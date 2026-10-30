"""Zero-shot / few-shot LLM baseline on the test split (GPU).

The LLM runs locally (16-bit; 4-bit only on GPUs with less than 20 GB), so no report text leaves the machine.
Outputs are binary (type flagged or not), stored in the common prediction format. --n_studies 0 scores the whole test
split, as in the paper; a positive value scores a patient-level random subset.

usage: python scripts/llm_zeroshot.py --data_dir DATA --out_dir results/runs/llm_qwen2.5-7b \
           --model Qwen/Qwen2.5-7B-Instruct --n_studies 0
       [--shots 4]                       few-shot: 4 training studies, each as error report + original report
       [--test_csv external_test.csv]    score the rule-based external set instead of the test split
"""
import argparse
import json
import os
import re
import sys
import time

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from rexerr.data import build_external_task, build_report_task  # noqa: E402
from rexerr.io import save_predictions  # noqa: E402
from rexerr.labels import NATIVE, NATIVE_DISPLAY, normalize_type  # noqa: E402

DEFINITIONS = {
    "add_medical_device": "a medical device (line, tube, catheter, pacemaker...) is mentioned that is unlikely to be present",
    "false_prediction": "a finding is reported that is unlikely to be present",
    "false_negation": "a finding is wrongly negated (stated as absent)",
    "add_repetitions": "a statement is repeated unnecessarily",
    "add_contradictions": "two statements in the report contradict each other",
    "change_name_of_device": "a device is named incorrectly",
    "change_position_of_device": "the position of a device is described incorrectly",
    "change_severity": "the severity/size of a finding is misstated",
    "change_location": "the anatomical location or laterality of a finding is wrong",
    "change_measurement": "a measurement or number is wrong",
    "change_to_homophone": "a word is replaced by a similar-sounding wrong word (speech-recognition error)",
    "add_typo": "a typographical error",
}

SYSTEM = "You are an experienced radiologist who proofreads chest X-ray reports for errors."


def instructions():
    lines = "\n".join(f'- "{NATIVE_DISPLAY[k]}": {v}' for k, v in DEFINITIONS.items())
    return (
        "The following chest X-ray report may contain errors of these types:\n"
        f"{lines}\n\n"
        "Many reports contain no error at all. Read the report and list every error type that is present.\n"
        'Answer with JSON only, for example {"errors": ["Add typo", "Change severity"]} or {"errors": []}.\n\n'
    )


def prompt(report):
    return instructions() + f"Report:\n{report}"


def parse(text):
    names = {NATIVE_DISPLAY[k].lower(): k for k in NATIVE}
    found = set()
    text = re.sub(r"<think>.*?(</think>|$)", "", text, flags=re.S)  # reasoning models (thinking is disabled anyway)
    m = re.search(r"\{.*\}", text, flags=re.S)
    if m:
        try:
            for e in json.loads(m.group(0)).get("errors", []):
                k = normalize_type(e)
                if k:
                    found.add(k)
            return found, True
        except (json.JSONDecodeError, AttributeError):
            pass
    low = text.lower()
    return {k for n, k in names.items() if n in low}, False


def pick_examples(data_dir, n_studies, seed, min_words=30, max_words=90):
    """Few-shot examples: training studies whose error reports cover as many error types as possible.

    Each study contributes its error report (with its sentence-level types) and its original report (no errors).
    """
    tr, ytr = build_report_task(data_dir, "train", "sentence", True)
    words = tr["text"].str.split().str.len()
    err = np.where((tr["variant"] == "error") & words.between(min_words, max_words) & (ytr.sum(1) > 0))[0]
    rng = np.random.default_rng(seed)
    pool = rng.choice(err, size=min(2000, len(err)), replace=False)
    chosen, covered = [], set()
    for _ in range(n_studies):
        gain = [len(set(np.where(ytr[i])[0]) - covered) for i in pool]
        best = pool[int(np.argmax(gain))]
        chosen.append(best)
        covered |= set(np.where(ytr[best])[0])
        pool = pool[pool != best]
    pairs = []
    for i in chosen:
        sid = tr["study_id"].iloc[i]
        orig = tr.index[(tr["study_id"] == sid) & (tr["variant"] == "original")]
        pairs.append((tr["text"].iloc[i], [NATIVE_DISPLAY[NATIVE[j]] for j in np.where(ytr[i])[0]], tr["uid"].iloc[i]))
        if len(orig):
            pairs.append((tr["text"].loc[orig[0]], [], tr["uid"].loc[orig[0]]))
    order = rng.permutation(len(pairs))
    return [pairs[k] for k in order]


def messages(report, examples):
    """Zero-shot: system + one user turn. Few-shot: the instructions once, then one turn per example report."""
    if not examples:
        return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt(report)}]
    msgs = [{"role": "system", "content": SYSTEM + "\n\n" + instructions().strip()}]
    for text, types, _ in examples:
        msgs += [{"role": "user", "content": f"Report:\n{text}"},
                 {"role": "assistant", "content": json.dumps({"errors": types})}]
    return msgs + [{"role": "user", "content": f"Report:\n{report}"}]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--n_studies", type=int, default=500, help="0 = the whole test split")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--max_new_tokens", type=int, default=80)
    ap.add_argument("--load_in_4bit", type=int, default=-1, help="-1 = only when all GPUs together have < 20 GB")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--shots", type=int, default=0, help="number of training studies shown as examples (0 = zero-shot)")
    ap.add_argument("--test_csv", default=None, help="external test set from scripts/make_external.py")
    args = ap.parse_args()

    t0 = time.time()
    if args.test_csv:
        te, Yte = build_external_task(args.test_csv)
    else:
        te, Yte = build_report_task(args.data_dir, "test", "sentence", True, max_studies=args.n_studies or None,
                                    seed=args.seed)
    va, Yva = te.iloc[:0], Yte[:0]
    examples = pick_examples(args.data_dir, args.shots, args.seed) if args.shots else []
    tok = AutoTokenizer.from_pretrained(args.model)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    cuda = torch.cuda.is_available()
    native_bf16 = cuda and torch.cuda.get_device_capability(0)[0] >= 8  # a T4 only emulates bf16
    dtype = (torch.bfloat16 if native_bf16 else torch.float16) if cuda else torch.float32
    if args.load_in_4bit == -1:
        total = sum(torch.cuda.get_device_properties(i).total_memory for i in range(torch.cuda.device_count()))
        args.load_in_4bit = int(cuda and total < 20e9)  # device_map="auto" spreads the model over all GPUs
    kw = {"dtype": dtype, "device_map": "auto"}
    if args.load_in_4bit:
        from transformers import BitsAndBytesConfig
        kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=dtype,
                                                       bnb_4bit_quant_type="nf4")
    model = AutoModelForCausalLM.from_pretrained(args.model, **kw).eval()

    texts = [tok.apply_chat_template(messages(r, examples), tokenize=False, add_generation_prompt=True,
                                     enable_thinking=False)  # Qwen3: answer directly; ignored by other templates
             for r in te["text"]]
    P = np.zeros((len(te), len(NATIVE)), dtype=np.float32)
    raw, parsed_ok = [], 0
    for i in range(0, len(texts), args.batch_size):
        enc = tok(texts[i:i + args.batch_size], return_tensors="pt", padding=True).to(model.device)
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=args.max_new_tokens, do_sample=False,
                                 pad_token_id=tok.pad_token_id)
        for j, seq in enumerate(out[:, enc["input_ids"].shape[1]:]):
            ans = tok.decode(seq, skip_special_tokens=True)
            found, ok = parse(ans)
            parsed_ok += ok
            raw.append(ans)
            for k in found:
                P[i + j, NATIVE.index(k)] = 1.0
        print(f"[llm] {min(i + args.batch_size, len(texts))}/{len(texts)} ({time.time() - t0:.0f}s)", flush=True)

    meta = {"model": args.model, "family": "llm_zeroshot", "task": "report", "seed": args.seed,
            "label_source": "sentence", "include_clean": 1, "n_test": len(te), "json_parse_rate": parsed_ok / len(te),
            "shots": args.shots, "example_uids": [u for _, _, u in examples],  # no report text in the meta file
            "test_set": "external" if args.test_csv else "test",
            "runtime_sec": time.time() - t0, "args": vars(args),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"}
    split = "external" if args.test_csv else "test"
    save_predictions(args.out_dir, "val", va, Yva, np.zeros((0, len(NATIVE))))
    save_predictions(args.out_dir, split, te, Yte, P, meta)
    with open(os.path.join(args.out_dir, "raw_outputs.jsonl"), "w") as f:
        for u, a in zip(te["uid"], raw):
            f.write(json.dumps({"uid": u, "output": a}) + "\n")
    print(f"[done] {args.out_dir} parse rate {parsed_ok / len(te):.3f} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
