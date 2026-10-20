"""Paraphrase every sentence of the original test reports with a local LLM (GPU), for the paraphrase control.

Each sentence is rewritten on its own, with an instruction to keep the clinical content unchanged; section headings
("Findings:", "Impression:") and de-identification placeholders are kept out of the prompt and put back afterwards.
scripts/make_paraphrase_control.py checks the rewrites and assembles error-free paraphrased reports.
The output contains report text, so it is not distributed with the code.

usage: python scripts/paraphrase.py --data_dir DATA --out results/paraphrase_sentences.csv --model Qwen/Qwen3-8B
"""
import argparse
import os
import re
import sys
import time

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from rexerr.data import build_report_task, load_sentences, normalize_text  # noqa: E402

HEADING = re.compile(r"^((?:[A-Z][A-Za-z ]{0,30}):\s*)")
INSTRUCTION = ("Rewrite the following sentence from a chest X-ray report in different words. Keep exactly the same "
               "clinical content: the same findings, negations, uncertainty, anatomical locations and sides, "
               "severity, sizes and measurements, devices and comparisons with prior studies. Do not add, remove or "
               "correct anything, and keep any \"___\" placeholder as it is. Reply with the rewritten sentence only.")


def split_heading(sentence):
    m = HEADING.match(sentence)
    return (m.group(1), sentence[m.end():]) if m else ("", sentence)


def clean_output(text):
    text = normalize_text(text.split("\n")[0] if text.strip() else text)
    text = re.sub(r"^(rewritten sentence|sentence|rewrite)\s*:\s*", "", text, flags=re.I)
    return text.strip().strip('"').strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--batch_size", type=int, default=96)
    ap.add_argument("--max_new_tokens", type=int, default=160)
    ap.add_argument("--max_studies", type=int, default=0, help="debug only")
    args = ap.parse_args()

    t0 = time.time()
    frame, _ = build_report_task(args.data_dir, "test", include_clean=True)
    orig = frame[frame["variant"] == "original"]
    if args.max_studies:
        orig = orig.head(args.max_studies)
    sent = load_sentences(args.data_dir, "test")
    rows = []
    for r in orig.itertuples():
        g = sent[sent["study_id"] == r.study_id]
        for i, s in enumerate(normalize_text(x) for x in g["original_sentence"]):
            if s:
                head, body = split_heading(s)
                rows.append({"study_id": r.study_id, "subject_id": r.subject_id, "position": i, "original": s,
                             "heading": head, "body": body})
    df = pd.DataFrame(rows)
    print(f"[data] {len(orig)} reports, {len(df)} sentences ({time.time() - t0:.0f}s)", flush=True)

    tok = AutoTokenizer.from_pretrained(args.model)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map="auto").eval()
    prompts = [tok.apply_chat_template([{"role": "user", "content": f"{INSTRUCTION}\n\nSentence: {b}"}], tokenize=False,
                                       add_generation_prompt=True, enable_thinking=False) for b in df["body"]]
    order = sorted(range(len(prompts)), key=lambda k: len(prompts[k]))  # similar lengths per batch, less padding
    out_text = [""] * len(prompts)
    for i in range(0, len(order), args.batch_size):
        idx = order[i:i + args.batch_size]
        enc = tok([prompts[k] for k in idx], return_tensors="pt", padding=True).to(model.device)
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=args.max_new_tokens, do_sample=False,
                                 pad_token_id=tok.pad_token_id)
        for k, seq in zip(idx, out[:, enc["input_ids"].shape[1]:]):
            out_text[k] = clean_output(tok.decode(seq, skip_special_tokens=True))
        print(f"[llm] {min(i + args.batch_size, len(order))}/{len(order)} ({time.time() - t0:.0f}s)", flush=True)
    df["paraphrase"] = [h + p if p else "" for h, p in zip(df["heading"], out_text)]
    df.drop(columns=["body"]).to_csv(args.out, index=False)
    print(f"[done] {args.out} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
