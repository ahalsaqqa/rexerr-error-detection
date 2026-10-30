"""Dataset statistics quoted in the manuscript that are not produced by the model evaluation (CPU).

- training studies whose error report is empty or identical to the original after normalization;
- agreement between sentence-level labels and text changes in the test split, and annotated erroneous sentences that
  were deleted (empty error sentence) and therefore cannot be scored at sentence level;
- report length in BioBERT tokens (training error reports);
- lexical traces of the generator in inserted device statements (training split);
- studies per patient in each split.

usage: python scripts/audit_extra.py --data_dir DATA --out results/audit_extra.json
"""
import argparse
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from rexerr.data import load_reports, load_sentences, normalize_text  # noqa: E402
from rexerr.labels import normalize_type  # noqa: E402

DEVICE = re.compile(r"\b(tubes?|lines?|catheters?|pacemakers?|pacers?|wires?|leads?|clips?|valves?|ports?|drains?|"
                    r"stents?|electrodes?|picc|defibrillators?|icd|sternotomy)\b", re.I)
PHRASES = {"starts with 'A central venous'": lambda s: s.lower().startswith("a central venous"),
           "starts with 'A pacemaker is'": lambda s: s.lower().startswith("a pacemaker is"),
           "contains 'is present'": lambda s: "is present" in s.lower()}


def share(n, d):
    return {"n": int(n), "of": int(d), "share": float(n / d) if d else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--out", default="results/audit_extra.json")
    ap.add_argument("--tokenizer", default="dmis-lab/biobert-v1.1")
    args = ap.parse_args()
    out = {}

    rep = load_reports(args.data_dir, "train")
    err, orig = rep["error_report"].map(normalize_text), rep["original_report"].map(normalize_text)
    out["train_error_report_empty"] = int((err == "").sum())
    out["train_error_report_identical"] = int(((err == orig) & (err != "")).sum())

    sent = load_sentences(args.data_dir, "test")
    e, o = sent["error_sentence"].map(normalize_text), sent["original_sentence"].map(normalize_text)
    labeled_err, unchanged = sent["error_present"] == 1, sent["error_present"] == 0
    out["test_erroneous_sentences_identical_to_original"] = share(((e == o) & labeled_err).sum(), labeled_err.sum())
    out["test_unchanged_sentences_that_differ"] = share(((e != o) & unchanged).sum(), unchanged.sum())
    deleted = labeled_err & (e == "")
    out["test_erroneous_sentences_deleted"] = share(deleted.sum(), labeled_err.sum())
    out["test_deleted_by_type"] = {k: int(v) for k, v in sent[deleted]["error_type"].map(normalize_type).value_counts().items()}

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    lengths = np.array([len(ids) for ids in tok(err[err != ""].tolist(), truncation=False)["input_ids"]])
    out["train_error_report_tokens"] = {"n": int(len(lengths)), "median": float(np.median(lengths)),
                                        "p99": float(np.percentile(lengths, 99)),
                                        "over_320": float(np.mean(lengths > 320)), "over_512": float(np.mean(lengths > 512))}

    tr = load_sentences(args.data_dir, "train")
    kinds = [normalize_type(t) if p == 1 and isinstance(t, str) else None for t, p in zip(tr["error_type"], tr["error_present"])]
    inserted = [normalize_text(s) for s, k in zip(tr["error_sentence"], kinds) if k == "add_medical_device"]
    inserted = [s for s in inserted if s]
    original = [s for s in (normalize_text(x) for x in tr["original_sentence"]) if s and DEVICE.search(s)]
    out["device_statements"] = {"inserted_by_generator": len(inserted), "original_mentioning_a_device": len(original),
                                "phrases": {name: {"inserted": share(sum(map(f, inserted)), len(inserted)),
                                                   "original": share(sum(map(f, original)), len(original))}
                                            for name, f in PHRASES.items()}}

    out["studies_per_patient"] = {}
    for split in ("train", "val", "test"):
        r = load_reports(args.data_dir, split)
        c = r.groupby("subject_id").size()
        out["studies_per_patient"][split] = {"studies": int(len(r)), "patients": int(len(c)), "mean": float(c.mean()),
                                             "max": int(c.max())}
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
