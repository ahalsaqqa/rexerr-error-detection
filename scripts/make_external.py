"""Build the external test set: the original test reports, each with one rule-based error, plus the originals.

usage: python scripts/make_external.py --data_dir DATA --out results/external_test.csv
The file contains report text, so it is not distributed with the code; it is rebuilt deterministically from ReXErr.
"""
import argparse
import os
import sys
from collections import Counter

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from rexerr.data import build_report_task, normalize_text  # noqa: E402
from rexerr.perturb import build  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    frame, _ = build_report_task(args.data_dir, "test", include_clean=True)
    orig = frame[frame["variant"] == "original"].reset_index(drop=True)
    rows = []
    for r, (text, kind) in zip(orig.itertuples(), build(orig["text"].tolist(), args.seed)):
        if text is None:
            continue
        rows.append((f"{r.study_id}_r", r.study_id, r.subject_id, "error", normalize_text(text), kind))
        rows.append((r.uid, r.study_id, r.subject_id, "original", r.text, ""))
    out = pd.DataFrame(rows, columns=["uid", "study_id", "subject_id", "variant", "text", "error_type"])
    out.to_csv(args.out, index=False)
    print(f"wrote {args.out}: {len(out)} reports;", dict(Counter(t for t in out["error_type"] if t)))


if __name__ == "__main__":
    main()
