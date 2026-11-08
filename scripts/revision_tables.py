"""Tables added in the revision (CPU, saved predictions).

1. Per-type precision, recall, F1, AUROC and AUPRC with the number of positives, for chosen runs.
2. Averages that are not dominated by rare types: micro-F1, support-weighted F1 and macro-F1 without the two types
   with fewer than 120 test positives (change measurement, change to homophone), mean over seeds where available.
3. Any-error sensitivity and specificity of the LLMs, whose answers are binary, so that they are compared at their
   single operating point instead of with AUROC/AUPRC.

usage: python scripts/revision_tables.py --runs_dir results/runs --out results/revision_tables.json
"""
import argparse
import glob
import json
import os
import re
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from rexerr import metrics as M  # noqa: E402
from rexerr.io import load_meta, load_predictions  # noqa: E402
from rexerr.labels import NATIVE, NATIVE_DISPLAY  # noqa: E402

RARE = ("change_measurement", "change_to_homophone")


def per_type(d):
    y, p = d["y"].astype(int), d["p"]
    rows = {}
    for j, c in enumerate(NATIVE):
        if y[:, j].sum() == 0:
            continue
        tp = int(((p[:, j] >= 0.5) & (y[:, j] == 1)).sum())
        fp = int(((p[:, j] >= 0.5) & (y[:, j] == 0)).sum())
        fn = int(((p[:, j] < 0.5) & (y[:, j] == 1)).sum())
        pr, rc, f = M._prf(tp, fp, fn)
        rows[c] = {"n": int(y[:, j].sum()), "precision": pr, "recall": rc, "f1": f,
                   "auroc": float(M._auroc(y[:, j], p[:, j])), "auprc": float(M._auprc(y[:, j], p[:, j]))}
    return rows


def averages(d):
    rows = per_type(d)
    y, yhat = d["y"].astype(int), (d["p"] >= 0.5).astype(int)
    act = y.sum(0) > 0
    tp, fp, fn = (yhat & y)[:, act].sum(), (yhat & (1 - y))[:, act].sum(), ((1 - yhat) & y)[:, act].sum()
    n = np.array([r["n"] for r in rows.values()])
    f1 = np.array([r["f1"] for r in rows.values()])
    keep = np.array([c not in RARE for c in rows])
    pos, s = d["y"].max(1) > 0, M.any_error_score(d["p"])
    return {"macro_f1": float(f1.mean()), "micro_f1": float(M._prf(tp, fp, fn)[2]),
            "weighted_f1": float((f1 * n).sum() / n.sum()), "macro_f1_without_rare": float(f1[keep].mean()),
            "macro_auroc": float(np.mean([r["auroc"] for r in rows.values()])),
            "macro_auprc": float(np.mean([r["auprc"] for r in rows.values()])),
            "any_error_auroc": float(M._auroc(pos.astype(int), s))}


def any_error(d):
    pos, s = d["y"].max(1) > 0, M.any_error_score(d["p"])
    return {"sensitivity": float((s[pos] >= 0.5).mean()), "specificity": float((s[~pos] < 0.5).mean())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_dir", default="results/runs")
    ap.add_argument("--per_type_runs", nargs="+", default=["biobert_s0", "cxrbert_s0"])
    ap.add_argument("--out", default="results/revision_tables.json")
    args = ap.parse_args()
    out = {"per_type": {}, "averages": {}, "llm_any_error": {}}
    for r in args.per_type_runs:
        out["per_type"][r] = per_type(load_predictions(os.path.join(args.runs_dir, r), "test"))
    groups = defaultdict(list)
    for run in sorted(glob.glob(os.path.join(args.runs_dir, "*"))):
        name = os.path.basename(run)
        if not os.path.exists(os.path.join(run, "preds_test.npz")):
            continue
        d = load_predictions(run, "test")
        if load_meta(run).get("family") == "llm_zeroshot":
            out["llm_any_error"][name] = {**any_error(d), **averages(d)}
            continue
        if re.fullmatch(r"[a-z0-9.\-]+_s\d", name) and "lr" not in name:  # encoders with the default setup
            groups[name.rsplit("_s", 1)[0]].append(averages(d))
        elif name.startswith(("tfidf", "length")):
            out["averages"][name] = {k: [v, 0.0, 1] for k, v in averages(d).items()}
    for model, runs in groups.items():
        out["averages"][model] = {k: [float(np.mean([r[k] for r in runs])), float(np.std([r[k] for r in runs], ddof=1))
                                      if len(runs) > 1 else 0.0, len(runs)] for k in runs[0]}
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)
    for m, v in sorted(out["averages"].items()):
        print(f"{m:<22}", {k: (round(x[0], 3), x[2]) for k, x in v.items()})
    for m, v in out["llm_any_error"].items():
        print(f"{m:<22}", {k: round(x, 3) for k, x in v.items()})


if __name__ == "__main__":
    main()
