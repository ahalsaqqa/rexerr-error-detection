"""Learning-rate sensitivity: every encoder at 5e-5 (seed 0) against the default 2e-5 (seed 0 and the seed mean),
plus BioBERT at 1e-5, with the rank agreement of the encoders between learning rates.

usage: python scripts/lr_sensitivity.py --runs_dir results/runs --lr_dir results/lr_runs --out results/lr_sensitivity.json
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
from scipy.stats import kendalltau, spearmanr

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
from revision_tables import averages  # noqa: E402
from rexerr.io import load_predictions  # noqa: E402

ENCODERS = ["radbert", "cxrbert", "biobert", "bluebert", "biomedbert", "bert-base-cased", "clinicalbert"]
KEYS = ("macro_f1", "micro_f1", "macro_auprc", "any_error_auroc")


def metrics(run):
    a = averages(load_predictions(run, "test"))
    return {k: a[k] for k in KEYS}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_dir", default="results/runs")
    ap.add_argument("--lr_dir", default="results/lr_runs")
    ap.add_argument("--out", default="results/lr_sensitivity.json")
    args = ap.parse_args()
    out = {"encoders": {}}
    for m in ENCODERS:
        row = {"lr2e-5_s0": metrics(os.path.join(args.runs_dir, f"{m}_s0"))}
        seeds = sorted(glob.glob(os.path.join(args.runs_dir, f"{m}_s[0-9]")))
        row["lr2e-5_seed_mean_macro_f1"] = float(np.mean([metrics(s)["macro_f1"] for s in seeds]))
        row["lr2e-5_n_seeds"] = len(seeds)
        for lr in ("1e-5", "5e-5"):
            run = os.path.join(args.lr_dir, f"{m}_lr{lr}_s0")
            if os.path.exists(os.path.join(run, "preds_test.npz")):
                row[f"lr{lr}_s0"] = metrics(run)
        out["encoders"][m] = row
    done = [m for m in ENCODERS if "lr5e-5_s0" in out["encoders"][m]]
    if len(done) >= 3:
        a = [out["encoders"][m]["lr2e-5_s0"]["macro_f1"] for m in done]
        mean = [out["encoders"][m]["lr2e-5_seed_mean_macro_f1"] for m in done]
        b = [out["encoders"][m]["lr5e-5_s0"]["macro_f1"] for m in done]
        out["rank_agreement"] = {
            "encoders": done,
            "spearman_s0": float(spearmanr(a, b)[0]), "kendall_s0": float(kendalltau(a, b)[0]),
            "spearman_seed_mean": float(spearmanr(mean, b)[0]), "kendall_seed_mean": float(kendalltau(mean, b)[0]),
            "gain_macro_f1": {m: y - x for m, x, y in zip(done, a, b)},
            "range_macro_f1_2e-5_s0": [min(a), max(a)], "range_macro_f1_5e-5": [min(b), max(b)],
        }
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)
    for m in ENCODERS:
        r = out["encoders"][m]
        print(f"{m:16s} 2e-5 s0 {r['lr2e-5_s0']['macro_f1']:.3f}  mean({r['lr2e-5_n_seeds']}) "
              f"{r['lr2e-5_seed_mean_macro_f1']:.3f}  5e-5 " +
              (f"{r['lr5e-5_s0']['macro_f1']:.3f}" if "lr5e-5_s0" in r else "  -  ") +
              (f"  1e-5 {r['lr1e-5_s0']['macro_f1']:.3f}" if "lr1e-5_s0" in r else ""))
    if "rank_agreement" in out:
        ra = out["rank_agreement"]
        print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in ra.items() if k != "gain_macro_f1"})


if __name__ == "__main__":
    main()
