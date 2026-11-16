"""All pairwise comparisons of the seven encoders on seed-averaged macro-F1, with a patient-level paired bootstrap
(10,000 resamples) and Holm's adjustment over the 21 comparisons.

Each resample draws test patients with replacement; macro-F1 (threshold 0.5, types with test positives) is computed
for every seed of both encoders and averaged over seeds before taking the difference, as in compare_seeds.py. The
resample is applied as per-report weights (the number of times the report's patient was drawn), which gives the same
counts as concatenating the drawn reports but is fast enough for 10,000 resamples of 21 runs. With B resamples a
bootstrap p-value of 0 means p < 2/B; 2/B is used for it in the adjustment.

usage: python scripts/pairwise_encoders.py --runs_dir results/runs --n_boot 10000 --out results/pairwise_encoders.json
"""
import argparse
import itertools
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
from compare_seeds import holm  # noqa: E402
from rexerr import metrics as M  # noqa: E402
from rexerr.io import load_predictions  # noqa: E402

ENCODERS = [("radbert", "RadBERT"), ("cxrbert", "CXR-BERT"), ("bluebert", "BlueBERT"), ("biobert", "BioBERT"),
            ("biomedbert", "BiomedBERT"), ("clinicalbert", "Bio+ClinicalBERT"), ("bert-base-cased", "BERT-base-cased")]


def macro_f1(w, y, yhat):
    """Weighted macro-F1 over the columns; w: (B, n) report weights, y and yhat: (n, K) binary."""
    tp = w @ (y & yhat)
    fp = w @ ((1 - y) & yhat)
    fn = w @ (y & (1 - yhat))
    den = 2 * tp + fp + fn
    f1 = np.where(den > 0, 2 * tp / np.maximum(den, 1), 0.0)
    return f1.mean(axis=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_dir", default="results/runs")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--n_boot", type=int, default=10000)
    ap.add_argument("--chunk", type=int, default=500)
    ap.add_argument("--out", default="results/pairwise_encoders.json")
    args = ap.parse_args()

    ref = None
    preds = {}
    for key, _ in ENCODERS:
        runs = [load_predictions(os.path.join(args.runs_dir, f"{key}_s{s}"), "test") for s in args.seeds]
        for d in runs:
            if ref is None:
                ref = d
            assert np.array_equal(d["uid"], ref["uid"])
        preds[key] = runs
    y_all = ref["y"].astype(int)
    active = M.active_classes(y_all)
    y = y_all[:, active]
    yhat = {k: [(d["p"][:, active] >= 0.5).astype(int) for d in runs] for k, runs in preds.items()}

    # report-level check against the evaluation code: the unweighted statistic equals summary()'s macro-F1
    for k, runs in preds.items():
        for d, yh in zip(runs, yhat[k]):
            assert abs(macro_f1(np.ones((1, len(y))), y, yh)[0] - M.summary(y_all, d["p"], 0.5)["f1_macro"]) < 1e-9

    groups = M._group_index(ref["subject_id"])
    patient_of = np.empty(len(y), dtype=int)
    for g, rows in enumerate(groups):
        patient_of[rows] = g
    rng = np.random.default_rng(0)
    boot = {k: [] for k, _ in ENCODERS}
    obs = {k: float(np.mean([macro_f1(np.ones((1, len(y))), y, yh)[0] for yh in yhat[k]])) for k, _ in ENCODERS}
    done = 0
    while done < args.n_boot:
        b = min(args.chunk, args.n_boot - done)
        counts = np.stack([np.bincount(rng.integers(0, len(groups), len(groups)), minlength=len(groups))
                           for _ in range(b)])  # (b, patients)
        w = counts[:, patient_of].astype(float)  # (b, reports)
        for k, _ in ENCODERS:
            boot[k].append(np.mean([macro_f1(w, y, yh) for yh in yhat[k]], axis=0))
        done += b
    boot = {k: np.concatenate(v) for k, v in boot.items()}

    names = dict(ENCODERS)
    rows = []
    for (a, _), (b, _) in itertools.combinations(ENCODERS, 2):
        if obs[a] < obs[b]:
            a, b = b, a  # report every difference as higher − lower mean macro-F1
        d = boot[a] - boot[b]
        p = min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean()))
        rows.append({"a": names[a], "b": names[b], "diff": obs[a] - obs[b],
                     "ci": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))], "p": float(p)})
    floor = 2 / args.n_boot
    for r, q in zip(rows, holm([max(r["p"], floor) for r in rows])):
        r["p_holm"] = float(q)
    rows.sort(key=lambda r: (-r["diff"]))
    out = {"n_boot": args.n_boot, "seeds": args.seeds, "mean_macro_f1": obs, "comparisons": rows}
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)
    for r in rows:
        print(f"{r['a']:>17} − {r['b']:<17} {r['diff']:+.4f} ({r['ci'][0]:+.4f}, {r['ci'][1]:+.4f}) "
              f"p={r['p']:.4f} Holm={r['p_holm']:.4f}")


if __name__ == "__main__":
    main()
