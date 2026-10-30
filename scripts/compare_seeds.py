"""Paired bootstrap comparison of models trained with several seeds, using seed-averaged metrics.

For each resample of test patients, macro-F1 and macro-AUPRC are averaged over the seeds of each model before taking
the difference, so the comparison does not rest on one favourable or unfavourable run. P-values are adjusted with
Holm's method over all pairwise comparisons of each metric; with B resamples a bootstrap p-value of 0 means p < 2/B,
and 2/B is used for it in the adjustment.

usage: python scripts/compare_seeds.py --runs_dir results/runs --models radbert cxrbert biobert --out results/seed_comparison.md
"""
import argparse
import itertools
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from rexerr import metrics as M  # noqa: E402
from rexerr.io import load_predictions  # noqa: E402


def stacked(runs_dir, model, seeds):
    ds = [load_predictions(os.path.join(runs_dir, f"{model}_s{s}"), "test") for s in seeds]
    for d in ds[1:]:
        assert np.array_equal(d["uid"], ds[0]["uid"])
    return ds[0], np.concatenate([d["p"] for d in ds], axis=1)


def seed_mean(metric, n_types, mask):
    def fn(y, P):
        return float(np.mean([M.summary(y, P[:, k:k + n_types], 0.5, mask=mask)[metric]
                              for k in range(0, P.shape[1], n_types)]))
    return fn


def holm(pvalues):
    """Holm-adjusted p-values, in the original order."""
    order = np.argsort(pvalues)
    m, running, adjusted = len(pvalues), 0.0, [0.0] * len(pvalues)
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvalues[i]))
        adjusted[i] = running
    return adjusted


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_dir", default="results/runs")
    ap.add_argument("--models", nargs="+", default=["radbert", "cxrbert", "biobert"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--n_boot", type=int, default=1000)
    ap.add_argument("--out", default="results/seed_comparison.md")
    args = ap.parse_args()
    data = {m: stacked(args.runs_dir, m, args.seeds) for m in args.models}
    ref = next(iter(data.values()))[0]
    y, n_types = ref["y"].astype(int), ref["y"].shape[1]
    mask = M.active_classes(y)
    rows, out = [], {}
    for a, b in itertools.combinations(args.models, 2):
        out[f"{a} vs {b}"] = {metric: M.paired_bootstrap(seed_mean(metric, n_types, mask), y, data[a][1], data[b][1],
                                                      ref["subject_id"], args.n_boot)
                           for metric in ("f1_macro", "auprc_macro")}
    floor = 2 / args.n_boot
    for metric in ("f1_macro", "auprc_macro"):
        adjusted = holm([max(r[metric]["p_value"], floor) for r in out.values()])
        for r, q in zip(out.values(), adjusted):
            r[metric]["p_holm"] = q

    def fmt(p):
        return f"< {floor:.3f}" if p < floor else f"= {p:.3f}"

    for name, res in out.items():
        cells = [f"{r['diff']:+.3f} ({r['ci'][0]:+.3f}, {r['ci'][1]:+.3f}), p {fmt(r['p_value'])}, Holm p "
                 f"{'= ' + format(r['p_holm'], '.3f')}" for r in res.values()]
        rows.append(f"| {name.replace(' vs ', ' − ')} | " + " | ".join(cells) + " |")
    lines = [f"## Seed-averaged paired comparisons (seeds {args.seeds}, {args.n_boot} patient-level resamples, "
             f"Holm over {len(out)} comparisons per metric)\n",
             "| Comparison | Δ macro-F1 (95% CI) | Δ macro-AUPRC (95% CI) |", "|---|---|---|"] + rows
    with open(args.out, "w") as f:
        f.write("\n".join(lines) + "\n")
    with open(os.path.splitext(args.out)[0] + ".json", "w") as f:
        json.dump(out, f, indent=1)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
