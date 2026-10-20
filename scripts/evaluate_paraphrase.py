"""Evaluate runs on the paraphrase control (scripts/make_paraphrase_control.py).

All reports in the control are error-free, so every flag is a false alarm. For each model the share of reports flagged
(any-error score >= threshold) is compared between the original reports and their partially and fully reworded
versions, paired by study, with patient-level bootstrap confidence intervals. Thresholds: 0.5 and the validation
threshold for a sensitivity of 0.90, as in the main evaluation.

usage: python scripts/evaluate_paraphrase.py --runs_dir results/paraphrase_runs --out results/paraphrase.md
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from rexerr import metrics as M  # noqa: E402
from rexerr.io import load_meta, load_predictions  # noqa: E402
from rexerr.labels import NATIVE, NATIVE_DISPLAY  # noqa: E402
from evaluate import display  # noqa: E402

VARIANTS = ("original", "para_partial", "para_full")


def paired(flags, groups, n_boot=1000, seed=0):
    """Flag-rate differences (reworded - original) with 95% patient-level bootstrap CIs."""
    rng = np.random.default_rng(seed)
    idx = M._group_index(groups)
    out = {}
    for v in VARIANTS[1:]:
        d = flags[v].astype(float) - flags["original"].astype(float)
        boots = [d[np.concatenate([idx[i] for i in rng.integers(0, len(idx), len(idx))])].mean() for _ in range(n_boot)]
        out[v] = {"diff": float(d.mean()), "ci": (float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5)))}
    return out


def evaluate(run_dir):
    d = load_predictions(run_dir, "paraphrase")
    s = M.any_error_score(d["p"])
    by = {v: d["variant"] == v for v in VARIANTS}
    sid = {v: d["study_id"][by[v]] for v in VARIANTS}
    assert all(np.array_equal(sid[v], sid["original"]) for v in VARIANTS)
    groups = d["subject_id"][by["original"]]
    t = load_predictions(run_dir, "test")
    test_any = M.any_error_score(t["p"])
    # the originals scored here must match the originals scored at the end of training
    orig_test = dict(zip(t["uid"], test_any))
    repro = float(np.max(np.abs(s[by["original"]] - np.array([orig_test[u] for u in d["uid"][by["original"]]]))))
    v = load_predictions(run_dir, "val")
    thresholds = {"0.5": 0.5, "val_sens_0.90": M.threshold_for_sensitivity(v["y"].max(1), M.any_error_score(v["p"]), 0.90)}
    res = {"n_studies": int(by["original"].sum()), "max_abs_diff_vs_test_scores": repro, "thresholds": thresholds}
    for name, thr in thresholds.items():
        flags = {k: s[by[k]] >= thr for k in VARIANTS}
        err = t["variant"] == "error"
        res[name] = {
            "flag_rate": {k: float(flags[k].mean()) for k in VARIANTS},
            "difference": paired(flags, groups),
            "rexerr_error_reports_flagged": float((test_any[err] >= thr).mean()),
            "rexerr_originals_flagged": float((test_any[t["variant"] == "original"] >= thr).mean()),
        }
    res["mean_score"] = {k: float(s[by[k]].mean()) for k in VARIANTS}
    full = d["p"][by["para_full"]]
    res["types_on_para_full"] = {NATIVE[j]: float((full[:, j] >= 0.5).mean()) for j in np.argsort(-full.mean(0))[:3]}
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_dir", default="results/paraphrase_runs")
    ap.add_argument("--out", default="results/paraphrase.md")
    ap.add_argument("--summary", default="results/paraphrase.json")
    ap.add_argument("--checks", default="results/paraphrase_checks.json")
    args = ap.parse_args()
    out = {}
    for rd in sorted(glob.glob(os.path.join(args.runs_dir, "*"))):
        if os.path.exists(os.path.join(rd, "preds_paraphrase.npz")):
            out[os.path.basename(rd)] = {"name": display(load_meta(rd)), **evaluate(rd)}
    with open(args.summary, "w") as f:
        json.dump(out, f, indent=1)
    L = ["## Paraphrase control (error-free reports reworded by Qwen3-8B; every flag is a false alarm)\n"]
    if os.path.exists(args.checks):
        c = json.load(open(args.checks))
        L.append(f"{c['usable']:,} of {c['sentences']:,} sentence rewrites ({c['usable_frac']:.1%}) passed the content "
                 f"checks; {c['reports_used']:,} of {c['reports']:,} reports have at least one. Partial: "
                 f"{c['partial_sentences_mean']:.1f} sentences reworded per report; full: "
                 f"{c['full_sentences_mean']:.1f} ({c['full_share_of_sentences_reworded']:.0%} of sentences). Median share "
                 f"of words changed per reworded sentence: {c['word_change_paraphrase_median']:.2f} (ReXErr erroneous "
                 f"sentences: {c['word_change_rexerr_error_sentence_median']:.2f}).\n")
    for thr in ("0.5", "val_sens_0.90"):
        L += [f"### Share of reports flagged, threshold {thr}\n",
              "| Model | Originals | Partially reworded (Δ, 95% CI) | Fully reworded (Δ, 95% CI) | ReXErr error reports |",
              "|---|---|---|---|---|"]
        for k, r in out.items():
            x = r[thr]
            cells = [f"{x['flag_rate'][v]:.3f} ({x['difference'][v]['diff']:+.3f}; {x['difference'][v]['ci'][0]:+.3f} to "
                     f"{x['difference'][v]['ci'][1]:+.3f})" for v in VARIANTS[1:]]
            L.append(f"| {r['name']} | {x['flag_rate']['original']:.3f} | " + " | ".join(cells) +
                     f" | {x['rexerr_error_reports_flagged']:.3f} |")
        L.append("")
    L += ["### Most frequent types predicted on fully reworded reports (share at 0.5)\n"]
    for k, r in out.items():
        L.append(f"- {r['name']}: " + ", ".join(f"{NATIVE_DISPLAY[c]} {v:.3f}" for c, v in r["types_on_para_full"].items())
                 + f" (originals re-scored with max |Δ| = {r['max_abs_diff_vs_test_scores']:.4f})")
    open(args.out, "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
