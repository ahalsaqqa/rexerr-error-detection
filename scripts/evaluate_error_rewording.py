"""Evaluate runs on the error-rewording control (scripts/make_paraphrase_control.py --target errors).

The erroneous sentences of each error-injected test report were reworded by another LLM while keeping the error, so
labels are unchanged. If a detector relied on how GPT-4o phrases inserted errors, detection should drop. Reported per
model: any-error sensitivity (threshold 0.5 and validation threshold) and macro type recall before and after
rewording, paired by study with patient-level bootstrap CIs, and type recall per reworded type.

usage: python scripts/evaluate_error_rewording.py --runs_dir results/error_rewording_runs --out results/error_rewording.md
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
from paraphrase import KEEP_AS_IS as KEPT  # noqa: E402


def paired_diff(a, b, groups, n_boot=1000, seed=0):
    """mean(b) - mean(a) for paired per-study values, with a patient-level bootstrap CI."""
    rng = np.random.default_rng(seed)
    idx = M._group_index(groups)
    d = b.astype(float) - a.astype(float)
    boots = [d[np.concatenate([idx[i] for i in rng.integers(0, len(idx), len(idx))])].mean() for _ in range(n_boot)]
    return {"before": float(a.mean()), "after": float(b.mean()), "diff": float(d.mean()),
            "ci": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]}


def evaluate(run_dir):
    d = load_predictions(run_dir, "mirror")
    before, after = d["variant"] == "error", d["variant"] == "error_reworded"
    assert np.array_equal(d["study_id"][before], d["study_id"][after])
    y, p = d["y"][before].astype(int), d["p"]
    groups = d["subject_id"][before]
    s = M.any_error_score(p)
    v = load_predictions(run_dir, "val")
    thr = M.threshold_for_sensitivity(v["y"].max(1), M.any_error_score(v["p"]), 0.90)
    res = {"n_reports": int(before.sum()), "threshold_val": float(thr),
           "flagged_0.5": paired_diff(s[before] >= 0.5, s[after] >= 0.5, groups),
           "flagged_val": paired_diff(s[before] >= thr, s[after] >= thr, groups)}
    # share of annotated (study, type) pairs whose type is predicted, per type
    per = {}
    for j, c in enumerate(NATIVE):
        rows = y[:, j] == 1
        if rows.sum() >= 20:
            per[c] = {"n": int(rows.sum()), "before": float((p[before][rows, j] >= 0.5).mean()),
                      "after": float((p[after][rows, j] >= 0.5).mean())}
    res["type_recall"] = per
    reworded = [x for c, x in per.items() if c not in KEPT]  # typos, homophones and repetitions were not reworded
    res["macro_type_recall"] = {"before": float(np.mean([x["before"] for x in reworded])),
                                "after": float(np.mean([x["after"] for x in reworded]))}
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_dir", default="results/error_rewording_runs")
    ap.add_argument("--out", default="results/error_rewording.md")
    ap.add_argument("--summary", default="results/error_rewording.json")
    ap.add_argument("--checks", default="results/error_rewording_checks.json")
    args = ap.parse_args()
    out = {}
    for rd in sorted(glob.glob(os.path.join(args.runs_dir, "*"))):
        if os.path.exists(os.path.join(rd, "preds_mirror.npz")):
            out[os.path.basename(rd)] = {"name": display(load_meta(rd)), **evaluate(rd)}
    with open(args.summary, "w") as f:
        json.dump(out, f, indent=1)
    L = ["## Error-rewording control (erroneous sentences reworded by Qwen3-8B, errors kept)\n"]
    if os.path.exists(args.checks):
        c = json.load(open(args.checks))
        L.append(f"{c['usable']:,} of {c['sentences']:,} erroneous sentences ({c['usable_frac']:.1%}) were reworded with "
                 f"all checks passed; {c['reports_used']:,} error-injected reports have at least one "
                 f"(mean {c['reworded_sentences_per_report_mean']:.1f} reworded sentences). Typos, homophones and "
                 "repetitions were not reworded.\n")
    L += ["| Model | Flagged at 0.5: before → after (Δ, 95% CI) | Flagged at validation threshold | Macro type recall (reworded types) |",
          "|---|---|---|---|"]
    for k, r in out.items():
        cells = [f"{r[m]['before']:.3f} → {r[m]['after']:.3f} ({r[m]['diff']:+.3f}; {r[m]['ci'][0]:+.3f} to "
                 f"{r[m]['ci'][1]:+.3f})" for m in ("flagged_0.5", "flagged_val")]
        L.append(f"| {r['name']} | " + " | ".join(cells) +
                 f" | {r['macro_type_recall']['before']:.3f} → {r['macro_type_recall']['after']:.3f} |")
    L += ["", "### Type recall before → after\n", "| Type | n | " + " | ".join(r["name"] for r in out.values()) + " |",
          "|---|---|" + "---|" * len(out)]
    types = sorted({c for r in out.values() for c in r["type_recall"]})
    for c in types:
        n = next(r["type_recall"][c]["n"] for r in out.values() if c in r["type_recall"])
        L.append(f"| {NATIVE_DISPLAY[c]} | {n} | " + " | ".join(
            f"{r['type_recall'][c]['before']:.2f} → {r['type_recall'][c]['after']:.2f}" if c in r["type_recall"] else "–"
            for r in out.values()) + " |")
    open(args.out, "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
