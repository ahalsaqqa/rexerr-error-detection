"""Evaluate runs on the rule-based cross-generator test set (scripts/make_external.py).

Each perturbed report carries exactly one error, so the question per type is whether that error is found: recall of the
injected type at 0.5, and whether the report is flagged at all (any-error score >= threshold). Thresholds for the
any-error score are taken from the validation split, as in the main evaluation.

usage: python scripts/evaluate_external.py --runs_dir results/runs --out results/external.md [--summary results/external.json]
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


def evaluate(run_dir):
    d = load_predictions(run_dir, "external")
    y, p = d["y"].astype(int), d["p"]
    y_any, s_any = y.max(1), M.any_error_score(p)
    res = {"n": int(len(y)), "any_error": {"at_0.5": M.binary(y_any, s_any, 0.5)}}
    val_path = os.path.join(run_dir, "preds_val.npz")
    v = load_predictions(run_dir, "val") if os.path.exists(val_path) else None
    if v is not None and len(v["y"]):
        t = M.threshold_for_sensitivity(v["y"].max(1), M.any_error_score(v["p"]), 0.90)
        res["any_error"]["val_sens_0.90"] = M.binary(y_any, s_any, t)
    per = {}
    for j, c in enumerate(NATIVE):
        rows = y[:, j] == 1
        if rows.sum() == 0:
            continue
        per[c] = {"n": int(rows.sum()), "type_recall": float((p[rows, j] >= 0.5).mean()),
                  "flagged": float((s_any[rows] >= 0.5).mean()),
                  "most_predicted": NATIVE[int(np.argmax(p[rows].mean(0)))]}
    res["per_type"] = per
    res["macro_type_recall"] = float(np.mean([v_["type_recall"] for v_ in per.values()]))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_dir", default="results/runs")
    ap.add_argument("--out", default="results/external.md")
    ap.add_argument("--summary", default="results/external.json")
    args = ap.parse_args()
    out = {}
    for rd in sorted(glob.glob(os.path.join(args.runs_dir, "*"))):
        if os.path.exists(os.path.join(rd, "preds_external.npz")):
            out[os.path.basename(rd)] = {"name": display(load_meta(rd)), **evaluate(rd)}
    with open(args.summary, "w") as f:
        json.dump(out, f, indent=1)
    keys = list(out)
    L = ["## Cross-generator test set (rule-based errors, one per perturbed report; original reports as controls)\n",
         "| Model | Any-error AUROC | Sensitivity at 0.5 | Specificity at 0.5 | Sensitivity / specificity (val threshold, Se 0.90) | Macro type recall |",
         "|---|---|---|---|---|---|"]
    for k in keys:
        r = out[k]
        a, b = r["any_error"]["at_0.5"], r["any_error"].get("val_sens_0.90")
        vb = f"{b['sensitivity']:.3f} / {b['specificity']:.3f}" if b else "–"
        L.append(f"| {r['name']} | {a['auroc']:.3f} | {a['sensitivity']:.3f} | {a['specificity']:.3f} | {vb} | "
                 f"{r['macro_type_recall']:.3f} |")
    L += ["", "### Per injected type: recall of the injected type / share of reports flagged (threshold 0.5)\n",
          "| Type | n | " + " | ".join(out[k]["name"] for k in keys) + " |", "|---|---|" + "---|" * len(keys)]
    types = sorted({c for k in keys for c in out[k]["per_type"]})
    for c in types:
        n = next(out[k]["per_type"][c]["n"] for k in keys if c in out[k]["per_type"])
        cells = [f"{out[k]['per_type'][c]['type_recall']:.2f} / {out[k]['per_type'][c]['flagged']:.2f}"
                 if c in out[k]["per_type"] else "–" for k in keys]
        L.append(f"| {NATIVE_DISPLAY[c]} | {n} | " + " | ".join(cells) + " |")
    open(args.out, "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
