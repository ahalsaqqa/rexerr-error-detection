"""Operating points behind the prevalence projections, with patient-level bootstrap intervals.

1. Any-error sensitivity by the number of annotated error types per error-injected report: most ReXErr error reports
   carry several types, so a report can be flagged because of any one of them, whereas a real erroneous report
   usually contains one error.
2. PPV and missed errors per 1,000 reports at prevalences of 3-20% for BioBERT at the validation threshold
   (sensitivity 0.90 on validation), using all error-injected reports, single-type reports only, or the
   specificity measured on fully reworded error-free reports (paraphrase control).
3. The same single-type view for the models scored on the cross-generator set, so that ReXErr and rule-based
   errors are compared with one error per report on both sides.

usage: python scripts/operating_points.py --out results/operating_points.json
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from rexerr import metrics as M  # noqa: E402
from rexerr.io import load_predictions  # noqa: E402

PREVALENCES = (0.03, 0.05, 0.10, 0.20)


def ppv(se, sp, pi):
    return se * pi / (se * pi + (1 - sp) * (1 - pi))


def val_threshold(run):
    v = load_predictions(run, "val")
    return M.threshold_for_sensitivity(v["y"].max(1), M.any_error_score(v["p"]), 0.90)


def by_count(run, thr):
    d = load_predictions(run, "test")
    k, s = d["y"].sum(1), M.any_error_score(d["p"])
    out = {}
    for name, sel in (("1", k == 1), ("2", k == 2), ("3", k == 3), ("4+", k >= 4), ("all", k > 0)):
        out[name] = {"n": int(sel.sum()), "se_0.5": float((s[sel] >= 0.5).mean()), "se_val": float((s[sel] >= thr).mean())}
    neg = k == 0  # as in the main evaluation: originals plus the 8 error-injected reports without annotated types
    out["specificity_val"] = float((s[neg] < thr).mean())
    out["specificity_0.5"] = float((s[neg] < 0.5).mean())
    return out


def bootstrap_ppv(run, thr, n_boot=1000, seed=0):
    """Se (all and single-type), Sp and PPV at each prevalence, resampling patients."""
    d = load_predictions(run, "test")
    k, s, g = d["y"].sum(1), M.any_error_score(d["p"]), d["subject_id"]
    neg = k == 0
    idx = M._group_index(g)
    rng = np.random.default_rng(seed)

    def stats(sel):
        kk, ss, nn = k[sel], s[sel], neg[sel]
        se, se1, sp = (ss[kk > 0] >= thr).mean(), (ss[kk == 1] >= thr).mean(), (ss[nn] < thr).mean()
        r = {"se": se, "se_single": se1, "sp": sp}
        for pi in PREVALENCES:
            r[f"ppv_{pi}"] = ppv(se, sp, pi)
            r[f"ppv_single_{pi}"] = ppv(se1, sp, pi)
        return r

    point = stats(np.arange(len(s)))
    boots = [stats(np.concatenate([idx[i] for i in rng.integers(0, len(idx), len(idx))])) for _ in range(n_boot)]
    return {key: {"value": float(v), "ci": [float(np.percentile([b[key] for b in boots], 2.5)),
                                            float(np.percentile([b[key] for b in boots], 97.5))]}
            for key, v in point.items()}


def single_type_external(run):
    """ReXErr single-type reports versus the cross-generator reports (one error each) for one retrained model."""
    thr = val_threshold(run)
    res = {}
    for split in ("test", "external"):
        d = load_predictions(run, split)
        k, s = d["y"].sum(1), M.any_error_score(d["p"])
        pos, neg = k == 1, d["variant"] == "original"
        y = np.r_[np.ones(pos.sum()), np.zeros(neg.sum())]
        res[split] = {"n_single": int(pos.sum()), "se_0.5": float((s[pos] >= 0.5).mean()),
                      "se_val": float((s[pos] >= thr).mean()), "auroc": float(M._auroc(y, np.r_[s[pos], s[neg]]))}
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_dir", default="results/runs")
    ap.add_argument("--external_dir", default="results/external_runs")
    ap.add_argument("--paraphrase", default="results/paraphrase.json")
    ap.add_argument("--out", default="results/operating_points.json")
    args = ap.parse_args()
    out = {"by_number_of_types": {}}
    for m in ("biobert", "cxrbert", "radbert"):
        for s in (0, 1, 2):
            run = os.path.join(args.runs_dir, f"{m}_s{s}")
            out["by_number_of_types"][f"{m}_s{s}"] = by_count(run, val_threshold(run))
    run = os.path.join(args.runs_dir, "biobert_s0")
    out["biobert_s0_val_threshold"] = bootstrap_ppv(run, val_threshold(run))
    # the retrained BioBERT of the paraphrase control, before and after rewording, at its validation threshold
    para = json.load(open(args.paraphrase))["biobert_s0_para"]["val_sens_0.90"]
    se_para = para["rexerr_error_reports_flagged"]
    out["paraphrase_scenario"] = {"se": se_para}
    for v in ("original", "para_full"):
        sp = 1 - para["flag_rate"][v]
        out["paraphrase_scenario"][v] = {"sp": sp, **{f"ppv_{pi}": ppv(se_para, sp, pi) for pi in PREVALENCES}}
    out["single_type_external"] = {m: single_type_external(os.path.join(args.external_dir, f"{m}_s0_ext"))
                                   for m in ("biobert", "cxrbert")}
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)
    b = out["biobert_s0_val_threshold"]
    print(json.dumps({k: (round(v["value"], 3), [round(x, 3) for x in v["ci"]]) for k, v in b.items()}, indent=0))
    print("paraphrase scenario", json.dumps(out["paraphrase_scenario"]))
    print("single-type external", json.dumps(out["single_type_external"], indent=0))


if __name__ == "__main__":
    main()
