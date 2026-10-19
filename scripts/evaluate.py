"""Evaluate every run in --runs_dir with one code path and write the paper's tables.

Outputs (in --out_dir):
  summary.json        all metrics, CIs, seed aggregates, paired comparisons, prevalence analysis
  tables.md           human-readable tables used in the manuscript
  error_examples.csv  qualitative false positives / negatives of the main model (needs --data_dir)

usage: python scripts/evaluate.py --runs_dir results/runs --out_dir results --main biobert [--data_dir DATA]
"""
import argparse
import glob
import json
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from rexerr import metrics as M  # noqa: E402
from rexerr.io import load_meta, load_predictions  # noqa: E402
from rexerr.labels import GROUP6, GROUP6_DISPLAY, NATIVE, NATIVE_DISPLAY, group6_matrix  # noqa: E402

DISPLAY = {
    "dmis-lab/biobert-v1.1": "BioBERT v1.1",
    "bert-base-cased": "BERT-base (cased)",
    "bert-base-uncased": "BERT-base (uncased)",
    "emilyalsentzer/Bio_ClinicalBERT": "Bio+ClinicalBERT",
    "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext": "BiomedBERT (PubMedBERT)",
    "StanfordAIMI/RadBERT": "RadBERT",
    "microsoft/BiomedVLP-CXR-BERT-general": "CXR-BERT",
    "UCSD-VA-health/RadBERT-RoBERTa-4m": "RadBERT (RoBERTa)",
    "bionlp/bluebert_pubmed_mimic_uncased_L-12_H-768_A-12": "BlueBERT",
    "unsloth/Meta-Llama-3.1-8B-Instruct": "Llama-3.1-8B-Instruct (zero-shot)",
    "tfidf": "TF-IDF + LR",
    "length": "Report length + LR",
    "Qwen/Qwen2.5-7B-Instruct": "Qwen2.5-7B-Instruct (zero-shot)",
    "Qwen/Qwen3-8B": "Qwen3-8B (zero-shot)",
}
HEADLINE = ["f1_macro", "f1_micro", "auroc_macro", "auprc_macro", "any_auroc", "any_auprc"]


def display(meta):
    name = DISPLAY.get(meta.get("model"), meta.get("model", "?"))
    if meta.get("task") == "sentence":
        name += " [sentence-level]"
    if str(meta.get("include_clean", 1)) == "0":
        name += " [trained without clean reports]"
    if meta.get("label_source") == "sampled":
        name += " [errors_sampled labels]"
    if meta.get("loss") in ("weighted", "focal"):
        name += " [weighted BCE]" if meta["loss"] == "weighted" else " [focal loss]"
    tag = (meta.get("args") or {}).get("tag")
    if tag:
        name += f" [{tag.strip('_')} training studies]" if tag.strip("_").endswith("k") else f" [{tag.strip('_')}]"
    return name


def aggregate_sentences(d):
    """Sentence predictions -> report predictions (max over the report's sentences)."""
    keys = np.char.add(d["study_id"].astype(str), np.char.add("_", d["variant"]))
    uniq, inv = np.unique(keys, return_inverse=True)
    y = np.zeros((len(uniq), d["y"].shape[1]), dtype=np.int8)
    p = np.zeros((len(uniq), d["p"].shape[1]), dtype=np.float32)
    np.maximum.at(y, inv, d["y"])
    np.maximum.at(p, inv, d["p"])
    first = np.zeros(len(uniq), dtype=int)
    first[inv[::-1]] = np.arange(len(inv))[::-1]
    return {"uid": uniq, "study_id": d["study_id"][first], "subject_id": d["subject_id"][first],
            "variant": d["variant"][first], "y": y, "p": p}


def headline_fn(mask):
    def fn(y, p):
        s = M.summary(y, p, 0.5, mask=mask)
        b = M.binary(y.max(1), M.any_error_score(p))
        return {"f1_macro": s["f1_macro"], "f1_micro": s["f1_micro"], "auroc_macro": s["auroc_macro"],
                "auprc_macro": s["auprc_macro"], "any_auroc": b["auroc"], "any_auprc": b["auprc"]}
    return fn


def evaluate_report_level(val, test, n_boot, seed=0):
    y, p, yv, pv = test["y"].astype(int), test["p"], val["y"].astype(int), val["p"]
    mask = M.active_classes(y)
    res = {"n_test": int(len(y)), "n_val": int(len(yv)), "active_classes": [c for c, m in zip(NATIVE, mask) if m]}
    res["native"] = M.summary(y, p, 0.5, mask=mask)
    res["native_per_class"] = M.per_class(y, p, NATIVE, 0.5)
    thr = M.tune_thresholds(yv, pv)
    res["native_tuned"] = M.summary(y, p, thr, mask=mask)
    res["native_tuned_per_class"] = M.per_class(y, p, NATIVE, thr)
    y6, p6 = group6_matrix(y), group6_matrix(p)
    res["group6"] = M.summary(y6, p6, 0.5)
    res["group6_per_class"] = M.per_class(y6, p6, GROUP6, 0.5)

    y_any, s_any = y.max(1), M.any_error_score(p)
    yv_any, sv_any = yv.max(1), M.any_error_score(pv)
    res["any_error"] = {"at_0.5": M.binary(y_any, s_any, 0.5)}
    for target in (0.90, 0.95):
        t = M.threshold_for_sensitivity(yv_any, sv_any, target)
        b = M.binary(y_any, s_any, t)
        res["any_error"][f"val_sens_{target:.2f}"] = b
        res["any_error"][f"prevalence_sens_{target:.2f}"] = M.prevalence_analysis(b["sensitivity"], b["specificity"])
    b = res["any_error"]["at_0.5"]
    res["any_error"]["prevalence_at_0.5"] = M.prevalence_analysis(b["sensitivity"], b["specificity"])

    err = test["variant"] == "error"
    if err.any() and (~err).any():  # error reports only = evaluation without controls
        res["error_reports_only"] = M.summary(y[err], p[err], 0.5, mask=M.active_classes(y[err]))

    res["ci95"] = M.bootstrap(headline_fn(mask), [y, p], test["subject_id"], n_boot=n_boot, seed=seed)
    return res


def evaluate_sentence_level(test):
    y, p = test["y"].astype(int), test["p"]
    mask = M.active_classes(y)
    err_rows = test["variant"] == "error"
    return {
        "n_sentences": int(len(y)),
        "localization": M.binary(y.max(1), M.any_error_score(p)),
        "localization_error_reports_only": M.binary(y[err_rows].max(1), M.any_error_score(p[err_rows])),
        "type": M.summary(y, p, 0.5, mask=mask),
        "type_per_class": M.per_class(y, p, NATIVE, 0.5),
    }


def fmt(x, nd=3):
    return "–" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{nd}f}"


def fmt_ci(v, ci):
    return f"{fmt(v)} ({fmt(ci[0])}–{fmt(ci[1])})"


def write_tables(out, path):
    L = []
    runs, groups = out["runs"], out["groups"]
    L.append("## Table A. Main comparison (official test split, report level, 12 native types)\n")
    L.append("Mean ± SD over seeds when several seeds were run; 95% CI (patient-level bootstrap) for the first seed.\n")
    L.append("| Model | Seeds | Macro-F1 | Micro-F1 | Macro-AUROC | Macro-AUPRC | Any-error AUROC | Any-error AUPRC |")
    L.append("|---|---|---|---|---|---|---|---|")
    for g in sorted(groups.values(), key=lambda g: -g["mean"]["auprc_macro"]):
        r0 = runs[g["runs"][0]]["report"]
        cells = []
        for k in HEADLINE:
            m, s = g["mean"][k], g["sd"][k]
            if len(g["runs"]) > 1:
                cells.append(f"{fmt(m)} ± {fmt(s)}")
            else:
                cells.append(fmt_ci(m, r0["ci95"][k]))
        L.append(f"| {g['name']} | {len(g['runs'])} | " + " | ".join(cells) + " |")

    main = out.get("main_run")
    if main:
        r = runs[main]["report"]
        L.append(f"\n## Table B. Per-type results of the main model ({runs[main]['name']}, threshold 0.5)\n")
        L.append("| Error type | Support | Precision | Recall | F1 | AUROC | AUPRC |")
        L.append("|---|---|---|---|---|---|---|")
        for c in r["native_per_class"]:
            if c["support"] == 0:
                continue
            L.append(f"| {NATIVE_DISPLAY[c['class']]} | {c['support']} | {fmt(c['precision'])} | {fmt(c['recall'])} | "
                     f"{fmt(c['f1'])} | {fmt(c['auroc'])} | {fmt(c['auprc'])} |")
        L.append(f"\n## Table C. Six-group results of the main model\n")
        L.append("| Group | Support | Precision | Recall | F1 | AUROC | AUPRC |")
        L.append("|---|---|---|---|---|---|---|")
        for c in r["group6_per_class"]:
            if c["support"] == 0:
                continue
            L.append(f"| {GROUP6_DISPLAY[c['class']]} | {c['support']} | {fmt(c['precision'])} | {fmt(c['recall'])} | "
                     f"{fmt(c['f1'])} | {fmt(c['auroc'])} | {fmt(c['auprc'])} |")
        L.append("\n## Table C2. Confusion counts of the main model for every error type (threshold 0.5)\n")
        L.append("| Error type | TP | FP | FN | TN | Specificity |")
        L.append("|---|---|---|---|---|---|")
        for c in r["native_per_class"]:
            if c["support"] == 0:
                continue
            L.append(f"| {NATIVE_DISPLAY[c['class']]} | {c['tp']} | {c['fp']} | {c['fn']} | {c['tn']} | "
                     f"{fmt(c['specificity'])} |")
        L.append("\n## Table C3. Per-type thresholds tuned on the validation split (secondary analysis)\n")
        t = r["native_tuned"]
        L.append(f"Macro-F1 {fmt(t['f1_macro'])} (vs. {fmt(r['native']['f1_macro'])} at 0.5); micro-F1 "
                 f"{fmt(t['f1_micro'])} (vs. {fmt(r['native']['f1_micro'])}).\n")
        a = r["any_error"]
        L.append("\n## Table D. Detecting whether a report contains any error (main model)\n")
        L.append("| Operating point | Threshold | Sensitivity | Specificity | Precision (test mix) |")
        L.append("|---|---|---|---|---|")
        for k, lab in (("at_0.5", "Default 0.5"), ("val_sens_0.90", "Sensitivity 0.90 on val"),
                       ("val_sens_0.95", "Sensitivity 0.95 on val")):
            b = a[k]
            L.append(f"| {lab} | {fmt(b['threshold'])} | {fmt(b['sensitivity'])} | {fmt(b['specificity'])} | {fmt(b['precision'])} |")
        L.append("\n## Table E. Expected performance at realistic error prevalence (main model, sensitivity-0.90 operating point)\n")
        L.append("| Prevalence | PPV | NPV | Alerts / 1000 reports | True alerts / 1000 | Missed errors / 1000 |")
        L.append("|---|---|---|---|---|---|")
        for row in a["prevalence_sens_0.90"]:
            L.append(f"| {row['prevalence']:.0%} | {fmt(row['ppv'])} | {fmt(row['npv'])} | {row['alerts_per_1000']:.0f} | "
                     f"{row['true_alerts_per_1000']:.0f} | {row['missed_per_1000']:.1f} |")

    if out["comparisons"]:
        L.append("\n## Table F. Paired bootstrap comparisons against the main model (difference = main − other)\n")
        L.append("| Other model | Δ Macro-F1 (95% CI) | p | Δ Macro-AUPRC (95% CI) | p |")
        L.append("|---|---|---|---|---|")
        for c in out["comparisons"]:
            f, a_ = c["f1_macro"], c["auprc_macro"]
            L.append(f"| {c['other']} | {f['diff']:+.3f} ({f['ci'][0]:+.3f}, {f['ci'][1]:+.3f}) | {f['p_value']:.3f} | "
                     f"{a_['diff']:+.3f} ({a_['ci'][0]:+.3f}, {a_['ci'][1]:+.3f}) | {a_['p_value']:.3f} |")

    for c in out.get("subset_comparisons", []):
        L.append(f"\n## Table H. {c['name']} vs the main model on the same {c['n']} test reports (threshold 0.5)\n")
        L.append("| Model | Macro-F1 | Micro-F1 | Any-error sensitivity | Any-error specificity | Any-error F1 |")
        L.append("|---|---|---|---|---|---|")
        for who, name in (("other", c["name"]), ("main", runs[main]["name"])):
            s_, a_ = c[who]["summary"], c[who]["any"]
            L.append(f"| {name} | {fmt(s_['f1_macro'])} | {fmt(s_['f1_micro'])} | {fmt(a_['sensitivity'])} | "
                     f"{fmt(a_['specificity'])} | {fmt(a_['f1'])} |")

    sent = [k for k, v in runs.items() if v["meta"].get("task") == "sentence"]
    if sent:
        L.append("\n## Table G. Sentence-level error localization\n")
        L.append("| Model | Sentences | Localization AUROC | AUPRC | F1 | Type macro-F1 | Report-level macro-F1 (aggregated) |")
        L.append("|---|---|---|---|---|---|---|")
        for k in sent:
            s, r = runs[k]["sentence"], runs[k]["report"]
            L.append(f"| {runs[k]['name']} | {s['n_sentences']} | {fmt(s['localization']['auroc'])} | "
                     f"{fmt(s['localization']['auprc'])} | {fmt(s['localization']['f1'])} | {fmt(s['type']['f1_macro'])} | "
                     f"{fmt(r['native']['f1_macro'])} |")
    with open(path, "w") as f:
        f.write("\n".join(L) + "\n")


def error_examples(run_dir, data_dir, path, k=5):
    import pandas as pd
    from rexerr.data import build_report_task

    meta = load_meta(run_dir)
    d = load_predictions(run_dir, "test")
    frame, _ = build_report_task(data_dir, "test", meta.get("label_source", "sentence"),
                                 bool(meta.get("include_clean", 1)))
    text = dict(zip(frame["uid"], frame["text"]))
    rows = []
    for j, c in enumerate(NATIVE):
        y, p = d["y"][:, j], d["p"][:, j]
        if y.sum() == 0:
            continue
        fp = np.where(y == 0)[0][np.argsort(-p[y == 0])][:k]
        fn = np.where(y == 1)[0][np.argsort(p[y == 1])][:k]
        for kind, idx in (("false_positive", fp), ("false_negative", fn)):
            for i in idx:
                rows.append({"type": c, "kind": kind, "prob": float(p[i]), "variant": d["variant"][i],
                             "true_types": ";".join(NATIVE[t] for t in np.where(d["y"][i])[0]),
                             "uid": d["uid"][i], "text": text.get(d["uid"][i], "")})
    orig = np.where(d["variant"] == "original")[0]
    s = d["p"][orig].max(1)
    for i in orig[np.argsort(-s)][: 2 * k]:
        rows.append({"type": NATIVE[int(d["p"][i].argmax())], "kind": "false_alarm_on_original_report",
                     "prob": float(d["p"][i].max()), "variant": "original", "true_types": "",
                     "uid": d["uid"][i], "text": text.get(d["uid"][i], "")})
    pd.DataFrame(rows).to_csv(path, index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_dir", default="results/runs")
    ap.add_argument("--out_dir", default="results")
    ap.add_argument("--main", default="biobert", help="substring identifying the main model's runs")
    ap.add_argument("--n_boot", type=int, default=1000)
    ap.add_argument("--data_dir", default=None)
    args = ap.parse_args()

    runs = {}
    for rd in sorted(glob.glob(os.path.join(args.runs_dir, "*"))):
        if not os.path.exists(os.path.join(rd, "preds_test.npz")):
            continue
        key = os.path.basename(rd)
        meta = load_meta(rd)
        val, test = load_predictions(rd, "val"), load_predictions(rd, "test")
        entry = {"dir": rd, "meta": {k: v for k, v in meta.items() if k != "history"}, "name": display(meta)}
        if meta.get("task") == "sentence":
            entry["sentence"] = evaluate_sentence_level(test)
            val, test = aggregate_sentences(val), aggregate_sentences(test)
        entry["report"] = evaluate_report_level(val, test, args.n_boot)
        runs[key] = entry
        print(f"[eval] {key}: macro-F1 {entry['report']['native']['f1_macro']:.4f} "
              f"macro-AUPRC {entry['report']['native']['auprc_macro']:.4f}", flush=True)

    # Seed aggregation: same model/task/labels -> one group.
    groups = defaultdict(list)
    for key, e in runs.items():
        m = e["meta"]
        gid = "|".join(str(m.get(x)) for x in ("model", "task", "label_source", "include_clean", "loss"))
        gid += "|" + str((m.get("args") or {}).get("tag", ""))
        groups[gid].append(key)
    out_groups = {}
    for gid, keys in groups.items():
        vals = {k: [] for k in HEADLINE}
        for key in keys:
            r = runs[key]["report"]
            for k in HEADLINE[:4]:
                vals[k].append(r["native"][k])
            vals["any_auroc"].append(r["any_error"]["at_0.5"]["auroc"])
            vals["any_auprc"].append(r["any_error"]["at_0.5"]["auprc"])
        out_groups[gid] = {"name": runs[keys[0]]["name"], "runs": keys,
                           "mean": {k: float(np.mean(v)) for k, v in vals.items()},
                           "sd": {k: float(np.std(v, ddof=1)) if len(v) > 1 else 0.0 for k, v in vals.items()}}

    # main model: standard configuration (report task, BCE loss, controls, sentence labels), lowest seed
    def is_standard(m):
        return (m.get("task") == "report" and m.get("loss", "bce") in (None, "bce")
                and str(m.get("include_clean", 1)) == "1" and m.get("label_source", "sentence") == "sentence")
    main_keys = [k for k in runs if args.main in k and is_standard(runs[k]["meta"])]
    main_run = min(main_keys, key=lambda k: (runs[k]["meta"].get("seed", 0), k)) if main_keys else None
    comparisons = []
    if main_run:
        d_main = load_predictions(runs[main_run]["dir"], "test")
        f1 = lambda y, p: M.summary(y, p, 0.5, mask=M.active_classes(d_main["y"]))["f1_macro"]  # noqa: E731
        ap_ = lambda y, p: M.summary(y, p, 0.5, mask=M.active_classes(d_main["y"]))["auprc_macro"]  # noqa: E731
        for gid, g in out_groups.items():
            other = sorted(g["runs"])[0]
            if other == main_run or main_run in g["runs"] or runs[other]["meta"].get("task") != "report":
                continue
            d_o = load_predictions(runs[other]["dir"], "test")
            if not np.array_equal(d_o["uid"], d_main["uid"]):
                continue
            comparisons.append({
                "other": g["name"],
                "f1_macro": M.paired_bootstrap(f1, d_main["y"], d_main["p"], d_o["p"], d_main["subject_id"], args.n_boot),
                "auprc_macro": M.paired_bootstrap(ap_, d_main["y"], d_main["p"], d_o["p"], d_main["subject_id"], args.n_boot),
            })

    subset = []
    if main_run:
        pos = {u: i for i, u in enumerate(d_main["uid"])}
        for key, e in runs.items():
            d_o = load_predictions(e["dir"], "test")
            if key == main_run or e["meta"].get("task") != "report" or len(d_o["uid"]) >= len(d_main["uid"]):
                continue
            if not all(u in pos for u in d_o["uid"]):
                continue
            idx = np.array([pos[u] for u in d_o["uid"]])
            y = d_main["y"][idx].astype(int)
            mask = M.active_classes(y)
            row = {"name": e["name"], "n": int(len(idx))}
            for who, p in (("other", d_o["p"]), ("main", d_main["p"][idx])):
                row[who] = {"summary": M.summary(y, p, 0.5, mask=mask), "any": M.binary(y.max(1), M.any_error_score(p)),
                            "per_class": M.per_class(y, p, NATIVE, 0.5)}
            subset.append(row)

    out = {"runs": runs, "groups": out_groups, "main_run": main_run, "comparisons": comparisons,
           "subset_comparisons": subset}
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "summary.json"), "w") as f:
        json.dump(out, f, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    write_tables(out, os.path.join(args.out_dir, "tables.md"))
    if main_run and args.data_dir:
        error_examples(runs[main_run]["dir"], args.data_dir, os.path.join(args.out_dir, "error_examples.csv"))
    print(open(os.path.join(args.out_dir, "tables.md")).read())


if __name__ == "__main__":
    main()
