"""Dataset audit for ReXErr-v1.

Produces results/audit.json with the statistics reported in the revised
manuscript: split sizes, patient overlap, label-source agreement, label
prevalence, report lengths, whitespace shortcut, and the leakage that a random
report-level split would introduce.

usage: python scripts/audit.py --data_dir DATA --out results/audit.json [--tokenizer dmis-lab/biobert-v1.1]
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from rexerr.data import SPLITS, load_reports, load_sentences, normalize_text, sentence_label_sets  # noqa: E402
from rexerr.labels import NATIVE, GROUP6, group6_matrix, parse_sampled  # noqa: E402
from rexerr.data import to_multihot  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--out", default="results/audit.json")
    ap.add_argument("--tokenizer", default=None, help="HF tokenizer for token-length stats")
    args = ap.parse_args()

    out = {"splits": {}, "labels": {}}
    reps = {s: load_reports(args.data_dir, s) for s in SPLITS}
    sents = {s: load_sentences(args.data_dir, s) for s in SPLITS}

    # Split sizes and patient overlap.
    for s in SPLITS:
        r = reps[s]
        out["splits"][s] = {
            "reports": int(len(r)), "subjects": int(r["subject_id"].nunique()),
            "unique_study_ids": int(r["study_id"].nunique()),
            "sentences": int(len(sents[s])),
            "studies_with_sentence_labels": int(sents[s]["study_id"].nunique()),
        }
    subj = {s: set(reps[s]["subject_id"]) for s in SPLITS}
    out["patient_overlap"] = {
        "train_test": len(subj["train"] & subj["test"]), "train_val": len(subj["train"] & subj["val"]),
        "val_test": len(subj["val"] & subj["test"]),
    }

    # Exact duplicate report texts across splits (templated normal reports).
    norm = {s: set(reps[s]["original_report"].map(normalize_text)) for s in SPLITS}
    test_norm = reps["test"]["original_report"].map(normalize_text)
    out["duplicates"] = {
        "test_original_reports_with_identical_text_in_train": int(test_norm.isin(norm["train"]).sum()),
        "test_reports": int(len(test_norm)),
    }
    out["unchanged_error_reports"] = {
        s: int((reps[s]["error_report"].map(normalize_text) == reps[s]["original_report"].map(normalize_text)).sum())
        for s in SPLITS
    }

    # Whitespace / formatting shortcut.
    def frac(series, pat):
        return float(series.fillna("").str.contains(pat, regex=True).mean())
    r = reps["train"]
    out["formatting_shortcut_train"] = {
        "original_with_double_space": frac(r["original_report"], r"  "),
        "error_with_double_space": frac(r["error_report"], r"  "),
        "original_with_trailing_space": frac(r["original_report"], r"\s$"),
        "error_with_trailing_space": frac(r["error_report"], r"\s$"),
        "original_with_newline": frac(r["original_report"], r"\n"),
        "error_with_newline": frac(r["error_report"], r"\n"),
    }

    # Lengths.
    lens = {}
    for v in ("original_report", "error_report"):
        t = r[v].map(normalize_text)
        w = t.str.split().str.len()
        lens[v] = {"chars_mean": float(t.str.len().mean()), "chars_median": float(t.str.len().median()),
                   "words_mean": float(w.mean()), "words_median": float(w.median()),
                   "words_p05": float(w.quantile(0.05)), "frac_under_50_words": float((w < 50).mean())}
    out["lengths_train"] = lens
    if args.tokenizer:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(args.tokenizer)
        sample = r["error_report"].map(normalize_text).sample(min(20000, len(r)), random_state=0).tolist()
        n = np.array([len(x) for x in tok(sample, add_special_tokens=True)["input_ids"]])
        out["tokens_error_report_train"] = {
            "tokenizer": args.tokenizer, "mean": float(n.mean()), "median": float(np.median(n)),
            "p95": float(np.percentile(n, 95)), "p99": float(np.percentile(n, 99)),
            "max": int(n.max()), "frac_over_256": float((n > 256).mean()), "frac_over_512": float((n > 512).mean()),
        }

    # Label sources: errors_sampled vs sentence-level annotations.
    for s in SPLITS:
        rep = reps[s]
        sampled = [set(parse_sampled(v)) for v in rep["errors_sampled"]]
        sent_sets = sentence_label_sets(sents[s])
        sent = [sent_sets.get(x, set()) for x in rep["study_id"]]
        ys, yt = to_multihot(sampled), to_multihot(sent)
        per = {}
        for j, k in enumerate(NATIVE):
            a, b = ys[:, j] == 1, yt[:, j] == 1
            per[k] = {
                "prevalence_sampled": float(a.mean()), "prevalence_sentence": float(b.mean()),
                "sampled_confirmed_by_sentence": float((a & b).sum() / a.sum()) if a.sum() else None,
                "sentence_also_sampled": float((a & b).sum() / b.sum()) if b.sum() else None,
            }
        g_s, g_t = group6_matrix(ys), group6_matrix(yt)
        out["labels"][s] = {
            "per_type": per,
            "mean_types_per_report_sampled": float(ys.sum(1).mean()),
            "mean_types_per_report_sentence": float(yt.sum(1).mean()),
            "exact_agreement_rate": float((ys == yt).all(1).mean()),
            "reports_with_no_sentence_level_error": float((yt.sum(1) == 0).mean()),
            "group6_prevalence_sampled": dict(zip(GROUP6, g_s.mean(0).round(4).tolist())),
            "group6_prevalence_sentence": dict(zip(GROUP6, g_t.mean(0).round(4).tolist())),
            "group6_mean_per_report_sampled": float(g_s.sum(1).mean()),
            "error_present_counts": {str(k): int(v) for k, v in sents[s]["error_present"].value_counts().items()},
        }

    # Leakage of a random 70/15/15 report-level split.
    allrep = pd.concat([reps[s] for s in SPLITS], ignore_index=True)
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(allrep))
    n_tr = int(0.70 * len(allrep)); n_va = int(0.15 * len(allrep))
    tr, te = allrep.iloc[perm[:n_tr]], allrep.iloc[perm[n_tr + n_va:]]
    out["random_split_leakage"] = {
        "test_reports_whose_patient_is_in_train": float(te["subject_id"].isin(set(tr["subject_id"])).mean()),
        "note": "70/15/15 random report-level split, seed 0",
    }

    # Clinician plausibility review shipped with the dataset.
    cr_path = os.path.join(args.data_dir, "clinician-review.csv")
    if os.path.exists(cr_path):
        cr = pd.read_csv(cr_path)
        out["clinician_review"] = {"n": int(len(cr)),
                                   "acceptable_rate": float(cr["acceptable"].astype(str).str.lower().str.startswith("y").mean())}

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
