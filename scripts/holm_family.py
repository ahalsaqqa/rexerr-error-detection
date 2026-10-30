"""Holm adjustment over the seven pairwise encoder comparisons on macro-F1 reported in the paper.

Family: the three seed-averaged comparisons of RadBERT, CXR-BERT and BioBERT (results/seed_comparison.json) and the
seed-0 comparisons of BioBERT with the four encoders trained with one seed (results/summary.json, "comparisons").
With 1,000 bootstrap resamples a p-value of 0 means p < 0.002; 0.002 is used for it in the adjustment, so the
adjusted value is an upper bound.

usage: python scripts/holm_family.py --out results/holm.json
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from compare_seeds import holm  # noqa: E402

SINGLE_SEED = {"BERT-base (cased)": "BERT-base-cased", "Bio+ClinicalBERT": "Bio+ClinicalBERT",
               "BiomedBERT (PubMedBERT)": "BiomedBERT", "BlueBERT": "BlueBERT"}
NAMES = {"radbert": "RadBERT", "cxrbert": "CXR-BERT", "biobert": "BioBERT"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", default="results/summary.json")
    ap.add_argument("--seeds", default="results/seed_comparison.json")
    ap.add_argument("--n_boot", type=int, default=1000)
    ap.add_argument("--out", default="results/holm.json")
    args = ap.parse_args()
    floor = 2 / args.n_boot
    rows = []
    for key, res in json.load(open(args.seeds)).items():
        a, b = key.replace(" vs ", "-").split("-")
        r = res["f1_macro"]
        rows.append({"comparison": f"{NAMES[a]} − {NAMES[b]}", "basis": "mean of seeds 0–2", "diff": r["diff"],
                     "ci": r["ci"], "p": r["p_value"]})
    for c in json.load(open(args.summary))["comparisons"]:
        if c["other"] in SINGLE_SEED:
            r = c["f1_macro"]
            rows.append({"comparison": f"BioBERT − {SINGLE_SEED[c['other']]}", "basis": "seed 0", "diff": r["diff"],
                         "ci": r["ci"], "p": r["p_value"]})
    assert len(rows) == 7, len(rows)
    for r, q in zip(rows, holm([max(r["p"], floor) for r in rows])):
        r["p_holm"] = q
    with open(args.out, "w") as f:
        json.dump(rows, f, indent=1)
    for r in rows:
        print(f"{r['comparison']:<32} {r['basis']:<18} {r['diff']:+.3f} p={r['p']:.3f} Holm={r['p_holm']:.3f}")


if __name__ == "__main__":
    main()
