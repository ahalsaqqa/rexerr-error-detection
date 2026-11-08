"""Human review of the rewording controls and of the cross-generator errors (same workbook format as label_audit.py).

make:    draw a blinded, shuffled sample of changed sentences from three sources and write a workbook plus an answer key:
           paraphrase  original-report sentences reworded by Qwen3-8B (expected: no error); half of them drawn from
                       reports that BioBERT flagged only after full rewording, to see whether those false alarms come
                       from real changes of meaning
           mirror      erroneous ReXErr sentences reworded by Qwen3-8B (expected: error still present, same type)
           rule        sentences changed by one cross-generator rule (expected: error present), per rule type
         Every row is compared with the original report, as in the label audit, and the reader also states whether the
         error could be recognized from the changed report alone.
analyze: share of each source judged erroneous, error types, recognizability, and agreement between two readers.

usage: python scripts/rewrite_audit.py make --data_dir DATA --paraphrase_sentences paraphrase_sentences.csv \
           --paraphrase_preds results/paraphrase_runs/biobert_s0_para/preds_paraphrase.npz \
           --error_sentences error_sentences.csv --external external_test.csv --out rewrite_audit.xlsx \
           --key rewrite_audit_key.csv --seed SEED
       python scripts/rewrite_audit.py analyze --key rewrite_audit_key.csv --sheets filled_A.xlsx [filled_B.xlsx]
"""
import argparse
import json
import os
import sys
from difflib import SequenceMatcher

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from label_audit import kappa, read_sheet, wilson, write_workbook  # noqa: E402
from make_paraphrase_control import check  # noqa: E402
from rexerr.data import load_reports, load_sentences, normalize_text  # noqa: E402
from rexerr.labels import NATIVE, NATIVE_DISPLAY, normalize_type  # noqa: E402
from rexerr.perturb import SENTENCE  # noqa: E402

SIZES = {"paraphrase_random": 15, "paraphrase_flagged": 15, "mirror": 15, "rule": 2}  # rule: per rule type
PURPOSE = ("We are checking sentences of de-identified chest X-ray reports (MIMIC-CXR, from the public ReXErr dataset) "
           "that were changed in different ways: some were reworded by a language model, some had an error inserted by "
           "a language model or by a simple rule. Some changed sentences contain an error and others do not; you will "
           "not be told which.")
RECOGNIZABLE = ("Recognizable without original?", 16, ["Yes", "No", "Unsure"],
                "Only if Yes in column E: could a radiologist notice this error from the changed report alone, without "
                "the original report or the image (for example a typo, a repetition or a statement that contradicts the "
                "rest of the report)? No: it reads as a plausible report and could only be caught with the original "
                "report or the image (for example a consistent swap of left and right).")


def sentence_diff(original, changed):
    """(original sentence, changed sentence) pairs that differ between two versions of a report."""
    a, b = SENTENCE.split(original), SENTENCE.split(changed)
    out = []
    for tag, i1, i2, j1, j2 in SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
        if tag == "replace":
            out.append((" ".join(a[i1:i2]), " ".join(b[j1:j2])))
        elif tag == "insert":
            out.append(("", " ".join(b[j1:j2])))
    return out


def pick(cand, n, rng, used):
    rows = []
    for i in rng.permutation(len(cand)):
        r = cand.iloc[i]
        if r["study_id"] in used:
            continue
        used.add(r["study_id"])
        rows.append(r)
        if len(rows) == n:
            break
    return rows


def sample(args):
    rng, used, rows = np.random.default_rng(args.seed), set(), []
    reports = load_reports(args.data_dir, "test").set_index("study_id")
    original = {sid: normalize_text(t) for sid, t in reports["original_report"].items()}

    para = pd.read_csv(args.paraphrase_sentences)
    para = para[[not check(o, r) for o, r in zip(para["original"], para["paraphrase"])]]
    z = np.load(args.paraphrase_preds)
    score = pd.DataFrame({"study_id": z["study_id"], "variant": z["variant"], "s": z["p"].max(axis=1)})
    s = score.pivot_table(index="study_id", columns="variant", values="s")
    flagged = set(s.index[(s["para_full"] >= 0.5) & (s["original"] < 0.5)])
    for name, cand in (("paraphrase_flagged", para[para["study_id"].isin(flagged)]),
                       ("paraphrase_random", para[~para["study_id"].isin(flagged)])):
        for r in pick(cand, SIZES[name], rng, used):
            rows.append({"source": name.split("_")[0], "subset": name, "study_id": int(r["study_id"]),
                         "expected_type": "", "original_sentence": normalize_text(r["original"]),
                         "modified_sentence": normalize_text(r["paraphrase"])})

    err = pd.read_csv(args.error_sentences)
    err = err[[not check(o, r) for o, r in zip(err["original"], err["paraphrase"])]]
    sent = load_sentences(args.data_dir, "test")
    truth = {(sid, normalize_text(e)): normalize_text(o) for sid, e, o, p in
             zip(sent["study_id"], sent["error_sentence"], sent["original_sentence"], sent["error_present"]) if p == 1}
    err = err.assign(true_original=[truth.get((sid, normalize_text(e))) for sid, e in zip(err["study_id"], err["original"])])
    for r in pick(err[err["true_original"].notna()], SIZES["mirror"], rng, used):
        rows.append({"source": "mirror", "subset": "mirror", "study_id": int(r["study_id"]),
                     "expected_type": normalize_type(r["error_type"]), "original_sentence": r["true_original"],
                     "modified_sentence": normalize_text(r["paraphrase"])})

    ext = pd.read_csv(args.external)
    pairs = ext.pivot_table(index="study_id", columns="variant", values="text", aggfunc="first")
    types = ext[ext["variant"] == "error"].set_index("study_id")["error_type"]
    changes = []
    for sid, t in types.items():
        d = sentence_diff(pairs.loc[sid, "original"], pairs.loc[sid, "error"])
        if len(d) == 1:  # one rule, one changed or added sentence
            changes.append({"study_id": sid, "type": t, "o": d[0][0], "e": d[0][1]})
    changes = pd.DataFrame(changes)
    for t in sorted(changes["type"].unique()):
        for r in pick(changes[changes["type"] == t], SIZES["rule"], rng, used):
            rows.append({"source": "rule", "subset": f"rule:{t}", "study_id": int(r["study_id"]), "expected_type": t,
                         "original_sentence": r["o"], "modified_sentence": r["e"]})

    df = pd.DataFrame(rows).sample(frac=1, random_state=args.seed).reset_index(drop=True)
    df["original_report"] = df["study_id"].map(original)
    df.insert(0, "item", [f"R{i + 1:03d}" for i in range(len(df))])
    return df


def read(path):
    df = pd.read_excel(path, sheet_name="Review", dtype=str).fillna("")
    out = read_sheet(path)
    out["recognizable"] = df[RECOGNIZABLE[0]] if RECOGNIZABLE[0] in df else ""
    return out


def analyze(key, sheets):
    k = pd.read_csv(key, keep_default_na=False)
    disp = {NATIVE_DISPLAY[c]: c for c in NATIVE}
    out, raters = {}, []
    for s in sheets:
        d = k.merge(read(s), on="item", how="left").fillna("")
        d["type_key"] = d["type"].map(lambda t: disp.get(t, "other" if t else ""))
        raters.append(d)
        res = {"answered": int(d["present"].ne("").sum())}
        for src, g in d[d["present"].isin(["Yes", "No"])].groupby("source"):
            y = g[g["present"] == "Yes"]
            res[src] = {"judged_erroneous": {"k": len(y), "n": len(g), "ci95": wilson(len(y), len(g))},
                        "recognizable_without_original": y["recognizable"].replace("", "missing").value_counts().to_dict(),
                        "significance": y["significance"].replace("", "missing").value_counts().to_dict()}
            if src != "paraphrase":
                res[src]["same_type"] = {"k": int((y["type_key"] == y["expected_type"]).sum()), "n": len(y)}
            if src == "paraphrase":
                res[src]["by_subset"] = {sub: {"k": int((gg["present"] == "Yes").sum()), "n": len(gg)}
                                         for sub, gg in g.groupby("subset")}
            if src == "rule":
                res[src]["by_rule"] = {sub: {"erroneous": int((gg["present"] == "Yes").sum()), "n": len(gg),
                                             "recognizable": int((gg["recognizable"] == "Yes").sum())}
                                       for sub, gg in g.groupby("subset")}
        out[os.path.basename(s)] = res
    if len(raters) == 2:
        a, b = raters
        both = a["present"].isin(["Yes", "No"]) & b["present"].isin(["Yes", "No"])
        out["inter_rater"] = {"n": int(both.sum()), "agreement": float((a["present"][both] == b["present"][both]).mean()),
                              "kappa_error_present": kappa(a["present"][both], b["present"][both])}
    print(json.dumps(out, indent=1, default=float))
    return out


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("make")
    for name in ("data_dir", "paraphrase_sentences", "paraphrase_preds", "error_sentences", "external", "out", "key"):
        m.add_argument(f"--{name}", required=True)
    m.add_argument("--seed", type=int, default=0)
    a = sub.add_parser("analyze")
    a.add_argument("--key", required=True)
    a.add_argument("--sheets", nargs="+", required=True)
    a.add_argument("--summary", default=None)
    args = ap.parse_args()
    if args.cmd == "make":
        df = sample(args)
        write_workbook(df, args.out, title_text="Radiology report rewording audit", purpose=PURPOSE, extra=RECOGNIZABLE)
        df[["item", "source", "subset", "study_id", "expected_type"]].to_csv(args.key, index=False)
        print(f"wrote {args.out} and {args.key}: {len(df)} rows;", df["subset"].value_counts().to_dict())
    else:
        res = analyze(args.key, args.sheets)
        if args.summary:
            with open(args.summary, "w") as f:
                json.dump(res, f, indent=1, default=float)


if __name__ == "__main__":
    main()
