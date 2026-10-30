"""Build the paraphrase control: error-free original test reports with some or all sentences reworded by an LLM.

Every reworded sentence from scripts/paraphrase.py is checked against its original; a rewrite is used only if it
keeps the numbers, sides, negation, uncertainty, severity and size words, device words and "___" placeholders, and
its length is within a factor of two. Two variants are written for every original report with at least one usable
rewrite, together with the original itself:
  partial: three randomly chosen sentences reworded (all usable ones if fewer), about as many sentences as ReXErr
           changes in an error report;
  full:    every sentence with a usable rewrite reworded.
All three are labelled error-free. The output contains report text, so it is not distributed with the code.

With --target errors (sentences from scripts/paraphrase.py --target errors), the erroneous sentences of each
error-injected report are replaced by their checked rewrites, which keep the error; the report keeps its labels, and
both the error-injected report and its reworded version are written (variants "error" and "error_reworded").

usage: python scripts/make_paraphrase_control.py --sentences paraphrase_sentences.csv --data_dir DATA \
           --out paraphrase_control.csv --stats results/paraphrase_checks.json [--target errors]
"""
import argparse
import json
import os
import re
import sys
from collections import Counter
from difflib import SequenceMatcher

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from rexerr.data import build_report_task, load_sentences, normalize_text  # noqa: E402
from rexerr.labels import NATIVE  # noqa: E402

NEGATION = r"\b(no|not|without|negative|absent|absence|none|neither|nor|free of|cannot|n't)\b"
UNCERTAIN = r"\b(may|might|could|possibl\w*|probabl\w*|likely|unlikely|suggest\w*|question\w*|concern\w*|" \
            r"cannot be (?:excluded|ruled out)|equivocal|indeterminate|versus|vs)\b"
SEVERITY = r"\b(mild\w*|moderate\w*|severe\w*|small|smaller|large|larger|minimal\w*|trace|tiny|massive|extensive\w*|" \
           r"slight\w*|subtle|marked\w*|significant\w*)\b"
DEVICE = r"\b(tube|line|catheter|pacemaker|pacer|wire|wires|lead|leads|clip|clips|valve|port|drain|stent|" \
         r"electrode\w*|picc|defibrillator|icd|sternotomy|prosthes\w*|hardware|coil\w*)\b"
CHECKS = {
    "numbers": lambda s: sorted(re.findall(r"\d+(?:\.\d+)?", s)),
    "sides": lambda s: (len(re.findall(r"\bleft\b", s)) > 0, len(re.findall(r"\bright\b", s)) > 0,
                        bool(re.search(r"\b(bilateral\w*|both)\b", s))),
    "negation": lambda s: bool(re.search(NEGATION, s)),
    "uncertainty": lambda s: bool(re.search(UNCERTAIN, s)),
    "severity": lambda s: sorted(set(re.findall(SEVERITY, s))),
    "devices": lambda s: sorted(set(re.sub(r"s$", "", w) for w in re.findall(DEVICE, s))),
    "placeholders": lambda s: s.count("___"),
}


def canon(s):
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()


def check(original, rewrite):
    """Names of the checks a rewrite fails ([] = usable), or ["unchanged"] / ["empty"]."""
    if not isinstance(rewrite, str) or not rewrite.strip():
        return ["empty"]
    if canon(original) == canon(rewrite):
        return ["unchanged"]
    o, r = original.lower(), rewrite.lower()
    failed = [name for name, f in CHECKS.items() if f(o) != f(r)]
    if not 0.5 <= len(rewrite) / max(1, len(original)) <= 2.0:
        failed.append("length")
    return failed


def locate(text, g):
    """(start, end, row) of every sentence in the report, sorted by start.

    The sentence-level file does not always list sentences in report order, so each one takes its first occurrence
    that does not overlap a sentence already placed.
    """
    spans = []
    for i in g.sort_values("position").index:
        s, start = g.loc[i, "original"], text.find(g.loc[i, "original"])
        while any(start < e and start + len(s) > b for b, e, _ in spans):
            start = text.find(s, start + 1)
        assert start >= 0, s
        spans.append((start, start + len(s), i))
    return sorted(spans)


def word_change(a, b):
    """Share of words not kept in order (1 - word-level similarity)."""
    return 1 - SequenceMatcher(None, canon(a).split(), canon(b).split()).ratio()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sentences", required=True)
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--stats", required=True)
    ap.add_argument("--k_partial", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--target", choices=["originals", "errors"], default="originals")
    args = ap.parse_args()

    df = pd.read_csv(args.sentences, keep_default_na=False)
    df["failed"] = [check(o, p) for o, p in zip(df["original"], df["paraphrase"])]
    df["usable"] = [not f for f in df["failed"]]
    fails = Counter(x for f in df["failed"] for x in f)
    if args.target == "errors":
        return reword_errors(df, fails, args)

    frame, _ = build_report_task(args.data_dir, "test", include_clean=True)
    orig = frame[frame["variant"] == "original"].set_index("study_id")
    rng = np.random.default_rng(args.seed)
    rows, n_partial, n_full, frac_full = [], [], [], []
    for sid, g in df.groupby("study_id", sort=False):
        text = orig.loc[sid, "text"]
        subject = int(orig.loc[sid, "subject_id"])
        use = g[g["usable"]]
        if use.empty:
            continue
        pick = set(rng.choice(use.index.to_numpy(), size=min(args.k_partial, len(use)), replace=False))
        spans = locate(text, g)
        variants = {}
        for name, chosen in (("para_partial", pick), ("para_full", set(use.index))):
            parts, cursor = [], 0
            for start, end, i in spans:
                parts += [text[cursor:start], g.loc[i, "paraphrase"] if i in chosen else text[start:end]]
                cursor = end
            variants[name] = normalize_text("".join(parts) + text[cursor:])
        rows.append((f"{sid}_o", sid, subject, "original", text, ""))
        rows.append((f"{sid}_pp", sid, subject, "para_partial", variants["para_partial"], ""))
        rows.append((f"{sid}_pf", sid, subject, "para_full", variants["para_full"], ""))
        n_partial.append(len(pick))
        n_full.append(len(use))
        frac_full.append(len(use) / len(g))
    out = pd.DataFrame(rows, columns=["uid", "study_id", "subject_id", "variant", "text", "error_type"])
    out.to_csv(args.out, index=False)

    # How much ReXErr itself rewrites an erroneous sentence, for comparison with the paraphrases.
    sent = load_sentences(args.data_dir, "test")
    err = sent[sent["error_present"] == 1]
    rexerr_change = [word_change(normalize_text(a), normalize_text(b))
                     for a, b in zip(err["original_sentence"], err["error_sentence"]) if isinstance(a, str) and a]
    stats = {
        "sentences": len(df), "usable": int(df["usable"].sum()), "usable_frac": float(df["usable"].mean()),
        "failed_checks": dict(fails.most_common()),
        "reports": int(df["study_id"].nunique()), "reports_used": len(n_full),
        "partial_sentences_mean": float(np.mean(n_partial)), "full_sentences_mean": float(np.mean(n_full)),
        "full_share_of_sentences_reworded": float(np.mean(frac_full)),
        "word_change_paraphrase_median": float(np.median([word_change(o, p) for o, p, u in
                                                          zip(df["original"], df["paraphrase"], df["usable"]) if u])),
        "word_change_rexerr_error_sentence_median": float(np.median(rexerr_change)),
    }
    with open(args.stats, "w") as f:
        json.dump(stats, f, indent=1)
    print(f"wrote {args.out}: {len(out)} reports")
    print(json.dumps(stats, indent=1))


def reword_errors(df, fails, args):
    frame, Y = build_report_task(args.data_dir, "test", include_clean=True)
    err = frame["variant"] == "error"
    reports = frame[err].set_index("study_id")
    labels = {sid: "|".join(NATIVE[j] for j in np.flatnonzero(y)) for sid, y in zip(frame["study_id"][err], Y[err])}
    rows, n_used, n_all = [], [], []
    for sid, g in df.groupby("study_id", sort=False):
        use = g[g["usable"]]
        if use.empty or not labels.get(sid):
            continue
        text, subject = reports.loc[sid, "text"], int(reports.loc[sid, "subject_id"])
        parts, cursor = [], 0
        for start, end, i in locate(text, use):
            parts += [text[cursor:start], use.loc[i, "paraphrase"]]
            cursor = end
        rows.append((f"{sid}_e", sid, subject, "error", text, labels[sid]))
        rows.append((f"{sid}_er", sid, subject, "error_reworded", normalize_text("".join(parts) + text[cursor:]),
                     labels[sid]))
        n_used.append(len(use))
        n_all.append(len(g))
    out = pd.DataFrame(rows, columns=["uid", "study_id", "subject_id", "variant", "text", "error_type"])
    out.to_csv(args.out, index=False)
    stats = {
        "sentences": len(df), "usable": int(df["usable"].sum()), "usable_frac": float(df["usable"].mean()),
        "failed_checks": dict(fails.most_common()), "reports": int(df["study_id"].nunique()), "reports_used": len(n_used),
        "reworded_sentences_per_report_mean": float(np.mean(n_used)),
        "share_of_eligible_error_sentences_reworded": float(np.sum(n_used) / np.sum(n_all)),
        "usable_by_type": {k: int(v) for k, v in df[df["usable"]]["error_type"].value_counts().items()},
        "word_change_paraphrase_median": float(np.median([word_change(o, p) for o, p, u in
                                                          zip(df["original"], df["paraphrase"], df["usable"]) if u])),
    }
    with open(args.stats, "w") as f:
        json.dump(stats, f, indent=1)
    print(f"wrote {args.out}: {len(out)} reports")
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
