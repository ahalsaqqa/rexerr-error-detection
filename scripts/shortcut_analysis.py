"""Do the detectors rely on how GPT-4o writes rather than on the errors? (CPU, saved predictions)

1. Style probe: classifiers trained on content-free style features only (length, sentence count and length, word
   length, type-token ratio, capitalization, digits, punctuation profile, reading ease) to separate error-injected from
   original reports. If style alone separated them, a detector could do the same without reading the errors.
2. Paired comparison: every error-injected report is compared with its own original (the same report before the
   errors were inserted), with the share of pairs in which the error report scores higher (paired AUROC) and the
   sensitivity, in subsets where a shortcut is unavailable:
     length-matched     same number of sentences and at most 2 words difference
     same punctuation   identical counts of every punctuation mark
     no GPT-4o vocabulary  the error report adds none of the k words or word pairs most over-represented in
                        error-injected training reports (log-odds with an informative prior)
3. False alarms and length: Spearman correlation of the any-error score with report length among original reports.

usage: python scripts/shortcut_analysis.py --data_dir DATA --runs results/runs/biobert_s0 results/runs/cxrbert_s0 \
           --out results/shortcuts.json
"""
import argparse
import json
import os
import re
import sys
from collections import Counter

import numpy as np
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
from operating_points import val_threshold  # noqa: E402
from rexerr import metrics as M  # noqa: E402
from rexerr.data import build_report_task  # noqa: E402
from rexerr.io import load_predictions  # noqa: E402
from rexerr.perturb import SENTENCE  # noqa: E402

PUNCT = [",", ".", ":", ";", "(", ")", "-", "/", "%"]
WORD = re.compile(r"[A-Za-z]+")
TOKEN = re.compile(r"[a-z]+")
STYLE_NAMES = (["log_chars", "log_words", "log_sentences", "words_per_sentence", "word_length", "type_token",
                "capitalized", "digits_per_100w", "long_words", "reading_ease"] + [f"punct_{p}" for p in PUNCT])


def syllables(w):
    return max(1, len(re.findall(r"[aeiouy]+", w.lower())))


def style(text):
    words = WORD.findall(text)
    n_w, n_s = max(1, len(words)), max(1, len(SENTENCE.split(text)))
    syl = [syllables(w) for w in words] or [1]
    feats = [np.log1p(len(text)), np.log1p(n_w), np.log1p(n_s), n_w / n_s, np.mean([len(w) for w in words] or [0]),
             len({w.lower() for w in words}) / n_w, sum(w[0].isupper() for w in words) / n_w,
             100 * sum(c.isdigit() for c in text) / n_w, sum(s >= 3 for s in syl) / n_w,
             206.835 - 1.015 * n_w / n_s - 84.6 * sum(syl) / n_w]
    return feats + [100 * text.count(p) / n_w for p in PUNCT]


def ngrams(text):
    t = TOKEN.findall(text.lower())
    return Counter(t + [a + " " + b for a, b in zip(t, t[1:])])


def characteristic(err_texts, orig_texts, k_max, min_count=20):
    """N-grams most over-represented in error-injected reports (Monroe et al. log-odds, informative Dirichlet prior)."""
    a, b = Counter(), Counter()
    for t in err_texts:
        a.update(ngrams(t))
    for t in orig_texts:
        b.update(ngrams(t))
    prior = a + b
    na, nb, n0 = sum(a.values()), sum(b.values()), sum(prior.values())
    scale = 0.01  # prior strength relative to the corpus counts
    z = {}
    for w, c in a.items():
        if c < min_count:
            continue
        aw = scale * prior[w]
        a0 = scale * n0
        d = np.log((c + aw) / (na + a0 - c - aw)) - np.log((b[w] + aw) / (nb + a0 - b[w] - aw))
        z[w] = d / np.sqrt(1 / (c + aw) + 1 / (b[w] + aw))
    return [w for w, _ in sorted(z.items(), key=lambda kv: -kv[1])[:k_max]]


def adds_any(err_text, orig_text, vocab):
    e, o = ngrams(err_text), ngrams(orig_text)
    return any(e[w] > o[w] for w in vocab)


def subset_metrics(s_e, s_o, s_neg, groups, thr, n_boot, k=None, rate_by_k=None, expected=None):
    def stat(se, so, g=None):
        return {"paired_auroc": float(np.mean((se > so) + 0.5 * (se == so))), "se_val": float(np.mean(se >= thr)),
                "se_0.5": float(np.mean(se >= 0.5))}
    out = {"n": int(len(s_e)), **stat(s_e, s_o)}
    if k is not None:  # subsets select reports with fewer errors, which are detected less often anyway
        out["mean_types"] = float(np.mean(k))
        out["se_val_expected_from_type_count"] = float(np.mean([rate_by_k[min(x, 4)] for x in k]))
        out["se_val_expected_from_types"] = float(np.mean(expected))
    out["auroc_vs_all_negatives"] = float(M._auroc(np.r_[np.ones(len(s_e)), np.zeros(len(s_neg))], np.r_[s_e, s_neg]))
    if len(s_e) >= 20:
        out["ci95"] = M.bootstrap(stat, [s_e, s_o], groups, n_boot=n_boot)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--train_studies", type=int, default=60000, help="studies used for the probe and the vocabulary")
    ap.add_argument("--vocab_k", nargs="+", type=int, default=[50, 200])
    ap.add_argument("--n_boot", type=int, default=1000)
    ap.add_argument("--out", default="results/shortcuts.json")
    args = ap.parse_args()
    out = {}

    tr, ytr = build_report_task(args.data_dir, "train", include_clean=True, max_studies=args.train_studies, seed=0)
    te, yte = build_report_task(args.data_dir, "test", include_clean=True)
    pos_tr, pos_te = ytr.max(1) > 0, yte.max(1) > 0
    X_tr = np.array([style(t) for t in tr["text"]])
    X_te = np.array([style(t) for t in te["text"]])
    out["style_probe"] = {"features": STYLE_NAMES, "train_reports": int(len(tr))}
    for name, clf in (("logistic", make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))),
                      ("gradient_boosting", HistGradientBoostingClassifier(max_iter=300, random_state=0))):
        clf.fit(X_tr, pos_tr)
        s = clf.predict_proba(X_te)[:, 1]
        out["style_probe"][name] = {"auroc": float(M._auroc(pos_te.astype(int), s)),
                                    "ci95": M.bootstrap(lambda y, p: {"auroc": M._auroc(y, p)},
                                                        [pos_te.astype(int), s], te["subject_id"].to_numpy(),
                                                        n_boot=args.n_boot)["auroc"]}

    err_tr = tr[(tr["variant"] == "error") & pos_tr]
    orig_tr = tr[tr["variant"] == "original"]
    vocab = characteristic(err_tr["text"], orig_tr["text"], max(args.vocab_k))
    out["gpt4o_vocabulary_top30"] = vocab[:30]

    text = dict(zip(te["uid"], te["text"]))
    for run in args.runs:
        d = load_predictions(run, "test")
        s, k = M.any_error_score(d["p"]), d["y"].sum(1)
        thr = val_threshold(run)
        score = dict(zip(d["uid"], s))
        sids = [u[:-2] for u, v, kk in zip(d["uid"], d["variant"], k) if v == "error" and kk > 0]
        subj = dict(zip(d["uid"], d["subject_id"]))
        neg = s[k == 0]
        kk_of = {u[:-2]: int(x) for u, v, x in zip(d["uid"], d["variant"], k) if v == "error"}
        y_of = {u[:-2]: yy for u, v, yy in zip(d["uid"], d["variant"], d["y"]) if v == "error"}
        pairs = []
        for sid in sids:
            e, o = text[f"{sid}_e"], text[f"{sid}_o"]
            fe, fo = style(e), style(o)
            pairs.append({"s_e": score[f"{sid}_e"], "s_o": score[f"{sid}_o"], "g": subj[f"{sid}_e"], "k": kk_of[sid], "y": y_of[sid],
                          "len_match": SENTENCE.split(e).__len__() == SENTENCE.split(o).__len__()
                          and abs(len(WORD.findall(e)) - len(WORD.findall(o))) <= 2,
                          "same_punct": all(e.count(p) == o.count(p) for p in PUNCT),
                          **{f"no_vocab_{kk}": not adds_any(e, o, vocab[:kk]) for kk in args.vocab_k}})
        res = {"threshold_val": float(thr)}
        all_k = np.array([min(p["k"], 4) for p in pairs])
        all_hit = np.array([p["s_e"] >= thr for p in pairs])
        rate_by_k = {x: float(all_hit[all_k == x].mean()) for x in range(1, 5)}
        # expected detection from which types a report contains (substitutions are harder than insertions)
        types = np.array([p["y"] for p in pairs])
        expected = LogisticRegression(max_iter=2000).fit(types, all_hit).predict_proba(types)[:, 1]
        for p, e in zip(pairs, expected):
            p["expected"] = e
        res["se_val_by_type_count"] = rate_by_k
        conds = ["all", "len_match", "same_punct"] + [f"no_vocab_{kk}" for kk in args.vocab_k]
        for c in conds:
            sel = [p for p in pairs if c == "all" or p[c]]
            res[c] = subset_metrics(np.array([p["s_e"] for p in sel]), np.array([p["s_o"] for p in sel]), neg,
                                    np.array([p["g"] for p in sel]), thr, args.n_boot,
                                    np.array([p["k"] for p in sel]), rate_by_k, [p["expected"] for p in sel])
        o_sel = d["variant"] == "original"
        n_words = np.array([len(WORD.findall(text[u])) for u in d["uid"][o_sel]])
        n_sents = np.array([len(SENTENCE.split(text[u])) for u in d["uid"][o_sel]])
        res["originals_score_vs_length"] = {"spearman_words": float(spearmanr(s[o_sel], n_words)[0]),
                                            "spearman_sentences": float(spearmanr(s[o_sel], n_sents)[0])}
        out[os.path.basename(run)] = res
        print(os.path.basename(run), json.dumps({c: (res[c]["n"], round(res[c]["paired_auroc"], 3), round(res[c]["se_val"], 3),
                                                     round(res[c]["se_val_expected_from_type_count"], 3),
                                                     round(res[c]["se_val_expected_from_types"], 3),
                                                     round(res[c]["mean_types"], 2)) for c in conds}))
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1, default=float)
    print("style probe:", {m: round(out["style_probe"][m]["auroc"], 3) for m in ("logistic", "gradient_boosting")})
    print("top GPT-4o n-grams:", vocab[:30])


if __name__ == "__main__":
    main()
