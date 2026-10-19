"""Non-neural baselines (CPU): report-length logistic regression and TF-IDF + logistic regression.

usage: python scripts/baselines.py --data_dir DATA --runs_dir results/runs --model tfidf|length
"""
import argparse
import os
import sys
import time

import numpy as np
from scipy.sparse import hstack
from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from rexerr.data import build_report_task  # noqa: E402
from rexerr.io import save_predictions  # noqa: E402


def length_features(texts):
    t = np.array(texts, dtype=object)
    chars = np.array([len(x) for x in t], dtype=float)
    words = np.array([len(x.split()) for x in t], dtype=float)
    sents = np.array([max(1, x.count(". ") + 1) for x in t], dtype=float)
    return np.log1p(np.stack([chars, words, sents], axis=1))


def fit_predict(X_tr, Y, X_eval, C):
    active = Y.sum(0) > 0
    # liblinear is not thread-safe, so the one-vs-rest fits run sequentially
    clf = OneVsRestClassifier(LogisticRegression(solver="liblinear", C=C, max_iter=1000))
    clf.fit(X_tr, Y[:, active])
    outs = []
    for X in X_eval:
        p = np.zeros((X.shape[0], Y.shape[1]), dtype=np.float32)
        p[:, active] = clf.predict_proba(X)
        outs.append(p)
    return outs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--runs_dir", default="results/runs")
    ap.add_argument("--model", choices=["tfidf", "length"], default="tfidf")
    ap.add_argument("--label_source", default="sentence", choices=["sentence", "sampled"])
    ap.add_argument("--include_clean", type=int, default=1)
    ap.add_argument("--max_train_studies", type=int, default=None)
    ap.add_argument("--C", type=float, default=4.0)
    ap.add_argument("--max_features", type=int, default=300000)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    t0 = time.time()
    tr, Ytr = build_report_task(args.data_dir, "train", args.label_source, bool(args.include_clean),
                                max_studies=args.max_train_studies)
    va, Yva = build_report_task(args.data_dir, "val", args.label_source, True)
    te, Yte = build_report_task(args.data_dir, "test", args.label_source, True)
    print(f"train {len(tr)} val {len(va)} test {len(te)} ({time.time() - t0:.0f}s)", flush=True)

    if args.model == "length":
        sc = StandardScaler().fit(length_features(tr.text))
        X_tr, X_va, X_te = (sc.transform(length_features(d.text)) for d in (tr, va, te))
        C = 1.0
    else:
        word = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=args.max_features,
                               sublinear_tf=True, dtype=np.float32, lowercase=False)
        # Character n-grams are hashed: an explicit vocabulary for 400k reports does not fit in 16 GB.
        hasher = HashingVectorizer(analyzer="char_wb", ngram_range=(3, 5), n_features=2 ** 20, alternate_sign=False,
                                   norm=None, dtype=np.float32, lowercase=False)
        char = TfidfTransformer(sublinear_tf=True)
        X_tr = hstack([word.fit_transform(tr.text), char.fit_transform(hasher.transform(tr.text))], format="csr",
                      dtype=np.float32)
        X_va = hstack([word.transform(va.text), char.transform(hasher.transform(va.text))], format="csr", dtype=np.float32)
        X_te = hstack([word.transform(te.text), char.transform(hasher.transform(te.text))], format="csr", dtype=np.float32)
        print(f"features {X_tr.shape[1]} nnz {X_tr.nnz} ({time.time() - t0:.0f}s)", flush=True)
        C = args.C

    p_va, p_te = fit_predict(X_tr, Ytr, [X_va, X_te], C)
    name = f"{args.model}_{args.label_source}_clean{args.include_clean}{args.tag}"
    out_dir = os.path.join(args.runs_dir, name)
    meta = {"model": args.model, "family": "baseline", "task": "report", "seed": 0,
            "label_source": args.label_source, "include_clean": args.include_clean,
            "n_train": len(tr), "runtime_sec": time.time() - t0, "args": vars(args)}
    save_predictions(out_dir, "val", va, Yva, p_va)
    save_predictions(out_dir, "test", te, Yte, p_te, meta)
    print(f"saved {out_dir} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
