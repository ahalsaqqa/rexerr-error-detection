"""Loading ReXErr-v1 and building the report-level and sentence-level tasks.

Design decisions (see README):
  * Official ReXErr/MIMIC-CXR train/val/test splits are used (patient-disjoint).
  * Each study contributes its error-injected report (labels = error types) and,
    when ``include_clean`` is set, its original report as an error-free
    negative (all-zero labels).
  * Primary labels come from the sentence-level annotations, which the dataset
    authors recommend over ``errors_sampled`` (sampled errors are not always
    present in the generated report).
  * Whitespace is collapsed because GPT-4o rewrites spacing, which would let a
    model separate original from error reports by formatting alone.
"""
import os
import re

import numpy as np
import pandas as pd

from .labels import NATIVE, normalize_type, parse_sampled

SPLITS = ("train", "val", "test")


def normalize_text(text):
    if not isinstance(text, str):
        return ""
    return re.sub(r"\s+", " ", text).strip()


def report_path(data_dir, split):
    return os.path.join(data_dir, "ReXErr-report-level", f"ReXErr-report-level_{split}.csv")


def sentence_path(data_dir, split):
    return os.path.join(data_dir, "ReXErr-sentence-level", f"ReXErr-sentence-level_{split}.csv")


def load_reports(data_dir, split):
    return pd.read_csv(report_path(data_dir, split))


def load_sentences(data_dir, split):
    return pd.read_csv(sentence_path(data_dir, split))


def sentence_label_sets(sent_df):
    """study_id -> set of native error types annotated at sentence level."""
    err = sent_df[sent_df["error_present"] == 1]
    out = {}
    for sid, t in zip(err["study_id"], err["error_type"]):
        key = normalize_type(t)
        if key:
            out.setdefault(sid, set()).add(key)
    return out


def to_multihot(label_sets):
    y = np.zeros((len(label_sets), len(NATIVE)), dtype=np.float32)
    idx = {k: i for i, k in enumerate(NATIVE)}
    for r, s in enumerate(label_sets):
        for k in s:
            y[r, idx[k]] = 1.0
    return y


def build_report_task(data_dir, split, label_source="sentence", include_clean=True,
                      max_studies=None, seed=0):
    """Return (frame, Y) for the report-level multi-label task.

    frame columns: uid, study_id, subject_id, variant ('error'|'original'), text
    Y: float32 array (n, 12) over ``labels.NATIVE``.
    """
    rep = load_reports(data_dir, split)
    if max_studies is not None and max_studies < len(rep):
        rng = np.random.default_rng(seed)
        subjects = rep["subject_id"].unique()
        rng.shuffle(subjects)
        keep, n = set(), 0
        counts = rep.groupby("subject_id").size()
        for s in subjects:
            if n >= max_studies:
                break
            keep.add(s)
            n += counts[s]
        rep = rep[rep["subject_id"].isin(keep)]

    if label_source == "sentence":
        sets = sentence_label_sets(load_sentences(data_dir, split))
        err_sets = [sets.get(s, set()) for s in rep["study_id"]]
    elif label_source == "sampled":
        err_sets = [set(parse_sampled(v)) for v in rep["errors_sampled"]]
    else:
        raise ValueError(label_source)

    rows, label_sets = [], []
    for (_, r), s in zip(rep.iterrows(), err_sets):
        err_text = normalize_text(r["error_report"])
        orig_text = normalize_text(r["original_report"])
        if err_text and err_text != orig_text:
            rows.append((f"{r.study_id}_e", r.study_id, r.subject_id, "error", err_text))
            label_sets.append(s)
        if include_clean and orig_text:
            rows.append((f"{r.study_id}_o", r.study_id, r.subject_id, "original", orig_text))
            label_sets.append(set())
    frame = pd.DataFrame(rows, columns=["uid", "study_id", "subject_id", "variant", "text"])
    return frame, to_multihot(label_sets)


def build_sentence_task(data_dir, split, include_clean=True, max_studies=None, seed=0):
    """Sentence-level localization task.

    Each example is (sentence, full report as context). Error sentences carry
    their error type; unchanged and 'neutral' (error_present == 2) sentences,
    and all sentences of the original report, are negatives.
    """
    sent = load_sentences(data_dir, split)
    rep = load_reports(data_dir, split).set_index("study_id")
    studies = sent["study_id"].unique()
    if max_studies is not None and max_studies < len(studies):
        rng = np.random.default_rng(seed)
        subj = sent.drop_duplicates("study_id").set_index("study_id")["subject_id"]
        subjects = subj.unique()
        rng.shuffle(subjects)
        keep, n = set(), 0
        per_subject = subj.groupby(subj).size()
        for s in subjects:
            if n >= max_studies:
                break
            keep.add(s)
            n += per_subject[s]
        sent = sent[sent["subject_id"].isin(keep)]

    rows, label_sets = [], []
    for sid, g in sent.groupby("study_id", sort=False):
        if sid not in rep.index:
            continue
        r = rep.loc[sid]
        err_ctx = normalize_text(r["error_report"])
        orig_ctx = normalize_text(r["original_report"])
        subject = int(g["subject_id"].iloc[0])
        for i, (es, ep, et) in enumerate(zip(g["error_sentence"], g["error_present"], g["error_type"])):
            es = normalize_text(es)
            if not es:
                continue
            key = normalize_type(et) if ep == 1 else None
            rows.append((f"{sid}_e{i}", sid, subject, "error", i, es, err_ctx, int(ep) if ep == ep else 0))
            label_sets.append({key} if key else set())
        if include_clean:
            for i, os_ in enumerate(g["original_sentence"]):
                os_ = normalize_text(os_)
                if not os_:
                    continue
                rows.append((f"{sid}_o{i}", sid, subject, "original", i, os_, orig_ctx, 0))
                label_sets.append(set())
    frame = pd.DataFrame(rows, columns=["uid", "study_id", "subject_id", "variant", "position",
                                        "text", "context", "error_present"])
    return frame, to_multihot(label_sets)


def build_external_task(path):
    """Report CSV written by scripts/make_external.py or scripts/make_paraphrase_control.py.

    error_type holds the native types of a report separated by "|" (empty for error-free reports).
    """
    df = pd.read_csv(path)
    sets = [set(t.split("|")) if isinstance(t, str) and t else set() for t in df["error_type"]]
    frame = df[["uid", "study_id", "subject_id", "variant", "text"]].reset_index(drop=True)
    return frame, to_multihot(sets)
