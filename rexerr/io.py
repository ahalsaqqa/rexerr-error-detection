"""Shared prediction-file format so that every model is evaluated by the same code."""
import json
import os

import numpy as np

from .labels import NATIVE


def save_predictions(out_dir, split, frame, y, p, meta=None):
    os.makedirs(out_dir, exist_ok=True)
    extra = {c: frame[c].to_numpy(dtype=np.int64) for c in ("position", "error_present") if c in frame}
    np.savez_compressed(
        os.path.join(out_dir, f"preds_{split}.npz"),
        uid=np.array(frame["uid"].astype(str).tolist(), dtype=str),
        study_id=frame["study_id"].to_numpy(dtype=np.int64), subject_id=frame["subject_id"].to_numpy(dtype=np.int64),
        variant=np.array(frame["variant"].astype(str).tolist(), dtype=str),
        y=np.asarray(y, dtype=np.int8), p=np.asarray(p, dtype=np.float32), classes=np.array(NATIVE), **extra,
    )
    if meta is not None:
        with open(os.path.join(out_dir, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2, default=str)


def load_predictions(run_dir, split):
    d = np.load(os.path.join(run_dir, f"preds_{split}.npz"), allow_pickle=False)
    d = {k: d[k] for k in d.files}
    if load_meta(run_dir).get("family") == "llm_zeroshot":
        d["p"] = fold_contradictions(d["p"])
    return d


def fold_contradictions(p):
    """Count contradiction answers as false predictions.

    The sentence-level annotation labels contradiction sentences as false predictions, which fine-tuned models learn
    from the training data; a zero-shot model cannot, so its contradiction answers are moved to that type.
    """
    p = np.array(p, dtype=np.float32, copy=True)
    c, f = NATIVE.index("add_contradictions"), NATIVE.index("false_prediction")
    if len(p):
        p[:, f] = np.maximum(p[:, f], p[:, c])
        p[:, c] = 0
    return p


def load_meta(run_dir):
    path = os.path.join(run_dir, "meta.json")
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)
