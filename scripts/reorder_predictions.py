"""Restore the row order of encoder predictions written before the evaluation-order fix in train_encoder.py.

With transformers 5.x, `train_sampling_strategy="group_by_length"` also makes Trainer.predict iterate in
length-grouped order (LengthGroupedSampler seeded with data_seed), while train_encoder.py paired the predictions
with the labels in dataset order. The sampler is deterministic, so this script rebuilds the evaluation datasets with
each run's tokenizer and arguments, recomputes the sampler order and writes the predictions back in dataset order.
A run is only rewritten when the reordered validation macro-F1 reproduces the best validation macro-F1 that the
Trainer logged during training; the original files are kept as preds_*.unordered.npz.

usage: python scripts/reorder_predictions.py --runs_dir results/runs --data_dir DATA [--eval_batch_size 64]
"""
import argparse
import glob
import json
import os
import shutil
import sys

import numpy as np
import torch
from transformers.trainer_pt_utils import LengthGroupedSampler

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
from rexerr.data import build_report_task, build_sentence_task  # noqa: E402
from rexerr.metrics import active_classes, summary  # noqa: E402
from train_encoder import EncodedDataset, encode, load_tokenizer  # noqa: E402


def sampler_order(ds, batch_size, seed):
    gen = torch.Generator().manual_seed(seed)
    return np.array(list(LengthGroupedSampler(batch_size, dataset=ds, generator=gen)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_dir", required=True)
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--eval_batch_size", type=int, default=64, help="per-device eval batch size x number of GPUs")
    ap.add_argument("--tol", type=float, default=2e-3)
    args = ap.parse_args()

    for meta_path in sorted(glob.glob(os.path.join(args.runs_dir, "*", "meta.json"))):
        run = os.path.dirname(meta_path)
        with open(meta_path) as f:
            meta = json.load(f)
        if meta.get("family") != "encoder" or os.path.exists(os.path.join(run, "preds_test.unordered.npz")):
            continue
        a = meta["args"]
        builder = build_report_task if a["task"] == "report" else build_sentence_task
        kw = {} if a["task"] == "sentence" else {"label_source": a["label_source"]}
        tok = load_tokenizer(a["model"])
        new = {}
        for split in ("val", "test"):
            frame, y = builder(args.data_dir, split, include_clean=True, max_studies=a.get("max_eval_studies"), **kw)
            ds = EncodedDataset(encode(tok, frame, a["task"], a["max_len"], a["lowercase"]), y)
            order = sampler_order(ds, args.eval_batch_size, a["seed"])
            d = dict(np.load(os.path.join(run, f"preds_{split}.npz")))
            assert len(order) == len(d["p"]) and np.array_equal(d["uid"], frame["uid"].astype(str).to_numpy())
            p = np.empty_like(d["p"])
            p[order] = d["p"]
            new[split] = {**d, "p": p}
        yv = new["val"]["y"].astype(int)
        f1 = summary(yv, new["val"]["p"], 0.5, mask=active_classes(yv))["f1_macro"]
        logged = meta["best_val_f1_macro"]
        ok = abs(f1 - logged) <= args.tol
        print(f"{os.path.basename(run)}: reordered val macro-F1 {f1:.4f}, logged {logged:.4f} -> "
              f"{'rewritten' if ok else 'NOT rewritten'}", flush=True)
        if not ok:
            continue
        for split, d in new.items():
            src = os.path.join(run, f"preds_{split}.npz")
            shutil.copy(src, os.path.join(run, f"preds_{split}.unordered.npz"))
            np.savez_compressed(src, **d)


if __name__ == "__main__":
    main()
