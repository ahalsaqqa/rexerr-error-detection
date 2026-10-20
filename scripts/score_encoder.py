"""Score a CSV of reports with a fine-tuned encoder saved by train_encoder.py (--save_model 1) (GPU).

The CSV has the columns written by scripts/make_external.py or scripts/make_paraphrase_control.py; predictions are
saved next to the model as preds_<name>.npz, in the CSV's order.

usage: python scripts/score_encoder.py --run_dir results/runs/radbert_s0 --csv paraphrase_control.csv --name paraphrase
"""
import argparse
import os
import sys

import numpy as np
import torch
from safetensors.torch import load_file

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from rexerr.data import build_external_task  # noqa: E402
from rexerr.io import load_meta, save_predictions  # noqa: E402
from rexerr.labels import NATIVE  # noqa: E402
from train_encoder import Collator, EncodedDataset, MultiLabelEncoder, encode, load_tokenizer, sigmoid  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--batch_size", type=int, default=64)
    args = ap.parse_args()

    run = load_meta(args.run_dir)["args"]
    frame, Y = build_external_task(args.csv)
    tok = load_tokenizer(run["model"])
    ds = EncodedDataset(encode(tok, frame, "report", run["max_len"], run["lowercase"]), Y)
    model = MultiLabelEncoder(run["model"], len(NATIVE), head=run["head"])
    state = load_file(os.path.join(args.run_dir, "model_fp16.safetensors"))
    model.load_state_dict({k: v.float() for k, v in state.items()})
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()
    loader = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=False, collate_fn=Collator(tok.pad_token_id))
    logits = []
    with torch.no_grad(), torch.autocast(device, dtype=torch.bfloat16, enabled=device == "cuda"):
        for b in loader:
            b = {k: v.to(device) for k, v in b.items() if k != "labels"}
            logits.append(model(**b)["logits"].float().cpu().numpy())
    save_predictions(args.run_dir, args.name, frame, Y, sigmoid(np.concatenate(logits)))
    print(f"[done] {args.run_dir}/preds_{args.name}.npz ({len(frame)} reports)")


if __name__ == "__main__":
    main()
