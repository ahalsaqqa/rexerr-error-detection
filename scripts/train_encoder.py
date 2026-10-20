"""Fine-tune a transformer encoder for multi-label error detection (GPU).

Report task: input = report text, output = 12 native ReXErr error types.
Sentence task (localization): input = (sentence, full report), output = the
error type of that sentence (all zeros if the sentence is correct).

Checkpoints are written to --ckpt_dir (default: --out_dir) and training resumes automatically from them, so
a disconnected session can simply be re-run. On Colab, put them on local disk: files deleted from a mounted
Google Drive go to its trash, which counts toward the storage quota.

usage: python scripts/train_encoder.py --data_dir DATA --model dmis-lab/biobert-v1.1 --seed 0 \
           --out_dir results/runs/biobert_s0
"""
import argparse
import glob
import inspect
import os
import sys
import time

import numpy as np
import torch
from torch import nn
from transformers import (AutoModel, AutoTokenizer, BertModel, BertTokenizerFast, EarlyStoppingCallback, Trainer,
                          TrainingArguments, set_seed)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from rexerr.data import build_external_task, build_report_task, build_sentence_task  # noqa: E402
from rexerr.io import save_predictions  # noqa: E402
from rexerr.labels import NATIVE  # noqa: E402
from rexerr.metrics import summary  # noqa: E402


def load_encoder(name):
    try:
        model = AutoModel.from_pretrained(name)
    except ValueError:  # e.g. BlueBERT, whose config.json has no model_type
        model = BertModel.from_pretrained(name)
    for p in model.parameters():  # weights converted from TensorFlow can be non-contiguous, which safetensors rejects
        p.data = p.data.contiguous()
    return model


def load_tokenizer(name):
    try:
        return AutoTokenizer.from_pretrained(name)
    except ValueError:
        return BertTokenizerFast.from_pretrained(name, do_lower_case="uncased" in name.lower())


class MultiLabelEncoder(nn.Module):
    """Encoder + classification head on the [CLS] representation.

    loss: "bce" (unweighted), "weighted" (BCE with per-type positive weights = negatives / positives, capped at
    100) or "focal" (binary focal loss).
    """

    def __init__(self, name, n_labels, head="mlp", dropout=0.1, loss="bce", pos_weight=None, focal_gamma=2.0):
        super().__init__()
        self.encoder = load_encoder(name)
        h = self.encoder.config.hidden_size
        if head == "mlp":  # 768 -> 256 -> ReLU -> labels
            self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(h, 256), nn.ReLU(), nn.Dropout(dropout),
                                      nn.Linear(256, n_labels))
        else:
            self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(h, n_labels))
        self.loss, self.focal_gamma = loss, focal_gamma
        self.register_buffer("pos_weight", torch.as_tensor(pos_weight if pos_weight is not None else np.ones(n_labels),
                                                           dtype=torch.float32))

    def compute_loss(self, logits, labels):
        if self.loss == "weighted":
            return nn.functional.binary_cross_entropy_with_logits(logits, labels, pos_weight=self.pos_weight)
        bce = nn.functional.binary_cross_entropy_with_logits(logits, labels, reduction="none")
        if self.loss == "focal":
            p_t = torch.exp(-bce)
            return ((1 - p_t) ** self.focal_gamma * bce).mean()
        return bce.mean()

    def forward(self, input_ids, attention_mask, token_type_ids=None, labels=None):
        kw = {"input_ids": input_ids, "attention_mask": attention_mask}
        if token_type_ids is not None:
            kw["token_type_ids"] = token_type_ids
        logits = self.head(self.encoder(**kw).last_hidden_state[:, 0])
        out = {"logits": logits}
        if labels is not None:
            out["loss"] = self.compute_loss(logits.float(), labels.float())
        return out


class EncodedDataset(torch.utils.data.Dataset):
    def __init__(self, enc, labels):
        self.ids = [np.asarray(x, dtype=np.int32) for x in enc["input_ids"]]
        self.tt = [np.asarray(x, dtype=np.int8) for x in enc["token_type_ids"]] if "token_type_ids" in enc else None
        self.labels = labels

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        item = {"input_ids": self.ids[i], "labels": self.labels[i]}
        if self.tt is not None:
            item["token_type_ids"] = self.tt[i]
        return item


class Collator:
    def __init__(self, pad_id):
        self.pad_id = pad_id

    def __call__(self, batch):
        n = max(len(b["input_ids"]) for b in batch)
        ids = torch.full((len(batch), n), self.pad_id, dtype=torch.long)
        mask = torch.zeros((len(batch), n), dtype=torch.long)
        tt = torch.zeros((len(batch), n), dtype=torch.long) if "token_type_ids" in batch[0] else None
        for i, b in enumerate(batch):
            k = len(b["input_ids"])
            ids[i, :k] = torch.from_numpy(b["input_ids"].astype(np.int64))
            mask[i, :k] = 1
            if tt is not None:
                tt[i, :k] = torch.from_numpy(b["token_type_ids"].astype(np.int64))
        out = {"input_ids": ids, "attention_mask": mask,
               "labels": torch.tensor(np.stack([b["labels"] for b in batch]), dtype=torch.float32)}
        if tt is not None:
            out["token_type_ids"] = tt
        return out


def encode(tok, frame, task, max_len, lowercase):
    text = frame["text"].tolist()
    if lowercase:
        text = [t.lower() for t in text]
    if task == "sentence":
        ctx = frame["context"].tolist()
        if lowercase:
            ctx = [c.lower() for c in ctx]
        return tok(text, ctx, truncation="only_second", max_length=max_len)
    return tok(text, truncation=True, max_length=max_len)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def versions():
    import platform

    import sklearn
    import tokenizers
    import transformers
    return {"python": platform.python_version(), "torch": torch.__version__, "cuda": torch.version.cuda,
            "transformers": transformers.__version__, "tokenizers": tokenizers.__version__,
            "scikit-learn": sklearn.__version__, "numpy": np.__version__}


class OrderedEvalTrainer(Trainer):
    """Length-grouped batches for training only; evaluation and prediction keep dataset order.

    With transformers 5.x, train_sampling_strategy="group_by_length" also reorders Trainer.predict, which would
    misalign the predictions with the labels when they are saved (see scripts/reorder_predictions.py).
    """

    def _get_eval_sampler(self, eval_dataset):
        return torch.utils.data.SequentialSampler(eval_dataset)


def training_arguments(**kw):
    """TrainingArguments across transformers 4.x and 5.x (renamed/removed options)."""
    params = inspect.signature(TrainingArguments.__init__).parameters
    if "group_by_length" in params:
        kw["group_by_length"] = True
    elif "train_sampling_strategy" in params:
        kw["train_sampling_strategy"] = "group_by_length"
    if "eval_strategy" not in params:
        kw["evaluation_strategy"] = kw.pop("eval_strategy")
    return TrainingArguments(**{k: v for k, v in kw.items() if k in params})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--ckpt_dir", default=None, help="where checkpoints go (default: --out_dir)")
    ap.add_argument("--model", default="dmis-lab/biobert-v1.1")
    ap.add_argument("--task", choices=["report", "sentence"], default="report")
    ap.add_argument("--label_source", choices=["sentence", "sampled"], default="sentence")
    ap.add_argument("--include_clean", type=int, default=1, help="add original reports as negatives to TRAINING")
    ap.add_argument("--max_train_studies", type=int, default=None)
    ap.add_argument("--max_eval_studies", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--grad_accum", type=int, default=1)
    ap.add_argument("--max_len", type=int, default=320)
    ap.add_argument("--head", choices=["mlp", "linear"], default="mlp")
    ap.add_argument("--loss", choices=["bce", "weighted", "focal"], default="bce")
    ap.add_argument("--focal_gamma", type=float, default=2.0)
    ap.add_argument("--save_model", type=int, default=1, help="keep the best weights (fp16) for later re-scoring")
    ap.add_argument("--lowercase", type=int, default=0)
    ap.add_argument("--evals_per_epoch", type=int, default=4)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--max_steps", type=int, default=-1, help="debug only")
    ap.add_argument("--extra_eval", default=None, help="also score the external set from scripts/make_external.py")
    args = ap.parse_args()

    ckpt_dir = args.ckpt_dir or args.out_dir
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)  # the weights are saved before the prediction files create it
    t0 = time.time()
    set_seed(args.seed)
    # bf16 on Ampere or newer GPUs (L4, A100, H100, G4); fp16 on T4
    # is_bf16_supported() is also True on a T4, where bf16 is emulated and several times slower than fp16
    use_bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    builder = build_report_task if args.task == "report" else build_sentence_task
    kw = {} if args.task == "sentence" else {"label_source": args.label_source}
    tr, Ytr = builder(args.data_dir, "train", include_clean=bool(args.include_clean),
                      max_studies=args.max_train_studies, seed=args.seed, **kw)
    # Evaluation always includes the error-free originals, whatever was used for training.
    va, Yva = builder(args.data_dir, "val", include_clean=True, max_studies=args.max_eval_studies, **kw)
    te, Yte = builder(args.data_dir, "test", include_clean=True, max_studies=args.max_eval_studies, **kw)
    print(f"[data] train {len(tr)} val {len(va)} test {len(te)} ({time.time() - t0:.0f}s)", flush=True)

    tok = load_tokenizer(args.model)
    ds = {name: EncodedDataset(encode(tok, f, args.task, args.max_len, args.lowercase), y)
          for name, f, y in (("train", tr, Ytr), ("val", va, Yva), ("test", te, Yte))}
    trunc = float(np.mean([len(x) >= args.max_len for x in ds["train"].ids]))
    print(f"[tok] truncated at {args.max_len}: {trunc:.4%} ({time.time() - t0:.0f}s)", flush=True)

    pos, neg = Ytr.sum(0), len(Ytr) - Ytr.sum(0)
    pos_weight = np.where(pos > 0, np.clip(neg / np.maximum(pos, 1), 1.0, 100.0), 1.0)
    model = MultiLabelEncoder(args.model, len(NATIVE), head=args.head, loss=args.loss,
                              pos_weight=pos_weight if args.loss == "weighted" else None, focal_gamma=args.focal_gamma)
    n_params = sum(p.numel() for p in model.parameters())
    n_dev = max(1, torch.cuda.device_count())
    steps_per_epoch = max(1, len(ds["train"]) // (args.batch_size * args.grad_accum * n_dev))
    eval_steps = max(1, steps_per_epoch // args.evals_per_epoch)
    active = Ytr.sum(0) > 0

    def compute_metrics(pred):
        p = sigmoid(pred.predictions)
        return summary(pred.label_ids, p, 0.5, mask=active & (pred.label_ids.sum(0) > 0))

    total_steps = args.max_steps if args.max_steps > 0 else int(steps_per_epoch * args.epochs)
    targs = training_arguments(
        output_dir=ckpt_dir, seed=args.seed, data_seed=args.seed,
        num_train_epochs=args.epochs, max_steps=args.max_steps, learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size, per_device_eval_batch_size=args.batch_size * 2,
        gradient_accumulation_steps=args.grad_accum, warmup_steps=max(1, int(0.1 * total_steps)),
        weight_decay=0.01, lr_scheduler_type="linear", max_grad_norm=1.0,
        fp16=torch.cuda.is_available() and not use_bf16, bf16=use_bf16,
        eval_strategy="steps", eval_steps=eval_steps, save_strategy="steps", save_steps=eval_steps,
        save_total_limit=2, load_best_model_at_end=True, metric_for_best_model="f1_macro",
        greater_is_better=True, logging_steps=max(1, eval_steps // 4), report_to="none",
        dataloader_num_workers=args.workers, remove_unused_columns=False, disable_tqdm=True,
    )
    trainer = OrderedEvalTrainer(model=model, args=targs, train_dataset=ds["train"], eval_dataset=ds["val"],
                      data_collator=Collator(tok.pad_token_id), compute_metrics=compute_metrics,
                      callbacks=[EarlyStoppingCallback(early_stopping_patience=args.patience)])
    resume = bool(glob.glob(os.path.join(ckpt_dir, "checkpoint-*")))
    train_out = trainer.train(resume_from_checkpoint=True if resume else None)

    preds = {}
    for name, frame, y in (("val", va, Yva), ("test", te, Yte)):
        logits = trainer.predict(ds[name]).predictions
        preds[name] = sigmoid(logits)
    history = [h for h in trainer.state.log_history if "eval_f1_macro" in h or "loss" in h]
    if args.save_model:
        from safetensors.torch import save_file
        state = {k: v.detach().to("cpu", torch.float16).contiguous() for k, v in trainer.model.state_dict().items()}
        save_file(state, os.path.join(args.out_dir, "model_fp16.safetensors"))
    meta = {
        "model": args.model, "family": "encoder", "task": args.task, "seed": args.seed,
        "label_source": args.label_source, "include_clean": args.include_clean,
        "n_train": len(tr), "n_params": n_params, "truncated_frac": trunc,
        "best_checkpoint": trainer.state.best_model_checkpoint, "best_val_f1_macro": trainer.state.best_metric,
        "train_runtime_sec": train_out.metrics.get("train_runtime"), "total_runtime_sec": time.time() - t0,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "precision": "bf16" if use_bf16 else ("fp16" if torch.cuda.is_available() else "fp32"),
        "global_steps": trainer.state.global_step, "loss": args.loss,
        "pos_weight": pos_weight.tolist() if args.loss == "weighted" else None,
        "versions": versions(), "resumed": resume, "args": vars(args), "history": history,
    }
    save_predictions(args.out_dir, "val", va, Yva, preds["val"])
    save_predictions(args.out_dir, "test", te, Yte, preds["test"], meta)
    if args.extra_eval and args.task == "report":
        ex, Yex = build_external_task(args.extra_eval)
        ds_ex = EncodedDataset(encode(tok, ex, "report", args.max_len, args.lowercase), Yex)
        save_predictions(args.out_dir, "external", ex, Yex, sigmoid(trainer.predict(ds_ex).predictions))
    for ck in glob.glob(os.path.join(ckpt_dir, "checkpoint-*")):  # optimizer states are not needed any more
        import shutil
        shutil.rmtree(ck, ignore_errors=True)
    print(f"[done] {args.out_dir} best val f1_macro={trainer.state.best_metric} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
