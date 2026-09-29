"""E5+ : fine-tune a stronger transformer (default DeBERTa-v3-base) on the v3 training data.

Differences from src/train_transformer.py (E4, DistilBERT):
  * backbone     - DeBERTa-v3 uses disentangled attention (content and word position are modelled separately),
                   which handles negation, contrast ("but"), sarcasm cues and long sentences better than DistilBERT.
  * data         - v3 training set (src/prepare_extra.py): + GoEmotions rows with a second emotion,
                   + mixed-emotion rows as soft targets, + ISEAR situation descriptions.
  * soft targets - the loss is cross-entropy against a probability distribution, so a text labelled
                   "joy + nervousness" teaches the model 50 % / 50 % instead of being thrown away.
  * optimisation - bf16 mixed precision, layer-wise learning-rate decay (lower layers change less),
                   length-grouped dynamic padding (much faster than padding every text to max length),
                   gradient accumulation, checkpoint selection twice per epoch.

Model selection uses only the frozen GoEmotions validation split (ISEAR validation is reported alongside).
The test sets are never touched here - use `python -m src.evaluate_v3` once for the chosen model.

Usage:
  uv run --extra transformer python -m src.train_v3 --exp-id E5 --out models/deberta_v3
"""
import argparse
import json
import math
import platform
import random
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, Sampler
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_cosine_schedule_with_warmup

from src import config
from src.data import load_classes, load_manifest, load_split, sha256_file
from src.metrics import (
    append_experiment,
    compute_metrics,
    expected_calibration_error,
    print_report,
    save_confusion_matrices,
    save_json,
    threshold_table,
)


class EncodedDataset(Dataset):
    def __init__(self, input_ids: list[list[int]], targets: np.ndarray):
        self.input_ids = input_ids
        self.targets = torch.as_tensor(targets, dtype=torch.float)

    def __len__(self) -> int:
        return len(self.input_ids)

    def __getitem__(self, i: int):
        return self.input_ids[i], self.targets[i]


class LengthGroupedSampler(Sampler):
    """Shuffle, then sort inside mega-batches by length: similar-length texts share a batch, so little padding."""

    def __init__(self, lengths: list[int], batch_size: int, seed: int):
        self.lengths, self.batch_size, self.seed, self.epoch = lengths, batch_size, seed, 0

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        self.epoch += 1
        idx = rng.permutation(len(self.lengths))
        mega = self.batch_size * 50
        batches = []
        for s in range(0, len(idx), mega):
            chunk = sorted(idx[s:s + mega], key=lambda i: self.lengths[i], reverse=True)
            batches += [chunk[b:b + self.batch_size] for b in range(0, len(chunk), self.batch_size)]
        rng.shuffle(batches)
        for b in batches:
            yield from b

    def __len__(self) -> int:
        return len(self.lengths)


def make_collate(pad_id: int):
    def collate(batch):
        ids, targets = zip(*batch)
        width = max(len(x) for x in ids)
        input_ids = torch.full((len(ids), width), pad_id, dtype=torch.long)
        mask = torch.zeros((len(ids), width), dtype=torch.long)
        for i, x in enumerate(ids):
            input_ids[i, :len(x)] = torch.as_tensor(x)
            mask[i, :len(x)] = 1
        return {"input_ids": input_ids, "attention_mask": mask}, torch.stack(targets)
    return collate


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def class_weights(mass: np.ndarray) -> torch.Tensor:
    """Square-root inverse class frequency (same rule as E4): helps small classes without drowning neutral."""
    w = np.sqrt(mass.max() / mass)
    return torch.tensor(w / w.mean(), dtype=torch.float)


def param_groups(model, lr: float, decay: float, weight_decay: float) -> list[dict]:
    """Layer-wise LR decay: the classifier learns at `lr`, each lower encoder layer at `decay` times the one above."""
    n_layers = model.config.num_hidden_layers
    groups: dict[tuple, dict] = {}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        m = re.search(r"\.layer\.(\d+)\.", name)
        if m:
            depth = int(m.group(1)) + 1
        elif "embeddings" in name or "rel_embeddings" in name or re.search(r"encoder\.LayerNorm", name):
            depth = 0
        else:
            depth = n_layers + 1                                   # pooler / classifier head
        scale = decay ** (n_layers + 1 - depth)
        wd = 0.0 if (p.ndim == 1 or name.endswith(".bias")) else weight_decay
        key = (depth, wd)
        groups.setdefault(key, {"params": [], "lr": lr * scale, "weight_decay": wd})["params"].append(p)
    return list(groups.values())


def encode(tokenizer, texts, max_len: int) -> list[list[int]]:
    return tokenizer(list(texts), truncation=True, max_length=max_len)["input_ids"]


@torch.inference_mode()
def collect_logits(model, ids: list[list[int]], device, pad_id: int, batch_size: int = 64) -> np.ndarray:
    model.eval()
    order = np.argsort([len(x) for x in ids])
    collate = make_collate(pad_id)
    out = np.zeros((len(ids), model.config.num_labels), dtype=np.float32)
    for s in range(0, len(order), batch_size):
        sel = order[s:s + batch_size]
        batch, _ = collate([(ids[i], torch.zeros(1)) for i in sel])
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            out[sel] = model(**batch).logits.float().cpu().numpy()
    return out


def fit_temperature(logits: np.ndarray, labels: np.ndarray) -> float:
    z, y = torch.tensor(logits), torch.tensor(labels)
    log_t = torch.zeros(1, requires_grad=True)
    optimizer = torch.optim.LBFGS([log_t], lr=0.1, max_iter=100)

    def closure():
        optimizer.zero_grad()
        loss = torch.nn.functional.cross_entropy(z / log_t.exp(), y)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(log_t.detach().exp())


def softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    p = np.exp(z)
    return p / p.sum(axis=1, keepdims=True)


def macro_f1(y_true_ids, logits, labels) -> float:
    return compute_metrics(np.asarray(y_true_ids), logits.argmax(1), labels)["macro_f1"]


def load_v3_train(classes: list[str], sources: list[str]) -> pd.DataFrame:
    manifest = json.loads((config.DATA_PROCESSED_V3 / "manifest.json").read_text())
    entry = manifest["files"]["train"]
    path = config.DATA_PROCESSED_V3 / entry["file"]
    if sha256_file(path) != entry["sha256"]:
        raise RuntimeError(f"{path} does not match data/processed_v3/manifest.json - rerun src.prepare_extra")
    if manifest["classes"] != classes:
        raise RuntimeError("class order in the v3 manifest differs from the v2 manifest")
    df = pd.read_parquet(path)
    return df[df["source"].isin(sources)].reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--exp-id", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model-name", default="microsoft/deberta-v3-base")
    parser.add_argument("--sources", default="goemotions,goemotions_partial,goemotions_mixed,isear",
                        help="comma list of v3 sources; 'goemotions' alone = the v2 training data")
    parser.add_argument("--epochs", type=float, default=4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--llrd", type=float, default=0.9, help="layer-wise LR decay factor (1.0 = off)")
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup", type=float, default=0.1)
    parser.add_argument("--max-len", type=int, default=256)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--freeze-embeddings", action=argparse.BooleanOptionalAction, default=True,
                        help="keep the (128k-token) word-embedding table fixed: saves ~1.2 GB GPU memory")
    parser.add_argument("--evals-per-epoch", type=int, default=2)
    parser.add_argument("--seed", type=int, default=config.SEED)
    parser.add_argument("--purpose", default="Stronger backbone + v3 data")
    parser.add_argument("--no-log", action="store_true", help="smoke test: do not write metrics/experiments.csv")
    args = parser.parse_args()

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else " - slow"))

    classes = load_classes()
    label2id = {c: i for i, c in enumerate(classes)}
    sources = [s.strip() for s in args.sources.split(",") if s.strip()]
    train = load_v3_train(classes, sources)
    targets = np.stack(train["targets"].to_numpy()).astype(np.float32)
    if args.label_smoothing:
        targets = targets * (1 - args.label_smoothing) + args.label_smoothing / len(classes)
    val = load_split("validation")
    y_val = val["label"].map(label2id).to_numpy()
    isear_val = pd.read_parquet(config.DATA_PROCESSED_V3 / "isear_validation.parquet")
    y_isear = isear_val["label"].map(label2id).to_numpy()
    isear_labels = sorted(set(y_isear))
    print(f"training rows {len(train):,} from {train['source'].value_counts().to_dict()}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name, num_labels=len(classes), id2label=dict(enumerate(classes)), label2id=label2id,
        dtype=torch.float32,  # the checkpoint is stored in fp16; training fp16 weights directly overflows (NaN)
    ).to(device)
    pad_id = tokenizer.pad_token_id
    if args.freeze_embeddings:
        model.get_input_embeddings().weight.requires_grad_(False)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"trainable parameters: {trainable / 1e6:.1f}M of {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")

    train_ids = encode(tokenizer, train["text"], args.max_len)
    val_ids = encode(tokenizer, val["text"], args.max_len)
    isear_ids = encode(tokenizer, isear_val["text"], args.max_len)
    truncated = sum(len(x) >= args.max_len for x in train_ids)
    print(f"max_len {args.max_len}: {truncated} training texts truncated")

    ds = EncodedDataset(train_ids, targets)
    sampler = LengthGroupedSampler([len(x) for x in train_ids], args.batch_size, args.seed)
    loader = DataLoader(ds, batch_size=args.batch_size, sampler=sampler, collate_fn=make_collate(pad_id))

    weights = class_weights(targets.sum(0)).to(device)
    loss_fn = torch.nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(param_groups(model, args.lr, args.llrd, args.weight_decay), lr=args.lr)
    steps_per_epoch = math.ceil(len(loader) / args.grad_accum)
    total_steps = int(steps_per_epoch * args.epochs)
    scheduler = get_cosine_schedule_with_warmup(optimizer, int(args.warmup * total_steps), total_steps)
    eval_every = max(1, steps_per_epoch // args.evals_per_epoch)

    settings = (f"{args.model_name}; data=v3[{'+'.join(sources)}] rows={len(train)}; epochs={args.epochs} "
                f"batch={args.batch_size}x{args.grad_accum} lr={args.lr} llrd={args.llrd} cosine warmup={args.warmup} "
                f"wd={args.weight_decay} max_len={args.max_len} bf16 length-grouped "
                f"{'frozen-word-embeddings ' if args.freeze_embeddings else ''}; soft-target CE "
                f"sqrt-inverse-freq weights; label_smoothing={args.label_smoothing}; seed={args.seed}")
    print(settings, flush=True)

    history, best = [], {"f1": -1.0}
    step, micro, running, t0 = 0, 0, 0.0, time.perf_counter()
    model.train()
    done = False
    while not done:
        for batch, target in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            target = target.to(device)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                logits = model(**batch).logits
            loss = loss_fn(logits.float(), target) / args.grad_accum
            loss.backward()
            running += loss.item()
            micro += 1
            if micro % args.grad_accum:
                continue
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            if step % 100 == 0:
                print(f"  step {step}/{total_steps} loss {running / 100:.4f} "
                      f"lr {scheduler.get_last_lr()[-1]:.2e} {time.perf_counter() - t0:.0f}s", flush=True)
                running = 0.0
            if step % eval_every == 0 or step == total_steps:
                vl = collect_logits(model, val_ids, device, pad_id)
                il = collect_logits(model, isear_ids, device, pad_id)
                f1 = macro_f1(y_val, vl, list(range(len(classes))))
                f1_isear = macro_f1(y_isear, il, isear_labels)
                history.append({"step": step, "epoch": round(step / steps_per_epoch, 2), "val_macro_f1": f1,
                                "isear_val_macro_f1": f1_isear, "seconds": round(time.perf_counter() - t0)})
                mark = ""
                if f1 > best["f1"]:
                    best = {"f1": f1, "isear": f1_isear, "step": step, "epoch": history[-1]["epoch"],
                            "state": {k: v.detach().to("cpu", copy=True) for k, v in model.state_dict().items()}}
                    mark = "  <- best"
                print(f"eval @ epoch {history[-1]['epoch']}: val macro F1 {f1:.4f} | ISEAR val macro F1 "
                      f"{f1_isear:.4f}{mark}", flush=True)
                model.train()
            if step >= total_steps:
                done = True
                break

    model.load_state_dict(best["state"])
    del best["state"]
    val_logits = collect_logits(model, val_ids, device, pad_id)
    temperature = fit_temperature(val_logits, y_val)
    probs = softmax(val_logits / temperature)
    pred = [classes[i] for i in probs.argmax(1)]
    correct = probs.argmax(1) == y_val
    ece = {"before_temperature": expected_calibration_error(softmax(val_logits).max(1), correct),
           "after_temperature": expected_calibration_error(probs.max(1), correct)}
    print(f"\nbest checkpoint: epoch {best['epoch']} (val macro F1 {best['f1']:.4f}, ISEAR val {best['isear']:.4f}); "
          f"temperature {temperature:.3f}; ECE {ece['before_temperature']:.4f} -> {ece['after_temperature']:.4f}")

    metrics = compute_metrics(val["label"], pred, classes)
    print_report(val["label"], pred, classes, f"{args.exp_id} (validation)")
    if args.no_log:
        print("--no-log: nothing saved")
        return
    save_confusion_matrices(val["label"], pred, classes, f"confusion_matrix_{args.exp_id}_validation",
                            f"{args.exp_id} - validation")

    now = datetime.now(timezone.utc)
    short = args.model_name.split("/")[-1]
    model_version = f"{load_manifest()['label_map_version']}-{short}-{args.exp_id.lower()}-{now:%Y%m%d}"
    args.out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.out)
    tokenizer.save_pretrained(args.out)
    card = {
        "model_version": model_version,
        "experiment": args.exp_id,
        "model_kind": short,
        "features": "transformer",
        "created_at": now.isoformat(timespec="seconds"),
        "task": "Emotion classification of user text (not a medical or psychological diagnosis)",
        "classes": classes,
        "settings": settings,
        "training_data": {"version": "v3", "sources": sources, "rows": len(train),
                          "by_source": train["source"].value_counts().to_dict()},
        "best_checkpoint": best,
        "history": history,
        "temperature": temperature,
        "max_length": args.max_len,
        "validation_metrics": {k: v for k, v in metrics.items() if k != "per_class"},
        "validation_per_class": metrics["per_class"],
        "validation_ece": ece,
        "validation_thresholds": threshold_table(probs.max(1), val["label"], pred),
        "versions": {"python": platform.python_version(), "torch": torch.__version__, "cuda": torch.version.cuda,
                     "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None},
    }
    save_json(card, args.out / "model_card.json")
    save_json({"id": args.exp_id, "model": short, "settings": settings, "eval_split": "validation",
               "metrics": metrics, "isear_validation_macro_f1": best["isear"], "history": history},
              config.METRICS / f"{args.exp_id}_validation.json")
    append_experiment(args.exp_id, short, settings, "validation", metrics, args.purpose, features="transformer-v3")
    print(f"\nSaved to {args.out} ({model_version})")


if __name__ == "__main__":
    main()
