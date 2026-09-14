"""E4 (optional, guide §22): fine-tune DistilBERT on the same frozen splits, on the GPU.

Selection is by validation macro F1 (best epoch). Probabilities are temperature-scaled on validation so the
API's confidence is meaningful. The test set is NOT touched here; use `make evaluate-transformer` once.

Usage: uv run --extra transformer python -m src.train_transformer [--epochs 4 --batch-size 32 --lr 3e-5 --max-len 128]
"""
import argparse
import json
import platform
import random
import time
from datetime import datetime, timezone

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from src import config
from src.data import load_classes, load_manifest, load_xy
from src.metrics import (
    append_experiment,
    compute_metrics,
    expected_calibration_error,
    print_report,
    save_confusion_matrices,
    save_json,
    threshold_table,
)


class TextDataset(Dataset):
    def __init__(self, encodings: dict, labels: np.ndarray):
        self.encodings = encodings
        self.labels = torch.as_tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, i: int) -> dict:
        item = {k: v[i] for k, v in self.encodings.items()}
        item["labels"] = self.labels[i]
        return item


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def class_weights(y: np.ndarray, n_classes: int) -> torch.Tensor:
    """Square-root inverse frequency: softer than 'balanced', so neutral is not over-penalised."""
    counts = np.bincount(y, minlength=n_classes).astype(float)
    w = np.sqrt(counts.max() / counts)
    return torch.tensor(w / w.mean(), dtype=torch.float)


def encode(tokenizer, texts, max_len: int) -> dict:
    return tokenizer(list(texts), padding="max_length", truncation=True, max_length=max_len, return_tensors="pt")


@torch.inference_mode()
def collect_logits(model, loader: DataLoader, device) -> np.ndarray:
    model.eval()
    out = []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items() if k != "labels"}
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            out.append(model(**batch).logits.float().cpu().numpy())
    return np.concatenate(out)


def fit_temperature(logits: np.ndarray, labels: np.ndarray) -> float:
    """Temperature scaling (Guo et al. 2017): one scalar fitted on validation to minimise NLL."""
    z = torch.tensor(logits)
    y = torch.tensor(labels)
    log_t = torch.zeros(1, requires_grad=True)
    optimizer = torch.optim.LBFGS([log_t], lr=0.1, max_iter=100)

    def closure():
        optimizer.zero_grad()
        loss = torch.nn.functional.cross_entropy(z / log_t.exp(), y)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(log_t.exp())


def softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    p = np.exp(z)
    return p / p.sum(axis=1, keepdims=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-name", default=config.TRANSFORMER_NAME)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--max-len", type=int, default=128)
    parser.add_argument("--weighted-loss", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    seed_everything(config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else " - this will be slow"))

    classes = load_classes()
    label2id = {c: i for i, c in enumerate(classes)}
    X_train, y_train = load_xy("train")
    X_val, y_val = load_xy("validation")
    y_train_ids = y_train.map(label2id).to_numpy()
    y_val_ids = y_val.map(label2id).to_numpy()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name, num_labels=len(classes), id2label=dict(enumerate(classes)), label2id=label2id
    ).to(device)

    train_loader = DataLoader(TextDataset(encode(tokenizer, X_train, args.max_len), y_train_ids),
                              batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(TextDataset(encode(tokenizer, X_val, args.max_len), y_val_ids), batch_size=128)

    weights = class_weights(y_train_ids, len(classes)).to(device) if args.weighted_loss else None
    loss_fn = torch.nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, int(0.1 * total_steps), total_steps)
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")

    settings = (f"{args.model_name}; epochs={args.epochs} batch={args.batch_size} lr={args.lr} max_len={args.max_len} "
                f"fp16 warmup=10% weight_decay=0.01 loss={'sqrt-inverse-freq weighted CE' if args.weighted_loss else 'CE'}")
    print(settings)

    history, best_f1, best_state, best_epoch = [], -1.0, None, -1
    for epoch in range(1, args.epochs + 1):
        model.train()
        start, running = time.perf_counter(), 0.0
        for step, batch in enumerate(train_loader, 1):
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch.pop("labels")
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                logits = model(**batch).logits
            loss = loss_fn(logits.float(), labels)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            running += loss.item()
            if step % 200 == 0:
                print(f"  epoch {epoch} step {step}/{len(train_loader)} loss {running / step:.4f}", flush=True)

        val_logits = collect_logits(model, val_loader, device)
        val_pred = val_logits.argmax(1)
        f1 = compute_metrics(y_val_ids, val_pred, list(range(len(classes))))["macro_f1"]
        history.append({"epoch": epoch, "train_loss": running / len(train_loader), "val_macro_f1": f1,
                        "seconds": time.perf_counter() - start})
        print(f"epoch {epoch}: train loss {running / len(train_loader):.4f} | val macro F1 {f1:.4f} "
              f"| {time.perf_counter() - start:.0f}s", flush=True)
        if f1 > best_f1:
            best_f1, best_epoch = f1, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    val_logits = collect_logits(model, val_loader, device)
    temperature = fit_temperature(val_logits, y_val_ids)
    probs = softmax(val_logits / temperature)
    pred_ids = probs.argmax(1)
    pred = [classes[i] for i in pred_ids]
    correct = pred_ids == y_val_ids
    ece_before = expected_calibration_error(softmax(val_logits).max(1), correct)
    ece_after = expected_calibration_error(probs.max(1), correct)
    print(f"\nbest epoch {best_epoch} (val macro F1 {best_f1:.4f}); temperature {temperature:.3f}; "
          f"ECE {ece_before:.4f} -> {ece_after:.4f}")

    metrics = compute_metrics(y_val, pred, classes)
    print_report(y_val, pred, classes, "E4 distilbert (validation)")
    save_confusion_matrices(y_val, pred, classes, "confusion_matrix_E4_validation", "E4 DistilBERT - validation")

    now = datetime.now(timezone.utc)
    model_version = f"{load_manifest()['label_map_version']}-distilbert-{now:%Y%m%d}"
    config.TRANSFORMER_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(config.TRANSFORMER_DIR)
    tokenizer.save_pretrained(config.TRANSFORMER_DIR)
    card = {
        "model_version": model_version,
        "model_kind": "distilbert",
        "features": "transformer",
        "created_at": now.isoformat(timespec="seconds"),
        "task": "Emotion classification of user text (not a medical or psychological diagnosis)",
        "classes": classes,
        "settings": settings,
        "best_epoch": best_epoch,
        "history": history,
        "temperature": temperature,
        "max_length": args.max_len,
        "validation_metrics": {k: v for k, v in metrics.items() if k != "per_class"},
        "validation_ece": {"before_temperature": ece_before, "after_temperature": ece_after},
        "validation_thresholds": threshold_table(probs.max(1), y_val, pred),
        "versions": {"python": platform.python_version(), "torch": torch.__version__,
                     "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None},
    }
    save_json(card, config.TRANSFORMER_DIR / "model_card.json")
    save_json({"id": "E4", "model": "distilbert", "features": "transformer", "settings": settings,
               "eval_split": "validation", "metrics": metrics, "history": history},
              config.METRICS / "E4_transformer_validation.json")
    append_experiment("E4", "distilbert", settings, "validation", metrics, "Transformer comparison (optional phase)",
                      features="transformer")
    print(f"\nSaved to {config.TRANSFORMER_DIR} ({model_version})")


if __name__ == "__main__":
    main()
