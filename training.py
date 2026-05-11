"""
training.py - Refined for Dual-AE (KT-Drift)
"""
from __future__ import annotations

import argparse
import json
import os
import time
import sys
from pathlib import Path
from typing import Any
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence

# ── Local imports ────────────────────────────────────────────────────────────
# Đã loại bỏ BaseKTModel
from autoencoder import KTAutoencoder
from get_feature import XES3G5M_exercises_embedding
from dataset import build_datasets


# ============================================================================
# Model Registry
# ============================================================================

MODEL_REGISTRY: dict[str, type[nn.Module]] = {
    "autoencoder": KTAutoencoder,
}


# ============================================================================
# Argument Parser
# ============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="KT-Drift training script",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Dataset / feature paths ──────────────────────────────────────────
    ds = p.add_argument_group("Dataset")
    ds.add_argument("--csv_path",           type=str, required=True)
    ds.add_argument("--kc_dict_path",       type=str, default="dataset/processed/XES3G5M/kc_dict.pkl")
    ds.add_argument("--question_dict_path", type=str, default="dataset/processed/XES3G5M/question_dict.pkl")
    ds.add_argument("--base_emb_path",      type=str, default="dataset/processed/XES3G5M/excercices_embedding")
    ds.add_argument("--ablation_mode",      type=str, default="full",
                    choices=["full", "no_text", "no_tree", "no_tree_no_text"])
    ds.add_argument("--no_cache",           action="store_true")
    ds.add_argument("--val_ratio",          type=float, default=0.1)
    ds.add_argument("--num_workers",        type=int, default=4)

    # ── Model ────────────────────────────────────────────────────────────
    m = p.add_argument_group("Model")
    m.add_argument("--model",       type=str, default="autoencoder", choices=list(MODEL_REGISTRY.keys()))
    m.add_argument("--lstm_hidden", type=int, default=64, help="BiLSTM hidden units")
    m.add_argument("--z_s_dim",     type=int, default=16, help="State latent dim")
    m.add_argument("--z_b_dim",     type=int, default=8,  help="Behavior latent dim")
    m.add_argument("--model_kwargs", type=str, default="{}", help="Extra JSON kwargs")

    # ── Training ─────────────────────────────────────────────────────────
    tr = p.add_argument_group("Training")
    tr.add_argument("--epochs",         type=int,   default=50)
    tr.add_argument("--batch_size",     type=int,   default=64)
    tr.add_argument("--lr",             type=float, default=1e-3)
    tr.add_argument("--weight_decay",   type=float, default=1e-4)
    tr.add_argument("--alpha",          type=float, default=1.0, help="Weight for behavior loss")
    tr.add_argument("--lambda_pred",    type=float, default=1.0, help="Weight for predictive loss")
    tr.add_argument("--beta",           type=float, default=0.1, help="Weight for separation loss")
    tr.add_argument("--max_grad_norm",  type=float, default=1.0)
    tr.add_argument("--patience",       type=int,   default=10)
    tr.add_argument("--seed",           type=int,   default=42)

    # ── I/O ────────────────────────────────────────────────────────────
    io = p.add_argument_group("I/O")
    io.add_argument("--output_dir",  type=str, default="checkpoints")
    io.add_argument("--resume",      type=str, default=None)
    io.add_argument("--log_every",   type=int, default=50)
    io.add_argument("--no_wandb",    action="store_true")
    io.add_argument("--wandb_project", type=str, default="kt-drift")
    io.add_argument("--run_name",    type=str, default=None)

    return p.parse_args()


def collate_fn(batch: list[dict]) -> dict:
    return {
        "x_sequence": pad_sequence([b["x_sequence"] for b in batch], batch_first=True),
        "user_id": [b["user_id"] for b in batch],
    }


class Logger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "a", encoding="utf-8")
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()
    def flush(self):
        pass


class MetricLogger:
    def __init__(self):
        self._sums: dict[str, float] = {}
        self._counts: dict[str, int] = {}
    def update(self, metrics: dict[str, Any], n: int = 1):
        for k, v in metrics.items():
            if isinstance(v, (float, int, torch.Tensor)):
                self._sums[k] = self._sums.get(k, 0.0) + float(v) * n
                self._counts[k] = self._counts.get(k, 0) + n
    def averages(self) -> dict[str, float]:
        return {k: self._sums[k] / self._counts[k] for k in self._sums}


def format_metrics(metrics: dict[str, float], prefix: str = "") -> str:
    return "  ".join([f"{prefix}{k}={v:.4f}" for k, v in metrics.items()])


# ============================================================================
# Build model
# ============================================================================

def build_model(args: argparse.Namespace, x_dim: int) -> nn.Module:
    model_cls = MODEL_REGISTRY[args.model]
    extra_kwargs = json.loads(args.model_kwargs)

    # Dựa theo thiết kế MD: State=33, Behavior=6. Tổng x_dim phải là 39.
    # Nếu x_dim khác (do ablation), ta cần tính toán lại split.
    behavior_dim = 6
    state_dim = x_dim - behavior_dim

    model = model_cls(
        state_dim    = state_dim,
        behavior_dim = behavior_dim,
        z_s_dim      = args.z_s_dim,
        z_b_dim      = args.z_b_dim,
        lstm_hidden  = args.lstm_hidden,
        **extra_kwargs,
    )
    return model


# ============================================================================
# Training / Validation step
# ============================================================================

def run_epoch(
    model:        nn.Module,
    loader:       DataLoader,
    optimizer:    torch.optim.Optimizer | None,
    device:       torch.device,
    args:         argparse.Namespace,
    epoch:        int,
    is_train:     bool,
    wandb_run=None,
) -> dict[str, float]:
    model.train(is_train)
    logger = MetricLogger()
    phase  = "train" if is_train else "val"

    for batch_idx, batch in enumerate(loader):
        x = batch["x_sequence"].to(device)

        with torch.set_grad_enabled(is_train):
            output    = model(x)
            # Truyền các hyperparams alpha, lambda, beta vào hàm loss
            loss_dict = model.compute_loss(
                x, output, 
                alpha=args.alpha, 
                lambda_pred=args.lambda_pred, 
                beta=args.beta
            )

        if is_train:
            optimizer.zero_grad()
            loss_dict["loss"].backward() # Dùng key "loss" thay vì "total"
            nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()

        logger.update(loss_dict, n=x.size(0))

        if is_train and (batch_idx + 1) % args.log_every == 0:
            avg = logger.averages()
            print(f"  Ep {epoch:03d} | Batch {batch_idx+1:04d} | {format_metrics(avg)}")
            if wandb_run:
                wandb_run.log({f"{phase}/{k}": v for k, v in avg.items()})

    return logger.averages()


# ============================================================================
# Checkpoint helpers
# ============================================================================

def save_checkpoint(path, model, optimizer, scheduler, epoch, best_val, args):
    torch.save({
        "epoch": epoch,
        "best_val": best_val,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "args": vars(args),
    }, path)


def load_checkpoint(path, model, optimizer, scheduler, device):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optimizer_state"])
    scheduler.load_state_dict(ckpt["scheduler_state"])
    return ckpt["epoch"], ckpt["best_val"]


# ============================================================================
# Main
# ============================================================================

def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # W&B logic
    wandb_run = None
    if not args.no_wandb:
        try:
            import wandb
            run_name = args.run_name or f"{args.model}-{args.ablation_mode}-{time.strftime('%m%d-%H%M')}"
            wandb_run = wandb.init(project=args.wandb_project, name=run_name, config=vars(args))
        except ImportError:
            pass

    run_name = args.run_name or f"{args.model}-{args.ablation_mode}"
    out_dir  = Path(args.output_dir) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    sys.stdout = Logger(out_dir / "train.log")

    # Load Data
    ex_emb = XES3G5M_exercises_embedding(args.csv_path, args.kc_dict_path, args.question_dict_path, args.base_emb_path)
    full_ds = build_datasets(args.csv_path, ex_emb, args.ablation_mode, cache=not args.no_cache)
    
    n_val = max(1, int(len(full_ds) * args.val_ratio))
    train_ds, val_ds = random_split(full_ds, [len(full_ds) - n_val, n_val], generator=torch.Generator().manual_seed(args.seed))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, collate_fn=collate_fn, num_workers=args.num_workers)

    # Model, Opt, Scheduler
    model = build_model(args, full_ds.x_dim).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)

    start_epoch, best_val, no_improve = 1, float("inf"), 0

    if args.resume:
        start_epoch, best_val = load_checkpoint(args.resume, model, optimizer, scheduler, device)
        start_epoch += 1

    # Training Loop
    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()
        train_m = run_epoch(model, train_loader, optimizer, device, args, epoch, is_train=True, wandb_run=wandb_run)
        val_m   = run_epoch(model, val_loader, None, device, args, epoch, is_train=False, wandb_run=wandb_run)
        scheduler.step()

        val_loss = val_m["loss"]
        print(f"Epoch {epoch:03d} | {time.time()-t0:.1f}s | {format_metrics(train_m, 'tr_')} | {format_metrics(val_m, 'val_')}")

        save_checkpoint(out_dir / "last.pt", model, optimizer, scheduler, epoch, best_val, args)
        if val_loss < best_val:
            best_val, no_improve = val_loss, 0
            save_checkpoint(out_dir / "best.pt", model, optimizer, scheduler, epoch, best_val, args)
            print(f"  ✓ Saved best.pt (loss={best_val:.4f})")
        else:
            no_improve += 1
            if no_improve >= args.patience:
                print(f"Early stopping at epoch {epoch}")
                break

    if wandb_run: wandb_run.finish()

if __name__ == "__main__":
    main()




## run
# python training.py \
#     --csv_path dataset/processed/XES3G5M/processed.csv \
#     --ablation_mode no_text \
#     --model autoencoder \
