"""
umap_visualize.py

Extract z_t từ checkpoint → UMAP 2D → lưu PNG.
Màu sắc theo 3 góc nhìn trên cùng 1 figure:
  [A] user_id  — kiểm tra xem z_t cluster theo user hay không
                  (nếu có → model chỉ học user identity, không học behavior)
  [B] timestep position (session_pos) — kiểm tra temporal drift
                  (nếu gradient rõ → z_t encode thời gian, tốt)
  [C] response (đúng/sai) — kiểm tra behavior signal
                  (nếu tách biệt → z_t phân biệt được engagement)
"""

import os
import argparse
import pickle
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torch.nn.utils.rnn import pad_sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm

try:
    import umap
except ImportError:
    raise ImportError("Cài umap-learn: pip install umap-learn")


# ─────────────────────────────────────────────
# Collate
# ─────────────────────────────────────────────
def collate_fn(batch):
    xs   = [b["x_sequence"] for b in batch]
    lens = [x.shape[0] for x in xs]
    x_padded     = pad_sequence(xs, batch_first=True, padding_value=0.0)
    T_max        = x_padded.shape[1]
    padding_mask = torch.zeros(len(batch), T_max, dtype=torch.bool)
    for i, l in enumerate(lens):
        padding_mask[i, l:] = True
    return {
        "x":            x_padded,
        "padding_mask": padding_mask,
        "lengths":      torch.tensor(lens),
        "user_ids":     [b["user_id"] for b in batch],
    }


# ─────────────────────────────────────────────
# Extract z_t
# ─────────────────────────────────────────────
@torch.no_grad()
def extract_z(model, loader, device, max_points: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    z_list, user_list, pos_list, resp_list = [], [], [], []
    collected = 0
    
    e_q_dim = model.e_q_dim # Lấy từ model để chuẩn xác

    for batch in loader:
        if collected >= max_points: break

        x = batch["x"].to(device)
        lens = batch["lengths"]
        uids = batch["user_ids"]

        # Sử dụng forward mới
        output = model(x)
        z = output["z"].cpu().float().numpy() 
        x_np = x.cpu().float().numpy()

        for i, (length, uid) in enumerate(zip(lens.tolist(), uids)):
            if collected >= max_points: break
            
            zi = z[i, :length]
            xi = x_np[i, :length]
            
            # Normalize z cho cosine metric
            zi = zi / (np.linalg.norm(zi, axis=1, keepdims=True) + 1e-8)
            
            z_list.append(zi)
            user_list.append(np.full(length, hash(uid) % 10000, dtype=np.int32))

            # Logic trích xuất từ feature vector x:
            # Cột cuối cùng là position (đã được normalize 0-1 trong dataset)
            pos_list.append(xi[:, -1])

            # Response: e_q_dim là wrong, e_q_dim + 1 là correct
            # Lấy xác suất đúng (cột thứ 2 của response one-hot)
            resp_correct = xi[:, e_q_dim + 1] 
            resp_list.append(resp_correct)

            collected += length

    z_all    = np.concatenate(z_list,    axis=0)[:max_points]
    user_all = np.concatenate(user_list, axis=0)[:max_points]
    pos_all  = np.concatenate(pos_list,  axis=0)[:max_points]
    resp_all = np.concatenate(resp_list, axis=0)[:max_points]

    print(f"[extract] {z_all.shape[0]} z_t points  z_dim={z_all.shape[1]}")
    return z_all, user_all, pos_all, resp_all


# ─────────────────────────────────────────────
# UMAP
# ─────────────────────────────────────────────
def run_umap(z: np.ndarray, n_neighbors: int, min_dist: float, seed: int) -> np.ndarray:
    print(f"[umap] n_neighbors={n_neighbors}  min_dist={min_dist}  seed={seed}")
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric="cosine",       # z_t đã L2-norm → cosine phù hợp hơn euclidean
        random_state=seed,
        verbose=True,
    )
    return reducer.fit_transform(z)


# ─────────────────────────────────────────────
# Plot
# ─────────────────────────────────────────────
def plot_umap(
    emb:        np.ndarray,    # [N, 2]
    user_all:   np.ndarray,
    pos_all:    np.ndarray,
    resp_all:   np.ndarray,
    output_path: str,
    n_points:   int,
):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Subsample để plot không quá nặng
    idx = np.random.default_rng(0).choice(len(emb), size=min(n_points, len(emb)), replace=False)
    e   = emb[idx]
    u   = user_all[idx]
    p   = pos_all[idx]
    r   = resp_all[idx]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("UMAP 2D of z_t", fontsize=14, fontweight="bold")

    dot = 2.5
    alpha = 0.4

    # [A] Color by user (up to 20 distinct users)
    ax = axes[0]
    unique_users = np.unique(u)[:20]
    colors_u = cm.tab20(np.linspace(0, 1, len(unique_users)))
    for ci, uid in enumerate(unique_users):
        mask = u == uid
        ax.scatter(e[mask, 0], e[mask, 1], s=dot, alpha=alpha,
                   color=colors_u[ci], rasterized=True)
    ax.set_title("[A] Color by user_id\n(blob đặc = model học user identity)")
    ax.set_xlabel("UMAP-1"); ax.set_ylabel("UMAP-2")
    ax.set_xticks([]); ax.set_yticks([])

    # [B] Color by session position (gradient xanh→đỏ)
    ax = axes[1]
    sc = ax.scatter(e[:, 0], e[:, 1], c=p, cmap="coolwarm",
                    s=dot, alpha=alpha, rasterized=True, vmin=0, vmax=1)
    plt.colorbar(sc, ax=ax, label="session pos (0=đầu, 1=cuối)")
    ax.set_title("[B] Color by session position\n(gradient rõ = z_t encode temporal drift)")
    ax.set_xlabel("UMAP-1"); ax.set_ylabel("UMAP-2")
    ax.set_xticks([]); ax.set_yticks([])

    # [C] Color by response (đỏ=sai, xanh=đúng)
    ax = axes[2]
    wrong = r < 0.5
    ax.scatter(e[wrong,  0], e[wrong,  1], s=dot, alpha=alpha,
               color="#e74c3c", label="wrong (0)", rasterized=True)
    ax.scatter(e[~wrong, 0], e[~wrong, 1], s=dot, alpha=alpha,
               color="#2ecc71", label="correct (1)", rasterized=True)
    ax.legend(markerscale=4, loc="upper right", fontsize=8)
    ax.set_title("[C] Color by response\n(tách biệt = z_t phân biệt engagement)")
    ax.set_xlabel("UMAP-1"); ax.set_ylabel("UMAP-2")
    ax.set_xticks([]); ax.set_yticks([])

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[saved] {output_path}")


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint",   type=str,
                        default="checkpoints/autoencoder-full/best.pt")
    parser.add_argument("--output",       type=str,
                        default="clustering/visualize/umap_2d_zt.png")
    parser.add_argument("--index",        type=int,   default=0)
    # Model dims — phải khớp với checkpoint
    parser.add_argument("--id_dim",       type=int,   default=32)
    parser.add_argument("--tree_dim",     type=int,   default=32)
    parser.add_argument("--text_dim",     type=int,   default=64)
    parser.add_argument("--scalar_dim",   type=int,   default=7)
    parser.add_argument("--d_model",      type=int,   default=128)
    parser.add_argument("--z_dim",        type=int,   default=64)
    parser.add_argument("--n_enc_layers", type=int,   default=4)
    parser.add_argument("--n_dec_layers", type=int,   default=2)
    parser.add_argument("--n_heads",      type=int,   default=8)
    parser.add_argument("--ffn_dim",      type=int,   default=256)
    parser.add_argument("--ablation",     type=str,   default="full")
    # Split
    parser.add_argument("--split",        type=str,   default="val",
                        choices=["val", "full"],
                        help="val = 10%% val set (seed=42) | full = toàn bộ dataset")
    parser.add_argument("--val_ratio",    type=float, default=0.1)
    parser.add_argument("--seed",         type=int,   default=42)
    # UMAP
    parser.add_argument("--n_neighbors",  type=int,   default=30)
    parser.add_argument("--min_dist",     type=float, default=0.1)
    # Plot
    parser.add_argument("--plot_points",  type=int,   default=20_000,
                        help="Số điểm subsample khi vẽ (không ảnh hưởng UMAP fit)")
    parser.add_argument("--batch_size",   type=int,   default=32)
    parser.add_argument("--num_workers",  type=int,   default=16)
    args = parser.parse_args()

    device = f"cuda:{args.index}" if torch.cuda.is_available() else "cpu"

    # ── Load model ──────────────────────────────
    from autoencoder import KTAutoencoder
    from torch.utils.data import random_split
    x_dim = args.id_dim + args.tree_dim + args.text_dim + args.scalar_dim
    model = KTAutoencoder(
        x_dim=x_dim,
        hidden_dim=args.d_model,
        latent_dim=args.z_dim
    )
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt["model_state"])
    model = model.to(device)
    print(f"[ckpt] epoch={ckpt['epoch']}  loss={ckpt.get('best_val', 0):.4f}")

    # ── Dataset + split ─────────────────────────
    from get_feature import XES3G5M_exercises_embedding
    from dataset import build_datasets

    csv_path = "dataset/processed/XES3G5M/processed.csv"
    ex_emb   = XES3G5M_exercises_embedding(
        csv_path=csv_path,
        kc_dict_path="dataset/processed/XES3G5M/kc_dict.pkl",
        question_dict_path="dataset/processed/XES3G5M/question_dict.pkl",
        base_emb_path="dataset/processed/XES3G5M/excercices_embedding",
    )
    full_ds = build_datasets(csv_path, ex_emb, args.ablation, cache=False)

    if args.split == "val":
        n_val   = max(1, int(len(full_ds) * args.val_ratio))
        n_train = len(full_ds) - n_val
        _, target_ds = random_split(
            full_ds,
            [n_train, n_val],
            generator=torch.Generator().manual_seed(args.seed),
        )
        print(f"[split] val set: {n_val} users  (val_ratio={args.val_ratio}, seed={args.seed})")
    else:
        target_ds = full_ds
        print(f"[split] full dataset: {len(full_ds)} users")

    loader = DataLoader(
        target_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # ── Extract toàn bộ z_t trong split ─────────
    # max_points=inf → lấy hết
    z_all, user_all, pos_all, resp_all = extract_z(
        model, loader, device, max_points=10**9
    )

    # ── UMAP ────────────────────────────────────
    emb_2d = run_umap(z_all, args.n_neighbors, args.min_dist, args.seed)

    # ── Plot ────────────────────────────────────
    plot_umap(emb_2d, user_all, pos_all, resp_all,
              output_path=args.output, n_points=args.plot_points)