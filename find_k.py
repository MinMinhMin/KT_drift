import os
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

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
# Extract z_t (Mục tiêu: z_b_prime)
# ─────────────────────────────────────────────
@torch.no_grad()
def extract_z(model, loader, device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    z_list, uid_list = [], []

    for batch in tqdm(loader, desc="Extracting behavioral regimes (z_b_prime)"):
        x     = batch["x"].to(device)
        lens  = batch["lengths"]
        uids  = batch["user_ids"]

        # Forward qua KTAutoencoder
        output = model(x)
        # Lấy z_b_prime theo yêu cầu (behavioral representation sau FiLM)
        z = output["z_b_prime"].cpu().float()   # [B, T, z_b_dim]

        for i, (length, uid) in enumerate(zip(lens.tolist(), uids)):
            zi = z[i, :length].numpy()
            # Normalize sang unit sphere để dùng Cosine Distance hiệu quả
            norm = np.linalg.norm(zi, axis=1, keepdims=True) + 1e-8
            zi = zi / norm
            
            z_list.append(zi)
            uid_list.append(np.full(length, uid, dtype=object))

    z_all   = np.concatenate(z_list,   axis=0)
    uid_all = np.concatenate(uid_list, axis=0)
    return z_all, uid_all

# ─────────────────────────────────────────────
# Spherical KMeans (cosine)
# ─────────────────────────────────────────────
def spherical_kmeans(z: np.ndarray, k: int, n_init: int = 5, max_iter: int = 300) -> np.ndarray:
    N, D = z.shape
    rng  = np.random.default_rng(42)
    best_labels   = None
    best_inertia  = float("inf")
    best_centroids = None

    for _ in range(n_init):
        idx       = rng.choice(N, size=k, replace=False)
        centroids = z[idx].copy()
        centroids = centroids / (np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-8)
        labels = np.zeros(N, dtype=np.int32)

        for _ in range(max_iter):
            sim    = z @ centroids.T
            new_labels = sim.argmax(axis=1)
            if np.all(new_labels == labels):
                break
            labels = new_labels
            for c in range(k):
                mask = labels == c
                if mask.sum() > 0:
                    centroids[c] = z[mask].mean(axis=0)
                    norm = np.linalg.norm(centroids[c])
                    if norm > 1e-8:
                        centroids[c] /= norm

        sim_assigned = sim[np.arange(N), labels]
        inertia      = 1.0 - sim_assigned.mean()
        if inertia < best_inertia:
            best_inertia  = inertia
            best_labels   = labels.copy()
            best_centroids = centroids.copy()
    return best_labels, best_centroids, best_inertia


# ─────────────────────────────────────────────
# Silhouette (cosine)
# ─────────────────────────────────────────────
def silhouette_cosine(z: np.ndarray, labels: np.ndarray, sample: int = 10_000) -> float:
    from sklearn.metrics import silhouette_score
    N = z.shape[0]
    if N > sample:
        rng = np.random.default_rng(0)
        idx = rng.choice(N, size=sample, replace=False)
        z_s, l_s = z[idx], labels[idx]
    else:
        z_s, l_s = z, labels
    try:
        return silhouette_score(z_s, l_s, metric="cosine")
    except:
        return 0.0


# ─────────────────────────────────────────────
# GMM diagonal → soft assignment
# ─────────────────────────────────────────────
def fit_gmm(z: np.ndarray, k: int) -> tuple:
    from sklearn.mixture import GaussianMixture
    gmm = GaussianMixture(
        n_components=k,
        covariance_type="diag",
        max_iter=200,
        n_init=3,
        random_state=42,
    )
    gmm.fit(z)
    bic        = gmm.bic(z)
    soft_probs = gmm.predict_proba(z)
    return gmm, bic, soft_probs


def print_cluster_stats(labels: np.ndarray, k: int, title: str):
    print(f"\n  {title}")
    counts = np.bincount(labels, minlength=k)
    total  = len(labels)
    for c in range(k):
        bar = "█" * int(30 * counts[c] / max(counts + [1]))
        print(f"    Cluster {c:2d}: {counts[c]:6d} ({100*counts[c]/total:.1f}%)  {bar}")


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint",   type=str, default="checkpoints/autoencoder/best.pt")
    parser.add_argument("--output",       type=str, default="clustering")
    parser.add_argument("--index",        type=int, default=0)
    
    # Model dims (khớp với autoencoder.py)
    parser.add_argument("--state_dim",    type=int, default=33)
    parser.add_argument("--behavior_dim", type=int, default=6)
    parser.add_argument("--z_s_dim",      type=int, default=16)
    parser.add_argument("--z_b_dim",      type=int, default=8)
    parser.add_argument("--lstm_hidden",  type=int, default=64)
    
    # Dataset & Ablation
    parser.add_argument("--ablation",     type=str, default="full", choices=["full", "no_text", "no_tree", "no_tree_no_text"])
    parser.add_argument("--split",        type=str, default="val", choices=["val", "full"])
    parser.add_argument("--val_ratio",    type=float, default=0.1)
    parser.add_argument("--seed",         type=int, default=42)
    
    # Clustering params
    parser.add_argument("--k_min",        type=int, default=3)
    parser.add_argument("--k_max",        type=int, default=15)
    parser.add_argument("--method",       type=str, default="both", choices=["spherical_kmeans", "gmm", "both"])
    parser.add_argument("--n_init",       type=int, default=5)
    
    # DataLoader
    parser.add_argument("--batch_size",   type=int, default=32)
    parser.add_argument("--num_workers",  type=int, default=8)
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    device = f"cuda:{args.index}" if torch.cuda.is_available() else "cpu"

    # ── Load model ──────────────────────────────
    from autoencoder import KTAutoencoder
    model = KTAutoencoder(
        state_dim=args.state_dim,
        behavior_dim=args.behavior_dim,
        z_s_dim=args.z_s_dim,
        z_b_dim=args.z_b_dim,
        lstm_hidden=args.lstm_hidden
    )
    
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt["model_state"])
    model = model.to(device)
    print(f"[ckpt] Loaded: epoch={ckpt.get('epoch', '?')}  val_loss={ckpt.get('best_val', 0):.4f}")

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
    
    # Truyền arg ablation vào build_datasets theo yêu cầu
    full_ds = build_datasets(csv_path, ex_emb, ablation_mode=args.ablation, cache=False)

    if args.split == "val":
        n_val   = max(1, int(len(full_ds) * args.val_ratio))
        n_train = len(full_ds) - n_val
        _, target_ds = random_split(
            full_ds, [n_train, n_val],
            generator=torch.Generator().manual_seed(args.seed),
        )
        print(f"[split] val: {n_val} users")
    else:
        target_ds = full_ds
        print(f"[split] full: {len(full_ds)} users")

    loader = DataLoader(
        target_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=args.num_workers, pin_memory=True,
    )

    # ── Extract z_t ─────────────────────────────
    z_all, uid_all = extract_z(model, loader, device)

    # ── Sweep k ─────────────────────────────────
    k_range = list(range(args.k_min, args.k_max + 1))
    sk_silhouettes, sk_inertias, gmm_bics = {}, {}, {}
    best_sk_labels, best_sk_centroids, best_gmm_probs = {}, {}, {}

    print(f"\n{'═'*60}\n  Sweeping k = {args.k_min} → {args.k_max} | Target: z_b_prime\n{'═'*60}")

    for k in k_range:
        print(f"\n── k={k} ──────────────────────────────────────")
        if args.method in ("spherical_kmeans", "both"):
            labels, centroids, inertia = spherical_kmeans(z_all, k, n_init=args.n_init)
            sil = silhouette_cosine(z_all, labels)
            sk_silhouettes[k], sk_inertias[k] = sil, inertia
            best_sk_labels[k], best_sk_centroids[k] = labels, centroids
            print(f"  [SKMeans] inertia={inertia:.4f}  sil={sil:.4f}")

        if args.method in ("gmm", "both"):
            gmm, bic, soft_probs = fit_gmm(z_all, k)
            hard_labels = soft_probs.argmax(axis=1)
            sil_gmm = silhouette_cosine(z_all, hard_labels)
            gmm_bics[k], best_gmm_probs[k] = bic, soft_probs
            print(f"  [GMM diag] BIC={bic:.1f}  sil={sil_gmm:.4f}")

    # ── Save & Report ───────────────────────────
    print(f"\n{'═'*60}\n  SUMMARY & SAVING\n{'═'*60}")

    np.save(os.path.join(args.output, "z_all.npy"),   z_all.astype(np.float32))
    np.save(os.path.join(args.output, "uid_all.npy"), uid_all)

    if args.method in ("spherical_kmeans", "both"):
        best_k_sk = max(sk_silhouettes, key=sk_silhouettes.get)
        np.save(os.path.join(args.output, "sk_labels.npy"), best_sk_labels[best_k_sk])
        np.save(os.path.join(args.output, "sk_centroids.npy"), best_sk_centroids[best_k_sk])
        # Soft assignment via temperature softmax
        sims = z_all @ best_sk_centroids[best_k_sk].T
        soft_sk = torch.softmax(torch.tensor(sims) / 0.1, dim=-1).numpy()
        np.save(os.path.join(args.output, "sk_soft_assignment.npy"), soft_sk)
        print(f"  [SK] Saved best k={best_k_sk}")

    if args.method in ("gmm", "both"):
        best_k_gmm = min(gmm_bics, key=gmm_bics.get)
        np.save(os.path.join(args.output, "gmm_soft_assignment.npy"), best_gmm_probs[best_k_gmm])
        np.save(os.path.join(args.output, "gmm_labels.npy"), best_gmm_probs[best_k_gmm].argmax(axis=1))
        print(f"  [GMM] Saved best k={best_k_gmm}")

    import json
    sweep = {
        "k_range": k_range,
        "sk_silhouettes": {str(k): float(v) for k, v in sk_silhouettes.items()},
        "gmm_bics": {str(k): float(v) for k, v in gmm_bics.items()},
    }
    with open(os.path.join(args.output, "sweep_results.json"), "w") as f:
        json.dump(sweep, f, indent=2)

    print(f"\n[done] Output: {args.output}/")


    # Example run:
    # python find_k.py \
    #     --checkpoint checkpoints/autoencoder-no_text/best.pt \
    #     --output clustering \
    #     --index 0 \
    #     --ablation no_text 