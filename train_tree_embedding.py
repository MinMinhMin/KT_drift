

import os
import argparse
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD


# ─────────────────────────────────────────────
# TF-IDF node initialization (giữ nguyên từ v3)
# ─────────────────────────────────────────────
def build_tfidf_node_init(node2idx: dict, kc_dict: dict, emb_dim: int) -> torch.Tensor:
    num_nodes = len(node2idx)
    idx2node  = {v: k for k, v in node2idx.items()}

    texts = []
    has_text = []
    for idx in range(num_nodes):
        node_id = idx2node.get(idx, -1)
        name    = kc_dict.get(node_id, "")
        texts.append(name if isinstance(name, str) and name.strip() else "unknown")
        has_text.append(bool(name and name.strip()))

    vectorizer = TfidfVectorizer(max_features=min(4096, num_nodes * 4), sublinear_tf=True)
    tfidf_mat  = vectorizer.fit_transform(texts)

    n_components = min(emb_dim, tfidf_mat.shape[1] - 1, num_nodes - 1)
    svd          = TruncatedSVD(n_components=n_components, random_state=42)
    dense        = svd.fit_transform(tfidf_mat).astype(np.float32)

    if n_components < emb_dim:
        pad   = np.zeros((num_nodes, emb_dim - n_components), dtype=np.float32)
        dense = np.concatenate([dense, pad], axis=1)

    norms = np.linalg.norm(dense, axis=1, keepdims=True)
    dense = dense / np.where(norms < 1e-8, 1.0, norms)

    print(f"[tfidf-init] {sum(has_text)}/{num_nodes} nodes có text name")
    return torch.tensor(dense, dtype=torch.float32)


# ─────────────────────────────────────────────
# Model (giữ nguyên từ v3)
# ─────────────────────────────────────────────
class HierarchicalTreeEncoder(nn.Module):
    def __init__(
        self,
        num_nodes:      int,
        routes_tensor:  torch.Tensor,
        weights_tensor: torch.Tensor,
        route_mask:     torch.Tensor,
        emb_dim:        int   = 64,
        n_levels:       int   = 4,
        dropout:        float = 0.1,
        init_weight:    torch.Tensor = None,
    ):
        super().__init__()
        self.emb_dim  = emb_dim
        self.n_levels = n_levels

        self.node_embeddings = nn.Embedding(num_nodes, emb_dim, padding_idx=0)
        if init_weight is not None:
            with torch.no_grad():
                self.node_embeddings.weight.copy_(init_weight)

        self.register_buffer("routes",     routes_tensor)
        self.register_buffer("weights",    weights_tensor)
        self.register_buffer("route_mask", route_mask)

        self.level_weights = nn.Parameter(torch.ones(n_levels))

        self.recon_head = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(emb_dim, num_nodes),
        )

    def forward(self, q_indices, return_levels=False):
        batch_routes  = self.routes[q_indices]
        batch_weights = self.weights[q_indices]
        batch_mask    = self.route_mask[q_indices]
        max_depth     = batch_routes.shape[-1]

        level_embs = []
        for level in range(self.n_levels):
            if level < max_depth:
                nodes_at_level = batch_routes[:, :, level]
                valid          = (nodes_at_level != 0).float()
                lev_emb        = self.node_embeddings(nodes_at_level)
                lev_emb        = lev_emb * valid.unsqueeze(-1)
                denom          = valid.sum(dim=1, keepdim=True).clamp(min=1)
                lev_emb        = lev_emb.sum(dim=1) / denom
            else:
                lev_emb = torch.zeros(
                    q_indices.shape[0], self.emb_dim, device=q_indices.device
                )
            level_embs.append(lev_emb)

        level_embs      = torch.stack(level_embs, dim=1)          # [N, L, D]
        level_embs_norm = F.normalize(level_embs, p=2, dim=-1)

        w        = F.softmax(self.level_weights, dim=0)
        tree_emb = (level_embs_norm * w.view(1, -1, 1)).sum(dim=1)
        tree_emb = F.normalize(tree_emb, p=2, dim=-1)

        if return_levels:
            return tree_emb, level_embs_norm
        return tree_emb

    def reconstruct(self, tree_emb):
        return self.recon_head(tree_emb)

    def get_normalized_embeddings(self, q_indices=None, device="cpu"):
        self.eval()
        with torch.no_grad():
            if q_indices is None:
                q_indices = torch.arange(self.routes.shape[0], device=device)
            embs = self.forward(q_indices)
        return F.normalize(embs.cpu(), p=2, dim=-1)

    def get_normalized_node_embeddings(self):
        return F.normalize(self.node_embeddings.weight.detach().cpu(), p=2, dim=-1)


# ─────────────────────────────────────────────
# In-batch All-pairs Loss
# ─────────────────────────────────────────────
class InBatchSoftLoss(nn.Module):
    """
    Nhận [B, D] embeddings + Jaccard matrix [B, B] precomputed.
    Tính cosine sim bằng matmul (1 kernel call) thay vì loop cặp.
    Kết hợp MSE soft loss + margin penalty cho diagonal-off negatives.
 
    Tại sao không dùng SoftPairDataset nữa:
      - v3: B step × 2 forwards = 2B vectors/step
      - v4: 1 forward = B vectors → B² cặp → throughput tăng B lần
      - Với B=2048: 2048 lần nhiều gradient signal hơn mỗi second
    """
    def __init__(self, margin: float = 0.05):
        super().__init__()
        self.margin = margin
 
    def forward(
        self,
        embs:       torch.Tensor,   # [B, D] — L2-normalized
        jac_matrix: torch.Tensor,   # [B, B] — Jaccard scores, precomputed
    ) -> torch.Tensor:
        # Cosine sim matrix: vì embs đã normalize → matmul = cosine
        sim_matrix = embs @ embs.T                              # [B, B]
 
        # Bỏ diagonal (self-similarity)
        B    = embs.shape[0]
        mask = ~torch.eye(B, dtype=torch.bool, device=embs.device)
 
        sim_flat = sim_matrix[mask]                             # [B*(B-1)]
        jac_flat = jac_matrix[mask]                             # [B*(B-1)]
 
        # MSE loss giữa cosine sim và Jaccard target
        loss = F.mse_loss(sim_flat, jac_flat)
 
        # Margin penalty: neg pairs (jac=0) có sim > margin → penalize
        if self.margin > 0:
            neg_mask = (jac_flat < 1e-6)
            if neg_mask.any():
                penalty = F.relu(sim_flat[neg_mask] - self.margin).pow(2).mean()
                loss    = loss + penalty
 
        return loss


# ─────────────────────────────────────────────
# Trainer
# ─────────────────────────────────────────────
class XES3G5M_train_tree_embedding:
    def __init__(
        self,
        csv_input_path:     str,
        question_dict_path: str,
        kc_dict_path:       str,
        index:              int   = 0,
        emb_dim:            int   = 64,
        n_levels:           int   = 4,
        dropout:            float = 0.1,
        batch_size:         int   = 2048,   # lớn hơn nhiều so với v3
        epochs:             int   = 150,
        lr:                 float = 3e-3,
        weight_decay:       float = 1e-4,
        recon_weight:       float = 0.3,
        margin:             float = 0.05,
        num_workers:        int   = 4,
        use_tfidf_init:     bool  = True,
        compile_model:      bool  = True,   # torch.compile (PyTorch ≥ 2.0)
    ):
        self.device       = f"cuda:{index}" if torch.cuda.is_available() else "cpu"
        self.emb_dim      = emb_dim
        self.epochs       = epochs
        self.lr           = lr
        self.weight_decay = weight_decay
        self.recon_weight = recon_weight
        self.num_workers  = num_workers
        self.batch_size   = batch_size

        with open(question_dict_path, "rb") as f:
            self.question_dict = pickle.load(f)
        with open(kc_dict_path, "rb") as f:
            self.kc_dict = pickle.load(f)

        self.df = pd.read_csv(csv_input_path)

        (
            self.num_nodes,
            self.tree_tensors,
            self.qid2idx,
            self.node2idx,
        ) = self.preprocess()

        self.num_questions = len(self.qid2idx)

        # ── Precompute Jaccard matrix trên CPU, lưu GPU ──
        # [num_q, num_q] float16 để tiết kiệm VRAM
        # 7618² × 2 bytes ≈ 110MB — hoàn toàn chấp nhận được
        self.path_sets   = self._build_path_sets()
        self.jac_matrix  = self._build_jaccard_matrix_gpu()

        # ── DataLoader: chỉ cần shuffle indices ──────────
        # Không cần pre-built pairs nữa
        q_indices = torch.arange(self.num_questions, dtype=torch.long)
        self.dataloader = DataLoader(
            TensorDataset(q_indices),
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,      
            pin_memory=False,
            drop_last=True,
        )

        print(f"[init] {self.num_questions} questions | "
              f"{self.num_nodes} nodes | "
              f"batch_size={batch_size} → {batch_size**2:,} pairs/step")

        # ── Model ────────────────────────────────────────
        init_weight = None
        if use_tfidf_init:
            init_weight = build_tfidf_node_init(self.node2idx, self.kc_dict, emb_dim)

        self.model = HierarchicalTreeEncoder(
            num_nodes=self.num_nodes,
            routes_tensor=self.tree_tensors["routes"],
            weights_tensor=self.tree_tensors["weights"],
            route_mask=self.tree_tensors["mask"],
            emb_dim=emb_dim,
            n_levels=n_levels,
            dropout=dropout,
            init_weight=init_weight,
        ).to(self.device)

        # torch.compile fuse các kernel nhỏ → ~20-30% nhanh hơn
        if compile_model and hasattr(torch, "compile"):
            print("[init] torch.compile enabled")
            self.model = torch.compile(self.model)

        self.loss_fn = InBatchSoftLoss(margin=margin)

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=epochs, eta_min=1e-5
        )

        # Node reconstruction targets — [num_q, num_nodes] float16
        self.node_targets = self._build_node_targets().to(self.device)

    # ── Preprocessing ──────────────────────────────────
    def preprocess(self):
        all_nodes = {0}
        for q_data in self.question_dict.values():
            for route in q_data.get("kc_routes", []):
                all_nodes.update([int(x) for x in route.split("----")])
        node2idx = {node_id: idx for idx, node_id in enumerate(sorted(all_nodes))}

        unique_questions = sorted(self.df["questions"].unique())
        qid2idx = {qid: idx for idx, qid in enumerate(unique_questions)}

        max_routes, max_depth = 0, 0
        temp_map = []
        for qid in unique_questions:
            routes = []
            if qid in self.question_dict:
                for r_str in self.question_dict[qid].get("kc_routes", []):
                    r_nodes = [node2idx[int(x)] for x in r_str.split("----")]
                    routes.append(r_nodes[::-1])
                    max_depth = max(max_depth, len(r_nodes))
            if not routes:
                routes.append([0])
            max_routes = max(max_routes, len(routes))
            temp_map.append(routes)

        num_q          = len(unique_questions)
        routes_t       = torch.zeros((num_q, max_routes, max_depth), dtype=torch.long)
        weights_t      = torch.zeros((num_q, max_routes, max_depth), dtype=torch.float)
        mask_t         = torch.zeros((num_q, max_routes), dtype=torch.float)
        gamma = 0.9
        for q_idx, routes in enumerate(temp_map):
            for r_idx, r_nodes in enumerate(routes):
                d = len(r_nodes)
                routes_t[q_idx, r_idx, :d]  = torch.tensor(r_nodes, dtype=torch.long)
                weights_t[q_idx, r_idx, :d] = torch.pow(
                    gamma, torch.arange(1, d + 1, dtype=torch.float)
                )
                mask_t[q_idx, r_idx] = 1.0

        return len(node2idx), {
            "routes": routes_t, "weights": weights_t, "mask": mask_t
        }, qid2idx, node2idx

    def _build_path_sets(self):
        path_sets = {}
        for qid, idx in self.qid2idx.items():
            nodes = set()
            if qid in self.question_dict:
                for r_str in self.question_dict[qid].get("kc_routes", []):
                    nodes.update([self.node2idx[int(x)] for x in r_str.split("----")])
            path_sets[idx] = nodes
        return path_sets

    def _build_jaccard_matrix_gpu(self) -> torch.Tensor:
        """
        Precompute toàn bộ Jaccard matrix [num_q, num_q] một lần.
        Dùng set operations trên CPU, lưu float16 lên GPU.

        Với 7618 questions: 7618² ≈ 58M phép tính → ~30s trên CPU.
        Tiết kiệm hơn nhiều so với tính lại mỗi batch mỗi epoch.
        """
        print("[init] Building Jaccard matrix (one-time cost)...")
        N   = self.num_questions
        mat = np.zeros((N, N), dtype=np.float16)

        sets = [self.path_sets[i] for i in range(N)]

        for i in tqdm(range(N), desc="Jaccard matrix"):
            si = sets[i]
            if not si:
                continue
            for j in range(i + 1, N):
                sj = sets[j]
                if not sj:
                    continue
                inter = len(si & sj)
                if inter == 0:
                    continue
                union     = len(si | sj)
                jac       = inter / union
                mat[i, j] = jac
                mat[j, i] = jac

        jac_tensor = torch.tensor(mat, dtype=torch.float16).to(self.device)
        print(f"[init] Jaccard matrix: {jac_tensor.shape}, "
              f"VRAM = {jac_tensor.numel() * 2 / 1024**2:.1f} MB")
        return jac_tensor

    def _build_node_targets(self) -> torch.Tensor:
        targets = torch.zeros(self.num_questions, self.num_nodes, dtype=torch.float16)
        for qid, idx in self.qid2idx.items():
            if qid in self.question_dict:
                nodes = set()
                for r_str in self.question_dict[qid].get("kc_routes", []):
                    nodes.update([self.node2idx[int(x)] for x in r_str.split("----")])
                for n in nodes:
                    targets[idx, n] = 1.0
        return targets

    # ── Training loop ────────────────────────────────
    def train(self, output_path):
        os.makedirs(output_path, exist_ok=True)
        log_path  = os.path.join(output_path, "training_log.txt")
        best_loss = float("inf")

        with open(log_path, "w", encoding="utf-8") as log_file:
            for epoch in range(1, self.epochs + 1):
                self.model.train()
                total_loss  = 0.0
                total_soft  = 0.0
                total_recon = 0.0
                total_var = 0.0
                n_batches   = 0

                pbar = tqdm(self.dataloader, desc=f"Epoch {epoch}")
                for (q_idx,) in pbar:
                    q_idx = q_idx.to(self.device)

                    self.optimizer.zero_grad()

                    # Forward: 1 lần cho toàn batch
                    embs, level_embs = self.model(q_idx,return_levels=True) # [B, D], [B, L, D]

                    # Jaccard sub-matrix cho batch này
                    jac_sub = self.jac_matrix[q_idx][:, q_idx]     # [B, B] float16
                    jac_sub = jac_sub.float()                       # cast cho loss

                    # Loss 1: In-batch all-pairs soft contrastive
                    loss_soft = self.loss_fn(embs, jac_sub)

                    # Loss 2: Reconstruction
                    recon_logits  = self.model.reconstruct(embs)    # [B, num_nodes]
                    recon_targets = self.node_targets[q_idx].float() # [B, num_nodes]
                    loss_recon    = F.binary_cross_entropy_with_logits(
                        recon_logits, recon_targets, reduction="mean"
                    )

                    # Loss3: auxiliary loss ép leaf level phải có variance cao hơn root
                    leaf_var = level_embs[:, 0,  :].var(dim=0).mean()
                    root_var = level_embs[:, -1, :].var(dim=0).mean()
                    loss_var = F.relu(root_var - leaf_var + 0.005)  # leaf phải > root
                    

                    loss = loss_soft + self.recon_weight * loss_recon + 0.1 * loss_var

                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()

                    total_loss  += loss.item()
                    total_soft  += loss_soft.item()
                    total_recon += loss_recon.item()
                    total_var += loss_var.item()
                    n_batches   += 1

                    pbar.set_postfix({
                        "loss":  f"{loss.item():.4f}",
                        "soft":  f"{loss_soft.item():.4f}",
                        "recon": f"{loss_recon.item():.4f}",
                        "loss_var":f"{loss_var.item():.4f}"
                    })

                self.scheduler.step()
                avg_loss = total_loss / max(n_batches, 1)

                log_line = (
                    f"Epoch {epoch:3d} | Loss: {avg_loss:.4f} | "
                    f"Soft: {total_soft/max(n_batches,1):.4f} | "
                    f"Var: {total_var/max(n_batches,1):.4f} | "
                    f"Recon: {total_recon/max(n_batches,1):.4f}\n"
                )
                print(log_line.strip())
                log_file.write(log_line)
                log_file.flush()

                if avg_loss < best_loss:
                    best_loss = avg_loss
                    self._save(output_path, epoch, best_loss)

        print(f"\n[done] Best loss: {best_loss:.4f}")

    # ── Save ─────────────────────────────────────────
    def _save(self, output_path, epoch, loss):
        self.model.eval()
        all_q = torch.arange(self.num_questions, device=self.device)

        with torch.no_grad():
            q_embs = self.model.get_normalized_embeddings(all_q, device=self.device)
            torch.save(q_embs, os.path.join(output_path, "question_tree_embeddings.pt"))

            _, level_embs = self.model(all_q, return_levels=True)
            torch.save(level_embs.cpu(),
                       os.path.join(output_path, "question_level_embeddings.pt"))

            node_embs = self.model.get_normalized_node_embeddings()
            torch.save(node_embs, os.path.join(output_path, "node_embeddings.pt"))

        with open(os.path.join(output_path, "qid2idx.pkl"), "wb") as f:
            pickle.dump(self.qid2idx, f)
        with open(os.path.join(output_path, "node2idx.pkl"), "wb") as f:
            pickle.dump(self.node2idx, f)

        torch.save({
            "epoch":       epoch,
            "model_state": self.model.state_dict(),
            "loss":        loss,
            "emb_dim":     self.emb_dim,
        }, os.path.join(output_path, "checkpoint.pt"))
        print(f"  ★ Saved checkpoint (loss={loss:.4f})")


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Tree Embedding")

    parser.add_argument("--csv",           type=str,
                        default="dataset/processed/XES3G5M/processed.csv")
    parser.add_argument("--question_dict", type=str,
                        default="dataset/processed/XES3G5M/question_dict.pkl")
    parser.add_argument("--kc_dict",       type=str,
                        default="dataset/processed/XES3G5M/kc_dict.pkl")
    parser.add_argument("--output",        type=str,
                        default="dataset/processed/XES3G5M/excercices_embedding/tree_embedding")

    parser.add_argument("--emb_dim",       type=int,   default=32)
    parser.add_argument("--n_levels",      type=int,   default=4)
    parser.add_argument("--dropout",       type=float, default=0.1)
    parser.add_argument("--no_tfidf_init", action="store_true")
    parser.add_argument("--no_compile",    action="store_true",
                        help="Tắt torch.compile (dùng nếu PyTorch < 2.0)")

    parser.add_argument("--index",         type=int,   default=0)
    parser.add_argument("--batch_size",    type=int,   default=2048,
                        help="Số questions/batch. Pairs/step = batch_size²")
    parser.add_argument("--epochs",        type=int,   default=150)
    parser.add_argument("--lr",            type=float, default=3e-3)
    parser.add_argument("--weight_decay",  type=float, default=1e-4)
    parser.add_argument("--recon_weight",  type=float, default=0.3)
    parser.add_argument("--margin",        type=float, default=0.05)
    parser.add_argument("--num_workers",   type=int,   default=4)

    args = parser.parse_args()

    trainer = XES3G5M_train_tree_embedding(
        csv_input_path=args.csv,
        question_dict_path=args.question_dict,
        kc_dict_path=args.kc_dict,
        index=args.index,
        emb_dim=args.emb_dim,
        n_levels=args.n_levels,
        dropout=args.dropout,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        recon_weight=args.recon_weight,
        margin=args.margin,
        num_workers=args.num_workers,
        use_tfidf_init=not args.no_tfidf_init,
        compile_model=not args.no_compile,
    )
    trainer.train(args.output)

    ## Ví dụ chạy:
    # python train_tree_embedding.py \
    #     --csv dataset/processed/XES3G5M/processed.csv \
    #     --question_dict dataset/processed/XES3G5M/question_dict.pkl \
    #     --kc_dict dataset/processed/XES3G5M/kc_dict.pkl \
    #     --output dataset/processed/XES3G5M/excercices_embedding/tree_embedding\
    #     --emb_dim 16 \
    #     --n_levels 4 \