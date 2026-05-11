"""
train_id_embedding.py — ID Embedding via Next-Question Contrastive Learning

Task: Cho sequence (q_1,r_1)...(q_t,r_t), predict q_{t+1}.
Why: Dự đoán đúng/sai quá dễ → embedding chỉ học "khó/dễ", KT chỉ học để predict đúng sai,embedding ít hưởng lợi. 
     Dự đoán "bài tiếp theo là gì" ép embedding học:
       - Prerequisite giữa các bài
       - Co-occurrence trong lộ trình học
       - Cấu trúc chapter/topic ẩn

Loss: Cross-Entropy toàn vocabulary (giống word2vec) + learned temperature.
Architecture: LSTM encoder → project về không gian q_emb → dot product.

Saved embeddings (question_id_embeddings.pt) đã được L2-normalize,
sẵn sàng dùng cho downstream model với cosine similarity / dot product.
"""

import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import pickle
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


# ─────────────────────────────────────────────
# Dataset: Next-Question Prediction
# ─────────────────────────────────────────────
class NextQuestionDataset(Dataset):
    def __init__(self, sequences, max_len=200):
        """
        sequences: list of (q_seq, r_seq)
        Input:  (q_0..q_{T-2}, r_0..r_{T-2})
        Target: q_1..q_{T-1}  (question ID tiếp theo)
        """
        self.samples = []
        for q_seq, r_seq in sequences:
            if len(q_seq) < 2:
                continue
            q_in     = q_seq[:-1][-max_len:]
            r_in     = r_seq[:-1][-max_len:]
            q_target = q_seq[1:][-max_len:]
            self.samples.append((q_in, r_in, q_target))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        q_in, r_in, q_target = self.samples[idx]
        return (
            torch.tensor(q_in,     dtype=torch.long),
            torch.tensor(r_in,     dtype=torch.long),
            torch.tensor(q_target, dtype=torch.long),
        )


def collate_fn(batch):
    q_seqs, r_seqs, targets = zip(*batch)
    q_padded = nn.utils.rnn.pad_sequence(q_seqs,  batch_first=True, padding_value=0)
    r_padded = nn.utils.rnn.pad_sequence(r_seqs,  batch_first=True, padding_value=0)
    t_padded = nn.utils.rnn.pad_sequence(targets, batch_first=True, padding_value=-1)
    return q_padded, r_padded, t_padded


# ─────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────
class IDEmbeddingModel(nn.Module):
    def __init__(
        self,
        num_questions: int,
        q_embed_dim:  int   = 64,
        r_embed_dim:  int   = 8,
        hidden_dim:   int   = 128,
        num_layers:   int   = 2,
        dropout:      float = 0.2,
    ):
        super().__init__()
        self.q_embed_dim = q_embed_dim

        self.question_embedding = nn.Embedding(num_questions, q_embed_dim)
        self.response_embedding = nn.Embedding(2, r_embed_dim)

        self.encoder = nn.LSTM(
            input_size=q_embed_dim + r_embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=False,
        )

        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, q_embed_dim),
        )

        # Learnable temperature — khởi tạo 1.0, để model tự học xuống
        self.temperature = nn.Parameter(torch.tensor(1.0))

    def forward(self, q, r):
        # q, r: [B, T]
        q_emb = self.question_embedding(q)      # [B, T, Dq]
        r_emb = self.response_embedding(r)      # [B, T, Dr]
        x = torch.cat([q_emb, r_emb], dim=-1)   # [B, T, Dq+Dr]

        h, _ = self.encoder(x)                  # [B, T, H]
        z = self.projection(h)                  # [B, T, Dq]
        z = F.normalize(z, p=2, dim=-1)
        return z

    def compute_logits(self, z):
        # z: [B, T, Dq] — đã normalized
        # all_q_emb: [num_q, Dq] — L2 normalized (nhất quán với embedding được save)
        all_q_emb = F.normalize(self.question_embedding.weight, p=2, dim=-1)
        logits = torch.matmul(z, all_q_emb.T) / torch.clamp(self.temperature, min=0.01)
        return logits  # [B, T, num_q]

    def get_normalized_embeddings(self):
        """Trả về embedding matrix đã L2-normalize — dùng khi save hoặc inference."""
        return F.normalize(self.question_embedding.weight.detach().cpu(), p=2, dim=-1)


# ─────────────────────────────────────────────
# Trainer
# ─────────────────────────────────────────────
class XES3G5M_train_id_embedding:
    def __init__(
        self,
        csv_input_path: str,
        index:       int   = 0,
        max_len:     int   = 200,
        embed_dim:   int   = 64,
        r_embed_dim: int   = 8,
        hidden_dim:  int   = 128,
        num_layers:  int   = 2,
        dropout:     float = 0.2,
        batch_size:  int   = 128,
        epochs:      int   = 100,
        lr:          float = 3e-3,
        weight_decay:float = 1e-4,
        num_workers: int   = 4,
    ):
        self.csv_input_path = csv_input_path
        self.df             = pd.read_csv(csv_input_path)
        self.max_len        = max_len
        self.device         = f"cuda:{index}" if torch.cuda.is_available() else "cpu"

        # Lưu các hyperparams để dùng trong train() và _save()
        self.embed_dim    = embed_dim
        self.batch_size   = batch_size
        self.epochs       = epochs
        self.lr           = lr
        self.weight_decay = weight_decay
        self.num_workers  = num_workers

        self.num_questions, self.user_sequences, self.qid2idx = self.preprocess()

        self.dataset = NextQuestionDataset(self.user_sequences, max_len=max_len)
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
        )

        self.model = IDEmbeddingModel(
            num_questions=self.num_questions,
            q_embed_dim=embed_dim,
            r_embed_dim=r_embed_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
        ).to(self.device)

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=epochs, eta_min=1e-5
        )

    # ── Preprocessing ──────────────────────────
    def preprocess(self):
        self.df = self.df.sort_values(["uid", "timestamps"])
        unique_questions = sorted(self.df["questions"].unique())
        qid2idx = {qid: idx for idx, qid in enumerate(unique_questions)}
        self.df["q_idx"] = self.df["questions"].map(qid2idx)

        user_sequences = []
        for uid, group in self.df.groupby("uid"):
            q_seq = group["q_idx"].tolist()
            r_seq = group["responses"].tolist()
            if len(q_seq) >= 2:
                user_sequences.append((q_seq, r_seq))

        return len(unique_questions), user_sequences, qid2idx

    # ── Training loop ───────────────────────────
    def train(self, output_path):
        os.makedirs(output_path, exist_ok=True)
        log_path  = os.path.join(output_path, "training_log.txt")
        best_loss = float("inf")

        with open(log_path, "w", encoding="utf-8") as log_file:
            for epoch in range(1, self.epochs + 1):
                self.model.train()
                total_loss = 0.0
                total_acc  = 0.0
                n_tokens   = 0

                pbar = tqdm(self.dataloader, desc=f"Epoch {epoch}")
                for q, r, target in pbar:
                    q      = q.to(self.device)
                    r      = r.to(self.device)
                    target = target.to(self.device)

                    self.optimizer.zero_grad()

                    z      = self.model(q, r)                  # [B, T, Dq]
                    logits = self.model.compute_logits(z)       # [B, T, num_q]

                    B, T, V     = logits.shape
                    logits_flat = logits.view(-1, V)
                    target_flat = target.view(-1)
                    mask        = target_flat != -1

                    loss = F.cross_entropy(
                        logits_flat[mask], target_flat[mask], reduction="mean"
                    )

                    with torch.no_grad():
                        pred = logits_flat[mask].argmax(dim=-1)
                        acc  = (pred == target_flat[mask]).float().mean()

                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()

                    total_loss += loss.item() * mask.sum().item()
                    total_acc  += acc.item()  * mask.sum().item()
                    n_tokens   += mask.sum().item()

                    pbar.set_postfix({
                        "loss": f"{loss.item():.4f}",
                        "acc":  f"{acc.item():.4f}",
                        "temp": f"{self.model.temperature.item():.3f}",
                    })

                self.scheduler.step()
                avg_loss = total_loss / max(n_tokens, 1)
                avg_acc  = total_acc  / max(n_tokens, 1)

                log_line = (
                    f"Epoch {epoch:3d} | Loss: {avg_loss:.4f} | "
                    f"Acc: {avg_acc:.4f} | Temp: {self.model.temperature.item():.3f}\n"
                )
                print(log_line.strip())
                log_file.write(log_line)
                log_file.flush()

                if avg_loss < best_loss:
                    best_loss = avg_loss
                    self._save(output_path, epoch, best_loss)

        print(f"\n[done] Best loss: {best_loss:.4f}")

    # ── Save ────────────────────────────────────
    def _save(self, output_path, epoch, loss):
        # L2-normalized embedding — nhất quán với compute_logits, sẵn sàng cho downstream
        q_emb_normalized = self.model.get_normalized_embeddings()
        torch.save(q_emb_normalized, os.path.join(output_path, "question_id_embeddings.pt"))

        with open(os.path.join(output_path, "qid2idx.pkl"), "wb") as f:
            pickle.dump(self.qid2idx, f)

        torch.save(
            {
                "epoch":           epoch,
                "model_state":     self.model.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "loss":            loss,
                "embed_dim":       self.embed_dim,
            },
            os.path.join(output_path, "checkpoint.pt"),
        )
        print(f"  ★ Saved checkpoint (loss={loss:.4f})")


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ID Embedding")
    parser.add_argument("--csv", type=str,
                        default="dataset/processed/XES3G5M/processed.csv")
    parser.add_argument("--output", type=str,
                        default="dataset/processed/XES3G5M/excercices_embedding/id_embedding")
    parser.add_argument("--index",        type=int,   default=0,    help="CUDA device index")
    parser.add_argument("--embed_dim",    type=int,   default=32,   help="Question embedding dim")
    parser.add_argument("--r_embed_dim",  type=int,   default=8,    help="Response embedding dim")
    parser.add_argument("--hidden_dim",   type=int,   default=128,  help="LSTM hidden size")
    parser.add_argument("--num_layers",   type=int,   default=2,    help="LSTM num layers")
    parser.add_argument("--dropout",      type=float, default=0.2)
    parser.add_argument("--batch_size",   type=int,   default=128)
    parser.add_argument("--epochs",       type=int,   default=100)
    parser.add_argument("--lr",           type=float, default=3e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--max_len",      type=int,   default=200)
    parser.add_argument("--num_workers",  type=int,   default=4)
    args = parser.parse_args()

    trainer = XES3G5M_train_id_embedding(
        csv_input_path=args.csv,
        index=args.index,
        max_len=args.max_len,
        embed_dim=args.embed_dim,
        r_embed_dim=args.r_embed_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        num_workers=args.num_workers,
    )
    trainer.train(args.output)


    ## Ví dụ chạy:
    # python train_id_embedding.py \
    #     --csv dataset/processed/XES3G5M/processed.csv \
    #     --output dataset/processed/XES3G5M/excercices_embedding\
    #     --embed_dim 16 \
    #     --r_embed_dim 4 \
    #     --hidden_dim 64 \