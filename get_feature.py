"""
get_feature.py

Quản lý toàn bộ feature precomputation và lookup cho XES3G5M.
Tất cả feature — kể cả scalar (gap_time, time_of_day, session_pos) —
đều được precompute ở đây để dataset.py chỉ cần gọi get_*.

Feature inventory:
  - id_embedding    : [id_dim]     learnable ID embedding
  - tree_embedding  : [tree_dim]   KC tree structure embedding
  - text_embedding  : [text_dim]   question content (SentenceTransformer + PCA + L2-norm)
  - difficulty      : [1]          1 - mean(responses) per question, clipped [0,1]
  - response        : [2]          one-hot {0,1}  ← tính tại dataset vì phụ thuộc interaction
  - gap_time        : [1]          log1p(clip(gap, 0, p99)), z-normalized
  - time_of_day     : [2]          sin/cos encoding of hour
  - session_pos     : [1]          position within user sequence, normalized [0,1]

gap_time, time_of_day, session_pos phụ thuộc vào toàn bộ user sequence nên
được precompute theo df row index, trả về tensor [N_total, dim].
"""

import os
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize


class XES3G5M_exercises_embedding:
    def __init__(
        self,
        csv_path:            str,
        kc_dict_path:        str,
        question_dict_path:  str,
        device_index:        int = 0,
        base_emb_path:       str = None,
    ):
        self.csv_path           = csv_path
        self.kc_dict_path       = kc_dict_path
        self.question_dict_path = question_dict_path
        self.index              = device_index
        self.base_emb_path      = base_emb_path or \
            "dataset/processed/XES3G5M/excercices_embedding"

        # ── Per-question embedding containers ────────────────
        self.id_data   = {"matrix": None, "qid2idx": None}
        self.tree_data = {"matrix": None, "qid2idx": None}
        self.text_dict: dict | None = None        # {qid -> Tensor[text_dim]}
        self.diff_dict: dict | None = None        # {qid -> float}

        # ── Per-row scalar features (precomputed from full df) ─
        # Populated by precompute_scalar_features() or _preload_scalars()
        self.gap_normalized: torch.Tensor | None  = None  # [N_total, 1]
        self.time_features:  torch.Tensor | None  = None  # [N_total, 2]
        self.session_pos:    torch.Tensor | None  = None  # [N_total, 1]
        self._gap_stats: dict | None = None

        self._preload_all()

    # ════════════════════════════════════════════
    # Preload
    # ════════════════════════════════════════════
    def _preload_all(self):
        print("[offline_embedding] Preloading...")
        self._preload_id()
        self._preload_tree()
        self._preload_text()
        self._preload_difficulty()
        self._preload_scalars()
        print("[offline_embedding] Done.")

    def _preload_id(self):
        path = os.path.join(self.base_emb_path, "id_embedding")
        if not os.path.exists(path):
            return
        self.id_data["matrix"] = torch.load(
            os.path.join(path, "question_id_embeddings.pt"), weights_only=True
        )
        with open(os.path.join(path, "qid2idx.pkl"), "rb") as f:
            self.id_data["qid2idx"] = pickle.load(f)

    def _preload_tree(self):
        path = os.path.join(self.base_emb_path, "tree_embedding")
        if not os.path.exists(path):
            return
        self.tree_data["matrix"] = torch.load(
            os.path.join(path, "question_tree_embeddings.pt"), weights_only=True
        )
        with open(os.path.join(path, "qid2idx.pkl"), "rb") as f:
            self.tree_data["qid2idx"] = pickle.load(f)

    def _preload_text(self):
        f = os.path.join(self.base_emb_path, "text_embedding/text_embedding.pkl")
        if not os.path.exists(f):
            return
        with open(f, "rb") as fp:
            raw = pickle.load(fp)
        # Normalize on load — đảm bảo đúng ngay cả với pkl cũ chưa normalize
        self.text_dict = {
            qid: F.normalize(emb.float().unsqueeze(0), p=2, dim=1).squeeze(0)
            for qid, emb in raw.items()
        }

    def _preload_difficulty(self):
        f = os.path.join(self.base_emb_path, "difficulty_embedding/difficulty_embedding.pkl")
        if not os.path.exists(f):
            return
        with open(f, "rb") as fp:
            self.diff_dict = pickle.load(fp)

    def _preload_scalars(self):
        """Load precomputed scalar features nếu đã tồn tại."""
        f = os.path.join(self.base_emb_path, "scalar_features/scalar_features.pt")
        if not os.path.exists(f):
            return
        data = torch.load(f, weights_only=True)
        self.gap_normalized = data["gap_normalized"]   # [N, 1]
        self.time_features  = data["time_features"]    # [N, 2]
        self.session_pos    = data["session_pos"]      # [N, 1]
        self._gap_stats     = data["gap_stats"]

    # ════════════════════════════════════════════
    # Precompute
    # ════════════════════════════════════════════
    def precompute_id_embedding(self, output_path: str):
        from train_id_embedding import XES3G5M_train_id_embedding
        XES3G5M_train_id_embedding(self.csv_path, self.index).train(output_path)
        self._preload_id()

    def precompute_tree_embedding(self, output_path: str):
        from train_tree_embedding import XES3G5M_train_tree_embedding
        XES3G5M_train_tree_embedding(
            self.csv_path, self.question_dict_path, self.kc_dict_path, self.index
        ).train(output_path)
        self._preload_tree()

    def precompute_text_embedding(self, output_path: str, text_dim: int = 64):
        """
        SentenceTransformer → PCA → L2-normalize → lưu pkl.

        Lý do L2-normalize sau PCA:
          - PCA output có variance không đồng đều giữa các dim
          - Normalize đảm bảo cosine similarity hoạt động đúng
          - Nhất quán với id/tree embedding đã normalize
        Lưu PCA object để có thể transform question mới sau này.
        """
        from sentence_transformers import SentenceTransformer

        with open(self.question_dict_path, "rb") as f:
            data = pickle.load(f)

        ids      = list(data.keys())
        contents = [data[i]["content"] for i in ids]

        model        = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2")
        emb_768      = model.encode(contents, show_progress_bar=True, convert_to_numpy=True)

        n_components = min(text_dim, emb_768.shape[1], len(ids) - 1)
        pca          = PCA(n_components=n_components, random_state=42)
        emb_reduced  = pca.fit_transform(emb_768).astype(np.float32)  # [N, text_dim]

        # L2-normalize — nhất quán với tree/id embedding
        emb_normed = normalize(emb_reduced, norm="l2", axis=1)

        result = {
            int(qid): torch.tensor(emb_normed[i], dtype=torch.float32)
            for i, qid in enumerate(ids)
        }

        os.makedirs(output_path, exist_ok=True)
        with open(os.path.join(output_path, "text_embedding.pkl"), "wb") as f:
            pickle.dump(result, f)
        # Lưu PCA để reuse
        with open(os.path.join(output_path, "pca.pkl"), "wb") as f:
            pickle.dump(pca, f)

        self._preload_text()

    def precompute_difficulty_embedding(self, output_path: str):
        """
        difficulty = 1 - mean(responses) per question.
        Clip [0, 1] để phòng noise (responses ngoài {0,1}).
        """
        df   = pd.read_csv(self.csv_path)
        diff = (
            df.groupby("questions")["responses"]
            .mean()
            .apply(lambda x: float(np.clip(1.0 - x, 0.0, 1.0)))
            .to_dict()
        )
        os.makedirs(output_path, exist_ok=True)
        with open(os.path.join(output_path, "difficulty_embedding.pkl"), "wb") as f:
            pickle.dump(diff, f)
        self._preload_difficulty()

    def precompute_scalar_features(self, output_path: str):
        """
        Precompute gap_time, time_of_day, session_pos cho toàn bộ df một lần.
        Kết quả được index theo row index của df (sau sort by user+timestamp).

        Lý do precompute ở đây thay vì dataset.py:
          - gap_time cần sort đúng thứ tự trước khi diff → phải biết toàn bộ sequence
          - session_pos cần biết độ dài sequence của từng user
          - Tránh duplicate logic giữa build_user_sequence và process_interaction
        """
        df = pd.read_csv(self.csv_path)
        if "uid" in df.columns:
            df = df.rename(columns={"uid": "user_id"})
        df = df.sort_values(["user_id", "timestamps"]).reset_index(drop=True)
        df["gap_time"] = df.groupby("user_id")["timestamps"].diff().fillna(0)

        N = len(df)

        # ── gap_time: log1p + z-normalize ──────────────
        raw_gap = df["gap_time"].values
        p99     = float(np.percentile(raw_gap, 99))
        clipped = np.clip(raw_gap, 0, p99)
        logged  = np.log1p(clipped).astype(np.float32)
        mean_g  = float(logged.mean())
        std_g   = float(logged.std()) + 1e-9
        gap_norm = torch.tensor(
            ((logged - mean_g) / std_g), dtype=torch.float32
        ).unsqueeze(1)   # [N, 1]

        gap_stats = {"p99": p99, "mean": mean_g, "std": std_g}

        # ── time_of_day: sin/cos(hour) ──────────────────
        # Dùng integer ms timestamp → hour trong ngày
        ts_ms  = df["timestamps"].values.astype(np.int64)
        hours  = (ts_ms // 1000 % 86400) // 3600
        angles = (2 * np.pi * hours / 24.0).astype(np.float32)
        time_f = torch.tensor(
            np.stack([np.sin(angles), np.cos(angles)], axis=1), dtype=torch.float32
        )  # [N, 2]

        # ── session_pos: position / (len - 1) ──────────
        user_lens  = df.groupby("user_id").size()
        df["_len"] = df["user_id"].map(user_lens)
        df["_pos"] = df.groupby("user_id").cumcount()
        sess_pos   = torch.tensor(
            (df["_pos"].values / np.maximum(df["_len"].values - 1, 1)).astype(np.float32),
            dtype=torch.float32,
        ).unsqueeze(1)   # [N, 1]

        os.makedirs(output_path, exist_ok=True)
        torch.save(
            {
                "gap_normalized": gap_norm,
                "time_features":  time_f,
                "session_pos":    sess_pos,
                "gap_stats":      gap_stats,
            },
            os.path.join(output_path, "scalar_features.pt"),
        )
        self._preload_scalars()

    # ════════════════════════════════════════════
    # Feature dims (để dataset.py tự build x_dim)
    # ════════════════════════════════════════════
    @property
    def id_dim(self) -> int:
        m = self.id_data["matrix"]
        return m.shape[1] if m is not None else 0

    @property
    def tree_dim(self) -> int:
        m = self.tree_data["matrix"]
        return m.shape[1] if m is not None else 0

    @property
    def text_dim(self) -> int:
        if self.text_dict:
            return next(iter(self.text_dict.values())).shape[0]
        return 0

    # ════════════════════════════════════════════
    # Per-question getters
    # ════════════════════════════════════════════
    def get_id_embedding(self, qid) -> torch.Tensor:
        if self.id_data["matrix"] is not None and qid in self.id_data["qid2idx"]:
            return self.id_data["matrix"][self.id_data["qid2idx"][qid]]
        return torch.zeros(self.id_dim or 64)

    def get_tree_embedding(self, qid) -> torch.Tensor:
        if self.tree_data["matrix"] is not None and qid in self.tree_data["qid2idx"]:
            return self.tree_data["matrix"][self.tree_data["qid2idx"][qid]]
        return torch.zeros(self.tree_dim or 64)

    def get_text_embedding(self, qid) -> torch.Tensor:
        if self.text_dict and qid in self.text_dict:
            return self.text_dict[qid]
        return torch.zeros(self.text_dim or 64)

    def get_difficulty(self, qid) -> torch.Tensor:
        if self.diff_dict and qid in self.diff_dict:
            return torch.tensor([self.diff_dict[qid]], dtype=torch.float32)
        return torch.tensor([0.5], dtype=torch.float32)   # prior trung tính

    # ════════════════════════════════════════════
    # Per-row scalar getters (indexed by df row)
    # ════════════════════════════════════════════
    def get_gap_normalized(self, row_indices: np.ndarray) -> torch.Tensor:
        """rows: int array → [T, 1]"""
        if self.gap_normalized is not None:
            return self.gap_normalized[row_indices]
        return torch.zeros(len(row_indices), 1)

    def get_time_features(self, row_indices: np.ndarray) -> torch.Tensor:
        """rows: int array → [T, 2]"""
        if self.time_features is not None:
            return self.time_features[row_indices]
        return torch.zeros(len(row_indices), 2)

    def get_session_pos(self, row_indices: np.ndarray) -> torch.Tensor:
        """rows: int array → [T, 1]"""
        if self.session_pos is not None:
            return self.session_pos[row_indices]
        return torch.zeros(len(row_indices), 1)
    

## Precompute text embedding example:
embedding = XES3G5M_exercises_embedding(
    csv_path="dataset/processed/XES3G5M/processed.csv",
    question_dict_path="dataset/processed/XES3G5M/question_dict.pkl",
    kc_dict_path="dataset/processed/XES3G5M/kc_dict.pkl",
)
embedding.precompute_text_embedding(
    output_path="dataset/processed/XES3G5M/excercices_embedding/text_embedding",
    text_dim=16,
)