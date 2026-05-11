"""
dataset.py

KTDataset — Knowledge Tracing dataset với ablation mode.

x_sequence shape phụ thuộc ablation_mode:
  ┌──────────────────────┬────────────────────────────────────────────────┐
  │ mode                 │ features included                              │
  ├──────────────────────┼────────────────────────────────────────────────┤
  │ "full"               │ id + tree + text + difficulty + response +     │
  │                      │ gap + time + session_pos                       │
  │ "no_text"            │ id + tree        + difficulty + response + ... │
  │ "no_tree"            │ id       + text  + difficulty + response + ... │
  │ "no_tree_no_text"    │ id               + difficulty + response + ... │
  └──────────────────────┴────────────────────────────────────────────────┘

Tất cả feature đều lấy từ offline_embedding — dataset.py không tự tính gì.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
import pandas as pd
from torch.utils.data import Dataset

from get_feature import XES3G5M_exercises_embedding

ABLATION_MODES = ("full", "no_text", "no_tree", "no_tree_no_text")


class XES3G5M_Interaction_Processor:
    def __init__(
        self,
        csv_path:             str,
        ex_embedding_manager: XES3G5M_exercises_embedding,
        ablation_mode:        str = "full",
    ):
        assert ablation_mode in ABLATION_MODES, \
            f"ablation_mode phải là một trong {ABLATION_MODES}"

        self.ex_emb        = ex_embedding_manager
        self.ablation_mode = ablation_mode

        self.df = pd.read_csv(csv_path)
        if "uid" in self.df.columns:
            self.df = self.df.rename(columns={"uid": "user_id"})
        self.df = self.df.sort_values(["user_id", "timestamps"]).reset_index(drop=True)

        # user_id → list[int row index]
        self._user_row_indices: dict = (
            self.df.groupby("user_id")
            .apply(lambda g: g.index.tolist())
            .to_dict()
        )

        self._qids      = self.df["questions"].values
        self._responses = self.df["responses"].values.astype(np.int64)

        self.x_dim = self._compute_x_dim()
        print(f"[dataset] ablation_mode={ablation_mode}  x_dim={self.x_dim}")

    def _compute_x_dim(self) -> int:
        ex   = self.ex_emb
        mode = self.ablation_mode
        dim  = ex.id_dim
        dim += ex.tree_dim if "no_tree" not in mode else 0
        dim += ex.text_dim if "no_text" not in mode else 0
        dim += 1    # difficulty
        dim += 2    # response one-hot
        dim += 1    # gap_time
        dim += 2    # time_of_day (sin, cos)
        dim += 1    # session_pos

        print("id_dim   =", ex.id_dim)
        print("tree_dim =", ex.tree_dim)
        print("text_dim =", ex.text_dim)

        return dim

    def build_user_sequence(self, uid) -> torch.Tensor:
        """[T, x_dim] — toàn bộ feature lấy từ ex_emb."""
        rows = np.array(self._user_row_indices[uid])
        qids = self._qids[rows]
        mode = self.ablation_mode
        ex   = self.ex_emb

        q_parts = []
        for qid in qids:
            parts = [ex.get_id_embedding(qid)]
            if "no_tree" not in mode:
                parts.append(ex.get_tree_embedding(qid))
            if "no_text" not in mode:
                parts.append(ex.get_text_embedding(qid))
            parts.append(ex.get_difficulty(qid))
            q_parts.append(torch.cat(parts))
        e_q = torch.stack(q_parts)                                  # [T, ...]

        resp = torch.tensor(self._responses[rows], dtype=torch.long)
        a_t  = F.one_hot(resp, num_classes=2).float()               # [T, 2]

        gap      = ex.get_gap_normalized(rows)                      # [T, 1]
        time_f   = ex.get_time_features(rows)                       # [T, 2]
        sess_pos = ex.get_session_pos(rows)                         # [T, 1]

        return torch.cat([e_q, a_t, gap, time_f, sess_pos], dim=1)  # [T, x_dim]


class KTDataset(Dataset):
    def __init__(
        self,
        processor:       XES3G5M_Interaction_Processor,
        cache_sequences: bool = True,
    ):
        self.processor = processor
        self.users     = processor.df["user_id"].unique()
        self._do_cache = cache_sequences
        self._cache:   dict[int, torch.Tensor] = {}

    def __len__(self) -> int:
        return len(self.users)

    def __getitem__(self, idx: int) -> dict:
        uid = self.users[idx]

        if self._do_cache and idx in self._cache:
            x_sequence = self._cache[idx]
        else:
            x_sequence = self.processor.build_user_sequence(uid)
            if self._do_cache:
                self._cache[idx] = x_sequence

        return {
            "x_sequence": x_sequence,   # [T, x_dim]
            "user_id":    uid,
        }

    @property
    def x_dim(self) -> int:
        return self.processor.x_dim


def build_datasets(
    csv_path:      str,
    ex_emb:        XES3G5M_exercises_embedding,
    ablation_mode: str  = "full",
    cache:         bool = True,
) -> KTDataset:
    """
    Shortcut build dataset 1 dòng.

        ds = build_datasets(csv, ex_emb, "full")
        ds = build_datasets(csv, ex_emb, "no_text")
        ds = build_datasets(csv, ex_emb, "no_tree")
        ds = build_datasets(csv, ex_emb, "no_tree_no_text")
    """
    proc = XES3G5M_Interaction_Processor(csv_path, ex_emb, ablation_mode)
    return KTDataset(proc, cache_sequences=cache)