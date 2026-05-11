"""
test_build_dataset.py

Kiểm tra từng bước:
  [1] Load offline_embedding — verify dim và shape
  [2] Scalar features — gap_time, time_of_day, session_pos
  [3] Per-question getters — id, tree, text, difficulty
  [4] Build user sequence — shape, dtype, NaN
  [5] KTDriftDataset __getitem__ — shape, bucket length
  [6] Ablation modes — x_dim đúng với từng mode
  [7] DataLoader batch — collate hoạt động
"""

import sys
import traceback
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

# ── Paths — chỉnh lại nếu cần ───────────────────────────────────────────────
CSV_PATH           = "dataset/processed/XES3G5M/processed.csv"
KC_DICT_PATH       = "dataset/processed/XES3G5M/kc_dict.pkl"
QUESTION_DICT_PATH = "dataset/processed/XES3G5M/question_dict.pkl"
BASE_EMB_PATH      = "dataset/processed/XES3G5M/excercices_embedding"

N_USERS_SMOKE = 5   # số user dùng cho smoke test sequence builder
# ─────────────────────────────────────────────────────────────────────────────

PASS = "\033[92m✓\033[0m"
FAIL = "\033[91m✗\033[0m"
SKIP = "\033[93m~\033[0m"

def check(name: str, cond: bool, detail: str = ""):
    mark = PASS if cond else FAIL
    print(f"  {mark} {name}" + (f"  ({detail})" if detail else ""))
    return cond

def section(title: str):
    print(f"\n{'─'*55}")
    print(f"  {title}")
    print(f"{'─'*55}")


# ════════════════════════════════════════════════════════
# [1] Load offline_embedding
# ════════════════════════════════════════════════════════
section("[1] Load XES3G5M_exercises_embedding")
try:
    from get_feature import XES3G5M_exercises_embedding
    ex_emb = XES3G5M_exercises_embedding(
        csv_path=CSV_PATH,
        kc_dict_path=KC_DICT_PATH,
        question_dict_path=QUESTION_DICT_PATH,
        base_emb_path=BASE_EMB_PATH,
    )

    check("id_data matrix loaded",   ex_emb.id_data["matrix"]   is not None,
          f"shape={tuple(ex_emb.id_data['matrix'].shape) if ex_emb.id_data['matrix'] is not None else 'None'}")
    check("tree_data matrix loaded", ex_emb.tree_data["matrix"] is not None,
          f"shape={tuple(ex_emb.tree_data['matrix'].shape) if ex_emb.tree_data['matrix'] is not None else 'None'}")
    check("text_dict loaded",        ex_emb.text_dict is not None,
          f"n={len(ex_emb.text_dict) if ex_emb.text_dict else 0}")
    check("diff_dict loaded",        ex_emb.diff_dict is not None,
          f"n={len(ex_emb.diff_dict) if ex_emb.diff_dict else 0}")

    print(f"\n  Dims  →  id={ex_emb.id_dim}  tree={ex_emb.tree_dim}  text={ex_emb.text_dim}")
    EMB_OK = True

except Exception:
    traceback.print_exc()
    print(f"  {FAIL} Không thể load offline_embedding — dừng test.")
    sys.exit(1)


# ════════════════════════════════════════════════════════
# [2] Scalar features
# ════════════════════════════════════════════════════════
section("[2] Scalar features (gap_time / time_of_day / session_pos)")

scalar_ok = ex_emb.gap_normalized is not None

if not scalar_ok:
    print(f"  {SKIP} scalar_features.pt chưa tồn tại → chạy precompute...")
    try:
        import os
        scalar_out = os.path.join(BASE_EMB_PATH, "scalar_features")
        ex_emb.precompute_scalar_features(scalar_out)
        scalar_ok = ex_emb.gap_normalized is not None
    except Exception:
        traceback.print_exc()

if scalar_ok:
    N = ex_emb.gap_normalized.shape[0]
    check("gap_normalized shape",   ex_emb.gap_normalized.shape == (N, 1),
          str(tuple(ex_emb.gap_normalized.shape)))
    check("time_features shape",    ex_emb.time_features.shape  == (N, 2),
          str(tuple(ex_emb.time_features.shape)))
    check("session_pos shape",      ex_emb.session_pos.shape    == (N, 1),
          str(tuple(ex_emb.session_pos.shape)))
    check("gap_normalized no NaN",  not torch.isnan(ex_emb.gap_normalized).any().item())
    check("time_features in [-1,1]",
          ex_emb.time_features.abs().max().item() <= 1.0 + 1e-5)
    check("session_pos in [0,1]",
          ex_emb.session_pos.min().item() >= -1e-5 and
          ex_emb.session_pos.max().item() <= 1.0 + 1e-5)
else:
    print(f"  {FAIL} Không thể precompute scalar features")


# ════════════════════════════════════════════════════════
# [3] Per-question getters
# ════════════════════════════════════════════════════════
section("[3] Per-question getters")

# Lấy một qid hợp lệ
sample_qid = next(iter(ex_emb.id_data["qid2idx"])) if ex_emb.id_data["qid2idx"] else None

if sample_qid is not None:
    id_emb   = ex_emb.get_id_embedding(sample_qid)
    tree_emb = ex_emb.get_tree_embedding(sample_qid)
    text_emb = ex_emb.get_text_embedding(sample_qid)
    diff     = ex_emb.get_difficulty(sample_qid)

    check("id_embedding shape",   id_emb.shape   == (ex_emb.id_dim,),   str(tuple(id_emb.shape)))
    check("tree_embedding shape", tree_emb.shape  == (ex_emb.tree_dim,), str(tuple(tree_emb.shape)))
    check("text_embedding shape", text_emb.shape  == (ex_emb.text_dim,), str(tuple(text_emb.shape)))
    check("difficulty shape",     diff.shape      == (1,),               str(tuple(diff.shape)))
    check("difficulty in [0,1]",  0.0 <= diff.item() <= 1.0,            f"{diff.item():.3f}")

    # L2-norm check — id/tree/text đều phải đã normalize
    for name, emb in [("id", id_emb), ("tree", tree_emb), ("text", text_emb)]:
        if emb.norm().item() > 0:
            check(f"{name}_embedding L2-norm ≈ 1",
                  abs(emb.norm().item() - 1.0) < 0.05,
                  f"norm={emb.norm().item():.4f}")

    # Fallback cho qid không tồn tại
    fake_qid   = -9999
    id_fall    = ex_emb.get_id_embedding(fake_qid)
    tree_fall  = ex_emb.get_tree_embedding(fake_qid)
    text_fall  = ex_emb.get_text_embedding(fake_qid)
    diff_fall  = ex_emb.get_difficulty(fake_qid)
    check("id fallback zeros",    id_fall.sum().item()   == 0.0)
    check("tree fallback zeros",  tree_fall.sum().item() == 0.0)
    check("text fallback zeros",  text_fall.sum().item() == 0.0)
    check("diff fallback 0.5",    diff_fall.item()       == 0.5)
else:
    print(f"  {SKIP} Không tìm được sample qid")


# ════════════════════════════════════════════════════════
# [4] Build user sequence
# ════════════════════════════════════════════════════════
section("[4] Build user sequence")
try:
    from dataset import XES3G5M_Interaction_Processor

    proc  = XES3G5M_Interaction_Processor(CSV_PATH, ex_emb, ablation_mode="full")
    users = proc.df["user_id"].unique()

    # Thử N_USERS_SMOKE user đầu
    ok_count = 0
    for uid in users[:N_USERS_SMOKE]:
        seq = proc.build_user_sequence(uid)
        T   = seq.shape[0]
        ok  = (
            seq.shape[1] == proc.x_dim
            and not torch.isnan(seq).any().item()
            and T >= 1
        )
        if ok:
            ok_count += 1

    check(f"build_user_sequence OK ({N_USERS_SMOKE} users)",
          ok_count == N_USERS_SMOKE,
          f"{ok_count}/{N_USERS_SMOKE} passed")

    # Kiểm tra shape chi tiết trên 1 user
    uid0 = users[0]
    seq0 = proc.build_user_sequence(uid0)
    T0   = seq0.shape[0]
    check("x_sequence dtype float32",   seq0.dtype == torch.float32)
    check("x_sequence dim == x_dim",    seq0.shape[1] == proc.x_dim,
          f"{seq0.shape[1]} == {proc.x_dim}")
    check("x_sequence no NaN",         not torch.isnan(seq0).any().item())
    check("x_sequence no Inf",         not torch.isinf(seq0).any().item())

    # Kiểm tra response one-hot — col [id+tree+text+1 : +2] là one-hot
    resp_start = ex_emb.id_dim + ex_emb.tree_dim + ex_emb.text_dim + 1
    resp_cols  = seq0[:, resp_start : resp_start + 2]
    check("response one-hot sums to 1", resp_cols.sum(dim=1).allclose(torch.ones(T0)))

    PROC_OK = True

except Exception:
    traceback.print_exc()
    PROC_OK = False


# ════════════════════════════════════════════════════════
# [5] KTDataset __getitem__
# ════════════════════════════════════════════════════════
section("[5] KTDataset __getitem__")
try:
    from dataset import KTDataset

    ds   = KTDataset(proc, cache_sequences=True)
    item = ds[0]

    check("keys present",       {"x_sequence", "user_id"} <= item.keys())
    check("x_sequence tensor",  isinstance(item["x_sequence"], torch.Tensor))
    check("x_sequence shape",   item["x_sequence"].shape[1] == proc.x_dim,
          f"dim={item['x_sequence'].shape[1]}")

    # Cache hit — second call phải trả về cùng tensor object
    item2 = ds[0]
    check("cache hit returns same tensor",
          item["x_sequence"].data_ptr() == item2["x_sequence"].data_ptr())

    DS_OK = True

except Exception:
    traceback.print_exc()
    DS_OK = False


# ════════════════════════════════════════════════════════
# [6] Ablation modes
# ════════════════════════════════════════════════════════
section("[6] Ablation modes — x_dim")
try:
    from dataset import build_datasets

    # Expected dims
    base_scalar = 1 + 2 + 1 + 2 + 1   # difficulty + response + gap + time + sess_pos = 7
    expected = {
        "full":           ex_emb.id_dim + ex_emb.tree_dim + ex_emb.text_dim + base_scalar,
        "no_text":        ex_emb.id_dim + ex_emb.tree_dim                   + base_scalar,
        "no_tree":        ex_emb.id_dim                   + ex_emb.text_dim + base_scalar,
        "no_tree_no_text":ex_emb.id_dim                                     + base_scalar,
    }

    for mode, exp_dim in expected.items():
        ds_mode = build_datasets(CSV_PATH, ex_emb, ablation_mode=mode, cache=False)
        got_dim = ds_mode.x_dim
        check(f"mode={mode:<16} x_dim={exp_dim}",
              got_dim == exp_dim,
              f"got {got_dim}")

except Exception:
    traceback.print_exc()


# ════════════════════════════════════════════════════════
# [7] DataLoader batch
# ════════════════════════════════════════════════════════
section("[7] DataLoader — collate batch of 4")
try:
    def collate_fn(batch):
        xs = pad_sequence([b["x_sequence"] for b in batch], batch_first=True)
        return {"x_sequence": xs, "user_id": [b["user_id"] for b in batch]}

    ds_full = build_datasets(CSV_PATH, ex_emb, "full", cache=True)
    loader  = DataLoader(ds_full, batch_size=4, shuffle=False, collate_fn=collate_fn)
    batch   = next(iter(loader))

    B, T, D = batch["x_sequence"].shape
    check("batch x_sequence rank 3",  len(batch["x_sequence"].shape) == 3,
          f"{tuple(batch['x_sequence'].shape)}")
    check("batch size == 4",          B == 4,  f"B={B}")
    check("x_dim matches",            D == ds_full.x_dim, f"{D} == {ds_full.x_dim}")
    check("no NaN in batch",          not torch.isnan(batch["x_sequence"]).any().item())

except Exception:
    traceback.print_exc()


# ════════════════════════════════════════════════════════
# Summary
# ════════════════════════════════════════════════════════
print(f"\n{'═'*55}")
print("  DONE")
print(f"{'═'*55}\n")