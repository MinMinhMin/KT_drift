# Usage — XES3G5M Dataset Pipeline

## Feature inventory

Mỗi timestep `t` trong `x_sequence` là concat của các feature sau:

| Feature | Dim | Nguồn | Mô tả |
|---|---|---|---|
| `id_embedding` | 32 | `train_id_embedding.py` | Learnable embedding cho question ID. Train bằng next-question prediction — bài có prerequisite/co-occurrence giống nhau sẽ gần nhau trong không gian embedding. |
| `tree_embedding` | 32 | `train_tree_embedding.py` | Embedding cấu trúc cây KC.  Path Reconstruction ép học kiến trúc cây|
| `text_embedding` | 64 | SentenceTransformer + PCA | Embedding nội dung đề bài (768d → PCA 64d → L2-norm). Bài có nội dung gần nghĩa sẽ gần nhau. |
| `difficulty` | 1 | `precompute_difficulty_embedding` | `1 - mean(responses)` per question, clip [0,1]. 0 = dễ nhất, 1 = khó nhất. Fallback = 0.5 nếu không có data. |
| `response` | 2 | dataset.py (online) | One-hot `{[1,0]=sai, [0,1]=đúng}`. Tính tại dataset vì phụ thuộc interaction cụ thể. |
| `gap_time` | 1 | `precompute_scalar_features` | `log1p(clip(gap_ms, 0, p99))` → z-normalize. Khoảng cách thời gian từ tương tác trước. Gap = 0 cho tương tác đầu tiên của mỗi user. |
| `time_of_day` | 2 | `precompute_scalar_features` | `[sin(2π·hour/24), cos(2π·hour/24)]`. Encoding tròn để tránh discontinuity ở 23:00→00:00. |
| `session_pos` | 1 | `precompute_scalar_features` | `position / (len - 1)`, normalize [0,1]. Vị trí tương tác trong toàn bộ lịch sử của user. |

**x_dim = id_dim + tree_dim + text_dim + 1 + 2 + 1 + 2 + 1 = 32+32+64+7 = 135** (full mode)

---

## Cài đặt và precompute (chạy một lần)

```python
from get_feature import XES3G5M_exercises_embedding

ex_emb = XES3G5M_exercises_embedding(
    csv_path           = "dataset/processed/XES3G5M/processed.csv",
    kc_dict_path       = "dataset/processed/XES3G5M/kc_dict.pkl",
    question_dict_path = "dataset/processed/XES3G5M/question_dict.pkl",
    base_emb_path      = "dataset/processed/XES3G5M/excercices_embedding",
)

# Chạy mỗi cái một lần, sau đó tự load từ disk
ex_emb.precompute_id_embedding(   "...excercices_embedding/id_embedding")
ex_emb.precompute_tree_embedding( "...excercices_embedding/tree_embedding")
ex_emb.precompute_text_embedding( "...excercices_embedding/text_embedding")
ex_emb.precompute_difficulty_embedding("...excercices_embedding/difficulty_embedding")
ex_emb.precompute_scalar_features("...excercices_embedding/scalar_features")
```

Sau khi precompute xong, `_preload_all()` tự chạy → các lần khởi tạo tiếp theo chỉ load từ disk, không tính lại.

---

## Build dataset

```python
from dataset import build_datasets

ds = build_datasets(
    csv_path      = "dataset/processed/XES3G5M/processed.csv",
    ex_emb        = ex_emb,
    ablation_mode = "full",   # xem bảng ablation bên dưới
    cache         = True,     # cache sequence vào RAM sau epoch đầu
)

print(ds.x_dim)  # input dim cho model
```

### Ablation modes

| Mode | Features | x_dim |
|---|---|---|
| `"full"` | id + tree + text + difficulty + response + gap + time + session_pos | 135 |
| `"no_text"` | id + tree + difficulty + response + gap + time + session_pos | 71 |
| `"no_tree"` | id + text + difficulty + response + gap + time + session_pos | 102 |
| `"no_tree_no_text"` | id + difficulty + response + gap + time + session_pos | 39 |

*Dim trên giả sử id=tree=32, text=64. Thực tế lấy từ `ex_emb.id_dim`, `ex_emb.tree_dim`, `ex_emb.text_dim`.*

```python
# Chạy 4 ablation song song
from dataset import build_datasets

modes = ["full", "no_text", "no_tree", "no_tree_no_text"]
datasets = {m: build_datasets(csv, ex_emb, m) for m in modes}

for m, ds in datasets.items():
    print(f"{m:<20} x_dim={ds.x_dim}")
```

---

## DataLoader

```python
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

def collate_fn(batch):
    return {
        "x_sequence": pad_sequence([b["x_sequence"] for b in batch], batch_first=True),  # [B, T_max, D]
        "user_id":    [b["user_id"] for b in batch],
    }

loader = DataLoader(ds, batch_size=32, shuffle=True, collate_fn=collate_fn, num_workers=4)

for batch in loader:
    x   = batch["x_sequence"]   # [B, T_max, x_dim]
    # ...
```

### Cấu trúc mỗi item

```python
item = ds[0]
item["x_sequence"]  # Tensor [T, x_dim]  — sequence feature của 1 user
item["user_id"]     # int/str             — user identifier
```

---

## Kiểm tra

```bash
python test_build_dataset.py
```

Chạy 7 nhóm check tự động: load, scalar shape, getter correctness, L2-norm, NaN/Inf, ablation dim, DataLoader batch.
