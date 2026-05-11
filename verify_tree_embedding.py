"""
verify_tree_embedding_v3.py

Kiểm tra embedding có thực sự nắm được cấu trúc cây KC hay không,
qua 5 nhóm kiểm tra độc lập:

  [1] Correlation — Pearson + Spearman(Jaccard, CosineSim), binned.

  [2] Tree-level monotonicity
      same_leaf > same_chapter > same_root > no_share
      → Bằng chứng trực tiếp embedding hiểu HIERARCHY.

  [3] Precision@K với Jaccard threshold tiers (0.1, 0.25, 0.5)
      Lý do KHÔNG dùng "Jaccard > 0":
        XES3G5M cây rất nông → ~84% cặp random share ít nhất 1 root
        → baseline = 0.84 → lift chỉ 1.2x dù embedding tốt (misleading).
      Dùng threshold cao hơn mới phân biệt "gần thật sự" vs "share root chung".

  [4] Same-leaf clustering — intra vs inter cluster similarity.

  [5] Level embedding variance — leaf phải discriminative hơn root.
"""

import os
import pickle
import numpy as np
import torch
from collections import defaultdict
from scipy.stats import pearsonr, spearmanr

# ═══════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════
BASE           = "dataset/processed/XES3G5M/excercices_embedding/tree_embedding"
QDICT_PATH     = "dataset/processed/XES3G5M/question_dict.pkl"
Q_EMB_PATH     = f"{BASE}/question_tree_embeddings.pt"
LEVEL_EMB_PATH = f"{BASE}/question_level_embeddings.pt"
QID2IDX_PATH   = f"{BASE}/qid2idx.pkl"

N_SAMPLE   = 20_000  # cặp random cho [1] correlation
N_MONO     = 30_000  # cặp random cho [2] monotonicity — tăng để same_leaf/chapter có đủ n
TOPK       = 20      # k cho [3] precision@K
N_RECALL_Q = 500     # số query cho [3]

# ═══════════════════════════════════════════════
# Load
# ═══════════════════════════════════════════════
def load_tensor(path: str) -> np.ndarray:
    t = torch.load(path, map_location="cpu", weights_only=True)
    return t.numpy() if isinstance(t, torch.Tensor) else np.array(t)

q_emb = load_tensor(Q_EMB_PATH)           # [num_q, D] — L2-normalized
print(f"[load] q_emb shape: {q_emb.shape}")

with open(QID2IDX_PATH, "rb") as f:
    qid2idx_raw = pickle.load(f)
with open(QDICT_PATH, "rb") as f:
    qdict_raw = pickle.load(f)

qid2idx = {int(k): int(v) for k, v in qid2idx_raw.items()
           if str(k).lstrip("-").isdigit() and str(v).lstrip("-").isdigit()}
qdict   = {int(k): v for k, v in qdict_raw.items()
           if str(k).lstrip("-").isdigit()}

valid_qids = [qid for qid in qid2idx
              if qid in qdict and qid2idx[qid] < len(q_emb)]
print(f"[info] Valid questions: {len(valid_qids)} / {len(qid2idx)}\n")

if not valid_qids:
    raise ValueError("Không có question nào khớp. Kiểm tra key type.")

# ═══════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════
def get_nodes(qid: int) -> set:
    nodes = set()
    for r in qdict[qid].get("kc_routes", []):
        nodes.update(r.split("----"))
    return nodes

# Cache node sets để tránh tính lại nhiều lần trong [3]
_node_cache: dict = {}
def get_nodes_cached(qid: int) -> set:
    if qid not in _node_cache:
        _node_cache[qid] = get_nodes(qid)
    return _node_cache[qid]

def jaccard(qid1: int, qid2: int) -> float:
    n1 = get_nodes_cached(qid1)
    n2 = get_nodes_cached(qid2)
    u  = len(n1 | n2)
    return len(n1 & n2) / u if u > 0 else 0.0

def get_level_nodes(qid: int, level: int) -> set:
    """level=0 → leaf (rightmost), level=-1 → root (leftmost)."""
    nodes = set()
    for r in qdict[qid].get("kc_routes", []):
        parts = r.split("----")
        if not parts:
            continue
        if level == 0:
            nodes.add(parts[-1])
        elif level == -1:
            nodes.add(parts[0])
        elif 0 < level < len(parts):
            nodes.add(parts[-(level + 1)])
    return nodes

# ═══════════════════════════════════════════════
# [1] Correlation
# ═══════════════════════════════════════════════
print("═"*55)
print("[1] CORRELATION  —  Jaccard vs CosineSim")
print("═"*55)

rng  = np.random.default_rng(42)
idx1 = rng.integers(0, len(valid_qids), size=N_SAMPLE)
idx2 = rng.integers(0, len(valid_qids), size=N_SAMPLE)

sims, jaccs = [], []
for a, b in zip(idx1, idx2):
    q1, q2 = valid_qids[a], valid_qids[b]
    if q1 == q2:
        continue
    i, j = qid2idx[q1], qid2idx[q2]
    # q_emb đã L2-normalize → dot product = cosine sim
    sims.append(float(q_emb[i] @ q_emb[j]))
    jaccs.append(jaccard(q1, q2))

sims  = np.array(sims)
jaccs = np.array(jaccs)

pearson_r,  pearson_p  = pearsonr(jaccs, sims)
spearman_r, spearman_p = spearmanr(jaccs, sims)
print(f"  Pearson  r = {pearson_r:.3f}  (p={pearson_p:.1e})")
print(f"  Spearman r = {spearman_r:.3f}  (p={spearman_p:.1e})")
print(f"  Hướng dẫn: Pearson > 0.85 = rất tốt | Spearman > 0.7 = tốt")

print("\n  Binned by Jaccard:")
for lo, hi in [(0.0,0.01),(0.01,0.1),(0.1,0.25),(0.25,0.5),(0.5,0.75),(0.75,1.01)]:
    mask = (jaccs >= lo) & (jaccs < hi)
    if mask.sum() > 0:
        print(f"    [{lo:.2f}, {hi:.2f}):  mean_sim={np.mean(sims[mask]):+.3f}  "
              f"std={np.std(sims[mask]):.3f}  n={mask.sum()}")

# ═══════════════════════════════════════════════
# [2] Tree-level monotonicity
# ═══════════════════════════════════════════════
print("\n" + "═"*55)
print("[2] TREE-LEVEL MONOTONICITY")
print("    Kỳ vọng: same_leaf > same_chapter > same_root > no_share")
print("═"*55)

groups  = {"same_leaf": [], "same_chapter": [], "same_root": [], "no_share": []}
rng2    = np.random.default_rng(0)
checked = 0

# Pool lớn hơn để đảm bảo same_leaf/chapter có đủ mẫu
pool_size = N_MONO * 5
a_pool = rng2.integers(0, len(valid_qids), size=pool_size)
b_pool = rng2.integers(0, len(valid_qids), size=pool_size)

for a, b in zip(a_pool, b_pool):
    if checked >= N_MONO:
        break
    q1, q2 = valid_qids[a], valid_qids[b]
    if q1 == q2:
        continue

    leaf1 = get_level_nodes(q1,  0);  leaf2 = get_level_nodes(q2,  0)
    chap1 = get_level_nodes(q1,  1);  chap2 = get_level_nodes(q2,  1)
    root1 = get_level_nodes(q1, -1);  root2 = get_level_nodes(q2, -1)

    i, j = qid2idx[q1], qid2idx[q2]
    sim  = float(q_emb[i] @ q_emb[j])
    jac  = jaccard(q1, q2)

    if leaf1 & leaf2:
        groups["same_leaf"].append(sim)
    elif chap1 & chap2:
        groups["same_chapter"].append(sim)
    elif root1 & root2:
        groups["same_root"].append(sim)
    elif jac == 0.0:
        groups["no_share"].append(sim)
    checked += 1

order     = ["same_leaf", "same_chapter", "same_root", "no_share"]
prev_mean = None
monotone  = True
for grp in order:
    vals = groups[grp]
    if not vals:
        print(f"  {grp:<16}: (không có mẫu)")
        continue
    mean = np.mean(vals)
    flag = ""
    if prev_mean is not None and mean > prev_mean:
        monotone = False
        flag = "  ← ⚠ VIOLATED"
    print(f"  {grp:<16}: mean_sim={mean:+.3f}  std={np.std(vals):.3f}  n={len(vals)}{flag}")
    prev_mean = mean

print(f"\n  Monotonicity: {'✓ OK' if monotone else '✗ VIOLATED — embedding chưa học đủ hierarchy'}")

# ═══════════════════════════════════════════════
# [3] Precision@K — Jaccard threshold tiers
# ═══════════════════════════════════════════════
print("\n" + "═"*55)
print(f"[3] PRECISION@{TOPK}  —  Jaccard threshold tiers")
print("    Kỳ vọng: precision >> baseline tại mỗi threshold")
print("    (Không dùng Jaccard>0 vì baseline~0.84 trong XES3G5M)")
print("═"*55)

JAC_THRESHOLDS = [0.1, 0.25, 0.5]
idx2qid        = {v: k for k, v in qid2idx.items()}
rng3           = np.random.default_rng(7)
query_ids      = rng3.choice(valid_qids,
                             size=min(N_RECALL_Q, len(valid_qids)),
                             replace=False)

# Baseline: P(jaccard >= thr | random pair)
print("\n  Estimating baselines (5000 random pairs)...")
rand_a    = rng3.choice(valid_qids, size=5000)
rand_b    = rng3.choice(valid_qids, size=5000)
rand_jacs = np.array([jaccard(int(a), int(b))
                      for a, b in zip(rand_a, rand_b) if a != b])
baseline_rates = {thr: float((rand_jacs >= thr).mean()) for thr in JAC_THRESHOLDS}

print("  Baseline rates (random pairs):")
for thr in JAC_THRESHOLDS:
    print(f"    Jaccard ≥ {thr:.2f}: {baseline_rates[thr]:.3f}")

# Precision@K — dùng dot product thay vì sklearn cosine_similarity
precision_at_k = {thr: [] for thr in JAC_THRESHOLDS}

for q in query_ids:
    i         = qid2idx[q]
    sims_row  = (q_emb @ q_emb[i]).copy()   # [num_q] — vectorized
    sims_row[i] = -2.0                       # exclude self
    topk_idx  = np.argpartition(sims_row, -TOPK)[-TOPK:]

    # Tính jaccard cho top-K một lần, tận dụng cache
    topk_jacs = np.array([
        jaccard(q, idx2qid[idx])
        for idx in topk_idx
        if idx in idx2qid
    ])

    for thr in JAC_THRESHOLDS:
        precision_at_k[thr].append(float((topk_jacs >= thr).mean()))

print(f"\n  Precision@{TOPK} vs baseline:")
overall_ok = True
for thr in JAC_THRESHOLDS:
    p    = float(np.mean(precision_at_k[thr]))
    base = baseline_rates[thr]
    lift = p / max(base, 1e-6)
    ok   = lift > 2.0
    if not ok:
        overall_ok = False
    print(f"    Jaccard ≥ {thr:.2f}:  precision={p:.3f}  "
          f"baseline={base:.3f}  lift={lift:.1f}x  {'✓' if ok else '⚠'}")

print(f"\n  {'✓ Embedding phân biệt tốt' if overall_ok else '⚠ Lift thấp — chưa đủ discriminative'}")
print(f"  Hướng dẫn: lift(≥0.25) > 3x = tốt, > 5x = rất tốt")

# ═══════════════════════════════════════════════
# [4] Same-leaf clustering
# ═══════════════════════════════════════════════
print("\n" + "═"*55)
print("[4] SAME-LEAF CLUSTERING")
print("    Kỳ vọng: intra_sim >> inter_sim")
print("═"*55)

leaf_groups: dict = defaultdict(list)
for qid in valid_qids:
    for leaf in get_level_nodes(qid, 0):
        leaf_groups[leaf].append(qid2idx[qid])

valid_groups = {k: v for k, v in leaf_groups.items() if len(v) >= 3}
print(f"  Leaf clusters có ≥ 3 questions: {len(valid_groups)}")

intra_sims, inter_sims = [], []
gap = None

if valid_groups:
    group_keys = list(valid_groups.keys())
    rng4       = np.random.default_rng(13)

    for key in rng4.choice(group_keys, size=min(100, len(group_keys)), replace=False):
        members = valid_groups[key]
        if len(members) < 2:
            continue
        for _ in range(min(20, len(members))):
            a, b = rng4.choice(members, 2, replace=False)
            intra_sims.append(float(q_emb[a] @ q_emb[b]))

    for _ in range(len(intra_sims)):
        k1, k2 = rng4.choice(group_keys, 2, replace=False)
        a = int(rng4.choice(valid_groups[k1]))
        b = int(rng4.choice(valid_groups[k2]))
        inter_sims.append(float(q_emb[a] @ q_emb[b]))

    gap = float(np.mean(intra_sims) - np.mean(inter_sims))
    print(f"  Intra-cluster mean sim : {np.mean(intra_sims):+.3f}  "
          f"(std={np.std(intra_sims):.3f}, n={len(intra_sims)})")
    print(f"  Inter-cluster mean sim : {np.mean(inter_sims):+.3f}  "
          f"(std={np.std(inter_sims):.3f}, n={len(inter_sims)})")
    print(f"  Gap (intra - inter)    : {gap:+.3f}")
    print(f"  {'✓ Phân cụm rõ theo leaf' if gap > 0.2 else '⚠ Gap nhỏ — leaf clustering chưa rõ'}")
else:
    print("  Không đủ cluster để test.")

# ═══════════════════════════════════════════════
# [5] Level embedding variance
# ═══════════════════════════════════════════════
print("\n" + "═"*55)
print("[5] LEVEL EMBEDDING VARIANCE  (leaf=0  →  root=n-1)")
print("    Kỳ vọng: var(leaf) > var(root)")
print("═"*55)

leaf_var = root_var = None
if os.path.exists(LEVEL_EMB_PATH):
    level_emb = load_tensor(LEVEL_EMB_PATH)   # [num_q, n_levels, D]
    n_levels  = level_emb.shape[1]

    for lv in range(n_levels):
        lv_mat = level_emb[:, lv, :]
        var    = float(np.var(lv_mat, axis=0).mean())
        norm   = float(np.linalg.norm(lv_mat, axis=1).mean())
        label  = "leaf" if lv == 0 else ("root" if lv == n_levels - 1 else f"mid-{lv}")
        print(f"  Level {lv} ({label:<6}): mean_var={var:.4f}  mean_norm={norm:.3f}")

    leaf_var = float(np.var(level_emb[:, 0,  :], axis=0).mean())
    root_var = float(np.var(level_emb[:, -1, :], axis=0).mean())
    if leaf_var > root_var:
        print(f"\n  ✓ leaf_var ({leaf_var:.4f}) > root_var ({root_var:.4f}) — hierarchy hợp lý")
    else:
        print(f"\n  ⚠ leaf_var ({leaf_var:.4f}) ≤ root_var ({root_var:.4f}) — root quá phân tán")
else:
    print("  Bỏ qua — không tìm thấy question_level_embeddings.pt")

# ═══════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════
print("\n" + "═"*55)
print("SUMMARY")
print("═"*55)
print(f"  [1] Pearson r            : {pearson_r:.3f}  "
      f"{'✓' if pearson_r > 0.7 else '⚠'}   "
      f"Spearman r: {spearman_r:.3f}  {'✓' if spearman_r > 0.6 else '⚠'}")
print(f"  [2] Monotonicity         : {'✓ OK' if monotone else '✗ VIOLATED'}")
if precision_at_k:
    for thr in JAC_THRESHOLDS:
        lift = float(np.mean(precision_at_k[thr])) / max(baseline_rates[thr], 1e-6)
        print(f"  [3] Precision@{TOPK}(≥{thr:.2f})  : lift={lift:.1f}x  "
              f"{'✓' if lift > 2 else '⚠'}")
if gap is not None:
    print(f"  [4] Cluster gap          : {gap:+.3f}  {'✓' if gap > 0.2 else '⚠'}")
if leaf_var is not None and root_var is not None:
    print(f"  [5] Leaf var > Root var  : {'✓' if leaf_var > root_var else '⚠'}  "
          f"({leaf_var:.4f} vs {root_var:.4f})")
print("═"*55)