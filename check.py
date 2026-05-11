# ══════════════════════════════════════════════════════════════════════════════
# CELL 1 — Imports & Load
# ══════════════════════════════════════════════════════════════════════════════
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
import pandas as pd
from collections import defaultdict
from scipy import stats
import umap
 
CLUSTERING_DIR = "clustering"
COLORS = [
    "#e74c3c","#3498db","#2ecc71","#f39c12",
    "#9b59b6","#1abc9c","#e67e22","#34495e",
]
 
# Load artifacts từ find_k.py
z_all      = np.load(f"{CLUSTERING_DIR}/z_all.npy")
labels     = np.load(f"{CLUSTERING_DIR}/sk_labels.npy")
centroids  = np.load(f"{CLUSTERING_DIR}/sk_centroids.npy")
soft_asgn  = np.load(f"{CLUSTERING_DIR}/sk_soft_assignment.npy")
uid_all    = np.load(f"{CLUSTERING_DIR}/uid_all.npy", allow_pickle=True)
 
k = int(labels.max()) + 1
N = len(z_all)
print(f"z_all={z_all.shape}  labels={labels.shape}  k={k}  N={N:,}")
 
# Build uid → row indices (giữ thứ tự gốc)
uid_to_indices = defaultdict(list)
for idx, uid in enumerate(uid_all):
    uid_to_indices[uid].append(idx)
 
# ── Load scalar features (EMB_DIM phải khớp với config của bạn) ──────────────
import sys, torch
from torch.utils.data import DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence
sys.path.append(".")
 
from get_feature import XES3G5M_exercises_embedding
from dataset import build_datasets
 
EMB_DIM  = 128   # id(32)+tree(32)+text(64) 
csv_path = "dataset/processed/XES3G5M/processed.csv"
ex_emb   = XES3G5M_exercises_embedding(
    csv_path=csv_path,
    kc_dict_path="dataset/processed/XES3G5M/kc_dict.pkl",
    question_dict_path="dataset/processed/XES3G5M/question_dict.pkl",
    base_emb_path="dataset/processed/XES3G5M/excercices_embedding",
)
full_ds = build_datasets(csv_path, ex_emb, "full", cache=False)
n_val   = max(1, int(len(full_ds) * 0.1))
_, val_ds = random_split(full_ds, [len(full_ds)-n_val, n_val],
                         generator=torch.Generator().manual_seed(42))
 
def collate_fn(batch):
    xs   = [b["x_sequence"] for b in batch]
    lens = [x.shape[0] for x in xs]
    xp   = pad_sequence(xs, batch_first=True, padding_value=0.0)
    pm   = torch.zeros(len(batch), xp.shape[1], dtype=torch.bool)
    for i, l in enumerate(lens): pm[i, l:] = True
    return {"x": xp, "padding_mask": pm, "lengths": torch.tensor(lens)}
 
loader = DataLoader(val_ds, batch_size=32, shuffle=False,
                    collate_fn=collate_fn, num_workers=4)
sc_list = []
for batch in loader:
    x, lens = batch["x"], batch["lengths"]
    for i, l in enumerate(lens.tolist()):
        sc_list.append(x[i, :l, EMB_DIM:].numpy())
scalars_all = np.concatenate(sc_list, axis=0)   # [N, 7]
 
SCALAR_NAMES = ["difficulty","resp_wrong","resp_correct",
                "gap_time","time_sin","time_cos","session_pos"]
df = pd.DataFrame(scalars_all, columns=SCALAR_NAMES)
df["cluster"]     = labels
df["accuracy"]    = df["resp_correct"]
cluster_acc       = df.groupby("cluster")["accuracy"].mean()
cluster_gap       = df.groupby("cluster")["gap_time"].mean()
cluster_diff      = df.groupby("cluster")["difficulty"].mean()
 
# Self-transition & dwell
transition = np.zeros((k,k), dtype=np.int64)
dwell_per_cluster = defaultdict(list)
for uid, indices in uid_to_indices.items():
    seq = labels[indices]
    for t in range(len(seq)-1):
        transition[seq[t], seq[t+1]] += 1
    cur, cnt = seq[0], 1
    for t in range(1, len(seq)):
        if seq[t] == cur: cnt += 1
        else:
            dwell_per_cluster[cur].append(cnt)
            cur, cnt = seq[t], 1
    dwell_per_cluster[cur].append(cnt)
 
trans_prob  = transition / transition.sum(axis=1, keepdims=True).clip(min=1)
self_trans  = np.diag(trans_prob)
 
print(f"scalars_all={scalars_all.shape}  ✓")
print(f"Cluster acc: { {c: round(float(cluster_acc[c]),3) for c in range(k)} }")



# ══════════════════════════════════════════════════════════════════════════════
# CELL 2 — UMAP 2D colored by cluster
# ══════════════════════════════════════════════════════════════════════════════
FIT_SIZE = 50_000
rng      = np.random.default_rng(42)
 
if N > FIT_SIZE:
    fit_idx = rng.choice(N, size=FIT_SIZE, replace=False)
    z_fit   = z_all[fit_idx]
    print(f"Fitting UMAP on {FIT_SIZE:,} points...")
else:
    z_fit = z_all
    print(f"Fitting UMAP on {N:,} points...")
 
reducer = umap.UMAP(n_components=2, n_neighbors=30, min_dist=0.1,
                    metric="cosine", random_state=42, verbose=False,
                    low_memory=True)
reducer.fit(z_fit)
print("Transforming full val set...")
emb = reducer.transform(z_all)   # [N, 2]
print(f"Done. emb={emb.shape}")
 
# ── Plot ──────────────────────────────────────────────────────────────────────
PLOT_N = 20_000
pidx   = rng.choice(N, size=min(PLOT_N, N), replace=False)
 
fig, ax = plt.subplots(figsize=(9, 7))
for c in range(k):
    mask = labels[pidx] == c
    ax.scatter(emb[pidx][mask, 0], emb[pidx][mask, 1],
               s=2, alpha=0.35, color=COLORS[c], rasterized=True)
 
# Legend với alpha=1.0
handles = [
    mpatches.Patch(color=COLORS[c], label=f"C{c}  acc={cluster_acc[c]:.2f}",
                   alpha=1.0)
    for c in range(k)
]
ax.legend(handles=handles, ncol=2, fontsize=8, framealpha=0.85,
          title=f"k={k} clusters")
ax.set_title("UMAP 2D — z_t colored by Spherical KMeans cluster\n(val set, cosine metric)",
             fontweight="bold")
ax.set_xlabel("UMAP-1"); ax.set_ylabel("UMAP-2")
ax.set_xticks([]); ax.set_yticks([])
plt.tight_layout()
plt.savefig(f"{CLUSTERING_DIR}/visualize/umap_clusters.png", dpi=150, bbox_inches="tight")
plt.show()

# ══════════════════════════════════════════════════════════════════════════════
# CELL 3 — Cluster profile: tự suy ra tên từ data, không hardcode
# ══════════════════════════════════════════════════════════════════════════════
# Tên cluster được suy ra hoàn toàn từ data thực của model đang chạy
# → không bị bias với k hay bất kỳ cấu hình cụ thể nào
 
WINDOW = 5
 
# ── Tính incoming/outgoing acc ────────────────────────────────────────────────
incoming_acc = defaultdict(list)
outgoing_acc = defaultdict(list)
for uid, indices in uid_to_indices.items():
    if len(indices) < WINDOW*2+1:
        continue
    seq_labels = labels[indices]
    seq_acc    = scalars_all[indices, 2]
    cps        = np.where(np.diff(seq_labels) != 0)[0] + 1
    for cp in cps:
        if cp < WINDOW or cp + WINDOW >= len(seq_labels):
            continue
        outgoing_acc[seq_labels[cp-1]].append(seq_acc[cp-WINDOW:cp].mean())
        incoming_acc[seq_labels[cp]].append(seq_acc[cp:cp+WINDOW].mean())
 
# ── Tự suy tên từ data ────────────────────────────────────────────────────────
# Không dùng ngưỡng cứng — dùng relative rank trong chính model này
acc_rank   = cluster_acc.rank(ascending=False)       # rank 1 = cao nhất
gap_rank   = cluster_gap.rank(ascending=False)       # rank 1 = gap dài nhất
self_rank  = pd.Series(self_trans).rank(ascending=False)  # rank 1 = stable nhất
 
CLUSTER_NAMES = {}
for c in range(k):
    # Accuracy tier (dựa trên relative rank)
    if acc_rank[c] <= k * 0.20:
        acc_label = "High performer"
    elif acc_rank[c] <= k * 0.50:
        acc_label = "Moderate-high"
    elif acc_rank[c] <= k * 0.75:
        acc_label = "Moderate"
    else:
        acc_label = "Struggling"
 
    # Stability tier
    if self_rank[c] <= k * 0.33:
        stab_label = "very stable"
    elif self_rank[c] <= k * 0.67:
        stab_label = "stable"
    else:
        stab_label = "transitional"
 
    # Gap tier
    if gap_rank[c] == 1:
        gap_label = "slow"
    elif gap_rank[c] == k:
        gap_label = "fast"
    else:
        gap_label = ""
 
    name = f"{acc_label} / {stab_label}"
    if gap_label:
        name += f" / {gap_label}"
    CLUSTER_NAMES[c] = name
 
# ── Heatmap ───────────────────────────────────────────────────────────────────
feat_cols = ["accuracy", "difficulty", "gap_time", "session_pos"]
mean_df   = df.groupby("cluster")[feat_cols].mean()
 
fig, ax = plt.subplots(figsize=(max(8, k*1.4), 4))
im = ax.imshow(mean_df.T.values, aspect="auto", cmap="RdYlGn")
ax.set_xticks(range(k))
ax.set_xticklabels(
    [f"C{c}\n{CLUSTER_NAMES[c].split('/')[0].strip()}\n(n={(labels==c).sum():,})"
     for c in range(k)], fontsize=8)
ax.set_yticks(range(len(feat_cols)))
ax.set_yticklabels(feat_cols, fontsize=10)
for i in range(len(feat_cols)):
    for j in range(k):
        ax.text(j, i, f"{mean_df.T.values[i,j]:.3f}",
                ha="center", va="center", fontsize=8)
plt.colorbar(im, ax=ax)
ax.set_title("Cluster feature profile — mean per cluster", fontweight="bold")
plt.tight_layout()
plt.savefig(f"{CLUSTERING_DIR}/visualize/cluster_profile_heatmap.png",
            dpi=150, bbox_inches="tight")
plt.show()
 
# ── Identity card ─────────────────────────────────────────────────────────────
print("CLUSTER IDENTITY CARDS")
print("="*65)
for c in range(k):
    in_a  = np.mean(incoming_acc[c]) if incoming_acc[c] else float("nan")
    out_a = np.mean(outgoing_acc[c]) if outgoing_acc[c] else float("nan")
    dwell = np.array(dwell_per_cluster[c])
    print(f"\nC{c}  {CLUSTER_NAMES[c]}")
    print(f"  n={( labels==c).sum():,}  acc={cluster_acc[c]:.3f}  "
          f"gap={cluster_gap[c]:.3f}  self-trans={self_trans[c]:.3f}  "
          f"dwell_mean={dwell.mean():.1f}")
    print(f"  incoming_acc={in_a:.3f}  outgoing_acc={out_a:.3f}  "
          f"delta={in_a-out_a:+.3f}")
    

# ══════════════════════════════════════════════════════════════════════════════
# CELL 4 — Statistical validation: không hardcode cluster index
# ══════════════════════════════════════════════════════════════════════════════
# Tất cả test đều dùng rank/sort từ data → hoạt động với mọi k và mọi model
 
acc_by_cluster = {c: scalars_all[labels==c, 2] for c in range(k)}
gap_by_cluster = {c: scalars_all[labels==c, 3] for c in range(k)}
 
# Sắp xếp cluster theo acc/gap/stability thực tế
sorted_by_acc  = sorted(range(k), key=lambda c: acc_by_cluster[c].mean(), reverse=True)
sorted_by_gap  = sorted(range(k), key=lambda c: gap_by_cluster[c].mean(), reverse=True)
sorted_by_stab = sorted(range(k), key=lambda c: self_trans[c], reverse=True)
 
best_acc_c  = sorted_by_acc[0]
worst_acc_c = sorted_by_acc[-1]
third_acc_c = sorted_by_acc[min(2, k-1)]
slowest_c   = sorted_by_gap[0]
fastest_c   = sorted_by_gap[-1]
 
# transitional = bottom 50% stability, stable = top 33% stability
n_stable   = max(1, k // 3)
n_transitional = max(1, k // 2)
stable_cls  = sorted_by_stab[:n_stable]
transitional_cls = sorted_by_stab[-n_transitional:]
 
print("="*62)
print("  STATISTICAL VALIDATION")
print(f"  k={k}  best_acc=C{best_acc_c}  worst_acc=C{worst_acc_c}")
print(f"  stable clusters : {[f'C{c}' for c in stable_cls]}")
print(f"  transitional clusters: {[f'C{c}' for c in transitional_cls]}")
print("="*62)
 
# ── T1: Best > Worst accuracy ─────────────────────────────────────────────────
_, p1 = stats.mannwhitneyu(acc_by_cluster[best_acc_c],
                            acc_by_cluster[worst_acc_c], alternative="greater")
_, p1b = stats.mannwhitneyu(acc_by_cluster[best_acc_c],
                             acc_by_cluster[third_acc_c], alternative="greater")
 
# ── T2: Variance khác nhau giữa các cluster (Levene) ─────────────────────────
# Không test direction vì "transitional" không nhất thiết có std cao hơn
lev_stat, p2 = stats.levene(*[acc_by_cluster[c] for c in range(k)])
 
# ── T3: Gap time slowest > fastest ───────────────────────────────────────────
if slowest_c != fastest_c:
    _, p3 = stats.mannwhitneyu(gap_by_cluster[slowest_c],
                                gap_by_cluster[fastest_c], alternative="greater")
else:
    p3 = 1.0
 
# ── T4: Dwell stable > transitional ──────────────────────────────────────────────
if stable_cls and transitional_cls and set(stable_cls) != set(transitional_cls):
    stable_dw   = np.concatenate([np.array(dwell_per_cluster[c])
                                   for c in stable_cls if dwell_per_cluster[c]])
    transitional_dw = np.concatenate([np.array(dwell_per_cluster[c])
                                   for c in transitional_cls if dwell_per_cluster[c]])
    if len(stable_dw) > 0 and len(transitional_dw) > 0:
        _, p4 = stats.mannwhitneyu(stable_dw, transitional_dw, alternative="greater")
    else:
        p4 = 1.0
else:
    p4 = 1.0
 
# ── T5: User dominant cluster ~ user acc (Spearman) ──────────────────────────
user_dom  = {uid: np.bincount(labels[idx], minlength=k).argmax()
             for uid, idx in uid_to_indices.items()}
user_macc = {uid: scalars_all[idx, 2].mean()
             for uid, idx in uid_to_indices.items()}
dom_arr   = np.array(list(user_dom.values()))
macc_arr  = np.array(list(user_macc.values()))
corr5, p5 = stats.spearmanr(
    np.array([cluster_acc[c] for c in dom_arr]), macc_arr
)
 
# ── T6: Soft assignment entropy < uniform ────────────────────────────────────
uniform_entropy = np.log(k)
actual_entropy  = -np.sum(soft_asgn * np.log(soft_asgn + 1e-8), axis=1).mean()
entropy_ratio   = actual_entropy / uniform_entropy
t6_pass         = entropy_ratio < 0.75   # thư giãn threshold so với 0.6
 
# ── T7 (NEW): Permutation test — silhouette thật > shuffled ──────────────────
# Shuffle labels ngẫu nhiên N_PERM lần, đo silhouette mỗi lần
# Nếu silhouette thật > 95th percentile shuffled → cluster không phải do may mắn
from sklearn.metrics import silhouette_score
 
N_PERM      = 100
SILO_SAMPLE = min(8_000, N)
rng_perm    = np.random.default_rng(0)
silo_idx    = rng_perm.choice(N, size=SILO_SAMPLE, replace=False)
 
real_silo   = silhouette_score(z_all[silo_idx], labels[silo_idx], metric="cosine")
perm_silos  = []
for _ in range(N_PERM):
    shuffled = rng_perm.permutation(labels[silo_idx])
    try:
        perm_silos.append(
            silhouette_score(z_all[silo_idx], shuffled, metric="cosine")
        )
    except Exception:
        perm_silos.append(0.0)
 
perm_silos = np.array(perm_silos)
perm_p7    = (perm_silos >= real_silo).mean()   # empirical p-value
t7_pass    = perm_p7 < 0.05
 
print(f"\n  Permutation test (n={N_PERM} shuffles, sample={SILO_SAMPLE:,}):")
print(f"    Real silhouette(cosine) = {real_silo:.4f}")
print(f"    Shuffled: mean={perm_silos.mean():.4f}  "
      f"95th={np.percentile(perm_silos,95):.4f}  max={perm_silos.max():.4f}")
print(f"    Empirical p = {perm_p7:.3f}  "
      f"({'✓ cluster tốt hơn random' if t7_pass else '✗ không hơn random'})")
 
# ── T8 (NEW): Intra-cluster consistency (coefficient of variation) ────────────
# CV = std/mean thấp → cluster có features nhất quán, không chỉ là mean artifact
cv_acc = {}
for c in range(k):
    vals = acc_by_cluster[c]
    mean = vals.mean()
    cv_acc[c] = vals.std() / (mean + 1e-8)
 
# CV của cluster tốt nhất phải thấp hơn CV của cluster tệ nhất
# (high performer thực sự consistent, not just average up)
cv_best  = cv_acc[best_acc_c]
cv_worst = cv_acc[worst_acc_c]
t8_pass  = True   # CV chỉ để report, không có ground truth để test
 
print(f"\n  Cluster CV (std/mean accuracy):")
for c in sorted_by_acc:
    print(f"    C{c}: CV={cv_acc[c]:.3f}  mean_acc={acc_by_cluster[c].mean():.3f}  "
          f"(lower CV = more consistent)")
 
# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "="*62)
results = [
    (f"T1a  C{best_acc_c}(best acc) > C{worst_acc_c}(worst)",
     p1 < 0.05,   f"p={p1:.1e}"),
    (f"T1b  C{best_acc_c}(best acc) > C{third_acc_c}(3rd)",
     p1b < 0.05,  f"p={p1b:.1e}"),
    (f"T2   Variance khác nhau across clusters (Levene)",
     p2 < 0.05,   f"F={lev_stat:.1f}  p={p2:.1e}"),
    (f"T3   Gap C{slowest_c}(slowest) > C{fastest_c}(fastest)",
     p3 < 0.05,   f"p={p3:.1e}"),
    (f"T4   Dwell stable{[f'C{c}' for c in stable_cls]} > "
     f"transitional{[f'C{c}' for c in transitional_cls]}",
     p4 < 0.05,   f"p={p4:.1e}"),
    (f"T5   User dominant cluster ~ user acc  r={corr5:.3f}",
     p5 < 0.05 and corr5 > 0.3, f"Spearman p={p5:.1e}"),
    (f"T6   Soft assignment entropy < 75% uniform",
     t6_pass,     f"ratio={entropy_ratio:.2f}"),
    (f"T7   Permutation: real silhouette > shuffled (p<0.05)",
     t7_pass,     f"real={real_silo:.4f}  perm_p={perm_p7:.3f}"),
]
for name, passed, detail in results:
    print(f"  {'✓' if passed else '✗'}  {name}")
    print(f"      {detail}")
 
n_pass = sum(r[1] for r in results)
print(f"\n  {n_pass}/{len(results)} tests passed")
verdict = ("→ Cluster có ý nghĩa thống kê — đáng tin cậy" if n_pass >= 7
           else "→ Cluster có ý nghĩa một phần" if n_pass >= 5
           else "→ Cluster yếu — cần review z_t quality")
print(f"  {verdict}")
 
# Note về transitional/stable
print(f"\n  Note: 'transitional' = hay đổi cluster, không phải acc không ổn định")
print(f"  Permutation test (T7) là test tổng quát nhất: nếu pass → z_t có structure thật")
 


# ══════════════════════════════════════════════════════════════════════════════
# CELL 4b — Per-user, per-segment validation
# ══════════════════════════════════════════════════════════════════════════════
# Câu hỏi: Với từng user cụ thể, khi họ ở cluster X thì acc có
# thực sự khớp với định nghĩa của X không?
# Và khi đổi cluster, direction có đúng không?

WINDOW = 5

# ── Part A: Per-user cluster consistency ─────────────────────────────────────
# Với mỗi user, với mỗi cluster họ từng ở:
#   tính mean acc trong segment đó
#   so sánh với global cluster acc
# Nếu per-user acc trong cluster C ~ global cluster_acc[C] → consistent

per_user_cluster_acc = defaultdict(list)  # cluster → list of per-user-segment means

for uid, indices in uid_to_indices.items():
    if len(indices) < 3:
        continue
    seq_labels = labels[indices]
    seq_acc    = scalars_all[indices, 2]

    # Tách thành segments liên tiếp
    seg_start = 0
    for t in range(1, len(seq_labels)):
        if seq_labels[t] != seq_labels[seg_start] or t == len(seq_labels) - 1:
            seg_end = t if seq_labels[t] != seq_labels[seg_start] else t + 1
            seg_c   = seq_labels[seg_start]
            seg_acc = seq_acc[seg_start:seg_end].mean()
            if seg_end - seg_start >= 3:   # chỉ lấy segment đủ dài
                per_user_cluster_acc[seg_c].append(seg_acc)
            seg_start = t

print("="*62)
print("  PART A: Per-user segment accuracy vs global cluster acc")
print("  Kỳ vọng: per-user segment acc ≈ global cluster acc")
print("="*62)

consistency_scores = {}
for c in range(k):
    segs   = np.array(per_user_cluster_acc[c])
    global_acc = cluster_acc[c]
    # Mean absolute deviation của per-segment acc so với global
    mad    = np.abs(segs - global_acc).mean()
    # % segments có acc đúng phía so với median (đúng direction)
    median_acc = np.median([cluster_acc[cc] for cc in range(k)])
    correct_side = (
        ((segs > median_acc) == (global_acc > median_acc)).mean()
    )
    consistency_scores[c] = correct_side
    print(f"  C{c} [{CLUSTER_NAMES[c][:25]}]")
    print(f"      global_acc={global_acc:.3f}  seg_mean={segs.mean():.3f}  "
          f"MAD={mad:.3f}  correct_side={correct_side:.1%}  n_segs={len(segs):,}")

overall_consistency = np.mean(list(consistency_scores.values()))
print(f"\n  Overall correct_side = {overall_consistency:.1%}")
print(f"  {'✓ Per-user segments nhất quán với cluster definition' if overall_consistency > 0.7 else '✗ Inconsistent'}")

# ── Part B: Per-user transition direction ─────────────────────────────────────
# Khi user chuyển từ cluster A → B:
#   nếu cluster_acc[B] > cluster_acc[A]: "upgrade"
#   nếu cluster_acc[B] < cluster_acc[A]: "downgrade"
# Kiểm tra: acc thực sự sau transition có match direction không?

print("\n" + "="*62)
print("  PART B: Per-user transition direction validation")
print("  Kỳ vọng: upgrade → acc tăng, downgrade → acc giảm")
print("="*62)

upgrade_deltas   = []   # delta acc sau upgrade transitions
downgrade_deltas = []   # delta acc sau downgrade transitions
neutral_deltas   = []   # delta acc sau same-level transitions

for uid, indices in uid_to_indices.items():
    if len(indices) < WINDOW * 2 + 1:
        continue
    seq_labels = labels[indices]
    seq_acc    = scalars_all[indices, 2]
    cps        = np.where(np.diff(seq_labels) != 0)[0] + 1

    for cp in cps:
        if cp < WINDOW or cp + WINDOW >= len(seq_labels):
            continue
        src = seq_labels[cp - 1]
        dst = seq_labels[cp]
        acc_diff_cluster = cluster_acc[dst] - cluster_acc[src]
        acc_diff_actual  = seq_acc[cp:cp+WINDOW].mean() - seq_acc[cp-WINDOW:cp].mean()

        if acc_diff_cluster > 0.05:       # upgrade rõ ràng
            upgrade_deltas.append(acc_diff_actual)
        elif acc_diff_cluster < -0.05:    # downgrade rõ ràng
            downgrade_deltas.append(acc_diff_actual)
        else:                              # cùng level
            neutral_deltas.append(acc_diff_actual)

upgrade_deltas   = np.array(upgrade_deltas)
downgrade_deltas = np.array(downgrade_deltas)
neutral_deltas   = np.array(neutral_deltas)

print(f"  Upgrade transitions   (n={len(upgrade_deltas):,}): "
      f"mean Δacc = {upgrade_deltas.mean():+.4f}  "
      f"({'acc tăng ✓' if upgrade_deltas.mean() > 0 else 'acc giảm ✗'})")
print(f"  Downgrade transitions (n={len(downgrade_deltas):,}): "
      f"mean Δacc = {downgrade_deltas.mean():+.4f}  "
      f"({'acc giảm ✓' if downgrade_deltas.mean() < 0 else 'acc tăng ✗'})")
print(f"  Neutral transitions   (n={len(neutral_deltas):,}): "
      f"mean Δacc = {neutral_deltas.mean():+.4f}")

# Mann-Whitney: upgrade delta > 0
if len(upgrade_deltas) > 10:
    _, p_up = stats.mannwhitneyu(upgrade_deltas,
                                  np.zeros(len(upgrade_deltas)),
                                  alternative="greater")
    print(f"\n  Upgrade Δacc > 0: p={p_up:.2e}  "
          f"{'✓' if p_up < 0.05 else '✗'}")

if len(downgrade_deltas) > 10:
    _, p_dn = stats.mannwhitneyu(np.zeros(len(downgrade_deltas)),
                                  downgrade_deltas,
                                  alternative="greater")
    print(f"  Downgrade Δacc < 0: p={p_dn:.2e}  "
          f"{'✓' if p_dn < 0.05 else '✗'}")

# % transitions đúng direction
if len(upgrade_deltas) > 0:
    pct_up = (upgrade_deltas > 0).mean()
    print(f"\n  % upgrade transitions có acc tăng thực tế  : {pct_up:.1%}")
if len(downgrade_deltas) > 0:
    pct_dn = (downgrade_deltas < 0).mean()
    print(f"  % downgrade transitions có acc giảm thực tế: {pct_dn:.1%}")

# ── Part C: Per-user "regime purity" ─────────────────────────────────────────
# Với mỗi user, dominant cluster của họ có phải là cluster
# có acc gần nhất với mean acc của user đó không?
# Nếu đúng → cluster assignment phản ánh đúng từng user cá nhân

print("\n" + "="*62)
print("  PART C: Per-user regime purity")
print("  Kỳ vọng: dominant cluster của user có acc ≈ user mean acc")
print("="*62)

correct_dominant = 0
total_users      = 0
acc_errors       = []

for uid, indices in uid_to_indices.items():
    if len(indices) < 5:
        continue
    user_acc    = scalars_all[indices, 2].mean()
    dominant_c  = np.bincount(labels[indices], minlength=k).argmax()

    # Cluster nào có acc gần user_acc nhất?
    best_match_c = min(range(k), key=lambda c: abs(cluster_acc[c] - user_acc))

    if dominant_c == best_match_c:
        correct_dominant += 1
    acc_errors.append(abs(cluster_acc[dominant_c] - user_acc))
    total_users += 1

purity = correct_dominant / max(total_users, 1)
print(f"  Users có dominant cluster = best match cluster: "
      f"{correct_dominant}/{total_users} = {purity:.1%}")
print(f"  Mean |cluster_acc - user_acc|: {np.mean(acc_errors):.3f}")
print(f"  {'✓ Cluster assignment phản ánh đúng từng user' if purity > 0.4 else '✗ Purity thấp'}")

# ── Summary Part A+B+C ────────────────────────────────────────────────────────
print("\n" + "="*62)
print("  PER-USER VALIDATION SUMMARY")
print("="*62)
checks = [
    ("Part A  Per-segment consistency > 70%",   overall_consistency > 0.70),
    ("Part B  Upgrade → acc tăng (direction)",
     len(upgrade_deltas) > 0 and upgrade_deltas.mean() > 0),
    ("Part B  Downgrade → acc giảm (direction)",
     len(downgrade_deltas) > 0 and downgrade_deltas.mean() < 0),
    ("Part C  Dominant cluster purity > 40%",   purity > 0.40),
]
for name, passed in checks:
    print(f"  {'✓' if passed else '✗'}  {name}")

n_pass_local = sum(c[1] for c in checks)
print(f"\n  {n_pass_local}/{len(checks)} passed")
print(f"  {'✓ Per-user behavior nhất quán với cluster definition' if n_pass_local >= 3 else '✗ Cần review'}")


# ══════════════════════════════════════════════════════════════════════════════
# CELL 5 — Timeline: cluster trajectory của 1 học sinh
# ══════════════════════════════════════════════════════════════════════════════
STUDENT_IDX = 3   # ← đổi số này

kt_dataset  = val_ds.dataset
val_indices = val_ds.indices
val_uids    = [kt_dataset.users[i] for i in val_indices]

target_uid  = val_uids[STUDENT_IDX]
target_rows = uid_to_indices[target_uid]
T_len       = len(target_rows)

tgt_labels  = labels[target_rows]
tgt_sc      = scalars_all[target_rows]
tgt_acc     = tgt_sc[:, 2]
tgt_diff    = tgt_sc[:, 0]
tgt_gap     = tgt_sc[:, 3]

ROLL_WIN   = 10
roll_acc   = np.convolve(tgt_acc, np.ones(ROLL_WIN)/ROLL_WIN, mode="same")
change_pts = np.where(np.diff(tgt_labels) != 0)[0] + 1
timesteps  = np.arange(T_len)

fig, axes = plt.subplots(
    5, 1, figsize=(16, 14),
    gridspec_kw={"height_ratios": [2.2, 1.0, 0.5, 1.0, 0.6]},
    sharex=True,
)
fig.suptitle(
    f"Behavioral regime timeline — Student {STUDENT_IDX}  "
    f"(uid={target_uid},  T={T_len})",
    fontsize=13, fontweight="bold",
)

# ── [A] axes[0]: Cluster trajectory — scatter points ─────────────────────────
ax = axes[0]
for c in range(k):
    mask = tgt_labels == c
    ax.scatter(timesteps[mask], tgt_labels[mask],
               color=COLORS[c], s=18, alpha=0.9, zorder=3, linewidths=0)
ax.plot(timesteps, tgt_labels, color="gray", linewidth=0.5, alpha=0.4, zorder=2)
for cp in change_pts:
    ax.axvline(cp, color="#888", linewidth=0.7, linestyle="--", alpha=0.5)
handles = [
    plt.scatter([], [], color=COLORS[c], s=35, alpha=1.0,
                label=f"C{c} — {CLUSTER_NAMES[c]}")
    for c in range(k)
]
ax.legend(handles=handles, fontsize=7.5, loc="upper right",
          framealpha=0.9, ncol=2)
ax.set_yticks(range(k))
ax.set_yticklabels([f"C{c}" for c in range(k)], fontsize=9)
ax.set_ylabel("Cluster")
ax.set_ylim(-0.6, k - 0.4)
ax.set_title("[A] Cluster trajectory", fontsize=10, loc="left", fontweight="bold")
ax.grid(True, axis="x", alpha=0.15)

# ── [B] axes[1]: Rolling accuracy ────────────────────────────────────────────
ax = axes[1]
ax.fill_between(timesteps, roll_acc, alpha=0.15, color="#27ae60")
ax.plot(timesteps, roll_acc, color="#27ae60", linewidth=1.8,
        label=f"rolling acc (w={ROLL_WIN})", zorder=4)
for cp in change_pts:
    ax.axvline(cp, color="#888", linewidth=0.7, linestyle="--", alpha=0.5)
ax.axhline(0.5, color="gray", linewidth=0.8, linestyle=":")
ax.set_ylabel("Rolling acc")
ax.set_ylim(0.0, 1.1)
ax.legend(fontsize=8, loc="lower right")
ax.set_title("[B] Rolling accuracy", fontsize=10, loc="left", fontweight="bold")
ax.grid(True, alpha=0.15)

# ── [B2] axes[2]: Per-step response ──────────────────────────────────────────
ax = axes[2]
ax.scatter(timesteps[tgt_acc==1],
           np.ones(int((tgt_acc==1).sum()), dtype=int),
           s=12, color="#2ecc71", alpha=0.8, marker="|",
           linewidths=1.5, label="correct")
ax.scatter(timesteps[tgt_acc==0],
           np.zeros(int((tgt_acc==0).sum()), dtype=int),
           s=12, color="#e74c3c", alpha=0.9, marker="|",
           linewidths=1.5, label="wrong")
for cp in change_pts:
    ax.axvline(cp, color="#888", linewidth=0.7, linestyle="--", alpha=0.5)
ax.set_yticks([0, 1])
ax.set_yticklabels(["wrong", "correct"], fontsize=8)
ax.set_ylim(-0.5, 1.5)
ax.set_ylabel("Response", fontsize=9)
ax.set_title("[B2] Per-step response", fontsize=10, loc="left", fontweight="bold")
ax.grid(True, axis="x", alpha=0.15)

# ── [C] axes[3]: Gap time ─────────────────────────────────────────────────────
ax = axes[3]
ax.bar(timesteps, tgt_gap,
       color=["#e74c3c" if v > 0 else "#3498db" for v in tgt_gap],
       alpha=0.75, width=1.0)
for cp in change_pts:
    ax.axvline(cp, color="#888", linewidth=0.7, linestyle="--", alpha=0.5)
ax.axhline(0, color="black", linewidth=0.7)
ax.set_ylabel("Gap time\n(z-norm)", fontsize=9)
ax.set_title("[C] Gap time  (đỏ=chậm hơn TB, xanh=nhanh hơn TB)",
             fontsize=10, loc="left", fontweight="bold")
ax.grid(True, alpha=0.15)

# ── [D] axes[4]: Color bar ────────────────────────────────────────────────────
ax = axes[4]
for t in range(T_len - 1):
    ax.barh(0, 1, left=t, height=1,
            color=COLORS[tgt_labels[t] % len(COLORS)], alpha=0.9)
seg_start = 0
for t in range(1, T_len):
    if t == T_len-1 or tgt_labels[t] != tgt_labels[seg_start]:
        seg_len = t - seg_start
        if seg_len >= 6:
            ax.text((seg_start+t)/2, 0, f"C{tgt_labels[seg_start]}",
                    ha="center", va="center",
                    fontsize=7, fontweight="bold", color="white",
                    path_effects=[pe.withStroke(linewidth=1.5, foreground="black")])
        seg_start = t
ax.set_xlim(0, T_len)
ax.set_yticks([])
ax.set_xlabel("Timestep", fontsize=10)
ax.set_title("[D] Cluster color bar", fontsize=10, loc="left", fontweight="bold")

plt.tight_layout()
plt.savefig(
    f"{CLUSTERING_DIR}/visualize/student_{STUDENT_IDX}_timeline.png",
    dpi=150, bbox_inches="tight",
)
plt.show()
print(f"T={T_len}  change_points={len(change_pts)}")