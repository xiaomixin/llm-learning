"""
Reproducible ROC vs PR figure for week01/knowledge.md.

Scenario: 10,000 samples, 17 positives (~0.17% — matches Kaggle creditcardfraud).
Simulate scores from a realistic classifier: positives come from N(2, 1),
negatives from N(0, 1). Then plot ROC and PR side by side and annotate
the same operating point (threshold) on both — so you can literally SEE
why "ROC looks great, PR tells the truth".

Run: python generate_roc_pr_chart.py  →  writes roc_vs_pr.png in the same dir.
"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score, average_precision_score

rng = np.random.default_rng(42)
N_NEG, N_POS = 9983, 17  # 0.17% positives, same order as Kaggle fraud
y = np.concatenate([np.zeros(N_NEG), np.ones(N_POS)])
scores_neg = rng.normal(loc=0.0, scale=1.0, size=N_NEG)
scores_pos = rng.normal(loc=2.0, scale=1.0, size=N_POS)   # clearly separable classifier
scores = np.concatenate([scores_neg, scores_pos])

fpr, tpr, thr_roc = roc_curve(y, scores)
prec, rec, thr_pr = precision_recall_curve(y, scores)
auc_roc = roc_auc_score(y, scores)
auc_pr = average_precision_score(y, scores)

# Pick the operating point with FPR ≈ 0.01 (the Recall@FPR=0.01 threshold used
# in fraud detection) and locate the SAME threshold on the PR curve.
idx_roc = np.argmin(np.abs(fpr - 0.01))
target_thr = thr_roc[idx_roc]
tpr_op, fpr_op = tpr[idx_roc], fpr[idx_roc]
idx_pr = np.argmin(np.abs(thr_pr - target_thr))
prec_op, rec_op = prec[idx_pr], rec[idx_pr]

pred = scores >= target_thr
TP = int(((pred == 1) & (y == 1)).sum())
FP = int(((pred == 1) & (y == 0)).sum())
FN = int(((pred == 0) & (y == 1)).sum())
TN = int(((pred == 0) & (y == 0)).sum())

print(f"operating point @ threshold={target_thr:.3f}")
print(f"  TP={TP}, FP={FP}, FN={FN}, TN={TN}")
print(f"  TPR(Recall)={tpr_op:.3f}, FPR={fpr_op:.4f}, Precision={prec_op:.3f}")
print(f"AUC-ROC={auc_roc:.3f}, AUC-PR={auc_pr:.3f}")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# ── ROC ────────────────────────────────────────────────────────────
ax = axes[0]
ax.plot(fpr, tpr, color="#1f77b4", lw=2.4, label=f"Classifier  (AUC-ROC={auc_roc:.3f})")
ax.plot([0, 1], [0, 1], color="gray", lw=1.0, ls="--", label="Random  (AUC=0.500)")
ax.scatter([fpr_op], [tpr_op], s=160, color="#d62728", zorder=5, edgecolor="white", linewidth=1.5,
           label=f"Op. point @ thr={target_thr:.2f}")
ax.annotate(
    f"FPR = {fpr_op:.3f}\nTPR = {tpr_op:.3f}\n\nTP={TP}  FP={FP}\nFN={FN}  TN={TN}",
    xy=(fpr_op, tpr_op), xytext=(0.20, 0.35),
    fontsize=11, ha="left", family="monospace",
    arrowprops=dict(arrowstyle="->", color="#d62728", lw=1.3),
    bbox=dict(boxstyle="round,pad=0.5", fc="#fff4f4", ec="#d62728"))
ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
ax.set_xlabel("FPR  =  FP / (FP + TN)      [denominator = ALL 9983 negatives]", fontsize=11)
ax.set_ylabel("TPR  =  TP / (TP + FN)  =  Recall", fontsize=11)
ax.set_title("ROC curve — 'looks great'\n112 FP divided by 9983 negatives is only 1.1%.\nThe curve HUGS the top-left; AUC-ROC = 0.957.",
             fontsize=11.5, pad=12)
ax.legend(loc="lower right", fontsize=10)
ax.grid(alpha=0.3)

# ── PR ────────────────────────────────────────────────────────────
ax = axes[1]
ax.plot(rec, prec, color="#2ca02c", lw=2.4, label=f"Classifier  (AUC-PR={auc_pr:.3f})")
baseline = N_POS / (N_POS + N_NEG)
ax.axhline(y=baseline, color="gray", lw=1.0, ls="--",
           label=f"Random  (AP={baseline:.4f})")
ax.scatter([rec_op], [prec_op], s=160, color="#d62728", zorder=5, edgecolor="white", linewidth=1.5,
           label=f"Op. point @ thr={target_thr:.2f}  (same threshold)")
ax.annotate(
    f"Recall    = {rec_op:.3f}\nPrecision = {prec_op:.3f}\n\n"
    f"Alert {TP+FP} times,\nonly {TP} are real fraud",
    xy=(rec_op, prec_op), xytext=(0.20, 0.55),
    fontsize=11, ha="left", family="monospace",
    arrowprops=dict(arrowstyle="->", color="#d62728", lw=1.3),
    bbox=dict(boxstyle="round,pad=0.5", fc="#fff4f4", ec="#d62728"))
ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
ax.set_xlabel("Recall  =  TP / (TP + FN)", fontsize=11)
ax.set_ylabel("Precision  =  TP / (TP + FP)      [denominator = predicted-positive only]",
              fontsize=11)
ax.set_title("PR curve — 'tells the truth'\nPrecision denominator excludes the 9871 true negatives.\nThe curve COLLAPSES; AUC-PR = 0.140.",
             fontsize=11.5, pad=12)
ax.legend(loc="upper right", fontsize=10)
ax.grid(alpha=0.3)

fig.suptitle(
    f"Same classifier, same threshold, two metrics — N={N_NEG+N_POS}, positive rate = {N_POS/(N_POS+N_NEG)*100:.2f}% (seed=42)",
    fontsize=13, y=1.00)
fig.tight_layout()

out = Path(__file__).parent / "roc_vs_pr.png"
fig.savefig(out, dpi=140, bbox_inches="tight")
print(f"saved: {out}")
