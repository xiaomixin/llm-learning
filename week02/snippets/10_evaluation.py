"""Concept 10 — Evaluation metrics under extreme imbalance.

Run: python 10_evaluation.py

Goal:
1. Compare AUC-ROC vs AUC-PR on a 1:500-imbalanced toy score set.
2. Show Recall @ FPR = 0.001 (the business-style metric).
3. Pick a threshold by maximum F1 and print the confusion matrix.
"""
import numpy as np
from sklearn.metrics import (average_precision_score, roc_auc_score,
                             precision_recall_curve, roc_curve,
                             confusion_matrix)


def toy_scores(n_neg=5000, n_pos=10, seed=0):
    """Mediocre classifier: positives scored higher on average, but
    some positives are buried among negatives."""
    rng = np.random.default_rng(seed)
    neg_scores = rng.beta(2, 20, size=n_neg)          # most near 0
    pos_scores = rng.beta(5, 2, size=n_pos) * 0.8 + 0.1  # higher, but noisy
    scores = np.concatenate([neg_scores, pos_scores])
    labels = np.concatenate([np.zeros(n_neg), np.ones(n_pos)])
    return labels, scores


def recall_at_fpr(y_true, scores, target_fpr=0.001):
    fpr, tpr, _ = roc_curve(y_true, scores)
    idx = np.searchsorted(fpr, target_fpr)
    idx = min(idx, len(tpr) - 1)
    return tpr[idx], fpr[idx]


def pick_best_f1(y_true, scores):
    prec, rec, thr = precision_recall_curve(y_true, scores)
    f1 = 2 * prec * rec / (prec + rec + 1e-9)
    # f1 is len(thr)+1 — last entry corresponds to threshold = inf (p=1, r=0).
    best = np.nanargmax(f1[:-1])
    return thr[best], f1[best], prec[best], rec[best]


def main():
    y, s = toy_scores()
    print(f"samples: {len(y)}  positives: {int(y.sum())}  "
          f"ratio: {y.mean():.5f}\n")

    ap = average_precision_score(y, s)
    roc = roc_auc_score(y, s)
    print(f"AUC-ROC = {roc:.4f}   <- looks great")
    print(f"AUC-PR  = {ap:.4f}    <- closer to the truth under imbalance")

    tpr_at, got_fpr = recall_at_fpr(y, s, 0.001)
    print(f"\nRecall @ FPR = 0.001 : {tpr_at:.3f}  (actual FPR at that bucket: {got_fpr:.5f})")
    print("  business reading: 'if we tolerate flagging 0.1% of normal txns, "
          "we catch {:.0%} of fraud.'".format(tpr_at))

    thr, f1, p, r = pick_best_f1(y, s)
    print(f"\nBest-F1 threshold = {thr:.4f}")
    print(f"  precision = {p:.3f}   recall = {r:.3f}   F1 = {f1:.3f}")
    y_pred = (s >= thr).astype(int)
    cm = confusion_matrix(y, y_pred)
    print("  confusion matrix [[TN, FP],[FN, TP]]:\n ", cm)

    print("\nTakeaway: AUC-ROC and AUC-PR can tell different stories.\n"
          "In imbalanced detection, report BOTH and rely on AUC-PR.")


if __name__ == "__main__":
    main()
