"""Concept 08 — Stratified split + 'scaler only fit on train'.

Run: python 08_split_scaler.py
(needs creditcard.csv — see ../snippets/_data.py for path resolution)

Goal:
1. Show stratified split keeps the fraud ratio stable across train/val/test.
2. Show the difference in scaler statistics between WRONG (fit on all)
   and RIGHT (fit on train only) — this is data leakage in 1 number.
"""
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from _data import load_creditcard


def main():
    X, y, cols = load_creditcard()
    print(f"full set: n={len(X):,}  fraud ratio={y.mean():.5f}")

    # ── 1. Stratified 70/15/15 split ─────────────────────────────
    X_tv, X_test, y_tv, y_test = train_test_split(
        X, y, test_size=0.15, stratify=y, random_state=42)
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_tv, y_tv, test_size=0.1765,   # 0.15 / 0.85
        stratify=y_tv, random_state=42)

    print("\n[stratified] fraud ratio per split:")
    print(f"  train = {y_tr.mean():.5f}  (n={len(y_tr):,}, pos={int(y_tr.sum())})")
    print(f"  val   = {y_val.mean():.5f}  (n={len(y_val):,}, pos={int(y_val.sum())})")
    print(f"  test  = {y_test.mean():.5f}  (n={len(y_test):,}, pos={int(y_test.sum())})")

    # Compare with non-stratified — positive count will swing
    X_nt, _, y_nt, _ = train_test_split(X, y, test_size=0.15, random_state=42)
    _, X_nt_val, _, y_nt_val = train_test_split(X_nt, y_nt, test_size=0.1765, random_state=42)
    print(f"\n[no stratify] val fraud ratio = {y_nt_val.mean():.5f}  "
          f"pos={int(y_nt_val.sum())}  (stratified gave pos={int(y_val.sum())})")
    print("  -> random splits can swing the positive count noticeably.")

    # ── 2. Leakage demo: scaler fit on ALL vs fit on TRAIN ──────
    scaler_all = StandardScaler().fit(X)            # WRONG — leaks val/test stats
    scaler_tr = StandardScaler().fit(X_tr)          # RIGHT

    # Compare mean of each feature — first 3 columns
    print("\nScaler mean (first 3 features):")
    print(f"  fit on all  : {scaler_all.mean_[:3]}")
    print(f"  fit on train: {scaler_tr.mean_[:3]}")
    delta = np.abs(scaler_all.mean_ - scaler_tr.mean_)
    print(f"  max |delta| across all features: {delta.max():.6f}")
    print(f"  mean std |delta|               : {delta.mean():.6f}")

    print("\nWhy this matters: the 'fit on all' path lets val/test stats\n"
          "bleed into training. Val metrics inflate; real-world performance\n"
          "drops when the model faces genuinely new data.\n"
          "Rule: any fit on data (StandardScaler, PCA, IQR bounds, class priors)\n"
          "      must use TRAIN ONLY.")


if __name__ == "__main__":
    main()
