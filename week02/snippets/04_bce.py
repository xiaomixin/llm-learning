"""Concept 04 — BCE loss, and why BCEWithLogitsLoss is not optional.

Run: python 04_bce.py

Goal: watch the naive "Sigmoid + BCELoss" blow up at extreme logits,
while BCEWithLogitsLoss stays finite.
"""
import math
import torch
import torch.nn as nn


def sigmoid(z):
    return 1.0 / (1.0 + math.exp(-z))


def naive_bce(z: float, y: int) -> float:
    """-y log sigma(z) - (1-y) log(1-sigma(z))  — the 'two-step' version."""
    p = sigmoid(z)
    # p may round to 0.0 or 1.0 for extreme z  => log explodes.
    if y == 1:
        return -math.log(p)
    else:
        return -math.log(1 - p)


def stable_bce(z: float, y: int) -> float:
    """max(z, 0) - z*y + log(1 + exp(-|z|))  — log-sum-exp form."""
    return max(z, 0.0) - z * y + math.log1p(math.exp(-abs(z)))


def main():
    print("logit z | y | naive BCE           | stable BCE")
    print("--------+---+---------------------+-----------")
    for z in [-50.0, -10.0, 0.0, 10.0, 50.0]:
        for y in (0, 1):
            try:
                n = f"{naive_bce(z, y):.6f}"
            except ValueError as e:
                n = f"ERROR: {e}"
            s = f"{stable_bce(z, y):.6f}"
            print(f"{z:+7.1f} | {y} | {n:<19} | {s}")

    # ── Compare with PyTorch built-ins ─────────────────────────────
    print("\nSame comparison with PyTorch (float32):")
    logits = torch.tensor([-50.0, -10.0, 0.0, 10.0, 50.0])
    targets = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0])  # y=1 everywhere

    # Naive: Sigmoid + BCELoss  (this is what you should NOT do)
    probs = torch.sigmoid(logits)
    try:
        naive = nn.BCELoss(reduction="none")(probs, targets)
        print("  Sigmoid + BCELoss   :", naive.tolist())
    except Exception as e:
        print("  Sigmoid + BCELoss   : ERROR", e)

    # Recommended: BCEWithLogitsLoss
    stable = nn.BCEWithLogitsLoss(reduction="none")(logits, targets)
    print("  BCEWithLogitsLoss   :", [f"{v:.6f}" for v in stable.tolist()])

    # For z = -50, y = 1, the "right" answer is ~50 (log(1+e^50) ≈ 50).
    # Naive gives inf; stable gives ~50.


if __name__ == "__main__":
    main()
