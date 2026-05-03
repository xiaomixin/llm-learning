"""Concept 07 — Focal Loss modulation.

Run: python 07_focal.py

Goal: plot the modulation factor (1 - p_t)^gamma across gamma values
so you can *see* how Focal kills gradient from easy samples.
Also print a few point values.
"""
import torch
import torch.nn as nn


def focal_loss(logits, targets, alpha=0.25, gamma=2.0, reduction="none"):
    """Numerically stable focal loss using BCEWithLogits inside."""
    p = torch.sigmoid(logits)
    ce = nn.functional.binary_cross_entropy_with_logits(
        logits, targets, reduction="none"
    )
    p_t = p * targets + (1 - p) * (1 - targets)
    alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
    fl = alpha_t * (1 - p_t) ** gamma * ce
    if reduction == "none":
        return fl
    return fl.mean()


def main():
    print("Modulation factor (1 - p_t)^gamma for a correctly-leaning sample")
    print(" p_t is 'prob model assigns to the TRUE class'.")
    print(" Easy sample: p_t near 1 (correct and confident).")
    print(" Hard sample: p_t near 0.5 or lower.\n")

    p_t_values = [0.99, 0.9, 0.75, 0.5, 0.25, 0.1, 0.01]
    print(f"{'p_t':>6} | " + " | ".join(f"g={g}" for g in (0, 1, 2, 5)))
    print("-" * 46)
    for p_t in p_t_values:
        row = [f"{(1 - p_t) ** g:.4f}" for g in (0, 1, 2, 5)]
        print(f"{p_t:>6.2f} | " + " | ".join(row))

    print("\nReadings:")
    print("  gamma=0 -> weight is always 1 (degenerates to weighted CE)")
    print("  gamma=2 -> easy sample (p_t=0.9) weight 0.01, hard (p_t=0.5) weight 0.25")
    print("  gamma=5 -> easy sample near zero weight; training focuses on hard cases")

    # ── Show full-pipeline focal loss on a few toy logits ─────────
    print("\nFocal Loss values on toy logits (y = 1 positive case):")
    logits = torch.tensor([-2.0, -0.5, 0.0, 0.5, 2.0, 5.0])
    targets = torch.ones_like(logits)
    for gamma in (0.0, 2.0, 5.0):
        fl = focal_loss(logits, targets, alpha=0.75, gamma=gamma)
        print(f"  gamma={gamma}: " + ", ".join(f"{v.item():.3f}" for v in fl))


if __name__ == "__main__":
    main()
