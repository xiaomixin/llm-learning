"""Concept 02 — nn.Linear is y = W x + b.

Run: python 02_linear.py

Goal: peek inside a Linear layer. See its .weight, .bias, their shapes,
and verify that forward() is literally a matrix multiply + add.
"""
import torch
import torch.nn as nn


def main():
    torch.manual_seed(0)

    # ── 1. Create a Linear mapping 3 features -> 2 outputs ─────────
    layer = nn.Linear(in_features=3, out_features=2)
    print("W shape:", layer.weight.shape, " <- (out, in)")
    print("b shape:", layer.bias.shape,   " <- (out,)")
    print("W =\n", layer.weight.data)
    print("b =", layer.bias.data)

    # ── 2. Forward on a single sample ──────────────────────────────
    x = torch.tensor([1.0, 2.0, 3.0])
    y = layer(x)
    print("\nlayer(x) =", y)

    # ── 3. Reproduce it by hand ────────────────────────────────────
    y_manual = layer.weight @ x + layer.bias   # (2,3) @ (3,) + (2,) = (2,)
    print("manual   =", y_manual)
    assert torch.allclose(y, y_manual)

    # ── 4. Batch version — the usual case in training ──────────────
    X = torch.tensor([[1.0, 2.0, 3.0],
                      [0.0, 1.0, 0.0],
                      [4.0, 5.0, 6.0]])                 # shape (3, 3) == (B, in)
    Y = layer(X)                                        # shape (3, 2) == (B, out)
    print("\nbatch in shape :", X.shape)
    print("batch out shape:", Y.shape)

    # Verify with matrix form: Y = X @ W.T + b
    Y_manual = X @ layer.weight.T + layer.bias
    assert torch.allclose(Y, Y_manual)
    print("Y = X @ W.T + b  holds.")

    # ── 5. Parameter count grows as in*out + out ───────────────────
    n = sum(p.numel() for p in layer.parameters())
    print(f"\nparams = {n}  = in*out + out = {3*2} + {2}")


if __name__ == "__main__":
    main()
