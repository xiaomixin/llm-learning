"""Concept 01 — Tensor and autograd.

Run: python 01_tensor.py

Goal: see with your own eyes that a tensor "remembers its origin",
and that calling .backward() fills in the .grad of every leaf.
"""
import torch


def main():
    # ── 1. A tensor is like a numpy array ──────────────────────────
    x = torch.tensor([1.0, 2.0, 3.0])
    print("x         =", x)
    print("x.shape   =", x.shape)
    print("x.dtype   =", x.dtype)

    # ── 2. But when requires_grad=True, it starts tracking ─────────
    x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    y = (x ** 2).sum()             # y = 1 + 4 + 9 = 14
    print("\ny         =", y.item(), "  <- scalar")
    print("y.grad_fn =", y.grad_fn, "  <- 'who made me'")

    # ── 3. .backward() fills x.grad with dy/dx ─────────────────────
    # y = x1^2 + x2^2 + x3^2  =>  dy/dx_i = 2*x_i  =>  [2, 4, 6]
    y.backward()
    print("x.grad    =", x.grad)

    # ── 4. .grad is additive — call backward() twice and see ───────
    x.grad.zero_()                 # reset to [0, 0, 0]
    y = (x ** 2).sum(); y.backward()
    print("\nfirst  backward  -> x.grad =", x.grad.tolist())
    y = (x ** 2).sum(); y.backward()  # WITHOUT zero_grad, adds on top
    print("second backward  -> x.grad =", x.grad.tolist(), "  <- doubled!")

    # ── 5. This is the whole reason optimizer.zero_grad() exists ───
    print("\nIf you forget zero_grad(), every step adds last step's grad on top.")
    print("Training becomes 'learning rate * epoch' — loss diverges.")


if __name__ == "__main__":
    main()
