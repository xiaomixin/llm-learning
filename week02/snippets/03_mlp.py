"""Concept 03 — MLP = Linear + ReLU stack.

Run: python 03_mlp.py

Goal:
1. Prove that Linear + Linear (no activation) collapses to a single Linear.
2. Build the same MLP the Week-2 notebook uses and count its params.
"""
import torch
import torch.nn as nn


def prove_linear_stack_is_still_linear():
    torch.manual_seed(0)
    stack_no_act = nn.Sequential(
        nn.Linear(3, 4),
        nn.Linear(4, 2),
    )
    x = torch.randn(5, 3)
    y_stack = stack_no_act(x)

    # Fold the two Linears by hand: (X @ W1.T + b1) @ W2.T + b2
    #                             = X @ (W2 @ W1).T + (W2 @ b1 + b2)
    W1, b1 = stack_no_act[0].weight, stack_no_act[0].bias
    W2, b2 = stack_no_act[1].weight, stack_no_act[1].bias
    W_eq = W2 @ W1
    b_eq = W2 @ b1 + b2
    y_eq = x @ W_eq.T + b_eq

    print("two Linears w/o activation == one equivalent Linear? ",
          torch.allclose(y_stack, y_eq, atol=1e-6))
    print("  -> stacking without a nonlinearity gains you NOTHING.")


def build_week2_mlp():
    class MLP(nn.Module):
        def __init__(self, in_dim=30, hidden=64, p_drop=0.3):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(p_drop),
                nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(p_drop),
                nn.Linear(hidden, 1),
            )

        def forward(self, x):
            return self.net(x).squeeze(-1)   # (B, 1) -> (B,)

    model = MLP(in_dim=30)
    x = torch.randn(8, 30)
    logits = model(x)
    print("\nWeek-2 MLP")
    print("  input :", x.shape)
    print("  output:", logits.shape, "(logits, before sigmoid)")
    n = sum(p.numel() for p in model.parameters())
    # 30*64+64 + 64*64+64 + 64*1+1 = 1984 + 4160 + 65 = 6209
    print(f"  params: {n}  (= 30*64+64 + 64*64+64 + 64*1+1)")


def show_relu_bending():
    """ReLU drops the negative half. Piecewise-linear bending is where
       non-linearity comes from."""
    x = torch.linspace(-3, 3, 7)
    print("\nReLU demo:")
    print("  x    =", x.tolist())
    print("  y    =", torch.relu(x).tolist())


if __name__ == "__main__":
    prove_linear_stack_is_still_linear()
    show_relu_bending()
    build_week2_mlp()
