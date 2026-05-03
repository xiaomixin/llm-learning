"""Concept 05 — The 5-step training loop, on a trivial regression.

Run: python 05_train_loop.py

Goal: fit y = 2x + 1 with a 1-parameter Linear. Watch the 5 components
(zero_grad / forward / loss / backward / step) actually move the weight.
"""
import torch
import torch.nn as nn


def main():
    torch.manual_seed(0)

    # ── Synthetic data: y = 2x + 1 + noise ─────────────────────────
    x = torch.linspace(-3, 3, 200).unsqueeze(1)       # (200, 1)
    y = 2 * x + 1 + 0.1 * torch.randn_like(x)

    # ── Model and loss ─────────────────────────────────────────────
    model = nn.Linear(1, 1)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)

    # Initial params — expect random, far from (W=2, b=1)
    print(f"init    W={model.weight.item():+.3f}  b={model.bias.item():+.3f}")

    # ── Training loop: exactly 5 steps inside ──────────────────────
    for epoch in range(1, 21):
        # 1. clear last step's grads
        optimizer.zero_grad()
        # 2. forward
        y_pred = model(x)
        # 3. loss
        loss = loss_fn(y_pred, y)
        # 4. backward — fills .grad of every parameter
        loss.backward()
        # 5. step — reads .grad, moves parameters
        optimizer.step()

        if epoch % 5 == 0 or epoch == 1:
            print(f"ep {epoch:02d}  loss={loss.item():.4f}  "
                  f"W={model.weight.item():+.3f}  b={model.bias.item():+.3f}")

    print(f"\nConverges to W~2, b~1 (true values). ✓")

    # ── What happens if you forget zero_grad? ─────────────────────
    print("\n--- Same loop, but WITHOUT zero_grad() ---")
    torch.manual_seed(0)
    model2 = nn.Linear(1, 1)
    opt2 = torch.optim.SGD(model2.parameters(), lr=0.05)
    for epoch in range(1, 11):
        # (no zero_grad!)
        y_pred = model2(x)
        loss = loss_fn(y_pred, y)
        loss.backward()
        opt2.step()
        if epoch in (1, 3, 5, 10):
            print(f"ep {epoch:02d}  loss={loss.item():.4f}  "
                  f"grad_norm={model2.weight.grad.abs().item():.3f}")
    print("grad keeps piling up => behaves like a runaway learning rate.")


if __name__ == "__main__":
    main()
