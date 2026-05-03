"""Concept 06 — pos_weight for class imbalance.

Run: python 06_pos_weight.py

Goal: build a 1:100 imbalanced toy dataset, train with and without
pos_weight, and see AUC-PR diverge. Also verify by hand that pos_weight
multiplies the positive-class gradient by exactly w.
"""
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score


def make_imbalanced(n_neg=10_000, n_pos=100, seed=0):
    """Two Gaussian clusters, 100:1 imbalanced, mildly separable."""
    g = torch.Generator().manual_seed(seed)
    X_neg = torch.randn(n_neg, 5, generator=g)
    X_pos = torch.randn(n_pos, 5, generator=g) + 0.8   # shift mean
    X = torch.cat([X_neg, X_pos])
    y = torch.cat([torch.zeros(n_neg), torch.ones(n_pos)])
    # shuffle
    perm = torch.randperm(len(y), generator=g)
    return X[perm], y[perm]


def train(X, y, pos_weight=None, epochs=200, seed=0):
    torch.manual_seed(seed)
    model = nn.Sequential(nn.Linear(5, 16), nn.ReLU(), nn.Linear(16, 1))
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    pw = None if pos_weight is None else torch.tensor([pos_weight])
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pw)
    for _ in range(epochs):
        opt.zero_grad()
        logits = model(X).squeeze(-1)
        loss = loss_fn(logits, y)
        loss.backward()
        opt.step()
    with torch.no_grad():
        scores = torch.sigmoid(model(X)).squeeze(-1).numpy()
    return average_precision_score(y.numpy(), scores), model


def verify_gradient_scales():
    """Confirm: BCEWithLogitsLoss(pos_weight=w) scales positive-sample
    gradient by exactly w."""
    logit = torch.tensor([0.5], requires_grad=True)
    target = torch.tensor([1.0])

    for w in (1.0, 10.0, 100.0):
        if logit.grad is not None:
            logit.grad.zero_()
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([w]))
        loss_fn(logit, target).backward()
        # analytic: d/dz = w * (sigma(z) - 1) for y=1
        expected = w * (torch.sigmoid(logit).item() - 1)
        print(f"  pos_weight={w:6.1f}  grad={logit.grad.item():+.6f}  "
              f"expected={expected:+.6f}")


def main():
    X, y = make_imbalanced()
    print(f"positives: {int(y.sum())} / {len(y)}  "
          f"= {y.mean():.4%}")

    neg, pos = int((y == 0).sum()), int((y == 1).sum())
    pw = neg / pos
    print(f"suggested pos_weight = neg/pos = {pw:.1f}\n")

    # A: no pos_weight
    ap_no, _ = train(X, y, pos_weight=None)
    # B: with pos_weight
    ap_yes, _ = train(X, y, pos_weight=pw)
    # C: wildly too big
    ap_too, _ = train(X, y, pos_weight=10 * pw)

    print("AUC-PR comparison")
    print(f"  no pos_weight            : {ap_no:.4f}")
    print(f"  pos_weight = {pw:5.1f}       : {ap_yes:.4f}  <- best")
    print(f"  pos_weight = {10*pw:5.1f} (too big): {ap_too:.4f}")

    print("\nGradient verification (positive sample, logit=0.5, y=1):")
    verify_gradient_scales()


if __name__ == "__main__":
    main()
