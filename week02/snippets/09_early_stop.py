"""Concept 09 — Early stopping + best-model checkpoint.

Run: python 09_early_stop.py

Goal:
1. Train a tiny MLP that will overfit on a small dataset.
2. Watch early stopping fire, and verify the rolled-back weights
   correspond to the best val epoch (not the last one).
3. Demonstrate the deepcopy pitfall: without it, 'best_state' silently
   tracks the live model.
"""
import copy
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score


def make_overfit_prone(n_train=200, n_val=500, dim=50, seed=0):
    """Small train set, larger val set -> easy to overfit."""
    g = torch.Generator().manual_seed(seed)
    X_tr = torch.randn(n_train, dim, generator=g)
    y_tr = (X_tr[:, 0] + 0.5 * torch.randn(n_train, generator=g) > 0).float()
    X_val = torch.randn(n_val, dim, generator=g)
    y_val = (X_val[:, 0] + 0.5 * torch.randn(n_val, generator=g) > 0).float()
    return X_tr, y_tr, X_val, y_val


def eval_ap(model, X, y):
    model.eval()
    with torch.no_grad():
        scores = torch.sigmoid(model(X)).squeeze(-1).numpy()
    return average_precision_score(y.numpy(), scores)


def train_with_early_stopping(X_tr, y_tr, X_val, y_val,
                              patience=5, max_epochs=100, use_deepcopy=True):
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(X_tr.shape[1], 64), nn.ReLU(),
                          nn.Linear(64, 1))
    opt = torch.optim.Adam(model.parameters(), lr=5e-3)
    loss_fn = nn.BCEWithLogitsLoss()

    best_ap, bad = -1.0, 0
    best_state = None
    best_epoch = -1

    for epoch in range(1, max_epochs + 1):
        model.train()
        opt.zero_grad()
        logits = model(X_tr).squeeze(-1)
        loss = loss_fn(logits, y_tr)
        loss.backward()
        opt.step()

        val_ap = eval_ap(model, X_val, y_val)
        if val_ap > best_ap:
            best_ap = val_ap
            best_epoch = epoch
            # The critical line:
            if use_deepcopy:
                best_state = copy.deepcopy(model.state_dict())
            else:
                best_state = model.state_dict()   # shallow — tensors still alive
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                print(f"  early stop @ epoch {epoch}, best was epoch {best_epoch} "
                      f"(val_ap={best_ap:.4f})")
                break

    model.load_state_dict(best_state)
    return model, best_ap, best_epoch


def main():
    X_tr, y_tr, X_val, y_val = make_overfit_prone()

    # With deepcopy — the correct way
    print("[correct] with copy.deepcopy")
    model, ap, ep = train_with_early_stopping(X_tr, y_tr, X_val, y_val,
                                              use_deepcopy=True)
    rolled_ap = eval_ap(model, X_val, y_val)
    print(f"  rolled-back AP = {rolled_ap:.4f}  (recorded best {ap:.4f})")
    assert abs(rolled_ap - ap) < 1e-6, "rolled-back model should match best"

    # Without deepcopy — the silent bug
    print("\n[buggy] WITHOUT deepcopy — state_dict shares tensor refs")
    model, ap, ep = train_with_early_stopping(X_tr, y_tr, X_val, y_val,
                                              use_deepcopy=False)
    rolled_ap = eval_ap(model, X_val, y_val)
    print(f"  rolled-back AP = {rolled_ap:.4f}  (recorded best {ap:.4f})")
    print("  -> if this differs, you were evaluating the LATEST weights, "
          "not the best ones.")


if __name__ == "__main__":
    main()
