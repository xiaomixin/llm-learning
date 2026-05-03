"""Shared helper for snippets that need creditcard.csv.

Looks up the CSV in the usual places or from the CREDITCARD_CSV env var.
Raises a friendly error if not found, pointing the user at the download step.
"""
from __future__ import annotations

import os
import pathlib


def find_creditcard_csv() -> pathlib.Path:
    """Return path to creditcard.csv or raise FileNotFoundError with a helpful hint."""
    # 1) explicit env var wins
    env = os.environ.get("CREDITCARD_CSV")
    if env:
        p = pathlib.Path(env).expanduser()
        if p.exists():
            return p

    # 2) a few common locations
    here = pathlib.Path(__file__).resolve().parent  # .../week02/snippets
    candidates = [
        here.parent / "data" / "creditcard.csv",                # week02/data/
        here.parent.parent / "data" / "creditcard.csv",          # learning/data/
        pathlib.Path.cwd() / "data" / "creditcard.csv",          # ./data/
        pathlib.Path("~/Code/binance/risk/openspec-projects/learning/week02/data/creditcard.csv").expanduser(),
    ]
    for c in candidates:
        if c.exists():
            return c

    tried = "\n  ".join(str(c) for c in candidates)
    raise FileNotFoundError(
        "creditcard.csv not found. Tried:\n  "
        + tried
        + "\n\nHow to get it:\n"
        "  1) `pip install kaggle`\n"
        "  2) `kaggle datasets download -d mlg-ulb/creditcardfraud -p week02/data --unzip`\n"
        "Or set the CREDITCARD_CSV env var to the absolute path.\n"
    )


def load_creditcard():
    """Return (X, y, feature_names) as numpy arrays + list[str]."""
    import pandas as pd

    csv = find_creditcard_csv()
    df = pd.read_csv(csv)
    feature_cols = [c for c in df.columns if c != "Class"]
    X = df[feature_cols].values.astype("float32")
    y = df["Class"].values.astype("float32")
    return X, y, feature_cols


if __name__ == "__main__":
    path = find_creditcard_csv()
    print(f"found: {path}")
    X, y, feats = load_creditcard()
    print(f"X shape: {X.shape}, y shape: {y.shape}, fraud ratio: {y.mean():.5f}")
