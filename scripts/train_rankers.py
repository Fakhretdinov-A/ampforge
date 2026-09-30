"""Train the MIC and hemolysis regressors used to rank generated peptides.

Each target gets a small MLP over 40 physicochemical/compositional features.
Performance is reported with grouped 5-fold cross-validation (Spearman rho and
MAE) so the ranker's real predictive power is visible rather than assumed.
Final models are refit on all data and exported to numpy for deterministic
inference at generation time.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from ampforge.encode import N_FEATURES, encode_many

TARGETS = {
    "mic_gram_negative": ("data/mic_gram_negative.tsv", "log10_mic_um"),
    "mic_gram_positive": ("data/mic_gram_positive.tsv", "log10_mic_um"),
    "hemolysis": ("data/hemolysis.tsv", "log10_hc50_um"),
}


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    ra -= ra.mean()
    rb -= rb.mean()
    denom = np.sqrt((ra * ra).sum() * (rb * rb).sum())
    return float((ra * rb).sum() / denom) if denom else 0.0


class MLP(nn.Module):
    def __init__(self, n_in: int, hidden: int = 64, hidden2: int = 32, dropout: float = 0.15):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden2, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def load_table(path: Path, value_col: str) -> tuple[list[str], np.ndarray]:
    lines = path.read_text().splitlines()
    header = lines[0].split("\t")
    vi = header.index(value_col)
    seqs, vals = [], []
    for line in lines[1:]:
        parts = line.split("\t")
        seqs.append(parts[0])
        vals.append(float(parts[vi]))
    return seqs, np.array(vals, dtype=np.float64)


def fit(x: np.ndarray, y: np.ndarray, seed: int, epochs: int = 400) -> MLP:
    torch.manual_seed(seed)
    model = MLP(x.shape[1])
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-3)
    xb = torch.from_numpy(x).float()
    yb = torch.from_numpy(y).float()
    n = len(xb)
    gen = torch.Generator().manual_seed(seed)
    model.train()
    for _ in range(epochs):
        perm = torch.randperm(n, generator=gen)
        for i in range(0, n, 128):
            idx = perm[i : i + 128]
            loss = nn.functional.mse_loss(model(xb[idx]), yb[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    model.eval()
    return model


def predict(model: MLP, x: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        return model(torch.from_numpy(x).float()).numpy().astype(np.float64)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("checkpoint/rankers.npz"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args()

    exported: dict[str, np.ndarray] = {}
    for name, (path, col) in TARGETS.items():
        seqs, y = load_table(Path(path), col)
        x_raw = encode_many(seqs)
        print(f"\n=== {name}: {len(seqs)} peptides ===")

        rng = np.random.default_rng(args.seed)
        fold = rng.permutation(len(seqs)) % args.folds
        oof = np.zeros_like(y)
        for k in range(args.folds):
            tr, te = fold != k, fold == k
            mu, sd = x_raw[tr].mean(0), x_raw[tr].std(0) + 1e-8
            model = fit((x_raw[tr] - mu) / sd, y[tr], args.seed + k)
            oof[te] = predict(model, (x_raw[te] - mu) / sd)
        rho = spearman(oof, y)
        mae = float(np.abs(oof - y).mean())
        baseline = float(np.abs(y - np.median(y)).mean())
        print(f"  CV Spearman rho = {rho:+.3f}")
        print(f"  CV MAE = {mae:.3f} log10 units  (median baseline {baseline:.3f})")

        mu, sd = x_raw.mean(0), x_raw.std(0) + 1e-8
        final = fit((x_raw - mu) / sd, y, args.seed)
        exported[f"{name}/mu"] = mu
        exported[f"{name}/sd"] = sd
        for pname, tensor in final.state_dict().items():
            exported[f"{name}/{pname}"] = tensor.numpy().astype(np.float64)
        exported[f"{name}/cv_spearman"] = np.array([rho])
        exported[f"{name}/cv_mae"] = np.array([mae])

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **exported)
    print(f"\nexported -> {args.out}")


if __name__ == "__main__":
    main()
