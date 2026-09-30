"""Train the AMP-likeness classifier used to screen the generated library.

Why this exists separately from the MIC rankers
-----------------------------------------------
Held-out wet-lab evidence showed that learned scorers cannot rank *potency*
within an already AMP-like set. They are, however, genuinely good at the
different and easier question of whether a sequence looks like an antimicrobial
peptide at all. Phase 1 of the competition scores the 50,000-sequence library
partly through exactly that kind of surrogate classifier, so this model is used
to screen the library, and deliberately not to order the top-100.

Data (all MIT, redistributed with the HydrAMP starter kit):
  positives  veltri_positive.csv        AMP Scanner v2 benchmark positives
  negatives  veltri_negative.csv        AMP Scanner v2 benchmark negatives
             unlabelled_negative.csv    UniProt non-AMPs, CD-HIT filtered
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from ampforge.encode import encode_many

STD_AA = set("ACDEFGHIKLMNPQRSTVWY")
MIN_LEN, MAX_LEN = 8, 50


def read_sequences(path: Path, limit: int | None = None) -> list[str]:
    out: list[str] = []
    with path.open() as fh:
        for row in csv.DictReader(fh):
            seq = (row.get("Sequence") or "").strip().upper()
            if set(seq) <= STD_AA and MIN_LEN <= len(seq) <= MAX_LEN:
                out.append(seq)
            if limit is not None and len(out) >= limit:
                break
    return out


class Classifier(nn.Module):
    def __init__(self, n_in: int, hidden: int = 96, hidden2: int = 48, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden2, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def auroc(scores: np.ndarray, positive: np.ndarray) -> float:
    a, b = scores[positive == 1], scores[positive == 0]
    return float((a[:, None] > b[None, :]).mean() + 0.5 * (a[:, None] == b[None, :]).mean())


def fit(x: np.ndarray, y: np.ndarray, seed: int, epochs: int = 120) -> Classifier:
    torch.manual_seed(seed)
    model = Classifier(x.shape[1])
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-3)
    xb, yb = torch.from_numpy(x).float(), torch.from_numpy(y).float()
    pos_weight = torch.tensor([(y == 0).sum() / max((y == 1).sum(), 1)])
    gen = torch.Generator().manual_seed(seed)
    model.train()
    for _ in range(epochs):
        perm = torch.randperm(len(xb), generator=gen)
        for i in range(0, len(xb), 256):
            idx = perm[i : i + 256]
            loss = nn.functional.binary_cross_entropy_with_logits(
                model(xb[idx]), yb[idx], pos_weight=pos_weight
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    model.eval()
    return model


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, required=True,
                    help="HydrAMP starter-kit data/training directory")
    ap.add_argument("--out", type=Path, default=Path("checkpoint/amp_classifier.npz"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-negatives", type=int, default=12000)
    args = ap.parse_args()

    positives = read_sequences(args.data_dir / "veltri_positive.csv")
    negatives = read_sequences(args.data_dir / "veltri_negative.csv")
    negatives += read_sequences(args.data_dir / "unlabelled_negative.csv", args.max_negatives)
    positives = sorted(set(positives))
    negatives = sorted(set(negatives) - set(positives))
    print(f"positives {len(positives)}  negatives {len(negatives)}")

    seqs = positives + negatives
    y = np.r_[np.ones(len(positives)), np.zeros(len(negatives))]
    x_raw = encode_many(seqs)

    rng = np.random.default_rng(args.seed)
    fold = rng.permutation(len(seqs)) % 5
    oof = np.zeros(len(seqs))
    for k in range(5):
        tr, te = fold != k, fold == k
        mu, sd = x_raw[tr].mean(0), x_raw[tr].std(0) + 1e-8
        model = fit((x_raw[tr] - mu) / sd, y[tr], args.seed + k)
        with torch.no_grad():
            oof[te] = model(torch.from_numpy((x_raw[te] - mu) / sd).float()).numpy()
    auc = auroc(oof, y)
    # Matthews correlation at the natural threshold
    pred = (oof > 0).astype(float)
    tp = ((pred == 1) & (y == 1)).sum(); tn = ((pred == 0) & (y == 0)).sum()
    fp = ((pred == 1) & (y == 0)).sum(); fn = ((pred == 0) & (y == 1)).sum()
    denom = np.sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))) or 1.0
    mcc = (tp * tn - fp * fn) / denom
    print(f"5-fold AUROC {auc:.3f}   MCC {mcc:.3f}")

    mu, sd = x_raw.mean(0), x_raw.std(0) + 1e-8
    final = fit((x_raw - mu) / sd, y, args.seed)
    arrays = {"mu": mu, "sd": sd, "cv_auroc": np.array([auc]), "cv_mcc": np.array([mcc])}
    for name, tensor in final.state_dict().items():
        arrays[name] = tensor.numpy().astype(np.float64)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **arrays)
    print(f"exported -> {args.out}")


if __name__ == "__main__":
    main()
