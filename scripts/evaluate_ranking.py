"""Compare ranking signals under a homology-aware split.

Why this replaces the earlier random-fold cross-validation
---------------------------------------------------------
Public MIC datasets are full of near-duplicates: analogue series, single-residue
variants, the same peptide re-measured by different groups. A random fold split
puts near-identical sequences on both sides of the split, so a model can score
well by recognising neighbours rather than by learning what makes a peptide
potent. That inflates the apparent skill of any flexible model and is the most
likely reason a regressor with random-fold Spearman 0.61 transferred at chance
to held-out wet-lab data.

This script clusters the peptides by sequence identity first and keeps whole
clusters together in a fold, which is the split BATTLE-AMP calls `homology_40`.
Signals are then compared on the same folds:

  * the trained MIC regressor
  * net charge alone
  * the Eisenberg hydrophobic moment alone
  * the composite cationic-amphipathic pharmacophore score

Both a rank correlation against log MIC and an AUROC at the competition's 16 µM
potency threshold are reported, because the threshold is what the competition
actually scores.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from ampforge.encode import encode_many
from ampforge.features import descriptors
from ampforge.rank import LOG_MIC_THRESHOLD, Rankers, pharmacophore_score


def load_table(path: Path) -> tuple[list[str], np.ndarray]:
    lines = path.read_text().splitlines()
    header = lines[0].split("\t")
    value_index = header.index("log10_mic_um")
    sequences, values = [], []
    for line in lines[1:]:
        parts = line.split("\t")
        sequences.append(parts[0])
        values.append(float(parts[value_index]))
    return sequences, np.array(values)


def cluster_by_identity(sequences: list[str], threshold: float) -> np.ndarray:
    """Greedy centroid clustering over pairwise Levenshtein identity, CD-HIT style.

    Single-linkage was tried first and is unusable here: over short peptides it
    chains almost the whole dataset into one component (2,893 of 3,293 members at
    40% identity), which makes the fold sizes wildly unequal and the resulting
    estimates both noisy and pessimistic. Greedy centroid assignment keeps every
    cluster member within `threshold` of its own centroid and does not chain.
    """
    from rapidfuzz import process
    from rapidfuzz.distance import Levenshtein

    similarity = process.cdist(
        sequences, sequences, scorer=Levenshtein.normalized_similarity, workers=-1
    )
    order = np.argsort([-len(s) for s in sequences], kind="stable")
    labels = np.full(len(sequences), -1)
    centroids: list[int] = []
    for i in order:
        for c, centroid in enumerate(centroids):
            if similarity[i, centroid] >= threshold:
                labels[i] = c
                break
        else:
            labels[i] = len(centroids)
            centroids.append(int(i))
    return labels


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean()
    rb -= rb.mean()
    denom = np.sqrt((ra * ra).sum() * (rb * rb).sum())
    return float((ra * rb).sum() / denom) if denom else 0.0


def auroc(scores: np.ndarray, positive: np.ndarray) -> float:
    a, b = scores[positive], scores[~positive]
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    return float((a[:, None] > b[None, :]).mean() + 0.5 * (a[:, None] == b[None, :]).mean())


def fit_mlp(x: np.ndarray, y: np.ndarray, seed: int, epochs: int = 400):
    import torch
    import torch.nn as nn

    torch.manual_seed(seed)
    model = nn.Sequential(
        nn.Linear(x.shape[1], 64), nn.ReLU(), nn.Dropout(0.15),
        nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.15),
        nn.Linear(32, 1),
    )
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-3)
    xb, yb = torch.from_numpy(x).float(), torch.from_numpy(y).float()
    gen = torch.Generator().manual_seed(seed)
    model.train()
    for _ in range(epochs):
        perm = torch.randperm(len(xb), generator=gen)
        for i in range(0, len(xb), 128):
            idx = perm[i : i + 128]
            loss = nn.functional.mse_loss(model(xb[idx]).squeeze(-1), yb[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    model.eval()
    return model


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", type=Path, default=Path("data/mic_gram_negative.tsv"))
    ap.add_argument("--identity", type=float, default=0.40)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    sequences, y = load_table(args.table)
    print(f"{len(sequences)} peptides from {args.table}")

    clusters = cluster_by_identity(sequences, args.identity)
    n_clusters = clusters.max() + 1
    sizes = np.bincount(clusters)
    print(f"{n_clusters} clusters at {args.identity:.0%} identity "
          f"(largest {sizes.max()}, singletons {(sizes == 1).sum()})")

    rng = np.random.default_rng(args.seed)
    cluster_fold = rng.permutation(n_clusters) % args.folds
    fold = cluster_fold[clusters]

    x_raw = encode_many(sequences)
    import torch

    oof = np.zeros(len(sequences))
    for k in range(args.folds):
        train, test = fold != k, fold == k
        mu, sd = x_raw[train].mean(0), x_raw[train].std(0) + 1e-8
        model = fit_mlp((x_raw[train] - mu) / sd, y[train], args.seed + k)
        with torch.no_grad():
            oof[test] = model(
                torch.from_numpy((x_raw[test] - mu) / sd).float()
            ).squeeze(-1).numpy()

    active = y <= LOG_MIC_THRESHOLD
    charge = np.array([descriptors(s)["charge"] for s in sequences])
    moment = np.array([descriptors(s)["hmoment"] for s in sequences])
    pharma = pharmacophore_score(sequences)

    print(f"\nactive at MIC <= 16 uM: {int(active.sum())} of {len(sequences)} "
          f"({active.mean():.1%})")
    print(f"\n{'signal':34s} {'Spearman':>10s} {'AUROC':>8s}")
    print("-" * 54)
    rows = [
        ("MIC regressor, homology split", -oof, spearman(-oof, -y), auroc(-oof, active)),
        ("net charge", charge, spearman(charge, -y), auroc(charge, active)),
        ("hydrophobic moment", moment, spearman(moment, -y), auroc(moment, active)),
        ("pharmacophore score", pharma, spearman(pharma, -y), auroc(pharma, active)),
    ]
    for name, _score, rho, auc in rows:
        print(f"{name:34s} {rho:+10.3f} {auc:8.3f}")

    # how much of the regressor's apparent skill came from near-duplicates
    rng2 = np.random.default_rng(args.seed)
    random_fold = rng2.permutation(len(sequences)) % args.folds
    oof_random = np.zeros(len(sequences))
    for k in range(args.folds):
        train, test = random_fold != k, random_fold == k
        mu, sd = x_raw[train].mean(0), x_raw[train].std(0) + 1e-8
        model = fit_mlp((x_raw[train] - mu) / sd, y[train], args.seed + k)
        with torch.no_grad():
            oof_random[test] = model(
                torch.from_numpy((x_raw[test] - mu) / sd).float()
            ).squeeze(-1).numpy()
    print("-" * 54)
    print(f"{'MIC regressor, random split':34s} "
          f"{spearman(-oof_random, -y):+10.3f} {auroc(-oof_random, active):8.3f}")
    print("\nThe gap between the two regressor rows is the part of its apparent")
    print("skill that comes from near-duplicate leakage rather than generalisation.")


if __name__ == "__main__":
    main()
