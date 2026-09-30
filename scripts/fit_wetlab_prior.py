"""Fit the wet-lab-calibrated activity prior used to rank AMPforge candidates.

Motivation
----------
Before building this, three families of scorer were tested against held-out
wet-lab measurements from the competition's own baseline methods:

  * an MLP trained on GRAMPA MIC data      -> AUROC 0.52 (chance)
  * the 13 OmegAMP XGBoost classifiers     -> AUROC 0.41-0.58, ensemble 0.49
  * classical physicochemical descriptors  -> AUROC 0.72 (net charge)

Learned AMP classifiers separate peptides from non-peptides well, but they carry
little signal for ranking *within* a library that is already AMP-like, which is
exactly the task here. The classical cationic-amphipathic pharmacophore does
carry signal. This module therefore fits a small, heavily regularised logistic
model on the pooled wet-lab outcomes and reports leave-one-out performance, so
the prior's real accuracy is measured rather than assumed.

Data: the experimental MIC tables shipped with the HydrAMP and AMP-Diffusion
starter kits (MIT), pooled; a peptide counts as active when its best
Gram-negative MIC is at or below the competition's 16 uM potency threshold.
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

from ampforge.features import descriptors

STD_AA = set("ACDEFGHIKLMNPQRSTVWY")
POTENCY_THRESHOLD_UM = 16.0
FEATURES = ["charge", "hmoment", "gravy", "helix_prop"]


def _load(path: Path, seq_col: str, mic_col: str, gram_neg_token: str) -> dict[str, float]:
    table: dict[str, list[float]] = defaultdict(list)
    with path.open() as fh:
        for row in csv.DictReader(fh):
            seq = row[seq_col].strip().upper()
            if not (set(seq) <= STD_AA and 8 <= len(seq) <= 50):
                continue
            if row["strain_type"].strip() != gram_neg_token:
                continue
            try:
                table[seq].append(float(row[mic_col]))
            except ValueError:
                continue
    return {seq: min(vals) for seq, vals in table.items()}


def load_pooled(ampdiff: Path, hydramp: Path) -> tuple[list[str], np.ndarray]:
    merged = {
        **_load(ampdiff, "sequence", "mic", "gram-"),
        **_load(hydramp, "sequence", "mic_uM", "-"),
    }
    seqs = sorted(merged)
    active = np.array([merged[s] <= POTENCY_THRESHOLD_UM for s in seqs], dtype=np.float64)
    return seqs, active


def auroc(scores: np.ndarray, positive: np.ndarray) -> float:
    a, b = scores[positive == 1], scores[positive == 0]
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    return float((a[:, None] > b[None, :]).mean() + 0.5 * (a[:, None] == b[None, :]).mean())


def fit_logistic(x: np.ndarray, y: np.ndarray, l2: float, steps: int = 4000) -> np.ndarray:
    w = np.zeros(x.shape[1] + 1)
    xb = np.hstack([np.ones((len(x), 1)), x])
    for _ in range(steps):
        p = 1.0 / (1.0 + np.exp(-xb @ w))
        grad = xb.T @ (p - y) / len(y) + l2 * np.r_[0.0, w[1:]]
        hess = xb.T @ (xb * (p * (1 - p))[:, None]) / len(y)
        hess[1:, 1:] += l2 * np.eye(x.shape[1])
        w -= np.linalg.solve(hess + 1e-6 * np.eye(len(w)), grad)
    return w


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ampdiff-mic", type=Path, required=True)
    ap.add_argument("--hydramp-mic", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("checkpoint/wetlab_prior.npz"))
    ap.add_argument("--l2", type=float, default=0.5)
    args = ap.parse_args()

    seqs, y = load_pooled(args.ampdiff_mic, args.hydramp_mic)
    x_raw = np.array([[descriptors(s)[k] for k in FEATURES] for s in seqs])
    mu, sd = x_raw.mean(0), x_raw.std(0) + 1e-8
    x = (x_raw - mu) / sd
    print(f"pooled wet-lab peptides: {len(seqs)}  active: {int(y.sum())}")

    # leave-one-out, the honest estimate at this sample size
    loo = np.zeros(len(seqs))
    for i in range(len(seqs)):
        keep = np.ones(len(seqs), dtype=bool)
        keep[i] = False
        w = fit_logistic(x[keep], y[keep], args.l2)
        loo[i] = w[0] + x[i] @ w[1:]
    print(f"leave-one-out AUROC: {auroc(loo, y):.3f}")
    print(f"  (single best descriptor, net charge: {auroc(x_raw[:, 0], y):.3f})")

    w = fit_logistic(x, y, args.l2)
    print("\nfitted coefficients (standardised):")
    print(f"  intercept {w[0]:+.3f}")
    for name, coef in zip(FEATURES, w[1:]):
        print(f"  {name:12s} {coef:+.3f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out, mu=mu, sd=sd, weights=w,
        feature_names=np.array(FEATURES),
        loo_auroc=np.array([auroc(loo, y)]),
        n_train=np.array([len(seqs)]),
    )
    print(f"\nexported -> {args.out}")


if __name__ == "__main__":
    main()
