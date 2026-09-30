"""Score every ranking component against wet-lab outcomes, under both label rules.

This exists because an earlier comparison in this project reached the wrong
conclusion. It measured the *composite* score, which deliberately includes a
synthesis-risk penalty that has nothing to do with potency, against an activity
label built from the *median* MIC across strains, where most values sit at the
64 uM assay ceiling. Both choices pushed the learned regressor's apparent skill
down to chance and led to it being under-weighted.

Every component is now scored separately, under both a permissive label (potent
against at least one Gram-negative strain) and a strict one (potent at the median
Gram-negative strain), so the effect of each choice is visible.
"""
from __future__ import annotations

import argparse
import csv
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np

from ampforge.encode import encode_many
from ampforge.features import descriptors
from ampforge.rank import Rankers, pharmacophore_score, score_candidates, synthesis_penalty

STD_AA = set("ACDEFGHIKLMNPQRSTVWY")
POTENCY_UM = 16.0


def auroc(scores: np.ndarray, positive: np.ndarray) -> float:
    a, b = scores[positive], scores[~positive]
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    return float((a[:, None] > b[None, :]).mean() + 0.5 * (a[:, None] == b[None, :]).mean())


def bootstrap_ci(scores: np.ndarray, positive: np.ndarray, n: int = 2000, seed: int = 0):
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n):
        idx = rng.integers(0, len(scores), len(scores))
        if positive[idx].sum() < 2 or (~positive[idx]).sum() < 2:
            continue
        draws.append(auroc(scores[idx], positive[idx]))
    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def _load(path: Path, mic_col: str, token: str) -> dict[str, list[float]]:
    table: dict[str, list[float]] = defaultdict(list)
    with path.open() as fh:
        for row in csv.DictReader(fh):
            seq = row["sequence"].strip().upper()
            if not (set(seq) <= STD_AA and 8 <= len(seq) <= 50):
                continue
            if row["strain_type"].strip() != token:
                continue
            try:
                table[seq].append(float(row[mic_col]))
            except ValueError:
                continue
    return table


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ampdiff-mic", type=Path, required=True)
    ap.add_argument("--hydramp-mic", type=Path, required=True)
    args = ap.parse_args()

    raw = {**_load(args.ampdiff_mic, "mic", "gram-"),
           **_load(args.hydramp_mic, "mic_uM", "-")}
    sequences = sorted(raw)
    best = np.array([min(raw[s]) for s in sequences])
    median = np.array([statistics.median(raw[s]) for s in sequences])

    rankers = Rankers.load(Path("checkpoint/rankers.npz"))
    features = encode_many(sequences)
    parts = score_candidates(sequences, rankers)
    signals = {
        "MIC regressor (Gram-negative)": -rankers.predict("mic_gram_negative", features),
        "predicted success rate, overall": parts["success_rate_overall"],
        "pharmacophore score": pharmacophore_score(sequences),
        "net charge": np.array([descriptors(s)["charge"] for s in sequences]),
        "hydrophobic moment": np.array([descriptors(s)["hmoment"] for s in sequences]),
        "predicted log safety window": parts["log_safety_window"],
        "synthesis penalty (negated)": -synthesis_penalty(sequences),
        "composite score as shipped": parts["composite"],
    }

    for label, values in (("potent against at least one Gram-negative strain", best),
                          ("potent at the median Gram-negative strain", median)):
        active = values <= POTENCY_UM
        print(f"\n=== label: {label} ===")
        print(f"n = {len(sequences)}, active = {int(active.sum())} ({active.mean():.0%})")
        print(f"\n{'signal':34s} {'AUROC':>7s}   {'95% bootstrap CI':>20s}")
        print("-" * 66)
        rows = sorted(((auroc(v, active), k, v) for k, v in signals.items()), reverse=True)
        for value, name, series in rows:
            lo, hi = bootstrap_ci(series, active)
            print(f"{name:34s} {value:7.3f}   {lo:8.3f} to {hi:6.3f}")


if __name__ == "__main__":
    main()
