"""Why the MIC regressor generalises to natural peptides but not to generated ones.

Under a homology-aware split on natural peptides the regressor beats every
physicochemical signal (AUROC 0.72 against 0.68 for net charge). On the 92
wet-lab-tested peptides that came out of generative models it scores at chance
while net charge reaches 0.72. This script tests the obvious explanation: the
generated peptides sit outside the regressor's training distribution.

For every wet-lab peptide it measures the highest sequence identity to any
training sequence, then reports scorer performance separately for peptides that
have a close training neighbour and peptides that do not.
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

from ampforge.features import descriptors
from ampforge.rank import Rankers, pharmacophore_score

STD_AA = set("ACDEFGHIKLMNPQRSTVWY")
POTENCY_UM = 16.0


def auroc(scores: np.ndarray, positive: np.ndarray) -> float:
    a, b = scores[positive], scores[~positive]
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    return float((a[:, None] > b[None, :]).mean() + 0.5 * (a[:, None] == b[None, :]).mean())


def _load(path: Path, mic_col: str, gram_neg_token: str) -> dict[str, float]:
    table: dict[str, list[float]] = defaultdict(list)
    with path.open() as fh:
        for row in csv.DictReader(fh):
            seq = row["sequence"].strip().upper()
            if not (set(seq) <= STD_AA and 8 <= len(seq) <= 50):
                continue
            if row["strain_type"].strip() != gram_neg_token:
                continue
            try:
                table[seq].append(float(row[mic_col]))
            except ValueError:
                continue
    return {s: min(v) for s, v in table.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ampdiff-mic", type=Path, required=True)
    ap.add_argument("--hydramp-mic", type=Path, required=True)
    ap.add_argument("--train-table", type=Path, default=Path("data/mic_gram_negative.tsv"))
    ap.add_argument("--near-threshold", type=float, default=0.60)
    args = ap.parse_args()

    wet = {**_load(args.ampdiff_mic, "mic", "gram-"),
           **_load(args.hydramp_mic, "mic_uM", "-")}
    sequences = sorted(wet)
    active = np.array([wet[s] <= POTENCY_UM for s in sequences])

    train = [l.split("\t")[0] for l in args.train_table.read_text().splitlines()[1:]]
    print(f"wet-lab peptides {len(sequences)}, active {int(active.sum())}")
    print(f"regressor training set {len(train)} peptides")

    from rapidfuzz import process
    from rapidfuzz.distance import Levenshtein

    identity = process.cdist(
        sequences, train, scorer=Levenshtein.normalized_similarity, workers=-1
    ).max(axis=1)

    print(f"\nhighest identity to any training sequence:")
    for q in (10, 25, 50, 75, 90):
        print(f"  {q}th percentile  {np.percentile(identity, q):.3f}")
    exact = int((identity >= 0.999).sum())
    print(f"  exact matches in training data: {exact}")

    rankers = Rankers.load(Path("checkpoint/rankers.npz"))
    from ampforge.encode import encode_many

    features = encode_many(sequences)
    regressor = -rankers.predict("mic_gram_negative", features)
    charge = np.array([descriptors(s)["charge"] for s in sequences])
    pharma = pharmacophore_score(sequences)

    near = identity >= args.near_threshold
    print(f"\nsplit at {args.near_threshold:.2f} identity: "
          f"{int(near.sum())} near, {int((~near).sum())} far")

    print(f"\n{'subset':28s} {'n':>4s} {'active':>7s} {'regressor':>10s} {'charge':>8s} {'pharma':>8s}")
    print("-" * 70)
    for label, mask in (("all wet-lab peptides", np.ones(len(sequences), bool)),
                        (f"near training data", near),
                        (f"far from training data", ~near)):
        if mask.sum() < 6:
            continue
        print(f"{label:28s} {int(mask.sum()):4d} {active[mask].mean():7.2f} "
              f"{auroc(regressor[mask], active[mask]):10.3f} "
              f"{auroc(charge[mask], active[mask]):8.3f} "
              f"{auroc(pharma[mask], active[mask]):8.3f}")

    print("\nIf the regressor recovers on the near subset and fails on the far one,")
    print("its weakness is distribution shift rather than a lack of signal, and the")
    print("relevant question becomes how far our own designs sit from training data.")


if __name__ == "__main__":
    main()
