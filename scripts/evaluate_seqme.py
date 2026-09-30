"""Score generated libraries with seqme, the framework the competition's Phase 1 uses.

Runs in its own environment because seqme pulls in torch, transformers and ESM-2,
none of which the submission's own entry point needs.

Usage:
    python scripts/evaluate_seqme.py --library generate/library.fasta \
        --baseline <hydramp-kit>/generate_broad_spectrum/library.fasta \
        --reference data/antibacterial.fasta
"""
from __future__ import annotations

import argparse
from pathlib import Path

import seqme as sm


def read_fasta(path: Path) -> list[str]:
    sequences, parts = [], []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if parts:
                sequences.append("".join(parts))
                parts = []
        else:
            parts.append(line.upper())
    if parts:
        sequences.append("".join(parts))
    return sequences


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--library", type=Path, default=None)
    ap.add_argument("--baseline", type=Path, default=None,
                    help="a competing library to compare against, e.g. the HydrAMP baseline")
    ap.add_argument("--group", action="append", default=[], metavar="NAME=PATH",
                    help="additional named library, repeatable")
    ap.add_argument("--reference", type=Path, required=True,
                    help="known AMPs; the distributional target and the novelty reference")
    ap.add_argument("--subsample", type=int, default=5000,
                    help="sequences per group for the embedding-based metrics")
    ap.add_argument("--esm2", default="facebook/esm2_t6_8M_UR50D")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    reference = read_fasta(args.reference)
    groups: dict[str, list[str]] = {}
    if args.library is not None:
        groups["AMPforge"] = read_fasta(args.library)
    for spec in args.group:
        name, _, path = spec.partition("=")
        groups[name] = read_fasta(Path(path))
    if args.baseline is not None:
        groups["baseline"] = read_fasta(args.baseline)
    groups["known AMPs"] = reference

    for name, seqs in groups.items():
        print(f"{name}: {len(seqs):,} sequences")

    # Improved precision/recall is defined for equal-sized samples, so the
    # reference is subsampled to the same size with a fixed seed.
    rng = __import__("numpy").random.default_rng(0)
    if len(reference) > args.subsample:
        idx = rng.choice(len(reference), args.subsample, replace=False)
        reference_matched = [reference[i] for i in sorted(idx)]
    else:
        reference_matched = reference

    cache = sm.Cache(models={
        "esm2": sm.models.ESM2(model_name=args.esm2, batch_size=256, device=args.device)
    })
    embedder = cache.model("esm2")

    metrics = [
        sm.metrics.Uniqueness(),
        sm.metrics.Novelty(reference=reference),
        sm.metrics.Subset(sm.metrics.Diversity(), n_samples=args.subsample),
        sm.metrics.Subset(sm.metrics.FBD(reference=reference, embedder=embedder),
                          n_samples=args.subsample),
        sm.metrics.Subset(sm.metrics.MMD(reference=reference, embedder=embedder),
                          n_samples=args.subsample),
        sm.metrics.Subset(sm.metrics.Precision(reference=reference_matched, embedder=embedder, n_neighbors=3),
                          n_samples=args.subsample),
        sm.metrics.Subset(sm.metrics.Recall(reference=reference_matched, embedder=embedder, n_neighbors=3),
                          n_samples=args.subsample),
        sm.metrics.Subset(sm.metrics.AuthPct(train_set=reference, embedder=embedder),
                          n_samples=args.subsample),
    ]

    df = sm.evaluate(groups, metrics)
    print()
    print(df.to_string())


if __name__ == "__main__":
    main()
