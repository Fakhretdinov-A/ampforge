"""Measure the trade-off that decides how hard to screen the library.

Phase 1 scores a library on two families that pull in opposite directions.
Surrogate activity prediction rewards a library filtered aggressively toward
predicted-active peptides. Distributional similarity rewards a library that looks
like the real distribution of known antimicrobial peptides. The aggregation
weights are held out until Phase 1 closes, so neither family can be optimised
blindly.

For reference, the official HydrAMP baseline library sits at one extreme: 97.7% of
its sequences pass an AMP classifier at 0.5, higher than the 66.1% of real known
AMPs that do, while its Fréchet distance to the reference is 7.7 against our 1.0.

This script samples a candidate pool once and builds library variants that differ
only in screening strength, then reports both families for each, so the shape of
the trade-off is visible rather than assumed.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from ampforge.generate import build_prefixes, generate_library, read_fasta
from ampforge.nn import PeptideLM
from ampforge.rank import AmpClassifier, Rankers, score_candidates


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", type=int, default=30000)
    ap.add_argument("--library", type=int, default=10000)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out-dir", type=Path, default=Path("screen_variants"))
    args = ap.parse_args()

    model = PeptideLM.load(Path("checkpoint/peptide_lm.npz"))
    screen = AmpClassifier.load(Path("checkpoint/amp_classifier.npz"))
    rankers = Rankers.load(Path("checkpoint/rankers.npz"))
    reference = read_fasta(Path("data/antibacterial.fasta"))

    rng = np.random.default_rng(args.seed)
    pool = generate_library(
        model, rng, args.pool, set(reference),
        batch_size=args.batch_size, temperature=1.0, top_p=0.98,
        max_batches=int(args.pool / args.batch_size * 4), verbose=False,
    )
    print(f"pool {len(pool):,} unique candidates")

    amp_logit = screen.logit(pool)
    order = np.argsort(-amp_logit)
    n = args.library

    variants: dict[str, list[str]] = {
        "no screen": [pool[i] for i in sorted(rng.choice(len(pool), n, replace=False))],
        "mild screen": [pool[i] for i in sorted(order[: int(n * 1.6)][:n])],
        "current screen": [pool[i] for i in sorted(order[:n])],
        "hard screen": [pool[i] for i in sorted(order[: n // 2])] +
                       [pool[i] for i in sorted(order[n // 2 : n])],
    }
    # a genuinely harder screen: the very top of a much larger pool
    if len(pool) >= 3 * n:
        variants["hard screen"] = [pool[i] for i in sorted(order[: n])]
        variants["hardest screen"] = [pool[i] for i in sorted(order[: n // 3])] * 3
        variants.pop("hardest screen")
        variants["top third of pool"] = [pool[i] for i in sorted(order[: len(pool) // 3])][:n]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n{'variant':20s} {'n':>7s} {'AMP p>0.5':>10s} {'AMP p>0.9':>10s} "
          f"{'pred MIC':>9s} {'SR>0.5':>7s}")
    print("-" * 70)
    for name, sequences in variants.items():
        path = args.out_dir / (name.replace(" ", "_") + ".fasta")
        path.write_text("".join(f">s{i}\n{s}\n" for i, s in enumerate(sequences)))
        idx = rng.choice(len(sequences), min(4000, len(sequences)), replace=False)
        sample = [sequences[i] for i in idx]
        p = screen.probability(sample)
        scored = score_candidates(sample, rankers)
        print(f"{name:20s} {len(sequences):7d} {(p > 0.5).mean():10.3f} "
              f"{(p > 0.9).mean():10.3f} "
              f"{10 ** np.median(scored['mic_gram_negative']):9.2f} "
              f"{(scored['success_rate_overall'] > 0.5).mean():7.3f}")

    print(f"\nwrote variants to {args.out_dir}/; score them with evaluate_seqme.py to")
    print("see what each costs in distributional similarity.")


if __name__ == "__main__":
    main()
