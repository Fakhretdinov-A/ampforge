"""Re-run top-100 selection over an existing library, for development iteration.

The library depends only on sampling and the AMP-likeness screen, so changing the
ranking does not change it. This script lets the ranking be iterated in seconds
instead of regenerating 50,000 sequences. The submitted files are always produced
by `uv run generate`, which is the single reproducible entry point.
"""
from __future__ import annotations

import argparse
import statistics
from pathlib import Path

import numpy as np

from ampforge.features import descriptors
from ampforge.generate import read_fasta, select_top, write_fasta
from ampforge.rank import MAX_SYNTHESIS_RISK, Rankers, score_candidates, synthesis_penalty


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--library", type=Path, default=Path("generate/library.fasta"))
    ap.add_argument("--reference", type=Path, default=Path("data/marlys_reference.fasta"))
    ap.add_argument("--rankers", type=Path, default=Path("checkpoint/rankers.npz"))
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--top-k", type=int, default=100)
    args = ap.parse_args()

    library = read_fasta(args.library)
    references = read_fasta(args.reference)
    rankers = Rankers.load(args.rankers)
    print(f"library {len(library):,}  reference {len(references):,}")

    scored = score_candidates(library, rankers)
    risk = synthesis_penalty(library)
    passing = int((risk <= MAX_SYNTHESIS_RISK).sum())
    print(f"{passing:,} pass the synthesis-risk gate ({passing/len(library):.1%})")

    top = select_top(library, scored["composite"], references, args.top_k,
                     risk=risk, verbose=False)
    if args.out is not None:
        write_fasta(top, args.out, "ampforge_top_")
        print(f"wrote {args.out}")

    stats = score_candidates(top, rankers)
    d = [descriptors(s) for s in top]
    print(f"\ntop-{args.top_k} properties")
    for key in ("charge", "hmoment", "gravy", "helix_prop"):
        values = [x[key] for x in d]
        print(f"  {key:12s} mean {statistics.mean(values):6.2f}   "
              f"range {min(values):6.2f} to {max(values):6.2f}")
    lengths = [len(s) for s in top]
    print(f"  {'length':12s} mean {statistics.mean(lengths):6.1f}   "
          f"range {min(lengths):6d} to {max(lengths):6d}")
    print(f"  cysteine-free {sum(1 for s in top if 'C' not in s)}/{len(top)}")
    print(f"\npredicted, mean over the {args.top_k}")
    print(f"  overall success rate   {stats['success_rate_overall'].mean():.3f}")
    print(f"  Gram-negative success  {stats['success_rate_gram_negative'].mean():.3f}")
    print(f"  Gram-positive success  {stats['success_rate_gram_positive'].mean():.3f}")
    print(f"  Gram-negative MIC      {10**stats['mic_gram_negative'].mean():.2f} uM")
    print(f"  safety window          {10**stats['log_safety_window'].mean():.0f}x")
    print(f"  synthesis risk         {synthesis_penalty(top).mean():.2f} "
          f"(gate at {MAX_SYNTHESIS_RISK})")


if __name__ == "__main__":
    main()
