"""AMPforge submission entry point for the AMP Challenge 2027.

`uv run generate` writes two files into `generate/`:

    generate/library.fasta  50,000 unique designed peptides
    generate/top.fasta      the 100 ranked candidates for wet-lab validation

Everything downstream of the trained weights runs in numpy float64 with a fixed
seed, so repeated runs -- on any platform -- produce byte-identical output.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from . import tokenizer as tk
from .nn import PeptideLM, sample
from .rank import (MAX_SYNTHESIS_RISK, AmpClassifier, Rankers, score_candidates,
                   synthesis_penalty)

LIBRARY_SIZE = 50_000
TOP_SIZE = 100
MIN_LEN, MAX_LEN = 8, 50

# the top-100 must stay below 80% identity to any known antibacterial peptide;
# we hold a margin so that a different identity definition (the organizers use
# MMseqs2 alignment identity, the starter kit uses a Levenshtein ratio) cannot
# push a candidate over the line
TOP_IDENTITY_LIMIT = 0.72
# Alignment identity over the aligned region, which is how the rule is specified. The
# limit is held below the rule's 0.80 because two reasonable implementations disagree
# near the boundary: on a candidate this gate scored at 0.800, MMseqs2's BLOSUM-scored
# alignment reported 0.827. Edit distance and substitution-matrix alignment need not
# agree, and the organizers use the latter, so the margin absorbs the difference.
TOP_LOCAL_IDENTITY_LIMIT = 0.79
# candidates in the top list are also kept apart from each other, so that the
# 25 peptides drawn at random are 25 genuinely different designs
TOP_INTERNAL_LIMIT = 0.65

PACKAGE_DIR = Path(__file__).resolve().parent          # <repo>/src/ampforge
REPO_ROOT = PACKAGE_DIR.parents[1]                     # <repo>


def _resolve(relative: str) -> Path:
    """Find a bundled data or checkpoint file regardless of the working directory."""
    for base in (Path.cwd(), REPO_ROOT, PACKAGE_DIR):
        candidate = base / relative
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"could not locate {relative!r} (cwd: {Path.cwd()})")


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


def write_fasta(sequences: list[str], path: Path, prefix: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        for i, seq in enumerate(sequences, start=1):
            fh.write(f">{prefix}{i}\n{seq}\n")


# How far the sampling mixture is pulled from the corpus frequencies toward short,
# strongly cationic designs. 0 reproduces the corpus; 1 is the fully tilted mixture.
# This is the dominant lever on the library's Phase 1 profile: a better density model
# moved Frechet distance by 3%, while this moves it several times more.
DEFAULT_TILT = 0.35


def build_prefixes(
    rng: np.random.Generator,
    n: int,
    tilt: float = DEFAULT_TILT,
    length_tilt: float | None = None,
    length_base: np.ndarray | None = None,
) -> np.ndarray:
    """Draw conditioning prefixes for sampling.

    Two considerations shape these mixtures.

    First, the conditioning tokens must be ones the model actually saw. Three of
    the eight activity-token combinations never occur in the training corpus, and
    conditioning on them asks the model to extrapolate, which pushes samples off
    the distribution of real antimicrobial peptides. Only supported tokens are used.

    Second, the library and the top-100 are optimised for different things. The
    computational phase rewards a library that resembles known antimicrobial
    peptides, so the mixture stays close to the corpus. Potency is bought later,
    at top-100 selection, by picking the cationic tail out of a large library
    rather than by shifting the whole library toward it. A modest tilt is still
    applied so that tail is well populated.

    EMPIRICAL_* are the corpus frequencies; TILTED_* favour short, strongly
    cationic designs; the two are blended by TILT.
    """
    # Bucket frequencies of the training corpus. Note these are *not* the frequencies
    # of data/antibacterial.fasta, the set the Phase 1 distributional metrics score
    # against: the corpus runs longer (0.2479 vs 0.2902 in the shortest bucket), the
    # tilt lengthens it further, and generation plus screening lengthens it again, so
    # the shipped library reaches 0.1509 there against a target of 0.2902.
    EMPIRICAL_LENGTH = np.array([0.2479, 0.1556, 0.2274, 0.1128, 0.1090, 0.0804, 0.0668])
    # Bucket frequencies of data/antibacterial.fasta itself, available as an alternative
    # base so the length prior can be aimed at the scored reference rather than the corpus.
    REFERENCE_LENGTH = np.array([0.2902, 0.1747, 0.2552, 0.1097, 0.0763, 0.0593, 0.0347])
    TILTED_LENGTH = np.array([0.1200, 0.2200, 0.2600, 0.2000, 0.1200, 0.0500, 0.0300])
    EMPIRICAL_CHARGE = np.array([0.2301, 0.2444, 0.2539, 0.1587, 0.0864, 0.0264])
    TILTED_CHARGE = np.array([0.0300, 0.1200, 0.2400, 0.2900, 0.2400, 0.0800])
    TILT = tilt

    base_length = EMPIRICAL_LENGTH if length_base is None else np.asarray(length_base, dtype=np.float64)
    length_tilt_value = TILT if length_tilt is None else length_tilt

    length_p = (1 - length_tilt_value) * base_length + length_tilt_value * TILTED_LENGTH
    charge_p = (1 - TILT) * EMPIRICAL_CHARGE + TILT * TILTED_CHARGE
    length_p /= length_p.sum()
    charge_p /= charge_p.sum()

    # activity tokens with real support in the corpus: 0 is a plain antimicrobial
    # annotation, 1 adds antibacterial, 3 adds anti-Gram-negative, 5 anti-Gram-positive.
    # Gram-negative is weighted up because 15 of the panel's 20 strains are Gram-negative.
    act_ids = np.array([0, 1, 3, 5])
    act_p = np.array([0.40, 0.05, 0.40, 0.15])

    length_tok = tk.LEN_OFFSET + rng.choice(len(length_p), size=n, p=length_p)
    charge_tok = tk.CHARGE_OFFSET + rng.choice(len(charge_p), size=n, p=charge_p)
    act_tok = tk.ACT_OFFSET + rng.choice(act_ids, size=n, p=act_p)
    return np.stack([length_tok, charge_tok, act_tok], axis=1).astype(np.int64)


# Low-complexity sequences (homopolymers, two-letter repeats) pass an
# AMP-versus-non-AMP classifier easily but make poor drug candidates: they
# aggregate, they are non-selective, and they collapse the library's diversity.
# They are rejected outright rather than merely down-weighted.
MIN_SEQUENCE_ENTROPY = 1.8
MAX_SINGLE_RESIDUE_FRACTION = 0.42
MAX_REPEAT_RUN = 4


def passes_quality_filter(seq: str) -> bool:
    """Reject degenerate sequences before they can reach the library."""
    n = len(seq)
    counts = np.array([seq.count(a) for a in sorted(set(seq))], dtype=np.float64)
    if counts.max() / n > MAX_SINGLE_RESIDUE_FRACTION:
        return False
    p = counts / n
    if float(-(p * np.log2(p)).sum()) < MIN_SEQUENCE_ENTROPY:
        return False
    run = 1
    for i in range(1, n):
        run = run + 1 if seq[i] == seq[i - 1] else 1
        if run > MAX_REPEAT_RUN:
            return False
    # reject simple two-residue periodic repeats such as (AL)n or (KR)n
    for period in (2, 3):
        if n >= 4 * period and all(seq[i] == seq[i % period] for i in range(n)):
            return False
    return True


def identity_ratio(a: str, b: str) -> float:
    """Levenshtein similarity ratio, matching the starter kit's compliance check."""
    import Levenshtein

    return Levenshtein.ratio(a, b)


# The alignment must cover at least this fraction of the candidate for its identity to
# count. This is the conventional reading of a sequence-identity threshold: the
# denominator is the aligned region, not the full sequence.
MIN_ALIGNMENT_COVERAGE = 0.80


def alignment_identity(queries: list[str], references: list[str]) -> np.ndarray:
    """Highest local-alignment identity of each query against any reference.

    Identity is measured over the aligned region, which is how MMseqs2 reports it and
    how the competition proposal specifies the 80% rule. A full-length ratio is not a
    substitute: a 31-residue design whose 25-residue core matches a natural peptide at
    83% has a full-length identity of only 0.68, so a full-length gate lets it through.
    Every window of the candidate covering at least `MIN_ALIGNMENT_COVERAGE` of it is
    therefore checked, and `partial_ratio` finds the best-matching stretch of the
    reference for each window.

    The dtype is forced because `process.cdist` infers float32 for normalised scorers,
    and float32(80.0) / 100.0 is 0.800000011920929, which sits just above a `> 0.80`
    gate and would reject candidates that are exactly on the limit.
    """
    from rapidfuzz import fuzz, process

    out = np.zeros(len(queries))
    for i, query in enumerate(queries):
        n = len(query)
        shortest = max(MIN_LEN, int(np.ceil(MIN_ALIGNMENT_COVERAGE * n)))
        windows = [
            query[start : start + width]
            for width in range(shortest, n + 1)
            for start in range(0, n - width + 1)
        ]
        scores = process.cdist(
            windows, references, scorer=fuzz.partial_ratio,
            workers=-1, dtype=np.float64,
        )
        out[i] = scores.max() / 100.0
    return out


def max_identity_against(seq: str, references: list[str], limit: float) -> float:
    """Highest identity of `seq` against `references`, short-circuiting past `limit`."""
    import Levenshtein

    n = len(seq)
    best = 0.0
    for ref in references:
        # ratio = 2*matches/(len_a+len_b) can never exceed 2*min/(la+lb)
        m = len(ref)
        if 2.0 * min(n, m) / (n + m) <= limit:
            continue
        r = Levenshtein.ratio(seq, ref)
        if r > best:
            best = r
            if best > limit:
                return best
    return best


def generate_library(
    model: PeptideLM,
    rng: np.random.Generator,
    n_target: int,
    forbidden: set[str],
    batch_size: int,
    temperature: float,
    top_p: float,
    max_batches: int,
    tilt: float = DEFAULT_TILT,
    length_tilt: float | None = None,
    length_base: np.ndarray | None = None,
    verbose: bool = True,
) -> list[str]:
    """Sample until `n_target` unique, valid, novel peptides have been collected."""
    collected: list[str] = []
    seen: set[str] = set()
    for batch in range(max_batches):
        prefixes = build_prefixes(rng, batch_size, tilt=tilt,
                                  length_tilt=length_tilt, length_base=length_base)
        peptides = sample(
            model, prefixes, rng,
            temperature=temperature, top_p=top_p,
            min_len=MIN_LEN, max_len=MAX_LEN,
        )
        for seq in peptides:
            if not (MIN_LEN <= len(seq) <= MAX_LEN):
                continue
            if seq in seen or seq in forbidden:
                continue
            if not passes_quality_filter(seq):
                continue
            seen.add(seq)
            collected.append(seq)
        if verbose:
            print(
                f"  batch {batch + 1}: {len(collected):,}/{n_target:,} unique peptides",
                flush=True,
            )
        if len(collected) >= n_target:
            break
    return collected[:n_target]


# Hard rules for the experimental shortlist, as distinct from the design library.
# The library is a design artifact and is left alone; these apply only to the 100
# peptides that can actually be sent for synthesis.
#
# Cysteine is excluded outright. The competition requires linear peptides with free
# termini and no disulfides, and free thiols in a linear peptide oxidise and form
# intermolecular bridges during synthesis, purification and storage. The official
# HydrAMP starter kit applies the same rule.
#
# Length is capped because stepwise yield in solid-phase synthesis compounds: a
# 40-mer is materially more likely to fail purity QC than a 25-mer, and a failed
# peptide is never retested, so it silently forfeits one of the team's 25 slots.
TOP_MAX_LENGTH = 32
TOP_FORBIDDEN_RESIDUES = frozenset("C")


def passes_shortlist_rules(seq: str) -> bool:
    if len(seq) > TOP_MAX_LENGTH:
        return False
    return not (TOP_FORBIDDEN_RESIDUES & set(seq))


def select_top(
    sequences: list[str],
    scores: np.ndarray,
    references: list[str],
    k: int,
    risk: np.ndarray | None = None,
    verbose: bool = True,
) -> list[str]:
    """Greedily take the best-scoring peptides that pass every gate.

    Gates: synthesis risk below threshold, novelty margin against the reference
    set, and mutual distinctness from the peptides already chosen.
    """
    order = np.argsort(-scores, kind="stable")

    # Local identity is screened in bulk, because a vectorised pass over the whole
    # reference set is far cheaper than one call per candidate. It is evaluated in
    # chunks along the ranked order and extended on demand, so no candidate is
    # unreachable: an earlier version capped it at a fixed prefix and then treated
    # "not yet computed" as "failed the gate", which silently made the rest of the
    # library invisible and blamed the pool size for the shortfall.
    chunk = max(4 * k, 2000)
    local_by_index: dict[int, float] = {}

    def local_identity(index: int, rank: int) -> float:
        if index not in local_by_index:
            block = order[rank : rank + chunk]
            values = alignment_identity([sequences[i] for i in block], references)
            local_by_index.update(
                {int(i): float(v) for i, v in zip(block, values)}
            )
        return local_by_index[index]

    chosen: list[str] = []
    for rank, idx in enumerate(order):
        if len(chosen) >= k:
            break
        seq = sequences[idx]
        if not passes_shortlist_rules(seq):
            continue
        if risk is not None and risk[idx] > MAX_SYNTHESIS_RISK:
            continue
        if local_identity(int(idx), rank) > TOP_LOCAL_IDENTITY_LIMIT:
            continue
        if max_identity_against(seq, references, TOP_IDENTITY_LIMIT) > TOP_IDENTITY_LIMIT:
            continue
        if any(identity_ratio(seq, other) > TOP_INTERNAL_LIMIT for other in chosen):
            continue
        chosen.append(seq)
        if verbose and len(chosen) % 20 == 0:
            print(f"  selected {len(chosen)}/{k} (scanned {rank + 1})", flush=True)
    if len(chosen) < k:
        raise RuntimeError(
            f"only {len(chosen)} of {k} candidates in a library of {len(sequences)} "
            "passed the shortlist gates; loosen a gate or enlarge the library"
        )
    return chosen


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the AMPforge submission.")
    parser.add_argument("--n-sequences", type=int, default=LIBRARY_SIZE)
    parser.add_argument("--top-k", type=int, default=TOP_SIZE)
    parser.add_argument("--seed", type=int, default=42)
    # Temperature 1.10 rather than 1.00: measured over five library variants it improves
    # six of eight Phase 1 metrics, including Frechet distance (0.781 against 0.843) and
    # authenticity (0.835 against 0.828) at the same time. At 1.00 the model concentrates
    # on its high-probability modes, which are the distribution's peaks rather than the
    # distribution, so spreading slightly covers the reference better and simultaneously
    # sits further from individual training sequences.
    parser.add_argument("--temperature", type=float, default=1.10)
    parser.add_argument("--top-p", type=float, default=0.98)
    parser.add_argument("--tilt", type=float, default=DEFAULT_TILT,
                        help="0 samples the corpus distribution, 1 the fully tilted mixture")
    parser.add_argument("--length-tilt", type=float, default=None,
                        help="tilt applied to the length prior alone; defaults to --tilt")
    parser.add_argument("--length-prior", choices=("corpus", "reference"), default="corpus",
                        help="base distribution for peptide length: the training corpus "
                             "(default) or data/antibacterial.fasta, the set the Phase 1 "
                             "distributional metrics score against")
    parser.add_argument("--oversample", type=float, default=1.5)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="defaults to a directory named after the entry point")
    parser.add_argument("--model", type=Path, default=None)
    parser.add_argument("--rankers", type=Path, default=None)
    args = parser.parse_args()

    # the template derives the output directory from the entry-point name, so the
    # same function serves both `generate` and `generate_broad_spectrum`
    entry_point = Path(sys.argv[0]).stem or "generate"
    out_dir = args.out_dir if args.out_dir is not None else Path(entry_point)

    model_path = args.model or _resolve("checkpoint/peptide_lm.npz")
    ranker_path = args.rankers or _resolve("checkpoint/rankers.npz")
    classifier_path = _resolve("checkpoint/amp_classifier.npz")
    reference_path = _resolve("data/antibacterial.fasta")

    print("AMPforge -- AMP Challenge 2027 submission", flush=True)
    print(f"  model      {model_path}", flush=True)
    print(f"  rankers    {ranker_path}", flush=True)
    print(f"  reference  {reference_path}", flush=True)

    references = read_fasta(reference_path)
    forbidden = set(references)
    print(f"  {len(references):,} known antibacterial peptides excluded", flush=True)

    # The wider MarLys set is excluded too, and used for the shortlist's novelty
    # gates. The starter kit's validator only checks the antimicrobial-labelled
    # subset, but the competition additionally screens submitted libraries for exact
    # matches against MarLys to count rediscovered known peptides. Screening against
    # the labelled subset alone left 595 rediscoveries in a 50,000-sequence library,
    # each one costing novelty for no benefit.
    try:
        novelty_reference = read_fasta(_resolve("data/marlys_reference.fasta"))
        before = len(forbidden)
        forbidden |= set(novelty_reference)
        print(f"  {len(novelty_reference):,} MarLys peptides used for the novelty gates "
              f"({len(forbidden) - before:,} further exclusions)", flush=True)
    except FileNotFoundError:
        novelty_reference = references
        print("  WARNING: data/marlys_reference.fasta is missing; the novelty gates "
              "fall back to the smaller antibacterial set", flush=True)

    model = PeptideLM.load(model_path)
    rankers = Rankers.load(ranker_path)
    screen = AmpClassifier.load(classifier_path)
    rng = np.random.default_rng(args.seed)

    n_pool = int(args.n_sequences * args.oversample)
    max_batches = max(8, int(n_pool / args.batch_size * 4))
    print(f"\nsampling a pool of {n_pool:,} candidates", flush=True)
    pool = generate_library(
        model, rng, n_pool, forbidden,
        batch_size=args.batch_size,
        temperature=args.temperature,
        top_p=args.top_p,
        max_batches=max_batches,
        tilt=args.tilt,
        length_tilt=args.length_tilt,
        length_base=(np.array([0.2902, 0.1747, 0.2552, 0.1097, 0.0763, 0.0593, 0.0347])
                     if args.length_prior == "reference" else None),
    )
    if len(pool) < args.n_sequences:
        sys.exit(f"ERROR: only sampled {len(pool)} unique peptides, need {args.n_sequences}")

    # The library and the top-100 are selected by different criteria on purpose.
    # Phase 1 scores the library partly through AMP-classification surrogates, a
    # task learned classifiers do well, so the library is screened by AMP-likeness.
    # Phase 2 measures potency, where those same classifiers were shown to carry no
    # ranking signal, so the top-100 is ordered by the composite score instead.
    print(f"\nscreening {len(pool):,} candidates for AMP-likeness", flush=True)
    screen_scores = screen.logit(pool)
    keep = np.argsort(-screen_scores, kind="stable")[: args.n_sequences]
    keep.sort()
    library = [pool[i] for i in keep]
    print(f"  kept {len(library):,}; "
          f"median AMP probability {np.median(1/(1+np.exp(-screen_scores[keep]))):.3f}",
          flush=True)

    print(f"\nscoring the library for candidate selection", flush=True)
    library_scores = score_candidates(library, rankers)["composite"]

    print(f"\nselecting the top {args.top_k}", flush=True)
    library_risk = synthesis_penalty(library)
    print(f"  {int((library_risk <= MAX_SYNTHESIS_RISK).sum()):,} of {len(library):,} "
          f"library peptides pass the synthesis-risk gate", flush=True)
    top = select_top(library, library_scores, novelty_reference, args.top_k,
                     risk=library_risk)

    write_fasta(library, out_dir / "library.fasta", "ampforge_lib_")
    write_fasta(top, out_dir / "top.fasta", "ampforge_top_")

    stats = score_candidates(top, rankers)
    print(f"\nwrote {len(library):,} peptides -> {out_dir / 'library.fasta'}")
    print(f"wrote {len(top)} peptides -> {out_dir / 'top.fasta'}")
    print("\ntop-100 summary")
    print(f"  mean predicted overall success rate : {stats['success_rate_overall'].mean():.3f}")
    print(f"  mean predicted log10 MIC (Gram-neg) : {stats['mic_gram_negative'].mean():.3f}")
    print(f"  mean predicted log10 safety window  : {stats['log_safety_window'].mean():.3f}")
    print(f"  mean length                         : {np.mean([len(s) for s in top]):.1f}")


if __name__ == "__main__":
    main()
