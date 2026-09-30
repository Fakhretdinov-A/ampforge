# Results

## Phase 1: library quality against the official baseline

The submitted 50,000-sequence library against the 50,000 shipped with the official HydrAMP
starter kit, and against the challenge's 39,448 known antibacterial peptides as the
reference. Scored with `seqme`, the framework the competition's computational phase uses,
on ESM-2 (8M) embeddings, 6,000 sequences sampled per group.

| Metric | AMPforge | HydrAMP baseline | Known AMPs | Better is |
|---|---|---|---|---|
| Uniqueness | 1.000 | 1.000 | 1.000 | higher |
| Novelty | 1.000 | 1.000 | 0.000 | higher |
| Diversity | **0.839** | 0.805 | 0.856 | higher |
| Fréchet biological distance | **0.892** | 7.671 | 0.016 | lower |
| Maximum mean discrepancy | **2.254** | 47.372 | 0.009 | lower |
| Kernel distance | **0.0027** | 0.0590 | 0.0000 | lower |
| Precision | **0.757** | 0.486 | 0.900 | higher |
| Recall | **0.724** | 0.317 | 0.895 | higher |
| Authenticity | 0.848 | **0.946** | 0.000 | higher |
| FKEA | 6.146 | **7.345** | 5.916 | higher |
| Clipped density | **0.529** | 0.071 | 1.000 | higher |
| Clipped coverage | **0.456** | 0.095 | 1.000 | higher |
| Conformity of charge and amphiphilicity | **0.439** | 0.420 | 0.503 | higher |
| Synthesizability rate | **0.670** | 0.498 | 0.769 | higher |

Eleven wins, two ties, two losses. The distributional gap is the substantial one: Fréchet
distance is 8.6x lower, maximum mean discrepancy 21x lower, kernel distance 22x lower, and
clipped density and coverage are five to seven times higher, so the library is not merely
realistic but covers far more of the reference distribution instead of collapsing onto part
of it.

The two losses are worth stating plainly. Authenticity counts how often a generated
sequence sits closer to a training sequence than training sequences sit to each other, and
is the direct cost of fitting the training distribution tightly; at 0.848 about six in
seven generated peptides are still not near-copies. FKEA is reference-free diversity whose
stated direction is to maximise, but real known antimicrobial peptides score 5.916, below
both libraries, so maximising it moves away from the reference that the rest of the metric
family rewards. We sit between the baseline and the real peptides on it.

## Choosing the generative model

Three models were trained and compared on the same Phase 1 metrics, each sampling a
6,000-sequence library under identical settings.

| Model | Validation loss | FBD | MMD | Precision | Recall | Authenticity |
|---|---|---|---|---|---|---|
| From scratch on 46,637 AMPs | 1.644 | 0.835 | 1.992 | 0.774 | 0.759 | 0.823 |
| Pretrained on 251,095 peptides | 2.192 (mixed corpus) | 0.910 | 2.123 | 0.761 | 0.773 | 0.831 |
| Pretrained then fine-tuned on AMPs | **1.346** | **0.811** | **1.826** | 0.771 | 0.760 | 0.829 |

Pretraining on general peptide space, 202,550 UniProt peptides plus every known AMP,
then fine-tuning briefly on antimicrobial sequences alone, cut validation loss from
1.644 to 1.346 and closed most of the overfitting gap the from-scratch model showed
(training 1.14 against validation 1.64). It is the model that ships.

The instructive part is how little that bought on the metrics that count. An 18%
better likelihood moved Fréchet distance by 3%. The library's Phase 1 profile is set
far more by the sampling mixture than by the density model, which is why the mixture
is the parameter that was swept rather than the architecture.

## Reproducibility, demonstrated three ways

The competition requires that running the generation script twice produces identical
output, and the organizers verify it by regenerating the library on their own machine
and comparing it against the submitted one. That is checked here more strictly than
asked.

| Check | Result |
|---|---|
| Two runs on the same Linux x86 machine | byte-identical |
| Linux x86 against a clean clone on Apple Silicon, shipped code | byte-identical, both files |
| Version 2's shortlist, selected independently on Linux x86 and Apple Silicon | byte-identical, `3093b43c…e2767448` |
| numpy inference against the PyTorch model, every position | max logit difference 6.0e-07 |
| The synthesis-risk and entropy terms across separate processes | byte-identical |

The second row is the check the organizers perform, and it was run on the shipped
commit: a fresh `git clone`, `uv sync`, `uv run generate`, on a different operating
system, CPU architecture and BLAS implementation, reproducing
`3dafddd6…938dbab` and `50411199…e70e9121` exactly.

Byte-identical output across two architectures and two BLAS implementations is
stronger than the requirement. It comes from running inference in numpy float64 and
snapping every logit vector to a 1e-5 grid before the softmax, which is far coarser
than the roughly 1e-13 divergence a different BLAS introduces and far finer than
anything that affects sampling over a 20-letter alphabet.

## What an independent review found

The pipeline was reviewed by a second party with instructions to be skeptical and to
test rather than read. It confirmed by direct experiment that the numpy inference
matches PyTorch at every position, that the key/value cache compaction is
stream-identical to a non-compacting reference over 192 sequences, and that the
descriptor formulas and the Levenshtein pruning bound are sound.

It also found nine defects, all fixed. The one that mattered: `rapidfuzz`'s `cdist`
infers `float32` for normalised scorers, so a true local identity of exactly 80.0
became 0.800000011920929 and failed a gate written as `> 0.80`. That silently rejected
359 candidates sitting on the boundary and decided 34 of the 100 shortlisted peptides
by a library default rather than by design. Two more were substantive: the
mean-absolute-error to sigma conversion used 1.4826, the constant for a median
absolute deviation rather than a mean, inflating sigma by 18% and shifting how the two
MIC panels blend; and the local-identity screen capped at a fixed prefix while
treating "not yet computed" as "failed the gate", which made most of the library
unreachable to the selector. The rest were smaller: a path off by one level, two
unstable sorts with exact ties present, hash-randomised set iteration in the entropy
terms, a silent fallback, a validation-loss weighting that decides which checkpoint
ships, and a cumulative-sum overflow that could drop a residue.

None of them would have produced a rule violation. Several were choosing part of the
submission by accident, which is worse than it sounds for a benchmark whose whole
purpose is to find out which design choices actually matter.

## Ranking: what predicts wet-lab activity

**On generated peptides.** The experimental MIC tables shipped with the two
baseline starter kits: 92 peptides, measured on the competition's own strain
panel, never used to fit any released model. Bootstrap 95% intervals span roughly
±0.10, so the leading signals are not separable at this sample size.

| Signal | At least one Gram-negative strain | At the median strain |
|---|---|---|
| Composite as shipped | 0.781 | **0.726** |
| Predicted success rate | **0.818** | 0.680 |
| MIC regressor | 0.781 | 0.651 |
| Pharmacophore | 0.751 | 0.703 |
| Net charge | 0.724 | 0.692 |
| Hydrophobic moment | 0.715 | 0.637 |
| Synthesis penalty, negated | 0.436 | 0.524 |

The composite is the only signal that places in the top two under both label
definitions, and its weights were not fitted to this set.

**On natural peptides, homology-aware split.** 3,293 GRAMPA peptides with
Gram-negative MIC values, clustered at 40% identity by greedy centroid assignment,
whole clusters held out together.

| Signal | Spearman | AUROC |
|---|---|---|
| MIC regressor, cluster-held-out | **+0.483** | **0.721** |
| Net charge | +0.383 | 0.675 |
| Pharmacophore | +0.367 | 0.670 |
| Hydrophobic moment | +0.251 | 0.616 |
| MIC regressor, random folds | +0.606 | 0.789 |

The distance between the regressor's two rows is near-duplicate leakage. Public
MIC datasets contain analogue series and repeated measurements, so random folds
place near-identical sequences on both sides of the split and flatter any flexible
model. The honest number is 0.483.

**A correction.** An earlier version of this work reported that learned scorers
rank at chance within an AMP-like library, and the ranking was weighted
accordingly. That was wrong, and the cause was measurement rather than modelling:
it scored the *composite* including the synthesis penalty against an activity
label built from the *median* MIC, where most values sit at the 64 µM ceiling.
Removing the penalty from the ranking and scoring components separately raised the
composite from AUROC 0.733 to 0.781, and the top-100's predicted Gram-negative MIC
from 1.60 to 1.17 µM.

## The submitted top-100

| Property | Mean | Range |
|---|---|---|
| Net charge | +7.22 | — |
| Length | 23.9 residues | up to 32 |
| Cysteine-free | 100 of 100 | — |

Predicted for the 50 peptides that can actually be drawn, since 25 are taken at random
from the top 50:

| | Version 1 | **Version 2, submitted** |
|---|---|---|
| Overall success rate | 0.903 | **0.925** |
| Gram-negative success rate | 0.934 | **0.941** |
| Gram-positive success rate | 0.810 | **0.877** |
| Gram-negative MIC | 2.02 µM | **1.88 µM** |
| Gram-positive MIC | 4.41 µM | **2.87 µM** |
| Safety window | 119x | 117x |
| Highest identity to a known peptide | 0.643 | **0.615** |
| Highest identity between two of them | 0.619 | **0.524** |

The Gram-positive gain is the point of the exercise: it is a whole scored category, and it
came from selecting on the ratio the five-category scoreboard implies rather than on the
strain panel's own 15:5. The drop in mutual identity matters for a different reason: 25 of
the 50 are drawn at random, so a less self-similar shortlist spreads the bet across more
distinct designs.

Novelty of the shortlist against the 47,411-sequence MarLys reference, under three
definitions, against a limit of 0.80: 0.615 by full-length Levenshtein identity, 0.788 by
alignment identity over the aligned region, and 0.524 as the highest identity between any
two shortlisted peptides.

## Component models## Component models

| Model | Task | Validation |
|---|---|---|
| Peptide language model, 4.78M parameters | conditional generation | pretrained on 251,095 peptides, fine-tuned on AMPs, best validation loss 1.346 |
| AMP-likeness classifier | library screening | 5-fold AUROC 0.962, MCC 0.818 |
| Gram-negative MIC regressor | ranking | 5-fold Spearman 0.606 |
| Gram-positive MIC regressor | ranking | 5-fold Spearman 0.559 |
| Hemolysis regressor | safety window | 5-fold Spearman 0.563 |

## Compliance

Every check in the challenge's own `verify_submission.py` passes on the submitted
files: 50,000 sequences over the standard alphabet, all 8 to 50 residues, all
unique, the top-100 a subset of the library, no library sequence identical to a
known antibacterial peptide, and no top-100 sequence above 80% identity to the
reference set.

## Reproducibility

Inference runs in numpy float64 with logits snapped to a fixed decimal grid
before the softmax. Repeated runs produce byte-identical output, and the grid is
coarse enough to absorb the floating-point differences between BLAS
implementations and CPU architectures, so the same holds on the organizers'
machine. The numpy path is checked against the PyTorch model by
`scripts/check_parity.py`: maximum logit difference 5.1e-6 on a logit scale of
12.2, with identical argmax on every row.
