# AMPforge: method and selection procedure

This document is the submission's required description of how the 50,000-sequence
library and the ranked top-100 were produced, including every filter applied.

## 1. Training data

| Source | Role | License |
|---|---|---|
| MarLys AMP database (MLAMP_db, 103,200 peptides), doi:10.17632/w4hb5grjwb.3 | Language-model corpus | CC-0 |
| GRAMPA, 51,345 MIC measurements (Witten & Witten 2019) | MIC regressors | MIT |
| GRAMPA / Hemolytik hemolysis table | Hemolysis regressor | MIT |
| AMP Scanner v2 benchmark sets, redistributed in the HydrAMP starter kit | AMP-likeness classifier | MIT |
| HydrAMP and AMP-Diffusion starter-kit experimental MIC tables | Held-out evaluation only | MIT |

From MarLys we keep sequences that are annotated antimicrobial, use only the 20
standard amino acids, are 8 to 50 residues long, and are unique. That yields
46,637 training sequences. No sequence was inspected or selected by hand.

## 2. Generative model

A 4.8M-parameter transformer decoder over the amino-acid alphabet: 6 layers,
256-dimensional embeddings, 8 attention heads, 1024-wide feed-forward blocks,
pre-layer-norm. Each training example carries a three-token conditioning prefix:

```
[length bucket] [net-charge bucket] [activity profile] [BOS] a1 ... an [EOS]
```

Length is bucketed at 12, 16, 20, 25, 32 and 40 residues. Net charge is bucketed
at 0, 2, 4, 6 and 9. The activity token is a three-bit combination of the
antibacterial, anti-Gram-negative and anti-Gram-positive annotations. Conditioning
this way steers sampling into a chosen region of design space directly, with no
rejection sampling and no discarded compute.

Training: AdamW, one-cycle schedule, peak learning rate 3e-4, batch size 256,
5% held out for validation, best-validation checkpoint retained.

## 3. Sampling

Conditioning prefixes are drawn from a fixed mixture centred on the region where
natural antimicrobial peptides live: lengths concentrated between 13 and 25
residues, net charge concentrated between +2 and +9, activity tokens weighted
toward Gram-negative annotations because 15 of the panel's 20 strains are
Gram-negative. Nucleus sampling, temperature 1.0, top-p 0.98.

Inference runs in numpy float64 with a preallocated key/value cache, and every
logit vector is snapped to a 1e-5 grid before the softmax. That grid is far
coarser than the ~1e-13 discrepancy a different BLAS library or CPU architecture
can introduce, and far finer than anything that affects sampling quality over a
20-letter alphabet, so the generated library is byte-identical on any platform.

## 3b. Choosing the sampling mixture by measurement

The mixture is the single parameter that most affects the library's computational
score, so it was swept rather than guessed. `TILT` interpolates between the corpus
frequencies (0) and a mixture favouring short, strongly cationic designs (1). Four
values were sampled into 6,000-sequence libraries and scored on both Phase 1
families.

| TILT | FBD | MMD | Recall | Authenticity | AMP classifier p>0.5 | median predicted MIC |
|---|---|---|---|---|---|---|
| 0.00 | 0.948 | 2.214 | 0.777 | 0.820 | 0.783 | 17.7 µM |
| 0.20 | 0.852 | 1.894 | 0.772 | 0.829 | 0.820 | 16.3 µM |
| **0.35** | **0.811** | **1.826** | 0.760 | 0.829 | 0.864 | 14.4 µM |
| 0.55 | 0.818 | 2.020 | 0.742 | 0.832 | 0.911 | 12.8 µM |

Surrogate activity rises monotonically with the tilt, as expected. Distributional
similarity does not: Fréchet distance is minimised at 0.35 and is worse at 0.00,
even though 0.00 reproduces the corpus charge distribution most faithfully (mean
+2.65 against +2.82 for known AMPs). The reason is that Fréchet distance is measured
in ESM-2 embedding space rather than property space, and a mild tilt toward
AMP-typical properties lands the samples in a more AMP-like region of that space
than sampling the raw corpus mixture does.

0.35 is therefore kept. It is the measured optimum for the two metrics on which this
library is furthest ahead of the baseline, and moving to 0.55 would trade 2.4% of
recall for 5.4% of surrogate activity while leaving a gap to the baseline's 0.981
that neither setting closes.

For context on what the other extreme looks like: the official HydrAMP baseline
library scores 0.981 on the AMP classifier, higher than the 0.660 of real known
antimicrobial peptides, because it is filtered through its own classifiers. It pays
for that with a Fréchet distance of 7.67 against our 0.81.

## 4. Hard filters, applied to every candidate

1. Alphabet restricted to the 20 standard amino acids, length 8 to 50, linear,
   free termini, no modifications.
2. Unique within the library.
3. Not identical to any sequence in the challenge reference set.
4. Sequence-quality filter, rejecting degenerate designs: Shannon entropy at
   least 1.8 bits, no single residue above 42% of the sequence, no run of more
   than 4 identical residues, no simple period-2 or period-3 repeat. This filter
   admits 90.3% of known natural antimicrobial peptides, so it removes degenerate
   output without narrowing the natural design space.

## 5. Library selection, 50,000 of 75,000 candidates

The candidate pool is screened by an AMP-likeness classifier and the top 50,000
are kept. The classifier is a small network over 40 physicochemical and
compositional features, trained on the AMP Scanner v2 benchmark, reaching 5-fold
AUROC 0.962 and Matthews correlation 0.818.

This screen is used for the library and deliberately not for the top-100. The
reason is in section 7.

## 6. Top-100 selection

Candidates are first gated, then ordered.

**Gates.** A candidate is excluded outright if it contains a cysteine, exceeds 32
residues, scores above a synthesis-risk threshold, exceeds 0.79 alignment identity
to any sequence in the MarLys reference, or exceeds 0.65 identity to a peptide
already selected.

Alignment identity is measured over the aligned region, by checking every window of
the candidate covering at least 80% of it, because that is how the rule is defined and
a full-length ratio cannot see the case that matters. A 31-residue design whose
25-residue core reproduces a natural peptide at 83% has a full-length identity of only
0.68; MMseqs2 scored exactly such a candidate at 0.827, and another at 0.875. The
limit is held at 0.79 rather than the rule's 0.80 because edit distance and MMseqs2's
substitution-matrix alignment disagree by roughly 0.03 near the boundary, and 45 of 100
candidates sat at exactly 0.800 under the window measure, where that disagreement is
what decides compliance.

Cysteine is excluded on a chemistry argument, not a rule. The competition requires
peptides that are "linear with free termini (no terminal modifications, including
amidation)" and excludes stapled peptides, peptidomimetics and chemically modified
variants; it does not name cysteine. But a free thiol oxidises and forms disulfide
bridges during synthesis, purification and storage, so a cysteine-bearing sequence
risks being assayed as something other than the molecule that was designed. The
official HydrAMP starter kit applies the same exclusion. Length is capped because stepwise yield
in solid-phase synthesis compounds, so a 40-mer is materially more likely to fail
purity QC than a 25-mer. The novelty margin is held below the stated 0.80 rule
because the organizers compute identity with MMseqs2 alignment while the starter
kit uses a Levenshtein ratio, and the two need not agree at the boundary. Mutual
distinctness matters because 25 of the top 50 are drawn at random for synthesis,
so the shortlist should represent 25 different designs rather than one design and
24 near-duplicates.

**Ordering.** Surviving candidates are ranked by a blend of three terms:

| Term | Weight | What it is |
|---|---|---|
| Predicted success rate | 0.45 | Two MIC regressors give a mean and a cross-validated error, which convert to a probability of clearing the 16 µM threshold; the Gram-negative and Gram-positive panels are combined 15:5 to match the strain panel |
| Cationic-amphipathic pharmacophore | 0.35 | Net charge peaking near +8, Eisenberg hydrophobic moment, helical propensity, minus excess bulk hydrophobicity |
| Safety window | 0.20 | Predicted log10 HC50 minus predicted log10 MIC, clipped |

Synthesis risk is deliberately absent from the ordering. Measured against activity
it is mildly anti-correlated (AUROC 0.44), because the traits it penalises — length,
hydrophobicity, cysteine — partly track potency. Including it in the score
corrupted the ranking without serving its actual purpose, which is to protect
experimental slots. It is a gate instead.

The charge term peaks rather than rising monotonically. A monotonic reward drifts
selection into the extreme tail, above +15, where the wet-lab evidence used to
justify the signal contains no peptides at all and where the literature reports
rising hemolysis without further potency gain.

## 7. What the ranking weights are based on

Signals were measured two ways, and an earlier version of this analysis got the
answer wrong, so both the method and the correction are recorded here.

**Test A, generated peptides.** The experimental MIC tables shipped with the
HydrAMP and AMP-Diffusion starter kits: 92 peptides measured on the competition's
own strain panel, none of them used to fit any released model. These are the only
wet-lab-tested peptides available that came out of generative models, which makes
them the closest proxy for what this submission is actually ranking.

| Signal | AUROC, potent on at least one Gram-negative strain | AUROC, potent at the median strain |
|---|---|---|
| Composite as shipped | 0.781 | **0.726** |
| Predicted success rate | **0.818** | 0.680 |
| MIC regressor | 0.781 | 0.651 |
| Pharmacophore | 0.751 | 0.703 |
| Net charge | 0.724 | 0.692 |
| Hydrophobic moment | 0.715 | 0.637 |
| Synthesis penalty, negated | 0.436 | 0.524 |

Bootstrap 95% intervals span roughly ±0.10, so the top signals are not
distinguishable from one another on 92 peptides. The composite is the only signal
in the top two under both label definitions, and its weights were set on the
principle that signals of indistinguishable measured skill deserve comparable
weight, not tuned against this set.

**Test B, natural peptides under a homology-aware split.** 3,293 GRAMPA peptides
with Gram-negative MIC values, clustered at 40% sequence identity by greedy
centroid assignment, with whole clusters held out together.

| Signal | Spearman | AUROC |
|---|---|---|
| MIC regressor, cluster-held-out | **+0.483** | **0.721** |
| Net charge | +0.383 | 0.675 |
| Pharmacophore | +0.367 | 0.670 |
| Hydrophobic moment | +0.251 | 0.616 |
| MIC regressor, random folds | +0.606 | 0.789 |

The gap between the regressor's two rows is leakage: public MIC datasets are full
of analogue series and re-measurements, so random folds place near-identical
sequences on both sides of the split. The honest estimate is 0.483, not 0.606.

**The correction.** An earlier version of this work reported that learned scorers
rank at chance within an AMP-like library, and down-weighted them accordingly.
That was a measurement error, not a finding. It compared the *composite* score,
which then included the synthesis penalty, against an activity label built from
the *median* MIC across strains, where most values sit at the 64 µM assay ceiling.
Both choices suppressed the learned signal. Scored properly, the learned surrogates
and the physicochemical signals both carry real skill and neither dominates, which
is why the shipped score blends them.

## 8. Manual intervention

None. No sequence was chosen, edited, rejected or reordered by hand at any stage.
Every filter and weight above is code in `src/ampforge/`, applied uniformly, and
the whole pipeline is reproducible with `uv run generate`.
