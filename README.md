# AMPforge

A submission to the [AMP Challenge 2027](https://szczurek-lab.github.io/amp-challenge-website/)
(NeurIPS 2026 Competition Track): de novo generative design of linear antimicrobial
peptides, evaluated computationally and then validated in the wet lab against a
20-strain panel of clinically relevant bacteria.

## Method abstract

AMPforge is a conditional character-level transformer language model over peptide
sequences, trained from scratch on a curated set of known antimicrobial peptides and
sampled under explicit control of length, net charge and activity annotation. Each
training sequence is prefixed with three conditioning tokens encoding its length
bucket, its net-charge bucket and its annotated activity profile, so at sampling time
the same prefix steers generation into a chosen region of design space without any
post-hoc rejection sampling.

Candidates are ranked by a composite score that estimates the quantity the competition
actually measures: the fraction of panel strains for which a peptide clears the 16 µM
potency threshold. Two MIC regressors (Gram-negative and Gram-positive panels) and a
hemolysis regressor each predict a mean and carry a cross-validated error estimate,
which converts into a per-panel probability of clearing the threshold and into a
predicted safety window HC50/MIC50. A synthesis-risk penalty down-weights peptides
likely to fail solid-phase synthesis, purification or solubility QC, because a peptide
that fails synthesis is never retested and silently forfeits one of the team's 25
experimental slots.

The final top-100 is selected greedily under two hard constraints: a novelty margin
against the challenge reference set, and a mutual-identity cap among the selected
peptides themselves, so that the 25 candidates drawn at random for synthesis represent
25 genuinely different designs rather than one design and 24 near-duplicates.

## What we measured before choosing the ranker

Ranking signals were tested two ways. First against the experimental MIC tables
shipped with the two baseline starter kits: 92 peptides measured on the
competition's own strain panel, produced by generative models, and never used to
fit any released model. Second against 3,293 natural peptides from GRAMPA under a
homology-aware split that clusters at 40% sequence identity and holds whole
clusters out.

| Signal | Generated peptides, AUROC | Natural peptides, AUROC |
|---|---|---|
| Composite as shipped | 0.781 | — |
| Predicted success rate | 0.818 | — |
| MIC regressor | 0.781 | 0.721 |
| Pharmacophore | 0.751 | 0.670 |
| Net charge | 0.724 | 0.675 |
| Synthesis penalty, negated | 0.436 | — |

Both the learned surrogates and the classical cationic-amphipathic pharmacophore
carry real signal, and on 92 peptides they are not distinguishable from one
another: bootstrap intervals span roughly ±0.10. The shipped score therefore
blends them rather than betting on either, with weights set on that principle
rather than tuned against a sample this small.

Two things were learned the hard way and are worth stating plainly. The synthesis
penalty is mildly anti-correlated with activity, so it belongs in the gates and not
in the ranking. And an earlier version of this analysis concluded that learned
scorers rank at chance, which was a measurement artifact rather than a finding;
`METHOD.md` records what went wrong.

## Quickstart

Requires [`uv`](https://docs.astral.sh/uv/). Model weights are committed under
`checkpoint/`, so no download is needed and the entry point runs offline.

```bash
uv run generate
```

Expect roughly 70 minutes on four CPU cores: about 45 minutes to sample the candidate
pool and about 25 minutes for the shortlist's novelty screen, which compares every
candidate window against a 47,411-sequence reference. There is no GPU path and none is
needed; inference is numpy by design, so that the output is byte-identical everywhere.

This writes two files into `generate/`:

```
generate/
  library.fasta   50,000 unique designed peptides
  top.fasta       the 100 ranked candidates
```

The starter kits name the broad-spectrum entry point `generate_broad_spectrum`; that
name is also registered and writes to `generate_broad_spectrum/` instead. Both resolve
to the same function.

| Flag | Default | Description |
|------|---------|-------------|
| `--n-sequences` | `50000` | Size of the submitted library |
| `--top-k` | `100` | Size of the ranked candidate list |
| `--seed` | `42` | Random seed |
| `--temperature` | `1.0` | Sampling temperature |
| `--top-p` | `0.98` | Nucleus sampling cutoff |
| `--oversample` | `1.5` | Candidate pool size relative to the library |
| `--batch-size` | `2048` | Sequences sampled per batch |

### Reproducibility

Generation is deterministic. Training uses PyTorch, but inference runs entirely in
numpy float64 and every logit vector is snapped to a fixed decimal grid before the
softmax, which erases the floating-point noise that differs between BLAS
implementations and CPU architectures. Repeated runs, on any platform, produce
byte-identical output.

Verify with the challenge's own validator:

```bash
uv run python scripts/verify_submission.py <github-url>
```

## Training data

Every source is public and redistributable, and each is listed with what it was used
for. No proprietary or non-public data was used.

| Source | Used for | License |
|---|---|---|
| [MarLys AMP database](https://doi.org/10.17632/w4hb5grjwb.3), 103,200 peptides | 46,637-sequence antimicrobial corpus for the language model; the 47,411-sequence valid subset is also the novelty reference | CC-0 |
| `Uniprot_0_25_{train,val}.csv` from the [HydrAMP starter kit](https://github.com/szczurek-lab/hydramp-starter-kit), 202,550 usable peptides | general peptide space for pretraining | MIT |
| `unlabelled_positive.csv` from the same kit, 1,908 further known AMPs | pretraining and fine-tuning corpus | MIT |
| `veltri_{positive,negative}.csv` from the same kit, the AMP Scanner v2 benchmark | AMP-likeness classifier | MIT |
| [GRAMPA](https://github.com/zswitten/Antimicrobial-Peptides) (Witten & Witten 2019), 51,345 MIC measurements | Gram-negative and Gram-positive MIC regressors | MIT |
| GRAMPA's cleaned Hemolytik table, 1,209 peptides | hemolysis regressor | MIT |
| `data/antibacterial.fasta` from the [challenge template](https://github.com/szczurek-lab/amp-challenge-2027) | exclusion and novelty reference, copied unmodified | BSD-3-Clause |
| Experimental MIC tables from the HydrAMP and AMP-Diffusion starter kits, 92 peptides | **evaluation only**; never used to fit any released model | MIT |

The last row matters for interpreting the numbers in `RESULTS.md`: those 92 peptides
are the held-out wet-lab test the ranker was judged against, and keeping them out of
every fit is what makes that test meaningful.

No manual curation of individual sequences happened at any stage. Every filter is a
rule in `src/ampforge/`, applied uniformly, and the whole pipeline runs from
`uv run generate`.

Derived tables are committed under `data/` so the pipeline reproduces from this
repository alone, without re-downloading anything.

## Rebuilding from scratch

Nothing here needs to be run to reproduce the submission: `uv run generate` uses the
committed weights. This is the path that produced them.

```bash
uv sync --extra train

# corpora
uv run python scripts/build_corpus.py --mlamp <MLAMP_db.json>
uv run python scripts/build_pretrain_corpus.py --hydramp-training <hydramp-kit>/data/training
uv run python scripts/build_activity_data.py \
    --grampa <grampa.csv> --hemolysis <Cleaned_hemolytic_data.csv>

# language model: pretrain on general peptide space, then fine-tune on AMPs
uv run python scripts/train_lm.py --corpus data/pretrain_corpus.tsv \
    --out checkpoint/peptide_lm_pretrained.npz --device cuda --epochs 25 --lr 4e-4
uv run python scripts/train_lm.py --corpus data/pretrain_corpus.tsv --amp-only \
    --init-from checkpoint/peptide_lm_pretrained.npz \
    --out checkpoint/peptide_lm.npz --device cuda --epochs 6 --lr 1.2e-4

# scorers
uv run python scripts/train_rankers.py
uv run python scripts/train_amp_classifier.py --data-dir <hydramp-kit>/data/training

# the numpy inference path must match the torch model before generating anything
uv run --extra train python scripts/check_parity.py
```

## Evaluation

```bash
# ranking signals against held-out wet-lab data, and against natural peptides
# under a homology-aware split
uv run python scripts/compare_signals.py --ampdiff-mic <...> --hydramp-mic <...>
uv run --extra train python scripts/evaluate_ranking.py

# the multivariate fit that was rejected: a four-feature logistic model on the
# pooled wet-lab data reaches leave-one-out AUROC 0.676, below net charge alone,
# which is why no fitted combination of descriptors is used
uv run python scripts/fit_wetlab_prior.py --ampdiff-mic <...> --hydramp-mic <...>

# Phase 1 metrics, in a separate environment since seqme pulls in torch and ESM-2
pip install "seqme[esm2]"
python scripts/evaluate_seqme.py --library generate/library.fasta \
    --baseline <hydramp-kit>/generate_broad_spectrum/library.fasta \
    --reference data/antibacterial.fasta

# novelty by alignment identity, which the full-length ratio does not catch
python scripts/check_identity_mmseqs.py --query generate/top.fasta \
    --target data/marlys_reference.fasta
```

## Repository layout

```
checkpoint/
  peptide_lm.npz            conditional peptide language model, 4.78M parameters
  amp_classifier.npz        AMP-likeness screen for the library
  rankers.npz               Gram-negative MIC, Gram-positive MIC, hemolysis
data/
  corpus.tsv                46,637 antimicrobial sequences, from MarLys
  pretrain_corpus.tsv       251,095 sequences, general peptide space plus AMPs
  mic_*.tsv, hemolysis.tsv  regressor training tables, from GRAMPA
  antibacterial.fasta       challenge reference set
  marlys_reference.fasta    wider novelty reference, 47,411 sequences
src/ampforge/
  tokenizer.py              vocabulary and the conditioning prefix
  nn.py                     numpy transformer inference, KV cache, determinism
  features.py               physicochemical descriptors
  encode.py                 the 40-feature encoding the scorers share
  rank.py                   scoring, gates, synthesis risk
  generate.py               entry point, filters, top-100 selection
scripts/
  build_*.py                corpus and table construction
  train_*.py                language model, rankers, AMP classifier
  check_parity.py           numpy against torch, run before generating
  compare_signals.py        ranking signals against wet-lab outcomes
  evaluate_ranking.py       homology-aware split on natural peptides
  evaluate_seqme.py         Phase 1 metrics
  check_identity_mmseqs.py  alignment-identity novelty check
  reselect_top.py           re-rank an existing library, for iteration
  screen_tradeoff.py        screening-strength trade-off
  verify_submission.py      the challenge's own validator, copied unmodified
METHOD.md                   what was done and why, with the measurements
RESULTS.md                  every number, including the ones that went against us
SUBMISSION.md               how to file the entry
submitted/                  the exact files uploaded, for hash checking
```

## Disclosure

Disclosed voluntarily, since no AI-assistance policy for this competition was located
either way: the code, the experiments and the documentation in this repository were
written with the assistance of a large language model (Claude), and the commit trailers
record it. The peptides themselves are not LLM output — they are sampled from the
4.78M-parameter transformer defined in `src/ampforge/nn.py` and trained by
`scripts/train_lm.py`. Every number reported here is produced by the pipeline in this
repository and reproducible from it.

## License

MIT. See [LICENSE](LICENSE).
