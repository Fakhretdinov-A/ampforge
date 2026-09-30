# Kaggle Writeup — copy-paste source

Team: iMakAI · Track: Computational

---

## Title  (80 char limit)

```
AMPforge: compact conditional peptide LM with byte-identical reproducibility
```
75 characters.

## Subtitle  (140 char limit)

```
50,000 rule-conformant peptides from a 4.78M-parameter conditional transformer, regenerating byte-for-byte on any platform.
```
122 characters.

---

## Project Description

### Abstract

AMPforge generates the library from a **4.78M-parameter character-level conditional
transformer** (6 layers, d_model 256, 8 heads, d_ff 1024) trained on 251,095 peptides in two
stages: the broad peptide corpus, then the measured-active subset. Each sequence is drawn
behind a three-token conditioning prefix — a length bucket, a net-charge bucket and an
activity profile — so length and charge are steered directly rather than filtered for
afterwards. Sampling runs at temperature 1.10, top-p 0.98, with the conditioning mixture held
close to corpus frequencies (tilt 0.35): potency is bought at shortlist selection out of a
large library, not by shifting the whole library toward the cationic tail.

Inference is **pure NumPy in float64**, with logits rounded to a 1e-5 grid. That rounding
absorbs the ~1e-13 BLAS divergence between platforms, so the 50,000-peptide library and the
top-100 regenerate **byte-for-byte identically on Linux x86-64 and Apple Silicon** — verified
by running both and comparing SHA-256. The NumPy path was checked against the PyTorch model
position by position (max deviation 5.98e-07) and the batch-compaction logic against a
reference implementation (0 mismatches over 192 sequences).

A pool of 75,000 candidates is screened to 50,000 by an AMP-likeness classifier (keep rate
0.667, median AMP probability 0.997). The shortlist is ranked by a composite of
**0.60 surrogate potency + 0.30 pharmacophore geometry + 0.10 predicted safety window**.

The composite is the only part of this we can validate against ground truth, so we did:
against the 92 peptides with measured MICs shipped in the two starter kits it reaches
**AUROC 0.813** (potent against at least one Gram-negative strain) and **0.728** (potent at
the median Gram-negative strain). For comparison, APEX-pathogen — the activity ensemble the
organisers vendor in their own AMP-Diffusion starter kit — reaches 0.724 and 0.634 on the
same peptides, so we kept our own scorer.

Library profile on the seqme panel against `data/antibacterial.fasta` at n = 9,000:
uniqueness 1.000, novelty 1.000, diversity 0.839, FBD 0.889, MMD 2.263, precision 0.727,
recall 0.740, authenticity 0.849.

### Data description

**Training data (251,095 sequences)**, all public:

- **MarLys AMP database** (MLAMP_db, Mendeley `doi:10.17632/w4hb5grjwb.3`, CC-0) — 46,637
  sequences after filtering to the standard 20 residues and 8–50 length.
- **UniProt general peptides** — 202,550 sequences, via `Uniprot_0_25_{train,val}.csv` from
  the HydrAMP starter kit.
- **Additional known AMPs** — 1,908 sequences from `unlabelled_positive.csv` in the same kit
  (originating in dbAMP, DRAMP and AMP Scanner).

**Scorer training data:** GRAMPA (`grampa.csv`) for the per-Gram-class MIC regressors;
`Cleaned_hemolytic_data.csv` for the hemolysis model.

**Disclosure.** The competition's own reference set, `data/antibacterial.fasta` (39,448
sequences), is wholly contained in the MarLys corpus, so those sequences are present in
training data. Every one of them is excluded from the submitted library by exact match —
0 of 50,000 overlap — and the library's novelty on the seqme panel is 1.000.

**Computational filters, in order.** Standard 20 amino acids only; length 8–50; exact-match
exclusion against all 39,448 reference sequences; deduplication; AMP-likeness screen from
75,000 candidates to 50,000. **No manual curation of sequences at any point** — the pipeline
is a single deterministic entry point (`uv run generate`).

### Top candidates selection procedure

The top-100 is selected from the 50,000-peptide library by descending composite score, under
hard constraints applied at selection time:

- **cysteine-free** (100/100), keeping the shortlist unambiguously linear as synthesised;
- length ≤ 32 residues (realised 13–32, mean 23.9);
- Shannon entropy ≥ 1.8, no single residue above 42% of the sequence, no homopolymer run
  above 4 — these reject degenerate sequences the surrogate scores well but a synthesiser
  cannot handle;
- **identity to any known AMP below every definition we could apply**: 0.615 full-length
  Levenshtein ratio, 0.788 alignment identity over windows covering ≥80% of the candidate,
  0.727 by MMseqs2 — all under the 0.80 limit, which is enforced at a stricter internal
  threshold of 0.72;
- internal pairwise identity ≤ 0.65, selected greedily, so the shortlist is not 100 variants
  of one motif;
- Gram-negative weighted 12:8 over Gram-positive in the scoring mixture, reflecting that 15
  of the panel's 20 strains are Gram-negative.

Shortlist predictions: mean overall success rate 0.889, mean predicted log10 MIC against
Gram-negatives 0.204 (≈1.6 µM), mean predicted log safety window 2.097.

### What we tested and rejected

Sixteen hypotheses were measured; most were rejected, and the rejections are documented with
numbers in `EXPERIMENTS.md`. The four that mattered most:

| Tested | Outcome |
|---|---|
| APEX-pathogen as the activity scorer | rejected — AUROC 0.724 vs our 0.813 on measured MICs |
| cysteine excess as the cause of our FBD gap | rejected — removing it moves FBD by 8% while costing 20% of recall and 55% of MMD |
| ESM-2 150M fine-tuned as the generator | rejected — 16× our FBD, 36× our MMD, recall 0.259; 148M parameters memorise 8,606 active sequences rather than generalise |
| length prior aimed at the reference distribution | real and priced: FBD −18%, MMD −35%, but the shortlist's predicted Gram-negative MIC moves 1.60 → 4.18 µM, because a 12-residue peptide cannot carry the +8 net charge the ranking rewards. Declined on the trade; retained in the code as `--length-prior`/`--length-tilt`, defaulting to the submitted behaviour |

**Known weakness, stated plainly.** The distributional metrics are where this library is
weakest: FBD 0.889 and MMD 2.260, against a lower bound of 0.010 and 0.007 — what a
9,000-sequence subsample of the reference set scores against the full reference set, an
overlapping comparison and so not an achievable target. We traced the gap to the length prior
being based on the training corpus rather than on the scored reference set, priced the
correction end to end, and declined it for the reason in the table above. We would rather
report a measured weakness with its cause identified than leave it for the panel to find.

---

## Project Links

- GitHub repository: https://github.com/Fakhretdinov-A/ampforge

## Project Files

| File | Size | SHA-256 |
|---|---|---|
| `library.fasta` | 2.1 MB | `5b505097dde766a72e954f767c241c85a86ed35c0497465d73bca1cf9e1ed43e` |
| `top.fasta` | 4.2 KB | `3093b43c161ce4bca70e675405b464937af125762e3c281585fe2879e2767448` |
