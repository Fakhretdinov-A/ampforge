# How to submit

Everything is built and verified. What follows is the exact sequence to file the
entry. Deadline as the Kaggle form states it: **1 October 2026, 01:00 GMT+3**, which is
30 September 22:00 UTC. An earlier revision of this file read "anywhere on Earth, 2 October
12:00 UTC", taken from the competition call rather than from the submission form; the form
is the operative deadline and it is roughly two days earlier.

## Step 0, how the entry is actually filed

The entry is a **Kaggle Writeup** on the AMP Challenge community hackathon page, not a
repository the organisers clone. Team **iMakAI**, computational track (the only track), due
1 Oct 2026 at 01:00 GMT+3.

An earlier revision of this file claimed the rules require an institutional email address and
reject free webmail. That claim was never verified against the rules and should not have been
stated as though it were; the team is registered and the question is moot. It is recorded here
rather than quietly deleted because it was wrong in a way worth leaving visible.

The form has six checklist items: title, subtitle, track (auto-selected), project description,
project links, project files. `submitted/KAGGLE_WRITEUP.md` holds the exact text for the first
four, sized to the 80- and 140-character limits.

## Step 1, the repository link

Published at **https://github.com/Fakhretdinov-A/ampforge** — public, so the panel can open it
without being granted access. This goes in *Project links*.

## Step 2, the files to upload

| File | What it is |
|---|---|
| `submitted/library.fasta` | the 50,000-sequence library |
| `submitted/top.fasta` | the ranked top-100 |

These are tracked in the repository so their hashes can be checked without a
70-minute regeneration. `uv run generate` writes the same two files into
`generate/`, which is not tracked.

SHA-256, so you can confirm nothing was altered in transit:

```
library.fasta  5b505097dde766a72e954f767c241c85a86ed35c0497465d73bca1cf9e1ed43e
top.fasta      3093b43c161ce4bca70e675405b464937af125762e3c281585fe2879e2767448
```

## Step 3, the method abstract

Paste this where the Kaggle form asks for a method summary.

> AMPforge is a conditional character-level transformer language model over
> peptide sequences. It is pretrained on 251,095 peptides spanning general peptide
> space and known antimicrobial peptides, then fine-tuned on antimicrobial
> sequences alone. Each sequence carries a three-token conditioning prefix encoding
> its length bucket, net-charge bucket and activity annotation, so sampling is
> steered directly rather than by rejection.
>
> The 50,000-sequence library is screened by an AMP-likeness classifier (5-fold
> AUROC 0.962). The top-100 is gated on synthesis risk, cysteine content, length,
> and two novelty measures, then ranked by a blend of a learned success-rate
> surrogate, the cationic-amphipathic pharmacophore, and a predicted safety window.
> The weights follow from measuring both signal families against held-out wet-lab
> MIC data and against natural peptides under a homology-aware split; neither
> family dominated, so neither was allowed to.
>
> Generation runs in numpy float64 with logits snapped to a fixed grid before the
> softmax. The 50,000-sequence library is byte-identical when generated on Apple
> Silicon and on Linux x86, which is stronger than the reproducibility the
> competition asks for.

## Step 4, the training-data statement

> All training data is public and redistributable. The language model is trained on
> the MarLys AMP database (CC-0) and on general peptides and additional AMPs
> redistributed with the HydrAMP starter kit (MIT). The MIC and hemolysis
> regressors are trained on GRAMPA (MIT). The AMP-likeness classifier is trained on
> the AMP Scanner v2 benchmark sets. The experimental MIC tables shipped with the
> HydrAMP and AMP-Diffusion starter kits were used only to evaluate the ranker and
> never to fit any released model. No proprietary or non-public data was used, and
> no sequence was selected or edited by hand at any stage.

## Step 5, disclose AI assistance

The competition permits it and requires disclosure.

> Prepared with the assistance of a large language model (Claude). All design
> decisions, evaluations and reported numbers are produced by the pipeline in the
> repository and are reproducible from it.

## What has been checked

Every check below was run independently on Linux x86 and on Apple Silicon, with
matching results.

| Check | Result |
|---|---|
| Library size, alphabet, length, uniqueness | 50,000, ACDEFGHIKLMNPQRSTVWY, 8 to 50, all unique |
| Top-100 is a subset of the library, no duplicates | pass |
| No library sequence identical to the challenge reference set | pass |
| Top-100 full-length identity to the reference set | maximum 0.615, limit 0.80 |
| Top-100 alignment identity over the aligned region | maximum 0.788, our gate 0.79 |
| Rediscovered MarLys peptides in the library | 0 of 50,000 |
| Highest identity between two shortlisted peptides | 0.524 |
| Cysteine-free | 100 of 100 |
| Longest candidate | 32 residues |
| Two generations byte-identical, same machine | verified twice |
| Generation byte-identical across Linux x86 and Apple Silicon | verified for the library and, independently, for the shortlist |
| numpy inference against the PyTorch model | maximum logit difference 6.0e-07 |
| Nine defects from an independent code review | all fixed, see RESULTS.md |
