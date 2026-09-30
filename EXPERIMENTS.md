# Improvement program

The submitted solution is preserved on the `submission-baseline` branch and at the
`submission-2026-10-01` tag, and its two files are tracked under `submitted/`. Nothing
below can lose it. Every hypothesis is recorded with its result, including the ones
that fail, because a negative result is the only thing that stops the same idea being
tried twice.

Where the ranking is decided:

* **Phase 1** gates entry. Four metric families, aggregation weights held out until it
  closes: surrogate activity prediction, sequence-level metrics, distributional
  similarity in protein-language-model embedding space against *two* reference sets,
  and property-distribution conformity.
* **Phase 2** awards the five categories. 25 peptides drawn at random from the top 50,
  MIC across 20 strains, HC50 against human red blood cells.

## Measurement gaps, before changing anything

| # | Hypothesis | Status | Result |
|---|---|---|---|
| M1 | Property conformity of charge and amphiphilicity is a named Phase 1 metric and has never been measured | done | conformity 0.452 against the baseline's 0.420, ceiling 0.503; KL of charge 1.87 against 3.64, of amphiphilicity 0.018 against 0.042. We lead on all three |
| M2 | Precision and recall are specified against two reference sets; only the known-AMP one was measured | done | against generic peptides: Fréchet 3.41 against 10.98, precision 0.553 against 0.406, recall 0.505 against 0.262. We lead |
| M3 | KID, clipped density, clipped coverage and FKEA are in the evaluation framework and were never run | done | KID 0.0032 against 0.0591, clipped density 0.564 against 0.078, clipped coverage 0.504 against 0.101. **FKEA 6.41 against 7.37, a loss** |
| M4 | The library's surrogate-activity profile has only been judged by our own classifier | **closed** | the baseline scores higher than us on an outside classifier, but real known AMPs score lowest of all; see below |
| M5 | Synthesizability rate is a named metric; ours is unmeasured against the baseline | done | 0.666 against the baseline's 0.510, but known AMPs reach 0.770, so there is headroom |

### What the measurements changed

Across eleven Phase 1 metrics measured so far, the submitted library leads the official
baseline on nine, ties on two and loses on two: authenticity (0.837 against 0.946) and
FKEA diversity (6.41 against 7.37).

Both losses point the same way, at diversity, and there is room to pay for them. The
Fréchet distance lead is eightfold, so trading some distributional tightness for
diversity is affordable in a way it would not be for a team that was merely level.

FKEA deserves a caveat: it is reference-free and its stated direction is to maximise,
yet real known antimicrobial peptides score 5.91, *below* both libraries. Maximising it
therefore moves away from the reference distribution that the other half of the metric
family rewards. We are between the baseline and the real peptides on it, which is a
defensible place to be.

## Improvement hypotheses

| # | Hypothesis | Expected effect | Status | Result |
|---|---|---|---|---|
| H1 | A hemolysis model on protein-language-model embeddings beats 40 hand features | selectivity category, currently the weakest component at Spearman 0.563 | **confirmed, and worse than expected** | see below |
| H2 | A MIC model on the same embeddings beats hand features under a homology split | top-100 quality, so Phase 2 across four categories | **rejected for adoption** | the gain does not survive repetition over seeds |
| H3 | Sampling temperature trades authenticity against Fréchet distance | authenticity is the only Phase 1 metric we lose | **confirmed, and there is no trade** | temperature 1.10 improves six of eight metrics; adopted |
| H4 | A library pooled from several variants covers the reference better | recall, coverage, FKEA | **rejected** | pooling is worse than the best single setting on Fréchet, recall and coverage |
| H5 | Conditioning on continuous properties beats bucketed conditioning | control over the sampled distribution | | |
| H6 | Raising the synthesizability rate of the library costs little elsewhere | a named Phase 1 metric | **rejected** | a synthesizability-first screen does not raise the rate (0.689 against 0.692) and worsens Fréchet distance |
| H7 | Selection weighted 15:5 toward Gram-negative maximises the broad-spectrum category but concedes the Gram-positive one, and the five categories carry equal weight | one of five categories | **confirmed and adopted** | selection moved to 12:8 |

## H1 in detail: the shipped hemolysis model does not transfer at all

| Features | Random folds | Homology folds | Held-out wet-lab HC50, 30 peptides |
|---|---|---|---|
| 40 hand descriptors, what ships | 0.587 | 0.398 | **-0.061** |
| ESM-2 embeddings, 35M parameters | 0.621 | **0.465** | — |
| Both | 0.602 | 0.464 | **+0.273** |

The shipped model reported Spearman 0.563 under random folds. Under folds that keep
sequence-similar peptides together it falls to 0.398, and on 30 peptides with measured
HC50 that were in no training set it is **-0.061**, which is to say it carries no
signal at all where it is actually used. Adding ESM-2 embeddings takes that to +0.273.

The safety-window term carries weight 0.20 in the shipped composite. On this evidence
that weight is spending a fifth of the ranking on noise.

A second hypothesis was tested and **rejected**: restricting the training set to free
termini and L stereochemistry, to match the competition's chemistry, since 1,445 of the
2,669 Hemolytik records are C-terminally amidated and amidation changes both charge and
hemolysis. It halves the data to 593 peptides and performs worse (homology Spearman
0.442 against 0.464, wet-lab +0.221 against +0.273). The sample-size loss outweighs the
chemistry match.

The open question this raises is whether the same holds for MIC, which drives four of
the five categories. If it does, the pipeline has a case for depending on a protein
language model at inference, which would cost the pure-numpy determinism that the
submission currently guarantees. That experiment is running.

## H2 in detail: the protocols disagree, and the relevant one favours embeddings

MIC drives four of the five categories, so this is the decisive experiment.

**On natural peptides, folds that keep sequence-similar peptides together:**

| Features | Gram-negative rho | AUROC | Gram-positive rho | AUROC |
|---|---|---|---|---|
| 40 hand descriptors | **0.493** | **0.731** | **0.420** | **0.715** |
| ESM-2, 35M parameters | 0.470 | 0.715 | 0.387 | 0.703 |
| Both | 0.486 | 0.727 | 0.396 | 0.709 |

Hand descriptors win. Embeddings win only under random folds (0.646 against 0.607),
which is the protocol that leaks near-duplicates, so on natural peptides the embedding
advantage is largely recognition of neighbours.

**On the 92 wet-lab peptides, which came out of generative models and were measured on
the competition's own strain panel:**

| Features | Spearman | AUROC |
|---|---|---|
| 40 hand descriptors | 0.399 | 0.725 |
| ESM-2 | 0.576 | 0.815 |
| Both | **0.643** | **0.862** |

Here embeddings win decisively, and the combination wins by more than either alone.

The two results are not in conflict; they are about different populations. Hand
descriptors encode the mechanism of membrane disruption directly, which is enough to
rank *natural* peptides. Generated peptides sit off that distribution, and there the
learned representation carries information the descriptors do not. The same pattern
appeared for hemolysis, where embeddings improved held-out transfer from -0.06 to +0.27.

Since the peptides we rank are generated ones, the 92-peptide result is the relevant
measurement even though it has a tenth of the sample. The gap, 0.862 against 0.725, is
roughly 2.7 standard errors at that sample size.

**Then the gain evaporated under repetition.** Both figures above came from a single fit
at one seed. Repeated over five seeds on the same 92 peptides:

| Features | AUROC, mean over seeds | sd | Spearman | sd |
|---|---|---|---|---|
| 40 hand descriptors | 0.782 | **0.016** | 0.501 | **0.006** |
| Hand + ESM-2 8M | 0.810 | 0.030 | 0.535 | 0.044 |
| Hand + ESM-2 35M | 0.818 | 0.022 | 0.582 | 0.043 |

The real gap is 0.782 against 0.818, about 0.036 of AUROC, against a seed-to-seed spread
of 0.016 to 0.022 and a sampling standard error near 0.05 at this sample size. The
single-seed 0.862 was a lucky draw, and so was the single-seed 0.725 it was compared
against.

**Rejected.** A gain of 0.036 that is not separable from noise does not justify torch
and transformers as runtime dependencies, vendored model weights, and the loss of the
bit-for-bit determinism the submission currently guarantees. The descriptor model is
also markedly more stable: its Spearman varies by 0.006 across seeds against 0.043 for
the embedding models, so the embedding model's ranking depends heavily on
initialisation, which is the last thing a one-shot 25-peptide draw wants.

The lesson generalises to the earlier hemolysis result, which was also single-seed and
is being re-measured the same way.

## An unplanned finding: the safety-window term was spending a fifth of the ranking on nothing

Since the hemolysis model transfers weakly, the obvious next question was what each term
of the composite is actually worth. The models are fixed, so this needed no training and
has no seed dependence. On the 92 wet-lab peptides:

| Term, alone | AUROC | 95% bootstrap CI |
|---|---|---|
| Predicted success rate | **0.817** | 0.722 to 0.900 |
| Pharmacophore | 0.751 | 0.640 to 0.856 |
| Safety window | **0.532** | 0.412 to 0.646 |

The weight surface falls monotonically as the safety-window weight rises: 0.816 at zero,
0.807 at 0.10, 0.781 at 0.20, 0.746 at 0.30. The shipped 0.45/0.35/0.20 scored 0.787;
the same ratio with the safety window removed scored 0.809.

That measurement is not entirely fair to the term, because the safety window is not meant
to predict activity, it is meant to serve the selectivity category. But the trade it
offers is a measurable loss in activity ranking for a selectivity gain that the hemolysis
experiment could not demonstrate.

**Changed to 0.60 surrogate, 0.30 pharmacophore, 0.10 safety window**, which scores 0.816
against the shipped 0.787. Three independent measurements point the same way and none was
fitted to this sample: the univariate AUROCs above, the homology-split result on 3,293
natural peptides (0.731 for the regressor against 0.670 for the pharmacophore), and the
monotone surface. The safety window keeps a token weight because selectivity is a scored
category and a peptide with a ruinous safety window is genuinely bad, not because the
model behind it has earned more.

The gain, +0.029, sits inside a confidence interval roughly 0.18 wide. It is adopted
because every measurement agrees on the direction, not because this sample proves it.

## H7: optimising the panel is not the same as optimising the scoreboard

The 20-strain panel is 15 Gram-negative and 5 Gram-positive, and the selection score
weights the two predicted success rates 15:5 to match it. That is correct for the
broad-spectrum category, which is scored across all 20 strains.

But there are five categories and each counts once: broad-spectrum, Gram-positive,
Gram-negative, multi-drug-resistant, and selectivity. The Gram-positive category is
scored on the Gram-positive panel alone, and is worth exactly as much as the
broad-spectrum one. The submitted shortlist has a predicted Gram-negative success rate
of 0.949 against 0.798 for Gram-positive, so a 15:5 selection weight is spending the
Gram-positive category to win the Gram-negative one twice.

Crucially there is no way to allocate peptides to categories: 25 are drawn at random from
the top 50 and every drawn peptide contributes to every category's team average. So the
only lever is the balance inside the selection score itself.

Re-scoring the submitted library's 23,211 shortlist-eligible sequences under different
blends, and reporting the predicted profile of the top 50, which is the set the 25 are
drawn from. The overall rate is always reported on the real 15:5 panel, so the first
column shows what a more even selection weighting costs the broad-spectrum category.

| Selection blend | SR overall | SR Gram-neg | SR Gram-pos | MIC gn | MIC gp | Window |
|---|---|---|---|---|---|---|
| 15:5, as shipped | 0.969 | 0.983 | 0.927 | 0.89 µM | 2.06 µM | 278x |
| **12:8** | **0.967** | **0.975** | **0.942** | 1.08 µM | 1.75 µM | 296x |
| 10:10 | 0.965 | 0.970 | 0.947 | 1.20 µM | 1.65 µM | 281x |
| 8:12 | 0.957 | 0.957 | 0.957 | 1.45 µM | 1.46 µM | 297x |
| 5:15 | 0.946 | 0.941 | 0.963 | 1.75 µM | 1.31 µM | 257x |

Moving to 12:8 buys 0.015 of Gram-positive success rate for 0.002 of overall and 0.008 of
Gram-negative, and improves the predicted safety window as a side effect.

**12:8 is not a fitted choice.** It is close to the ratio the scoreboard implies. Each of
the five categories counts once. Broad-spectrum depends on the Gram-negative panel for
15 of 20 strains and the Gram-positive panel for 5. The Gram-negative and Gram-positive
categories depend wholly on one each. The multi-drug-resistant category is scored on 8
isolates that are 5 Gram-negative and 3 Gram-positive, far more even than the full panel.
Adding that up gives 2.375 units of demand for Gram-negative performance against 1.625
for Gram-positive, a ratio of about 1.46 to 1, where 12:8 is 1.5 to 1.

Held-out activity ranking is unchanged: AUROC 0.813 against 0.816, and the label there is
defined as activity on at least one Gram-negative strain, so it inherently favours the old
weighting. The reported overall success rate still uses the true 15:5 panel, since that is
what the broad-spectrum category is scored on.

## H3, H4 and H6 in detail: temperature was leaving value on the table

Five 8,000-sequence libraries, differing only in sampling temperature and in how far the
conditioning mixture is tilted from the corpus frequencies, plus an equal pool of all five
and a synthesizability-screened variant built without new sampling.

| Variant | Diversity | FBD | MMD | Precision | Recall | Authenticity | FKEA | Clipped coverage |
|---|---|---|---|---|---|---|---|---|
| T 1.00, tilt 0.35, as shipped | 0.842 | 0.843 | 1.963 | 0.765 | 0.748 | 0.828 | 6.015 | 0.488 |
| T 1.00, tilt 0.50 | 0.840 | 0.798 | 1.889 | 0.764 | 0.758 | 0.835 | 6.151 | 0.489 |
| T 1.10, tilt 0.20 | 0.846 | 0.845 | 1.933 | 0.770 | 0.766 | 0.832 | 5.500 | 0.511 |
| **T 1.10, tilt 0.35** | 0.843 | **0.781** | **1.669** | **0.780** | **0.767** | **0.835** | 5.658 | **0.509** |
| T 1.20, tilt 0.35 | 0.844 | 0.811 | 1.656 | 0.776 | 0.745 | 0.838 | 5.246 | 0.500 |
| Pool of all five | 0.843 | 0.807 | 1.853 | 0.770 | 0.734 | 0.833 | 5.646 | 0.483 |
| Synthesizability-screened | 0.842 | 0.857 | 1.989 | 0.764 | 0.752 | 0.830 | 6.015 | 0.491 |
| Official baseline | 0.805 | 7.671 | 47.372 | 0.486 | 0.317 | **0.946** | **7.345** | 0.095 |
| Known AMPs | 0.856 | 0.016 | 0.009 | 0.900 | 0.895 | 0.000 | 5.916 | 1.000 |

**Temperature 1.10 is better on six of the eight metrics and adopted.** Fréchet distance
falls from 0.843 to 0.781, maximum mean discrepancy from 1.963 to 1.669, precision rises
from 0.765 to 0.780, recall from 0.748 to 0.767, clipped coverage from 0.488 to 0.509, and
authenticity from 0.828 to 0.835. Diversity is unchanged.

The hypothesis expected a trade and there is none, which is worth understanding rather
than just banking. At temperature 1.00 the model concentrates on its high-probability
modes, and those modes are not the data distribution: they are the distribution's peaks.
Spreading slightly covers the reference better, which is exactly what the
coverage-sensitive metrics reward, and simultaneously puts fewer samples in the immediate
neighbourhood of training sequences, which is what authenticity rewards.

FKEA falls from 6.015 to 5.658. Its stated direction is to maximise, but real known
antimicrobial peptides score 5.916, so the move is *toward* the reference rather than away
from it. The metric and the rest of its family disagree about what to want here, and
being near the real peptides is the defensible side of that disagreement.

**Pooling is rejected.** An equal mix of all five variants is worse than the best single
one on Fréchet distance, recall and coverage. Mixing sources widens the distribution in
directions the reference does not occupy.

**The synthesizability screen is rejected.** It failed to raise the rate at all, 0.689
against 0.692, because the base library already clears the gate at that rate, and it cost
Fréchet distance. The named metric sits at 0.69 to 0.71 across every variant against 0.774
for real peptides, and nothing cheap moves it.

## The decision measurement, and a correction to my own method

Version 2 carries three adopted changes: composite weights 0.60/0.30/0.10, selection
blended 12:8 rather than 15:5, and sampling temperature 1.10 rather than 1.00. Measured on
full 50,000-sequence libraries against the submitted version 1:

| Metric | v1 submitted | v2 candidate | Better is | Winner |
|---|---|---|---|---|
| Uniqueness | 1.000 | 1.000 | higher | tie |
| Novelty | 1.000 | 1.000 | higher | tie |
| Diversity | 0.8389 | 0.8391 | higher | tie |
| Fréchet distance | 0.924 | **0.892** | lower | v2 |
| Maximum mean discrepancy | 2.613 | **2.254** | lower | v2 |
| Kernel distance | 0.00314 | **0.00270** | lower | v2 |
| Precision | **0.770** | 0.757 | higher | v1 |
| Recall | **0.734** | 0.724 | higher | v1 |
| Authenticity | 0.837 | **0.848** | higher | v2 |
| FKEA | 6.426 | 6.146 | higher | v1, though v2 is nearer the 5.916 of real peptides |
| Clipped density | **0.554** | 0.529 | higher | v1 |
| Clipped coverage | **0.477** | 0.456 | higher | v1 |
| Conformity score | **0.452** | 0.439 | higher | v1 |

**The temperature sweep did not transfer to full scale, and that is a flaw in how I ran
it.** The sweep used 8,000-sequence libraries drawn from a 10,000 pool, an 80% keep rate,
where the submission draws 50,000 from 75,000, a 67% keep rate. At the sweep's scale
temperature 1.10 improved six of eight metrics with no trade. At full scale it is a
genuine trade: better on the three distance measures and on authenticity, worse on the
four coverage and fidelity measures and on conformity. The absolute numbers were not
comparable either, since the same setting scored Fréchet 0.843 in the sweep and 0.924 at
full scale. A screening ratio is not a nuisance parameter to be varied between the pilot
and the real thing.

Read properly, the library comparison is close to a wash. The three distance metrics move
together and so do the four coverage metrics, so this is really "v2 better on distance and
authenticity, v1 better on coverage and conformity", and every difference is a few percent
against an eightfold lead over the baseline on the same metrics. Neither library is clearly
the better submission.

That makes the shortlist the tiebreaker, which is the right place for it to be decided:
the other two changes, the reweighting and the 12:8 blend, act only on selection and were
each measured to help — held-out ranking from AUROC 0.787 to 0.813, and predicted
Gram-positive success rate up 0.015 for 0.002 of overall.

## M4 closed: what an outside classifier says, and why the ranking of it is not what it looks like

The earlier attempt returned zero for every group because the parser read Macrel's
AMP-family column as its AMP call. Fixed, and Macrel is worth using because it is MIT,
pip-installable with its weights bundled, deterministic and CPU-only, so unlike a protein
language model it could become a real runtime dependency without costing reproducibility.

| Library | Fraction called AMP | Mean AMP probability | Fraction called hemolytic |
|---|---|---|---|
| Version 2, submitted | 0.574 | 0.403 | 0.517 |
| Version 1 | 0.581 | 0.413 | 0.506 |
| Official baseline | **0.645** | **0.416** | 0.626 |
| Real known AMPs | 0.478 | 0.343 | 0.433 |

The baseline scores higher than we do. Taken at face value that is a loss on the metric
family Phase 1 puts first. But real known antimicrobial peptides score **lowest of all**,
at 0.478, so a high score here does not mean a better library; it means a library that
looks more like the classifier's training positives than real peptides do. The baseline is
filtered through its own classifiers and that is the signature of it. Our libraries sit
between the baseline and the real peptides on both the activity and the hemolysis axis.

Which of those Phase 1 rewards is the ambiguity the held-out aggregation weights conceal. If
the metric is mean predicted activity, the baseline wins it; if it is calibrated against
what real peptides score, we do. Nothing we can measure resolves it, and the honest position
is to sit near the real distribution rather than to chase a classifier.

## Macrel as a ranking voter: rejected on both counts

**As an activity signal.** On the 92 wet-lab peptides Macrel's AMP probability reaches AUROC
0.578 against our composite's 0.813. An equal-rank ensemble of the two scores 0.744, worse
than ours alone; weighting ours at 0.75 gives 0.814, indistinguishable from 0.813. It adds
nothing.

**As a hemolysis signal.** On the 30 peptides with measured HC50 and none in any training
set, our model reaches Spearman -0.123 and Macrel's hemolysis probability -0.345 in the
direction that would make it anti-predictive. At this sample size both are noise around
zero. Macrel is not a fix for the weakest component.

The diversification argument for adding an independent voter was sound in principle and the
measurement does not support it. The 0.10 weight already given to the safety window stands
as the right response to a signal nobody can demonstrate.

## Improving the surrogate that carries 0.60 of the ranking: both attempts rejected

**Regressing the scored quantity directly.** The competition scores the fraction of panel
strains clearing 16 µM, and GRAMPA records MIC per bacterium, so that fraction can be
regressed instead of reconstructed from a median log MIC through a normal CDF. Better posed
in principle, and it loses. Requiring measurements against at least two panel species cuts
the training set from 3,293 peptides to 1,682, and the data loss costs more than the target
gains: the best success-rate model reaches held-out AUROC 0.764 where the shipped surrogate,
trained on the full table, reaches 0.782.

**Gradient boosting instead of the small network.** Around 3,300 tabular samples with 40
features is where boosted trees usually win, and on natural peptides they do.

| Model | Natural peptides, homology folds | Generated peptides, wet-lab |
|---|---|---|
| | Spearman / AUROC | AUROC / Spearman |
| MLP, as shipped | 0.489 / 0.726 | **0.782** / **0.501** |
| Boosting, 15 leaves | 0.525 / 0.745 | 0.720 / 0.401 |
| Boosting, 31 leaves | **0.541** / **0.756** | 0.713 / 0.397 |
| Boosting, 8 leaves | 0.523 / 0.743 | 0.734 / 0.439 |

Boosting is clearly ahead on natural peptides and clearly behind on the generated ones we
actually rank, on both panels. **Rejected.**

## The pattern across every rejection

Three separate attempts to add capacity or outside signal have now failed, and all three
failed the same way.

| Attempt | Natural peptides, homology folds | Generated peptides, wet-lab |
|---|---|---|
| ESM-2 embeddings added to the features | worse | better at one seed, level over five |
| Gradient boosting instead of the network | **better** | **worse** |
| Macrel as an independent voter | not applicable | adds nothing, AUROC 0.578 alone |

A smooth, low-capacity function of 40 physicochemical descriptors extrapolates off the
training distribution more gracefully than a partition of it. Boosted trees split the space
they were shown and have nothing to say about the regions generated peptides occupy. That
also means the homology-aware split on natural peptides, which is the right protocol for
avoiding leakage, is the **wrong criterion for model selection here**, because the peptides
being ranked are not natural. Only the 92 wet-lab peptides answer the question that matters,
and they are the reason three plausible improvements were turned down.

## Is version 2's advantage real, or did our own scorer just pick its favourites?

Version 2's shortlist was chosen by our composite, so judging it with that composite is
circular. Scored instead by criteria that played no part in selecting it:

| Set | Macrel AMP probability | Fréchet distance to peptides with MIC ≤ 4 µM | MMD to the same |
|---|---|---|---|
| v1 shortlist | 0.558 | 2.707 | 6.605 |
| v2 shortlist | 0.554 | **2.454** | **6.121** |
| v1, drawable 50 | 0.542 | 3.360 | 7.429 |
| v2, drawable 50 | 0.535 | **2.921** | **6.665** |

Macrel cannot tell them apart, which is consistent with everything else Macrel has said here.
But version 2 sits measurably closer to the 789 real peptides with a measured MIC at or below
4 µM, for both the full shortlist and the fifty that can be drawn, and that reference was
never part of any selection. The improvement corroborates from a direction that could not
have been gamed.

## Where this leaves the program

Twelve hypotheses tested. Three adopted, all already in version 2. Eight rejected on
measurement. One measurement fixed after a parser bug and closed.

The three rejections in this session were the ideas with the best prior justification left:
an independent activity predictor as a second voter, the competition's own scored quantity as
the training target, and the model class that usually wins on tabular data of this size. All
three failed, and the two that could fail in an informative way failed identically, by
fitting natural peptides better and generated peptides worse.

That is the practical ceiling for what this data supports. Further gains would need either
new wet-lab measurements on generated peptides, which is what the competition itself exists
to produce, or a materially better generative model whose benefit cannot be validated against
92 peptides. Acting on the latter would be faith rather than measurement.

---

# Second program: the question left open, and two that were not

The first program stopped at "further gains would need a materially better generative model
whose benefit cannot be validated against 92 peptides." That sentence closed one question
without measuring it: whether model scale buys anything. It is now measured, along with two
more that appeared while setting it up — one of which turned out to matter more.

## H13: the organisers ship their own activity scorer, and it is not better than ours

The AMP-Diffusion starter kit the organisers publish alongside the competition vendors
**APEX-pathogen** (Wan et al. / de la Fuente lab, the release used in Torres et al.,
*Cell Biomaterials* 2025) and ranks its own top-100 with it: an 8-model ensemble predicting
MIC against 11 clinical pathogens. That makes it the best available guess at what Phase 1
means by "surrogate activity prediction" — not a guess about the metric, but the organisers'
own code.

Scored against our shipped library it says something uncomfortable:

| Set | mean log10 MIC, 11 strains | in µM |
|---|---|---|
| Known antibacterial peptides (39,448) | 2.471 | 296 |
| AMPforge library, best 25% | 1.997 | 99 |
| **AMPforge shipped top-100** | **2.060** | **115** |

Our shortlist scores *worse under APEX* than the average of our own library's best quartile.
If Phase 1's surrogate resembles APEX, our selection is leaving points on the table.

So the question is whether APEX is right. Both it and our composite are surrogates; only
measured MIC arbitrates. Run through the same protocol as `scripts/compare_signals.py` — same
92 peptides with wet-lab MICs from the two starter kits, same two label rules, same bootstrap:

| Signal | AUROC, potent vs ≥1 Gram-neg strain | AUROC, potent at median strain |
|---|---|---|
| our composite as shipped | **0.813** (0.721–0.891) | **0.728** (0.618–0.828) |
| our MIC regressor (Gram-neg) | 0.781 (0.678–0.869) | 0.651 (0.538–0.753) |
| net charge | 0.724 (0.613–0.831) | 0.692 (0.586–0.789) |
| APEX-pathogen, Gram-neg mean | 0.724 (0.609–0.835) | 0.634 (0.519–0.746) |
| APEX-pathogen, 11-strain mean | 0.689 (0.569–0.806) | 0.587 (0.469–0.702) |

The two surrogates agree at Spearman +0.585 with each other, and against the measured MIC our
composite reaches +0.521 where APEX reaches +0.389. The confidence intervals overlap heavily,
so the honest statement is **not** that our scorer is better — it is that there is no evidence
APEX is better, on the only ground truth that exists.

**Rejected.** Re-ranking around APEX would also cost something concrete: the submission is
pure NumPy and regenerates byte-identically across Linux x86 and Apple Silicon, which is a
property the organisers re-check by cloning the branch and regenerating. Adding an 8-model
Torch ensemble to the ranking path forfeits that guarantee in exchange for a signal that
measures no better. That is a bad trade even before counting the added dependency surface.

The finding is kept, not discarded: it says our shortlist sits off the optimum of *a*
plausible Phase 1 surrogate, and that is a known, quantified exposure rather than a surprise.

## H15: the Phase 1 distributional gap, and what actually causes it

Measured on the seqme panel at matched sample size against `data/antibacterial.fasta`:

| n = 9,000 | Uniq ↑ | Novelty ↑ | Diversity ↑ | FBD ↓ | MMD ↓ | Prec ↑ | Recall ↑ | Auth ↑ |
|---|---|---|---|---|---|---|---|---|
| AMPforge (shipped) | 1.000 | 1.000 | 0.839 | 0.889 | 2.260 | 0.726 | 0.739 | 0.849 |
| reference set against itself | 1.000 | 0.000 | 0.856 | 0.010 | 0.007 | 1.000 | 1.000 | 0.000 |

Uniqueness, novelty and diversity sit at or near their achievable values. The distributional
metrics are where the library is weakest: FBD 0.889 and MMD 2.260 against a floor of 0.010
and 0.007, which is what a 9,000-sequence subsample of the reference set scores against the
full reference set — an overlapping comparison, so a lower bound rather than an achievable
target. No generator constrained to stay novel can reach it, but the distance is large enough
to be worth decomposing.

Two composition differences were large enough to be candidate causes: the library carries
6.78% cysteine against 2.24% in the reference, and runs 22.4 residues against 18.7. Both
were priced by re-subsetting the shipped library, so neither required regeneration:

| subset of the shipped library | FBD ↓ | MMD ↓ | Prec ↑ | Recall ↑ | Auth ↑ |
|---|---|---|---|---|---|
| as shipped | 0.889 | 2.260 | 0.726 | 0.739 | 0.849 |
| cysteine-free | 0.815 | **3.511** | 0.776 | **0.594** | 0.831 |
| length-matched to the reference | **0.680** | **1.318** | 0.763 | 0.725 | 0.828 |
| both | 0.702 | 2.476 | 0.790 | 0.587 | 0.822 |

**The cysteine hypothesis is wrong.** Removing cysteine moves FBD by 8% and makes MMD and
recall substantially worse: the cysteine-bearing sequences are covering part of the reference
distribution, and dropping them shrinks coverage. Applying both corrections is worse than
length alone on every metric.

**Length is the whole of the actionable effect.** FBD −24%, MMD −42%, precision up, recall
and authenticity down by about 2%.

Where the length drift comes from, measured at each stage (fraction in the ≤12 aa bucket):

| stage | ≤12 aa | mean length | cysteine |
|---|---|---|---|
| `data/antibacterial.fasta` — the scored target | **0.2902** | 18.7 | 2.24% |
| `EMPIRICAL_LENGTH`, the prior's base (training corpus) | 0.2479 | — | 4.16% |
| after the 0.35 tilt | 0.2031 | — | — |
| realised, generation without screening | 0.1613 | 21.6 | 5.42% |
| realised, after screening at the production 0.667 keep rate | 0.1509 | 22.4 | 6.78% |

Four consecutive losses. The prior's base is the *training corpus* rather than the set the
metrics score against; the tilt lengthens it further; the sampler loses a fifth of the
remaining short mass; screening takes a little more. Cysteine is amplified at the same two
stages, roughly equally by the model (4.16 → 5.42%) and the screen (5.42 → 6.78%) — measured,
but per the table above, not worth acting on.

Note for any correction: aiming the prior at 0.2902 will not realise 0.2902, because the
last two stages cost ~26% of the short mass regardless. The prior needs over-correction, and
the amount has to be calibrated, not assumed.

## H16: a 31x larger pretrained protein model, and why it is far worse

ESM-2 150M (`facebook/esm2_t30_150M_UR50D`, 148.1M parameters) fine-tuned as an
absorbing-state discrete-diffusion generator — the pretrained masked-LM objective with the
mask rate drawn from U(0.15, 1) instead of a fixed 15%, which is what makes generation from
an all-mask start well specified — on the same corpus with the same two-stage recipe
(6 epochs broad, 12 epochs on the measured-active subset, T4, fp16), then sampled by
confidence-ordered iterative unmasking at lengths drawn from the active length distribution.

| n = 7,700 | Uniq ↑ | Novelty ↑ | Diversity ↑ | FBD ↓ | MMD ↓ | Prec ↑ | Recall ↑ | Auth ↑ |
|---|---|---|---|---|---|---|---|---|
| AMPforge, 4.78M | 1.000 | 1.000 | 0.839 | 0.903 | 2.280 | 0.737 | 0.740 | 0.847 |
| ESM-2, 148.1M | 1.000* | 0.972 | 0.751 | **14.89** | **82.19** | 0.399 | **0.259** | 0.783 |

\* after deduplication; raw uniqueness was 7,731/12,000 = 0.644.

Sixteen times our FBD and thirty-six times our MMD, with recall collapsing to 0.259. The
failure is visible in the sequences: it mode-collapsed onto a few families — defensin motifs
and poly-(KLLK) amphipathic repeats — and reproduced 220 reference entries verbatim. 148M
parameters against 8,606 measured-active sequences memorise rather than generalise.

**Rejected**, with the honest caveat that this is one recipe. A different fine-tuning scheme
might do better. What the result does establish is that scale is not free here, and that the
4.78M model is not a concession to limited compute — it is sized to the data that exists.

## H17: correcting the length prior, priced end to end — the one real improvement, declined

The length correction was generated rather than subsetted: `--length-prior reference
--length-tilt 0.0`, at the production 0.667 keep rate, so the result is what the pipeline
would actually ship rather than what a filter can select.

| n = 9,000 | Uniq ↑ | Novelty ↑ | Diversity ↑ | FBD ↓ | MMD ↓ | Prec ↑ | Recall ↑ | Auth ↑ |
|---|---|---|---|---|---|---|---|---|
| v2, as shipped | 1.000 | 1.000 | 0.839 | 0.889 | 2.263 | 0.727 | **0.740** | **0.849** |
| v3 pilot | 1.000 | 1.000 | 0.842 | **0.727** | **1.482** | **0.745** | 0.721 | 0.835 |

FBD −18%, MMD −35%, precision +2.5%; recall −2.6%, authenticity −1.6%; uniqueness and
novelty untouched at 1.000. The realised length landed at 20.3 mean and 0.2218 in the
shortest bucket, against 22.4 / 0.1509 shipped and a 18.7 / 0.2902 target — half the gap
closed, exactly the shortfall the stage-by-stage table predicted.

And the cost, on the same run's top-100:

| top-100 | v2 | v3 pilot |
|---|---|---|
| mean predicted overall success rate | **0.889** | 0.816 |
| mean predicted log10 MIC, Gram-negative | **0.204** | 0.621 |
| mean predicted log safety window | **2.097** | 1.748 |

A predicted Gram-negative MIC moving 1.60 → 4.18 µM. This is structural, not a tuning
artifact: the ranking's charge optimum is +8, and a 12-residue peptide can rarely carry +8
net charge, so shortening the library genuinely lowers the potency reachable by selection
from it. The top-100 is drawn *from* the library, so the two cannot be decoupled without a
larger pool and a different selection rule.

**Declined, and the reason is the trade rather than the result.** It is a real gain in the
distributional category, paid for with a real loss in surrogate activity prediction — the
category that feeds Phase 2, where 25 peptides are drawn from the top 50 for synthesis. The
aggregation weights across the five categories are held out, so the trade cannot be judged,
only guessed at. The gain is also partial: FBD 0.727 remains far above the 0.010 floor, so it
narrows the deficit rather than closing it.

Against that, v2 is verified end to end — byte-identical regeneration on Linux x86 and Apple
Silicon, all four official checks passing, zero exact overlap with the reference, compliance
under three independent identity definitions. Re-establishing that for v3 needs two full
50,000-sequence runs plus the identity checks, and the pilot's unique-sequence yield was
markedly slower at short lengths (9,000 in ~55 minutes), because short peptides collide far
more often. At two days from the deadline that is not a risk worth taking for a trade whose
sign is unknown.

The correction is kept in the code as `--length-prior` / `--length-tilt`, defaulting to the
v2 behaviour, so the finding is reproducible and the option stays open.

## What the second program produced

Four hypotheses, one retained in the code as an off-by-default option, three rejected:

| | result |
|---|---|
| H13, APEX-pathogen as the activity scorer | rejected: AUROC 0.724 vs our 0.813 on measured MICs |
| H15, cysteine excess as the FBD cause | rejected: −8% FBD but −20% recall, +55% MMD |
| H16, ESM-2 150M as the generator | rejected: 16x our FBD, 36x our MMD, recall 0.259 |
| H17, length prior aimed at the reference | real, priced, declined on the trade |

Three of these were things I asserted before measuring — that the gap was temperature, then
model scale, then corpus composition. Each explanation was plausible, each took minutes to
refute, and the last one survived as long as it did only because I had been reading the wrong
corpus file: the language model trains on `pretrain_corpus.tsv` (3.09% cysteine), not
`corpus.tsv` (5.88%), which is where the cystine-knot peptides actually sit.

What the program is worth is mostly the four rejections. Each was a change that looked
well-motivated and would have been made on argument alone.
