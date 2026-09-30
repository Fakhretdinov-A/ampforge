"""Scoring and candidate ranking for AMPforge.

The composite score is built to match how the competition actually scores teams
rather than to maximise a generic "AMP-likeness" number:

* Success rate is the fraction of panel strains with MIC <= 16 uM. Each regressor
  predicts a mean log10 MIC; combined with its cross-validated error that gives a
  probability of clearing the 16 uM threshold, which is an unbiased estimate of
  the per-panel success rate.
* The panel is 15 Gram-negative and 5 Gram-positive strains, so the overall
  success rate is their 15:5 weighted average.
* The selectivity category scores the safety window HC50 / MIC50, which in log
  space is simply the difference of the two predictions.
* Two families of signal were measured against held-out wet-lab results from the
  competition's own baseline methods, and against natural peptides under a
  homology-aware split. Both carry real signal and neither dominates: on 92
  wet-lab peptides the predicted success rate reaches AUROC 0.82 and net charge
  0.72, with confidence intervals that overlap heavily, while on 3,293 natural
  peptides split by sequence cluster the regressor reaches 0.72 against 0.68 for
  net charge. The score therefore blends a learned and a mechanistic signal
  rather than betting on either.
* Synthesis risk is a gate, not a ranking term. Measured against activity it is
  mildly anti-correlated (AUROC 0.44), because the traits it penalises -- length,
  hydrophobicity, cysteine -- partly track potency. Mixing it into the ranking
  score therefore corrupted the ranking while doing nothing extra for its actual
  purpose. It now excludes candidates above a risk threshold and is otherwise
  left out of the ordering.
* Peptides that are unlikely to survive solid-phase synthesis, purification and
  solubility QC are penalised, because a failed peptide is never retested and
  silently wastes one of the team's 25 experimental slots.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .encode import encode_many

# competition potency threshold: MIC <= 16 uM
LOG_MIC_THRESHOLD = float(np.log10(16.0))
MAE_TO_SIGMA = float(np.sqrt(np.pi / 2.0))

# The 20-strain panel is 15 Gram-negative and 5 Gram-positive. That ratio is what the
# broad-spectrum category is scored on, so it is what the reported overall success rate
# uses.
N_GRAM_NEGATIVE, N_GRAM_POSITIVE = 15, 5

# Selection uses a different ratio, because five categories are scored and each counts
# once: broad-spectrum on all 20 strains, Gram-negative on 15, Gram-positive on 5,
# multi-drug-resistant on 8 isolates that are 5 Gram-negative and 3 Gram-positive, and
# selectivity on the safety window. Adding up how much each category depends on each
# panel gives 2.375 units of demand for Gram-negative performance and 1.625 for
# Gram-positive, a ratio near 1.46 to 1 rather than 3 to 1. Optimising the panel is not
# the same as optimising the scoreboard, and there is no way to allocate peptides to
# categories: 25 are drawn at random from the top 50 and every one contributes to every
# category's team average.
SELECT_GRAM_NEGATIVE, SELECT_GRAM_POSITIVE = 12, 8

MODEL_NAMES = ("mic_gram_negative", "mic_gram_positive", "hemolysis")


def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0.0)


def _normal_cdf(z: np.ndarray) -> np.ndarray:
    # Abramowitz-Stegun 26.2.17, accurate to ~7.5e-8, ample for a ranking signal
    t = 1.0 / (1.0 + 0.2316419 * np.abs(z))
    poly = t * (0.319381530 + t * (-0.356563782 + t * (1.781477937
            + t * (-1.821255978 + t * 1.330274429))))
    tail = np.exp(-0.5 * z * z) / np.sqrt(2.0 * np.pi) * poly
    return np.where(z >= 0.0, 1.0 - tail, tail)


class Rankers:
    """Numpy inference for the exported MIC and hemolysis regressors."""

    def __init__(self, arrays: dict[str, np.ndarray]):
        self.models: dict[str, dict] = {}
        for name in MODEL_NAMES:
            self.models[name] = {
                "mu": arrays[f"{name}/mu"],
                "sd": arrays[f"{name}/sd"],
                "w0": arrays[f"{name}/net.0.weight"], "b0": arrays[f"{name}/net.0.bias"],
                "w1": arrays[f"{name}/net.3.weight"], "b1": arrays[f"{name}/net.3.bias"],
                "w2": arrays[f"{name}/net.6.weight"], "b2": arrays[f"{name}/net.6.bias"],
                # sqrt(pi/2) converts a mean absolute error to a standard deviation
                # for normal residuals. An earlier version used 1.4826, which is the
                # factor for a *median* absolute deviation, and inflated sigma by 18%.
                "sigma": float(arrays[f"{name}/cv_mae"][0]) * MAE_TO_SIGMA,
                "spearman": float(arrays[f"{name}/cv_spearman"][0]),
            }

    @classmethod
    def load(cls, path: Path) -> "Rankers":
        with np.load(path) as data:
            return cls({k: data[k] for k in data.files})

    def predict(self, name: str, features: np.ndarray) -> np.ndarray:
        m = self.models[name]
        x = (features - m["mu"]) / m["sd"]
        h = _relu(x @ m["w0"].T + m["b0"])
        h = _relu(h @ m["w1"].T + m["b1"])
        return (h @ m["w2"].T + m["b2"]).ravel()

    def sigma(self, name: str) -> float:
        return self.models[name]["sigma"]


# ---------------------------------------------------------------- synthesis risk

_HYDROPHOBIC = set("AVILMFWC")


def synthesis_penalty(sequences: list[str]) -> np.ndarray:
    """Heuristic risk that a peptide fails synthesis, purification or solubility.

    Each term reflects a well-known failure mode of solid-phase peptide synthesis
    or of downstream handling, and larger values mean higher risk.
    """
    out = np.zeros(len(sequences))
    for i, seq in enumerate(sequences):
        n = len(seq)
        risk = 0.0
        # cysteines oxidise and form uncontrolled disulfides in a linear peptide
        n_cys = seq.count("C")
        risk += 0.45 * n_cys if n_cys else 0.0
        # methionine and tryptophan oxidise during cleavage
        risk += 0.10 * seq.count("M")
        risk += 0.05 * max(0, seq.count("W") - 2)
        # long hydrophobic stretches drive on-resin aggregation and poor solubility
        run = best = 0
        for aa in seq:
            run = run + 1 if aa in _HYDROPHOBIC else 0
            best = max(best, run)
        risk += 0.25 * max(0, best - 5)
        # overall hydrophobicity: insoluble peptides never reach the assay
        hydro_frac = sum(seq.count(a) for a in _HYDROPHOBIC) / n
        risk += 4.0 * max(0.0, hydro_frac - 0.60)
        # Asn-Gly deamidates; Asp-Pro is acid-labile during TFA cleavage
        risk += 0.20 * seq.count("NG")
        risk += 0.20 * seq.count("DP")
        # low-complexity sequences (homopolymers, two-letter repeats) aggregate, are
        # poor drug candidates, and are a known degenerate optimum for any score that
        # rewards charge or hydrophobicity monotonically
        counts = np.array([seq.count(a) for a in sorted(set(seq))], dtype=np.float64)
        p = counts / n
        entropy = float(-(p * np.log2(p)).sum())
        risk += 2.0 * max(0.0, 2.6 - entropy)
        risk += 1.5 * max(0.0, counts.max() / n - 0.35)
        # length drives both cost and stepwise failure probability; the penalty is
        # steep past 30 residues, where solid-phase yields fall off noticeably
        risk += 0.12 * max(0, n - 28)
        out[i] = risk
    return out


# ---------------------------------------------------------------- pharmacophore

# Net charge is the strongest validated single predictor, but its benefit
# saturates: beyond roughly +8 the literature reports rising hemolysis and no
# further potency gain, and our wet-lab evidence contains no peptides in that range
# to support extrapolating further. The reward therefore plateaus rather than
# growing without bound, which also stops the ranker degenerating toward polylysine.
CHARGE_OPTIMUM = 8.0
CHARGE_WIDTH = 4.5
HMOMENT_SCALE = 0.7
GRAVY_PENALTY_ABOVE = 0.5


def pharmacophore_score(sequences: list[str]) -> np.ndarray:
    """Cationic-amphipathic character, the best wet-lab-validated ranking signal.

    The charge term peaks near +8 and falls away on both sides rather than rising
    without bound. A monotonic reward drifts the selection into the extreme tail
    (net charge above +15), where the wet-lab evidence used to justify the signal
    contains no peptides at all and where the literature reports rising hemolysis
    without further potency gain. Peaking keeps the selection inside the range the
    evidence actually covers.
    """
    from .features import descriptors

    out = np.zeros(len(sequences))
    for i, seq in enumerate(sequences):
        d = descriptors(seq)
        charge_term = float(np.exp(-0.5 * ((d["charge"] - CHARGE_OPTIMUM) / CHARGE_WIDTH) ** 2))
        moment_term = np.tanh(d["hmoment"] / HMOMENT_SCALE)
        helix_term = np.clip((d["helix_prop"] - 0.9) / 0.4, 0.0, 1.0)
        # excess bulk hydrophobicity tracks hemolysis rather than potency
        gravy_term = -np.clip(d["gravy"] - GRAVY_PENALTY_ABOVE, 0.0, 2.0) / 2.0
        out[i] = 0.45 * charge_term + 0.35 * moment_term + 0.10 * helix_term + 0.10 * gravy_term
    return out


# ------------------------------------------------------------------- composite

# A candidate above this synthesis-risk score is excluded outright rather than
# down-ranked. The threshold admits the great majority of natural antimicrobial
# peptides while removing designs that are likely to fail synthesis, purification
# or solubility QC, each of which would silently forfeit an experimental slot.
MAX_SYNTHESIS_RISK = 1.2


def score_candidates(
    sequences: list[str],
    rankers: Rankers,
    surrogate_weight: float = 0.60,
    pharmacophore_weight: float = 0.30,
    safety_weight: float = 0.10,
    safety_cap: float = 1.6,
) -> dict[str, np.ndarray]:
    """Return per-sequence predictions and the composite ranking score."""
    features = encode_many(sequences)
    mic_gn = rankers.predict("mic_gram_negative", features)
    mic_gp = rankers.predict("mic_gram_positive", features)
    hc50 = rankers.predict("hemolysis", features)

    sr_gn = _normal_cdf((LOG_MIC_THRESHOLD - mic_gn) / rankers.sigma("mic_gram_negative"))
    sr_gp = _normal_cdf((LOG_MIC_THRESHOLD - mic_gp) / rankers.sigma("mic_gram_positive"))
    sr_overall = (N_GRAM_NEGATIVE * sr_gn + N_GRAM_POSITIVE * sr_gp) / (
        N_GRAM_NEGATIVE + N_GRAM_POSITIVE
    )
    # what selection optimises, as distinct from what the broad-spectrum category scores
    sr_selection = (SELECT_GRAM_NEGATIVE * sr_gn + SELECT_GRAM_POSITIVE * sr_gp) / (
        SELECT_GRAM_NEGATIVE + SELECT_GRAM_POSITIVE
    )

    # log10 safety window, using the more potent panel as the MIC50 proxy
    log_sw = hc50 - np.minimum(mic_gn, mic_gp)
    penalty = synthesis_penalty(sequences)

    pharmacophore = pharmacophore_score(sequences)

    # The weights follow three measurements that agree, not a fit to the 92-peptide
    # wet-lab set, which is far too small to fit three weights against.
    #
    # On those 92 peptides each term alone scores AUROC 0.817 for the predicted success
    # rate, 0.751 for the pharmacophore and 0.532 for the safety window. On 3,293
    # natural peptides under folds that keep sequence-similar peptides together, the
    # same ordering holds: 0.731 for the MIC regressor against 0.670 for the
    # pharmacophore. And the weight surface falls monotonically as the safety-window
    # weight rises, from 0.816 at zero to 0.746 at 0.30.
    #
    # So the surrogate leads, the pharmacophore is kept as a mechanistic hedge against
    # the surrogate failing off-distribution, and the safety window keeps a token weight
    # because selectivity is one of the five scored categories and a peptide with a
    # ruinous safety window is genuinely bad, even though the hemolysis model behind it
    # transfers weakly. An earlier 0.45/0.35/0.20 scored 0.787 where this scores 0.814;
    # the difference is inside the confidence interval, but every independent
    # measurement points the same way.
    composite = (
        surrogate_weight * sr_selection
        + pharmacophore_weight * pharmacophore
        + safety_weight * np.clip(log_sw, -safety_cap, safety_cap) / safety_cap
    )
    return {
        "pharmacophore": pharmacophore,
        "mic_gram_negative": mic_gn,
        "mic_gram_positive": mic_gp,
        "hc50": hc50,
        "success_rate_gram_negative": sr_gn,
        "success_rate_gram_positive": sr_gp,
        "success_rate_overall": sr_overall,
        "success_rate_selection": sr_selection,
        "log_safety_window": log_sw,
        "synthesis_penalty": penalty,
        "composite": composite,
    }


# ------------------------------------------------------- AMP-likeness screen

class AmpClassifier:
    """Numpy inference for the AMP-versus-non-AMP screen.

    Used to filter the submitted library, never to order the top-100: on held-out
    wet-lab data this family of model separates peptides from non-peptides well
    (5-fold AUROC 0.96) but does not rank potency among peptides.
    """

    def __init__(self, arrays: dict[str, np.ndarray]):
        self.mu, self.sd = arrays["mu"], arrays["sd"]
        self.w0, self.b0 = arrays["net.0.weight"], arrays["net.0.bias"]
        self.w1, self.b1 = arrays["net.3.weight"], arrays["net.3.bias"]
        self.w2, self.b2 = arrays["net.6.weight"], arrays["net.6.bias"]
        self.cv_auroc = float(arrays["cv_auroc"][0])
        self.cv_mcc = float(arrays["cv_mcc"][0])

    @classmethod
    def load(cls, path: Path) -> "AmpClassifier":
        with np.load(path) as data:
            return cls({k: data[k] for k in data.files})

    def logit(self, sequences: list[str]) -> np.ndarray:
        x = (encode_many(sequences) - self.mu) / self.sd
        h = _relu(x @ self.w0.T + self.b0)
        h = _relu(h @ self.w1.T + self.b1)
        return (h @ self.w2.T + self.b2).ravel()

    def probability(self, sequences: list[str]) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-self.logit(sequences)))
