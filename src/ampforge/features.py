"""Physicochemical descriptors for short peptides.

Pure numpy, no external bioinformatics dependency, so that the generation entry
point stays light and fully deterministic.
"""
from __future__ import annotations

import math

import numpy as np

AA = "ACDEFGHIKLMNPQRSTVWY"
AA_INDEX = {a: i for i, a in enumerate(AA)}

# Kyte-Doolittle hydropathy
KD = {
    "A": 1.8, "C": 2.5, "D": -3.5, "E": -3.5, "F": 2.8, "G": -0.4, "H": -3.2,
    "I": 4.5, "K": -3.9, "L": 3.8, "M": 1.9, "N": -3.5, "P": -1.6, "Q": -3.5,
    "R": -4.5, "S": -0.8, "T": -0.7, "V": 4.2, "W": -0.9, "Y": -1.3,
}

# Eisenberg consensus hydrophobicity (used for the hydrophobic moment)
EISENBERG = {
    "A": 0.62, "C": 0.29, "D": -0.90, "E": -0.74, "F": 1.19, "G": 0.48,
    "H": -0.40, "I": 1.38, "K": -1.50, "L": 1.06, "M": 0.64, "N": -0.78,
    "P": 0.12, "Q": -0.85, "R": -2.53, "S": -0.18, "T": -0.05, "V": 1.08,
    "W": 0.81, "Y": 0.26,
}

# side-chain pKa values for net-charge calculation
PKA_SIDE = {"D": 3.65, "E": 4.25, "C": 8.18, "Y": 10.07, "H": 6.00, "K": 10.53, "R": 12.48}
POSITIVE = {"K", "R", "H"}
PKA_NTERM, PKA_CTERM = 9.69, 2.34

# Chou-Fasman helix propensity
HELIX_PROP = {
    "A": 1.42, "C": 0.70, "D": 1.01, "E": 1.51, "F": 1.13, "G": 0.57, "H": 1.00,
    "I": 1.08, "K": 1.16, "L": 1.21, "M": 1.45, "N": 0.67, "P": 0.57, "Q": 1.11,
    "R": 0.98, "S": 0.77, "T": 0.83, "V": 1.06, "W": 1.08, "Y": 0.69,
}

# Boman index (protein-binding potential), kcal/mol
BOMAN = {
    "A": -0.17, "C": 0.24, "D": -1.23, "E": -2.02, "F": 1.13, "G": -0.01,
    "H": -0.96, "I": 1.16, "K": -0.99, "L": 1.18, "M": 1.18, "N": -0.42,
    "P": -0.45, "Q": -0.58, "R": -0.81, "S": -0.13, "T": -0.14, "V": 1.13,
    "W": 1.85, "Y": 0.94,
}

MW = {
    "A": 71.08, "C": 103.14, "D": 115.09, "E": 129.12, "F": 147.18, "G": 57.05,
    "H": 137.14, "I": 113.16, "K": 128.17, "L": 113.16, "M": 131.19, "N": 114.10,
    "P": 97.12, "Q": 128.13, "R": 156.19, "S": 87.08, "T": 101.10, "V": 99.13,
    "W": 186.21, "Y": 163.18,
}

FEATURE_NAMES = [
    "length", "charge", "charge_density", "gravy", "hmoment", "hydrophobic_frac",
    "aromatic_frac", "helix_prop", "boman", "pi", "aliphatic_index",
    "cys_frac", "met_frac", "pro_frac", "gly_frac", "lys_frac", "arg_frac",
    "max_hydrophobic_run", "kr_ratio", "net_charge_per_res",
]


def net_charge(seq: str, ph: float = 7.4) -> float:
    """Net charge from Henderson-Hasselbalch over side chains and termini."""
    pos = 1.0 / (1.0 + 10 ** (ph - PKA_NTERM))
    neg = 1.0 / (1.0 + 10 ** (PKA_CTERM - ph))
    for aa in seq:
        pka = PKA_SIDE.get(aa)
        if pka is None:
            continue
        if aa in POSITIVE:
            pos += 1.0 / (1.0 + 10 ** (ph - pka))
        else:
            neg += 1.0 / (1.0 + 10 ** (pka - ph))
    return pos - neg


def isoelectric_point(seq: str) -> float:
    lo, hi = 0.0, 14.0
    for _ in range(60):
        mid = (lo + hi) / 2.0
        if net_charge(seq, mid) > 0:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def hydrophobic_moment(seq: str, angle_deg: float = 100.0, window: int = 11) -> float:
    """Maximum Eisenberg hydrophobic moment over sliding windows (per residue)."""
    vals = [EISENBERG.get(a, 0.0) for a in seq]
    n = len(vals)
    w = min(window, n)
    rad = math.radians(angle_deg)
    best = 0.0
    for start in range(0, n - w + 1):
        sin_sum = cos_sum = 0.0
        for i in range(w):
            h = vals[start + i]
            sin_sum += h * math.sin(rad * i)
            cos_sum += h * math.cos(rad * i)
        best = max(best, math.sqrt(sin_sum * sin_sum + cos_sum * cos_sum) / w)
    return best


def max_hydrophobic_run(seq: str) -> int:
    hydrophobic = set("AVILMFWC")
    best = run = 0
    for a in seq:
        run = run + 1 if a in hydrophobic else 0
        best = max(best, run)
    return best


def aliphatic_index(seq: str) -> float:
    n = len(seq)
    if n == 0:
        return 0.0
    a = seq.count("A") / n * 100
    v = seq.count("V") / n * 100
    il = (seq.count("I") + seq.count("L")) / n * 100
    return a + 2.9 * v + 3.9 * il


def descriptors(seq: str) -> dict[str, float]:
    n = len(seq)
    if n == 0:
        return {k: 0.0 for k in FEATURE_NAMES}
    q = net_charge(seq)
    hydrophobic = sum(seq.count(a) for a in "AVILMFWC")
    aromatic = sum(seq.count(a) for a in "FWY")
    return {
        "length": float(n),
        "charge": q,
        "charge_density": q / n,
        "gravy": sum(KD[a] for a in seq) / n,
        "hmoment": hydrophobic_moment(seq),
        "hydrophobic_frac": hydrophobic / n,
        "aromatic_frac": aromatic / n,
        "helix_prop": sum(HELIX_PROP[a] for a in seq) / n,
        "boman": sum(BOMAN[a] for a in seq) / n,
        "pi": isoelectric_point(seq),
        "aliphatic_index": aliphatic_index(seq),
        "cys_frac": seq.count("C") / n,
        "met_frac": seq.count("M") / n,
        "pro_frac": seq.count("P") / n,
        "gly_frac": seq.count("G") / n,
        "lys_frac": seq.count("K") / n,
        "arg_frac": seq.count("R") / n,
        "max_hydrophobic_run": float(max_hydrophobic_run(seq)),
        "kr_ratio": (seq.count("K") + 1.0) / (seq.count("R") + 1.0),
        "net_charge_per_res": q / n,
    }


def feature_matrix(sequences: list[str]) -> np.ndarray:
    return np.array(
        [[descriptors(s)[k] for k in FEATURE_NAMES] for s in sequences],
        dtype=np.float64,
    )
