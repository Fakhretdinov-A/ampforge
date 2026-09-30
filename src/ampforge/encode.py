"""Fixed-width feature encoding shared by the ranker's training and inference."""
from __future__ import annotations

import numpy as np

from .features import AA, FEATURE_NAMES, descriptors

COMPOSITION_NAMES = [f"frac_{a}" for a in AA]
ENCODING_NAMES = FEATURE_NAMES + COMPOSITION_NAMES
N_FEATURES = len(ENCODING_NAMES)


def encode_one(seq: str) -> np.ndarray:
    d = descriptors(seq)
    n = max(len(seq), 1)
    comp = [seq.count(a) / n for a in AA]
    return np.array([d[k] for k in FEATURE_NAMES] + comp, dtype=np.float64)


def encode_many(sequences: list[str]) -> np.ndarray:
    return np.stack([encode_one(s) for s in sequences]) if sequences else np.zeros((0, N_FEATURES))
