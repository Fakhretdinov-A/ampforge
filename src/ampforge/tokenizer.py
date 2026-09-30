"""Vocabulary and conditional prefix encoding for the AMPforge peptide language model.

Token layout of one training example:

    [LEN_BUCKET] [CHARGE_BUCKET] [ACTIVITY] [BOS] a1 a2 ... an [EOS]

The three conditioning tokens act as a prefix that steers generation toward a
desired length range, net charge range and activity profile.
"""
from __future__ import annotations

AA = "ACDEFGHIKLMNPQRSTVWY"

PAD, BOS, EOS = 0, 1, 2
AA_OFFSET = 3
LEN_OFFSET = AA_OFFSET + 20          # 23
CHARGE_OFFSET = LEN_OFFSET + 7       # 30
ACT_OFFSET = CHARGE_OFFSET + 6       # 36
# eight activity-annotation combinations, plus one token for peptides that carry no
# antimicrobial annotation at all. The extra token exists so the model can be
# pretrained on general peptide space without being told those peptides are AMPs.
ACT_NON_AMP = ACT_OFFSET + 8         # 44
VOCAB_SIZE = ACT_NON_AMP + 1         # 45

MAX_PEPTIDE_LEN = 50
MAX_SEQ_LEN = 3 + 1 + MAX_PEPTIDE_LEN + 1  # prefix + BOS + peptide + EOS

LEN_BINS = [12, 16, 20, 25, 32, 40]      # 7 buckets
CHARGE_BINS = [0.0, 2.0, 4.0, 6.0, 9.0]  # 6 buckets

AA_TO_ID = {a: AA_OFFSET + i for i, a in enumerate(AA)}
ID_TO_AA = {v: k for k, v in AA_TO_ID.items()}


def _bucket(value: float, bins: list[float]) -> int:
    for i, edge in enumerate(bins):
        if value <= edge:
            return i
    return len(bins)


def length_token(length: int) -> int:
    return LEN_OFFSET + _bucket(length, LEN_BINS)


def charge_token(charge: float) -> int:
    return CHARGE_OFFSET + _bucket(charge, CHARGE_BINS)


def activity_token(antibacterial: bool, gram_neg: bool, gram_pos: bool) -> int:
    bits = (1 if antibacterial else 0) | (2 if gram_neg else 0) | (4 if gram_pos else 0)
    return ACT_OFFSET + bits


def encode(seq: str, length_tok: int, charge_tok: int, act_tok: int) -> list[int]:
    return [length_tok, charge_tok, act_tok, BOS] + [AA_TO_ID[a] for a in seq] + [EOS]


def decode(token_ids: list[int]) -> str:
    return "".join(ID_TO_AA[t] for t in token_ids if t in ID_TO_AA)
