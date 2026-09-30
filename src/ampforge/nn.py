"""Pure-numpy inference for the AMPforge conditional peptide language model.

Two properties matter here and both are deliberate:

* **Cross-platform determinism.** Inference runs in float64 and every logit vector
  is rounded to a fixed decimal grid before the softmax. Rounding erases the
  floating-point noise that differs between BLAS implementations and CPU
  architectures, so the sampled library is bit-for-bit identical on the author's
  laptop and on the organizers' Linux workstation.
* **Speed.** A key/value cache keeps generation linear rather than quadratic in
  peptide length.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from . import tokenizer as tk

# logits are snapped to this grid before softmax; 1e-5 is far coarser than the
# ~1e-13 discrepancy a different BLAS can introduce, and far finer than anything
# that affects sampling quality over a 20-letter alphabet
LOGIT_GRID = 100_000.0


def _layer_norm(x: np.ndarray, weight: np.ndarray, bias: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    mu = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return (x - mu) / np.sqrt(var + eps) * weight + bias


def _erf(x: np.ndarray) -> np.ndarray:
    """Rational approximation of erf (Numerical Recipes 6.2.2), ~1.2e-7 accurate.

    Evaluated with Horner's method so the coefficient list stays readable.
    """
    coefficients = (
        0.17087277, -0.82215223, 1.48851587, -1.13520398, 0.27886807,
        -0.18628806, 0.09678418, 0.37409196, 1.00002368, -1.26551223,
    )
    ax = np.abs(x)
    t = 1.0 / (1.0 + 0.5 * ax)
    poly = np.zeros_like(ax)
    for c in coefficients:
        poly = poly * t + c
    tau = t * np.exp(-ax * ax + poly)
    return np.sign(x) * (1.0 - tau)


def _gelu(x: np.ndarray) -> np.ndarray:
    return 0.5 * x * (1.0 + _erf(x / np.sqrt(2.0)))


class PeptideLM:
    """Numpy re-implementation of the trained transformer decoder, with KV cache."""

    def __init__(self, weights: dict[str, np.ndarray]):
        cfg = weights["__config__"]
        self.d_model, self.n_layers, self.n_heads, self.d_ff = (int(v) for v in cfg)
        self.head_dim = self.d_model // self.n_heads
        w = {k: v.astype(np.float64) for k, v in weights.items() if k != "__config__"}
        self.tok_emb = w["tok.weight"]
        # the checkpoint is authoritative about vocabulary size, so a model trained
        # before the non-AMP token was added still loads and runs
        self.vocab_size = self.tok_emb.shape[0]
        self.pos_emb = w["pos.weight"]
        self.ln_f_w, self.ln_f_b = w["ln_f.weight"], w["ln_f.bias"]
        self.head = w["head.weight"]
        self.blocks = []
        for i in range(self.n_layers):
            p = f"blocks.{i}."
            self.blocks.append(
                {
                    "ln1_w": w[p + "ln1.weight"], "ln1_b": w[p + "ln1.bias"],
                    "qkv_w": w[p + "qkv.weight"].T.copy(), "qkv_b": w[p + "qkv.bias"],
                    "proj_w": w[p + "proj.weight"].T.copy(), "proj_b": w[p + "proj.bias"],
                    "ln2_w": w[p + "ln2.weight"], "ln2_b": w[p + "ln2.bias"],
                    "fc1_w": w[p + "fc1.weight"].T.copy(), "fc1_b": w[p + "fc1.bias"],
                    "fc2_w": w[p + "fc2.weight"].T.copy(), "fc2_b": w[p + "fc2.bias"],
                }
            )

    @classmethod
    def load(cls, path: Path) -> "PeptideLM":
        with np.load(path) as data:
            return cls({k: data[k] for k in data.files})

    def new_cache(self, batch: int, max_positions: int) -> dict:
        """Preallocate the key/value cache.

        Growing the cache with concatenate would reallocate and copy it on every
        one of the ~50 generation steps, for every layer, which dominates runtime.
        """
        shape = (self.n_layers, batch, self.n_heads, max_positions, self.head_dim)
        return {"k": np.zeros(shape), "v": np.zeros(shape), "filled": 0}

    def compact(self, cache: dict, keep: np.ndarray) -> None:
        """Drop finished rows from the cache so they stop consuming compute."""
        cache["k"] = np.ascontiguousarray(cache["k"][:, keep])
        cache["v"] = np.ascontiguousarray(cache["v"][:, keep])

    def step(self, tokens: np.ndarray, position: int, cache: dict) -> np.ndarray:
        """Advance one position. tokens: (batch,) int. Returns logits (batch, vocab)."""
        b = tokens.shape[0]
        slot = cache["filled"]
        upto = slot + 1
        scale = 1.0 / np.sqrt(self.head_dim)
        x = self.tok_emb[tokens] + self.pos_emb[position][None]
        for layer, blk in enumerate(self.blocks):
            h = _layer_norm(x, blk["ln1_w"], blk["ln1_b"])
            qkv = h @ blk["qkv_w"] + blk["qkv_b"]
            q, k, v = np.split(qkv, 3, axis=-1)
            cache["k"][layer, :, :, slot, :] = k.reshape(b, self.n_heads, self.head_dim)
            cache["v"][layer, :, :, slot, :] = v.reshape(b, self.n_heads, self.head_dim)
            kk = cache["k"][layer, :, :, :upto, :]
            vv = cache["v"][layer, :, :, :upto, :]
            q = q.reshape(b, self.n_heads, self.head_dim)
            # (batch, heads, upto) attention logits without materialising a 4-D tensor
            att = np.einsum("bhd,bhtd->bht", q, kk) * scale
            att -= att.max(axis=-1, keepdims=True)
            np.exp(att, out=att)
            att /= att.sum(axis=-1, keepdims=True)
            out = np.einsum("bht,bhtd->bhd", att, vv).reshape(b, self.d_model)
            x = x + (out @ blk["proj_w"] + blk["proj_b"])
            h = _layer_norm(x, blk["ln2_w"], blk["ln2_b"])
            h = _gelu(h @ blk["fc1_w"] + blk["fc1_b"])
            x = x + (h @ blk["fc2_w"] + blk["fc2_b"])
        cache["filled"] = upto
        x = _layer_norm(x, self.ln_f_w, self.ln_f_b)
        return x @ self.head.T


def sample(
    model: PeptideLM,
    prefixes: np.ndarray,
    rng: np.random.Generator,
    temperature: float = 1.0,
    top_p: float = 0.95,
    min_len: int = 8,
    max_len: int = 50,
) -> list[str]:
    """Sample one peptide per row of `prefixes` (batch, 3 conditioning tokens)."""
    b = prefixes.shape[0]
    context = np.concatenate([prefixes, np.full((b, 1), tk.BOS, dtype=np.int64)], axis=1)
    cache = model.new_cache(b, context.shape[1] + max_len)
    logits = None
    for pos in range(context.shape[1]):
        logits = model.step(context[:, pos], pos, cache)

    aa_lo, aa_hi = tk.AA_OFFSET, tk.AA_OFFSET + 20
    # token ids per position, 0 where the peptide has already ended
    drawn = np.zeros((b, max_len), dtype=np.int64)
    position = context.shape[1]
    # indices into the original batch for the rows still being generated; finished
    # rows are dropped so the remaining steps do not pay for them. Mean peptide
    # length is well under the 50-residue cap, so this roughly halves the work.
    active = np.arange(b)

    for step_i in range(max_len):
        masked = np.full_like(logits, -np.inf)
        masked[:, aa_lo:aa_hi] = logits[:, aa_lo:aa_hi]
        if step_i >= min_len:
            masked[:, tk.EOS] = logits[:, tk.EOS]

        masked = masked / temperature
        masked = np.round(masked * LOGIT_GRID) / LOGIT_GRID  # determinism anchor
        masked = masked - np.nanmax(np.where(np.isfinite(masked), masked, -np.inf), axis=1, keepdims=True)
        probs = np.where(np.isfinite(masked), np.exp(masked), 0.0)
        probs = probs / probs.sum(axis=1, keepdims=True)

        if top_p < 1.0:
            order = np.argsort(-probs, axis=1, kind="stable")
            sorted_p = np.take_along_axis(probs, order, axis=1)
            cum = np.cumsum(sorted_p, axis=1)
            sorted_p[(cum - sorted_p) > top_p] = 0.0
            probs = np.zeros_like(probs)
            np.put_along_axis(probs, order, sorted_p, axis=1)
            probs = probs / probs.sum(axis=1, keepdims=True)

        cum = np.cumsum(probs, axis=1)
        # Draw for the full original batch and index the active rows, so the random
        # stream stays identical whether or not rows have been dropped. This is
        # load-bearing for the compaction below: do not replace it with
        # rng.random(len(active)).
        draws = rng.random(b)[active][:, None]
        choice = (cum < draws).sum(axis=1)
        # If float normalisation leaves the last cumulative just below the draw, the
        # index runs past the end. Fall back to the most probable token rather than
        # clipping onto a conditioning token, which the decoder maps to nothing.
        overflow = choice >= probs.shape[1]
        if overflow.any():
            choice = np.where(overflow, probs.argmax(axis=1), choice)

        ended = choice == tk.EOS
        drawn[active[~ended], step_i] = choice[~ended]
        if ended.all():
            break
        if ended.any():
            keep = ~ended
            active = active[keep]
            choice = choice[keep]
            model.compact(cache, keep)

        logits = model.step(choice, position, cache)
        position += 1

    lookup = np.full(tk.VOCAB_SIZE, "", dtype=object)
    for token_id, aa in tk.ID_TO_AA.items():
        lookup[token_id] = aa
    return ["".join(lookup[row[row > 0]]) for row in drawn]
