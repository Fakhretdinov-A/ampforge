"""Check that the numpy inference path reproduces the trained PyTorch model.

Generation runs in numpy so that output is byte-identical across platforms, but
that only matters if the numpy implementation is actually the same model. This
compares both forward passes on random conditioned inputs and fails loudly if
they disagree. Run it after every training run, before generating a library.

    uv run --extra train python scripts/check_parity.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ampforge import tokenizer as tk
from ampforge.nn import PeptideLM as NumpyLM
from train_lm import PeptideLM as TorchLM

TOLERANCE = 1e-3


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, default=Path("checkpoint/peptide_lm.npz"))
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--length", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    data = dict(np.load(args.checkpoint))
    cfg = data["__config__"]
    torch_model = TorchLM(d_model=int(cfg[0]), n_layers=int(cfg[1]),
                          n_heads=int(cfg[2]), d_ff=int(cfg[3]))
    torch_model.load_state_dict(
        {k: torch.from_numpy(v) for k, v in data.items() if k != "__config__"}
    )
    torch_model.eval()
    numpy_model = NumpyLM.load(args.checkpoint)

    rng = np.random.default_rng(args.seed)
    b, n = args.batch, args.length
    idx = np.concatenate([
        rng.integers(tk.LEN_OFFSET, tk.LEN_OFFSET + 7, (b, 1)),
        rng.integers(tk.CHARGE_OFFSET, tk.CHARGE_OFFSET + 6, (b, 1)),
        rng.integers(tk.ACT_OFFSET, tk.ACT_OFFSET + 8, (b, 1)),
        np.full((b, 1), tk.BOS),
        rng.integers(tk.AA_OFFSET, tk.AA_OFFSET + 20, (b, n)),
    ], axis=1).astype(np.int64)

    with torch.no_grad():
        reference = torch_model(torch.from_numpy(idx)).numpy()[:, -1]

    cache = numpy_model.new_cache(b, idx.shape[1])
    produced = None
    for position in range(idx.shape[1]):
        produced = numpy_model.step(idx[:, position], position, cache)

    difference = float(np.abs(produced - reference).max())
    scale = float(np.abs(reference).max())
    argmax_agrees = bool((produced.argmax(1) == reference.argmax(1)).all())

    print(f"checkpoint            {args.checkpoint}")
    print(f"max logit difference  {difference:.3e}  (logit scale {scale:.2f})")
    print(f"argmax agrees         {argmax_agrees}")

    if difference < TOLERANCE and argmax_agrees:
        print("\nPARITY OK")
        return
    print("\nPARITY FAILED", file=sys.stderr)
    raise SystemExit(1)


if __name__ == "__main__":
    main()
