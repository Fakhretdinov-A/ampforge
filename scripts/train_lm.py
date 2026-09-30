"""Train the AMPforge conditional peptide language model and export numpy weights.

Training uses PyTorch (optional `train` extra). Inference at submission time runs
in pure numpy so that generation is bit-for-bit reproducible on any platform.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ampforge import tokenizer as tk


class Block(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.n_heads = n_heads
        self.ln1 = nn.LayerNorm(d_model)
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, d = x.shape
        h = self.ln1(x)
        q, k, v = self.qkv(h).chunk(3, dim=-1)
        shape = (b, t, self.n_heads, d // self.n_heads)
        q, k, v = (z.view(shape).transpose(1, 2) for z in (q, k, v))
        att = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        att = att.transpose(1, 2).contiguous().view(b, t, d)
        x = x + self.drop(self.proj(att))
        h = self.ln2(x)
        x = x + self.drop(self.fc2(F.gelu(self.fc1(h))))
        return x


class PeptideLM(nn.Module):
    def __init__(self, d_model=256, n_layers=6, n_heads=8, d_ff=1024, dropout=0.1):
        super().__init__()
        self.tok = nn.Embedding(tk.VOCAB_SIZE, d_model)
        self.pos = nn.Embedding(tk.MAX_SEQ_LEN, d_model)
        self.blocks = nn.ModuleList(
            [Block(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)]
        )
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, tk.VOCAB_SIZE, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        t = idx.shape[1]
        pos = torch.arange(t, device=idx.device)
        x = self.drop(self.tok(idx) + self.pos(pos)[None])
        for blk in self.blocks:
            x = blk(x)
        return self.head(self.ln_f(x))


def load_corpus(path: Path, amp_only: bool = False) -> list[list[int]]:
    """Read a corpus. A trailing `is_amp` column marks general peptides as non-AMP."""
    lines = path.read_text().splitlines()
    header = lines[0].split("\t")
    has_flag = "is_amp" in header
    flag_index = header.index("is_amp") if has_flag else None

    examples = []
    for row in lines[1:]:
        parts = row.split("\t")
        seq, bits, length, charge = parts[0], int(parts[1]), int(parts[2]), float(parts[3])
        is_amp = int(parts[flag_index]) == 1 if has_flag else True
        if amp_only and not is_amp:
            continue
        activity = (
            tk.activity_token(bool(bits & 1), bool(bits & 2), bool(bits & 4))
            if is_amp
            else tk.ACT_NON_AMP
        )
        examples.append(
            tk.encode(seq, tk.length_token(length), tk.charge_token(charge), activity)
        )
    return examples


def export_numpy(model: PeptideLM, path: Path, config: dict) -> None:
    arrays = {k: v.detach().cpu().float().numpy() for k, v in model.state_dict().items()}
    arrays["__config__"] = np.array(
        [config["d_model"], config["n_layers"], config["n_heads"], config["d_ff"]],
        dtype=np.int64,
    )
    np.savez_compressed(path, **arrays)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", type=Path, default=Path("data/corpus.tsv"))
    ap.add_argument("--out", type=Path, default=Path("checkpoint/peptide_lm.npz"))
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    ap.add_argument("--amp-only", action="store_true",
                    help="keep only antimicrobial-annotated sequences (fine-tuning stage)")
    ap.add_argument("--init-from", type=Path, default=None,
                    help="start from an exported checkpoint instead of random weights")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    examples = load_corpus(args.corpus, amp_only=args.amp_only)
    print(f"loaded {len(examples)} sequences from {args.corpus}"
          + (" (antimicrobial only)" if args.amp_only else ""))

    maxlen = max(len(e) for e in examples)
    data = np.full((len(examples), maxlen), tk.PAD, dtype=np.int64)
    for i, ex in enumerate(examples):
        data[i, : len(ex)] = ex

    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(data))
    n_val = max(1, len(data) // 20)
    val_idx, train_idx = perm[:n_val], perm[n_val:]
    train = torch.from_numpy(data[train_idx])
    val = torch.from_numpy(data[val_idx])

    config = {"d_model": 256, "n_layers": 6, "n_heads": 8, "d_ff": 1024}
    model = PeptideLM(**config)
    if args.init_from is not None:
        loaded = dict(np.load(args.init_from))
        loaded.pop("__config__", None)
        model.load_state_dict({k: torch.from_numpy(v) for k, v in loaded.items()})
        print(f"initialised from {args.init_from}")
    model = model.to(args.device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model parameters: {n_params/1e6:.2f}M  device={args.device}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    steps_per_epoch = math.ceil(len(train) / args.batch_size)
    total_steps = steps_per_epoch * args.epochs
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=total_steps, pct_start=0.1
    )

    best_val = float("inf")
    step = 0
    for epoch in range(args.epochs):
        model.train()
        order = torch.randperm(len(train))
        running = 0.0
        for i in range(0, len(train), args.batch_size):
            batch = train[order[i : i + args.batch_size]].to(args.device)
            logits = model(batch[:, :-1])
            loss = F.cross_entropy(
                logits.reshape(-1, tk.VOCAB_SIZE),
                batch[:, 1:].reshape(-1),
                ignore_index=tk.PAD,
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            running += loss.item()
            step += 1

        model.eval()
        with torch.no_grad():
            # weight by the number of scored tokens, so a short final batch does not
            # count as much as a full one; this value selects the exported checkpoint
            total_loss = 0.0
            total_tokens = 0
            for i in range(0, len(val), args.batch_size):
                batch = val[i : i + args.batch_size].to(args.device)
                logits = model(batch[:, :-1])
                targets = batch[:, 1:].reshape(-1)
                loss = F.cross_entropy(
                    logits.reshape(-1, tk.VOCAB_SIZE), targets,
                    ignore_index=tk.PAD, reduction="sum",
                )
                total_loss += loss.item()
                total_tokens += int((targets != tk.PAD).sum().item())
            vl = total_loss / max(total_tokens, 1)

        print(
            f"epoch {epoch+1:3d}/{args.epochs}  train {running/steps_per_epoch:.4f}  val {vl:.4f}"
            + ("  *" if vl < best_val else "")
        )
        if vl < best_val:
            best_val = vl
            args.out.parent.mkdir(parents=True, exist_ok=True)
            export_numpy(model, args.out, config)

    print(f"best val loss {best_val:.4f} -> {args.out}")


if __name__ == "__main__":
    main()
