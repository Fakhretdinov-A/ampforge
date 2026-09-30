"""Assemble the pretraining corpus: general peptide space plus every known AMP.

The AMP-only corpus has 46,637 sequences, which is small enough that a 4.8M
parameter model starts memorising after about 40 epochs (training loss 1.14
against validation 1.64). Pretraining on a much larger sample of ordinary peptide
space first gives the model a prior over what peptides look like in general, so the
AMP fine-tune has less to learn and less room to memorise.

Sources, all redistributed with the HydrAMP starter kit under MIT:
  Uniprot_0_25_{train,val}.csv  general peptides, no antimicrobial annotation
  unlabelled_positive.csv       known AMPs from dbAMP, DRAMP and AMP Scanner
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

from ampforge.features import descriptors

STD_AA = set("ACDEFGHIKLMNPQRSTVWY")
MIN_LEN, MAX_LEN = 8, 50


def read_column(path: Path, column: str = "Sequence") -> list[str]:
    out = []
    with path.open() as fh:
        for row in csv.DictReader(fh):
            seq = (row.get(column) or "").strip().upper()
            if set(seq) <= STD_AA and MIN_LEN <= len(seq) <= MAX_LEN:
                out.append(seq)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hydramp-training", type=Path, required=True)
    ap.add_argument("--amp-corpus", type=Path, default=Path("data/corpus.tsv"))
    ap.add_argument("--out", type=Path, default=Path("data/pretrain_corpus.tsv"))
    args = ap.parse_args()

    amp_rows = args.amp_corpus.read_text().splitlines()[1:]
    amp_sequences = {row.split("\t")[0] for row in amp_rows}

    general: list[str] = []
    for name in ("Uniprot_0_25_train.csv", "Uniprot_0_25_val.csv"):
        general += read_column(args.hydramp_training / name)
    extra_amps = read_column(args.hydramp_training / "unlabelled_positive.csv")

    general = sorted(set(general) - amp_sequences - set(extra_amps))
    extra_amps = sorted(set(extra_amps) - amp_sequences)
    print(f"AMP corpus              {len(amp_sequences):7,}")
    print(f"additional known AMPs   {len(extra_amps):7,}")
    print(f"general peptides        {len(general):7,}")

    lines = ["sequence\tlabel_bits\tlength\tcharge\tgravy\thmoment\tis_amp"]
    for row in amp_rows:
        lines.append(row + "\t1")
    for seq in extra_amps:
        d = descriptors(seq)
        # annotated antimicrobial and antibacterial, no gram-specific information
        lines.append(f"{seq}\t1\t{len(seq)}\t{d['charge']:.3f}\t{d['gravy']:.3f}"
                     f"\t{d['hmoment']:.3f}\t1")
    for seq in general:
        d = descriptors(seq)
        lines.append(f"{seq}\t0\t{len(seq)}\t{d['charge']:.3f}\t{d['gravy']:.3f}"
                     f"\t{d['hmoment']:.3f}\t0")

    args.out.write_text("\n".join(lines) + "\n")
    print(f"wrote {len(lines) - 1:,} sequences -> {args.out}")


if __name__ == "__main__":
    main()
