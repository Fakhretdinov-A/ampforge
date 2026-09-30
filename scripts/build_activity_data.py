"""Assemble MIC and hemolysis training tables from public datasets.

Sources
-------
GRAMPA (Witten & Witten 2019), https://github.com/zswitten/Antimicrobial-Peptides
  * data/grampa.csv           - 51k MIC measurements, log10 uM, per bacterium
  * Hemolysis/Cleaned_hemolytic_data.csv - log10 HC50, uM, from Hemolytik

Only unmodified, linear peptides over the 20 standard amino acids and 8-50
residues long are kept, matching the competition's sequence constraints.
"""
from __future__ import annotations

import argparse
import csv
import statistics
from collections import defaultdict
from pathlib import Path

STD_AA = set("ACDEFGHIKLMNPQRSTVWY")

# species appearing in the competition's 20-strain panel
GRAM_NEGATIVE = [
    "A. baumannii", "E. cloacae", "E. coli", "K. pneumoniae",
    "P. aeruginosa", "S. enterica", "S. typhimurium",
]
GRAM_POSITIVE = ["B. subtilis", "S. aureus", "E. faecalis", "E. faecium"]


def _valid(seq: str) -> bool:
    return bool(seq) and set(seq) <= STD_AA and 8 <= len(seq) <= 50


def build_mic(grampa_csv: Path, out_dir: Path) -> None:
    per_panel: dict[str, dict[str, list[float]]] = {
        "gram_negative": defaultdict(list),
        "gram_positive": defaultdict(list),
    }
    kept = 0
    with grampa_csv.open() as fh:
        for row in csv.DictReader(fh):
            if row["is_modified"] != "False":
                continue
            seq = row["sequence"].strip().upper()
            if not _valid(seq):
                continue
            try:
                value = float(row["value"])
            except ValueError:
                continue
            bug = row["bacterium"].strip()
            if bug in GRAM_NEGATIVE:
                per_panel["gram_negative"][seq].append(value)
                kept += 1
            elif bug in GRAM_POSITIVE:
                per_panel["gram_positive"][seq].append(value)
                kept += 1

    for panel, table in per_panel.items():
        path = out_dir / f"mic_{panel}.tsv"
        lines = ["sequence\tlog10_mic_um\tn_measurements"]
        for seq, vals in sorted(table.items()):
            lines.append(f"{seq}\t{statistics.median(vals):.4f}\t{len(vals)}")
        path.write_text("\n".join(lines) + "\n")
        print(f"{panel}: {len(table)} unique peptides -> {path}")
    print(f"(from {kept} panel-relevant measurements)")


def build_hemolysis(hemo_csv: Path, out_dir: Path) -> None:
    table: dict[str, list[float]] = defaultdict(list)
    with hemo_csv.open() as fh:
        for row in csv.DictReader(fh):
            seq = (row.get("Sequence") or "").strip()
            # lowercase letters mark D-amino acids; those peptides are out of scope
            if seq != seq.upper():
                continue
            seq = seq.upper()
            if not _valid(seq):
                continue
            raw = (row.get("log10_HC50") or "").strip()
            try:
                value = float(raw)
            except ValueError:
                continue
            if not (-3.0 <= value <= 5.0):
                continue
            table[seq].append(value)

    path = out_dir / "hemolysis.tsv"
    lines = ["sequence\tlog10_hc50_um\tn_measurements"]
    for seq, vals in sorted(table.items()):
        lines.append(f"{seq}\t{statistics.median(vals):.4f}\t{len(vals)}")
    path.write_text("\n".join(lines) + "\n")
    print(f"hemolysis: {len(table)} unique peptides -> {path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--grampa", type=Path, required=True)
    ap.add_argument("--hemolysis", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, default=Path("data"))
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    build_mic(args.grampa, args.out_dir)
    build_hemolysis(args.hemolysis, args.out_dir)


if __name__ == "__main__":
    main()
