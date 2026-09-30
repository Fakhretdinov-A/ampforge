"""Build the AMPforge training corpus from the public MarLys AMP database (CC-0).

Source: Marczak, Bocian, Lyskowski. MarLys AMP database (MLAMP_db), Mendeley Data,
doi:10.17632/w4hb5grjwb.3 -- released CC-0.

Outputs data/corpus.tsv with columns: sequence, label_bits, length, charge, gravy, hmoment.
label_bits is a small integer bitmask of curated activity annotations.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

STD_AA = set("ACDEFGHIKLMNPQRSTVWY")
MIN_LEN, MAX_LEN = 8, 50

# activity annotations we keep, in bit order
LABELS = ["antibacterial", "anti-gram-", "anti-gram+", "anticancer", "anti-biofilm"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mlamp", type=Path, required=True, help="MLAMP_db.json")
    ap.add_argument("--out", type=Path, default=Path("data/corpus.tsv"))
    args = ap.parse_args()

    records = json.loads(args.mlamp.read_text())
    args.out.parent.mkdir(parents=True, exist_ok=True)

    seen: set[str] = set()
    rows: list[str] = []
    for rec in records:
        seq = (rec.get("sequence") or "").strip().upper()
        if not seq or seq in seen:
            continue
        if not set(seq) <= STD_AA:
            continue
        if not (MIN_LEN <= len(seq) <= MAX_LEN):
            continue
        acts = set(rec.get("activity") or [])
        if "antimicrobial" not in acts:
            continue
        seen.add(seq)
        bits = 0
        for i, lab in enumerate(LABELS):
            if lab in acts:
                bits |= 1 << i
        props = rec.get("properties") or {}
        rows.append(
            "\t".join(
                [
                    seq,
                    str(bits),
                    str(len(seq)),
                    f"{float(props.get('charge', 0.0)):.3f}",
                    f"{float(props.get('gravy', 0.0)):.3f}",
                    f"{float(props.get('hmoment', 0.0)):.3f}",
                ]
            )
        )

    header = "sequence\tlabel_bits\tlength\tcharge\tgravy\thmoment"
    args.out.write_text(header + "\n" + "\n".join(rows) + "\n")
    print(f"wrote {len(rows)} sequences -> {args.out}")


if __name__ == "__main__":
    main()
