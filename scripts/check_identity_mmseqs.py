"""Check the top-100 against the reference set using MMseqs2 alignment identity.

The starter kit's validator measures novelty with a Levenshtein ratio, but the
competition proposal states that the 80% identity rule is computed with MMseqs2
pairwise alignment against the MarLys reference database. The two disagree in a
way that matters: local alignment ignores unaligned overhangs, so a short peptide
that matches a window of a long natural peptide can have a low Levenshtein ratio
and a high alignment identity at the same time. Passing only the Levenshtein check
is therefore not sufficient.

This reports, for every candidate, the highest alignment identity found, both as
MMseqs2's own percent identity over the aligned region and rescaled to the query
length, which is the stricter reading.
"""
from __future__ import annotations

import argparse
import subprocess
import tempfile
from pathlib import Path


def read_fasta(path: Path) -> list[str]:
    sequences, parts = [], []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if parts:
                sequences.append("".join(parts))
                parts = []
        else:
            parts.append(line.upper())
    if parts:
        sequences.append("".join(parts))
    return sequences


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--query", type=Path, required=True, help="top.fasta")
    ap.add_argument("--target", type=Path, required=True, help="reference FASTA")
    ap.add_argument("--mmseqs", default="mmseqs")
    ap.add_argument("--limit", type=float, default=0.80)
    ap.add_argument("--sensitivity", type=float, default=7.5)
    ap.add_argument("--coverage", type=float, default=0.80,
                    help="minimum fraction of the query the alignment must cover")
    args = ap.parse_args()

    queries = read_fasta(args.query)
    print(f"{len(queries)} candidates against {args.target}")

    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        query_fasta = tmpdir / "query.fasta"
        query_fasta.write_text("".join(f">q{i}\n{s}\n" for i, s in enumerate(queries)))
        hits = tmpdir / "hits.m8"
        cmd = [
            args.mmseqs, "easy-search", str(query_fasta), str(args.target), str(hits),
            str(tmpdir / "work"),
            "-s", str(args.sensitivity),
            "--max-seqs", "4000",
            "-e", "1e5",
            "--min-seq-id", "0",
            # coverage is required over the query. Without it, MMseqs2 reports 100%
            # identity for a five-residue local match, which is not what an 80%
            # sequence-identity rule can possibly mean: under that reading even the
            # organizers' own baseline shortlist fails.
            "--cov-mode", "2", "-c", str(args.coverage),
            "--format-output", "query,target,pident,alnlen,qlen,tlen",
        ]
        run = subprocess.run(cmd, capture_output=True, text=True)
        if run.returncode != 0:
            print(run.stdout[-2000:])
            raise SystemExit(run.stderr[-2000:])

        best_aligned: dict[str, float] = {}
        best_over_query: dict[str, float] = {}
        for line in hits.read_text().splitlines():
            q, _t, pident, alnlen, qlen, _tlen = line.split("\t")[:6]
            pid = float(pident)
            pid = pid / 100.0 if pid > 1.0 else pid
            over_query = pid * int(alnlen) / int(qlen)
            if pid > best_aligned.get(q, 0.0):
                best_aligned[q] = pid
            if over_query > best_over_query.get(q, 0.0):
                best_over_query[q] = over_query

    def summarise(label: str, table: dict[str, float]) -> int:
        values = [table.get(f"q{i}", 0.0) for i in range(len(queries))]
        over = [i for i, v in enumerate(values) if v > args.limit]
        peak = max(values) if values else 0.0
        no_hit = sum(1 for i in range(len(queries)) if f"q{i}" not in table)
        print(f"\n{label}")
        print(f"  highest identity found      {peak:.3f}")
        print(f"  candidates above {args.limit:.2f}       {len(over)}")
        print(f"  candidates with no hit      {no_hit}")
        if over:
            worst = sorted(over, key=lambda i: -values[i])[:5]
            for i in worst:
                print(f"    q{i} {values[i]:.3f}  {queries[i]}")
        return len(over)

    failures = summarise("identity over the aligned region (MMseqs2 pident)", best_aligned)
    failures += summarise("identity rescaled to query length (stricter)", best_over_query)

    print()
    if failures == 0:
        print(f"PASS: no candidate exceeds {args.limit:.0%} identity under either definition.")
    else:
        raise SystemExit(f"FAIL: {failures} candidate-definition pairs exceed the limit.")


if __name__ == "__main__":
    main()
