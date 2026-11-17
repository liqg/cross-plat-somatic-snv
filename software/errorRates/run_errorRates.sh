#!/usr/bin/env bash
# Run errorRates-v1.3 on one aligned BAM and write an .er.txt.gz table
# (columns: strand b5 ref b3 base read12 cycle qual count).
#
#   bash software/errorRates/run_errorRates.sh \
#     -b sample.bam -f genome.fasta -l reference/panel/T364V1.hg19.homoref.bed \
#     -o sample.er.txt.gz -t 16
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
BIN="${ERRORRATES_BIN:-$HERE/errorRates-v1.3}"

usage() { echo "usage: $0 -b BAM -f FASTA -l BED -o OUT.er.txt.gz [-t THREADS]" >&2; exit 2; }
BAM=""; FASTA=""; BED=""; OUT=""; THREADS=8
while getopts "b:f:l:o:t:" opt; do
  case "$opt" in
    b) BAM=$OPTARG ;;
    f) FASTA=$OPTARG ;;
    l) BED=$OPTARG ;;
    o) OUT=$OPTARG ;;
    t) THREADS=$OPTARG ;;
    *) usage ;;
  esac
done
[ -n "$BAM" ] && [ -n "$FASTA" ] && [ -n "$BED" ] && [ -n "$OUT" ] || usage
[ -x "$BIN" ] || { echo "errorRates binary not found: $BIN" >&2; exit 1; }

mkdir -p "$(dirname "$OUT")"
"$BIN" -t "$THREADS" -f "$FASTA" -l "$BED" "$BAM" | bgzip -c > "${OUT}.tmp"
mv "${OUT}.tmp" "$OUT"
echo "wrote $OUT"
