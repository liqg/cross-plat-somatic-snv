#!/usr/bin/env bash
# Combine FigS1.pdf ... FigS12.pdf into one A4 PDF (Additional file 2).
#
#   bash analysis/07_figures_tables/combine_supp_figures.sh \
#       "$CPSNV_RESULTS/figures/supplementary" \
#       "$CPSNV_RESULTS/Additional_file_2_Supplementary_figures.pdf"
set -euo pipefail

DIR="${1:?usage: combine_supp_figures.sh <figure-dir> <output.pdf>}"
OUT="${2:?usage: combine_supp_figures.sh <figure-dir> <output.pdf>}"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

# FigS0 is the legend/cover page; FigS1-S12 follow.
for i in $(seq 0 12); do
  src="$DIR/FigS${i}.pdf"
  [ -s "$src" ] || { echo "missing $src" >&2; exit 1; }
  gs -q -sDEVICE=pdfwrite -sPAPERSIZE=a4 -dPDFFitPage -dFIXEDMEDIA \
     -dCompatibilityLevel=1.4 -o "$TMP/page_$(printf '%02d' "$((i+1))").pdf" "$src"
done
pdfunite "$TMP"/page_*.pdf "$OUT"
echo "wrote $OUT"
