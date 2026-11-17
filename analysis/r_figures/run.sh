#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
Rscript 01_fig1b_fig2d_fig3.R
Rscript 02_fig4.R
