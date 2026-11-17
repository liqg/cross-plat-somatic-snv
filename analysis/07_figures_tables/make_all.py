#!/usr/bin/env python3
"""Run the analysis / figure scripts in dependency order.

Steps
-----
  error_rates   errorRates -> error_rates_by_q_clean.csv
  consistency   Fig. 1b
  fig2d         input tables for Fig. 2d (R script draws the panel)
  fig3          input tables for Fig. 3 (R script draws the panels)
  fig4          input tables for Fig. 4 (R script draws the panels)
  supplement    Tables S1-S35 and Additional file 2 figures S1-S12
  all           every step above

Use ``--dry-run`` to print the commands.  Each step is skipped with a warning
if its required input is absent, so a fresh checkout can run the smoke test
without the full sequencing data.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "analysis"))
import common  # noqa: E402
PY = sys.executable


def script(rel):
    return [PY, REPO / rel]


STEPS = {
    "error_rates": [
        script("analysis/01_error_rates/build_error_rates_by_q.py"),
    ],
    "consistency": [
        script("analysis/02_consistency/consistency_from_posfeature.py"),
    ],
    "fig2d": [
        script("analysis/03_benchmark/make_fig2d_data.py"),
    ],
    "fig3": [
        script("analysis/04_model/make_fig3_data.py"),
        script("analysis/04_model/make_fig3_roc.py"),
    ],
    "fig4": [
        script("analysis/06_twist/derive_twist_tables.py"),
    ],
    "supplement": [
        script("analysis/07_figures_tables/build_supplement_v4.py"),
        script("analysis/07_figures_tables/make_bmc_supp_figures.py"),
        ["bash", REPO / "analysis" / "07_figures_tables" / "combine_supp_figures.sh",
         str(common.SUPP_FIG_DIR), str(common.RESULTS_ROOT / "Additional_file_2_Supplementary_figures.pdf")],
    ],
}


def run(cmd, dry):
    print(("DRY-RUN: " if dry else "RUN: ") + " ".join(str(x) for x in cmd), flush=True)
    if not dry:
        subprocess.run([str(x) for x in cmd], check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", default="all", choices=list(STEPS) + ["all"])
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    steps = list(STEPS) if a.step == "all" else [a.step]
    for step in steps:
        for cmd in STEPS[step]:
            run(cmd, a.dry_run)


if __name__ == "__main__":
    main()
