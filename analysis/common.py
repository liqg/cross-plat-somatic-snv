#!/usr/bin/env python3
"""Shared paths, constants and small helpers for the paper analyses.

The repository contains no machine-specific absolute paths.  Every data
location is resolved from one of three environment variables:

    CPSNV_DATA_ROOT   root of the downloaded / derived data tree
                      (default: <repo>/data)
    CPSNV_WORK_ROOT   root of pipeline outputs
                      (default: $CPSNV_DATA_ROOT/work)
    CPSNV_RESULTS     root for tables, figures and logs
                      (default: <repo>/results)

The expected data layout is documented in docs/data_layout.md.  Scripts that
were used for the revision (A1-A4, B1-B1c, C1-C3, D1-D1b, G1, R24, R27)
accept the same layout so that they can be run unchanged on a fresh checkout.
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

DATA_ROOT = Path(os.environ.get("CPSNV_DATA_ROOT", REPO_ROOT / "data")).resolve()
WORK_ROOT = Path(os.environ.get("CPSNV_WORK_ROOT", DATA_ROOT / "work")).resolve()
RESULTS_ROOT = Path(os.environ.get("CPSNV_RESULTS", REPO_ROOT / "results")).resolve()

TABLES = RESULTS_ROOT / "tables"
FIGURES = RESULTS_ROOT / "figures"
LOGS = RESULTS_ROOT / "logs"
INTERMEDIATE = RESULTS_ROOT / "intermediate"
SUPP_FIG_DIR = FIGURES / "supplementary"
for _d in (TABLES, FIGURES, LOGS, INTERMEDIATE, SUPP_FIG_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Reference inputs
# ---------------------------------------------------------------------------
REF_DIR = REPO_ROOT / "reference"
GENOME_FASTA = Path(os.environ.get("CPSNV_GENOME", DATA_ROOT / "reference" / "GRCh37" / "genome.fasta"))
PANEL_BED = Path(os.environ.get("CPSNV_PANEL_BED", REF_DIR / "panel" / "T364V1.hg19.bed"))
GIAB_DIR = Path(os.environ.get("CPSNV_GIAB", DATA_ROOT / "giab" / "v4.2.1"))
GIAB_HC_DIR = Path(os.environ.get("CPSNV_GIAB_HC", DATA_ROOT / "giab" / "highconf"))

# ---------------------------------------------------------------------------
# Pipeline outputs (written by workflow/)
# ---------------------------------------------------------------------------
ALIGN_ROOT = WORK_ROOT / "0.align" / "pipeline" / "bwamem"

# Directory names follow the original analysis working tree so that the
# revision scripts can be re-used unchanged.
BENCH_ROOT = WORK_ROOT / "3.performance"       # single-pair benchmark (Fig. 2d)
TEST_ROOT = WORK_ROOT / "4.test"               # test pairs 45 and 54 (Fig. 3)
TRAIN_ROOT = WORK_ROOT / "4.train"             # six training pairs (12-32)
ER_1_ROOT = WORK_ROOT / "1.errorRates"       # full-depth batch
ER_7_ROOT = WORK_ROOT / "7.errorRates_bq30_PE70M"
ER_9_ROOT = WORK_ROOT / "9.errorRates_bq30_twist_wt_PE15M"
CONSISTENCY_ROOT = WORK_ROOT / "2.consistency" # Fig. 1b
TWIST_PHYS_ROOT = WORK_ROOT / "6.twist_pair"
TWIST_MIX_ROOT = WORK_ROOT / "8.twist_HG001_mix"

# SomaticXGB feature matrices and predictions
XGB_TRAIN_ROOT = WORK_ROOT / "5.somaticxgb" / "train"
XGB_TEST_ROOT = WORK_ROOT / "5.somaticxgb" / "test"
TRAIN_MATRIX = Path(
    os.environ.get("CPSNV_TRAIN_MATRIX", XGB_TRAIN_ROOT / "feature_matrix.tsv.gz")
)

# Inputs that are not produced by this repository
TWIST_TRUTH_VCF = Path(
    os.environ.get(
        "CPSNV_TWIST_TRUTH", DATA_ROOT / "twist" / "truth" / "twist_cfdna_pancancer_v1.hg19.vcf"
    )
)
TWIST_TRUTH_221 = Path(
    os.environ.get("CPSNV_TWIST_TRUTH_221", INTERMEDIATE / "twist_truth_221.tsv")
)
GIAB_PANEL_VARS = Path(
    os.environ.get("CPSNV_GIAB_PANEL_VARS", INTERMEDIATE / "giab_panel_vars.pkl")
)
TRAIN_ONLY_DIR = Path(
    os.environ.get("CPSNV_TRAIN_ONLY_DIR", REPO_ROOT / "workflow" / "pos_files" / "train")
)

# ---------------------------------------------------------------------------
# Platforms
# ---------------------------------------------------------------------------
PLATFORM_ORDER = [
    "FASTASeq 300",
    "GenoLab M",
    "SURFSeq 5000",
    "NovaSeq 6000",
    "NovaSeq X",
    "DNBSEQ-T7",
]
TRUSEQ_PLATFORMS = [p for p in PLATFORM_ORDER if p != "DNBSEQ-T7"]

PLATFORM_MAP = {
    "FA": "FASTASeq 300",
    "FASTASeq_300": "FASTASeq 300",
    "GL": "GenoLab M",
    "GenoLab_M": "GenoLab M",
    "SF": "SURFSeq 5000",
    "SURFSeq_5000": "SURFSeq 5000",
    "NV": "NovaSeq 6000",
    "NovaSeq_6000": "NovaSeq 6000",
    "NX": "NovaSeq X",
    "NovaSeq_X": "NovaSeq X",
    "T7": "DNBSEQ-T7",
    "MGISEQ-T7": "DNBSEQ-T7",
    "DNBSEQ-T7": "DNBSEQ-T7",
}

# Colour-blind-safe palette used in the manuscript figures.
PLATFORM_COLORS = {
    "FASTASeq 300": "#D55E00",
    "GenoLab M": "#0072B2",
    "SURFSeq 5000": "#009E73",
    "NovaSeq 6000": "#CC79A7",
    "NovaSeq X": "#E69F00",
    "DNBSEQ-T7": "#8C564B",
}

READ_DEPTH_LABELS = {25: "900X", 50: "1800X", 75: "2700X", 100: "3600X", 125: "4500X"}

POSITION_COLUMNS = [
    "CHROM", "POS", "P", "REF", "ALT", "GT", "f", "b", "GT2", "group", "VAF",
]


def platform_name(value: str) -> str:
    """Normalise a platform ID such as ``SURFSeq_5000`` to its display name."""
    return PLATFORM_MAP.get(value, value.replace("_", " "))


def ensure_dirs(*paths: Path) -> None:
    for p in paths:
        Path(p).mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Small I/O helpers
# ---------------------------------------------------------------------------
def read_position_table(path, **kwargs):
    """Read a PosMix position-frequency table (no header, 11 columns)."""
    import pandas as pd

    df = pd.read_csv(path, sep="\t", header=None, names=POSITION_COLUMNS, **kwargs)
    df["CHROM"] = df["CHROM"].astype(str).str.replace("^chr", "", regex=True)
    df["POS"] = pd.to_numeric(df["POS"], errors="coerce").astype("Int64")
    df["var"] = (
        df["CHROM"].astype(str) + ":" + df["POS"].astype(str) + ":"
        + df["REF"].astype(str) + ":" + df["ALT"].astype(str)
    )
    return df


def nearest_distance(pos, chrom_positions):
    """Distance from each position to the nearest position in ``chrom_positions``."""
    import numpy as np

    pos = np.asarray(pos, dtype=np.int64)
    ref = np.sort(np.asarray(chrom_positions, dtype=np.int64))
    if len(ref) == 0:
        return np.full(len(pos), np.iinfo(np.int64).max, dtype=np.int64)
    idx = np.searchsorted(ref, pos)
    left = np.where(idx > 0, pos - ref[np.clip(idx - 1, 0, len(ref) - 1)], np.iinfo(np.int64).max)
    right = np.where(idx < len(ref), ref[np.clip(idx, 0, len(ref) - 1)] - pos, np.iinfo(np.int64).max)
    return np.minimum(left, right)


def add_target_distance(df, dist_col: str = "dist"):
    """Within-chromosome distance to the nearest other target (capped at 10 kb)."""
    import numpy as np

    out = np.full(len(df), 10000, dtype=np.int64)
    for _, sub in df.groupby("CHROM", sort=False):
        pos = sub["POS"].astype(np.int64).values
        n = len(pos)
        if n == 1:
            d = np.array([10000])
        else:
            d = np.full(n, 10000, dtype=np.int64)
            if n > 1:
                gaps = np.diff(pos)
                d[1:] = np.minimum(d[1:], gaps)
                d[:-1] = np.minimum(d[:-1], gaps)
        out[sub.index.to_numpy()] = d
    df = df.copy()
    df[dist_col] = out
    return df


def vcf_variants(path, pass_only: bool = True):
    """Return a set of ``CHROM:POS:REF:ALT`` SNV keys from a VCF/VCF.gz.

    Uses ``bcftools`` (available in the Docker image).  Multi-allelic sites are
    skipped, matching the manuscript's SNV evaluation.
    """
    cmd = ["bcftools", "query", "-f", "%CHROM\t%POS\t%REF\t%ALT\t%FILTER\n", str(path)]
    res = subprocess.run(cmd, capture_output=True, text=True, check=True)
    out = set()
    for line in res.stdout.splitlines():
        if not line or line.startswith("#"):
            continue
        chrom, pos, ref, alt, flt = line.split("\t")
        if pass_only and flt not in ("PASS", ".") and "PASS" not in flt:
            continue
        if len(ref) == 1 and len(alt) == 1 and "," not in alt:
            out.add(f"{chrom.replace('chr', '')}:{pos}:{ref}:{alt}")
    return out


def metrics(tp: int, fp: int, fn: int) -> dict:
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    f1 = 2 * recall * precision / (recall + precision) if (recall + precision) else 0.0
    return {"Recall": recall, "Precision": precision, "F1": f1}
