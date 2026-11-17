#!/usr/bin/env python3
"""A3: multi-threshold and per-quality mismatch rates (Reviewer 2, Comment 8).

Reads ``.er.txt.gz`` tables with columns
``strand b5 ref b3 base read12 cycle qual count`` and computes, for every
sequencing library:

* retained bases, mismatches, mismatch rate and fraction of bases retained at
  quality thresholds Q = 0, 10, 20, 25, 30, 35 and 40;
* bases, mismatches and mismatch rate at each reported quality value 0-40.

Results are aggregated per sample and per platform (mean / SD / range across
libraries).  The script also reproduces the published single-threshold table
``results/analysis/error_rates_by_q_clean.csv`` from the ``7.errorRates`` files and
writes an auditable comparison, plus a comparison of the same library IDs
processed in ``1.errorRates`` versus ``7.errorRates``.

Data sources and de-duplication
-------------------------------
1. ``1.errorRates/pipeline/bwamem``: the paper's GIAB libraries.  It contains
   60 files for the five TruSeq platforms (12 libraries x 5 platforms) and 15
   DNBSEQ-T7 files (3 libraries per donor).  Only the 60 non-T7 files enter the
   primary analysis.
2. ``7.errorRates_bq30_PE70M/pipeline/bwamem``: bq30/PE70M reprocessing of a
   subset of the same libraries (25 non-T7 + 5 T7).  The 25 non-T7 files are
   *not* independent: they are the same 5 libraries per platform as in
   ``1.errorRates`` after down-sampling/trimming (e.g. FA_HG001 Q0 total bases
   9.04e9 versus 1.36e10 in ``1.errorRates``).  They are excluded from the primary
   tables and used only to reproduce the published CSV.  The 5 T7 files are the
   primary T7 dataset, as requested.
3. ``9.errorRates_bq30_twist_wt_PE15M/pipeline/bwamem``: 5 Twist wild-type files,
   one per platform, all primary.

No file is used twice in the primary sample set.  Within the raw inventory,
30 sample IDs occur in both ``1.errorRates`` and ``7.errorRates`` but the files differ
(md5 and base totals); this is logged explicitly.

Usage
-----
    /usr/bin/python3 A3_error_rates.py

All paths have defaults under the project and addendum directories; override
with ``--base-dir`` and ``--out-dir`` if needed.
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
import common  # noqa: E402

import argparse
import gzip
import hashlib
import logging
import re
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
DEFAULT_BASE = common.WORK_ROOT
DEFAULT_OUT = common.RESULTS_ROOT
DEFAULT_ANALYSIS_DIR = common.INTERMEDIATE

ROOTS = {
    "1.errorRates": "1.errorRates/pipeline/bwamem",
    "7.errorRates_bq30_PE70M": "7.errorRates_bq30_PE70M/pipeline/bwamem",
    "9.errorRates_bq30_twist_wt_PE15M": "9.errorRates_bq30_twist_wt_PE15M/pipeline/bwamem",
}
PROCESSING = {
    "1.errorRates": "full",
    "7.errorRates_bq30_PE70M": "bq30_PE70M",
    "9.errorRates_bq30_twist_wt_PE15M": "bq30_PE15M",
}
PLATFORM_NAMES = {
    "FA": "FASTASeq 300",
    "GL": "GenoLab M",
    "SF": "SURFSeq 5000",
    "NV": "NovaSeq 6000",
    "NX": "NovaSeq X",
    "T7": "DNBSEQ-T7",
}
PLATFORM_ORDER = ["FA", "GL", "SF", "NV", "NX", "T7"]
COLORS = {
    "FASTASeq 300": "#e41a1c",
    "GenoLab M": "#377eb8",
    "SURFSeq 5000": "#4daf4a",
    "NovaSeq 6000": "#984ea3",
    "NovaSeq X": "#ff7f00",
    "DNBSEQ-T7": "#a65628",
}
# Short codes are used in file names; the full names in tables and figures.
EXPECTED_COLUMNS = [
    "strand",
    "b5",
    "ref",
    "b3",
    "base",
    "read12",
    "cycle",
    "qual",
    "count",
]
REQUIRED_COLUMNS = {"ref", "base", "qual", "count"}
MAX_QUAL = 40
QUALITIES = np.arange(0, MAX_QUAL + 1, dtype=int)
THRESHOLDS = [0, 10, 20, 25, 30, 35, 40]
# Figure points supported by fewer than this many bases are omitted from the
# plots (but retained in the tables, with support_flag="low_bases").  This
# matters for the extreme T7 tail: Q39/Q40 carry only ~450/800 bases out of
# ~4.8e10, so their mismatch rates are not interpretable.
MIN_PLOT_BASES = 1_000_000


# --------------------------------------------------------------------------
# Logging
# --------------------------------------------------------------------------
def setup_logging(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("A3")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    logger.propagate = False
    return logger


def log_dataframe(logger: logging.Logger, name: str, df: pd.DataFrame, max_rows: int = 80) -> None:
    if df.empty:
        logger.info("%s: <empty>", name)
        return
    with pd.option_context("display.width", 220, "display.max_columns", 50):
        text = df.to_string(index=False, max_rows=max_rows)
    logger.info("%s:\n%s", name, text)


# --------------------------------------------------------------------------
# File discovery
# --------------------------------------------------------------------------
def md5sum(path: Path, block: int = 1 << 20) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(block), b""):
            h.update(chunk)
    return h.hexdigest()


def discover_files(base: Path, logger: logging.Logger) -> list[dict]:
    """Find all .er.txt.gz files and annotate platform / sample / role."""
    records: list[dict] = []
    seen_sample_per_root: dict[str, set[str]] = {}
    for batch, rel in ROOTS.items():
        root = base / rel
        if not root.is_dir():
            logger.error("Missing root directory: %s", root)
            continue
        paths = sorted(root.glob("*/*.er.txt.gz"))
        logger.info("Root %-32s %s  (%d files)", batch, root, len(paths))
        seen_sample_per_root[batch] = set()
        for path in paths:
            sample = path.parent.name
            if path.name != f"{sample}.er.txt.gz":
                logger.warning("Unexpected file name (parent=%s): %s", sample, path)
            if sample in seen_sample_per_root[batch]:
                logger.error("Duplicate sample ID within %s: %s", batch, sample)
            seen_sample_per_root[batch].add(sample)

            prefix = sample.split("_", 1)[0]
            if prefix not in PLATFORM_NAMES:
                logger.error("Unknown platform prefix %r for %s", prefix, sample)
                continue
            platform = PLATFORM_NAMES[prefix]
            hg_match = re.search(r"(HG\d{3})", sample)
            if hg_match:
                hg = hg_match.group(1)
            elif "twist" in sample.lower():
                hg = "twist_wt"
            else:
                hg = "NA"
                logger.warning("Could not parse donor from sample name: %s", sample)

            # Primary-set rules and exclusion reasons (see module docstring).
            if batch == "1.errorRates" and prefix != "T7":
                dataset = "GIAB"
                include, reason = True, ""
            elif batch == "1.errorRates" and prefix == "T7":
                dataset = "T7"
                include = False
                reason = (
                    "Excluded from primary: T7 is represented by the 7.errorRates "
                    "files; five of these 15 libraries share library IDs with "
                    "7.errorRates and would otherwise be counted twice."
                )
            elif batch == "7.errorRates_bq30_PE70M" and prefix == "T7":
                dataset = "T7"
                include, reason = True, ""
            elif batch == "7.errorRates_bq30_PE70M" and prefix != "T7":
                dataset = "GIAB"
                include = False
                reason = (
                    "Excluded from primary: bq30/PE70M reprocessing of GIAB "
                    "libraries already represented by 1.errorRates (same library "
                    "IDs, ~65% of the bases); used only for the sanity check."
                )
            elif batch == "9.errorRates_bq30_twist_wt_PE15M":
                dataset = "Twist-WT"
                include, reason = True, ""
            else:
                dataset = "NA"
                include, reason = False, "Unclassified"

            records.append(
                {
                    "source_batch": batch,
                    "processing": PROCESSING[batch],
                    "dataset": dataset,
                    "platform": platform,
                    "platform_short": prefix,
                    "sample": sample,
                    "hg": hg,
                    "path": str(path),
                    "include_primary": include,
                    "exclusion_reason": reason,
                }
            )
    return records


# --------------------------------------------------------------------------
# Per-file aggregation
# --------------------------------------------------------------------------
def check_header(path: Path, logger: logging.Logger) -> tuple[bool, list[str]]:
    try:
        with gzip.open(path, "rt", encoding="utf-8", errors="replace") as fh:
            first = fh.readline().rstrip("\n").rstrip("\r")
    except Exception as exc:  # noqa: BLE001
        logger.error("Cannot read header of %s: %s", path, exc)
        return False, []
    cols = first.split("\t")
    if cols and cols[0].startswith("﻿"):
        cols[0] = cols[0].lstrip("﻿")
    if not REQUIRED_COLUMNS.issubset(set(cols)):
        logger.error("Missing required columns in %s: header=%s", path, cols)
        return False, cols
    if cols != EXPECTED_COLUMNS:
        logger.warning("Unexpected column order/set in %s: %s", path, cols)
    return True, cols


def aggregate_file(
    rec: dict, chunksize: int, logger: logging.Logger
) -> dict:
    """Stream one .er.txt.gz and accumulate per-quality bases/mismatches.

    Quality values are clipped to [0, MAX_QUAL] for accumulation; values above
    MAX_QUAL are extremely rare (none expected) and are logged.  The Q=40
    threshold therefore means "qual >= 40".
    """
    path = Path(rec["path"])
    total = 0.0
    bases = np.zeros(MAX_QUAL + 1, dtype=np.float64)
    mism = np.zeros(MAX_QUAL + 1, dtype=np.float64)
    n_rows = 0
    qual_min = None
    qual_max = None
    qual_values: set[int] = set()
    n_bad = 0
    n_negative = 0
    n_above_max = 0
    non_acgt = 0
    error = ""

    try:
        reader = pd.read_csv(
            path,
            sep="\t",
            usecols=["ref", "base", "qual", "count"],
            dtype={"ref": str, "base": str},
            chunksize=chunksize,
            compression="gzip",
            keep_default_na=False,
            na_values=[],
        )
        for chunk in reader:
            n_rows += len(chunk)
            ref = chunk["ref"].astype(str).str.strip().str.upper()
            base = chunk["base"].astype(str).str.strip().str.upper()
            q_raw = pd.to_numeric(chunk["qual"], errors="coerce")
            c_raw = pd.to_numeric(chunk["count"], errors="coerce")

            bad = q_raw.isna() | c_raw.isna() | ~np.isfinite(c_raw)
            if bad.any():
                n_bad += int(bad.sum())
                if n_bad <= 5:
                    logger.warning(
                        "%s: %d non-numeric/NaN rows in chunk; dropping",
                        rec["sample"],
                        int(bad.sum()),
                    )
                ref = ref[~bad]
                base = base[~bad]
                q_raw = q_raw[~bad]
                c_raw = c_raw[~bad]
            if len(c_raw) == 0:
                continue

            # Non-ACGT reference/observed bases are kept (mismatch = ref != base)
            non_acgt += int((~ref.isin(["A", "C", "G", "T"])).sum())
            non_acgt += int((~base.isin(["A", "C", "G", "T"])).sum())

            neg = c_raw < 0
            if neg.any():
                n_negative += int(neg.sum())
                c_raw = c_raw.copy()
                c_raw[neg] = 0.0

            q_int = np.rint(q_raw.to_numpy(dtype=np.float64)).astype(np.int64)
            if np.any(q_raw.to_numpy(dtype=np.float64) != q_int):
                logger.warning("%s: non-integer quality values were rounded", rec["sample"])
            q_min_chunk = int(q_int.min())
            q_max_chunk = int(q_int.max())
            qual_min = q_min_chunk if qual_min is None else min(qual_min, q_min_chunk)
            qual_max = q_max_chunk if qual_max is None else max(qual_max, q_max_chunk)
            qual_values.update(np.unique(q_int).tolist())

            above = q_int > MAX_QUAL
            if above.any():
                n_above_max += int(above.sum())
                q_int = np.clip(q_int, 0, MAX_QUAL)
            else:
                q_int = np.clip(q_int, 0, MAX_QUAL)
            if np.any(q_int < 0):
                logger.warning("%s: negative quality values clipped to 0", rec["sample"])

            c = c_raw.to_numpy(dtype=np.float64)
            mm = (ref.to_numpy() != base.to_numpy())
            bases += np.bincount(q_int, weights=c, minlength=MAX_QUAL + 1)
            if mm.any():
                mism += np.bincount(
                    q_int[mm], weights=c[mm], minlength=MAX_QUAL + 1
                )
            total += float(c.sum())
    except Exception as exc:  # noqa: BLE001
        error = f"{type(exc).__name__}: {exc}"
        logger.error("FAILED reading %s: %s", path, error)

    if n_bad:
        logger.warning("%s: total dropped non-numeric rows = %d", rec["sample"], n_bad)
    if n_negative:
        logger.warning("%s: rows with negative count = %d", rec["sample"], n_negative)
    if n_above_max:
        logger.warning(
            "%s: rows with qual > %d = %d (accumulated in the Q>=40 bucket)",
            rec["sample"], MAX_QUAL, n_above_max,
        )
    if non_acgt:
        logger.warning("%s: non-ACGT ref/base rows = %d", rec["sample"], non_acgt)

    rec.update(
        {
            "status": "ok" if not error and total > 0 else "error",
            "error": error,
            "n_rows": n_rows,
            "total_bases": int(round(total)),
            "bases_by_q": bases,
            "mism_by_q": mism,
            "qual_min": qual_min,
            "qual_max": qual_max,
            "n_qual_levels": len(qual_values),
        }
    )
    return rec


# --------------------------------------------------------------------------
# Table construction
# --------------------------------------------------------------------------
SAMPLE_THRESHOLD_COLUMNS = [
    "dataset",
    "platform",
    "platform_short",
    "sample",
    "hg",
    "source_batch",
    "processing",
    "quality_threshold",
    "total_bases",
    "retained_bases",
    "mismatches",
    "error_rate",
    "fraction_retained",
]


def sample_threshold_table(records: list[dict]) -> pd.DataFrame:
    rows = []
    for rec in records:
        if rec.get("status") != "ok":
            continue
        bases = rec["bases_by_q"]
        mism = rec["mism_by_q"]
        total = float(rec["total_bases"])
        for q in THRESHOLDS:
            retained = float(bases[q:].sum())
            n_mism = float(mism[q:].sum())
            rows.append(
                {
                    "dataset": rec["dataset"],
                    "platform": rec["platform"],
                    "platform_short": rec["platform_short"],
                    "sample": rec["sample"],
                    "hg": rec["hg"],
                    "source_batch": rec["source_batch"],
                    "processing": rec["processing"],
                    "quality_threshold": q,
                    "total_bases": int(rec["total_bases"]),
                    "retained_bases": int(round(retained)),
                    "mismatches": int(round(n_mism)),
                    "error_rate": (n_mism / retained) if retained > 0 else np.nan,
                    "fraction_retained": (retained / total) if total > 0 else np.nan,
                }
            )
    df = pd.DataFrame(rows, columns=SAMPLE_THRESHOLD_COLUMNS)
    return df.sort_values(
        ["dataset", "platform", "sample", "quality_threshold"], kind="stable"
    ).reset_index(drop=True)


def sample_per_quality_table(records: list[dict]) -> pd.DataFrame:
    rows = []
    for rec in records:
        if rec.get("status") != "ok":
            continue
        bases = rec["bases_by_q"]
        mism = rec["mism_by_q"]
        total = float(rec["total_bases"])
        for q in QUALITIES:
            b = float(bases[q])
            m = float(mism[q])
            rows.append(
                {
                    "dataset": rec["dataset"],
                    "platform": rec["platform"],
                    "platform_short": rec["platform_short"],
                    "sample": rec["sample"],
                    "hg": rec["hg"],
                    "source_batch": rec["source_batch"],
                    "processing": rec["processing"],
                    "qual": int(q),
                    "bases": int(round(b)),
                    "mismatches": int(round(m)),
                    "error_rate": (m / b) if b > 0 else np.nan,
                    "fraction_of_sample_bases": (b / total) if total > 0 else np.nan,
                }
            )
    return pd.DataFrame(rows)


def platform_threshold_table(st: pd.DataFrame) -> pd.DataFrame:
    rows = []
    group_cols = ["dataset", "platform", "platform_short", "quality_threshold"]
    for keys, g in st.groupby(group_cols, sort=False, dropna=False):
        dataset, platform, pshort, q = keys
        valid = g[g["retained_bases"] > 0]
        err = valid["error_rate"].astype(float)
        frac = g["fraction_retained"].astype(float)
        total_bases = int(g["total_bases"].sum())
        retained = int(g["retained_bases"].sum())
        n_mism = int(g["mismatches"].sum())
        rows.append(
            {
                "dataset": dataset,
                "platform": platform,
                "platform_short": pshort,
                "quality_threshold": int(q),
                "n_samples": int(len(g)),
                "n_samples_with_retained_bases": int(len(valid)),
                "mean_error_rate": err.mean(),
                "sd_error_rate": err.std(ddof=1),
                "min_error_rate": err.min(),
                "max_error_rate": err.max(),
                "median_error_rate": err.median(),
                "mean_fraction_retained": frac.mean(),
                "sd_fraction_retained": frac.std(ddof=1),
                "min_fraction_retained": frac.min(),
                "max_fraction_retained": frac.max(),
                "min_sample_retained_bases": int(g["retained_bases"].min()),
                "max_sample_retained_bases": int(g["retained_bases"].max()),
                "total_bases": total_bases,
                "total_retained_bases": retained,
                "total_mismatches": n_mism,
                "pooled_error_rate": (n_mism / retained) if retained > 0 else np.nan,
                "pooled_fraction_retained": (retained / total_bases) if total_bases else np.nan,
                "support_flag": "ok" if retained >= MIN_PLOT_BASES else "low_bases",
            }
        )
    df = pd.DataFrame(rows)
    # Qualitative platform order, then threshold.
    order = {p: i for i, p in enumerate([PLATFORM_NAMES[x] for x in PLATFORM_ORDER])}
    df["_porder"] = df["platform"].map(order).fillna(99).astype(int)
    df = df.sort_values(
        ["dataset", "_porder", "quality_threshold"], kind="stable"
    ).drop(columns="_porder")
    return df.reset_index(drop=True)


def platform_per_quality_table(pq: pd.DataFrame) -> pd.DataFrame:
    rows = []
    group_cols = ["dataset", "platform", "platform_short", "qual"]
    for keys, g in pq.groupby(group_cols, sort=False, dropna=False):
        dataset, platform, pshort, q = keys
        valid = g[g["bases"] > 0]
        err = valid["error_rate"].astype(float)
        frac = g["fraction_of_sample_bases"].astype(float)
        bases = int(g["bases"].sum())
        mism = int(g["mismatches"].sum())
        rows.append(
            {
                "dataset": dataset,
                "platform": platform,
                "platform_short": pshort,
                "qual": int(q),
                "n_samples": int(len(g)),
                "n_samples_with_bases": int(len(valid)),
                "mean_error_rate": err.mean(),
                "sd_error_rate": err.std(ddof=1),
                "min_error_rate": err.min(),
                "max_error_rate": err.max(),
                "pooled_error_rate": (mism / bases) if bases else np.nan,
                "mean_bases": float(valid["bases"].mean()) if len(valid) else np.nan,
                "min_sample_bases": int(valid["bases"].min()) if len(valid) else 0,
                "max_sample_bases": int(valid["bases"].max()) if len(valid) else 0,
                "total_bases": bases,
                "total_mismatches": mism,
                "mean_fraction_of_sample_bases": frac.mean(),
                "min_fraction_of_sample_bases": frac.min(),
                "max_fraction_of_sample_bases": frac.max(),
                "support_flag": "ok" if bases >= MIN_PLOT_BASES else "low_bases",
            }
        )
    df = pd.DataFrame(rows)
    order = {p: i for i, p in enumerate([PLATFORM_NAMES[x] for x in PLATFORM_ORDER])}
    df["_porder"] = df["platform"].map(order).fillna(99).astype(int)
    df = df.sort_values(["dataset", "_porder", "qual"], kind="stable").drop(columns="_porder")
    return df.reset_index(drop=True)


# --------------------------------------------------------------------------
# Sanity checks
# --------------------------------------------------------------------------
def compare_thresholds_with_existing(
    ours: pd.DataFrame, analysis_dir: Path
) -> tuple[pd.DataFrame, dict]:
    path = analysis_dir / "error_rates_by_q_clean.csv"
    clean = pd.read_csv(path)
    clean = clean.rename(columns={"q": "quality_threshold"})
    right = ours[
        [
            "sample",
            "quality_threshold",
            "retained_bases",
            "mismatches",
            "error_rate",
            "fraction_retained",
        ]
    ].rename(
        columns={
            "retained_bases": "retained_bases_a3",
            "mismatches": "mismatches_a3",
            "error_rate": "error_rate_a3",
            "fraction_retained": "fraction_retained_a3",
        }
    )
    merged = clean.merge(
        right, on=["sample", "quality_threshold"], how="outer", indicator=True
    )
    merged["abs_diff_error_rate"] = (merged["err"] - merged["error_rate_a3"]).abs()
    denom = merged["err"].abs().where(merged["err"].abs() > 0)
    merged["rel_diff_error_rate"] = merged["abs_diff_error_rate"] / denom
    merged["abs_diff_fraction_retained"] = (
        merged["frac"] - merged["fraction_retained_a3"]
    ).abs()
    merged["n_diff"] = merged["n"] - merged["retained_bases_a3"]

    both = merged[merged["_merge"] == "both"]
    # Compare only rows where both sides have a finite rate (Q40 can be empty).
    finite = both[np.isfinite(both["err"]) & np.isfinite(both["error_rate_a3"])]
    nan_both = both[~np.isfinite(both["err"]) & ~np.isfinite(both["error_rate_a3"])]
    stats = {
        "existing_file": str(path),
        "existing_rows": int(len(clean)),
        "a3_rows": int(len(ours)),
        "merged_rows": int(len(merged)),
        "samples_existing": int(clean["sample"].nunique()),
        "samples_a3": int(ours["sample"].nunique()),
        "samples_overlap": int(both["sample"].nunique()),
        "rows_both": int(len(both)),
        "only_existing": int((merged["_merge"] == "left_only").sum()),
        "only_a3": int((merged["_merge"] == "right_only").sum()),
        "n_exact_matches": int((both["n_diff"] == 0).sum()),
        "n_rows_finite_rate": int(len(finite)),
        "max_abs_diff_error_rate": float(finite["abs_diff_error_rate"].max()) if len(finite) else np.nan,
        "max_rel_diff_error_rate": float(finite["rel_diff_error_rate"].max()) if len(finite) else np.nan,
        "max_abs_diff_fraction_retained": float(both["abs_diff_fraction_retained"].max()) if len(both) else np.nan,
        "n_rows_nan_rate_both": int(len(nan_both)),
        "n_rows_rate_mismatch_nan": int(len(both) - len(finite) - len(nan_both)),
    }
    # The existing clean table does not carry mismatch counts; keep only the
    # columns that exist on both sides.
    cols = [
        "sample",
        "platform",
        "hg",
        "quality_threshold",
        "frac",
        "fraction_retained_a3",
        "abs_diff_fraction_retained",
        "err",
        "error_rate_a3",
        "abs_diff_error_rate",
        "rel_diff_error_rate",
        "n",
        "retained_bases_a3",
        "n_diff",
        "mismatches_a3",
        "_merge",
    ]
    out = merged[[c for c in cols if c in merged.columns]].rename(
        columns={
            "fraction_retained_a3": "fraction_retained",
            "error_rate_a3": "error_rate",
            "retained_bases_a3": "retained_bases",
            "mismatches_a3": "mismatches",
        }
    )
    return out, stats


def compare_per_quality_with_existing(
    ours: pd.DataFrame, analysis_dir: Path
) -> tuple[pd.DataFrame, dict]:
    path = analysis_dir / "error_rates_by_q.csv"
    existing = pd.read_csv(path)
    n_existing_raw = len(existing)
    # The existing file concatenates the five cumulative-threshold rows
    # (0/10/20/30/40) first and the per-quality rows afterwards.  For quality
    # values that are both a threshold and an observed score there are two
    # rows; keep the appended per-quality row (last occurrence).
    existing = existing.drop_duplicates(["sample", "qual"], keep="last").copy()
    n_duplicates_removed = n_existing_raw - len(existing)

    merged = existing.merge(
        ours[["sample", "qual", "bases", "mismatches", "error_rate", "fraction_of_sample_bases"]],
        on=["sample", "qual"],
        how="outer",
        indicator=True,
    )
    both = merged[merged["_merge"] == "both"].copy()
    # Only qualities at which this sample actually has bases are comparable
    # at the per-quality level; cumulative threshold rows at unobserved scores
    # (e.g. Q40 for FASTASeq) are not per-quality observations.
    both = both[both["bases"] > 0].copy()
    both["abs_diff_error_rate"] = (both["err_rate"] - both["error_rate"]).abs()
    denom = both["err_rate"].abs().where(both["err_rate"].abs() > 0)
    both["rel_diff_error_rate"] = both["abs_diff_error_rate"] / denom
    both["abs_diff_fraction_retained"] = (
        both["frac_retained"] - both["fraction_of_sample_bases"]
    ).abs()
    both["n_diff"] = both["n"] - both["bases"]
    both["mismatch_diff"] = both["err"] - both["mismatches"]
    finite = both[np.isfinite(both["err_rate"]) & np.isfinite(both["error_rate"])]
    stats = {
        "existing_file": str(path),
        "existing_rows_raw": int(n_existing_raw),
        "existing_duplicate_rows_removed": int(n_duplicates_removed),
        "existing_rows_deduplicated": int(len(existing)),
        "a3_rows": int(len(ours)),
        "rows_compared_with_bases": int(len(both)),
        "n_exact_matches": int((both["n_diff"] == 0).sum()),
        "n_rows_finite_rate": int(len(finite)),
        "max_abs_diff_error_rate": float(finite["abs_diff_error_rate"].max()) if len(finite) else np.nan,
        "max_rel_diff_error_rate": float(finite["rel_diff_error_rate"].max()) if len(finite) else np.nan,
        "max_abs_diff_fraction_retained": float(both["abs_diff_fraction_retained"].max()) if len(both) else np.nan,
        "max_abs_diff_mismatches": float(both["mismatch_diff"].abs().max()),
    }
    cols = [
        "sample",
        "qual",
        "platform",
        "n",
        "bases",
        "n_diff",
        "err_rate",
        "error_rate",
        "abs_diff_error_rate",
        "rel_diff_error_rate",
        "frac_retained",
        "fraction_of_sample_bases",
        "abs_diff_fraction_retained",
        "err",
        "mismatches",
        "mismatch_diff",
    ]
    return both[[c for c in cols if c in both.columns]], stats


def source_comparison(
    all_thresholds: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare 1.errorRates and 7.errorRates for the same 30 sample IDs."""
    a = all_thresholds[all_thresholds["source_batch"] == "1.errorRates"]
    b = all_thresholds[all_thresholds["source_batch"] == "7.errorRates_bq30_PE70M"]
    common = sorted(set(a["sample"]) & set(b["sample"]))
    a = a[a["sample"].isin(common)]
    b = b[b["sample"].isin(common)]
    merged = a.merge(
        b,
        on=["sample", "platform", "hg", "quality_threshold"],
        suffixes=("_batch1", "_batch7"),
        how="inner",
    )
    merged["abs_diff_error_rate"] = (
        merged["error_rate_batch1"] - merged["error_rate_batch7"]
    ).abs()
    denom = merged["error_rate_batch1"].abs().where(
        merged["error_rate_batch1"].abs() > 0
    )
    merged["rel_diff_error_rate"] = merged["abs_diff_error_rate"] / denom
    merged["abs_diff_fraction_retained"] = (
        merged["fraction_retained_batch1"] - merged["fraction_retained_batch7"]
    ).abs()
    cols = [
        "sample",
        "platform",
        "hg",
        "quality_threshold",
        "total_bases_batch1",
        "total_bases_batch7",
        "error_rate_batch1",
        "error_rate_batch7",
        "abs_diff_error_rate",
        "rel_diff_error_rate",
        "fraction_retained_batch1",
        "fraction_retained_batch7",
        "abs_diff_fraction_retained",
    ]
    detail = merged[cols].sort_values(
        ["platform", "sample", "quality_threshold"]
    )
    summary = (
        merged.groupby("quality_threshold")
        .agg(
            n=("sample", "size"),
            median_rel_diff_error_rate=("rel_diff_error_rate", "median"),
            max_rel_diff_error_rate=("rel_diff_error_rate", "max"),
            median_abs_diff_error_rate=("abs_diff_error_rate", "median"),
            max_abs_diff_error_rate=("abs_diff_error_rate", "max"),
            median_abs_diff_fraction_retained=("abs_diff_fraction_retained", "median"),
        )
        .reset_index()
    )
    return detail, summary


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------
def setup_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10.5,
            "axes.titlesize": 11.5,
            "axes.labelsize": 10.5,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "legend.fontsize": 8.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _platform_order(df: pd.DataFrame) -> list[str]:
    present = set(df["platform"])
    return [PLATFORM_NAMES[p] for p in PLATFORM_ORDER if PLATFORM_NAMES[p] in present]


def _positive_limits(values: np.ndarray) -> tuple[float, float]:
    vals = values[np.isfinite(values) & (values > 0)]
    if len(vals) == 0:
        return 1e-5, 1.0
    return max(vals.min() / 2.5, 1e-8), vals.max() * 2.5


def make_threshold_figure(pt: pd.DataFrame, fig_dir: Path, logger: logging.Logger) -> None:
    setup_matplotlib()
    fig, axes = plt.subplots(
        1, 2, figsize=(10.2, 4.2), gridspec_kw={"width_ratios": [1.35, 1.0]}
    )
    panels = [
        ("GIAB + DNBSEQ-T7", ["GIAB", "T7"], axes[0]),
        ("Twist wild-type", ["Twist-WT"], axes[1]),
    ]
    for panel_title, datasets, ax in panels:
        sub_all = pt[pt["dataset"].isin(datasets)]
        omitted = sub_all[
            (sub_all["support_flag"] != "ok")
            & np.isfinite(sub_all["mean_error_rate"])
        ]
        sub = sub_all[sub_all["support_flag"] == "ok"]
        for platform in _platform_order(sub):
            g = sub[sub["platform"] == platform].sort_values("quality_threshold")
            color = COLORS[platform]
            ax.plot(
                g["quality_threshold"],
                g["mean_error_rate"],
                color=color,
                lw=1.5,
                marker="o",
                ms=4.2,
                label=platform,
                zorder=3,
            )
            lo = g["mean_error_rate"] - g["sd_error_rate"]
            hi = g["mean_error_rate"] + g["sd_error_rate"]
            if np.isfinite(lo).any():
                ax.fill_between(
                    g["quality_threshold"],
                    lo,
                    hi,
                    color=color,
                    alpha=0.12,
                    lw=0,
                    zorder=1,
                )
        ax.set_yscale("log")
        ylo, yhi = _positive_limits(sub["mean_error_rate"].to_numpy(dtype=float))
        ax.set_ylim(ylo, yhi)
        ax.set_xlim(-1.5, 41.5)
        ax.set_xticks(THRESHOLDS)
        ax.set_xticklabels([f"Q{q}" for q in THRESHOLDS])
        ax.axvline(30, color="0.65", lw=0.7, ls=":", zorder=0)
        ax.set_xlabel("Quality threshold (bases with qual ≥ Q retained)")
        ax.set_ylabel("Mismatch rate among retained bases")
        ax.set_title(panel_title, loc="left", fontweight="bold")
        if panel_title.startswith("GIAB"):
            n_note = "means ± SD across libraries (GIAB n=12/platform; T7 n=5)"
        else:
            n_note = "one library per platform: no SD"
        if len(omitted):
            om = "; ".join(
                f"{r.platform} Q{int(r.quality_threshold)} "
                f"({int(r.total_retained_bases):,} bases)"
                for r in omitted.itertuples()
            )
            n_note += f"\nOmitted (<{MIN_PLOT_BASES:,} bases): {om}"
        ax.text(
            0.01,
            0.02,
            n_note,
            transform=ax.transAxes,
            fontsize=7.6,
            color="0.35",
            va="bottom",
        )
        ax.legend(frameon=False, loc="upper right", ncol=1, handlelength=1.4, labelspacing=0.3)
    axes[0].text(-0.13, 1.06, "a", transform=axes[0].transAxes, fontsize=15, fontweight="bold", va="top")
    axes[1].text(-0.16, 1.06, "b", transform=axes[1].transAxes, fontsize=15, fontweight="bold", va="top")
    fig.tight_layout(w_pad=2.0)
    for ext in ("png", "pdf"):
        out = fig_dir / f"A3_error_rate_vs_q_threshold.{ext}"
        fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
        logger.info("Wrote %s", out)
    plt.close(fig)


def make_per_quality_figure(pq: pd.DataFrame, fig_dir: Path, logger: logging.Logger) -> None:
    setup_matplotlib()
    fig, axes = plt.subplots(
        1, 2, figsize=(10.2, 4.2), gridspec_kw={"width_ratios": [1.35, 1.0]}
    )
    panels = [
        ("GIAB + DNBSEQ-T7", ["GIAB", "T7"], axes[0]),
        ("Twist wild-type", ["Twist-WT"], axes[1]),
    ]
    for panel_title, datasets, ax in panels:
        sub_all = pq[pq["dataset"].isin(datasets)]
        omitted = sub_all[
            (sub_all["support_flag"] != "ok")
            & (sub_all["n_samples_with_bases"] > 0)
        ]
        sub = sub_all[sub_all["support_flag"] == "ok"]
        for platform in _platform_order(sub):
            g = sub[
                (sub["platform"] == platform) & (sub["n_samples_with_bases"] > 0)
            ].sort_values("qual")
            if g.empty:
                continue
            color = COLORS[platform]
            ax.plot(
                g["qual"],
                g["mean_error_rate"],
                color=color,
                lw=1.2,
                marker="o",
                ms=3.4,
                label=platform,
                zorder=3,
            )
            ax.fill_between(
                g["qual"],
                g["min_error_rate"],
                g["max_error_rate"],
                color=color,
                alpha=0.10,
                lw=0,
                zorder=1,
            )
        ax.set_yscale("log")
        ylo, yhi = _positive_limits(sub["mean_error_rate"].to_numpy(dtype=float))
        ax.set_ylim(ylo, yhi)
        ax.set_xlim(-1, 41)
        ax.set_xticks([0, 5, 10, 15, 20, 25, 30, 35, 40])
        ax.set_xlabel("Reported base quality (QUAL)")
        ax.set_ylabel("Mismatch rate at the reported quality")
        ax.set_title(panel_title, loc="left", fontweight="bold")
        ax.legend(frameon=False, loc="lower left", ncol=1, handlelength=1.4, labelspacing=0.3)
        if len(omitted):
            parts = []
            for platform, g in omitted.groupby("platform", sort=False):
                vals = sorted(int(v) for v in g["qual"].unique())
                parts.append(f"{platform}: Q" + ", Q".join(map(str, vals)))
            ax.text(
                0.99,
                0.98,
                f"Omitted (<{MIN_PLOT_BASES:,} bases):\n" + "\n".join(parts),
                transform=ax.transAxes,
                fontsize=7.4,
                color="0.35",
                ha="right",
                va="top",
            )
    axes[0].text(-0.13, 1.06, "a", transform=axes[0].transAxes, fontsize=15, fontweight="bold", va="top")
    axes[1].text(-0.16, 1.06, "b", transform=axes[1].transAxes, fontsize=15, fontweight="bold", va="top")
    fig.tight_layout(w_pad=2.0)
    for ext in ("png", "pdf"):
        out = fig_dir / f"A3_error_rate_per_quality.{ext}"
        fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
        logger.info("Wrote %s", out)
    plt.close(fig)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-dir", type=Path, default=DEFAULT_BASE)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--analysis-dir", type=Path, default=DEFAULT_ANALYSIS_DIR)
    ap.add_argument("--chunksize", type=int, default=250_000)
    ap.add_argument("--skip-md5", action="store_true", help="Skip md5 inventory (faster)")
    args = ap.parse_args()

    out_dir: Path = args.out_dir
    table_dir = out_dir / "tables"
    fig_dir = out_dir / "figures"
    log_dir = out_dir / "logs"
    for d in (table_dir, fig_dir, log_dir):
        d.mkdir(parents=True, exist_ok=True)

    logger = setup_logging(log_dir / "A3.log")
    t_start = time.time()
    logger.info("=" * 78)
    logger.info("A3: multi-threshold and per-quality mismatch rates")
    logger.info("=" * 78)
    logger.info("Python %s", sys.version.replace("\n", " "))
    logger.info("pandas %s | numpy %s | matplotlib %s", pd.__version__, np.__version__, matplotlib.__version__)
    logger.info("base dir: %s", args.base_dir)
    logger.info("out dir : %s", out_dir)
    logger.info("thresholds: %s", THRESHOLDS)
    logger.info("per-quality range: %d-%d", QUALITIES.min(), QUALITIES.max())
    logger.info("chunksize: %d", args.chunksize)

    # ---------------- Inventory ----------------
    records = discover_files(args.base_dir, logger)
    if not records:
        logger.error("No input files found; aborting.")
        return 2
    logger.info("Total .er.txt.gz files discovered: %d", len(records))

    by_root = (
        pd.DataFrame(records)
        .groupby(["source_batch", "platform", "dataset"])
        .size()
        .reset_index(name="n_files")
    )
    log_dataframe(logger, "Raw inventory by batch/platform/dataset", by_root)

    # Header checks and md5/duplication audit
    n_header_bad = 0
    for rec in records:
        ok, cols = check_header(Path(rec["path"]), logger)
        rec["header_ok"] = ok
        rec["header_columns"] = ",".join(cols)
        if not ok:
            n_header_bad += 1
        rec["size_bytes"] = Path(rec["path"]).stat().st_size
        rec["md5"] = "" if args.skip_md5 else md5sum(Path(rec["path"]))
    if n_header_bad:
        logger.error("Files with a bad header (will be skipped): %d", n_header_bad)

    # Duplication audit: same sample ID in more than one root, identical md5.
    inv = pd.DataFrame(records)
    sample_roots = (
        inv.groupby("sample")["source_batch"]
        .agg(lambda s: ";".join(sorted(set(s))))
        .reset_index(name="source_batches")
    )
    cross = sample_roots[sample_roots["source_batches"].str.contains(";")]
    logger.info(
        "Sample IDs present in more than one source batch: %d", len(cross)
    )
    if len(cross):
        log_dataframe(logger, "Cross-batch sample IDs (first 40)", cross.head(40))
    if not args.skip_md5:
        md5_dups = inv[inv["md5"].duplicated(keep=False) & (inv["md5"] != "")]
        logger.info("Byte-identical files (same md5) across the three roots: %d", len(md5_dups))
        if len(md5_dups):
            log_dataframe(logger, "Identical files", md5_dups[["sample", "source_batch", "path", "md5"]])
    logger.info(
        "Note: cross-batch files are NOT identical; 7.errorRates is a bq30/PE70M "
        "reprocessing of a subset of the same libraries. Only the T7 files from "
        "7.errorRates enter the primary set; its 25 non-T7 files are used only for "
        "the published-table sanity check. All 15 T7 files in 1.errorRates are "
        "excluded from the primary set to avoid counting overlapping libraries."
    )

    excluded = inv[~inv["include_primary"].astype(bool)]
    logger.info("Excluded from primary set: %d files", len(excluded))
    if len(excluded):
        log_dataframe(
            logger,
            "Excluded files by batch/platform (reasons in inventory table)",
            excluded.groupby(["source_batch", "platform", "exclusion_reason"]).size().reset_index(name="n_files"),
            max_rows=20,
        )

    # ---------------- Aggregate all files ----------------
    logger.info("-" * 78)
    logger.info("Aggregating %d files (chunked)", len(records))
    t_agg = time.time()
    ok_records = []
    for i, rec in enumerate(records, 1):
        if not rec.get("header_ok", False):
            rec["status"] = "error"
            rec["error"] = "bad header"
            continue
        rec = aggregate_file(rec, args.chunksize, logger)
        ok_records.append(rec)
        if i % 10 == 0 or i == len(records):
            logger.info(
                "[%3d/%3d] %-34s total=%13d  qual=%s..%s  nqual=%2s  status=%s  elapsed=%.1fs",
                i,
                len(records),
                rec["sample"],
                rec.get("total_bases", 0),
                rec.get("qual_min"),
                rec.get("qual_max"),
                rec.get("n_qual_levels"),
                rec.get("status"),
                time.time() - t_agg,
            )
        if rec.get("status") != "ok":
            logger.error(
                "Skipping %s (%s): %s", rec["sample"], rec["source_batch"], rec.get("error", "unknown")
            )
    logger.info(
        "Aggregation done: %d ok, %d failed, %.1fs",
        sum(r.get("status") == "ok" for r in ok_records),
        len(records) - sum(r.get("status") == "ok" for r in ok_records),
        time.time() - t_agg,
    )

    # ---------------- Per-sample tables ----------------
    st_all = sample_threshold_table(ok_records)
    pq_all = sample_per_quality_table(ok_records)

    primary_ids = {
        (r["source_batch"], r["sample"])
        for r in ok_records
        if r.get("include_primary") and r.get("status") == "ok"
    }
    logger.info("Primary samples: %d", len(primary_ids))
    if len(primary_ids) != len({r["sample"] for r in ok_records if r.get("include_primary") and r.get("status") == "ok"}):
        logger.error("Duplicate sample IDs in the primary set!")

    st_primary = st_all.merge(
        pd.DataFrame(
            [(b, s) for b, s in primary_ids], columns=["source_batch", "sample"]
        ),
        on=["source_batch", "sample"],
        how="inner",
    )
    pq_primary = pq_all.merge(
        pd.DataFrame(
            [(b, s) for b, s in primary_ids], columns=["source_batch", "sample"]
        ),
        on=["source_batch", "sample"],
        how="inner",
    )
    # Keep the canonical column order (merge may reorder nothing, but be explicit).
    st_primary = st_primary[SAMPLE_THRESHOLD_COLUMNS].sort_values(
        ["dataset", "platform", "sample", "quality_threshold"], kind="stable"
    )
    pq_primary = pq_primary[
        [
            "dataset",
            "platform",
            "platform_short",
            "sample",
            "hg",
            "source_batch",
            "processing",
            "qual",
            "bases",
            "mismatches",
            "error_rate",
            "fraction_of_sample_bases",
        ]
    ].sort_values(["dataset", "platform", "sample", "qual"], kind="stable")

    coverage = (
        st_primary.groupby(["dataset", "platform"])["sample"]
        .nunique()
        .reset_index(name="n_samples")
    )
    log_dataframe(logger, "Primary coverage by dataset/platform", coverage)

    # Platform summaries
    pt = platform_threshold_table(st_primary)
    ppq = platform_per_quality_table(pq_primary)

    # ---------------- Write primary tables ----------------
    p_sample = table_dir / "A3_multithreshold_by_sample.csv"
    p_platform = table_dir / "A3_multithreshold_by_platform.csv"
    p_pq_sample = table_dir / "A3_per_quality_by_sample.csv.gz"
    p_pq_platform = table_dir / "A3_per_quality_by_platform.csv"
    st_primary.to_csv(p_sample, index=False, float_format="%.10g")
    pt.to_csv(p_platform, index=False, float_format="%.10g")
    pq_primary.to_csv(p_pq_sample, index=False, compression="gzip", float_format="%.10g")
    ppq.to_csv(p_pq_platform, index=False, float_format="%.10g")
    for p in (p_sample, p_platform, p_pq_sample, p_pq_platform):
        logger.info("Wrote %s", p)

    # ---------------- File inventory table ----------------
    inv_cols = [
        "source_batch",
        "processing",
        "dataset",
        "platform",
        "platform_short",
        "sample",
        "hg",
        "path",
        "size_bytes",
        "md5",
        "header_columns",
        "include_primary",
        "exclusion_reason",
        "status",
        "error",
        "n_rows",
        "total_bases",
        "qual_min",
        "qual_max",
        "n_qual_levels",
    ]
    inventory = pd.DataFrame(ok_records)
    for c in inv_cols:
        if c not in inventory.columns:
            inventory[c] = np.nan
    inventory = inventory[inv_cols].sort_values(["source_batch", "platform", "sample"])
    p_inv = table_dir / "A3_file_inventory.csv"
    inventory.to_csv(p_inv, index=False)
    logger.info("Wrote %s", p_inv)

    # ---------------- Sanity check 1: exact reproduction of published CSV ----------------
    logger.info("-" * 78)
    logger.info("Sanity check against published single-threshold table")
    sanity_records = [r for r in ok_records if r["source_batch"] == "7.errorRates_bq30_PE70M"]
    sanity_st = sample_threshold_table(sanity_records)
    sanity_pq = sample_per_quality_table(sanity_records)
    logger.info(
        "Sanity set (7.errorRates_bq30_PE70M): %d files, %d samples",
        len(sanity_records),
        sanity_st["sample"].nunique(),
    )
    check_st, stats_st = compare_thresholds_with_existing(sanity_st, args.analysis_dir)
    p_chk = table_dir / "A3_sanity_check_vs_existing_thresholds.csv"
    check_st.to_csv(p_chk, index=False, float_format="%.10g")
    logger.info("Wrote %s", p_chk)
    logger.info("Threshold sanity-check statistics:")
    for k, v in stats_st.items():
        logger.info("  %-34s %s", k, v)

    check_pq, stats_pq = compare_per_quality_with_existing(sanity_pq, args.analysis_dir)
    p_chkp = table_dir / "A3_sanity_check_vs_existing_per_quality.csv"
    check_pq.to_csv(p_chkp, index=False, float_format="%.10g")
    logger.info("Wrote %s", p_chkp)
    logger.info("Per-quality sanity-check statistics:")
    for k, v in stats_pq.items():
        logger.info("  %-34s %s", k, v)

    # ---------------- Sanity check 2: 1.errorRates vs 7.errorRates same library IDs ----------------
    logger.info("-" * 78)
    logger.info("Source comparison: 1.errorRates (full) versus 7.errorRates (bq30/PE70M)")
    source_detail, source_summary = source_comparison(st_all)
    p_src = table_dir / "A3_source_comparison_batch1_vs_batch7.csv"
    source_detail.to_csv(p_src, index=False, float_format="%.10g")
    logger.info("Wrote %s", p_src)
    log_dataframe(logger, "Source comparison by threshold (same 30 library IDs)", source_summary)

    # ---------------- Key summaries in the log ----------------
    logger.info("-" * 78)
    logger.info("PRIMARY RESULTS")
    q_sel = pt[pt["quality_threshold"].isin([0, 10, 20, 25, 30, 35, 40])]
    wide = q_sel.pivot_table(
        index=["dataset", "platform", "n_samples"],
        columns="quality_threshold",
        values="mean_error_rate",
        aggfunc="first",
    ).reset_index()
    wide.columns = [str(c) if not isinstance(c, str) else c for c in wide.columns]
    rename = {str(q): f"Q{q}_mean_err" for q in THRESHOLDS}
    wide = wide.rename(columns=rename)
    log_dataframe(logger, "Platform mean mismatch rates by threshold", wide)

    frac30 = q_sel[q_sel["quality_threshold"] == 30][
        [
            "dataset",
            "platform",
            "n_samples",
            "n_samples_with_retained_bases",
            "mean_fraction_retained",
            "sd_fraction_retained",
            "min_fraction_retained",
            "max_fraction_retained",
            "pooled_fraction_retained",
            "pooled_error_rate",
        ]
    ]
    log_dataframe(logger, "Q30 retained fraction and pooled error rate", frac30)

    # Quality-level coverage: platforms use very different, discrete score sets.
    pq_obs = pq_primary[pq_primary["bases"] > 0]
    qcov = (
        pq_obs.groupby(["dataset", "platform"])["qual"]
        .nunique()
        .reset_index(name="n_distinct_qual_values")
    )
    obs = (
        pq_obs.groupby(["dataset", "platform"])["qual"]
        .agg(lambda s: ",".join(str(x) for x in sorted(set(s.astype(int)))))
        .reset_index(name="observed_qual_values")
    )
    qcov = qcov.merge(obs, on=["dataset", "platform"], how="left")
    log_dataframe(logger, "Reported quality vocabularies by platform", qcov)

    # Low-support points: retained in tables, omitted from figures.
    low_thr = pt[(pt["support_flag"] != "ok") & np.isfinite(pt["mean_error_rate"])][
        ["dataset", "platform", "quality_threshold", "n_samples",
         "total_retained_bases", "mean_error_rate", "sd_error_rate"]
    ]
    log_dataframe(
        logger,
        f"Threshold points with <{MIN_PLOT_BASES:,} retained bases "
        "(kept in tables; omitted from figure)",
        low_thr,
    )
    low_q = ppq[
        (ppq["support_flag"] != "ok") & (ppq["n_samples_with_bases"] > 0)
    ][
        ["dataset", "platform", "qual", "n_samples_with_bases", "total_bases",
         "mean_error_rate", "sd_error_rate"]
    ]
    log_dataframe(
        logger,
        f"Per-quality points with <{MIN_PLOT_BASES:,} total bases "
        "(kept in tables; omitted from figure)",
        low_q,
    )

    # ---------------- Figures ----------------
    logger.info("-" * 78)
    logger.info("Making figures")
    make_threshold_figure(pt, fig_dir, logger)
    make_per_quality_figure(ppq, fig_dir, logger)

    elapsed = time.time() - t_start
    logger.info("=" * 78)
    logger.info("DONE in %.1f s", elapsed)
    logger.info("Primary samples: %d | platform-threshold rows: %d | per-quality rows: %d",
                st_primary["sample"].nunique(), len(pt), len(ppq))
    logger.info("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
