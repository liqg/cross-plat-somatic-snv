#!/usr/bin/env python3
"""
D1: sensitivity analysis matching platforms on deduplicated (Mutect2) depth
instead of raw sequencing depth.

Inputs
------
4.test/{45,54}/pred/{platform}_reads{N}M_pred.txt.gz
    One row per candidate site; DP is the Mutect2 depth after duplicate
    removal, label is the truth label, pred is the full-17-feature model score.
4.test/{45,54}/posmix/{platform}_T_reads{N}M.bam
    Tumour BAMs, used only for a BAM-level check of the depth proxy.
tables/A2_paired_by_condition.csv
    Previously computed per-condition full17 F1 values, used as a cross-check.

Outputs (all under response/addendum-analyses/)
-----------------------------------------------
tables/D1_dedup_depth_by_condition.csv
tables/D1_f1_vs_dedup_depth.csv
tables/D1_matched_pairs.csv
tables/D1_rank_at_matched_depth.csv
tables/D1_pairwise_matched_depth.csv          (supporting)
tables/D1_matched_direction_counts.csv        (supporting)
tables/D1_raw_depth_ranking.csv               (raw-depth control)
tables/D1_interpolation_fit_stats.csv         (supporting)
tables/D1_bam_depth_validation.csv            (if BAM results present)
tables/D1_bam_depth_correlation.csv           (if BAM results present)
figures/D1_f1_vs_dedup_depth.png/.pdf
logs/D1.log
logs/D1_panel_sample.bed                      (BAM validation intervals)

Notes
-----
* F1@0.1 is F1 with pred > 0.1.  F1_best is the maximum F1 over all score
  cut-offs (same definition as A2_summary.py).
* Raw-depth matching compares conditions at the same nominal raw depth.
  Dedup-depth matching compares conditions with the closest median Mutect2 DP
  at candidate sites, allowing different nominal raw depths.
* No significance tests are reported for matched F1 differences; only
  descriptive sign counts and effect sizes.
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
import common  # noqa: E402

import argparse
import glob
import logging
import os
import re
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LinearRegression

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --------------------------------------------------------------------------
# Paths and constants
# --------------------------------------------------------------------------
OUTDIR = common.RESULTS_ROOT
TABDIR = common.TABLES
FIGDIR = common.FIGURES
LOGDIR = common.LOGS
PRED_ROOT = common.TEST_ROOT
A2_FILE = TABDIR / "A2_paired_by_condition.csv"
PANEL_BED = common.REF_DIR / "panel" / "intersect_HG004_HG005_panel.sort.bed"
BAM_RAW = LOGDIR / "D1_bam_depth_raw.tsv"
SAMPLE_BED = LOGDIR / "D1_panel_sample.bed"

GROUPS = [45, 54]
READS = [60, 80, 100, 120, 140]
PLATFORMS = [
    "FASTASeq 300",
    "GenoLab M",
    "DNBSEQ-T7",
    "NovaSeq 6000",
    "NovaSeq X",
    "SURFSeq 5000",
]
SURF = "SURFSeq 5000"
TARGET = "SURFSeq 5000"  # alias for readability

# pred-file platform IDs -> display names used in the manuscript/A2
FILE_TO_DISPLAY = {
    "FASTASeq_300": "FASTASeq 300",
    "GenoLab_M": "GenoLab M",
    "MGISEQ-T7": "DNBSEQ-T7",
    "NovaSeq_6000": "NovaSeq 6000",
    "NovaSeq_X": "NovaSeq X",
    "SURFSeq_5000": "SURFSeq 5000",
}
DISPLAY_TO_FILE = {v: k for k, v in FILE_TO_DISPLAY.items()}
A2_TO_DISPLAY = {
    "FASTASeq 300": "FASTASeq 300",
    "GenoLab M": "GenoLab M",
    "DNBSEQ-T7": "DNBSEQ-T7",
    "MGISEQ-T7": "DNBSEQ-T7",
    "NovaSeq 6000": "NovaSeq 6000",
    "NovaSeq X": "NovaSeq X",
    "SURFSeq 5000": "SURFSeq 5000",
}
COLORS = {
    "FASTASeq 300": "#1f77b4",
    "GenoLab M": "#2ca02c",
    "DNBSEQ-T7": "#9467bd",
    "NovaSeq 6000": "#8c564b",
    "NovaSeq X": "#e377c2",
    "SURFSeq 5000": "#d62728",
}
METRICS = ["F1@0.1", "F1_best"]
TIE_TOL = 0.002          # half a percentage point on the F1 scale
DP_TIE_LOG2 = 0.05       # ~3.5% median-DP difference, used for near-tie audit
GRID_N_ALL = 40
GRID_N_PAIR = 20


# --------------------------------------------------------------------------
# Logging
# --------------------------------------------------------------------------
def setup_logger() -> logging.Logger:
    LOGDIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("D1")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    fh = logging.FileHandler(LOGDIR / "D1.log", mode="w")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def log_df(logger: logging.Logger, df: pd.DataFrame, title: str) -> None:
    logger.info("")
    logger.info("---- %s ----", title)
    if df.empty:
        logger.info("(empty)")
        return
    for line in df.to_string(index=True).splitlines():
        logger.info(line)


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------
def f1_at(y: np.ndarray, p: np.ndarray, thr: float = 0.1) -> float:
    c = p > thr
    tp = int((c & (y == 1)).sum())
    fp = int((c & (y == 0)).sum())
    fn = int(((~c) & (y == 1)).sum())
    return 2.0 * tp / max(2 * tp + fp + fn, 1)


def best_f1(y: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    """Maximum F1 over score cut-offs, and the corresponding threshold."""
    order = np.argsort(-p, kind="mergesort")
    pp = p[order]
    yy = y[order]
    tp = np.cumsum(yy)
    fp = np.cumsum(1 - yy)
    fn = yy.sum() - tp
    f1 = 2 * tp / np.maximum(2 * tp + fp + fn, 1)
    i = int(np.argmax(f1))
    thr = pp[i] if i == len(pp) - 1 else (pp[i] + pp[i + 1]) / 2.0
    return float(f1[i]), float(thr)


# --------------------------------------------------------------------------
# Load condition-level table
# --------------------------------------------------------------------------
def load_conditions(logger: logging.Logger) -> pd.DataFrame:
    rows = []
    for group in GROUPS:
        files = sorted(glob.glob(str(PRED_ROOT / str(group) / "pred" / "*_pred.txt.gz")))
        if not files:
            raise FileNotFoundError(f"No pred files for group {group} under {PRED_ROOT}")
        for f in files:
            name = os.path.basename(f)
            m = re.match(r"(.+)_reads(\d+)M_pred\.txt\.gz$", name)
            if not m:
                raise ValueError(f"Unexpected pred filename: {name}")
            file_plat, reads = m.group(1), int(m.group(2))
            if file_plat not in FILE_TO_DISPLAY:
                raise ValueError(f"Unexpected platform id: {file_plat}")
            plat = FILE_TO_DISPLAY[file_plat]
            d = pd.read_csv(f, sep="\t", usecols=["DP", "label", "pred"])
            y = d["label"].to_numpy(dtype=int)
            p = d["pred"].to_numpy(dtype=float)
            fb, thr = best_f1(y, p)
            rows.append(
                {
                    "Group": int(group),
                    "Platform": plat,
                    "PlatformFileID": file_plat,
                    "Reads": int(reads),
                    "N_candidates": int(len(d)),
                    "N_pos": int(y.sum()),
                    "Median_DP": float(d["DP"].median()),
                    "Mean_DP": float(d["DP"].mean()),
                    "Q25_DP": float(d["DP"].quantile(0.25)),
                    "Q75_DP": float(d["DP"].quantile(0.75)),
                    "F1@0.1": f1_at(y, p, 0.1),
                    "F1_best": fb,
                    "Best_threshold": thr,
                }
            )
    df = pd.DataFrame(rows)

    # Cross-check against A2 paired table where available
    if A2_FILE.exists():
        a2 = pd.read_csv(A2_FILE)
        a2["Platform"] = a2["Platform"].map(lambda s: A2_TO_DISPLAY.get(s, s))
        a2["Group"] = a2["Group"].astype(int)
        a2 = a2.rename(
            columns={
                "full17_F1@0.1": "A2_F1@0.1",
                "full17_F1best": "A2_F1_best",
            }
        )
        keep = ["Group", "Platform", "Reads", "A2_F1@0.1", "A2_F1_best"]
        df = df.merge(a2[keep], on=["Group", "Platform", "Reads"], how="left")
        df["Delta_F1@0.1_vs_A2"] = df["F1@0.1"] - df["A2_F1@0.1"]
        df["Delta_F1_best_vs_A2"] = df["F1_best"] - df["A2_F1_best"]
        n_missing = int(df["A2_F1@0.1"].isna().sum())
        if n_missing:
            logger.warning("A2 merge: %d conditions without an A2 match", n_missing)
    else:
        logger.warning("A2 file not found: %s", A2_FILE)
        for c in ["A2_F1@0.1", "A2_F1_best", "Delta_F1@0.1_vs_A2", "Delta_F1_best_vs_A2"]:
            df[c] = np.nan

    # Within-group ranks by median DP and raw depth
    df["MedianDP_rank_in_group"] = df.groupby("Group")["Median_DP"].rank(method="min").astype(int)
    df["RawReads_rank_in_group"] = df.groupby("Group")["Reads"].rank(method="min").astype(int)
    df = df.sort_values(["Group", "Platform", "Reads"]).reset_index(drop=True)
    return df


# --------------------------------------------------------------------------
# Descriptions of the depth proxy
# --------------------------------------------------------------------------
def describe_dp(df: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
    logger.info("")
    logger.info("================ DEDUPLICATED-DEPTH PROXY ================")
    logger.info(
        "DP in the pred table is the Mutect2 depth at candidate sites after "
        "duplicate removal.  It is not a whole-panel average; it is used here "
        "for relative comparisons across platforms at the same nominal raw depth."
    )
    for g in GROUPS:
        sub = df[df.Group == g]
        piv = sub.pivot(index="Platform", columns="Reads", values="Median_DP")
        piv = piv.reindex(PLATFORMS)[READS]
        log_df(logger, piv.round(1), f"Group {g}: median candidate DP by platform x raw depth (x)")

        # spread across platforms at each raw depth
        spread = []
        for r in READS:
            vals = sub[sub.Reads == r].set_index("Platform")["Median_DP"].reindex(PLATFORMS)
            spread.append(
                {
                    "Group": g,
                    "Reads": r,
                    "Min_medianDP": vals.min(),
                    "Max_medianDP": vals.max(),
                    "Max_over_min": vals.max() / vals.min(),
                    "CV_percent": 100 * vals.std(ddof=1) / vals.mean(),
                    "Highest_platform": vals.idxmax(),
                    "Lowest_platform": vals.idxmin(),
                }
            )
        log_df(logger, pd.DataFrame(spread).set_index("Reads"), f"Group {g}: cross-platform median-DP spread at each raw depth")

        rho, pval = spearmanr(sub["Reads"], sub["Median_DP"])
        logger.info(
            "Group %s: Spearman(raw reads, median candidate DP) across all %d conditions = %.3f (p=%.2g)",
            g, len(sub), rho, pval,
        )
        per_plat = []
        for plat in PLATFORMS:
            s = sub[sub.Platform == plat].sort_values("Reads")
            rho_p, p_p = spearmanr(s["Reads"], s["Median_DP"])
            ratio = s["Median_DP"].iloc[-1] / s["Median_DP"].iloc[0]
            per_plat.append(
                {
                    "Platform": plat,
                    "Spearman_raw_vs_medianDP": rho_p,
                    "p": p_p,
                    "MedianDP_140M_over_60M": ratio,
                }
            )
        log_df(logger, pd.DataFrame(per_plat).set_index("Platform").round(4), f"Group {g}: within-platform depth scaling")

    return df


# --------------------------------------------------------------------------
# Nearest-neighbour matching
# --------------------------------------------------------------------------
def _row_dict(r) -> dict:
    return {
        "Platform": r.Platform,
        "Reads": int(r.Reads),
        "Median_DP": float(r.Median_DP),
        "F1@0.1": float(r["F1@0.1"]),
        "F1_best": float(r.F1_best),
        "N_candidates": int(r.N_candidates),
    }


def nearest_condition(anchor_row, candidates: pd.DataFrame) -> tuple[pd.Series, float]:
    diffs = (np.log2(anchor_row["Median_DP"]) - np.log2(candidates["Median_DP"])).abs()
    idx = diffs.idxmin()
    return candidates.loc[idx], float(diffs.loc[idx])


def build_matched_pairs(df: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
    rows = []

    # (a) Pair-level optimal matching: every unordered platform pair
    for g in GROUPS:
        sub = df[df.Group == g]
        for i in range(len(PLATFORMS)):
            for j in range(i + 1, len(PLATFORMS)):
                pa, pb = PLATFORMS[i], PLATFORMS[j]
                # orient SURFSeq as A when present
                if pb == SURF:
                    pa, pb = pb, pa
                dfa = sub[sub.Platform == pa]
                dfb = sub[sub.Platform == pb]
                row, ad, bd = closest_pair_fixed(dfa, dfb)
                row["Group"] = g
                if pa == SURF or pb == SURF:
                    comp = pb if pa == SURF else pa
                    row["Comparator"] = comp
                    row["Anchor_Platform"] = SURF
                    delta = row[f"Delta_F1@0.1_A_minus_B"]
                    if pa != SURF:
                        delta = -delta
                    row["Delta_F1@0.1_SURF_minus_Comparator"] = delta
                    delta_b = row["Delta_F1_best_A_minus_B"]
                    if pa != SURF:
                        delta_b = -delta_b
                    row["Delta_F1_best_SURF_minus_Comparator"] = delta_b
                else:
                    row["Comparator"] = ""
                    row["Anchor_Platform"] = ""
                    row["Delta_F1@0.1_SURF_minus_Comparator"] = np.nan
                    row["Delta_F1_best_SURF_minus_Comparator"] = np.nan
                rows.append(row)

    # (b)/(c) Anchor-level nearest neighbour in both directions, SURFSeq vs each comparator
    for g in GROUPS:
        sub = df[df.Group == g]
        for comp in [p for p in PLATFORMS if p != SURF]:
            surf_df = sub[sub.Platform == SURF]
            comp_df = sub[sub.Platform == comp]
            # SURF anchor -> nearest comparator condition
            for _, a in surf_df.iterrows():
                b, diff = nearest_condition(a, comp_df)
                rows.append(anchor_row(g, "surf_anchor", SURF, comp, a, b, diff))
            # comparator anchor -> nearest SURF condition
            for _, a in comp_df.iterrows():
                b, diff = nearest_condition(a, surf_df)
                rows.append(anchor_row(g, "comparator_anchor", SURF, comp, b, a, diff, anchor_reads=a.Reads))
    mp = pd.DataFrame(rows)
    # Sort for readability
    mp["Level"] = pd.Categorical(
        mp["Level"], ["pair_optimal", "surf_anchor", "comparator_anchor"], ordered=True
    )
    mp = mp.sort_values(["Group", "Level", "Comparator", "Anchor_Reads"]).reset_index(drop=True)
    return mp


def closest_pair_fixed(dfa: pd.DataFrame, dfb: pd.DataFrame) -> tuple[dict, dict, dict]:
    """Same as nearest_pair_rows but returns clean records without tuple hacks."""
    combos = []
    for a in dfa.to_dict("records"):
        for b in dfb.to_dict("records"):
            score = abs(np.log2(a["Median_DP"] / b["Median_DP"]))
            combos.append((score, abs(a["Median_DP"] - b["Median_DP"]), a, b))
    combos.sort(key=lambda z: (z[0], z[1]))
    best = combos[0]
    near = [c for c in combos if c[0] <= best[0] + DP_TIE_LOG2]
    a, b = best[2], best[3]
    row = {
        "Level": "pair_optimal",
        "Platform_A": a["Platform"],
        "Platform_B": b["Platform"],
        "Reads_A": int(a["Reads"]),
        "Reads_B": int(b["Reads"]),
        "MedianDP_A": float(a["Median_DP"]),
        "MedianDP_B": float(b["Median_DP"]),
        "AbsLog2DPDiff": float(best[0]),
        "DP_ratio_A_over_B": float(a["Median_DP"] / b["Median_DP"]),
        "N_candidates_A": int(a["N_candidates"]),
        "N_candidates_B": int(b["N_candidates"]),
        "NearTie_Count_within_0.05_log2": int(len(near)),
        "NearTie_Reads_pairs": ";".join(f"{int(c[2]['Reads'])}-{int(c[3]['Reads'])}" for c in near),
    }
    for metric in METRICS:
        row[f"{metric}_A"] = float(a[metric])
        row[f"{metric}_B"] = float(b[metric])
        row[f"Delta_{metric}_A_minus_B"] = float(a[metric] - b[metric])
        vals = [c[2][metric] - c[3][metric] for c in near]
        row[f"NearTie_Delta_{metric}_min"] = float(np.min(vals))
        row[f"NearTie_Delta_{metric}_max"] = float(np.max(vals))
    return row, a, b


def anchor_row(group, level, anchor_plat, comp, surf_row, comp_row, diff, anchor_reads=None) -> dict:
    """One SURFSeq-vs-comparator matched comparison, always oriented SURF - comparator."""
    row = {
        "Group": group,
        "Level": level,
        "Anchor_Platform": anchor_plat,
        "Comparator": comp,
        "Anchor_Reads": int(anchor_reads if anchor_reads is not None else surf_row["Reads"]),
        "Platform_A": SURF,
        "Platform_B": comp,
        "Reads_A": int(surf_row["Reads"]),
        "Reads_B": int(comp_row["Reads"]),
        "MedianDP_A": float(surf_row["Median_DP"]),
        "MedianDP_B": float(comp_row["Median_DP"]),
        "AbsLog2DPDiff": float(diff),
        "DP_ratio_A_over_B": float(surf_row["Median_DP"] / comp_row["Median_DP"]),
        "N_candidates_A": int(surf_row["N_candidates"]),
        "N_candidates_B": int(comp_row["N_candidates"]),
        "NearTie_Count_within_0.05_log2": np.nan,
        "NearTie_Reads_pairs": "",
        "NearTie_Delta_F1@0.1_min": np.nan,
        "NearTie_Delta_F1@0.1_max": np.nan,
        "NearTie_Delta_F1_best_min": np.nan,
        "NearTie_Delta_F1_best_max": np.nan,
    }
    for metric in METRICS:
        row[f"{metric}_A"] = float(surf_row[metric])
        row[f"{metric}_B"] = float(comp_row[metric])
        row[f"Delta_{metric}_A_minus_B"] = float(surf_row[metric] - comp_row[metric])
    row["Delta_F1@0.1_SURF_minus_Comparator"] = row["Delta_F1@0.1_A_minus_B"]
    row["Delta_F1_best_SURF_minus_Comparator"] = row["Delta_F1_best_A_minus_B"]
    return row


def summarise_matched_pairs(mp: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
    rows = []
    sub = mp[mp["Comparator"].notna() & (mp["Comparator"] != "")]
    for g in GROUPS + ["ALL"]:
        s0 = sub if g == "ALL" else sub[sub.Group == g]
        for level in ["pair_optimal", "surf_anchor", "comparator_anchor"]:
            s1 = s0[s0.Level == level]
            for metric in METRICS:
                delta_col = f"Delta_{metric}_SURF_minus_Comparator"
                for comp in [p for p in PLATFORMS if p != SURF] + ["ALL"]:
                    s = s1 if comp == "ALL" else s1[s1.Comparator == comp]
                    if s.empty:
                        continue
                    d = s[delta_col].dropna().to_numpy()
                    if len(d) == 0:
                        continue
                    rows.append(
                        {
                            "Group": g,
                            "Level": level,
                            "F1_metric": metric,
                            "Comparator": comp,
                            "N": int(len(d)),
                            "SURF_higher": int((d > TIE_TOL).sum()),
                            "Tie": int((np.abs(d) <= TIE_TOL).sum()),
                            "SURF_lower": int((d < -TIE_TOL).sum()),
                            "Exact_tie": int((d == 0).sum()),
                            "SURF_higher_0.005": int((d > 0.005).sum()),
                            "Tie_0.005": int((np.abs(d) <= 0.005).sum()),
                            "SURF_lower_0.005": int((d < -0.005).sum()),
                            "Mean_delta": float(np.mean(d)),
                            "Median_delta": float(np.median(d)),
                            "Min_delta": float(np.min(d)),
                            "Max_delta": float(np.max(d)),
                            "Mean_AbsLog2DPDiff": float(s["AbsLog2DPDiff"].mean()),
                            "Max_AbsLog2DPDiff": float(s["AbsLog2DPDiff"].max()),
                            "N_within_0.05_log2": int((s["AbsLog2DPDiff"] <= DP_TIE_LOG2).sum()),
                        }
                    )
    out = pd.DataFrame(rows)
    log_df(logger, out, "Matched-pair direction counts (delta = SURFSeq - comparator; tie = |delta| <= 0.002)")
    return out


# --------------------------------------------------------------------------
# Interpolation curves
# --------------------------------------------------------------------------
def common_range(dfp: pd.DataFrame) -> tuple[float, float]:
    lo = dfp.groupby("Platform")["Median_DP"].min().max()
    hi = dfp.groupby("Platform")["Median_DP"].max().min()
    return float(lo), float(hi)


def fit_models(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Fit linear (log10 DP) and isotonic curves for every group/platform/metric."""
    fits = {}
    stat_rows = []
    for g in GROUPS:
        for plat in PLATFORMS:
            s = df[(df.Group == g) & (df.Platform == plat)].sort_values("Median_DP")
            x = np.log10(s["Median_DP"].to_numpy())
            for metric in METRICS:
                y = s[metric].to_numpy()
                lin = LinearRegression().fit(x.reshape(-1, 1), y)
                iso = IsotonicRegression(increasing=True, out_of_bounds="clip").fit(x, y)
                rho, pval = spearmanr(x, y)
                fits[(g, plat, metric)] = {"linear": lin, "isotonic": iso, "x": x, "y": y}
                stat_rows.append(
                    {
                        "Group": g,
                        "Platform": plat,
                        "F1_metric": metric,
                        "Linear_slope_per_log10DP": float(lin.coef_[0]),
                        "Linear_intercept": float(lin.intercept_),
                        "Linear_R2": float(lin.score(x.reshape(-1, 1), y)),
                        "Spearman_logDP_F1": float(rho),
                        "Spearman_p": float(pval),
                        "Observed_DP_min": float(s["Median_DP"].min()),
                        "Observed_DP_max": float(s["Median_DP"].max()),
                    }
                )
    stats = pd.DataFrame(stat_rows).sort_values(["Group", "F1_metric", "Platform"])
    return stats, fits


def predict(fits, g, plat, metric, model, dp_grid) -> np.ndarray:
    xg = np.log10(np.asarray(dp_grid, dtype=float)).reshape(-1, 1)
    if model == "linear_log10_DP":
        pred = fits[(g, plat, metric)]["linear"].predict(xg)
    elif model == "isotonic_log10_DP":
        pred = fits[(g, plat, metric)]["isotonic"].predict(xg.ravel())
    else:
        raise ValueError(model)
    return np.clip(pred, 0.0, 1.0)


def build_interpolation_tables(df: pd.DataFrame, fits, logger: logging.Logger):
    # Observed rows
    obs = []
    for _, r in df.iterrows():
        for metric in METRICS:
            obs.append(
                {
                    "Group": int(r.Group),
                    "Scope": "observed",
                    "Comparator": "",
                    "Platform": r.Platform,
                    "F1_metric": metric,
                    "Source": "observed",
                    "RawReads": int(r.Reads),
                    "MedianDP": float(r.Median_DP),
                    "F1": float(r[metric]),
                    "WithinObservedRange": True,
                }
            )
    obs = pd.DataFrame(obs)

    fit_rows = []
    rank_rows = []
    pair_rows = []
    pair_summary = []

    for g in GROUPS:
        sub = df[df.Group == g]
        lo, hi = common_range(sub)
        logger.info("Group %s: all-platform common median-DP range = %.1f to %.1f x", g, lo, hi)
        grids = {
            "all6": np.geomspace(lo, hi, GRID_N_ALL),
        }
        for comp in [p for p in PLATFORMS if p != SURF]:
            flo = max(
                sub[sub.Platform == SURF]["Median_DP"].min(),
                sub[sub.Platform == comp]["Median_DP"].min(),
            )
            fhi = min(
                sub[sub.Platform == SURF]["Median_DP"].max(),
                sub[sub.Platform == comp]["Median_DP"].max(),
            )
            grids[f"pair::{comp}"] = np.geomspace(flo, fhi, GRID_N_PAIR)

        for scope, grid in grids.items():
            comp = scope.split("::", 1)[1] if scope.startswith("pair::") else ""
            plats = [SURF, comp] if comp else PLATFORMS
            for metric in METRICS:
                for model in ["linear_log10_DP", "isotonic_log10_DP"]:
                    preds = {}
                    for plat in plats:
                        preds[plat] = predict(fits, g, plat, metric, model, grid)
                        obs_lo = sub[sub.Platform == plat]["Median_DP"].min()
                        obs_hi = sub[sub.Platform == plat]["Median_DP"].max()
                        for i, dp in enumerate(grid):
                            fit_rows.append(
                                {
                                    "Group": g,
                                    "Scope": scope if comp else "all6",
                                    "Comparator": comp,
                                    "Platform": plat,
                                    "F1_metric": metric,
                                    "Source": model,
                                    "RawReads": np.nan,
                                    "MedianDP": float(dp),
                                    "F1": float(preds[plat][i]),
                                    "WithinObservedRange": bool(obs_lo <= dp <= obs_hi),
                                }
                            )
                    # Rankings at each grid point
                    for i, dp in enumerate(grid):
                        vals = pd.Series({p: preds[p][i] for p in plats})
                        ranks = vals.rank(method="min", ascending=False).astype(int)
                        surf_val = vals[SURF]
                        for plat in plats:
                            rec = {
                                "Group": g,
                                "Scope": scope if comp else "all6",
                                "Comparator": comp,
                                "F1_metric": metric,
                                "Model": model,
                                "GridIndex": i + 1,
                                "MedianDP": float(dp),
                                "Platform": plat,
                                "Predicted_F1": float(vals[plat]),
                                "Rank": int(ranks[plat]),
                                "Delta_vs_SURFSeq": float(vals[plat] - surf_val),
                                "SURFSeq_Rank": int(ranks[SURF]),
                                "WithinObservedRange": bool(
                                    sub[sub.Platform == plat]["Median_DP"].min() <= dp
                                    <= sub[sub.Platform == plat]["Median_DP"].max()
                                ),
                            }
                            if comp:
                                pair_rows.append(rec)
                            else:
                                rank_rows.append(rec)
                    if comp:
                        d = preds[SURF] - preds[comp]
                        pair_summary.append(
                            {
                                "Group": g,
                                "Comparator": comp,
                                "F1_metric": metric,
                                "Model": model,
                                "DP_min": float(grid[0]),
                                "DP_max": float(grid[-1]),
                                "N_grid": len(grid),
                                "Mean_delta_SURF_minus_comparator": float(np.mean(d)),
                                "Median_delta": float(np.median(d)),
                                "Min_delta": float(np.min(d)),
                                "Max_delta": float(np.max(d)),
                                "SURF_higher": int((d > TIE_TOL).sum()),
                                "Tie": int((np.abs(d) <= TIE_TOL).sum()),
                                "SURF_lower": int((d < -TIE_TOL).sum()),
                                "SURF_higher_grid_count": int(
                                    sum(
                                        preds[SURF][i] > preds[comp][i] + TIE_TOL
                                        for i in range(len(grid))
                                    )
                                ),
                                "SURF_not_lower_grid_count": int(
                                    sum(
                                        preds[SURF][i] >= preds[comp][i] - TIE_TOL
                                        for i in range(len(grid))
                                    )
                                ),
                            }
                        )

    fit_df = pd.concat([obs, pd.DataFrame(fit_rows)], ignore_index=True)
    rank_df = pd.DataFrame(rank_rows)
    pair_df = pd.DataFrame(pair_rows)
    pair_sum = pd.DataFrame(pair_summary)
    return fit_df, rank_df, pair_df, pair_sum


def summarise_interpolation(rank_df: pd.DataFrame, pair_sum: pd.DataFrame, logger: logging.Logger):
    logger.info("")
    logger.info("================ INTERPOLATION AT MATCHED DEDUP DEPTH ================")
    for g in GROUPS:
        for metric in METRICS:
            for model in ["linear_log10_DP", "isotonic_log10_DP"]:
                s = rank_df[
                    (rank_df.Group == g)
                    & (rank_df["F1_metric"] == metric)
                    & (rank_df.Model == model)
                ]
                if s.empty:
                    continue
                # one row per grid point
                per_grid = s.drop_duplicates("GridIndex")
                counts = per_grid["SURFSeq_Rank"].value_counts().reindex([1, 2, 3, 4, 5, 6], fill_value=0)
                logger.info(
                    "Group %s | %s | %s: SURFSeq rank distribution over %d DP levels = %s; mean rank %.2f",
                    g, metric, model, len(per_grid),
                    dict(counts), per_grid["SURFSeq_Rank"].mean(),
                )
                # mean delta vs each comparator across the grid
                d = (
                    s[~s.Platform.eq(SURF)]
                    .groupby("Platform")["Delta_vs_SURFSeq"]
                    .agg(["mean", "min", "max"])
                    .reindex([p for p in PLATFORMS if p != SURF])
                )
                log_df(
                    logger, d.round(4),
                    f"Group {g} | {metric} | {model}: predicted F1(platform) - F1(SURFSeq), all6 grid",
                )
    logger.info("")
    logger.info("Pairwise-overlap interpolation (SURFSeq vs each comparator), linear model:")
    s = pair_sum[(pair_sum.Model == "linear_log10_DP")]
    log_df(
        logger, s.round(4).set_index(["Group", "Comparator", "F1_metric"]),
        "Pairwise common-DP ranges and predicted delta (SURFSeq - comparator)",
    )


# --------------------------------------------------------------------------
# Raw-depth control
# --------------------------------------------------------------------------
def raw_depth_control(df: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
    rows = []
    for g in GROUPS:
        for r in READS:
            for metric in METRICS:
                s = df[(df.Group == g) & (df.Reads == r)].copy()
                s["Rank"] = s[metric].rank(method="min", ascending=False).astype(int)
                best = s[metric].max()
                surf_val = s.loc[s.Platform == SURF, metric].iloc[0]
                for _, row in s.iterrows():
                    rows.append(
                        {
                            "Group": g,
                            "Reads": r,
                            "F1_metric": metric,
                            "Platform": row.Platform,
                            "Median_DP": row.Median_DP,
                            "F1": row[metric],
                            "Rank": int(row.Rank),
                            "Delta_vs_SURFSeq": float(row[metric] - surf_val),
                            "Delta_vs_Best": float(row[metric] - best),
                        }
                    )
    out = pd.DataFrame(rows)

    logger.info("")
    logger.info("================ RAW-DEPTH-MATCHED CONTROL ================")
    for metric in METRICS:
        for g in GROUPS:
            s = out[(out.Group == g) & (out["F1_metric"] == metric)]
            surf = s[s.Platform == SURF]
            counts = surf["Rank"].value_counts().reindex([1, 2, 3, 4, 5, 6], fill_value=0)
            logger.info(
                "Group %s | %s: SURFSeq raw-depth rank distribution over 5 depths = %s; mean rank %.1f; mean delta vs best %.4f",
                g, metric, dict(counts), surf["Rank"].mean(),
                surf["Delta_vs_Best"].mean(),
            )
            d = (
                s[s.Platform != SURF]
                .groupby("Platform")["Delta_vs_SURFSeq"]
                .agg(["mean", "min", "max"])
                .reindex([p for p in PLATFORMS if p != SURF])
            )
            log_df(logger, d.round(4), f"Group {g} | {metric}: raw-depth-matched F1(platform) - F1(SURFSeq)")
    return out


# --------------------------------------------------------------------------
# BAM-level validation of the depth proxy
# --------------------------------------------------------------------------
def make_sample_bed(step: int = 20) -> Path:
    if not PANEL_BED.exists():
        raise FileNotFoundError(f"Panel BED not found: {PANEL_BED}")
    b = pd.read_csv(PANEL_BED, sep="\t", header=None, names=["chrom", "start", "end"])
    s = b.iloc[::step].copy()
    SAMPLE_BED.parent.mkdir(parents=True, exist_ok=True)
    s.to_csv(SAMPLE_BED, sep="\t", header=False, index=False)
    return SAMPLE_BED


def _depth_stats(bam: Path, bed: Path, panel_bp: int) -> dict:
    with tempfile.NamedTemporaryFile(suffix=".tsv", delete=False) as tmp:
        tmpname = tmp.name
    try:
        with open(tmpname, "w") as out:
            subprocess.run(
                ["samtools", "depth", "-b", str(bed), str(bam)],
                check=True, stdout=out, stderr=subprocess.PIPE, text=True,
            )
        d = pd.read_csv(tmpname, sep="\t", header=None, names=["chrom", "pos", "depth"])
        n = len(d)
        s = int(d["depth"].sum())
        med = float(d["depth"].median()) if n else np.nan
        return {
            "PanelBp": int(panel_bp),
            "CoveredBp": int(n),
            "SumDepth": s,
            "MeanDepthAllBp": s / panel_bp,
            "MeanDepthCovered": s / n if n else np.nan,
            "MedianDepth": med,
        }
    finally:
        if os.path.exists(tmpname):
            os.unlink(tmpname)


def run_bam_validation(group: int, reads: int, platforms: list[str], logger: logging.Logger,
                       workers: int = 6) -> pd.DataFrame:
    bed = make_sample_bed(20)
    panel_bp = int(
        pd.read_csv(bed, sep="\t", header=None, names=["c", "s", "e"])
        .eval("e - s").sum()
    )
    logger.info(
        "BAM validation: group %s, %s M raw reads, %d sampled panel intervals, %d bp (%.1f%% of panel)",
        group, reads, len(pd.read_csv(bed, sep="\t", header=None)), panel_bp,
        100 * panel_bp / int(
            pd.read_csv(PANEL_BED, sep="\t", header=None, names=["c", "s", "e"]).eval("e - s").sum()
        ),
    )
    bamdir = PRED_ROOT / str(group) / "posmix"
    out = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {}
        for plat in platforms:
            f = bamdir / f"{DISPLAY_TO_FILE[plat]}_T_reads{reads}M.bam"
            if not f.exists():
                alt = list(bamdir.glob(f"*_T_reads{reads}M.bam"))
                logger.warning("BAM not found for %s: %s (available: %s)", plat, f, [x.name for x in alt])
                continue
            futs[ex.submit(_depth_stats, f, bed, panel_bp)] = plat
        for fut in as_completed(futs):
            plat = futs[fut]
            try:
                st = fut.result()
                st.update({"Group": group, "Reads": reads, "Platform": plat})
                out.append(st)
                logger.info("BAM depth done: group %s %sM %s -> mean %.1f, median %.1f",
                            group, reads, plat, st["MeanDepthAllBp"], st["MedianDepth"])
            except Exception as e:
                logger.error("BAM depth failed for %s: %s", plat, e)
    return pd.DataFrame(out)


def merge_bam_validation(df: pd.DataFrame, logger: logging.Logger):
    if not BAM_RAW.exists():
        logger.warning("BAM validation raw table not found (%s); skipping BAM check.", BAM_RAW)
        return
    b = pd.read_csv(BAM_RAW, sep="\t")
    if "TotalBp" in b.columns and "PanelBp" not in b.columns:
        b = b.rename(columns={"TotalBp": "PanelBp"})
    b["Platform"] = b["Platform"].map(lambda s: A2_TO_DISPLAY.get(s, FILE_TO_DISPLAY.get(s, s)))
    b["Group"] = b["Group"].astype(int)
    b["Reads"] = b["Reads"].astype(int)
    if b.empty:
        logger.warning("BAM validation raw table is empty; skipping BAM check.")
        return
    m = b.merge(
        df[["Group", "Platform", "Reads", "Median_DP", "Mean_DP", "Q25_DP", "Q75_DP"]],
        on=["Group", "Platform", "Reads"], how="left",
    )
    m["Ratio_BAMmean_to_PredMedian"] = m["MeanDepthAllBp"] / m["Median_DP"]
    m["Ratio_BAMmedian_to_PredMedian"] = m["MedianDepth"] / m["Median_DP"]
    m["PctDiff_BAMmean_vs_PredMedian"] = 100 * (m["Ratio_BAMmean_to_PredMedian"] - 1)
    m["Rank_PredMedian"] = m.groupby(["Group", "Reads"])["Median_DP"].rank(method="min", ascending=False).astype(int)
    m["Rank_BAMmean"] = m.groupby(["Group", "Reads"])["MeanDepthAllBp"].rank(method="min", ascending=False).astype(int)
    m["Rank_BAMmedian"] = m.groupby(["Group", "Reads"])["MedianDepth"].rank(method="min", ascending=False).astype(int)
    cols = [
        "Group", "Reads", "Platform", "PanelBp", "CoveredBp", "SumDepth",
        "MeanDepthAllBp", "MeanDepthCovered", "MedianDepth",
        "Median_DP", "Mean_DP", "Q25_DP", "Q75_DP",
        "Ratio_BAMmean_to_PredMedian", "Ratio_BAMmedian_to_PredMedian",
        "PctDiff_BAMmean_vs_PredMedian", "Rank_PredMedian", "Rank_BAMmean", "Rank_BAMmedian",
    ]
    m = m[[c for c in cols if c in m.columns]].sort_values(["Group", "Reads", "Platform"])
    m.to_csv(TABDIR / "D1_bam_depth_validation.csv", index=False, float_format="%.6f")

    corr_rows = []
    for (g, r), s in m.groupby(["Group", "Reads"]):
        s = s.dropna(subset=["Median_DP", "MeanDepthAllBp"])
        if len(s) >= 3:
            pr_mean, pp_mean = pearsonr(s["Median_DP"], s["MeanDepthAllBp"])
            sr_mean, sp_mean = spearmanr(s["Median_DP"], s["MeanDepthAllBp"])
            pr_med, pp_med = pearsonr(s["Median_DP"], s["MedianDepth"])
            sr_med, sp_med = spearmanr(s["Median_DP"], s["MedianDepth"])
            corr_rows.append({
                "Group": g, "Reads": r, "N": len(s),
                "Pearson_PredMedian_vs_BAMmean": pr_mean, "p_Pearson_mean": pp_mean,
                "Spearman_PredMedian_vs_BAMmean": sr_mean, "p_Spearman_mean": sp_mean,
                "Pearson_PredMedian_vs_BAMmedian": pr_med, "p_Pearson_median": pp_med,
                "Spearman_PredMedian_vs_BAMmedian": sr_med, "p_Spearman_median": sp_med,
                "Mean_ratio_BAMmean_over_PredMedian": float(s["Ratio_BAMmean_to_PredMedian"].mean()),
                "SD_ratio": float(s["Ratio_BAMmean_to_PredMedian"].std(ddof=1)),
                "Min_ratio": float(s["Ratio_BAMmean_to_PredMedian"].min()),
                "Max_ratio": float(s["Ratio_BAMmean_to_PredMedian"].max()),
            })
    # pooled across conditions (e.g. 100M + 140M)
    s = m.dropna(subset=["Median_DP", "MeanDepthAllBp"])
    if len(s) >= 3:
        pr, pp = pearsonr(s["Median_DP"], s["MeanDepthAllBp"])
        sr, sp = spearmanr(s["Median_DP"], s["MeanDepthAllBp"])
        corr_rows.append({
            "Group": "ALL", "Reads": "ALL", "N": len(s),
            "Pearson_PredMedian_vs_BAMmean": pr, "p_Pearson_mean": pp,
            "Spearman_PredMedian_vs_BAMmean": sr, "p_Spearman_mean": sp,
            "Pearson_PredMedian_vs_BAMmedian": np.nan, "p_Pearson_median": np.nan,
            "Spearman_PredMedian_vs_BAMmedian": np.nan, "p_Spearman_median": np.nan,
            "Mean_ratio_BAMmean_over_PredMedian": float(s["Ratio_BAMmean_to_PredMedian"].mean()),
            "SD_ratio": float(s["Ratio_BAMmean_to_PredMedian"].std(ddof=1)),
            "Min_ratio": float(s["Ratio_BAMmean_to_PredMedian"].min()),
            "Max_ratio": float(s["Ratio_BAMmean_to_PredMedian"].max()),
        })
    corr = pd.DataFrame(corr_rows)
    corr.to_csv(TABDIR / "D1_bam_depth_correlation.csv", index=False, float_format="%.6f")
    logger.info("")
    logger.info("================ BAM CHECK OF THE DEDUP-DEPTH PROXY ================")
    panel_total = int(
        pd.read_csv(PANEL_BED, sep="\t", header=None, names=["c", "s", "e"]).eval("e - s").sum()
    )
    logger.info(
        "BAM mean depth is from samtools depth over %d sampled panel intervals "
        "(%.1f kb, ~%.1f%% of the %.2f Mb panel); duplicate-marked reads are excluded by default.",
        len(pd.read_csv(SAMPLE_BED, sep="\t", header=None)),
        m["PanelBp"].iloc[0] / 1000,
        100 * m["PanelBp"].iloc[0] / panel_total,
        panel_total / 1e6,
    )
    log_df(
        logger,
        m[["Group", "Reads", "Platform", "Median_DP", "MeanDepthAllBp", "MedianDepth",
           "Ratio_BAMmean_to_PredMedian", "Rank_PredMedian", "Rank_BAMmean"]]
        .round(3).set_index(["Group", "Reads", "Platform"]),
        "Predicted median candidate DP vs BAM depth",
    )
    log_df(logger, corr.round(4), "Correlation across platforms")


# --------------------------------------------------------------------------
# Figure
# --------------------------------------------------------------------------
def make_figure(fit_df: pd.DataFrame, df: pd.DataFrame, logger: logging.Logger):
    plt.rcParams.update({
        "font.size": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.6), sharex="col")
    for ri, g in enumerate(GROUPS):
        sub = df[df.Group == g]
        lo, hi = common_range(sub)
        for ci, metric in enumerate(METRICS):
            ax = axes[ri, ci]
            ax.axvspan(lo, hi, color="#d9d9d9", alpha=0.45, lw=0, zorder=0)
            for plat in PLATFORMS:
                s = sub[sub.Platform == plat].sort_values("Median_DP")
                color = COLORS[plat]
                lw = 3.2 if plat == SURF else 1.8
                ms = 7.5 if plat == SURF else 5.0
                z = 5 if plat == SURF else 3
                ax.plot(s["Median_DP"], s[metric], "-o", color=color, lw=1.0,
                        ms=ms, alpha=0.95, zorder=z, markeredgecolor="white",
                        markeredgewidth=0.6)
                grid = np.geomspace(lo, hi, GRID_N_ALL)
                fitted = fit_df[
                    (fit_df.Group == g) & (fit_df.Scope == "all6")
                    & (fit_df.Platform == plat) & (fit_df["F1_metric"] == metric)
                    & (fit_df.Source == "linear_log10_DP")
                ].sort_values("MedianDP")
                ax.plot(fitted["MedianDP"], fitted["F1"], "-", color=color, lw=lw,
                        alpha=0.95, zorder=z)
            ax.set_xscale("log")
            ax.set_xlabel("Median Mutect2 DP at candidate sites (deduplicated, log scale)")
            ax.set_ylabel("F1" if metric == "F1@0.1" else "Best achievable F1")
            ax.set_title(
                f"{'Group 45' if g == 45 else 'Group 54'}  |  "
                f"{'F1 at pred > 0.1' if metric == 'F1@0.1' else 'F1 best (threshold-free)'}",
                fontsize=10,
            )
            ax.grid(axis="y", color="#e6e6e6", lw=0.6)
            ax.text(
                0.02, 0.025,
                f"shaded: all-platform overlap {lo:.0f}-{hi:.0f}x\n"
                f"points: 5 raw depths/platform; lines: linear fit",
                transform=ax.transAxes, fontsize=7.2, color="#444444", va="bottom",
            )
            if ri == 0 and ci == 0:
                # Legend handled globally below to avoid duplicate clutter
                pass
    handles = [
        plt.Line2D([0], [0], color=COLORS[p], marker="o", lw=2.5 if p == SURF else 1.6,
                   ms=7 if p == SURF else 5, label=p)
        for p in PLATFORMS
    ]
    fig.legend(handles=handles, loc="upper center", ncol=6, frameon=False,
               bbox_to_anchor=(0.5, 0.985), fontsize=8.5)
    fig.suptitle(
        "F1 versus deduplicated on-target depth (candidate-site Mutect2 DP)",
        y=0.945, fontsize=11.5,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    for ext in ["png", "pdf"]:
        fig.savefig(FIGDIR / f"D1_f1_vs_dedup_depth.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("Wrote %s and %s", FIGDIR / "D1_f1_vs_dedup_depth.png", FIGDIR / "D1_f1_vs_dedup_depth.pdf")


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    global PRED_ROOT
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pred-root", type=Path, default=PRED_ROOT)
    ap.add_argument("--run-bam", action="store_true",
                    help="Run BAM depth validation (slow: ~2-3 min per BAM).")
    ap.add_argument("--bam-group", type=int, default=45)
    ap.add_argument("--bam-reads", type=int, nargs="+", default=[100],
                    help="One or more nominal raw depths for BAM validation.")
    ap.add_argument("--bam-workers", type=int, default=6)
    ap.add_argument("--skip-figure", action="store_true")
    args = ap.parse_args()

    PRED_ROOT = args.pred_root
    TABDIR.mkdir(parents=True, exist_ok=True)
    FIGDIR.mkdir(parents=True, exist_ok=True)
    LOGDIR.mkdir(parents=True, exist_ok=True)

    logger = setup_logger()
    t0 = time.time()
    logger.info("D1 matched-deduplicated-depth sensitivity analysis")
    logger.info("pred root: %s", PRED_ROOT)
    logger.info("A2 cross-check: %s", A2_FILE)
    logger.info("panel BED: %s", PANEL_BED)
    logger.info("python: %s", sys.version.replace("\n", " "))
    logger.info("pandas %s | numpy %s", pd.__version__, np.__version__)

    # Optional BAM validation first (so its results can be included below)
    if args.run_bam:
        frames = []
        for r in args.bam_reads:
            frames.append(run_bam_validation(args.bam_group, r, PLATFORMS, logger,
                                             workers=args.bam_workers))
        if frames:
            out = pd.concat(frames, ignore_index=True)
            out.to_csv(BAM_RAW, sep="\t", index=False, float_format="%.6f")
            logger.info("Wrote BAM raw depth table: %s", BAM_RAW)

    df = load_conditions(logger)
    df = describe_dp(df, logger)

    # Cross-check A2
    if df["A2_F1@0.1"].notna().any():
        logger.info("")
        logger.info("---- Cross-check against A2_paired_by_condition.csv (60 conditions) ----")
        logger.info(
            "F1@0.1: max |recomputed - A2| = %.4f; F1_best: max |difference| = %.4f. "
            "The pred tables are the released-model scores used here; A2 used its own rerun.",
            df["Delta_F1@0.1_vs_A2"].abs().max(),
            df["Delta_F1_best_vs_A2"].abs().max(),
        )

    # Condition table
    cond_cols = [
        "Group", "Platform", "PlatformFileID", "Reads", "N_candidates", "N_pos",
        "Median_DP", "Mean_DP", "Q25_DP", "Q75_DP",
        "F1@0.1", "F1_best", "Best_threshold",
        "A2_F1@0.1", "A2_F1_best", "Delta_F1@0.1_vs_A2", "Delta_F1_best_vs_A2",
        "MedianDP_rank_in_group", "RawReads_rank_in_group",
    ]
    df[cond_cols].to_csv(TABDIR / "D1_dedup_depth_by_condition.csv", index=False,
                         float_format="%.6f")
    logger.info("Wrote %s", TABDIR / "D1_dedup_depth_by_condition.csv")

    # F1 matrices for the log
    for g in GROUPS:
        log_df(
            logger,
            df[df.Group == g].pivot(index="Platform", columns="Reads", values="F1@0.1")
            .reindex(PLATFORMS)[READS].round(4),
            f"Group {g}: observed F1@0.1 by platform x raw depth",
        )
        log_df(
            logger,
            df[df.Group == g].pivot(index="Platform", columns="Reads", values="F1_best")
            .reindex(PLATFORMS)[READS].round(4),
            f"Group {g}: observed F1_best by platform x raw depth",
        )

    # Matching
    logger.info("")
    logger.info("================ NEAREST-NEIGHBOUR MATCHING ================")
    mp = build_matched_pairs(df, logger)
    mp.to_csv(TABDIR / "D1_matched_pairs.csv", index=False, float_format="%.6f")
    logger.info("Wrote %s", TABDIR / "D1_matched_pairs.csv")
    dir_counts = summarise_matched_pairs(mp, logger)
    dir_counts.to_csv(TABDIR / "D1_matched_direction_counts.csv", index=False,
                      float_format="%.6f")

    # Interpolation
    logger.info("")
    logger.info("================ INTERPOLATION ================")
    fit_stats, fits = fit_models(df)
    fit_stats.to_csv(TABDIR / "D1_interpolation_fit_stats.csv", index=False,
                     float_format="%.6f")
    log_df(logger, fit_stats.round(4).set_index(["Group", "F1_metric", "Platform"]),
           "Interpolation fit statistics")
    fit_df, rank_df, pair_df, pair_sum = build_interpolation_tables(df, fits, logger)
    fit_df.to_csv(TABDIR / "D1_f1_vs_dedup_depth.csv", index=False, float_format="%.6f")
    rank_df.to_csv(TABDIR / "D1_rank_at_matched_depth.csv", index=False, float_format="%.6f")
    pair_df.to_csv(TABDIR / "D1_pairwise_matched_depth.csv", index=False, float_format="%.6f")
    pair_sum.to_csv(TABDIR / "D1_pairwise_interpolation_summary.csv", index=False,
                    float_format="%.6f")
    logger.info("Wrote %s", TABDIR / "D1_f1_vs_dedup_depth.csv")
    logger.info("Wrote %s", TABDIR / "D1_rank_at_matched_depth.csv")
    logger.info("Wrote %s", TABDIR / "D1_pairwise_matched_depth.csv")
    summarise_interpolation(rank_df, pair_sum, logger)

    # Raw-depth control
    raw = raw_depth_control(df, logger)
    raw.to_csv(TABDIR / "D1_raw_depth_ranking.csv", index=False, float_format="%.6f")
    logger.info("Wrote %s", TABDIR / "D1_raw_depth_ranking.csv")

    # BAM validation
    merge_bam_validation(df, logger)

    # Figure
    if not args.skip_figure:
        make_figure(fit_df, df, logger)

    # Final descriptive summary
    logger.info("")
    logger.info("================ FINAL DESCRIPTIVE SUMMARY ================")
    for metric in METRICS:
        for g in GROUPS:
            raw_s = raw[(raw.Group == g) & (raw["F1_metric"] == metric) & (raw.Platform == SURF)]
            rank_s = rank_df[
                (rank_df.Group == g) & (rank_df["F1_metric"] == metric)
                & (rank_df.Model == "linear_log10_DP")
            ].drop_duplicates("GridIndex")
            logger.info(
                "Group %s | %s: raw-depth SURF rank mean %.2f (ranks %s); "
                "all6 matched-DP SURF rank mean %.2f (ranks %s)",
                g, metric,
                raw_s["Rank"].mean(), sorted(raw_s["Rank"].tolist()),
                rank_s["SURFSeq_Rank"].mean(),
                rank_s["SURFSeq_Rank"].value_counts().sort_index().to_dict(),
            )
    for level in ["pair_optimal", "surf_anchor", "comparator_anchor"]:
        for metric in METRICS:
            s = dir_counts[
                (dir_counts.Group == "ALL") & (dir_counts.Level == level)
                & (dir_counts["F1_metric"] == metric) & (dir_counts.Comparator == "ALL")
            ]
            if not s.empty:
                r = s.iloc[0]
                logger.info(
                    "Matched NN | ALL | %s | %s: SURF higher %d, tie %d, lower %d; mean delta %.4f",
                    level, metric, r["SURF_higher"], r["Tie"], r["SURF_lower"],
                    r["Mean_delta"],
                )
    logger.info("")
    logger.info("Done in %.1f s", time.time() - t0)


if __name__ == "__main__":
    main()
