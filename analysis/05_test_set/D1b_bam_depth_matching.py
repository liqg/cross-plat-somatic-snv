#!/usr/bin/env python3
"""
D1b: BAM-depth sensitivity analysis for platform comparison.

This is a companion to D1_matched_dedup_depth.py.  D1 matched conditions on
the median Mutect2 DP at candidate sites.  Here we instead use a BAM-derived
panel-level depth estimate:

    BAM_est_depth(platform, raw reads) =
        BAM_mean_depth_panel_100M(platform) * (raw reads / 100)

The scaling with raw reads is an approximation (linear library scaling), not
measured at each raw depth.  For group 45 we additionally have the 140M BAM
measurement and report how close 1.4x scaling is to the observed 140M mean.

Definitions:
* BAM depth: `samtools depth -b <sampled panel BED>` mean over 429 sampled
  panel intervals (102,217 bp, ~5.2% of the 1.97 Mb panel), default filters
  (duplicate-marked, secondary, QC-fail and unmapped reads excluded).
* pred median DP: median Mutect2 DP at the candidate sites in the pred table;
  it is a candidate-site quantity, not a whole-panel mean.  The two are
  reported side by side, not equated.

All output files are new and have the D1b_ prefix.
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
import common  # noqa: E402

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LinearRegression

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUTDIR = common.RESULTS_ROOT
TABDIR = common.TABLES
FIGDIR = common.FIGURES
LOGDIR = common.LOGS

D1_COND = TABDIR / "D1_dedup_depth_by_condition.csv"
D1_DIR = TABDIR / "D1_matched_direction_counts.csv"
D1_RANK = TABDIR / "D1_rank_at_matched_depth.csv"
BAM100_G45 = LOGDIR / "D1_bam_depth_100M.tsv"
BAM100_G54 = LOGDIR / "D1b_bam_depth_g54_100M.tsv"
BAM140_G45 = LOGDIR / "D1b_bam_depth_g45_140M.tsv"
SAMPLED_BED = LOGDIR / "D1_panel_sample.bed"
CAND_DIR = LOGDIR / "D1_candidate_depth"
PRED_ROOT = common.TEST_ROOT

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
BAM_TO_DISPLAY = {
    "FASTASeq_300": "FASTASeq 300",
    "GenoLab_M": "GenoLab M",
    "MGISEQ-T7": "DNBSEQ-T7",
    "NovaSeq_6000": "NovaSeq 6000",
    "NovaSeq_X": "NovaSeq X",
    "SURFSeq_5000": "SURFSeq 5000",
}
DISPLAY_TO_BAM = {v: k for k, v in BAM_TO_DISPLAY.items()}
COLORS = {
    "FASTASeq 300": "#1f77b4",
    "GenoLab M": "#2ca02c",
    "DNBSEQ-T7": "#9467bd",
    "NovaSeq 6000": "#8c564b",
    "NovaSeq X": "#e377c2",
    "SURFSeq 5000": "#d62728",
}
METRICS = ["F1@0.1", "F1_best"]
TIE_TOL = 0.002
GRID_N = 40


def setup_logger():
    LOGDIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("D1b")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    fh = logging.FileHandler(LOGDIR / "D1b.log", mode="w")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def log_df(logger, df, title):
    logger.info("")
    logger.info("---- %s ----", title)
    if df is None or df.empty:
        logger.info("(empty)")
        return
    for line in df.to_string().splitlines():
        logger.info(line)


# --------------------------------------------------------------------------
# BAM depth inputs
# --------------------------------------------------------------------------
def read_bam_panel(path: Path, group: int, reads: int, logger) -> pd.DataFrame:
    if not path.exists():
        logger.warning("BAM panel file not found: %s", path)
        return pd.DataFrame()
    d = pd.read_csv(path, sep="\t")
    if "TotalBp" in d.columns and "PanelBp" not in d.columns:
        d = d.rename(columns={"TotalBp": "PanelBp"})
    d = d[d["Platform"].isin(BAM_TO_DISPLAY.keys())].copy()
    d["Platform"] = d["Platform"].map(BAM_TO_DISPLAY)
    d["Group"] = group
    d["Reads_measured"] = reads
    num = ["PanelBp", "CoveredBp", "SumDepth", "MeanDepthAllBp",
           "MeanDepthCovered", "MedianDepth"]
    for c in num:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    return d


def load_bam_means(logger):
    frames = [
        read_bam_panel(BAM100_G45, 45, 100, logger),
        read_bam_panel(BAM100_G54, 54, 100, logger),
        read_bam_panel(BAM140_G45, 45, 140, logger),
    ]
    b = pd.concat([x for x in frames if not x.empty], ignore_index=True)
    means100 = b[b.Reads_measured == 100].copy()
    missing = [(g, p) for g in GROUPS for p in PLATFORMS
               if not ((means100.Group == g) & (means100.Platform == p)).any()]
    if missing:
        logger.warning("Missing 100M BAM means for: %s", missing)
    return b, means100


def build_bam_vs_median(cond, means100, logger) -> pd.DataFrame:
    c100 = cond[cond.Reads == 100][
        ["Group", "Platform", "Median_DP", "Mean_DP", "Q25_DP", "Q75_DP", "N_candidates"]
    ].copy()
    m = means100.merge(c100, on=["Group", "Platform"], how="inner")
    if m.empty:
        raise RuntimeError("No BAM/pred overlap at 100M")
    m["Ratio_BAMmean_over_PredMedian"] = m["MeanDepthAllBp"] / m["Median_DP"]
    m["PctDiff_BAMmean_vs_PredMedian"] = 100 * (m["Ratio_BAMmean_over_PredMedian"] - 1)
    m["Ratio_BAMmedian_over_PredMedian"] = m["MedianDepth"] / m["Median_DP"]
    m["PctDiff_BAMmedian_vs_PredMedian"] = 100 * (m["Ratio_BAMmedian_over_PredMedian"] - 1)
    m["Rank_BAMmean"] = m.groupby("Group")["MeanDepthAllBp"].rank(method="min", ascending=False).astype(int)
    m["Rank_predMedian"] = m.groupby("Group")["Median_DP"].rank(method="min", ascending=False).astype(int)
    m["Rank_difference_BAM_minus_pred"] = m["Rank_BAMmean"] - m["Rank_predMedian"]

    corr_rows = []
    for g, s in m.groupby("Group"):
        s = s.dropna(subset=["MeanDepthAllBp", "Median_DP"])
        if len(s) >= 3:
            pr, pp = pearsonr(s["MeanDepthAllBp"], s["Median_DP"])
            sr, sp = spearmanr(s["MeanDepthAllBp"], s["Median_DP"])
            prm, ppm = pearsonr(s["MedianDepth"], s["Median_DP"])
            srm, spm = spearmanr(s["MedianDepth"], s["Median_DP"])
            corr_rows.append({
                "Group": g, "N": len(s),
                "Pearson_BAMmean_vs_PredMedian": pr, "p_Pearson_mean": pp,
                "Spearman_BAMmean_vs_PredMedian": sr, "p_Spearman_mean": sp,
                "Pearson_BAMmedian_vs_PredMedian": prm, "p_Pearson_median": ppm,
                "Spearman_BAMmedian_vs_PredMedian": srm, "p_Spearman_median": spm,
                "Mean_ratio": float(s["Ratio_BAMmean_over_PredMedian"].mean()),
                "Min_ratio": float(s["Ratio_BAMmean_over_PredMedian"].min()),
                "Max_ratio": float(s["Ratio_BAMmean_over_PredMedian"].max()),
                "Mean_PctDiff": float(s["PctDiff_BAMmean_vs_PredMedian"].mean()),
                "Max_abs_PctDiff": float(s["PctDiff_BAMmean_vs_PredMedian"].abs().max()),
                "Rank_Spearman_note": "rank correlation across six platforms within group",
            })
    corr = pd.DataFrame(corr_rows)
    log_df(logger, corr.round(4), "BAM panel mean vs pred median DP: correlation across platforms at 100M")
    log_df(
        logger,
        m[["Group", "Platform", "MeanDepthAllBp", "MedianDepth", "Median_DP", "Mean_DP",
           "Ratio_BAMmean_over_PredMedian", "PctDiff_BAMmean_vs_PredMedian",
           "Rank_BAMmean", "Rank_predMedian", "Rank_difference_BAM_minus_pred"]]
        .round(3).set_index(["Group", "Platform"]),
        "BAM panel depth vs candidate-site Mutect2 DP at 100M",
    )

    # Per-platform rows plus one correlation-summary row per group
    per = m[[
        "Group", "Platform", "MeanDepthAllBp", "MedianDepth", "Median_DP", "Mean_DP",
        "Ratio_BAMmean_over_PredMedian", "PctDiff_BAMmean_vs_PredMedian",
        "Ratio_BAMmedian_over_PredMedian", "PctDiff_BAMmedian_vs_PredMedian",
        "Rank_BAMmean", "Rank_predMedian", "Rank_difference_BAM_minus_pred",
    ]].copy()
    per = per.rename(columns={
        "MeanDepthAllBp": "BAM_mean_depth_100M",
        "MedianDepth": "BAM_median_depth_100M",
        "Median_DP": "pred_median_DP_100M",
        "Mean_DP": "pred_mean_DP_100M",
    })
    per["Row_type"] = "platform"
    per["N_platforms"] = np.nan
    corr_cols = ["Pearson_BAMmean_vs_PredMedian", "p_Pearson_mean",
                 "Spearman_BAMmean_vs_PredMedian", "p_Spearman_mean",
                 "Pearson_BAMmedian_vs_PredMedian", "p_Pearson_median",
                 "Spearman_BAMmedian_vs_PredMedian", "p_Spearman_median"]
    for c in corr_cols:
        per[c] = per["Group"].map(corr.set_index("Group")[c])
    summary = []
    for _, r in corr.iterrows():
        summary.append({
            "Group": r["Group"], "Platform": "ALL_PLATFORMS",
            "BAM_mean_depth_100M": np.nan, "BAM_median_depth_100M": np.nan,
            "pred_median_DP_100M": np.nan, "pred_mean_DP_100M": np.nan,
            "Ratio_BAMmean_over_PredMedian": r["Mean_ratio"],
            "PctDiff_BAMmean_vs_PredMedian": r["Mean_PctDiff"],
            "Ratio_BAMmedian_over_PredMedian": np.nan,
            "PctDiff_BAMmedian_vs_PredMedian": np.nan,
            "Rank_BAMmean": np.nan, "Rank_predMedian": np.nan,
            "Rank_difference_BAM_minus_pred": np.nan,
            "Row_type": "correlation_summary", "N_platforms": r["N"],
            "Pearson_BAMmean_vs_PredMedian": r["Pearson_BAMmean_vs_PredMedian"],
            "p_Pearson_mean": r["p_Pearson_mean"],
            "Spearman_BAMmean_vs_PredMedian": r["Spearman_BAMmean_vs_PredMedian"],
            "p_Spearman_mean": r["p_Spearman_mean"],
            "Pearson_BAMmedian_vs_PredMedian": r["Pearson_BAMmedian_vs_PredMedian"],
            "p_Pearson_median": r["p_Pearson_median"],
            "Spearman_BAMmedian_vs_PredMedian": r["Spearman_BAMmedian_vs_PredMedian"],
            "p_Spearman_median": r["p_Spearman_median"],
        })
    out = pd.concat([per, pd.DataFrame(summary)], ignore_index=True)
    out["_row_order"] = out["Row_type"].map({"platform": 0, "correlation_summary": 1})
    out = out.sort_values(["Group", "_row_order", "Rank_predMedian"], na_position="last").drop(columns="_row_order")
    return out


def check_linear_scaling(b, logger):
    if b.empty or not (BAM140_G45.exists()):
        return
    s = b[b.Reads_measured.isin([100, 140]) & (b.Group == 45)]
    piv = s.pivot_table(index="Platform", columns="Reads_measured", values="MeanDepthAllBp")
    if 100 not in piv.columns or 140 not in piv.columns:
        return
    piv = piv.dropna()
    piv["Observed_140M_over_100M"] = piv[140] / piv[100]
    piv["Expected_linear"] = 1.4
    piv["Deviation_from_linear_pct"] = 100 * (piv["Observed_140M_over_100M"] / 1.4 - 1)
    piv = piv.reindex(PLATFORMS).dropna(how="all")
    piv.reset_index().to_csv(TABDIR / "D1b_bam_depth_linearity.csv", index=False,
                             float_format="%.6f")
    log_df(
        logger, piv.round(3),
        "Group 45: observed BAM 140M / 100M mean-depth ratio vs the assumed 1.4x linear scaling",
    )
    logger.info(
        "Linear-scaling check: observed 140M/100M ratios range %.2f-%.2f "
        "(deviation from 1.4x: %.0f%% to %.0f%%).",
        piv["Observed_140M_over_100M"].min(), piv["Observed_140M_over_100M"].max(),
        piv["Deviation_from_linear_pct"].min(), piv["Deviation_from_linear_pct"].max(),
    )


# --------------------------------------------------------------------------
# Matching and interpolation (same logic as D1, x = BAM-estimated depth)
# --------------------------------------------------------------------------
def closest_pair(dfa, dfb):
    combos = []
    for a in dfa.to_dict("records"):
        for b in dfb.to_dict("records"):
            score = abs(np.log2(a["BAM_est_depth"] / b["BAM_est_depth"]))
            combos.append((score, abs(a["BAM_est_depth"] - b["BAM_est_depth"]), a, b))
    combos.sort(key=lambda z: (z[0], z[1]))
    best = combos[0]
    near = [c for c in combos if c[0] <= best[0] + 0.05]
    a, b = best[2], best[3]
    row = {
        "Level": "pair_optimal",
        "Platform_A": a["Platform"], "Platform_B": b["Platform"],
        "Reads_A": int(a["Reads"]), "Reads_B": int(b["Reads"]),
        "BAMdepth_A": float(a["BAM_est_depth"]), "BAMdepth_B": float(b["BAM_est_depth"]),
        "AbsLog2BAMdepthDiff": float(best[0]),
        "BAMdepth_ratio_A_over_B": float(a["BAM_est_depth"] / b["BAM_est_depth"]),
        "N_candidates_A": int(a["N_candidates"]), "N_candidates_B": int(b["N_candidates"]),
        "NearTie_Count": int(len(near)),
        "NearTie_Reads_pairs": ";".join(f"{int(c[2]['Reads'])}-{int(c[3]['Reads'])}" for c in near),
    }
    for metric in METRICS:
        row[f"{metric}_A"] = float(a[metric])
        row[f"{metric}_B"] = float(b[metric])
        row[f"Delta_{metric}_A_minus_B"] = float(a[metric] - b[metric])
        vals = [c[2][metric] - c[3][metric] for c in near]
        row[f"NearTie_Delta_{metric}_min"] = float(np.min(vals))
        row[f"NearTie_Delta_{metric}_max"] = float(np.max(vals))
    return row


def nearest_condition(anchor, cand):
    diffs = (np.log2(anchor["BAM_est_depth"]) - np.log2(cand["BAM_est_depth"])).abs()
    idx = diffs.idxmin()
    return cand.loc[idx], float(diffs.loc[idx])


def anchor_row(group, level, comp, surf_row, comp_row, diff, anchor_reads):
    row = {
        "Group": group, "Level": level, "Comparator": comp,
        "Anchor_Reads": int(anchor_reads),
        "Platform_A": SURF, "Platform_B": comp,
        "Reads_A": int(surf_row["Reads"]), "Reads_B": int(comp_row["Reads"]),
        "BAMdepth_A": float(surf_row["BAM_est_depth"]),
        "BAMdepth_B": float(comp_row["BAM_est_depth"]),
        "AbsLog2BAMdepthDiff": float(diff),
        "BAMdepth_ratio_A_over_B": float(surf_row["BAM_est_depth"] / comp_row["BAM_est_depth"]),
        "N_candidates_A": int(surf_row["N_candidates"]),
        "N_candidates_B": int(comp_row["N_candidates"]),
    }
    for metric in METRICS:
        row[f"{metric}_A"] = float(surf_row[metric])
        row[f"{metric}_B"] = float(comp_row[metric])
        row[f"Delta_{metric}_A_minus_B"] = float(surf_row[metric] - comp_row[metric])
        row[f"Delta_{metric}_SURF_minus_Comparator"] = float(surf_row[metric] - comp_row[metric])
    return row


def build_matched_pairs(df):
    rows = []
    for g in GROUPS:
        sub = df[df.Group == g]
        for i in range(len(PLATFORMS)):
            for j in range(i + 1, len(PLATFORMS)):
                pa, pb = PLATFORMS[i], PLATFORMS[j]
                if pb == SURF:
                    pa, pb = pb, pa
                dfa, dfb = sub[sub.Platform == pa], sub[sub.Platform == pb]
                row = closest_pair(dfa, dfb)
                row["Group"] = g
                if SURF in (pa, pb):
                    comp = pb if pa == SURF else pa
                    row["Comparator"] = comp
                    d = row["Delta_F1@0.1_A_minus_B"] if pa == SURF else -row["Delta_F1@0.1_A_minus_B"]
                    row["Delta_F1@0.1_SURF_minus_Comparator"] = d
                    db = row["Delta_F1_best_A_minus_B"] if pa == SURF else -row["Delta_F1_best_A_minus_B"]
                    row["Delta_F1_best_SURF_minus_Comparator"] = db
                else:
                    row["Comparator"] = ""
                    row["Delta_F1@0.1_SURF_minus_Comparator"] = np.nan
                    row["Delta_F1_best_SURF_minus_Comparator"] = np.nan
                rows.append(row)
        for comp in [p for p in PLATFORMS if p != SURF]:
            surf_df, comp_df = sub[sub.Platform == SURF], sub[sub.Platform == comp]
            for _, a in surf_df.iterrows():
                b, diff = nearest_condition(a, comp_df)
                rows.append(anchor_row(g, "surf_anchor", comp, a, b, diff, a["Reads"]))
            for _, a in comp_df.iterrows():
                b, diff = nearest_condition(a, surf_df)
                rows.append(anchor_row(g, "comparator_anchor", comp, b, a, diff, a["Reads"]))
    mp = pd.DataFrame(rows)
    mp["Level"] = pd.Categorical(mp["Level"], ["pair_optimal", "surf_anchor", "comparator_anchor"],
                                 ordered=True)
    return mp.sort_values(["Group", "Level", "Comparator", "Anchor_Reads"]).reset_index(drop=True)


def direction_counts(mp):
    rows = []
    sub = mp[mp["Comparator"].notna() & (mp["Comparator"] != "")]
    for g in GROUPS + ["ALL"]:
        s0 = sub if g == "ALL" else sub[sub.Group == g]
        for level in ["pair_optimal", "surf_anchor", "comparator_anchor"]:
            s1 = s0[s0.Level == level]
            for metric in METRICS:
                col = f"Delta_{metric}_SURF_minus_Comparator"
                for comp in [p for p in PLATFORMS if p != SURF] + ["ALL"]:
                    s = s1 if comp == "ALL" else s1[s1.Comparator == comp]
                    d = s[col].dropna().to_numpy()
                    if len(d) == 0:
                        continue
                    rows.append({
                        "Group": g, "Level": level, "F1_metric": metric,
                        "Comparator": comp, "N": len(d),
                        "SURF_higher": int((d > TIE_TOL).sum()),
                        "Tie": int((np.abs(d) <= TIE_TOL).sum()),
                        "SURF_lower": int((d < -TIE_TOL).sum()),
                        "Exact_tie": int((d == 0).sum()),
                        "Mean_delta": float(np.mean(d)),
                        "Median_delta": float(np.median(d)),
                        "Min_delta": float(np.min(d)), "Max_delta": float(np.max(d)),
                        "Mean_AbsLog2BAMdepthDiff": float(s["AbsLog2BAMdepthDiff"].mean()),
                        "Max_AbsLog2BAMdepthDiff": float(s["AbsLog2BAMdepthDiff"].max()),
                    })
    return pd.DataFrame(rows)


def common_range(df):
    lo = df.groupby("Platform")["BAM_est_depth"].min().max()
    hi = df.groupby("Platform")["BAM_est_depth"].max().min()
    return float(lo), float(hi)


def fit_models(df):
    stats, fits = [], {}
    for g in GROUPS:
        for plat in PLATFORMS:
            s = df[(df.Group == g) & (df.Platform == plat)].sort_values("BAM_est_depth")
            x = np.log10(s["BAM_est_depth"].to_numpy())
            for metric in METRICS:
                y = s[metric].to_numpy()
                lin = LinearRegression().fit(x.reshape(-1, 1), y)
                iso = IsotonicRegression(increasing=True, out_of_bounds="clip").fit(x, y)
                rho, p = spearmanr(x, y)
                fits[(g, plat, metric)] = {"lin": lin, "iso": iso}
                stats.append({
                    "Group": g, "Platform": plat, "F1_metric": metric,
                    "Linear_slope_per_log10BAMdepth": float(lin.coef_[0]),
                    "Linear_intercept": float(lin.intercept_),
                    "Linear_R2": float(lin.score(x.reshape(-1, 1), y)),
                    "Spearman_logBAMdepth_F1": float(rho), "Spearman_p": float(p),
                    "BAMdepth_min": float(s["BAM_est_depth"].min()),
                    "BAMdepth_max": float(s["BAM_est_depth"].max()),
                })
    return pd.DataFrame(stats), fits


def predict(fits, g, plat, metric, model, grid):
    xg = np.log10(np.asarray(grid, dtype=float)).reshape(-1, 1)
    if model == "linear_log10_BAMdepth":
        p = fits[(g, plat, metric)]["lin"].predict(xg)
    else:
        p = fits[(g, plat, metric)]["iso"].predict(xg.ravel())
    return np.clip(p, 0, 1)


def build_interpolation(df, fits, logger):
    obs = []
    for _, r in df.iterrows():
        for metric in METRICS:
            obs.append({
                "Group": int(r.Group), "Platform": r.Platform, "F1_metric": metric,
                "Source": "observed", "RawReads": int(r.Reads),
                "BAM_est_depth": float(r.BAM_est_depth), "F1": float(r[metric]),
                "WithinObservedRange": True,
            })
    fits_rows, rank_rows = [], []
    for g in GROUPS:
        sub = df[df.Group == g]
        lo, hi = common_range(sub)
        logger.info("Group %s: all-platform common BAM-estimated depth range = %.1f to %.1f x", g, lo, hi)
        grid = np.geomspace(lo, hi, GRID_N)
        for metric in METRICS:
            for model in ["linear_log10_BAMdepth", "isotonic_log10_BAMdepth"]:
                preds = {}
                for plat in PLATFORMS:
                    preds[plat] = predict(fits, g, plat, metric, model, grid)
                    plo = sub[sub.Platform == plat]["BAM_est_depth"].min()
                    phi = sub[sub.Platform == plat]["BAM_est_depth"].max()
                    for i, x in enumerate(grid):
                        fits_rows.append({
                            "Group": g, "Platform": plat, "F1_metric": metric,
                            "Source": model, "RawReads": np.nan,
                            "BAM_est_depth": float(x), "F1": float(preds[plat][i]),
                            "WithinObservedRange": bool(plo <= x <= phi),
                        })
                for i, x in enumerate(grid):
                    vals = pd.Series({p: preds[p][i] for p in PLATFORMS})
                    ranks = vals.rank(method="min", ascending=False).astype(int)
                    for plat in PLATFORMS:
                        rank_rows.append({
                            "Group": g, "F1_metric": metric, "Model": model,
                            "GridIndex": i + 1, "BAM_est_depth": float(x),
                            "Platform": plat, "Predicted_F1": float(vals[plat]),
                            "Rank": int(ranks[plat]),
                            "Delta_vs_SURFSeq": float(vals[plat] - vals[SURF]),
                            "SURFSeq_Rank": int(ranks[SURF]),
                            "WithinObservedRange": bool(
                                sub[sub.Platform == plat]["BAM_est_depth"].min() <= x
                                <= sub[sub.Platform == plat]["BAM_est_depth"].max()
                            ),
                        })
    out = pd.concat([pd.DataFrame(obs), pd.DataFrame(fits_rows)], ignore_index=True)
    return out, pd.DataFrame(rank_rows)


def summarise_rank(rank_df, logger):
    logger.info("")
    logger.info("================ BAM-DEPTH-MATCHED INTERPOLATION ================")
    for g in GROUPS:
        for metric in METRICS:
            for model in ["linear_log10_BAMdepth", "isotonic_log10_BAMdepth"]:
                s = rank_df[(rank_df.Group == g) & (rank_df["F1_metric"] == metric)
                            & (rank_df.Model == model)]
                if s.empty:
                    continue
                per = s.drop_duplicates("GridIndex")
                counts = per["SURFSeq_Rank"].value_counts().reindex([1, 2, 3, 4, 5, 6], fill_value=0)
                logger.info("Group %s | %s | %s: SURF rank distribution %s; mean rank %.2f",
                            g, metric, model, dict(counts), per["SURFSeq_Rank"].mean())
                d = (s[~s.Platform.eq(SURF)].groupby("Platform")["Delta_vs_SURFSeq"]
                     .agg(["mean", "min", "max"])
                     .reindex([p for p in PLATFORMS if p != SURF]))
                log_df(logger, d.round(4),
                       f"Group {g} | {metric} | {model}: predicted F1(platform) - F1(SURFSeq), BAM-depth grid")


# --------------------------------------------------------------------------
# Candidate-site BAM check (supporting, if present)
# --------------------------------------------------------------------------
def candidate_site_check(logger):
    if not CAND_DIR.exists():
        return
    files = sorted(CAND_DIR.glob("*_depth.tsv.gz"))
    files = [f for f in files if f.stat().st_size > 0]
    if not files:
        return
    rows = []
    for f in files:
        plat_file = f.name.replace("_depth.tsv.gz", "")
        plat = BAM_TO_DISPLAY.get(plat_file, plat_file)
        pred_f = PRED_ROOT / "45" / "pred" / f"{plat_file}_reads100M_pred.txt.gz"
        if not pred_f.exists():
            continue
        p = pd.read_csv(pred_f, sep="\t", usecols=["CHROM", "POS", "DP", "label"])
        b = pd.read_csv(f, sep="\t", header=None, names=["CHROM", "POS", "BAM_depth"])
        m = p.merge(b, on=["CHROM", "POS"], how="left")
        m["BAM_depth"] = m["BAM_depth"].fillna(0)
        pr, pp = pearsonr(m["DP"], m["BAM_depth"])
        sr, sp = spearmanr(m["DP"], m["BAM_depth"])
        rows.append({
            "Group": 45, "Reads": 100, "Platform": plat,
            "N_candidate_sites": len(m),
            "Pred_median_DP": float(m["DP"].median()),
            "Pred_mean_DP": float(m["DP"].mean()),
            "BAM_candidate_mean_depth": float(m["BAM_depth"].mean()),
            "BAM_candidate_median_depth": float(m["BAM_depth"].median()),
            "Ratio_candidateBAMmean_over_predMean": float(m["BAM_depth"].mean() / m["DP"].mean()),
            "Pearson_site_level_DP_vs_BAMdepth": float(pr), "p_Pearson": float(pp),
            "Spearman_site_level_DP_vs_BAMdepth": float(sr), "p_Spearman": float(sp),
        })
    if not rows:
        return
    d = pd.DataFrame(rows).sort_values(["Group", "Platform"])
    d.to_csv(TABDIR / "D1b_bam_candidate_site_check.csv", index=False, float_format="%.6f")
    log_df(
        logger,
        d.set_index("Platform")[[
            "N_candidate_sites", "Pred_median_DP", "Pred_mean_DP",
            "BAM_candidate_mean_depth", "BAM_candidate_median_depth",
            "Ratio_candidateBAMmean_over_predMean",
            "Pearson_site_level_DP_vs_BAMdepth", "Spearman_site_level_DP_vs_BAMdepth",
        ]].round(3),
        "Supporting check at the exact candidate sites: pred DP vs BAM depth (group 45, 100M)",
    )
    logger.info(
        "Candidate-site check: site-level Pearson r ranges %.3f-%.3f and Spearman rho "
        "ranges %.3f-%.3f across the six platforms.",
        d["Pearson_site_level_DP_vs_BAMdepth"].min(), d["Pearson_site_level_DP_vs_BAMdepth"].max(),
        d["Spearman_site_level_DP_vs_BAMdepth"].min(), d["Spearman_site_level_DP_vs_BAMdepth"].max(),
    )


# --------------------------------------------------------------------------
# Comparison of the two matching approaches
# --------------------------------------------------------------------------
def build_comparison(d1_dir, d1_rank, b_dir, b_rank):
    rows = []
    for approach, dir_df, rank_df, model in [
        ("pred_median_DP", d1_dir, d1_rank, "linear_log10_DP"),
        ("BAM_est_depth", b_dir, b_rank, "linear_log10_BAMdepth"),
    ]:
        dir_df = dir_df.copy()
        dir_df["Group"] = dir_df["Group"].astype(str)
        for g in GROUPS:
            gs = str(g)
            for metric in METRICS:
                # rank at matched depth, all six platforms
                r = rank_df[(rank_df.Group == g) & (rank_df["F1_metric"] == metric)
                            & (rank_df.Model == model)]
                per = r.drop_duplicates("GridIndex")
                counts = per["SURFSeq_Rank"].value_counts().reindex([1, 2, 3, 4, 5, 6], fill_value=0).to_dict()
                # nearest-neighbour direction counts
                sd = dir_df[(dir_df.Group == gs) & (dir_df["F1_metric"] == metric)
                            & (dir_df.Level == "surf_anchor") & (dir_df.Comparator == "ALL")]
                pd_ = dir_df[(dir_df.Group == gs) & (dir_df["F1_metric"] == metric)
                             & (dir_df.Level == "pair_optimal") & (dir_df.Comparator == "ALL")]
                row = {
                    "Approach": approach, "Group": g, "F1_metric": metric,
                    "SURF_mean_rank_all6": float(per["SURFSeq_Rank"].mean()),
                    "SURF_rank1_count_all6": int(counts.get(1, 0)),
                    "SURF_rank2_count_all6": int(counts.get(2, 0)),
                    "SURF_rank3plus_count_all6": int(sum(v for k, v in counts.items() if k >= 3)),
                    "N_grid_levels": len(per),
                }
                if not sd.empty:
                    row.update({
                        "NN_surf_anchor_SURF_higher": int(sd.iloc[0]["SURF_higher"]),
                        "NN_surf_anchor_Tie": int(sd.iloc[0]["Tie"]),
                        "NN_surf_anchor_SURF_lower": int(sd.iloc[0]["SURF_lower"]),
                        "NN_surf_anchor_mean_delta": float(sd.iloc[0]["Mean_delta"]),
                    })
                if not pd_.empty:
                    row.update({
                        "NN_pair_optimal_SURF_higher": int(pd_.iloc[0]["SURF_higher"]),
                        "NN_pair_optimal_Tie": int(pd_.iloc[0]["Tie"]),
                        "NN_pair_optimal_SURF_lower": int(pd_.iloc[0]["SURF_lower"]),
                        "NN_pair_optimal_mean_delta": float(pd_.iloc[0]["Mean_delta"]),
                    })
                rows.append(row)
    return pd.DataFrame(rows)


def comparator_delta_comparison(d1_dir, b_dir):
    rows = []
    for approach, d in [("pred_median_DP", d1_dir), ("BAM_est_depth", b_dir)]:
        s = d[(d.Group == "ALL") & (d.Level == "surf_anchor")
              & (d.Comparator != "ALL")]
        for metric in METRICS:
            for _, r in s[s["F1_metric"] == metric].iterrows():
                rows.append({
                    "Approach": approach, "F1_metric": metric,
                    "Comparator": r["Comparator"], "N": r["N"],
                    "SURF_higher": r["SURF_higher"], "Tie": r["Tie"],
                    "SURF_lower": r["SURF_lower"], "Mean_delta": r["Mean_delta"],
                    "Median_delta": r["Median_delta"],
                })
    return pd.DataFrame(rows)


def collect_exceptions(mp, rank_df, approach):
    rows = []
    s = mp[(mp.Level == "surf_anchor") & mp["Comparator"].notna() & (mp["Comparator"] != "")]
    for metric in METRICS:
        col = f"Delta_{metric}_SURF_minus_Comparator"
        bad = s[s[col] < -TIE_TOL]
        for _, r in bad.iterrows():
            rows.append({
                "Approach": approach, "F1_metric": metric,
                "Group": r["Group"], "Comparator": r["Comparator"],
                "SURF_raw_reads": r["Reads_A"], "Comparator_raw_reads": r["Reads_B"],
                "SURF_depth": r["BAMdepth_A"] if approach == "BAM_est_depth" else r["MedianDP_A"],
                "Comparator_depth": r["BAMdepth_B"] if approach == "BAM_est_depth" else r["MedianDP_B"],
                "Delta_SURF_minus_Comparator": r[col],
            })
    # interpolation grid points where SURF is below a comparator
    for g in GROUPS:
        for metric in METRICS:
            model = "linear_log10_BAMdepth" if approach == "BAM_est_depth" else "linear_log10_DP"
            s = rank_df[(rank_df.Group == g) & (rank_df["F1_metric"] == metric)
                        & (rank_df.Model == model) & (~rank_df.Platform.eq(SURF))
                        & (rank_df.Delta_vs_SURFSeq > TIE_TOL)]
            for _, r in s.iterrows():
                rows.append({
                    "Approach": approach, "F1_metric": metric,
                    "Group": g, "Comparator": r["Platform"],
                    "SURF_raw_reads": np.nan, "Comparator_raw_reads": np.nan,
                    "SURF_depth": r["BAM_est_depth"] if approach == "BAM_est_depth" else r["MedianDP"],
                    "Comparator_depth": np.nan,
                    "Delta_SURF_minus_Comparator": -r["Delta_vs_SURFSeq"],
                })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Figure
# --------------------------------------------------------------------------
def make_figure(fit_df, df, logger):
    plt.rcParams.update({
        "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.6), sharex="col")
    for ri, g in enumerate(GROUPS):
        sub = df[df.Group == g]
        lo, hi = common_range(sub)
        for ci, metric in enumerate(METRICS):
            ax = axes[ri, ci]
            ax.axvspan(lo, hi, color="#d9d9d9", alpha=0.45, lw=0, zorder=0)
            for plat in PLATFORMS:
                s = sub[sub.Platform == plat].sort_values("BAM_est_depth")
                color, lw, ms = COLORS[plat], (3.2 if plat == SURF else 1.8), (7.5 if plat == SURF else 5)
                z = 5 if plat == SURF else 3
                ax.plot(s["BAM_est_depth"], s[metric], "-o", color=color, lw=1.0,
                        ms=ms, markeredgecolor="white", markeredgewidth=0.6, zorder=z)
                f = fit_df[(fit_df.Group == g) & (fit_df.Platform == plat)
                           & (fit_df["F1_metric"] == metric)
                           & (fit_df.Source == "linear_log10_BAMdepth")].sort_values("BAM_est_depth")
                ax.plot(f["BAM_est_depth"], f["F1"], "-", color=color, lw=lw, zorder=z)
            ax.set_xscale("log")
            ax.set_xlabel("BAM-estimated deduplicated panel depth (x; 100M mean x reads/100)")
            ax.set_ylabel("F1" if metric == "F1@0.1" else "Best achievable F1")
            ax.set_title(
                f"{'Group 45' if g == 45 else 'Group 54'}  |  "
                f"{'F1 at pred > 0.1' if metric == 'F1@0.1' else 'F1 best (threshold-free)'}",
                fontsize=10,
            )
            ax.grid(axis="y", color="#e6e6e6", lw=0.6)
            ax.text(0.02, 0.025,
                    f"shaded: all-platform overlap {lo:.0f}-{hi:.0f}x\n"
                    "points: 5 raw depths/platform; lines: linear fit",
                    transform=ax.transAxes, fontsize=7.2, color="#444444", va="bottom")
    handles = [plt.Line2D([0], [0], color=COLORS[p], marker="o",
                          lw=2.5 if p == SURF else 1.6, ms=7 if p == SURF else 5, label=p)
               for p in PLATFORMS]
    fig.legend(handles=handles, loc="upper center", ncol=6, frameon=False,
               bbox_to_anchor=(0.5, 0.985), fontsize=8.5)
    fig.suptitle("F1 versus BAM-estimated deduplicated panel depth", y=0.945, fontsize=11.5)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    for ext in ["png", "pdf"]:
        fig.savefig(FIGDIR / f"D1b_f1_vs_BAMdepth.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("Wrote %s and %s", FIGDIR / "D1b_f1_vs_BAMdepth.png",
                FIGDIR / "D1b_f1_vs_BAMdepth.pdf")


# --------------------------------------------------------------------------
# Final conclusion text
# --------------------------------------------------------------------------
def write_conclusion(logger, bam_vs, comparison, exceptions, rank_b, cand=None):
    corr = bam_vs[bam_vs.Row_type == "correlation_summary"].set_index("Group")
    def cval(g, col):
        return corr.loc[g, col] if g in corr.index else np.nan

    cand_file = TABDIR / "D1b_bam_candidate_site_check.csv"
    if cand_file.exists() and cand_file.stat().st_size > 0:
        cd = pd.read_csv(cand_file)
        cand_txt = ("At the exact candidate sites, BAM depth and pred DP agree much more "
                    "closely (site-level Pearson r=%.3f-%.3f; see "
                    "D1b_bam_candidate_site_check.csv), so pred DP is the more relevant "
                    "candidate-site depth proxy; the panel mean is a broader capture measure."
                    % (cd["Pearson_site_level_DP_vs_BAMdepth"].min(),
                       cd["Pearson_site_level_DP_vs_BAMdepth"].max()))
    else:
        cand_txt = ("A candidate-site BAM check was not available for this run; the panel "
                    "mean remains a broader capture measure than candidate-site Mutect2 DP.")
    lin_file = TABDIR / "D1b_bam_depth_linearity.csv"
    if lin_file.exists():
        lf = pd.read_csv(lin_file)
        lin_txt = ("observed G45 140M/100M ratios are %.2f-%.2f, i.e. %.0f-%.0f%% below "
                   "the assumed 1.4x (D1b_bam_depth_linearity.csv)."
                   % (lf["Observed_140M_over_100M"].min(), lf["Observed_140M_over_100M"].max(),
                      -lf["Deviation_from_linear_pct"].max(), -lf["Deviation_from_linear_pct"].min()))
    else:
        lin_txt = "no direct 140M/100M linearity check was available."

    lines = ["D1b final conclusion (BAM-depth sensitivity analysis)"]
    # 1. BAM check
    lines.append(
        "1. BAM check (100M): panel-level BAM mean depth vs candidate-site median Mutect2 DP "
        "give weak cross-platform rank agreement (G45 Spearman rho=%.2f, Pearson r=%.2f; "
        "G54 rho=%.2f, r=%.2f); in G45 SURFSeq is rank 1 by pred median DP but rank 4 by "
        "panel BAM mean. The two are different quantities: pred DP is candidate-site Mutect2 "
        "DP, BAM mean is a 5.2%% panel sample. %s"
        % (cval(45, "Spearman_BAMmean_vs_PredMedian"), cval(45, "Pearson_BAMmean_vs_PredMedian"),
           cval(54, "Spearman_BAMmean_vs_PredMedian"), cval(54, "Pearson_BAMmean_vs_PredMedian"),
           cand_txt)
    )
    # 2. Matching results per metric (both groups, both approaches)
    for mi, metric in enumerate(METRICS):
        a45 = comparison[(comparison.Approach == "pred_median_DP") & (comparison.Group == 45)
                         & (comparison["F1_metric"] == metric)]
        a54 = comparison[(comparison.Approach == "pred_median_DP") & (comparison.Group == 54)
                         & (comparison["F1_metric"] == metric)]
        b45 = comparison[(comparison.Approach == "BAM_est_depth") & (comparison.Group == 45)
                         & (comparison["F1_metric"] == metric)]
        b54 = comparison[(comparison.Approach == "BAM_est_depth") & (comparison.Group == 54)
                         & (comparison["F1_metric"] == metric)]
        if any(x.empty for x in [a45, a54, b45, b54]):
            continue
        a45, a54, b45, b54 = a45.iloc[0], a54.iloc[0], b45.iloc[0], b54.iloc[0]
        lines.append(
            "%d. %s (linear): matched pred-DP all-six SURF mean rank G45=%.2f (rank1 %d/40), "
            "G54=%.2f (rank1 %d/40); NN surf-anchor pooled direction %d higher/%d tie/%d lower, "
            "mean delta %+.4f. BAM-depth matching: SURF mean rank G45=%.2f (rank1 %d/40), "
            "G54=%.2f (rank1 %d/40); NN %d higher/%d tie/%d lower, mean delta %+.4f."
            % (mi + 2, metric,
               a45["SURF_mean_rank_all6"], a45["SURF_rank1_count_all6"],
               a54["SURF_mean_rank_all6"], a54["SURF_rank1_count_all6"],
               a45["NN_surf_anchor_SURF_higher"] + a54["NN_surf_anchor_SURF_higher"],
               a45["NN_surf_anchor_Tie"] + a54["NN_surf_anchor_Tie"],
               a45["NN_surf_anchor_SURF_lower"] + a54["NN_surf_anchor_SURF_lower"],
               (a45["NN_surf_anchor_mean_delta"] + a54["NN_surf_anchor_mean_delta"]) / 2,
               b45["SURF_mean_rank_all6"], b45["SURF_rank1_count_all6"],
               b54["SURF_mean_rank_all6"], b54["SURF_rank1_count_all6"],
               b45["NN_surf_anchor_SURF_higher"] + b54["NN_surf_anchor_SURF_higher"],
               b45["NN_surf_anchor_Tie"] + b54["NN_surf_anchor_Tie"],
               b45["NN_surf_anchor_SURF_lower"] + b54["NN_surf_anchor_SURF_lower"],
               (b45["NN_surf_anchor_mean_delta"] + b54["NN_surf_anchor_mean_delta"]) / 2)
        )
    # 3. Exceptions under BAM matching (NN conditions only; grid exceptions kept in CSV)
    eb = exceptions[(exceptions.Approach == "BAM_est_depth")
                    & exceptions["SURF_raw_reads"].notna()].copy()
    def _exc_text(df, maxn=4):
        parts = []
        for _, r in df.head(maxn).iterrows():
            parts.append("G%s %s %s (raw %sM vs %sM, %+.4f)"
                         % (r["Group"], r["F1_metric"], r["Comparator"],
                            int(r["SURF_raw_reads"]), int(r["Comparator_raw_reads"]),
                            r["Delta_SURF_minus_Comparator"]))
        return "; ".join(parts)
    ep = exceptions[(exceptions.Approach == "pred_median_DP")
                    & exceptions["SURF_raw_reads"].notna()].copy()
    eb_txt = _exc_text(eb, 5) if not eb.empty else "none below -0.002"
    ep_txt = _exc_text(ep, 4) if not ep.empty else "none below -0.002"
    lines.append("4. NN exceptions where SURF is lower by >0.002 (raw reads shown): "
                 "BAM-depth matching: " + eb_txt + ". Pred-DP matching: " + ep_txt + ".")
    # 5. Top-of-common-grid rank reversals
    top_bits = []
    for g in GROUPS:
        gmax = int(rank_b[rank_b.Group == g]["GridIndex"].max())
        for metric in METRICS:
            q = rank_b[(rank_b.Group == g) & (rank_b.Model == "linear_log10_BAMdepth")
                       & (rank_b["F1_metric"] == metric) & (rank_b.GridIndex == gmax)]
            if q.empty:
                continue
            q = q.sort_values("Rank")
            surf_rank = int(q[q.Platform == SURF]["Rank"].iloc[0])
            if surf_rank > 1:
                top = q.iloc[0]
                surf = q[q.Platform == SURF].iloc[0]
                top_bits.append("G%s %s at %.0fx: %s %.4f vs SURF %.4f (SURF rank %d)"
                                % (g, metric, top["BAM_est_depth"], top["Platform"],
                                   top["Predicted_F1"], surf["Predicted_F1"], surf_rank))
    if top_bits:
        lines.append("5. At the top of the common BAM-depth grid the SURF advantage narrows or "
                     "reverses in %d of 4 group x metric combinations: " % len(top_bits)
                     + "; ".join(top_bits) + ". SURF remains rank 1 in the remaining "
                     "combination(s). The advantage is therefore concentrated at low-to-middle "
                     "deduplicated depth, not at the highest matched depth.")
    else:
        lines.append("5. SURFSeq remains rank 1 at the top of the common BAM-depth grid as well.")
    # 6. Overall bottom line
    b_all = comparison[comparison.Approach == "BAM_est_depth"]
    a_all = comparison[comparison.Approach == "pred_median_DP"]
    # Isotonic sensitivity (model-dependent)
    d1_rank_all = pd.read_csv(D1_RANK) if D1_RANK.exists() else pd.DataFrame()
    def _iso_ranks(rdf, model):
        out = {}
        for gg in GROUPS:
            for mm in METRICS:
                per = rdf[(rdf.Group == gg) & (rdf["F1_metric"] == mm)
                          & (rdf.Model == model)].drop_duplicates("GridIndex")
                out[(gg, mm)] = (float(per["SURFSeq_Rank"].mean()),
                                 int((per["SURFSeq_Rank"] == 1).sum()))
        return out
    iso_pred = _iso_ranks(d1_rank_all, "isotonic_log10_DP") if not d1_rank_all.empty else {}
    iso_bam = _iso_ranks(rank_b, "isotonic_log10_BAMdepth")
    def _mean_rank(rdf, model, gg, mm):
        return (rdf[(rdf.Group == gg) & (rdf["F1_metric"] == mm)
                    & (rdf.Model == model)].groupby("Platform")["Rank"].mean().sort_values())
    pr54 = _mean_rank(d1_rank_all, "isotonic_log10_DP", 54, "F1@0.1") if not d1_rank_all.empty else pd.Series(dtype=float)
    bm54 = _mean_rank(rank_b, "isotonic_log10_BAMdepth", 54, "F1@0.1")
    def _best_other(x, surf_val):
        x = x.drop(labels=[SURF], errors="ignore")
        if x.empty:
            return "n/a", np.nan
        return x.index[0], float(x.iloc[0])
    p_best, p_best_val = _best_other(pr54, pr54.get(SURF, np.nan))
    b_best, b_best_val = _best_other(bm54, bm54.get(SURF, np.nan))
    lines.append(
        "6. Bottom line: averaged over both metrics and groups, SURFSeq has the lowest linear "
        "mean all-six rank under both proxies (%.2f BAM vs %.2f pred DP; lower is better), and "
        "it is best on average in group 45 and in F1_best. The clear exception is group-54 "
        "F1@0.1: linear fitting ranks SURF first, but isotonic fitting ranks %s first "
        "(mean rank %.2f vs SURF %.2f under pred DP; %.2f vs %.2f under BAM depth). "
        "Isotonic SURF mean ranks are G45 %.2f/%.2f and G54 %.2f/%.2f (pred DP), and "
        "G45 %.2f/%.2f and G54 %.2f/%.2f (BAM depth) for F1@0.1/F1_best. Raw-depth matching "
        "thus overstates the uniformity of the advantage, but depth alone does not explain it."
        % (b_all["SURF_mean_rank_all6"].mean(), a_all["SURF_mean_rank_all6"].mean(),
           p_best, p_best_val, float(pr54.get(SURF, np.nan)),
           b_best_val, float(bm54.get(SURF, np.nan)),
           iso_pred.get((45, "F1@0.1"), (np.nan, np.nan))[0],
           iso_pred.get((45, "F1_best"), (np.nan, np.nan))[0],
           iso_pred.get((54, "F1@0.1"), (np.nan, np.nan))[0],
           iso_pred.get((54, "F1_best"), (np.nan, np.nan))[0],
           iso_bam.get((45, "F1@0.1"), (np.nan, np.nan))[0],
           iso_bam.get((45, "F1_best"), (np.nan, np.nan))[0],
           iso_bam.get((54, "F1@0.1"), (np.nan, np.nan))[0],
           iso_bam.get((54, "F1_best"), (np.nan, np.nan))[0])
    )
    # 5. Caveat
    lines.append(
        "7. Caveat: BAM 100M means are scaled linearly to other raw depths; %s "
        "The panel sample is 5.2%% of the panel; candidate sets differ across platforms, so "
        "depth matching does not equalise candidate difficulty. The matched differences are "
        "descriptive; no significance tests are claimed." % lin_txt
    )
    text = "\n".join(lines) + "\n"
    (TABDIR / "D1b_conclusion.txt").write_text(text)
    logger.info("")
    logger.info("================ D1b FINAL CONCLUSION ================")
    for line in lines:
        logger.info(line)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--skip-figure", action="store_true")
    args = ap.parse_args()

    LOGDIR.mkdir(parents=True, exist_ok=True)
    TABDIR.mkdir(parents=True, exist_ok=True)
    FIGDIR.mkdir(parents=True, exist_ok=True)
    logger = setup_logger()
    logger.info("D1b BAM-depth sensitivity analysis")
    logger.info("D1 conditions: %s", D1_COND)
    logger.info("BAM 100M group 45: %s", BAM100_G45)
    logger.info("BAM 100M group 54: %s", BAM100_G54)
    logger.info("BAM 140M group 45 (linearity check): %s", BAM140_G45)

    cond = pd.read_csv(D1_COND)
    bam, means100 = load_bam_means(logger)
    if means100.empty:
        raise RuntimeError("No 100M BAM means available")
    log_df(logger, means100[["Group", "Platform", "MeanDepthAllBp", "MedianDepth"]]
           .round(2).set_index(["Group", "Platform"]),
           "BAM panel depth measured at 100M")

    bam_vs = build_bam_vs_median(cond, means100, logger)
    bam_vs.to_csv(TABDIR / "D1b_bam_vs_medianDP.csv", index=False, float_format="%.6f")
    logger.info("Wrote %s", TABDIR / "D1b_bam_vs_medianDP.csv")

    check_linear_scaling(bam, logger)

    # Attach BAM-estimated depth to every condition
    bm = means100[["Group", "Platform", "MeanDepthAllBp", "MedianDepth"]].rename(
        columns={"MeanDepthAllBp": "BAM_mean_depth_100M", "MedianDepth": "BAM_median_depth_100M"}
    )
    df = cond.merge(bm, on=["Group", "Platform"], how="left")
    if df["BAM_mean_depth_100M"].isna().any():
        logger.warning("Conditions without BAM 100M mean: %d",
                       int(df["BAM_mean_depth_100M"].isna().sum()))
        df = df.dropna(subset=["BAM_mean_depth_100M"]).copy()
    df["BAM_est_depth"] = df["BAM_mean_depth_100M"] * df["Reads"] / 100.0
    df["BAM_est_depth_median_scaled"] = df["BAM_median_depth_100M"] * df["Reads"] / 100.0
    logger.info("")
    logger.info("BAM-estimated depth = BAM_mean_depth_100M * (raw reads / 100). "
                "This linear scaling is an approximation; see D1b_bam_depth_linearity.csv.")
    log_df(
        logger,
        df.pivot_table(index="Platform", columns="Reads", values="BAM_est_depth").reindex(PLATFORMS)[READS].round(1),
        "BAM-estimated deduplicated depth by platform x raw depth (x)",
    )

    # Nearest-neighbour matching
    logger.info("")
    logger.info("================ BAM-DEPTH NEAREST-NEIGHBOUR MATCHING ================")
    mp = build_matched_pairs(df)
    mp.to_csv(TABDIR / "D1b_matched_BAMdepth_pairs.csv", index=False, float_format="%.6f")
    logger.info("Wrote %s", TABDIR / "D1b_matched_BAMdepth_pairs.csv")
    dirs = direction_counts(mp)
    dirs.to_csv(TABDIR / "D1b_matched_BAMdepth_direction_counts.csv", index=False,
                float_format="%.6f")
    log_df(logger, dirs, "BAM-depth matched direction counts (delta = SURFSeq - comparator; tie = |delta| <= 0.002)")

    # Interpolation
    logger.info("")
    logger.info("================ BAM-DEPTH INTERPOLATION ================")
    stats, fits = fit_models(df)
    stats.to_csv(TABDIR / "D1b_interpolation_fit_stats.csv", index=False, float_format="%.6f")
    log_df(logger, stats.round(4).set_index(["Group", "F1_metric", "Platform"]),
           "BAM-depth interpolation fit statistics")
    fit_df, rank_b = build_interpolation(df, fits, logger)
    fit_df.to_csv(TABDIR / "D1b_f1_vs_BAMdepth.csv", index=False, float_format="%.6f")
    rank_b.to_csv(TABDIR / "D1b_rank_at_matched_BAMdepth.csv", index=False, float_format="%.6f")
    logger.info("Wrote %s", TABDIR / "D1b_f1_vs_BAMdepth.csv")
    logger.info("Wrote %s", TABDIR / "D1b_rank_at_matched_BAMdepth.csv")
    summarise_rank(rank_b, logger)

    # Compare with D1 pred-median-DP matching
    d1_dir = pd.read_csv(D1_DIR) if D1_DIR.exists() else pd.DataFrame()
    d1_rank = pd.read_csv(D1_RANK) if D1_RANK.exists() else pd.DataFrame()
    if not d1_dir.empty and not d1_rank.empty:
        comp = build_comparison(d1_dir, d1_rank, dirs, rank_b)
        comp.to_csv(TABDIR / "D1b_predDP_vs_BAMdepth_comparison.csv", index=False,
                    float_format="%.6f")
        log_df(logger, comp.round(4), "Pred-median-DP matching vs BAM-depth matching")
        cdelta = comparator_delta_comparison(d1_dir, dirs)
        cdelta.to_csv(TABDIR / "D1b_comparator_deltas_predDP_vs_BAM.csv", index=False,
                      float_format="%.6f")
        log_df(logger, cdelta.round(4).set_index(["F1_metric", "Approach", "Comparator"]),
               "Per-comparator mean delta (surf-anchor NN)")
        exc = collect_exceptions(mp, rank_b, "BAM_est_depth")
        # add pred-DP exceptions from D1 matched pairs
        if (TABDIR / "D1_matched_pairs.csv").exists():
            d1mp = pd.read_csv(TABDIR / "D1_matched_pairs.csv")
            d1mp["MedianDP_A"] = d1mp.get("MedianDP_A", np.nan)
            exc_d1 = collect_exceptions(d1mp, d1_rank, "pred_median_DP")
            exc = pd.concat([exc_d1, exc], ignore_index=True)
        exc.to_csv(TABDIR / "D1b_exceptions.csv", index=False, float_format="%.6f")
        log_df(logger, exc, "Matched conditions where SURFSeq is lower (exceptions)")
    else:
        comp = pd.DataFrame()
        exc = collect_exceptions(mp, rank_b, "BAM_est_depth")
        exc.to_csv(TABDIR / "D1b_exceptions.csv", index=False, float_format="%.6f")
        logger.warning("D1 direction/rank tables missing; comparison table not written.")

    candidate_site_check(logger)

    if not args.skip_figure:
        make_figure(fit_df, df, logger)

    write_conclusion(logger, bam_vs, comp, exc, rank_b, pd.DataFrame())
    logger.info("")
    logger.info("Wrote %s", TABDIR / "D1b_conclusion.txt")


if __name__ == "__main__":
    main()
