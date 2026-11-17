#!/usr/bin/env python3
"""R27: paired/hierarchical analyses, confidence intervals and audit for
Reviewer 2 comment 7 and Reviewer 3 comment 5.

Outputs are written only under response/addendum-analyses/:
  tables/R27_platform_mixed_models.csv
  tables/R27_method_mixed_models.csv
  tables/R27_paired_bootstrap.csv
  tables/R27_cluster_bootstrap_sensitivity.csv
  tables/R27_twist_ci.csv
  tables/R27_error_f1_relation.csv
  tables/R27_error_f1_points.csv
  tables/R27_supp_table_draft.csv
  tables/R27_conclusion.txt
  figures/R27_forest_platform.png/.pdf
  figures/R27_forest_method.png/.pdf
  logs/R27.log

Design notes
------------
* GIAB platform comparison: SomaticXGB fixed.  condition = donor direction x depth
  (10 blocks); the six platforms are measured within every condition.  A random
  intercept for condition is therefore a paired/blocked analysis.
* GIAB method comparison: method varies within condition = donor x platform x
  depth (60 blocks).  A random intercept for condition is a paired analysis.
* Cluster bootstrap resamples whole conditions, preserving the correlation among
  methods/platforms measured in the same condition.  Extra hierarchical bootstraps
  resample platforms, depths, and both, as a sensitivity analysis.
* Twist: three independent library replicates.  Paired t (df = 2) is the primary
  conservative interval; MixedLM/block OLS are reported as requested but their CIs
  are narrower and can be anti-conservative with three replicate groups.
* All confidence intervals are two-sided 95%.  MixedLM intervals use a t reference
  distribution with a conservative residual df (n_obs - n_groups - n_fixed); the
  corresponding z p-value is also retained.  This is not a Satterthwaite/KR
  approximation and is labelled accordingly.
* No existing file is modified.
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
import common  # noqa: E402

import itertools
import json
import logging
import math
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.stats.multitest import multipletests

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------------
BASE = common.REPO_ROOT
TABLES = common.TABLES
FIGS = common.FIGURES
LOGS = common.LOGS
for d in (TABLES, FIGS, LOGS):
    d.mkdir(parents=True, exist_ok=True)

B1C = TABLES / "B1c_all_depths_by_condition.csv"
A3 = TABLES / "A3_multithreshold_by_platform.csv"
TWIST_PHYS = TABLES / "supp_table_S9a_original_twist.csv"
TWIST_HYB = TABLES / "supp_table_S9b_hybrid_twist.csv"
OLD_XLSX = common.REPO_ROOT / "reference" / "supplementary" / "supplementary_tables_revision.xlsx"  # optional archive; absence is tolerated
NEW_XLSX = common.REPO_ROOT / "reference" / "supplementary" / "sup_tables-submit-revised-261112.xlsx"

SEED = 20261113
N_BOOT = 20000
ALPHA = 0.05

PLATFORMS = [
    "SURFSeq 5000",
    "FASTASeq 300",
    "GenoLab M",
    "NovaSeq 6000",
    "NovaSeq X",
    "DNBSEQ-T7",
]
TRUSEQ = [p for p in PLATFORMS if p != "DNBSEQ-T7"]

RULE_LABEL = {
    "strelka_default": "Strelka2 (default PASS)",
    "mutect2_tlod8": "Mutect2 (TLOD>=8)",
    "mutect2_tlod10": "Mutect2 (TLOD>=10)",
    "union_tlod8": "Strelka2+Mutect2 union (TLOD>=8)",
    "union_tlod10": "Strelka2+Mutect2 union (TLOD>=10)",
    "varscan_ssc3": "VarScan2 (SSC>=3, fair-tuned)",
    "xgb_pred01": "SomaticXGB (0.1)",
    "xgb_pred06": "SomaticXGB (0.6)",
}

PRIMARY_RULES = [
    "xgb_pred01",
    "strelka_default",
    "mutect2_tlod8",
    "union_tlod8",
    "varscan_ssc3",
    "xgb_pred06",
]
SENS_RULES = ["mutect2_tlod10", "union_tlod10"]
ALL_RULES = PRIMARY_RULES + SENS_RULES

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_PATH = LOGS / "R27.log"
logger = logging.getLogger("R27")
logger.setLevel(logging.INFO)
logger.handlers.clear()
fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", "%Y-%m-%d %H:%M:%S")
fh = logging.FileHandler(LOG_PATH, mode="w")
fh.setFormatter(fmt)
sh = logging.StreamHandler(sys.stdout)
sh.setFormatter(fmt)
logger.addHandler(fh)
logger.addHandler(sh)


def log_section(title: str) -> None:
    logger.info("")
    logger.info("=" * 78)
    logger.info(title)
    logger.info("=" * 78)


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------
def bh_fdr(pvals) -> np.ndarray:
    p = np.asarray(pvals, dtype=float)
    out = np.full(p.shape, np.nan, dtype=float)
    mask = np.isfinite(p)
    if mask.any():
        out[mask] = multipletests(p[mask], method="fdr_bh")[1]
    return out


def t_ci(est: float, se: float, df: float) -> tuple[float, float]:
    if not np.isfinite(est) or not np.isfinite(se) or df <= 0:
        return (np.nan, np.nan)
    crit = stats.t.ppf(0.975, df)
    return (est - crit * se, est + crit * se)


def fmt_p(p: float) -> str:
    if not np.isfinite(p):
        return "NA"
    if p < 1e-4:
        return f"{p:.2e}"
    return f"{p:.4f}"


def safe_pearson(x, y) -> float:
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return np.nan
    return float(stats.pearsonr(x, y)[0])


def safe_spearman(x, y) -> float:
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return np.nan
    return float(stats.spearmanr(x, y)[0])


def fit_mixed_contrast(
    data: pd.DataFrame,
    outcome: str,
    factor: str,
    reference: str,
    group_col: str,
    levels: list[str] | None = None,
) -> tuple[pd.DataFrame | None, dict]:
    """Fit MixedLM(outcome ~ factor, groups=group_col) with a reference level.

    Returns a tidy table.  For the reference row, estimate is the model mean and
    estimate_type is 'mean'.  For every other level, estimate is the difference
    from the reference (level - reference) and estimate_type is 'difference'.
    """
    if levels is None:
        levels = [reference] + [x for x in sorted(data[factor].dropna().unique()) if x != reference]
    levels = [reference] + [x for x in levels if x != reference]
    missing = [x for x in levels if x not in set(data[factor].unique())]
    if missing:
        raise ValueError(f"levels missing from data: {missing}")

    X = pd.DataFrame({"Intercept": np.ones(len(data), dtype=float)}, index=data.index)
    for lev in levels[1:]:
        X[f"{factor}[{lev}]"] = (data[factor] == lev).astype(float).to_numpy()

    y = data[outcome].astype(float)
    groups = data[group_col].astype(str)
    n_obs = len(data)
    n_groups = int(groups.nunique())
    n_fixed = int(X.shape[1])
    df = max(1, n_obs - n_groups - n_fixed)

    info = {
        "n_obs": n_obs,
        "n_groups": n_groups,
        "n_fixed": n_fixed,
        "df": df,
        "converged": False,
        "error": None,
        "warnings": [],
    }
    with warnings.catch_warnings(record=True) as wlist:
        warnings.simplefilter("always")
        try:
            fit = sm.MixedLM(y, X, groups=groups).fit(reml=True)
        except Exception as exc:  # noqa: BLE001 - recorded, not swallowed silently
            info["error"] = f"{type(exc).__name__}: {exc}"
            logger.error("MixedLM failed for %s ~ %s: %s", outcome, factor, info["error"])
            return None, info
        for wi in wlist:
            msg = str(wi.message)
            info["warnings"].append(msg)
            logger.warning("MixedLM warning (%s ~ %s): %s", outcome, factor, msg)

    info["converged"] = bool(getattr(fit, "converged", False))
    group_var = float(fit.cov_re.iloc[0, 0]) if fit.cov_re.size else np.nan
    resid_var = float(fit.scale)
    icc = group_var / (group_var + resid_var) if (group_var + resid_var) > 0 else np.nan
    info.update(group_var=group_var, resid_var=resid_var, icc=icc)

    rows: list[dict] = []
    # Reference row: absolute model mean
    est = float(fit.params["Intercept"])
    se = float(fit.bse["Intercept"])
    lo, hi = t_ci(est, se, df)
    rows.append(
        {
            "term": reference,
            "estimate_type": "mean",
            "estimate": est,
            "SE": se,
            "CI95_low": lo,
            "CI95_high": hi,
            "p": np.nan,
            "p_z": np.nan,
            "df": df,
        }
    )
    for lev in levels[1:]:
        term = f"{factor}[{lev}]"
        est = float(fit.params.get(term, np.nan))
        se = float(fit.bse.get(term, np.nan))
        tval = est / se if np.isfinite(se) and se > 0 else np.nan
        p_t = 2 * stats.t.sf(abs(tval), df) if np.isfinite(tval) else np.nan
        p_z = float(fit.pvalues.get(term, np.nan))
        lo, hi = t_ci(est, se, df)
        rows.append(
            {
                "term": lev,
                "estimate_type": "difference",
                "estimate": est,
                "SE": se,
                "CI95_low": lo,
                "CI95_high": hi,
                "p": p_t,
                "p_z": p_z,
                "df": df,
            }
        )

    out = pd.DataFrame(rows)
    out["p_fdr"] = np.nan
    mask = out["estimate_type"] == "difference"
    out.loc[mask, "p_fdr"] = bh_fdr(out.loc[mask, "p"].to_numpy())
    out["model_mean"] = np.nan
    intercept = float(fit.params["Intercept"])
    for i, row in out.iterrows():
        if row["estimate_type"] == "mean":
            out.loc[i, "model_mean"] = intercept
        else:
            out.loc[i, "model_mean"] = intercept + row["estimate"]
    obs = data.groupby(factor, observed=True)[outcome].agg(["mean", "size"]).reset_index()
    obs_map = dict(zip(obs[factor], obs["mean"]))
    size_map = dict(zip(obs[factor], obs["size"]))
    out["mean_observed"] = out["term"].map(obs_map)
    out["n_level_obs"] = out["term"].map(size_map)
    out["n_conditions"] = n_groups
    out["n_obs"] = n_obs
    out["group_var"] = group_var
    out["resid_var"] = resid_var
    out["icc"] = icc
    out["ci_method"] = f"Wald t, df={df}"
    out["converged"] = info["converged"]
    out["factor"] = factor
    out["reference"] = reference
    return out, info


def fit_block_ols(
    data: pd.DataFrame,
    outcome: str,
    factor: str,
    reference: str,
    block: str,
) -> pd.DataFrame | None:
    """OLS with factor and block fixed effects (paired complete-block analysis)."""
    d = data.copy()
    levels = [reference] + [x for x in sorted(d[factor].dropna().unique()) if x != reference]
    d[factor] = pd.Categorical(d[factor], categories=levels)
    d[block] = pd.Categorical(d[block], categories=sorted(d[block].dropna().unique()))
    try:
        with warnings.catch_warnings(record=True) as wlist:
            warnings.simplefilter("always")
            fit = smf.ols(f"{outcome} ~ C({factor}) + C({block})", data=d).fit()
            for wi in wlist:
                logger.warning("Block OLS warning (%s ~ %s): %s", outcome, factor, wi.message)
    except Exception as exc:  # noqa: BLE001
        logger.error("Block OLS failed for %s ~ %s: %s", outcome, factor, exc)
        return None

    rows = []
    est = float(fit.params["Intercept"])
    se = float(fit.bse["Intercept"])
    lo, hi = t_ci(est, se, fit.df_resid)
    rows.append(
        {
            "term": reference,
            "estimate_type": "mean",
            "estimate": est,
            "SE": se,
            "CI95_low": lo,
            "CI95_high": hi,
            "p": np.nan,
            "df": float(fit.df_resid),
        }
    )
    for lev in levels[1:]:
        term = f"C({factor})[T.{lev}]"
        if term not in fit.params.index:
            logger.warning("Block OLS term missing: %s", term)
            continue
        est = float(fit.params[term])
        se = float(fit.bse[term])
        tval = est / se if se > 0 else np.nan
        pval = 2 * stats.t.sf(abs(tval), fit.df_resid) if np.isfinite(tval) else np.nan
        lo, hi = t_ci(est, se, fit.df_resid)
        rows.append(
            {
                "term": lev,
                "estimate_type": "difference",
                "estimate": est,
                "SE": se,
                "CI95_low": lo,
                "CI95_high": hi,
                "p": pval,
                "df": float(fit.df_resid),
            }
        )
    out = pd.DataFrame(rows)
    out["p_fdr"] = np.nan
    mask = out["estimate_type"] == "difference"
    out.loc[mask, "p_fdr"] = bh_fdr(out.loc[mask, "p"].to_numpy())
    out["n_obs"] = len(data)
    out["resid_var"] = float(fit.mse_resid)
    out["ci_method"] = f"OLS t, df={int(fit.df_resid)}"
    out["reference"] = reference
    return out


def condition_bootstrap(
    diff_matrix: np.ndarray,
    contrasts: list[str],
    rng: np.random.Generator,
    n_boot: int = N_BOOT,
) -> pd.DataFrame:
    """Cluster bootstrap over rows (conditions) of an n x k difference matrix."""
    n = diff_matrix.shape[0]
    idx = rng.integers(0, n, size=(n_boot, n))
    boot = diff_matrix[idx]  # B x n x k
    mean_boot = boot.mean(axis=1)
    median_boot = np.median(boot, axis=1)
    rows = []
    for j, name in enumerate(contrasts):
        d = diff_matrix[:, j]
        n_up = int((d > 0).sum())
        n_down = int((d < 0).sum())
        n_tie = int(n - n_up - n_down)
        if n_up + n_down > 0:
            sign_p = float(stats.binomtest(n_up, n_up + n_down, 0.5).pvalue)
        else:
            sign_p = np.nan
        try:
            w_p = float(stats.wilcoxon(d, zero_method="wilcox", alternative="two-sided").pvalue)
        except ValueError:
            w_p = np.nan
        try:
            t_p = float(stats.ttest_1samp(d, 0.0).pvalue)
        except Exception:  # noqa: BLE001
            t_p = np.nan
        mlow, mhigh = np.percentile(mean_boot[:, j], [2.5, 97.5])
        medlow, medhigh = np.percentile(median_boot[:, j], [2.5, 97.5])
        rows.append(
            {
                "contrast": name,
                "n_units": n,
                "mean_diff": float(np.mean(d)),
                "mean_CI_low": float(mlow),
                "mean_CI_high": float(mhigh),
                "median_diff": float(np.median(d)),
                "median_CI_low": float(medlow),
                "median_CI_high": float(medhigh),
                "n_up": n_up,
                "n_tie": n_tie,
                "n_down": n_down,
                "sign_test_p": sign_p,
                "paired_t_p": t_p,
                "wilcoxon_p": w_p,
                "n_boot": n_boot,
            }
        )
    return pd.DataFrame(rows)


def bootstrap_vector_mean(
    values: np.ndarray,
    rng: np.random.Generator,
    n_boot: int = N_BOOT,
) -> tuple[float, float, float]:
    """Percentile CI for the mean of a vector, resampling its entries."""
    n = len(values)
    idx = rng.integers(0, n, size=(n_boot, n))
    boot = values[idx].mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return float(np.mean(values)), float(lo), float(hi)


def bootstrap_cells_mean(
    cell_values: np.ndarray,
    rng: np.random.Generator,
    n_boot: int = N_BOOT,
) -> tuple[float, float, float]:
    """Two/three-way cluster bootstrap: resample every axis independently.

    cell_values is an array with one axis per crossed factor (e.g. group x depth
    or group x platform x depth).  Each bootstrap replicate samples indices with
    replacement along every axis and averages the selected cells.
    """
    shape = cell_values.shape
    idxs = [rng.integers(0, s, size=(n_boot, s)) for s in shape]
    if len(shape) == 2:
        boot = cell_values[idxs[0][:, :, None], idxs[1][:, None, :]]
    elif len(shape) == 3:
        boot = cell_values[
            idxs[0][:, :, None, None],
            idxs[1][:, None, :, None],
            idxs[2][:, None, None, :],
        ]
    else:
        raise ValueError("bootstrap_cells_mean supports 2 or 3 crossed factors")
    boot_mean = boot.reshape(n_boot, -1).mean(axis=1)
    lo, hi = np.percentile(boot_mean, [2.5, 97.5])
    return float(cell_values.mean()), float(lo), float(hi)


def paired_t_summary(diffs: np.ndarray) -> dict:
    """Paired t and exact Wilcoxon summary for a vector of paired differences."""
    d = np.asarray(diffs, dtype=float)
    n = len(d)
    m = float(np.mean(d))
    se = float(stats.sem(d)) if n > 1 else np.nan
    lo, hi = t_ci(m, se, n - 1)
    t_p = float(stats.ttest_1samp(d, 0.0).pvalue) if np.std(d) > 0 else np.nan
    try:
        w_p = float(stats.wilcoxon(d, zero_method="wilcox", alternative="two-sided").pvalue)
    except ValueError:
        w_p = np.nan
    n_up = int((d > 0).sum())
    n_down = int((d < 0).sum())
    n_tie = int(n - n_up - n_down)
    sign_p = float(stats.binomtest(n_up, n_up + n_down, 0.5).pvalue) if (n_up + n_down) > 0 else np.nan
    return {
        "n": n,
        "mean_diff": m,
        "SE": se,
        "CI95_low": lo,
        "CI95_high": hi,
        "CI95_width": hi - lo if np.isfinite(hi - lo) else np.nan,
        "paired_t_p": t_p,
        "wilcoxon_p": w_p,
        "n_up": n_up,
        "n_tie": n_tie,
        "n_down": n_down,
        "sign_test_p": sign_p,
        "differences": ";".join(f"{v:.6f}" for v in d),
    }


def exact_permutation_p(x: np.ndarray, y: np.ndarray, method: str) -> tuple[float, int]:
    """Exact two-sided permutation p by enumerating all permutations of y."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if method == "pearson":
        obs = safe_pearson(x, y)
        fn = safe_pearson
    elif method == "spearman":
        obs = safe_spearman(x, y)
        fn = safe_spearman
    else:
        raise ValueError(method)
    if not np.isfinite(obs):
        return np.nan, 0
    stats_list = []
    for perm in itertools.permutations(range(len(y))):
        stats_list.append(fn(x, y[list(perm)]))
    vals = np.asarray(stats_list, dtype=float)
    p = float(np.mean(np.abs(vals) >= abs(obs) - 1e-12))
    return p, len(vals)


def bootstrap_corr_ci(
    x: np.ndarray,
    y: np.ndarray,
    method: str,
    rng: np.random.Generator,
    n_boot: int = N_BOOT,
) -> tuple[float, float]:
    """Percentile bootstrap CI for a correlation, resampling platform pairs."""
    n = len(x)
    fn = safe_pearson if method == "pearson" else safe_spearman
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        v = fn(x[idx], y[idx])
        if np.isfinite(v):
            vals.append(v)
    if not vals:
        return (np.nan, np.nan)
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(lo), float(hi)


def fisher_ci(r: float, n: int) -> tuple[float, float]:
    if not np.isfinite(r) or abs(r) >= 1 or n <= 3:
        return (np.nan, np.nan)
    z = np.arctanh(r)
    se = 1.0 / math.sqrt(n - 3)
    return (float(np.tanh(z - 1.96 * se)), float(np.tanh(z + 1.96 * se)))


# ---------------------------------------------------------------------------
# Load and validate inputs
# ---------------------------------------------------------------------------
def load_b1c() -> pd.DataFrame:
    df = pd.read_csv(B1C)
    logger.info("B1c path: %s", B1C)
    logger.info("B1c shape: %s", df.shape)
    logger.info("B1c columns: %s", df.columns.tolist())
    for col in ["Method", "Method_Label", "Rule_Key", "Depth_M", "Group", "Platform"]:
        vals = sorted(df[col].dropna().unique().tolist())
        logger.info("B1c unique %s (%d): %s", col, len(vals), vals)
    df["condition"] = df["Group"].astype(str) + "_" + df["Depth_M"].astype(str)
    df["cond_method"] = df["condition"] + "_" + df["Platform"].astype(str)
    logger.info("B1c conditions (donor x depth): %s", sorted(df["condition"].unique()))
    logger.info("B1c method conditions (donor x platform x depth): %d", df["cond_method"].nunique())
    logger.info("Rows per condition (first 5): %s", df.groupby("condition").size().head().to_dict())
    logger.info("Rows per method-condition (first 5): %s", df.groupby("cond_method").size().head().to_dict())
    # Structural checks
    assert len(df) == 480, len(df)
    assert df.groupby("condition")["Platform"].nunique().eq(6).all()
    assert df.groupby("condition")["Rule_Key"].nunique().eq(8).all()
    assert not df.duplicated(["Group", "Depth_M", "Platform", "Rule_Key"]).any()
    assert df[["Recall", "Precision", "F1"]].notna().all().all()
    logger.info("B1c structural checks passed: 480 rows, 10 conditions x 6 platforms x 8 rules.")
    return df


def load_a3() -> pd.DataFrame:
    df = pd.read_csv(A3)
    logger.info("A3 path: %s", A3)
    logger.info("A3 shape: %s", df.shape)
    logger.info("A3 datasets: %s", sorted(df["dataset"].unique().tolist()))
    logger.info(
        "A3 platforms by dataset: %s",
        df.groupby("dataset")["platform"].apply(lambda s: sorted(s.unique().tolist())).to_dict(),
    )
    logger.info("A3 quality thresholds: %s", sorted(df["quality_threshold"].unique().tolist()))
    logger.info("A3 support flags: %s", df["support_flag"].value_counts().to_dict())
    return df


def load_twist(path: Path, dataset: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    logger.info("%s path: %s shape: %s", dataset, path, df.shape)
    logger.info("%s platforms: %s", dataset, sorted(df["Platform"].unique().tolist()))
    logger.info("%s replicates: %s", dataset, sorted(df["Replicate"].unique().tolist()))
    logger.info("%s methods: %s", dataset, sorted(df["Method"].unique().tolist()))
    return df


def twist_three_methods(df: pd.DataFrame, dataset: str) -> pd.DataFrame:
    """Return the three method rows at SomaticXGB threshold 0.1 only."""
    if dataset == "Physical":
        xgb = df[(df["Method"] == "SomaticXGB") & np.isclose(df["Threshold"].astype(float), 0.10)].copy()
        xgb["Method"] = "SomaticXGB (0.1)"
        other = df[df["Method"].isin(["Strelka2_Mutect2", "VarScan2_fpfilter"])].copy()
        out = pd.concat([xgb, other], ignore_index=True)
    else:
        out = df[df["Method"].isin(["SomaticXGB (0.1)", "Strelka2_Mutect2", "VarScan2_fpfilter"])].copy()
    keep = ["Platform", "Replicate", "Method", "Recall", "Precision", "F1"]
    out = out[keep].copy()
    assert out.groupby(["Platform", "Replicate", "Method"]).size().eq(1).all()
    assert out.groupby("Method").size().eq(15).all()
    assert out.groupby(["Method", "Replicate"])["Platform"].nunique().eq(5).all()
    logger.info("%s analysis rows: %d (5 platforms x 3 replicates x 3 methods)", dataset, len(out))
    return out


# ---------------------------------------------------------------------------
# Analysis 1: GIAB platform comparison (SomaticXGB fixed)
# ---------------------------------------------------------------------------
def analysis_platform(df: pd.DataFrame, rng: np.random.Generator):
    log_section("1. GIAB platform comparison, SomaticXGB fixed (R2-7 / R3-5)")
    mixed_rows = []
    boot_rows = []
    cluster_rows = []
    model_infos = []

    threshold_rules = [("xgb_pred01", "SomaticXGB 0.1"), ("xgb_pred06", "SomaticXGB 0.6")]
    metrics = ["F1", "Recall", "Precision"]

    for rule, threshold_label in threshold_rules:
        d = df[df["Rule_Key"] == rule].copy()
        assert len(d) == 60
        for metric in metrics:
            # MixedLM with condition random intercept
            mdl, info = fit_mixed_contrast(
                d, metric, "Platform", "SURFSeq 5000", "condition", levels=PLATFORMS
            )
            model_infos.append({"analysis": "platform", "threshold": threshold_label, "metric": metric, **info})
            if mdl is None:
                logger.error("Platform MixedLM failed for %s / %s", threshold_label, metric)
                continue
            mdl["analysis"] = "GIAB_platform"
            mdl["metric"] = metric
            mdl["threshold"] = threshold_label
            mdl["rule_key"] = rule
            mixed_rows.append(mdl)
            logger.info(
                "Platform MixedLM | %s | %s | n_obs=%d n_groups=%d df=%d group_var=%.6g resid_var=%.6g",
                threshold_label, metric, info["n_obs"], info["n_groups"], info["df"],
                info["group_var"], info["resid_var"],
            )
            for _, r in mdl.iterrows():
                if r["estimate_type"] == "difference":
                    logger.info(
                        "  %-15s diff=%+.4f SE=%.4f CI=[%+.4f, %+.4f] p_t=%.4g p_fdr=%.4g",
                        r["term"], r["estimate"], r["SE"], r["CI95_low"], r["CI95_high"],
                        r["p"], r["p_fdr"],
                    )

            # Condition cluster bootstrap
            piv = d.pivot_table(index="condition", columns="Platform", values=metric)
            comps = [p for p in PLATFORMS if p != "SURFSeq 5000"]
            diff = piv[comps].sub(piv["SURFSeq 5000"], axis=0)
            bres = condition_bootstrap(diff.to_numpy(), comps, rng)
            bres["analysis"] = "GIAB_platform"
            bres["metric"] = metric
            bres["threshold"] = threshold_label
            bres["reference"] = "SURFSeq 5000"
            # group- and depth-specific mean differences
            long = diff.copy()
            long["condition"] = long.index
            long["Group"] = long["condition"].str.split("_").str[0].astype(int)
            long["Depth_M"] = long["condition"].str.split("_").str[1].astype(int)
            for i, comp in enumerate(comps):
                sub = long[[comp, "Group", "Depth_M"]]
                bres.loc[i, "mean_diff_group_45"] = sub.loc[sub["Group"] == 45, comp].mean()
                bres.loc[i, "mean_diff_group_54"] = sub.loc[sub["Group"] == 54, comp].mean()
                bres.loc[i, "mean_diff_depth_min"] = sub.groupby("Depth_M")[comp].mean().min()
                bres.loc[i, "mean_diff_depth_max"] = sub.groupby("Depth_M")[comp].mean().max()
                bres.loc[i, "n_conditions_group_45"] = int((sub["Group"] == 45).sum())
                bres.loc[i, "n_conditions_group_54"] = int((sub["Group"] == 54).sum())
            bres["n_boot"] = N_BOOT
            bres["seed"] = SEED
            boot_rows.append(bres)

            # Hierarchical sensitivity: resample donor direction x depth cells
            cell = np.empty((2, 5), dtype=float)
            groups = [45, 54]
            depths = sorted(long["Depth_M"].unique().tolist())
            for gi, g in enumerate(groups):
                for di, dep in enumerate(depths):
                    cell[gi, di] = long.loc[(long["Group"] == g) & (long["Depth_M"] == dep), comps[0]].mean()
            # one cell array per comparator
            for i, comp in enumerate(comps):
                for gi, g in enumerate(groups):
                    for di, dep in enumerate(depths):
                        cell[gi, di] = long.loc[
                            (long["Group"] == g) & (long["Depth_M"] == dep), comp
                        ].mean()
                m, lo, hi = bootstrap_cells_mean(cell, rng)
                cluster_rows.append(
                    {
                        "analysis": "GIAB_platform",
                        "metric": metric,
                        "threshold": threshold_label,
                        "contrast": comp,
                        "boot_unit": "group_x_depth",
                        "n_units": 10,
                        "mean_diff": m,
                        "CI_low": lo,
                        "CI_high": hi,
                        "excludes_zero": bool(lo > 0 or hi < 0),
                        "n_boot": N_BOOT,
                        "seed": SEED,
                    }
                )

    platform_mixed = pd.concat(mixed_rows, ignore_index=True)
    platform_mixed["platform"] = platform_mixed["term"]
    platform_mixed["CI95"] = platform_mixed.apply(
        lambda r: f"{r['CI95_low']:+.4f} to {r['CI95_high']:+.4f}", axis=1
    )
    platform_boot = pd.concat(boot_rows, ignore_index=True)
    platform_boot["mean_CI95"] = platform_boot.apply(
        lambda r: f"{r['mean_CI_low']:+.4f} to {r['mean_CI_high']:+.4f}", axis=1
    )
    platform_boot["median_CI95"] = platform_boot.apply(
        lambda r: f"{r['median_CI_low']:+.4f} to {r['median_CI_high']:+.4f}", axis=1
    )
    platform_cluster = pd.DataFrame(cluster_rows)
    platform_cluster["CI95"] = platform_cluster.apply(
        lambda r: f"{r['CI_low']:+.4f} to {r['CI_high']:+.4f}", axis=1
    )
    return platform_mixed, platform_boot, platform_cluster, model_infos


# ---------------------------------------------------------------------------
# Analysis 2: GIAB method comparison (platform not fixed)
# ---------------------------------------------------------------------------
def analysis_method(df: pd.DataFrame, rng: np.random.Generator):
    log_section("2. GIAB method comparison (method within condition; R3-5)")
    mixed_rows = []
    boot_rows = []
    cluster_rows = []
    model_infos = []

    metrics = ["F1", "Recall", "Precision"]
    d = df[df["Rule_Key"].isin(PRIMARY_RULES)].copy()
    assert len(d) == 360
    assert d.groupby("cond_method")["Rule_Key"].nunique().eq(6).all()

    for metric in metrics:
        mdl, info = fit_mixed_contrast(
            d, metric, "Rule_Key", "xgb_pred01", "cond_method", levels=PRIMARY_RULES
        )
        model_infos.append({"analysis": "method", "threshold": "primary6", "metric": metric, **info})
        if mdl is None:
            logger.error("Method MixedLM failed for %s", metric)
            continue
        mdl["analysis"] = "GIAB_method"
        mdl["comparison_set"] = "primary (6 methods)"
        mdl["threshold"] = "primary (6 methods)"
        mdl["metric"] = metric
        mdl["method"] = mdl["term"].map(lambda x: RULE_LABEL.get(x, x))
        mdl["reference"] = "SomaticXGB (0.1)"
        mixed_rows.append(mdl)
        logger.info(
            "Method MixedLM | %s | n_obs=%d n_groups=%d df=%d group_var=%.6g resid_var=%.6g",
            metric, info["n_obs"], info["n_groups"], info["df"], info["group_var"], info["resid_var"],
        )
        for _, r in mdl.iterrows():
            if r["estimate_type"] == "difference":
                logger.info(
                    "  %-38s diff=%+.4f SE=%.4f CI=[%+.4f, %+.4f] p_t=%.4g p_fdr=%.4g",
                    RULE_LABEL.get(r["term"], r["term"]), r["estimate"], r["SE"],
                    r["CI95_low"], r["CI95_high"], r["p"], r["p_fdr"],
                )

        # Condition cluster bootstrap
        piv = d.pivot_table(index="cond_method", columns="Rule_Key", values=metric)
        comps = [r for r in PRIMARY_RULES if r != "xgb_pred01"]
        diff = piv[comps].sub(piv["xgb_pred01"], axis=0)
        bres = condition_bootstrap(diff.to_numpy(), comps, rng)
        bres["analysis"] = "GIAB_method"
        bres["metric"] = metric
        bres["threshold"] = "primary (6 methods)"
        bres["reference"] = "SomaticXGB (0.1)"
        # Attach condition metadata
        meta = d[["cond_method", "Group", "Platform", "Depth_M"]].drop_duplicates().set_index("cond_method")
        long = diff.copy()
        long = long.join(meta)
        for i, comp in enumerate(comps):
            sub = long[[comp, "Group", "Platform", "Depth_M"]]
            bres.loc[i, "mean_diff_group_45"] = sub.loc[sub["Group"] == 45, comp].mean()
            bres.loc[i, "mean_diff_group_54"] = sub.loc[sub["Group"] == 54, comp].mean()
            bres.loc[i, "mean_diff_platform_min"] = sub.groupby("Platform")[comp].mean().min()
            bres.loc[i, "mean_diff_platform_max"] = sub.groupby("Platform")[comp].mean().max()
            bres.loc[i, "mean_diff_depth_min"] = sub.groupby("Depth_M")[comp].mean().min()
            bres.loc[i, "mean_diff_depth_max"] = sub.groupby("Depth_M")[comp].mean().max()
        bres["n_boot"] = N_BOOT
        bres["seed"] = SEED
        boot_rows.append(bres)

        # Hierarchical sensitivity: platform, depth, and group x platform x depth
        # platform cluster
        for i, comp in enumerate(comps):
            vals = long.groupby("Platform")[comp].mean().to_numpy()
            m, lo, hi = bootstrap_vector_mean(vals, rng)
            cluster_rows.append(
                {
                    "analysis": "GIAB_method", "metric": metric, "threshold": "primary (6 methods)",
                    "contrast": comp, "boot_unit": "platform", "n_units": len(vals),
                    "mean_diff": m, "CI_low": lo, "CI_high": hi,
                    "excludes_zero": bool(lo > 0 or hi < 0),
                    "n_boot": N_BOOT, "seed": SEED,
                }
            )
            vals = long.groupby("Depth_M")[comp].mean().to_numpy()
            m, lo, hi = bootstrap_vector_mean(vals, rng)
            cluster_rows.append(
                {
                    "analysis": "GIAB_method", "metric": metric, "threshold": "primary (6 methods)",
                    "contrast": comp, "boot_unit": "depth", "n_units": len(vals),
                    "mean_diff": m, "CI_low": lo, "CI_high": hi,
                    "excludes_zero": bool(lo > 0 or hi < 0),
                    "n_boot": N_BOOT, "seed": SEED,
                }
            )
            cell = np.empty((2, 6, 5), dtype=float)
            groups = [45, 54]
            plats = PLATFORMS
            depths = sorted(long["Depth_M"].unique().tolist())
            for gi, g in enumerate(groups):
                for pi, p in enumerate(plats):
                    for di, dep in enumerate(depths):
                        cell[gi, pi, di] = long.loc[
                            (long["Group"] == g) & (long["Platform"] == p) & (long["Depth_M"] == dep),
                            comp,
                        ].mean()
            m, lo, hi = bootstrap_cells_mean(cell, rng)
            cluster_rows.append(
                {
                    "analysis": "GIAB_method", "metric": metric, "threshold": "primary (6 methods)",
                    "contrast": comp, "boot_unit": "group_x_platform_x_depth", "n_units": 60,
                    "mean_diff": m, "CI_low": lo, "CI_high": hi,
                    "excludes_zero": bool(lo > 0 or hi < 0),
                    "n_boot": N_BOOT, "seed": SEED,
                }
            )

    # Sensitivity model: add Mutect2/union TLOD>=10, F1 only
    d8 = df[df["Rule_Key"].isin(ALL_RULES)].copy()
    assert len(d8) == 480
    mdl8, info8 = fit_mixed_contrast(
        d8, "F1", "Rule_Key", "xgb_pred01", "cond_method", levels=ALL_RULES
    )
    model_infos.append({"analysis": "method", "threshold": "sensitivity8", "metric": "F1", **info8})
    if mdl8 is not None:
        mdl8["analysis"] = "GIAB_method"
        mdl8["comparison_set"] = "sensitivity (8 methods, F1 only)"
        mdl8["threshold"] = "sensitivity (8 methods, F1 only)"
        mdl8["metric"] = "F1"
        mdl8["method"] = mdl8["term"].map(lambda x: RULE_LABEL.get(x, x))
        mdl8["reference"] = "SomaticXGB (0.1)"
        # retain only sensitivity terms and reference
        keep = mdl8["term"].isin(["xgb_pred01"] + SENS_RULES)
        mixed_rows.append(mdl8[keep].copy())
        for _, r in mdl8[keep].iterrows():
            if r["estimate_type"] == "difference":
                logger.info(
                    "  sensitivity %-38s diff=%+.4f SE=%.4f CI=[%+.4f, %+.4f] p_t=%.4g",
                    RULE_LABEL.get(r["term"], r["term"]), r["estimate"], r["SE"],
                    r["CI95_low"], r["CI95_high"], r["p"],
                )

    # Bootstrap sensitivity contrasts from the 8-rule data
    for metric in ["F1"]:
        piv = d8.pivot_table(index="cond_method", columns="Rule_Key", values=metric)
        for comp in SENS_RULES:
            diff = (piv[comp] - piv["xgb_pred01"]).to_numpy()
            bres = condition_bootstrap(diff[:, None], [comp], rng)
            bres["analysis"] = "GIAB_method"
            bres["metric"] = metric
            bres["threshold"] = "sensitivity (8 methods, F1 only)"
            bres["reference"] = "SomaticXGB (0.1)"
            bres["n_boot"] = N_BOOT
            bres["seed"] = SEED
            boot_rows.append(bres)

    method_mixed = pd.concat(mixed_rows, ignore_index=True)
    method_mixed["CI95"] = method_mixed.apply(
        lambda r: f"{r['CI95_low']:+.4f} to {r['CI95_high']:+.4f}", axis=1
    )
    method_boot = pd.concat(boot_rows, ignore_index=True)
    method_boot["mean_CI95"] = method_boot.apply(
        lambda r: f"{r['mean_CI_low']:+.4f} to {r['mean_CI_high']:+.4f}", axis=1
    )
    method_boot["median_CI95"] = method_boot.apply(
        lambda r: f"{r['median_CI_low']:+.4f} to {r['median_CI_high']:+.4f}", axis=1
    )
    method_cluster = pd.DataFrame(cluster_rows)
    method_cluster["CI95"] = method_cluster.apply(
        lambda r: f"{r['CI_low']:+.4f} to {r['CI_high']:+.4f}", axis=1
    )
    return method_mixed, method_boot, method_cluster, model_infos


# ---------------------------------------------------------------------------
# Analysis 3: Twist three-replicate data
# ---------------------------------------------------------------------------
def analysis_twist(phys: pd.DataFrame, hyb: pd.DataFrame):
    log_section("3. Twist per-replicate analyses (3 independent libraries)")
    all_rows = []
    for dataset, df in [("Physical Twist 1%", phys), ("Twist variants in HG001", hyb)]:
        d = twist_three_methods(df, "Physical" if dataset.startswith("Physical") else "Hybrid")
        metrics = ["F1", "Recall", "Precision"]

        # --- Platform comparison holding method = SomaticXGB 0.1 ---
        dp = d[d["Method"] == "SomaticXGB (0.1)"].copy()
        for metric in metrics:
            # MixedLM
            mdl, info = fit_mixed_contrast(
                dp, metric, "Platform", "SURFSeq 5000", "Replicate", levels=PT_PLATFORMS
            )
            if mdl is not None:
                mdl["dataset"] = dataset
                mdl["comparison_type"] = "platform"
                mdl["metric"] = metric
                mdl["inference"] = "MixedLM (random intercept for replicate)"
                mdl["primary"] = False
                mdl["notes"] = (
                    "Three replicate groups; MixedLM assumes homogeneous within-replicate "
                    "variance and its CI is narrower than the paired-t CI. Interpret as sensitivity."
                )
                all_rows.append(mdl)
                logger.info(
                    "Twist platform MixedLM | %s | %s | df=%d group_var=%.6g resid_var=%.6g",
                    dataset, metric, info["df"], info["group_var"], info["resid_var"],
                )
            else:
                all_rows.append(
                    pd.DataFrame(
                        [{
                            "dataset": dataset, "comparison_type": "platform", "metric": metric,
                            "term": "all contrasts", "estimate_type": "difference",
                            "estimate": np.nan, "SE": np.nan, "CI95_low": np.nan,
                            "CI95_high": np.nan, "p": np.nan, "p_fdr": np.nan, "df": np.nan,
                            "inference": "MixedLM (random intercept for replicate)",
                            "primary": False,
                            "notes": f"MixedLM not estimable: {info['error']}",
                        }]
                    )
                )

            # Block OLS with replicate fixed effect
            ols = fit_block_ols(dp, metric, "Platform", "SURFSeq 5000", "Replicate")
            if ols is not None:
                ols["dataset"] = dataset
                ols["comparison_type"] = "platform"
                ols["metric"] = metric
                ols["inference"] = "Block OLS (replicate fixed effect)"
                ols["primary"] = False
                ols["notes"] = "Complete-block OLS; assumes no platform x replicate interaction."
                all_rows.append(ols)

            # Paired t across the three replicate differences
            piv = dp.pivot_table(index="Replicate", columns="Platform", values=metric)
            for comp in [p for p in PT_PLATFORMS if p != "SURFSeq 5000"]:
                s = paired_t_summary((piv[comp] - piv["SURFSeq 5000"]).to_numpy())
                all_rows.append(
                    pd.DataFrame(
                        [{
                            "dataset": dataset, "comparison_type": "platform", "metric": metric,
                            "term": comp, "reference": "SURFSeq 5000", "estimate_type": "difference",
                            "estimate": s["mean_diff"], "SE": s["SE"],
                            "CI95_low": s["CI95_low"], "CI95_high": s["CI95_high"],
                            "CI95_width": s["CI95_width"], "p": s["paired_t_p"],
                            "p_fdr": np.nan, "df": 2.0,
                            "inference": "Paired t (3 replicates, df=2)",
                            "primary": True,
                            "wilcoxon_p": s["wilcoxon_p"],
                            "n_up": s["n_up"], "n_tie": s["n_tie"], "n_down": s["n_down"],
                            "differences": s["differences"],
                            "notes": "Primary conservative interval for generalization to new replicate libraries.",
                        }]
                    )
                )

        # --- Method comparison: average across platforms within replicate ---
        a = d.groupby(["Replicate", "Method"])[metrics].mean().reset_index()
        assert len(a) == 9
        for metric in metrics:
            # Try MixedLM (often singular with 3 groups); record failure honestly
            mdl, info = fit_mixed_contrast(
                a, metric, "Method", "SomaticXGB (0.1)", "Replicate",
                levels=["SomaticXGB (0.1)", "Strelka2_Mutect2", "VarScan2_fpfilter"],
            )
            if mdl is not None:
                mdl["dataset"] = dataset
                mdl["comparison_type"] = "method"
                mdl["metric"] = metric
                mdl["inference"] = "MixedLM (random intercept for replicate)"
                mdl["primary"] = False
                mdl["notes"] = (
                    "Three replicate groups; MixedLM may be at the variance boundary. "
                    "Paired t is the primary interval."
                )
                all_rows.append(mdl)
            else:
                all_rows.append(
                    pd.DataFrame(
                        [{
                            "dataset": dataset, "comparison_type": "method", "metric": metric,
                            "term": "all method contrasts", "estimate_type": "difference",
                            "estimate": np.nan, "SE": np.nan, "CI95_low": np.nan,
                            "CI95_high": np.nan, "p": np.nan, "p_fdr": np.nan, "df": np.nan,
                            "inference": "MixedLM (random intercept for replicate)",
                            "primary": False,
                            "notes": f"MixedLM not estimable with 3 replicate groups: {info['error']}",
                        }]
                    )
                )

            # Block OLS
            ols = fit_block_ols(a, metric, "Method", "SomaticXGB (0.1)", "Replicate")
            if ols is not None:
                ols["dataset"] = dataset
                ols["comparison_type"] = "method"
                ols["metric"] = metric
                ols["inference"] = "Block OLS (replicate fixed effect)"
                ols["primary"] = False
                ols["notes"] = "Complete-block OLS; assumes no method x replicate interaction."
                all_rows.append(ols)

            # Paired t across replicates (method - SomaticXGB 0.1)
            piv = a.pivot(index="Replicate", columns="Method", values=metric)
            for comp in ["Strelka2_Mutect2", "VarScan2_fpfilter"]:
                s = paired_t_summary((piv[comp] - piv["SomaticXGB (0.1)"]).to_numpy())
                all_rows.append(
                    pd.DataFrame(
                        [{
                            "dataset": dataset, "comparison_type": "method", "metric": metric,
                            "term": comp, "reference": "SomaticXGB (0.1)",
                            "estimate_type": "difference",
                            "estimate": s["mean_diff"], "SE": s["SE"],
                            "CI95_low": s["CI95_low"], "CI95_high": s["CI95_high"],
                            "CI95_width": s["CI95_width"], "p": s["paired_t_p"],
                            "p_fdr": np.nan, "df": 2.0,
                            "inference": "Paired t (3 replicates, df=2)",
                            "primary": True,
                            "wilcoxon_p": s["wilcoxon_p"],
                            "n_up": s["n_up"], "n_tie": s["n_tie"], "n_down": s["n_down"],
                            "differences": s["differences"],
                            "notes": "Primary conservative interval; negative estimate means the comparator is lower than SomaticXGB 0.1.",
                        }]
                    )
                )

    out = pd.concat(all_rows, ignore_index=True)
    out["CI95"] = out.apply(
        lambda r: f"{r['CI95_low']:+.4f} to {r['CI95_high']:+.4f}"
        if pd.notna(r["CI95_low"]) and pd.notna(r["CI95_high"]) else "NA",
        axis=1,
    )
    # FDR within dataset x comparison_type x metric x inference
    out["p_fdr"] = np.nan
    for keys, grp in out.groupby(["dataset", "comparison_type", "metric", "inference"], dropna=False):
        mask = grp["estimate_type"] == "difference"
        if mask.sum() > 0:
            out.loc[grp.index[mask], "p_fdr"] = bh_fdr(grp.loc[mask, "p"].to_numpy())
    return out


# Twist platform order (five platforms; T7 not present)
PT_PLATFORMS = [
    "SURFSeq 5000",
    "FASTASeq 300",
    "GenoLab M",
    "NovaSeq 6000",
    "NovaSeq X",
]


# ---------------------------------------------------------------------------
# Analysis 4: error rate vs F1 (platform level, n = 6)
# ---------------------------------------------------------------------------
def analysis_error_f1(df_b1c: pd.DataFrame, df_a3: pd.DataFrame, rng: np.random.Generator):
    log_section("4. Platform error rate vs calling performance (R3-5, third item)")
    # Error rates: GIAB five TruSeq platforms + DNBSEQ-T7 from its own dataset
    err_rows = []
    for dataset, platform_subset in [("GIAB", TRUSEQ), ("T7", ["DNBSEQ-T7"])]:
        sub = df_a3[
            (df_a3["dataset"] == dataset)
            & (df_a3["quality_threshold"].isin([0, 30]))
            & (df_a3["support_flag"] == "ok")
            & (df_a3["platform"].isin(platform_subset))
        ]
        for _, r in sub.iterrows():
            err_rows.append(
                {
                    "platform": r["platform"],
                    "error_threshold": int(r["quality_threshold"]),
                    "mean_error_rate": float(r["mean_error_rate"]),
                    "pooled_error_rate": float(r["pooled_error_rate"]),
                    "n_samples": int(r["n_samples"]),
                    "a3_dataset": dataset,
                }
            )
    err = pd.DataFrame(err_rows)
    logger.info("Platform error-rate table:\n%s", err.to_string(index=False))
    assert err.groupby("error_threshold")["platform"].nunique().eq(6).all()

    # F1 platform means from B1c
    f1_rows = []
    for rule, label in [("xgb_pred01", "SomaticXGB (0.1)"), ("xgb_pred06", "SomaticXGB (0.6)")]:
        sub = df_b1c[df_b1c["Rule_Key"] == rule]
        means = sub.groupby("Platform")["F1"].mean()
        for p, v in means.items():
            f1_rows.append(
                {
                    "platform": p,
                    "f1_method": label,
                    "f1_mean": float(v),
                    "n_conditions": int((sub["Platform"] == p).sum()),
                }
            )
    f1 = pd.DataFrame(f1_rows)
    assert f1.groupby("f1_method").size().eq(6).all()

    points = f1.merge(err, on=["platform"], how="left")
    assert points[["mean_error_rate", "f1_mean"]].notna().all().all()
    points["platform_set"] = "all6"
    points.to_csv(TABLES / "R27_error_f1_points.csv", index=False)
    logger.info("Wrote %s", TABLES / "R27_error_f1_points.csv")

    # Within-depth F1 means for a consistency check
    depth_means = {}
    for rule, label in [("xgb_pred01", "SomaticXGB (0.1)"), ("xgb_pred06", "SomaticXGB (0.6)")]:
        sub = df_b1c[df_b1c["Rule_Key"] == rule]
        dm = sub.groupby(["Depth_M", "Platform"])["F1"].mean().reset_index()
        depth_means[label] = dm

    rows = []
    for error_threshold in [0, 30]:
        for f1_method in ["SomaticXGB (0.1)", "SomaticXGB (0.6)"]:
            for platform_set, plats in [("all6", PLATFORMS), ("TruSeq5_excluding_T7", TRUSEQ)]:
                sub = points[
                    (points["error_threshold"] == error_threshold)
                    & (points["f1_method"] == f1_method)
                    & (points["platform"].isin(plats))
                ].copy()
                # preserve platform order
                sub["platform"] = pd.Categorical(sub["platform"], categories=plats, ordered=True)
                sub = sub.sort_values("platform")
                x = sub["mean_error_rate"].to_numpy(dtype=float)
                y = sub["f1_mean"].to_numpy(dtype=float)
                n = len(x)
                pear_r = safe_pearson(x, y)
                pear_p = float(stats.pearsonr(x, y)[1]) if np.isfinite(pear_r) else np.nan
                pear_perm, n_perm = exact_permutation_p(x, y, "pearson")
                pear_boot = bootstrap_corr_ci(x, y, "pearson", rng)
                pear_fisher = fisher_ci(pear_r, n)
                spear_rho = safe_spearman(x, y)
                spear_p = float(stats.spearmanr(x, y)[1]) if np.isfinite(spear_rho) else np.nan
                spear_perm, n_perm_s = exact_permutation_p(x, y, "spearman")
                spear_boot = bootstrap_corr_ci(x, y, "spearman", rng)

                # Leave-one-platform-out
                loo_r, loo_rho = [], []
                for i in range(n):
                    keep = np.arange(n) != i
                    if len(x[keep]) >= 3:
                        loo_r.append(safe_pearson(x[keep], y[keep]))
                        loo_rho.append(safe_spearman(x[keep], y[keep]))
                loo_r = [v for v in loo_r if np.isfinite(v)]
                loo_rho = [v for v in loo_rho if np.isfinite(v)]

                # Within-depth Spearman
                dm = depth_means[f1_method]
                within = []
                for dep in sorted(dm["Depth_M"].unique().tolist()):
                    s = dm[dm["Depth_M"] == dep].set_index("Platform")["F1"]
                    s = s.reindex(plats)
                    xx = err[(err["error_threshold"] == error_threshold) & (err["platform"].isin(plats))]
                    xx = xx.set_index("platform").reindex(plats)["mean_error_rate"]
                    if s.notna().all() and xx.notna().all():
                        within.append(safe_spearman(xx.to_numpy(), s.to_numpy()))
                within = [v for v in within if np.isfinite(v)]

                rows.append(
                    {
                        "error_threshold": error_threshold,
                        "f1_method": f1_method,
                        "platform_set": platform_set,
                        "n_platforms": n,
                        "platforms": ";".join(plats),
                        "pearson_r": pear_r,
                        "pearson_p_asymptotic": pear_p,
                        "pearson_p_permutation": pear_perm,
                        "pearson_CI_low": pear_fisher[0],
                        "pearson_CI_high": pear_fisher[1],
                        "pearson_CI95": f"{pear_fisher[0]:+.4f} to {pear_fisher[1]:+.4f}",
                        "pearson_CI_method": "Fisher z (approximate)",
                        "pearson_boot_CI_low": pear_boot[0],
                        "pearson_boot_CI_high": pear_boot[1],
                        "spearman_rho": spear_rho,
                        "spearman_p_asymptotic": spear_p,
                        "spearman_p_permutation": spear_perm,
                        "spearman_CI_low": spear_boot[0],
                        "spearman_CI_high": spear_boot[1],
                        "spearman_CI95": f"{spear_boot[0]:+.4f} to {spear_boot[1]:+.4f}",
                        "spearman_CI_method": "percentile bootstrap",
                        "loo_pearson_min": min(loo_r) if loo_r else np.nan,
                        "loo_pearson_max": max(loo_r) if loo_r else np.nan,
                        "loo_spearman_min": min(loo_rho) if loo_rho else np.nan,
                        "loo_spearman_max": max(loo_rho) if loo_rho else np.nan,
                        "within_depth_spearman_min": min(within) if within else np.nan,
                        "within_depth_spearman_max": max(within) if within else np.nan,
                        "n_depths_with_negative_spearman": int(sum(v < 0 for v in within)),
                        "n_depths": len(within),
                        "n_permutations": n_perm,
                        "notes": (
                            "Platform-level correlation (n=6, or n=5 excluding DNBSEQ-T7). "
                            "Permutation p is exact over all platform label permutations. "
                            "Not a causal test; six platforms are underpowered. "
                            "DNBSEQ-T7 used a separate library preparation."
                        ),
                    }
                )
    out = pd.DataFrame(rows)
    logger.info(
        "Error-F1 primary results:\n%s",
        out[out["platform_set"].isin(["all6"])][
            ["error_threshold", "f1_method", "platform_set", "pearson_r", "pearson_p_permutation",
             "spearman_rho", "spearman_p_permutation"]
        ].to_string(index=False),
    )
    return out


# ---------------------------------------------------------------------------
# S11 audit and draft replacement
# ---------------------------------------------------------------------------
def audit_s11(hyb: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    log_section("5. Supplementary Table 11 audit and replacement draft")
    audit = {
        "new_xlsx": str(NEW_XLSX),
        "old_xlsx": str(OLD_XLSX),
        "new_s11_title": None,
        "header": None,
        "first_data_row": None,
        "n_rows": None,
        "has_ci_column": False,
        "header_data_misalignment": None,
        "old_s11_summary_exists": None,
        "old_s11_ci_method": None,
    }
    try:
        import openpyxl
        wb = openpyxl.load_workbook(NEW_XLSX, read_only=True)
        ws = wb["Table S11"]
        rows = []
        for i, row in enumerate(ws.iter_rows(values_only=True)):
            rows.append(row)
            if i >= 4:
                break
        audit["new_s11_title"] = rows[0][0]
        audit["header"] = list(rows[1][:4])
        audit["first_data_row"] = list(rows[2][:4])
        audit["n_rows"] = ws.max_row
        hdr = [str(x).strip() if x is not None else "" for x in rows[1][:4]]
        data = rows[2][:4]
        # The current sheet puts the numeric value in the "Metric" column and the
        # metric name in the "Value" column.
        audit["has_ci_column"] = any("ci" in h.lower() or "confidence" in h.lower() for h in hdr)
        audit["header_data_misalignment"] = bool(
            isinstance(data[2], (int, float)) and isinstance(data[3], str)
        )
        logger.info("Table S11 title: %s", audit["new_s11_title"])
        logger.info("Table S11 header: %s", audit["header"])
        logger.info("Table S11 first data row: %s", audit["first_data_row"])
        logger.info("Table S11 rows: %s", audit["n_rows"])
        logger.info("Table S11 has CI column: %s", audit["has_ci_column"])
        logger.info("Table S11 header/data misalignment: %s", audit["header_data_misalignment"])
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not inspect new Table S11: %s", exc)

    try:
        wb = pd.ExcelFile(OLD_XLSX)
        if "S11_hybrid_summary" in wb.sheet_names:
            old = pd.read_excel(OLD_XLSX, sheet_name="S11_hybrid_summary")
            audit["old_s11_summary_exists"] = True
            audit["old_s11_ci_method"] = (
                "naive t over 15 platform-replicate observations; treats non-independent "
                "observations as independent"
            )
            logger.info("Old S11_hybrid_summary exists with columns: %s", old.columns.tolist())
            logger.info("Old S11_hybrid_summary rows: %d", len(old))
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not inspect old S11 summary: %s", exc)

    # Build a replacement S11 summary: method x metric across hybrid 15 obs.
    # Primary CI: two-way cluster bootstrap resampling platforms and replicates.
    # Sensitivity: t interval on the three replicate-level means (df=2).
    d = twist_three_methods(hyb, "Hybrid")
    rng = np.random.default_rng(SEED + 11)
    rows = []
    for method in ["SomaticXGB (0.1)", "Strelka2_Mutect2", "VarScan2_fpfilter"]:
        sub = d[d["Method"] == method]
        for metric in ["F1", "Recall", "Precision"]:
            piv = sub.pivot_table(index="Platform", columns="Replicate", values=metric)
            piv = piv.reindex(PT_PLATFORMS)
            vals = piv.to_numpy(dtype=float)
            assert vals.shape == (5, 3)
            mean15 = float(vals.mean())
            sd15 = float(vals.std(ddof=1))
            rep_means = vals.mean(axis=0)  # mean across platforms within each replicate
            se_rep = float(rep_means.std(ddof=1) / math.sqrt(3))
            lo_t, hi_t = t_ci(float(rep_means.mean()), se_rep, 2)
            # two-way cluster bootstrap
            n_boot = N_BOOT
            pi = rng.integers(0, 5, size=(n_boot, 5))
            ri = rng.integers(0, 3, size=(n_boot, 3))
            boot = vals[pi[:, :, None], ri[:, None, :]].reshape(n_boot, -1).mean(axis=1)
            lo_b, hi_b = np.percentile(boot, [2.5, 97.5])
            rows.append(
                {
                    "proposed_table": "S11 (replacement)",
                    "section": "Hybrid method summary",
                    "dataset": "Twist variants in HG001",
                    "analysis": "Descriptive mean with design-aware CI",
                    "metric": metric,
                    "threshold": "",
                    "term": method,
                    "reference": "",
                    "n_units": 15,
                    "n_replicates": 3,
                    "n_platforms": 5,
                    "estimate": mean15,
                    "SE": se_rep,
                    "CI95_low": float(lo_b),
                    "CI95_high": float(hi_b),
                    "p": np.nan,
                    "p_fdr": np.nan,
                    "inference": "Mean of 15 platform-replicate observations; 95% CI two-way cluster bootstrap over platforms and replicates",
                    "value_rep1": float(rep_means[0]),
                    "value_rep2": float(rep_means[1]),
                    "value_rep3": float(rep_means[2]),
                    "CI95_replicate_t_low": lo_t,
                    "CI95_replicate_t_high": hi_t,
                    "notes": (
                        f"SD across 15 observations = {sd15:.4f}. The 3 replicate means are independent libraries; "
                        "the replicate t CI (df=2) is a sensitivity interval. Platform means: "
                        + "; ".join(f"{p}={v:.4f}" for p, v in zip(PT_PLATFORMS, vals.mean(axis=1)))
                    ),
                }
            )
    out = pd.DataFrame(rows)
    for _, r in out.iterrows():
        logger.info(
            "S11 draft | %-20s %-9s mean=%.4f bootCI=[%.4f, %.4f] rep_tCI=[%.4f, %.4f]",
            r["term"], r["metric"], r["estimate"], r["CI95_low"], r["CI95_high"],
            r["CI95_replicate_t_low"], r["CI95_replicate_t_high"],
        )
    return out, audit


# ---------------------------------------------------------------------------
# Conclusion and supplementary draft
# ---------------------------------------------------------------------------
def build_conclusion(platform_mixed, platform_boot, method_mixed, method_boot, twist, err, s11_audit) -> str:
    # Helper to pull a row
    def pm(metric, term, threshold="SomaticXGB 0.1"):
        sub = platform_mixed[
            (platform_mixed["metric"] == metric)
            & (platform_mixed["threshold"] == threshold)
            & (platform_mixed["term"] == term)
            & (platform_mixed["estimate_type"] == "difference")
        ]
        return sub.iloc[0] if len(sub) else None

    def pb(metric, term, threshold="SomaticXGB 0.1"):
        sub = platform_boot[
            (platform_boot["metric"] == metric)
            & (platform_boot["threshold"] == threshold)
            & (platform_boot["contrast"] == term)
        ]
        return sub.iloc[0] if len(sub) else None

    def mm(metric, term, comparison_set="primary (6 methods)"):
        sub = method_mixed[
            (method_mixed["metric"] == metric)
            & (method_mixed["comparison_set"] == comparison_set)
            & (method_mixed["term"] == term)
            & (method_mixed["estimate_type"] == "difference")
        ]
        return sub.iloc[0] if len(sub) else None

    def mb(metric, term):
        sub = method_boot[
            (method_boot["metric"] == metric)
            & (method_boot["contrast"] == term)
        ]
        return sub.iloc[0] if len(sub) else None

    def tw(dataset, ctype, metric, term, inference_prefix="Paired t"):
        sub = twist[
            (twist["dataset"] == dataset)
            & (twist["comparison_type"] == ctype)
            & (twist["metric"] == metric)
            & (twist["term"] == term)
            & (twist["inference"].astype(str).str.startswith(inference_prefix))
        ]
        return sub.iloc[0] if len(sub) else None

    def er(q, f1m, pset):
        sub = err[
            (err["error_threshold"] == q)
            & (err["f1_method"] == f1m)
            & (err["platform_set"] == pset)
        ]
        return sub.iloc[0] if len(sub) else None

    # Platform F1 summaries
    plat_f1 = []
    for p in ["DNBSEQ-T7", "FASTASeq 300", "GenoLab M", "NovaSeq 6000", "NovaSeq X"]:
        r = pm("F1", p)
        b = pb("F1", p)
        if r is not None and b is not None:
            plat_f1.append(
                f"{p} {r['estimate']:+.3f} (MixedLM CI {r['CI95_low']:+.3f} to {r['CI95_high']:+.3f}; "
                f"bootstrap {b['mean_CI_low']:+.3f} to {b['mean_CI_high']:+.3f})"
            )
    plat_f1_txt = "; ".join(plat_f1)

    # Which platform recall/precision CIs exclude zero
    def excl_list(metric):
        out = []
        for p in ["DNBSEQ-T7", "FASTASeq 300", "GenoLab M", "NovaSeq 6000", "NovaSeq X"]:
            r = pm(metric, p)
            b = pb(metric, p)
            if r is not None and b is not None and (r["CI95_low"] > 0 or r["CI95_high"] < 0):
                out.append(f"{p} ({r['estimate']:+.3f}, {r['CI95_low']:+.3f} to {r['CI95_high']:+.3f})")
        return out

    rec_excl = excl_list("Recall")
    prec_excl = excl_list("Precision")

    # Method F1 contrasts (method - XGB0.1)
    mv = mm("F1", "varscan_ssc3")
    mu = mm("F1", "union_tlod8")
    ms = mm("F1", "strelka_default")
    mmu = mm("F1", "mutect2_tlod8")
    mx6 = mm("F1", "xgb_pred06")
    mbv = mb("F1", "varscan_ssc3")
    mbu = mb("F1", "union_tlod8")
    mbx6 = mb("F1", "xgb_pred06")
    mvr = mm("Recall", "varscan_ssc3")
    mvp = mm("Precision", "varscan_ssc3")
    msp = mm("Precision", "strelka_default")

    # Twist
    tw_plat_widths = []
    for ds in ["Physical Twist 1%", "Twist variants in HG001"]:
        ws = []
        for p in ["FASTASeq 300", "GenoLab M", "NovaSeq 6000", "NovaSeq X"]:
            r = tw(ds, "platform", "F1", p)
            if r is not None:
                ws.append(r["CI95_width"])
        if ws:
            tw_plat_widths.append(f"{ds}: {min(ws):.3f}-{max(ws):.3f}")
    tw_plat_txt = "; ".join(tw_plat_widths)
    hyb_str = tw("Twist variants in HG001", "method", "F1", "Strelka2_Mutect2")
    hyb_var = tw("Twist variants in HG001", "method", "F1", "VarScan2_fpfilter")
    hyb_var_rec = tw("Twist variants in HG001", "method", "Recall", "VarScan2_fpfilter")
    hyb_var_prec = tw("Twist variants in HG001", "method", "Precision", "VarScan2_fpfilter")
    phys_str = tw("Physical Twist 1%", "method", "F1", "Strelka2_Mutect2")
    phys_var = tw("Physical Twist 1%", "method", "F1", "VarScan2_fpfilter")

    # Error-F1
    e30 = er(30, "SomaticXGB (0.1)", "all6")
    e30_5 = er(30, "SomaticXGB (0.1)", "TruSeq5_excluding_T7")
    e0 = er(0, "SomaticXGB (0.1)", "all6")

    lines = []
    lines.append(
        "R27 addendum (2026-11-13), direct answer to R2-7 and R3-5. Independent units: GIAB condition = donor direction x depth "
        f"(platform varies within condition, n=10 blocks); GIAB method condition = donor x platform x depth (n=60 blocks); Twist = 3 "
        "independent library replicates. GIAB conditions are blocks, not biological replicates; inference is conditional on these two "
        "donor directions, five depths and six platforms and must not be extrapolated to new library pairs."
    )
    lines.append(
        "GIAB platform comparison, SomaticXGB 0.1, F1: SURFSeq 5000 is highest. MixedLM, the paired condition bootstrap, and the more "
        f"conservative group x depth cluster bootstrap all agree in direction and in excluding zero for every comparator: {plat_f1_txt}. "
        "DNBSEQ-T7 shows the largest difference but uses a separate library preparation, so that contrast is workflow-level, not instrument-level."
    )
    lines.append(
        "GIAB platform recall/precision (MixedLM): recall CIs excluding zero versus SURFSeq 5000 are "
        f"{'; '.join(rec_excl) if rec_excl else 'none'}; precision CIs excluding zero are "
        f"{'; '.join(prec_excl) if prec_excl else 'none'} (FASTASeq 300 precision includes zero). The F1 advantage of SURFSeq 5000 is therefore "
        "not driven by a single metric, although precision differences are smaller than recall differences."
    )
    lines.append(
        "GIAB method comparison, F1 (estimate = method minus SomaticXGB 0.1; negative favours SomaticXGB): vs fair-tuned VarScan2 SSC>=3 "
        f"{mv['estimate']:+.3f} (MixedLM 95% CI {mv['CI95_low']:+.3f} to {mv['CI95_high']:+.3f}; paired bootstrap {mbv['mean_CI_low']:+.3f} to "
        f"{mbv['mean_CI_high']:+.3f}); vs union TLOD>=8 {mu['estimate']:+.3f} ({mu['CI95_low']:+.3f} to {mu['CI95_high']:+.3f}; bootstrap "
        f"{mbu['mean_CI_low']:+.3f} to {mbu['mean_CI_high']:+.3f}); vs Strelka2 {ms['estimate']:+.3f} ({ms['CI95_low']:+.3f} to {ms['CI95_high']:+.3f}); "
        f"vs Mutect2 {mmu['estimate']:+.3f} ({mmu['CI95_low']:+.3f} to {mmu['CI95_high']:+.3f}). All four CIs exclude zero. SomaticXGB 0.6 vs 0.1 "
        f"F1 is {mx6['estimate']:+.3f}; the MixedLM CI ({mx6['CI95_low']:+.3f} to {mx6['CI95_high']:+.3f}) crosses zero because it pools a much larger "
        f"residual variance across methods, while the contrast-specific bootstrap CI ({mbx6['mean_CI_low']:+.3f} to {mbx6['mean_CI_high']:+.3f}) excludes zero; "
        "the three-way group x platform x depth cluster bootstrap also excludes zero (see R27_cluster_bootstrap_sensitivity.csv). We do not treat a "
        "0.016 F1 change as a substantive gain. All four baseline CIs also exclude zero under the three-way cluster bootstrap."
    )
    lines.append(
        "GIAB method recall/precision: SomaticXGB 0.1 minus fair-tuned VarScan2 SSC>=3 is "
        f"{-mvr['estimate']:+.3f} for recall (95% CI {-mvr['CI95_high']:+.3f} to {-mvr['CI95_low']:+.3f}) and "
        f"{-mvp['estimate']:+.3f} for precision (95% CI {-mvp['CI95_high']:+.3f} to {-mvp['CI95_low']:+.3f}), so it dominates that baseline on both axes "
        f"in this grid. Strelka2/union have slightly higher precision than SomaticXGB 0.1 (Strelka2 {msp['estimate']:+.3f}, CI {msp['CI95_low']:+.3f} to "
        f"{msp['CI95_high']:+.3f}) but much lower recall and F1; the SomaticXGB advantage is therefore mainly a recall/F1 advantage, not a precision advantage "
        "over the high-precision conventional callers."
    )
    lines.append(
        f"Twist: with only 3 replicate libraries, three of four physical-platform F1 paired-t 95% CIs span zero; GenoLab M is the only raw p<0.05 "
        "(difference -0.044, CI -0.082 to -0.006, raw p=0.038, BH q=0.104). In the hybrid, all four F1 CIs span zero. CI widths are "
        f"{tw_plat_txt}; exact Wilcoxon p is >=0.25 for every platform contrast, the smallest value possible with n=3. MixedLM/block-OLS intervals "
        "are narrower because they pool replicate x platform residual variance and should not be over-read. In the hybrid, SomaticXGB 0.1 minus "
        f"Strelka2 F1 is {-hyb_str['estimate']:+.3f} (paired-t 95% CI {-hyb_str['CI95_high']:+.3f} to {-hyb_str['CI95_low']:+.3f}) and minus VarScan2 is "
        f"{-hyb_var['estimate']:+.3f} (CI {-hyb_var['CI95_high']:+.3f} to {-hyb_var['CI95_low']:+.3f}); both exclude zero, but the exact Wilcoxon p is 0.25. "
        f"VarScan2 recall is slightly higher than SomaticXGB (VarScan2 - SomaticXGB = {hyb_var_rec['estimate']:+.3f}, CI {hyb_var_rec['CI95_low']:+.3f} to "
        f"{hyb_var_rec['CI95_high']:+.3f}), while SomaticXGB precision is much higher (SomaticXGB - VarScan2 = {-hyb_var_prec['estimate']:+.3f}, CI "
        f"{-hyb_var_prec['CI95_high']:+.3f} to {-hyb_var_prec['CI95_low']:+.3f})."
    )
    lines.append(
        f"Physical Twist 1%: SomaticXGB 0.1 minus Strelka2 F1 is {-phys_str['estimate']:+.3f} (paired-t 95% CI {-phys_str['CI95_high']:+.3f} to "
        f"{-phys_str['CI95_low']:+.3f}), not distinguishable from zero; minus VarScan2 it is {-phys_var['estimate']:+.3f} (CI {-phys_var['CI95_high']:+.3f} to "
        f"{-phys_var['CI95_low']:+.3f}). This is consistent with the manuscript's statement that the unmodified Twist 1% background makes the SomaticXGB and "
        "Strelka2_Mutect2 F1 values similar."
    )
    lines.append(
        f"Error rate vs F1 (platform level, n=6): Q30 Pearson r={e30['pearson_r']:.3f} (exact permutation p={e30['pearson_p_permutation']:.3f}), "
        f"Spearman rho={e30['spearman_rho']:.3f} (p={e30['spearman_p_permutation']:.3f}); Q0 Pearson r={e0['pearson_r']:.3f} (p={e0['pearson_p_permutation']:.3f}), "
        f"Spearman rho={e0['spearman_rho']:.3f} (p={e0['spearman_p_permutation']:.3f}). Excluding DNBSEQ-T7 (separate library), Q30 Pearson r={e30_5['pearson_r']:.3f} "
        f"(p={e30_5['pearson_p_permutation']:.3f}), Spearman rho={e30_5['spearman_rho']:.3f} (p={e30_5['spearman_p_permutation']:.3f}). "
        f"Leave-one-platform-out Pearson r ranges from {e30['loo_pearson_min']:.3f} to {e30['loo_pearson_max']:.3f} with all six platforms and from "
        f"{e30_5['loo_pearson_min']:.3f} to {e30_5['loo_pearson_max']:.3f} with the five TruSeq platforms. The direction is negative (lower error, higher F1) "
        "but six platforms are underpowered, the association is not robust to including T7, and this is not a causal test."
    )
    lines.append(
        "S11 audit: the 261112 workbook Table S11 contains 135 raw platform-method-metric values, no mean/SD/CI, and its header "
        f"({', '.join(str(x) for x in (s11_audit.get('header') or []))}) is misaligned with the data (first data row: "
        f"{s11_audit.get('first_data_row')}). It cannot support manuscript line 51's claim that 95% CIs are in S11. Replace S11 with the summary in "
        "R27_supp_table_draft.csv (two-way cluster-bootstrap CI over platforms and replicates, plus the replicate-level t CI) and say so in the text, or add "
        "a new S21 and cite that. Do not restore the old S11 naive 15-observation t CI, which treats non-independent platform-replicate observations as independent."
    )
    lines.append(
        "The same supplement renumbering affects manuscript citations S12 (TLOD+DP baseline), S13 (error rate vs Q) and S14 (individual Strelka2/Mutect2 results): "
        "those contents are no longer at those numbers in the 261112 workbook (S12/S13 are paired platform tables; S14 is a calling-method difference table). "
        "Reconcile all supplementary citations before resubmission; the S11 problem is not isolated."
    )
    return "\n".join(lines) + "\n"


def build_supp_draft(platform_mixed, platform_boot, method_mixed, method_boot, twist, err, s11) -> pd.DataFrame:
    rows = []
    # S11 replacement rows
    for _, r in s11.iterrows():
        rows.append(r.to_dict())
    # S21A platform mixed models (F1 primary; recall/precision for XGB0.1)
    for _, r in platform_mixed.iterrows():
        if r["threshold"] != "SomaticXGB 0.1" and r["metric"] != "F1":
            continue
        rows.append(
            {
                "proposed_table": "S21A (new)",
                "section": "GIAB platform MixedLM",
                "dataset": "GIAB",
                "analysis": "MixedLM, condition random intercept",
                "metric": r["metric"],
                "threshold": r["threshold"],
                "term": r["term"],
                "reference": r["reference"],
                "n_units": r["n_conditions"],
                "n_replicates": np.nan,
                "n_platforms": 6,
                "estimate": r["estimate"],
                "SE": r["SE"],
                "CI95_low": r["CI95_low"],
                "CI95_high": r["CI95_high"],
                "p": r["p"],
                "p_fdr": r["p_fdr"],
                "inference": "MixedLM (condition random intercept); Wald t CI",
                "notes": f"estimate_type={r['estimate_type']}; model mean={r['model_mean']:.4f}; observed mean={r['mean_observed']:.4f}",
            }
        )
    # S21B method mixed models
    for _, r in method_mixed.iterrows():
        rows.append(
            {
                "proposed_table": "S21B (new)",
                "section": "GIAB method MixedLM",
                "dataset": "GIAB",
                "analysis": "MixedLM, condition random intercept",
                "metric": r["metric"],
                "threshold": r["comparison_set"],
                "term": RULE_LABEL.get(r["term"], r["term"]),
                "reference": "SomaticXGB (0.1)",
                "n_units": r["n_conditions"],
                "n_replicates": np.nan,
                "n_platforms": 6,
                "estimate": r["estimate"],
                "SE": r["SE"],
                "CI95_low": r["CI95_low"],
                "CI95_high": r["CI95_high"],
                "p": r["p"],
                "p_fdr": r["p_fdr"],
                "inference": "MixedLM (condition random intercept); Wald t CI",
                "notes": f"estimate_type={r['estimate_type']}; negative estimate = method lower than SomaticXGB 0.1",
            }
        )
    # S21C paired bootstrap
    for _, r in platform_boot.iterrows():
        rows.append(
            {
                "proposed_table": "S21C (new)",
                "section": "GIAB platform paired bootstrap",
                "dataset": "GIAB",
                "analysis": "Paired cluster bootstrap (conditions)",
                "metric": r["metric"],
                "threshold": r["threshold"],
                "term": r["contrast"],
                "reference": "SURFSeq 5000",
                "n_units": r["n_units"],
                "n_replicates": np.nan,
                "n_platforms": 6,
                "estimate": r["mean_diff"],
                "SE": np.nan,
                "CI95_low": r["mean_CI_low"],
                "CI95_high": r["mean_CI_high"],
                "p": r["paired_t_p"],
                "p_fdr": np.nan,
                "inference": "Percentile cluster bootstrap over conditions; paired t p shown",
                "notes": f"median diff={r['median_diff']:+.4f} [{r['median_CI_low']:+.4f}, {r['median_CI_high']:+.4f}]; up/tie/down={r['n_up']}/{r['n_tie']}/{r['n_down']}",
            }
        )
    for _, r in method_boot.iterrows():
        if r["threshold"] != "primary (6 methods)":
            continue
        rows.append(
            {
                "proposed_table": "S21C (new)",
                "section": "GIAB method paired bootstrap",
                "dataset": "GIAB",
                "analysis": "Paired cluster bootstrap (conditions)",
                "metric": r["metric"],
                "threshold": r["threshold"],
                "term": RULE_LABEL.get(r["contrast"], r["contrast"]),
                "reference": "SomaticXGB (0.1)",
                "n_units": r["n_units"],
                "n_replicates": np.nan,
                "n_platforms": 6,
                "estimate": r["mean_diff"],
                "SE": np.nan,
                "CI95_low": r["mean_CI_low"],
                "CI95_high": r["mean_CI_high"],
                "p": r["paired_t_p"],
                "p_fdr": np.nan,
                "inference": "Percentile cluster bootstrap over conditions; paired t p shown",
                "notes": f"median diff={r['median_diff']:+.4f} [{r['median_CI_low']:+.4f}, {r['median_CI_high']:+.4f}]; negative = comparator lower than SomaticXGB 0.1",
            }
        )
    # S21D Twist paired-t CIs (primary)
    for _, r in twist.iterrows():
        if not bool(r.get("primary", False)):
            continue
        rows.append(
            {
                "proposed_table": "S21D (new)",
                "section": f"Twist {r['comparison_type']} paired t",
                "dataset": r["dataset"],
                "analysis": "Paired t (3 replicate libraries)",
                "metric": r["metric"],
                "threshold": "",
                "term": r["term"],
                "reference": "SURFSeq 5000" if r["comparison_type"] == "platform" else "SomaticXGB (0.1)",
                "n_units": 3,
                "n_replicates": 3,
                "n_platforms": 5,
                "estimate": r["estimate"],
                "SE": r["SE"],
                "CI95_low": r["CI95_low"],
                "CI95_high": r["CI95_high"],
                "p": r["p"],
                "p_fdr": r["p_fdr"],
                "inference": "Paired t, df=2; exact Wilcoxon p also available",
                "notes": f"CI width={r['CI95_width']:.4f}; Wilcoxon p={r['wilcoxon_p']}; up/tie/down={r.get('n_up', 'NA')}/{r.get('n_tie', 'NA')}/{r.get('n_down', 'NA')}",
            }
        )
    # S21E error-F1
    for _, r in err.iterrows():
        rows.append(
            {
                "proposed_table": "S21E (new)",
                "section": "Platform error rate vs F1",
                "dataset": "GIAB" if r["platform_set"] == "all6" else "GIAB TruSeq5",
                "analysis": "Platform-level correlation (n=6 or 5)",
                "metric": "F1 vs error rate",
                "threshold": f"Q{r['error_threshold']}",
                "term": r["f1_method"],
                "reference": "",
                "n_units": r["n_platforms"],
                "n_replicates": np.nan,
                "n_platforms": r["n_platforms"],
                "estimate": r["spearman_rho"],
                "SE": np.nan,
                "CI95_low": r["spearman_CI_low"],
                "CI95_high": r["spearman_CI_high"],
                "p": r["spearman_p_permutation"],
                "p_fdr": np.nan,
                "inference": "Spearman rho; exact permutation p; percentile bootstrap CI",
                "notes": f"Pearson r={r['pearson_r']:.3f} [Fisher {r['pearson_CI_low']:.3f}, {r['pearson_CI_high']:.3f}], exact p={r['pearson_p_permutation']:.3f}; n={r['n_platforms']}; not causal",
            }
        )
    out = pd.DataFrame(rows)
    return out


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def forest_plot(
    mixed: pd.DataFrame,
    boot: pd.DataFrame,
    metric: str,
    threshold: str,
    title: str,
    xlabel: str,
    outstem: str,
    reference: str,
    rows_order: list[str],
):
    m = mixed[
        (mixed["metric"] == metric)
        & (mixed["threshold"] == threshold)
        & (mixed["estimate_type"] == "difference")
    ].set_index("term")
    b = boot[
        (boot["metric"] == metric)
        & (boot["threshold"] == threshold)
    ].set_index("contrast")
    rows = [r for r in rows_order if r in m.index]
    if not rows:
        logger.warning("No rows for forest plot %s / %s", metric, threshold)
        return
    fig, ax = plt.subplots(figsize=(7.4, max(2.6, 0.55 * (len(rows) + 1))))
    ypos = np.arange(len(rows))[::-1]
    offset = 0.13
    for i, term in enumerate(rows):
        y = ypos[i]
        mr = m.loc[term]
        # MixedLM
        ax.errorbar(
            mr["estimate"], y + offset,
            xerr=[[mr["estimate"] - mr["CI95_low"]], [mr["CI95_high"] - mr["estimate"]]],
            fmt="o", color="#1f4e79", ecolor="#1f4e79", elinewidth=1.8,
            capsize=3.5, markersize=5.5, label="MixedLM (condition random intercept)" if i == 0 else None,
        )
        # Bootstrap
        if term in b.index:
            br = b.loc[term]
            ax.errorbar(
                br["mean_diff"], y - offset,
                xerr=[[br["mean_diff"] - br["mean_CI_low"]], [br["mean_CI_high"] - br["mean_diff"]]],
                fmt="s", mfc="white", mec="#c55a11", color="#c55a11", ecolor="#c55a11",
                elinewidth=1.8, capsize=3.5, markersize=5.5,
                label="Paired cluster bootstrap (conditions)" if i == 0 else None,
            )
    ax.axvline(0, color="black", lw=1, ls="--")
    ax.set_yticks(ypos)
    labels = []
    for term in rows:
        mr = m.loc[term]
        labels.append(f"{term}\n({mr['estimate']:+.3f}; p={fmt_p(mr['p'])})")
    ax.set_yticklabels(labels, fontsize=8.5)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_title(title, fontsize=10.5, loc="left")
    ax.tick_params(axis="x", labelsize=8.5)
    ax.grid(axis="x", color="0.9", lw=0.6)
    ax.set_axisbelow(True)
    ax.legend(fontsize=7.5, loc="lower left", frameon=False)
    # annotate direction
    ax.text(0.99, 0.02, "reference = " + reference, transform=ax.transAxes,
            ha="right", va="bottom", fontsize=7.5, color="0.35")
    fig.tight_layout()
    png = FIGS / f"{outstem}.png"
    pdf = FIGS / f"{outstem}.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    logger.info("Wrote %s and %s", png, pdf)


def make_figures(platform_mixed, platform_boot, method_mixed, method_boot):
    log_section("6. Figures")
    platform_order = ["DNBSEQ-T7", "FASTASeq 300", "GenoLab M", "NovaSeq 6000", "NovaSeq X"]
    method_order = ["strelka_default", "mutect2_tlod8", "union_tlod8", "varscan_ssc3", "xgb_pred06"]

    for metric in ["F1", "Recall", "Precision"]:
        forest_plot(
            platform_mixed, platform_boot, metric, "SomaticXGB 0.1",
            f"GIAB platform comparison - SomaticXGB 0.1 - {metric}",
            "Difference from SURFSeq 5000 (95% CI)",
            f"R27_forest_platform_{metric}",
            "SURFSeq 5000",
            platform_order,
        )
        forest_plot(
            method_mixed, method_boot, metric, "primary (6 methods)",
            f"GIAB method comparison - {metric}",
            "Difference from SomaticXGB 0.1 (95% CI); negative favours SomaticXGB 0.1",
            f"R27_forest_method_{metric}",
            "SomaticXGB 0.1",
            method_order,
        )

    # Combined 3-panel figures (requested filenames)
    for kind, mixed, boot, threshold, order, ref, stem, base_title in [
        ("platform", platform_mixed, platform_boot, "SomaticXGB 0.1", platform_order,
         "SURFSeq 5000", "R27_forest_platform", "GIAB platform comparison (SomaticXGB 0.1)"),
        ("method", method_mixed, method_boot, "primary (6 methods)", method_order,
         "SomaticXGB 0.1", "R27_forest_method", "GIAB method comparison"),
    ]:
        fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.2), sharey=False)
        for ax, metric in zip(axes, ["F1", "Recall", "Precision"]):
            m = mixed[
                (mixed["metric"] == metric)
                & (mixed["threshold"] == threshold)
                & (mixed["estimate_type"] == "difference")
            ].set_index("term")
            b = boot[
                (boot["metric"] == metric)
                & (boot["threshold"] == threshold)
            ].set_index("contrast")
            rows = [r for r in order if r in m.index]
            ypos = np.arange(len(rows))[::-1]
            off = 0.13
            for i, term in enumerate(rows):
                y = ypos[i]
                mr = m.loc[term]
                ax.errorbar(
                    mr["estimate"], y + off,
                    xerr=[[mr["estimate"] - mr["CI95_low"]], [mr["CI95_high"] - mr["estimate"]]],
                    fmt="o", color="#1f4e79", ecolor="#1f4e79", elinewidth=1.7,
                    capsize=3, markersize=5,
                )
                if term in b.index:
                    br = b.loc[term]
                    ax.errorbar(
                        br["mean_diff"], y - off,
                        xerr=[[br["mean_diff"] - br["mean_CI_low"]], [br["mean_CI_high"] - br["mean_diff"]]],
                        fmt="s", mfc="white", mec="#c55a11", color="#c55a11", ecolor="#c55a11",
                        elinewidth=1.7, capsize=3, markersize=5,
                    )
            ax.axvline(0, color="black", lw=1, ls="--")
            ax.set_yticks(ypos)
            if kind == "platform":
                labels = [f"{t}\n({m.loc[t, 'estimate']:+.3f}; p={fmt_p(m.loc[t, 'p'])})" for t in rows]
            else:
                labels = [f"{RULE_LABEL.get(t, t)}\n({m.loc[t, 'estimate']:+.3f}; p={fmt_p(m.loc[t, 'p'])})" for t in rows]
            ax.set_yticklabels(labels, fontsize=7.4)
            ax.set_title(metric, fontsize=10)
            ax.set_xlabel(f"Difference from {ref} (95% CI)", fontsize=8.2)
            ax.tick_params(axis="x", labelsize=7.5)
            ax.grid(axis="x", color="0.9", lw=0.6)
            ax.set_axisbelow(True)
        # legend handles
        from matplotlib.lines import Line2D
        handles = [
            Line2D([0], [0], marker="o", color="#1f4e79", lw=1.7, markersize=5, label="MixedLM (condition random intercept)"),
            Line2D([0], [0], marker="s", color="#c55a11", lw=1.7, markersize=5, mfc="white", label="Paired cluster bootstrap (conditions)"),
        ]
        fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, fontsize=8.5)
        fig.suptitle(base_title, fontsize=11.5, x=0.01, ha="left")
        if kind == "method":
            footer = f"Ref = {ref}; negative favours SomaticXGB 0.1."
        else:
            footer = f"Ref = {ref}; negative = lower."
        fig.text(0.01, 0.005, footer, fontsize=7.5, color="0.35")
        fig.tight_layout(rect=[0, 0.08, 1, 0.94])
        png = FIGS / f"{stem}.png"
        pdf = FIGS / f"{stem}.pdf"
        fig.savefig(png, dpi=300, bbox_inches="tight")
        fig.savefig(pdf, bbox_inches="tight")
        plt.close(fig)
        logger.info("Wrote %s and %s", png, pdf)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    t0 = time.time()
    log_section("R27 mixed models, paired/hierarchical inference and S11 audit")
    logger.info("Command: %s", " ".join(sys.argv))
    logger.info("Python: %s", sys.version.replace("\n", " "))
    logger.info("numpy=%s pandas=%s scipy=%s statsmodels=%s matplotlib=%s",
                np.__version__, pd.__version__, stats.__version__ if hasattr(stats, "__version__") else "n/a",
                sm.__version__ if hasattr(sm, "__version__") else "n/a", matplotlib.__version__)
    logger.info("Seed=%d, bootstrap replicates=%d, alpha=%.2f", SEED, N_BOOT, ALPHA)

    b1c = load_b1c()
    a3 = load_a3()
    phys = load_twist(TWIST_PHYS, "Physical Twist")
    hyb = load_twist(TWIST_HYB, "Hybrid Twist")

    rng = np.random.default_rng(SEED)

    platform_mixed, platform_boot, platform_cluster, platform_infos = analysis_platform(b1c, rng)
    method_mixed, method_boot, method_cluster, method_infos = analysis_method(b1c, rng)
    twist = analysis_twist(phys, hyb)
    err = analysis_error_f1(b1c, a3, rng)
    s11, s11_audit = audit_s11(hyb)

    # Write tables
    log_section("7. Writing tables")
    outputs = {}
    outputs["R27_platform_mixed_models.csv"] = platform_mixed
    outputs["R27_method_mixed_models.csv"] = method_mixed
    outputs["R27_paired_bootstrap.csv"] = pd.concat([platform_boot, method_boot], ignore_index=True)
    outputs["R27_cluster_bootstrap_sensitivity.csv"] = pd.concat([platform_cluster, method_cluster], ignore_index=True)
    outputs["R27_twist_ci.csv"] = twist
    outputs["R27_error_f1_relation.csv"] = err
    # The draft combines the S11 replacement and the new S21 sections.
    draft = build_supp_draft(platform_mixed, platform_boot, method_mixed, method_boot, twist, err, s11)
    outputs["R27_supp_table_draft.csv"] = draft
    for name, df in outputs.items():
        path = TABLES / name
        df.to_csv(path, index=False)
        logger.info("Wrote %s (%d rows x %d cols)", path, df.shape[0], df.shape[1])

    # Conclusion
    conclusion = build_conclusion(
        platform_mixed, platform_boot, method_mixed, method_boot, twist, err, s11_audit
    )
    concl_path = TABLES / "R27_conclusion.txt"
    concl_path.write_text(conclusion, encoding="utf-8")
    logger.info("Wrote %s", concl_path)
    logger.info("Conclusion:\n%s", conclusion)

    # Audit note
    audit_path = TABLES / "R27_s11_audit.json"
    audit_path.write_text(json.dumps(s11_audit, indent=2, default=str), encoding="utf-8")
    logger.info("Wrote %s", audit_path)

    make_figures(platform_mixed, platform_boot, method_mixed, method_boot)

    # Model convergence summary
    n_fail = sum(1 for x in platform_infos + method_infos if x.get("error") is not None)
    logger.info("MixedLM fits attempted: %d; failures: %d", len(platform_infos + method_infos), n_fail)
    logger.info("Twist MixedLM failures are recorded in R27_twist_ci.csv notes.")

    elapsed = time.time() - t0
    log_section("8. Done")
    logger.info("Elapsed: %.1f s", elapsed)
    logger.info("All outputs are under %s", ADD)
    logger.info("R27 completed successfully.")


if __name__ == "__main__":
    main()
