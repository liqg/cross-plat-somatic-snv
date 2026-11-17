#!/usr/bin/env python3

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
import common  # noqa: E402
# -*- coding: utf-8 -*-
"""
C2: PosMix false-positive provenance -- distance of FP calls to nearest simulated
target (spiked donor-private SNV) site.

For every method x group x platform:
  * calls are taken exactly as in scripts/make_fig3_data.py:
      Strelka2  : strelka/{sid}/results/variants/somatic.snvs.vcf.gz (PASS, 1-bp SNV)
      Mutect2   : mutect/{sid}.filter.vcf.gz                       (PASS, 1-bp SNV)
      union     : Strelka2 | Mutect2
      VarScan2  : varscan_novo/{sid}.snp.pass.vcf.gz              (PASS, 1-bp SNV)
      SomaticXGB: pred/{sid}_pred.txt.gz with pred > 0.1
  * pf{g}_low.txt distance rule copied from make_fig3_data.add_dist:
      dist >= 50 -> truth (scorable), dist < 50 -> near
  * calls exactly matching a near variant are removed (same as F1 calculation)
  * TP = call in truth, FP = call not in truth (and not near)
  * distance = bp to nearest truth site on the same chromosome; cross-chromosome
    calls are kept in the by-call table with distance=NaN and reported separately,
    never binned.

Null distribution: 200,000 positions drawn length-weighted from the (merged)
panel BED, stratified by chromosome, fixed seed, separately for each group.
Positions are sampled only on chromosomes that carry truth sites for that group
(chr18 has no truth site in group 54); this is equivalent to sampling the full
BED and dropping cross-chromosome points, which the spec excludes from bins.

Outputs (under response/addendum-analyses/):
  tables/C2_fp_distance_by_call.csv.gz
  tables/C2_fp_distance_bins.csv
  tables/C2_fp_distance_summary.csv
  tables/C2_fp_distance_pooled_method.csv
  tables/C2_fp_score_by_distance.csv
  tables/C2_fp_distance_localnull_sensitivity.csv   (extra: target-interval null)
  figures/C2_fp_distance_enrichment.png/.pdf
"""

import os
import re
import gzip
import math
import subprocess
import datetime as dt

import numpy as np
import pandas as pd
from scipy.stats import binomtest

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator

# ----------------------------------------------------------------------------- paths
BASE = str(common.WORK_ROOT)
ROOT = str(common.RESULTS_ROOT)
TAB = os.path.join(ROOT, "tables")
FIG = os.path.join(ROOT, "figures")
LOG = os.path.join(ROOT, "logs")
BED = str(common.REF_DIR / "panel" / "intersect_HG004_HG005_panel.sort.bed")

GROUPS = ["45", "54"]
SIDS = [
    "FASTASeq_300_reads100M",
    "GenoLab_M_reads100M",
    "MGISEQ-T7_reads100M",
    "NovaSeq_6000_reads100M",
    "NovaSeq_X_reads100M",
    "SURFSeq_5000_reads100M",
]
# same mapping as make_fig3_data.py
PLAT_MAP = {
    "FA": "FASTASeq 300", "GL": "GenoLab M", "NV": "NovaSeq 6000",
    "NX": "NovaSeq X", "SF": "SURFSeq 5000", "T7": "DNBSEQ-T7",
    "MGISEQ-T7": "DNBSEQ-T7",
}
METHODS = ["SomaticXGB 0.1", "Strelka2", "Mutect2", "Strelka2_Mutect2", "VarScan2_fpfilter"]

BINS = [
    ("0", 0, 1),
    ("1-50", 1, 50),
    ("50-100", 50, 100),
    ("100-150", 100, 150),
    ("150-250", 150, 250),
    ("250-500", 250, 500),
    ("500-1000", 500, 1000),
    ("1000-5000", 1000, 5000),
    (">5000", 5000, np.inf),
]
BIN_LABELS = [b[0] for b in BINS]
BIN_ORDER = {b: i for i, b in enumerate(BIN_LABELS)}
CUM_THRESHOLDS = [0, 50, 150, 300, 1000]
N_NULL = 200_000
SEED = 20261113

os.makedirs(TAB, exist_ok=True)
os.makedirs(FIG, exist_ok=True)
os.makedirs(LOG, exist_ok=True)


def log(msg):
    print(f"[{dt.datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


# ----------------------------------------------------------------------------- helpers
def add_dist(pf):
    """Exact copy of add_dist() in scripts/make_fig3_data.py."""
    df = pf.copy()
    df["CHROM"] = df.CHROM.astype(str)
    df["POS"] = pd.to_numeric(df.POS)
    d = []
    for chrom in df.CHROM.unique():
        sub = df[df.CHROM == chrom]
        pos = sub.POS.values
        n = len(pos)
        if n == 1:
            dd = [10000]
        elif n == 2:
            dd = [pos[1] - pos[0]] * 2
        else:
            d1 = np.r_[10000, pos[1:] - pos[:-1]]
            d2 = np.r_[pos[1:] - pos[:-1], 10000]
            dd = np.minimum(d1, d2)
        d.extend(dd)
    df["dist"] = d
    return df


def norm_chrom(c):
    """mvcfvars() uses c.replace('chr',''); do the same."""
    return str(c).replace("chr", "")


def vcfvars(path):
    """Exact rules of vcfvars() in scripts/make_fig3_data.py."""
    res = set()
    q = subprocess.run(
        ["bcftools", "query", "-f", "%CHROM\t%POS\t%REF\t%ALT\t%FILTER\n", path],
        capture_output=True, text=True,
    )
    if q.returncode != 0:
        raise RuntimeError(f"bcftools failed for {path}: {q.stderr}")
    for line in q.stdout.strip().split("\n"):
        if not line:
            continue
        c, p, r, a, flt = line.split("\t")
        if flt not in ("PASS", ".") and "PASS" not in flt:
            continue
        if len(r) == 1 and len(a) == 1 and "," not in a:
            res.add(c.replace("chr", "") + ":" + p + ":" + r + ":" + a)
    return res


def split_var(v):
    c, p, r, a = v.split(":")
    return c, int(p), r, a


def load_truth(group):
    pf = pd.read_csv(
        f"{BASE}/4.test/{group}/pf{group}_low.txt", sep="\t", header=None,
        names=["CHROM", "POS", "P", "REF", "ALT", "GT", "f", "b", "GT2", "group", "AF"],
    )
    pf = add_dist(pf)
    pf["CHROM"] = pf.CHROM.astype(str)
    pf["var"] = pf.CHROM + ":" + pf.POS.astype(str) + ":" + pf.REF + ":" + pf.ALT
    truth = set(pf.loc[pf.dist >= 50, "var"])
    near = set(pf.loc[pf.dist < 50, "var"])
    truth_df = pf.loc[pf.dist >= 50, ["CHROM", "POS", "REF", "ALT", "var"]].copy()
    assert truth_df["var"].is_unique
    return pf, truth, near, truth_df


def load_bed():
    b = pd.read_csv(BED, sep="\t", header=None, names=["chrom", "start", "end"])
    b["chrom"] = b.chrom.astype(str).map(norm_chrom)
    b = b.sort_values(["chrom", "start"]).reset_index(drop=True)
    # merge overlapping / touching intervals
    merged = []
    for c, sub in b.groupby("chrom", sort=False):
        s = sub.start.values
        e = sub.end.values
        cs, ce = int(s[0]), int(e[0])
        for ss, ee in zip(s[1:], e[1:]):
            ss, ee = int(ss), int(ee)
            if ss <= ce:
                ce = max(ce, ee)
            else:
                merged.append((c, cs, ce))
                cs, ce = ss, ee
        merged.append((c, cs, ce))
    m = pd.DataFrame(merged, columns=["chrom", "start", "end"])
    m["len"] = m.end - m.start
    return m


def chrom_truth_positions(truth_df):
    out = {}
    for c, sub in truth_df.groupby("CHROM", sort=False):
        out[str(c)] = np.sort(sub.POS.unique().astype(np.int64))
    return out


def nearest_distances(chroms, positions, ctp):
    """Distance (bp) to nearest truth position on the same chromosome.

    Returns (dist int64 array with -1 for chromosomes lacking truth, nearest
    truth POS array with -1, nearest truth var-index friend not returned).
    """
    chroms = np.asarray(chroms, dtype=object)
    positions = np.asarray(positions, dtype=np.int64)
    dist = np.full(len(positions), -1, dtype=np.int64)
    npos = np.full(len(positions), -1, dtype=np.int64)
    for c in np.unique(chroms):
        mask = chroms == c
        if c not in ctp:
            continue
        tp = ctp[c]
        idx = np.searchsorted(tp, positions[mask])
        left = np.clip(idx - 1, 0, len(tp) - 1)
        right = np.clip(idx, 0, len(tp) - 1)
        dl = np.abs(positions[mask] - tp[left])
        dr = np.abs(positions[mask] - tp[right])
        use_left = dl <= dr
        dd = np.where(use_left, dl, dr)
        nn = np.where(use_left, tp[left], tp[right])
        dist[mask] = dd
        npos[mask] = nn
    return dist, npos


def dist_bin(d):
    """Bin label for one distance (int >=0)."""
    for label, lo, hi in BINS:
        if lo <= d < hi:
            return label
    return ">5000"


def summarize_distances(dvals, n_total):
    """Return dict with counts, quantiles and cumulative proportions (valid dist only)."""
    d = np.asarray([x for x in dvals if not pd.isna(x)], dtype=np.int64)
    n = len(d)
    rec = {
        "n_fp": int(n_total),
        "n_fp_interchrom": int(n_total - n),
        "n_fp_dist_valid": int(n),
        "dist_median": float(np.median(d)) if n else np.nan,
        "dist_mean": float(np.mean(d)) if n else np.nan,
        "dist_q25": float(np.percentile(d, 25)) if n else np.nan,
        "dist_q75": float(np.percentile(d, 75)) if n else np.nan,
        "dist_min": int(d.min()) if n else np.nan,
        "dist_max": int(d.max()) if n else np.nan,
        "n_dist0": int((d == 0).sum()),
    }
    for t in CUM_THRESHOLDS:
        rec[f"n_le{t}"] = int((d <= t).sum())
        rec[f"prop_le{t}"] = float((d <= t).mean()) if n else np.nan
    return rec


def sample_positions(intervals, n, seed):
    """Length-weighted, chromosome-stratified random 1-based positions."""
    rng = np.random.default_rng(seed)
    lens = intervals.groupby("chrom")["len"].sum()
    p = lens / lens.sum()
    # exact chromosome-stratified allocation proportional to interval length
    raw = n * p.values
    counts = np.floor(raw).astype(int)
    rem = n - int(counts.sum())
    if rem > 0:
        counts[np.argsort(-(raw - counts))[:rem]] += 1
    frames = []
    for (c, n_c) in zip(lens.index.values, counts):
        if n_c == 0:
            continue
        sub = intervals[intervals.chrom == c]
        starts = sub.start.values.astype(np.int64)
        ends = sub.end.values.astype(np.int64)
        ilen = ends - starts
        cum = np.cumsum(ilen)
        u = rng.integers(0, int(cum[-1]), size=int(n_c))
        idx = np.searchsorted(cum, u, side="right")
        offset = u - (cum[idx] - ilen[idx])
        pos = starts[idx] + offset + 1  # BED 0-based -> 1-based
        frames.append(pd.DataFrame({"chrom": c, "pos1": pos}))
    out = pd.concat(frames, ignore_index=True)
    assert len(out) == n
    return out


def sample_null_positions(bed, truth_df, group, n=N_NULL, seed=SEED):
    """Length-weighted, chromosome-stratified random positions inside the BED.

    Only chromosomes with truth sites are used (see module docstring).
    Returns DataFrame(chrom, pos1).
    """
    ctp = chrom_truth_positions(truth_df)
    b = bed[bed.chrom.isin(ctp.keys())].copy()
    dropped = bed[~bed.chrom.isin(ctp.keys())]
    if len(dropped):
        log(f"  group {group}: null BED excludes {len(dropped)} intervals on "
            f"chromosomes without truth: {sorted(dropped.chrom.unique())} "
            f"({dropped.len.sum():,} bp of {bed.len.sum():,} BED bp)")
    return sample_positions(b, n, seed + int(group))


def target_intervals(bed, truth_df):
    """BED intervals that contain >=1 truth site."""
    bi = bed.copy()
    bi["has"] = False
    for _, r in truth_df.iterrows():
        m = ((bi.chrom == str(r["CHROM"])) &
             (bi.start <= int(r["POS"]) - 1) & (bi.end > int(r["POS"]) - 1))
        bi.loc[m, "has"] = True
    return bi[bi.has].copy().reset_index(drop=True)


def interval_membership(chroms, positions, intervals):
    """Boolean: is each 1-based position inside one of the intervals?"""
    chroms = np.asarray(chroms, dtype=str)
    positions = np.asarray(positions, dtype=np.int64)
    out = np.zeros(len(positions), dtype=bool)
    for c in np.unique(chroms):
        m = chroms == c
        sub = intervals[intervals.chrom == c]
        if len(sub) == 0:
            continue
        st = sub.start.values.astype(np.int64)
        en = sub.end.values.astype(np.int64)
        i = np.searchsorted(st, positions[m] - 1, side="right") - 1
        ok = i >= 0
        ii = np.clip(i, 0, len(st) - 1)
        ok &= (st[ii] <= positions[m] - 1) & (positions[m] - 1 < en[ii])
        out[m] = ok
    return out


def null_stats(bed, truth_df, group, ctp):
    null = sample_null_positions(bed, truth_df, group)
    d, _ = nearest_distances(null.chrom.values, null.pos1.values, ctp)
    assert (d >= 0).all(), "null position without same-chromosome truth"
    labels = np.array([dist_bin(int(x)) for x in d])
    bin_counts = pd.Series(labels).value_counts().reindex(BIN_LABELS, fill_value=0)
    rec = {
        "n_null": len(d),
        "null_median": float(np.median(d)),
        "null_mean": float(np.mean(d)),
        "null_q25": float(np.percentile(d, 25)),
        "null_q75": float(np.percentile(d, 75)),
    }
    for t in CUM_THRESHOLDS:
        rec[f"null_n_le{t}"] = int((d <= t).sum())
        rec[f"null_prop_le{t}"] = float((d <= t).mean())
    for lab in BIN_LABELS:
        rec[f"null_bin_{lab}"] = int(bin_counts[lab])
    return rec, null, d


def bh_fdr(p):
    p = np.asarray(p, dtype=float)
    out = np.full_like(p, np.nan)
    ok = ~np.isnan(p)
    pv = p[ok]
    n = len(pv)
    if n == 0:
        return out
    order = np.argsort(pv)
    ranked = pv[order]
    q = ranked * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0, 1)
    tmp = np.empty(n)
    tmp[order] = q
    out[ok] = tmp
    return out


# ----------------------------------------------------------------------------- run
def main():
    log("C2: FP distance-to-nearest-target analysis")
    log(f"BED: {BED}")
    if not os.path.exists(BED):
        raise SystemExit(f"panel BED not found: {BED}")
    bed = load_bed()
    log(f"BED intervals (merged): {len(bed)}, total bp: {bed.len.sum():,}, "
        f"chromosomes: {bed.chrom.nunique()}")

    # group-level truth / null
    group_data = {}
    for g in GROUPS:
        pf, truth, near, truth_df = load_truth(g)
        ctp = chrom_truth_positions(truth_df)
        log(f"group {g}: pf records={len(pf)}, truth(dist>=50)={len(truth)}, "
            f"near(dist<50)={len(near)}, truth chromosomes={len(ctp)}")
        nstat, null_df, null_d = null_stats(bed, truth_df, g, ctp)
        log(f"  null n={nstat['n_null']:,}, median distance={nstat['null_median']:.0f} bp, "
            f"<=150bp={nstat['null_prop_le150']:.4f}, <=300bp={nstat['null_prop_le300']:.4f}, "
            f"<=1000bp={nstat['null_prop_le1000']:.4f}")

        # extra sensitivity null: only BED intervals that contain >=1 truth site,
        # i.e. conditional on being in a target-dense interval
        ti = target_intervals(bed, truth_df)
        ti = ti[ti.chrom.isin(ctp.keys())].copy()
        ti["len"] = ti.end - ti.start
        local_null = sample_positions(ti, N_NULL, SEED + int(g) + 500)
        ld, _ = nearest_distances(local_null.chrom.values, local_null.pos1.values, ctp)
        assert (ld >= 0).all()
        local_stat = {
            "n_intervals": int(len(ti)),
            "bp": int(ti.len.sum()),
            "bp_frac": float(ti.len.sum() / bed.len.sum()),
            "n_null": int(len(ld)),
            "median": float(np.median(ld)),
            "q25": float(np.percentile(ld, 25)),
            "q75": float(np.percentile(ld, 75)),
        }
        for t in CUM_THRESHOLDS:
            local_stat[f"prop_le{t}"] = float((ld <= t).mean())
        log(f"  local-null (target-containing intervals): intervals={local_stat['n_intervals']}, "
            f"bp={local_stat['bp']:,} ({local_stat['bp_frac']:.1%}), median={local_stat['median']:.0f}bp, "
            f"<=150bp={local_stat['prop_le150']:.4f}, <=300bp={local_stat['prop_le300']:.4f}")
        group_data[g] = dict(pf=pf, truth=truth, near=near, truth_df=truth_df,
                             ctp=ctp, null=nstat, null_df=null_df, null_d=null_d,
                             target_intervals=ti, local_null=local_stat, local_null_d=ld)

    by_call_rows = []
    summary_rows = []
    bin_rows = []

    for g in GROUPS:
        gd = group_data[g]
        truth, near, ctp, nstat = gd["truth"], gd["near"], gd["ctp"], gd["null"]
        for sid in SIDS:
            plat = PLAT_MAP.get(sid.split("_reads")[0], sid.split("_reads")[0])
            s = vcfvars(f"{BASE}/4.test/{g}/strelka/{sid}/results/variants/somatic.snvs.vcf.gz")
            m = vcfvars(f"{BASE}/4.test/{g}/mutect/{sid}.filter.vcf.gz")
            v = vcfvars(f"{BASE}/4.test/{g}/varscan_novo/{sid}.snp.pass.vcf.gz")
            pred = pd.read_csv(
                f"{BASE}/4.test/{g}/pred/{sid}_pred.txt.gz", sep="\t", compression="gzip"
            )
            x = set(pred.loc[pred.pred > 0.1, "var"])
            call_sets = [
                ("SomaticXGB 0.1", x),
                ("Strelka2", s),
                ("Mutect2", m),
                ("Strelka2_Mutect2", s | m),
                ("VarScan2_fpfilter", v),
            ]
            log(f"group {g} / {plat}: raw calls raw Strelka2={len(s)}, Mutect2={len(m)}, "
                f"union={len(s | m)}, VarScan2={len(v)}, SomaticXGB>0.1={len(x)}")

            for method, calls_raw in call_sets:
                calls = set(calls_raw) - near          # same as make_fig3_data.py
                tp = len(calls & truth)
                fp_vars = sorted(calls - truth)
                n_fp = len(fp_vars)

                if fp_vars:
                    parts = [split_var(vv) for vv in fp_vars]
                    fchrom = [p[0] for p in parts]
                    fpos = np.array([p[1] for p in parts], dtype=np.int64)
                    d, npos = nearest_distances(fchrom, fpos, ctp)
                else:
                    d = np.array([], dtype=np.int64)
                    npos = np.array([], dtype=np.int64)

                # by-call table
                for i, vv in enumerate(fp_vars):
                    c, p, r, a = parts[i]
                    dd = int(d[i]) if d[i] >= 0 else np.nan
                    by_call_rows.append({
                        "Method": method,
                        "Group": g,
                        "Platform": plat,
                        "var": vv,
                        "CHROM": c,
                        "POS": p,
                        "REF": r,
                        "ALT": a,
                        "distance": dd,
                        "dist_bin": dist_bin(dd) if not pd.isna(dd) else "interchrom_NA",
                        "interchrom": bool(pd.isna(dd)),
                        "nearest_truth_POS": int(npos[i]) if npos[i] >= 0 else np.nan,
                    })

                dvals = d.astype(float)
                dvals[d < 0] = np.nan
                rec = summarize_distances(dvals, n_fp)
                rec.update({
                    "Method": method, "Group": g, "Platform": plat,
                    "n_calls_after_near_filter": len(calls),
                    "n_tp": tp,
                })
                for t in CUM_THRESHOLDS:
                    nk, nnull = rec[f"n_le{t}"], nstat[f"null_n_le{t}"]
                    pp = rec[f"prop_le{t}"]
                    p0 = nstat[f"null_prop_le{t}"]
                    rec[f"null_prop_le{t}"] = p0
                    rec[f"enrich_le{t}"] = pp / p0 if p0 > 0 else np.nan
                    if rec["n_fp_dist_valid"] > 0 and p0 > 0:
                        rec[f"p_binom_le{t}"] = float(binomtest(
                            nk, rec["n_fp_dist_valid"], p0, alternative="two-sided"
                        ).pvalue)
                    else:
                        rec[f"p_binom_le{t}"] = np.nan
                summary_rows.append(rec)

                # bins table (valid distances only)
                valid_d = d[d >= 0]
                labels = np.array([dist_bin(int(xx)) for xx in valid_d])
                vc = pd.Series(labels).value_counts().reindex(BIN_LABELS, fill_value=0)
                for lab in BIN_LABELS:
                    k = int(vc[lab])
                    n = rec["n_fp_dist_valid"]
                    p0 = nstat[f"null_bin_{lab}"] / nstat["n_null"]
                    pp = k / n if n else np.nan
                    bin_rows.append({
                        "Method": method, "Group": g, "Platform": plat,
                        "bin": lab, "bin_order": BIN_ORDER[lab],
                        "fp_count": k,
                        "fp_prop": pp,
                        "n_fp_dist_valid": n,
                        "null_count": nstat[f"null_bin_{lab}"],
                        "null_prop": p0,
                        "n_null": nstat["n_null"],
                        "enrichment": (pp / p0) if (p0 > 0 and n > 0) else np.nan,
                    })

            # progress
            n_no_fp = sum(1 for r in summary_rows
                          if r["Group"] == g and r["Platform"] == plat and r["n_fp"] == 0)
            log(f"  platform done for group {g}: summary rows={len(summary_rows)}")

    by_call = pd.DataFrame(by_call_rows)
    summary = pd.DataFrame(summary_rows)
    bins = pd.DataFrame(bin_rows)

    # add nearest truth ALT match (informational; distance-0 rows necessarily mismatch)
    tv = {}
    for g in GROUPS:
        for _, row in group_data[g]["truth_df"].iterrows():
            tv[(g, row["CHROM"], int(row["POS"]))] = row["var"]
    if len(by_call):
        by_call["nearest_truth_var"] = [
            tv.get((r.Group, r.CHROM, int(r.nearest_truth_POS)), np.nan)
            if not pd.isna(r.nearest_truth_POS) else np.nan
            for r in by_call.itertuples()
        ]
        by_call["alt_match_nearest_truth"] = [
            (isinstance(nv, str)) and (nv.split(":")[3] == alt)
            for nv, alt in zip(by_call["nearest_truth_var"].tolist(),
                               by_call["ALT"].tolist())
        ]
    else:
        by_call["nearest_truth_var"] = []
        by_call["alt_match_nearest_truth"] = []

    # extra sensitivity: is the FP inside a BED interval that contains a truth site?
    by_call["in_target_interval"] = False
    for g in GROUPS:
        gd = group_data[g]
        idx = by_call.index[by_call.Group.astype(str) == g]
        if len(idx):
            by_call.loc[idx, "in_target_interval"] = interval_membership(
                by_call.loc[idx, "CHROM"].values,
                by_call.loc[idx, "POS"].values,
                gd["target_intervals"],
            )

    sens_rows = []
    for (method, group, plat), sub in by_call.groupby(
            ["Method", "Group", "Platform"], sort=False):
        group = str(group)
        gd = group_data[group]
        ln = gd["local_null"]
        valid = sub.distance.notna()
        s = sub[valid]
        n_valid = len(s)
        n_in = int(s.in_target_interval.sum()) if n_valid else 0
        rec = {
            "Method": method, "Group": group, "Platform": plat,
            "n_fp": len(sub),
            "n_fp_dist_valid": n_valid,
            "n_fp_in_target_intervals": n_in,
            "frac_fp_in_target_intervals": (n_in / n_valid) if n_valid else np.nan,
            "target_interval_bp_frac": ln["bp_frac"],
            "density_ratio_in_target_intervals":
                (n_in / n_valid / ln["bp_frac"]) if (n_valid and ln["bp_frac"] > 0) else np.nan,
        }
        for t in [50, 150, 300, 1000]:
            obs_global = float((s.distance <= t).mean()) if n_valid else np.nan
            p0 = ln[f"prop_le{t}"]
            obs_cond = float((s.loc[s.in_target_interval, "distance"] <= t).mean()) if n_in else np.nan
            k = int((s.loc[s.in_target_interval, "distance"] <= t).sum()) if n_in else 0
            rec[f"global_prop_le{t}"] = obs_global
            rec[f"localnull_prop_le{t}"] = p0
            rec[f"global_enrich_vs_localnull_le{t}"] = obs_global / p0 if n_valid else np.nan
            rec[f"cond_in_target_prop_le{t}"] = obs_cond
            rec[f"cond_enrich_le{t}"] = obs_cond / p0 if n_in else np.nan
            rec[f"cond_p_binom_le{t}"] = (
                float(binomtest(k, n_in, p0, alternative="two-sided").pvalue)
                if (n_in and 0 < p0 < 1) else np.nan)
        sens_rows.append(rec)
    sens = pd.DataFrame(sens_rows)
    if len(sens):
        for t in [50, 150, 300, 1000]:
            sens[f"cond_q_binom_le{t}"] = bh_fdr(sens[f"cond_p_binom_le{t}"].values)
        sens["_m"] = sens.Method.map({m: i for i, m in enumerate(METHODS)})
        sens["_g"] = sens.Group.map({"45": 0, "54": 1})
        sens["_p"] = sens.Platform.map(
            {PLAT_MAP.get(s.split("_reads")[0], s.split("_reads")[0]): i
             for i, s in enumerate(SIDS)})
        sens = sens.sort_values(["_m", "_g", "_p"]).drop(columns=["_m", "_g", "_p"]).reset_index(drop=True)

    # FDR on binomial p-values
    for t in CUM_THRESHOLDS:
        col = f"p_binom_le{t}"
        summary[f"q_binom_le{t}"] = bh_fdr(summary[col].values)

    # order columns
    sum_front = ["Method", "Group", "Platform", "n_calls_after_near_filter", "n_tp", "n_fp",
                 "n_fp_interchrom", "n_fp_dist_valid", "n_dist0"]
    sum_rest = [c for c in summary.columns if c not in sum_front]
    summary = summary[sum_front + sum_rest]
    summary["Method"] = pd.Categorical(summary["Method"], METHODS, ordered=True)
    summary["Group"] = pd.Categorical(summary["Group"], GROUPS, ordered=True)
    summary["Platform"] = pd.Categorical(summary["Platform"],
                                         [PLAT_MAP.get(s.split("_reads")[0], s.split("_reads")[0])
                                          for s in SIDS], ordered=True)
    summary = summary.sort_values(["Method", "Group", "Platform"]).reset_index(drop=True)
    summary["Method"] = summary["Method"].astype(str)
    summary["Group"] = summary["Group"].astype(str)
    summary["Platform"] = summary["Platform"].astype(str)

    bins["Method"] = pd.Categorical(bins["Method"], METHODS, ordered=True)
    bins["Group"] = pd.Categorical(bins["Group"], GROUPS, ordered=True)
    bins["Platform"] = pd.Categorical(bins["Platform"],
                                      [PLAT_MAP.get(s.split("_reads")[0], s.split("_reads")[0])
                                       for s in SIDS], ordered=True)
    bins = bins.sort_values(["Method", "Group", "Platform", "bin_order"]).reset_index(drop=True)
    for c in ["Method", "Group", "Platform"]:
        bins[c] = bins[c].astype(str)

    by_call["Method"] = pd.Categorical(by_call["Method"], METHODS, ordered=True)
    by_call["Group"] = pd.Categorical(by_call["Group"], GROUPS, ordered=True)
    by_call["Platform"] = pd.Categorical(by_call["Platform"],
                                         [PLAT_MAP.get(s.split("_reads")[0], s.split("_reads")[0])
                                          for s in SIDS], ordered=True)
    by_call = by_call.sort_values(["Method", "Group", "Platform", "distance", "var"]).reset_index(drop=True)
    for c in ["Method", "Group", "Platform"]:
        by_call[c] = by_call[c].astype(str)

    # ------------------------------------------------------------------ pooled by method
    pooled_rows = []
    for method in METHODS:
        sub = by_call[by_call.Method == method]
        d = sub["distance"].dropna().values
        # group-specific null proportions, weighted by the number of valid FP calls
        # contributed by each group
        w = {}
        for g in GROUPS:
            w[g] = int(((sub.Group == g) & sub.distance.notna()).sum())
        n_valid = int(len(d))
        rec = {"Method": method, "n_fp": int(len(sub)), "n_fp_interchrom": int(sub.distance.isna().sum()),
               "n_fp_dist_valid": n_valid,
               "dist_median": float(np.median(d)) if n_valid else np.nan,
               "dist_mean": float(np.mean(d)) if n_valid else np.nan,
               "dist_q25": float(np.percentile(d, 25)) if n_valid else np.nan,
               "dist_q75": float(np.percentile(d, 75)) if n_valid else np.nan,
               "n_dist0": int((d == 0).sum()) if n_valid else 0}
        for t in [50, 150, 300, 1000]:
            k = int((d <= t).sum()) if n_valid else 0
            p0 = sum(w[g] * group_data[g]["null"][f"null_prop_le{t}"] for g in GROUPS) / n_valid \
                if n_valid else np.nan
            pp = k / n_valid if n_valid else np.nan
            rec[f"n_le{t}"] = k
            rec[f"prop_le{t}"] = pp
            rec[f"null_prop_le{t}"] = p0
            rec[f"enrich_le{t}"] = pp / p0 if (n_valid and p0 and p0 > 0) else np.nan
            rec[f"p_binom_le{t}"] = float(binomtest(k, n_valid, p0, alternative="two-sided").pvalue) \
                if (n_valid and p0 and p0 > 0) else np.nan
        rec["p_binom_le150_fdr"] = np.nan  # filled below across methods
        rec["p_binom_le300_fdr"] = np.nan
        pooled_rows.append(rec)
    pooled = pd.DataFrame(pooled_rows)
    pooled["p_binom_le150_fdr"] = bh_fdr(pooled["p_binom_le150"].values)
    pooled["p_binom_le300_fdr"] = bh_fdr(pooled["p_binom_le300"].values)
    pooled["Method"] = pd.Categorical(pooled["Method"], METHODS, ordered=True)
    pooled = pooled.sort_values("Method").reset_index(drop=True)
    pooled["Method"] = pooled["Method"].astype(str)

    # ------------------------------------------------------------------ extra: score vs distance
    score_frames = []
    for g in GROUPS:
        for sid in SIDS:
            plat = PLAT_MAP.get(sid.split("_reads")[0], sid.split("_reads")[0])
            p = pd.read_csv(f"{BASE}/4.test/{g}/pred/{sid}_pred.txt.gz",
                            sep="\t", compression="gzip")
            keep = [c for c in ["var", "pred", "avg_pos_as_fraction", "avg_basequality",
                                "avg_mapping_quality", "DP", "ECNT", "TLOD", "NALOD"] if c in p.columns]
            p = p[keep].copy()
            p["Group"] = g
            p["Platform"] = plat
            score_frames.append(p)
    pred_all = pd.concat(score_frames, ignore_index=True)
    pred_all = pred_all.drop_duplicates(["Group", "Platform", "var"])

    fp_ann = by_call.merge(pred_all, on=["Group", "Platform", "var"], how="left")
    fp_ann["dist_class"] = np.where(fp_ann.distance.isna(), "interchrom_NA",
                                    np.where(fp_ann.distance <= 300, "<=300bp", ">300bp"))
    extra_rows = []
    for (method, group, plat), sub in fp_ann.groupby(["Method", "Group", "Platform"], sort=False):
        for cls in ["<=300bp", ">300bp", "interchrom_NA"]:
            s = sub[sub.dist_class == cls]
            if len(s) == 0:
                continue
            rec = {"Method": method, "Group": group, "Platform": plat,
                   "dist_class": cls, "n_fp": len(s),
                   "n_pred_annotated": int(s["pred"].notna().sum()),
                   "frac_pred_annotated": float(s["pred"].notna().mean())}
            for col in ["pred", "avg_pos_as_fraction", "avg_basequality",
                        "avg_mapping_quality", "DP", "ECNT", "TLOD", "NALOD"]:
                if col in s.columns:
                    rec[f"median_{col}"] = float(s[col].median()) if s[col].notna().any() else np.nan
            extra_rows.append(rec)
    extra = pd.DataFrame(extra_rows)
    if len(extra):
        morder = {m: i for i, m in enumerate(METHODS)}
        porder = {PLAT_MAP.get(s.split("_reads")[0], s.split("_reads")[0]): i for i, s in enumerate(SIDS)}
        extra["_m"] = extra.Method.map(morder)
        extra["_g"] = extra.Group.astype(str).map({"45": 0, "54": 1})
        extra["_p"] = extra.Platform.map(porder)
        extra["_c"] = extra.dist_class.map({"<=300bp": 0, ">300bp": 1, "interchrom_NA": 2})
        extra = extra.sort_values(["_m", "_g", "_p", "_c"]).drop(columns=["_m", "_g", "_p", "_c"])
        extra = extra.reset_index(drop=True)

    # ------------------------------------------------------------------ write tables
    p_call = os.path.join(TAB, "C2_fp_distance_by_call.csv.gz")
    by_call.to_csv(p_call, index=False, compression="gzip")
    p_bins = os.path.join(TAB, "C2_fp_distance_bins.csv")
    bins.to_csv(p_bins, index=False)
    p_sum = os.path.join(TAB, "C2_fp_distance_summary.csv")
    summary.to_csv(p_sum, index=False)
    p_pool = os.path.join(TAB, "C2_fp_distance_pooled_method.csv")
    pooled.to_csv(p_pool, index=False)
    p_extra = os.path.join(TAB, "C2_fp_score_by_distance.csv")
    if len(extra):
        extra.to_csv(p_extra, index=False)
    p_sens = os.path.join(TAB, "C2_fp_distance_localnull_sensitivity.csv")
    if len(sens):
        sens.to_csv(p_sens, index=False)

    log(f"wrote {p_call} ({len(by_call)} FP calls)")
    log(f"wrote {p_bins} ({len(bins)} rows)")
    log(f"wrote {p_sum} ({len(summary)} method x group x platform rows)")
    log(f"wrote {p_pool}")
    if len(extra):
        log(f"wrote {p_extra}")
    if len(sens):
        log(f"wrote {p_sens}")

    # ------------------------------------------------------------------ figure
    make_figure(by_call, group_data, pooled)

    # ------------------------------------------------------------------ console summary
    log("\n=== Pooled across groups/platforms ===")
    show = pooled[["Method", "n_fp", "n_fp_interchrom", "dist_median",
                   "prop_le150", "null_prop_le150", "enrich_le150",
                   "prop_le300", "null_prop_le300", "enrich_le300",
                   "prop_le1000", "null_prop_le1000", "enrich_le1000",
                   "p_binom_le300", "p_binom_le300_fdr"]].copy()
    show.columns = ["Method", "n_FP", "interchrom", "median_bp",
                    "FP<=150", "null<=150", "ER150",
                    "FP<=300", "null<=300", "ER300",
                    "FP<=1000", "null<=1000", "ER1000",
                    "p300", "q300"]
    log("\n" + show.to_string(index=False, float_format=lambda x: f"{x:.4g}"))

    log("\n=== FP counts (method x group x platform) ===")
    pivot = summary.pivot_table(index=["Method", "Group"], columns="Platform",
                                values="n_fp", aggfunc="sum")
    log("\n" + pivot.to_string())

    log("\n=== Null reference (200k positions per group) ===")
    for g in GROUPS:
        ns = group_data[g]["null"]
        log(f"group {g}: median={ns['null_median']:.0f}bp, mean={ns['null_mean']:.0f}bp, "
            f"q25={ns['null_q25']:.0f}, q75={ns['null_q75']:.0f}, "
            f"<=50={ns['null_prop_le50']:.4f}, <=150={ns['null_prop_le150']:.4f}, "
            f"<=300={ns['null_prop_le300']:.4f}, <=1000={ns['null_prop_le1000']:.4f}")

    log("\n=== Bin enrichment summary (FP/null, pooled methods per group) ===")
    for g in GROUPS:
        sub = bins[bins.Group == g]
        agg = sub.groupby("bin", sort=False)[["fp_count"]].sum()
        n = agg.fp_count.sum()
        agg["fp_prop"] = agg.fp_count / n
        agg["null_prop"] = [group_data[g]["null"][f"null_bin_{lab}"] /
                            group_data[g]["null"]["n_null"] for lab in agg.index]
        agg["enrichment"] = agg.fp_prop / agg.null_prop
        agg = agg.reindex(BIN_LABELS)
        log(f"\ngroup {g} (n FP with valid distance = {n}):")
        log(agg.to_string(float_format=lambda x: f"{x:.4g}"))

    log("\n=== Local-null sensitivity: conditional on target-containing BED intervals ===")
    for g in GROUPS:
        ln = group_data[g]["local_null"]
        log(f"group {g}: target intervals cover {ln['bp_frac']:.1%} of panel bp; "
            f"local null <=150={ln['prop_le150']:.3f}, <=300={ln['prop_le300']:.3f}")
        for method in METHODS:
            ss = sens[(sens.Group == g) & (sens.Method == method)]
            if len(ss) == 0:
                continue
            n_valid = int(ss.n_fp_dist_valid.sum())
            n_in = int(ss.n_fp_in_target_intervals.sum())
            if not n_valid:
                continue
            line = (f"  {method:20s} FPs={n_valid:5d} in_target={n_in/n_valid:.3f} "
                    f"(density x{n_in/n_valid/ln['bp_frac']:.2f})")
            for t in [150, 300]:
                k = int((ss.n_fp_in_target_intervals *
                         ss[f"cond_in_target_prop_le{t}"]).round().sum())
                cond = k / n_in if n_in else np.nan
                p0 = ln[f"prop_le{t}"]
                p = binomtest(k, n_in, p0, alternative="two-sided").pvalue if n_in else np.nan
                line += f" | cond<={t}: {cond:.3f} vs {p0:.3f} (ER {cond/p0:.2f}, p={p:.2g})"
            log(line)

    log("\nDone.")


def make_figure(by_call, group_data, pooled):
    method_colors = {
        "SomaticXGB 0.1": "#1f77b4",
        "Strelka2": "#ff7f0e",
        "Mutect2": "#2ca02c",
        "Strelka2_Mutect2": "#d62728",
        "VarScan2_fpfilter": "#9467bd",
    }
    markers = {"SomaticXGB 0.1": "o", "Strelka2": "s", "Mutect2": "^",
               "Strelka2_Mutect2": "D", "VarScan2_fpfilter": "v"}
    # for plotting, merge distance 0 and 1-49 into one 0-50 category
    # (the 0 bin can be a single call; both are tabulated separately in the CSVs)
    fig_bins = ["0-50", "50-100", "100-150", "150-250", "250-500",
                "500-1000", "1000-5000", ">5000"]

    def fig_bin_counts(frame):
        c = frame.dist_bin.value_counts()
        out = pd.Series(0, index=fig_bins, dtype=int)
        out["0-50"] = int(c.get("0", 0) + c.get("1-50", 0))
        for lab in fig_bins[1:]:
            out[lab] = int(c.get(lab, 0))
        return out

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for row, g in enumerate(GROUPS):
        sub = by_call[by_call.Group == g]
        ns = group_data[g]["null"]
        # left: enrichment by distance bin (methods pooled over platforms)
        ax = axes[row, 0]
        x = np.arange(len(fig_bins))
        for method in METHODS:
            msub = sub[(sub.Method == method) & sub.distance.notna()]
            if len(msub) == 0:
                continue
            counts = fig_bin_counts(msub)
            props = counts / counts.sum()
            nullp = np.array([
                (ns["null_bin_0"] + ns["null_bin_1-50"]) / ns["n_null"]
                if lab == "0-50" else ns[f"null_bin_{lab}"] / ns["n_null"]
                for lab in fig_bins
            ])
            er = np.where(nullp > 0, props.values / nullp, np.nan)
            ax.plot(x, er, marker=markers[method], color=method_colors[method],
                    lw=1.6, ms=5, label=f"{method} (n={len(msub)})")
        ax.axhline(1, color="grey", ls="--", lw=1)
        ax.set_xticks(x)
        ax.set_xticklabels(fig_bins, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("Enrichment ratio (FP / null)")
        ax.set_title(f"Group {g}: FP distance-bin enrichment", fontsize=11)
        ax.legend(fontsize=7, frameon=False, ncol=2)
        ax.grid(alpha=0.25, lw=0.5)
        ax.set_ylim(bottom=0)

        # right: CDF of distance to nearest truth vs null
        ax = axes[row, 1]
        grid = np.unique(np.concatenate([
            np.logspace(0, 7.5, 400),
            np.array([50, 100, 150, 250, 300, 500, 1000, 5000], dtype=float),
        ]))
        for method in METHODS:
            msub = sub[(sub.Method == method) & sub.distance.notna()]
            if len(msub) == 0:
                continue
            d = msub.distance.values
            cdf = np.array([(d <= t).mean() for t in grid])
            ax.plot(grid, cdf, color=method_colors[method], lw=1.5,
                    label=f"{method} (n={len(d)})")
        null_d = group_data[g]["null_d"]
        null_cdf = np.array([(null_d <= t).mean() for t in grid])
        ax.plot(grid, null_cdf, color="black", ls="--", lw=2.0, label="null (panel)")
        for t in [150, 300, 1000]:
            ax.axvline(t, color="grey", ls=":", lw=0.8)
        ax.set_xscale("log")
        ax.set_xlabel("Distance to nearest truth site (bp, same chromosome)")
        ax.set_ylabel("Cumulative proportion of FPs within D")
        ax.set_title(f"Group {g}: CDF vs null", fontsize=11)
        ax.set_xlim(1, 3e7)
        ax.set_ylim(0, 1.02)
        ax.legend(fontsize=7, frameon=False, loc="lower right")
        ax.grid(alpha=0.25, lw=0.5, which="both")
        ax.xaxis.set_major_locator(LogLocator(base=10, numticks=10))

    fig.suptitle("C2  PosMix false positives: proximity to spiked donor target sites\n"
                 "(FPs exclude calls matching truth or near(<50bp) target variants; "
                 "null = length-weighted random panel positions)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    for ext in ["png", "pdf"]:
        p = os.path.join(FIG, f"C2_fp_distance_enrichment.{ext}")
        fig.savefig(p, dpi=300, bbox_inches="tight")
        log(f"wrote {p}")
    plt.close(fig)


if __name__ == "__main__":
    main()
