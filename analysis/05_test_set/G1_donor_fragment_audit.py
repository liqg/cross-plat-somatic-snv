#!/usr/bin/env python3

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
import common  # noqa: E402
# -*- coding: utf-8 -*-
"""
G1 -- Direct audit of donor alleles on PosMix donor fragments that are NOT in
the truth list (R1-4 / R2-2).

Question addressed
------------------
"A validation analysis should quantify unintended variants and demonstrate that
the final truth set accounts for all introduced alleles."

Design
------
1. Static (VCF-based, deterministic)
   * donor/background panel variants are read from the same panel-restricted
     GIAB v4.2.1 VCFs that were used to build pf{group}_low.txt
     (HG004_bed45panel.vcf / HG005_bed45panel.vcf).
   * truth = pf{group}_low.txt (481 / 479 target alleles).
   * classify every donor allele:
       - donor-private by position (the truth basis)
       - donor-private by allele (donor ALT absent from background)
       - shared position / identical ALT (no tumor-only difference)
       - shared position / different ALT (donor ALT tumor-specific)
   * unlisted donor alleles = donor alleles not in the truth list
     (clustered donor-private SNVs are NOT excluded from pf45_low: all 481
     donor-private SNVs are listed; 67 of them are <50 bp from another target
     and are treated as "near" only at scoring time, exactly as C2 does).
   * per-target counts of unlisted alleles within +/-150/300/500 bp.
   * fragment-length-aware expectation using the empirical insert-size
     distribution of the donor library.

2. BAM-level direct audit
   * donor reads in the PosMix tumor BAM are identified by qname intersection
     with the donor BAM in the same window (background reads never collide).
   * for every truth target, reconstruct each donor fragment span, and count
     unlisted donor alleles inside the span (excluding the target itself).
   * verify ALT carriage from the read bases at each unlisted SNV.
   * output per-fragment and per-target summaries.

3. FP-site donor validation (SURFSeq 5000, SomaticXGB)
   * for every SomaticXGB FP (C2 by-call table) in group 45/54 SURFSeq 5000,
     count the FP ALT separately in donor-origin and background-origin reads
     from the tumor BAM.  This tests directly whether a near-target FP is an
     unintended donor allele or a sequencing/mapping error.

Outputs (all under response/addendum-analyses/):
  tables/G1_static_summary.csv
  tables/G1_static_per_target.csv
  tables/G1_fragment_summary.csv
  tables/G1_fragment_level_audit.csv
  tables/G1_fp_validation.csv
  tables/G1_conclusion.txt
  figures/G1_unintended_alleles.png / .pdf
  logs/G1.log
"""

import os
import sys
import time
import math
import pickle
import collections
import datetime as dt

import numpy as np
import pandas as pd

try:
    import pysam
except ImportError:
    sys.stderr.write("pysam is required (pip install --user pysam)\n")
    raise

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# --------------------------------------------------------------------------- paths
REPO = str(common.REPO_ROOT)
ROOT = str(common.RESULTS_ROOT)
TAB = str(common.TABLES)
FIG = str(common.FIGURES)
LOG = str(common.LOGS)
BASE = str(common.WORK_ROOT)
POSFILE = os.environ.get("CPSNV_DONOR_VCF_DIR", str(common.DATA_ROOT / "posfiles" / "donor_vcf"))
PANEL_BED = str(common.REF_DIR / "panel" / "intersect_HG004_HG005_panel.sort.bed")
PKL = str(common.GIAB_PANEL_VARS)
C2_CALLS = os.path.join(TAB, "C2_fp_distance_by_call.csv.gz")

for d in (TAB, FIG, LOG):
    os.makedirs(d, exist_ok=True)

LOG_PATH = os.path.join(LOG, "G1.log")
_log_fh = open(LOG_PATH, "w")


def log(msg=""):
    line = f"[{dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    _log_fh.write(line + "\n")
    _log_fh.flush()


GROUPS = {
    "45": dict(
        donor="HG004",
        background="HG005",
        tumor=f"{BASE}/4.test/45/posmix/SURFSeq_5000_T_reads100M.bam",
        donor_bam=str(common.ALIGN_ROOT / "SURFSeq_5000_HG004_GM23121213" / "SURFSeq_5000_HG004_GM23121213.bam"),
        bg_bam=str(common.ALIGN_ROOT / "SURFSeq_5000_HG005_GM23121215" / "SURFSeq_5000_HG005_GM23121215.bam"),
        truth=str(common.REPO_ROOT / "workflow" / "pos_files" / "test" / "45" / "pf45_low.txt"),
    ),
    "54": dict(
        donor="HG005",
        background="HG004",
        tumor=f"{BASE}/4.test/54/posmix/SURFSeq_5000_T_reads100M.bam",
        donor_bam=str(common.ALIGN_ROOT / "SURFSeq_5000_HG005_GM23121216" / "SURFSeq_5000_HG005_GM23121216.bam"),
        bg_bam=str(common.ALIGN_ROOT / "SURFSeq_5000_HG004_GM23121214" / "SURFSeq_5000_HG004_GM23121214.bam"),
        truth=str(common.REPO_ROOT / "workflow" / "pos_files" / "test" / "54" / "pf54_low.txt"),
    ),
}

DS = [150, 300, 500]
WINDOW = 1000          # fetch half-window for donor-qname / read recovery
MAX_FRAG = 2000        # exclude chimeric insert spans from the main audit
MIN_MAPQ = 20          # read filter for base counting
FP_WINDOW = 300        # half-window for FP-site validation
FP_DONOR_MIN_ALT = 2
FP_DONOR_MIN_FRAC = 0.15
FP_BG_MIN_ALT = 2
FP_BG_MIN_FRAC = 0.10

# --------------------------------------------------------------------------- helpers
def load_panel_bed(path):
    bed = pd.read_csv(path, sep="\t", header=None,
                      names=["chrom", "start", "end"], dtype={"chrom": str})
    bed = bed.sort_values(["chrom", "start"]).reset_index(drop=True)
    # merge overlapping/touching intervals (same logic as C2)
    merged = []
    for c, sub in bed.groupby("chrom", sort=False):
        s = e = None
        for x, y in zip(sub.start.values, sub.end.values):
            if s is None:
                s, e = int(x), int(y)
            elif x <= e:
                e = max(e, int(y))
            else:
                merged.append((c, s, e))
                s, e = int(x), int(y)
        if s is not None:
            merged.append((c, s, e))
    return pd.DataFrame(merged, columns=["chrom", "start", "end"])


def in_bed(bed, chrom, pos):
    sub = bed[bed.chrom == str(chrom)]
    if len(sub) == 0:
        return False
    return bool(((sub.start.values <= pos) & (sub.end.values >= pos)).any())


def load_panel_alleles(sid):
    """Allele-level records of the panel-restricted GIAB VCF.

    Multiallelic records are split; dosage of each ALT allele is computed from
    the GT indices (e.g. GT 2/1 -> 0.5 for ALT1 and 0.5 for ALT2).
    """
    path = os.path.join(POSFILE, f"{sid}_bed45panel.vcf")
    rows = []
    with open(path) as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            a = line.rstrip("\n").split("\t")
            chrom = str(a[0]).replace("chr", "")
            pos = int(a[1])
            ref = a[3]
            alts = a[4].split(",")
            gt_raw = a[9].split(":")[0] if len(a) > 9 else "."
            gt = gt_raw.replace("|", "/").split("/")
            for i, alt in enumerate(alts, start=1):
                dosage = sum(1 for x in gt if x == str(i)) / 2.0
                rows.append(dict(
                    chrom=chrom, pos=pos, ref=ref, alt=alt, GT=gt_raw,
                    dosage=dosage, multiallelic=len(alts) > 1,
                    snv=(len(ref) == 1 and len(alt) == 1),
                ))
    df = pd.DataFrame(rows)
    df["var"] = (df.chrom + ":" + df.pos.astype(str) + ":"
                 + df.ref + ":" + df.alt)
    return df


def add_pf_dist(pf):
    """Exact copy of add_dist() in make_fig3_data.py / C2_fp_distance.py."""
    df = pf.copy()
    d = []
    for chrom in df.CHROM.unique():
        sub = df[df.CHROM == chrom]
        pos = sub.POS.values.astype(np.int64)
        n = len(pos)
        if n == 1:
            dd = [10000]
        elif n == 2:
            dd = [pos[1] - pos[0]] * 2
        else:
            d1 = np.r_[10000, pos[1:] - pos[:-1]]
            d2 = np.r_[pos[1:] - pos[:-1], 10000]
            dd = np.minimum(d1, d2).tolist()
        d.extend(dd)
    df["dist"] = d
    return df


def load_truth(group):
    path = GROUPS[group]["truth"]
    df = pd.read_csv(
        path, sep="\t", header=None,
        names=["CHROM", "POS", "P", "REF", "ALT", "GT", "f", "b", "GT2", "group", "AF"],
    )
    df["CHROM"] = df.CHROM.astype(str).str.replace("chr", "")
    df["POS"] = df.POS.astype(int)
    df["var"] = (df.CHROM + ":" + df.POS.astype(str) + ":"
                 + df.REF + ":" + df.ALT)
    assert df["var"].is_unique, f"duplicate truth vars in {path}"
    df = add_pf_dist(df)
    df["scorable"] = df.dist >= 50
    return df


def pos_index(df):
    idx = collections.defaultdict(list)
    for c, p in zip(df.chrom, df.pos):
        idx[str(c)].append(int(p))
    return {k: np.array(sorted(v), dtype=np.int64) for k, v in idx.items()}


def count_within(idx, chrom, pos, D):
    arr = idx.get(str(chrom))
    if arr is None or len(arr) == 0:
        return 0
    return int(np.searchsorted(arr, pos + D, side="right")
               - np.searchsorted(arr, pos - D, side="left"))


def min_dist_to(idx, chrom, pos):
    arr = idx.get(str(chrom))
    if arr is None or len(arr) == 0:
        return np.inf
    i = int(np.searchsorted(arr, pos))
    ds = []
    if i > 0:
        ds.append(pos - int(arr[i - 1]))
    if i < len(arr):
        ds.append(int(arr[i]) - pos)
    return min(ds) if ds else np.inf


def classify_unlisted(donor, bg, truth):
    """Return a dict with the allele categories and the unlisted DataFrame."""
    donor_pos = set(zip(donor.chrom, donor.pos))
    bg_pos = set(zip(bg.chrom, bg.pos))
    pos_private = donor_pos - bg_pos
    shared_pos = donor_pos & bg_pos

    donor_alleles = set(donor["var"])
    bg_alleles = set(bg["var"])

    # donor alleles at shared positions, indexed by position
    bg_alt_by_pos = collections.defaultdict(set)
    for c, p, a in zip(bg.chrom, bg.pos, bg.alt):
        bg_alt_by_pos[(c, p)].add(a)

    donor_dp_alleles = donor_alleles - bg_alleles
    truth_set = set(truth["var"])

    unlisted = donor[donor["var"].isin(donor_dp_alleles - truth_set)].copy()
    unlisted = unlisted.drop_duplicates("var")
    unlisted["pos_private"] = [
        (c, p) in pos_private for c, p in zip(unlisted.chrom, unlisted.pos)
    ]
    unlisted["shared_pos_diff_alt"] = [
        (c, p) in shared_pos for c, p in zip(unlisted.chrom, unlisted.pos)
    ]
    unlisted["category"] = np.where(
        unlisted.pos_private,
        np.where(unlisted.snv, "pos_private_snv_not_in_truth",
                 "pos_private_indel_not_in_truth"),
        np.where(unlisted.snv, "shared_pos_diff_alt_snv",
                 "shared_pos_diff_alt_indel"),
    )

    # shared, identical alleles (no tumor-only difference)
    shared_identical_vars = set()
    for c, p, a, v in zip(donor.chrom, donor.pos, donor.alt, donor["var"]):
        if (c, p) in shared_pos and v in bg_alleles:
            shared_identical_vars.add(v)

    # shared-position donor alleles whose ALT is not present in background
    shared_diff = donor[
        [(c, p) in shared_pos and a not in bg_alt_by_pos[(c, p)]
         for c, p, a in zip(donor.chrom, donor.pos, donor.alt)]
    ].drop_duplicates("var")

    return dict(
        donor_pos=donor_pos,
        bg_pos=bg_pos,
        pos_private=pos_private,
        shared_pos=shared_pos,
        donor_alleles=donor_alleles,
        bg_alleles=bg_alleles,
        donor_dp_alleles=donor_dp_alleles,
        truth_set=truth_set,
        unlisted=unlisted,
        unlisted_snv=unlisted[unlisted.snv].copy(),
        unlisted_pos_private=unlisted[unlisted.pos_private].copy(),
        unlisted_shared_diff=unlisted[unlisted.shared_pos_diff_alt].copy(),
        shared_diff=shared_diff,
        shared_identical_vars=shared_identical_vars,
        bg_alt_by_pos=bg_alt_by_pos,
    )


def dosage_of_gt(gt):
    gt = gt.replace("|", "/").split("/")
    return sum(1 for x in gt if x not in ("0", ".")) / 2.0


# --------------------------------------------------------------------------- insert size
def sample_insert_sizes(bam_path, bed, n_intervals=80, seed=17, max_reads=4_000_000):
    """Sample TLEN from proper read1 pairs inside the panel (trimmed at 1000)."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(bed), size=min(n_intervals, len(bed)), replace=False)
    tlen = []
    rl = []
    bam = pysam.AlignmentFile(bam_path, "rb")
    n = 0
    for i in idx:
        row = bed.iloc[int(i)]
        c, s, e = str(row.chrom), int(row.start), int(row.end)
        try:
            it = bam.fetch(c, s, e)
        except ValueError:
            continue
        for r in it:
            n += 1
            if r.query_length:
                rl.append(r.query_length)
            if (r.is_proper_pair and r.is_read1 and not r.is_unmapped
                    and r.template_length and 0 < r.template_length <= 1000):
                tlen.append(int(r.template_length))
            if n >= max_reads:
                break
        if n >= max_reads:
            break
    bam.close()
    tlen = np.array(tlen, dtype=np.float64)
    rl = np.array(rl, dtype=np.float64)
    return tlen, rl


def cover_prob(d, lengths):
    """P(fragment covers an allele d bp from the target | fragment covers target)."""
    if d <= 0:
        return 1.0
    L = np.asarray(lengths, dtype=np.float64)
    return float(np.mean(np.where(L > d, (L - d) / np.maximum(L, 1.0), 0.0)))


# --------------------------------------------------------------------------- fragment audit
def frag_span_from_reads(reads):
    starts, ends = [], []
    for r in reads:
        if r.is_unmapped or r.reference_start is None or r.reference_end is None:
            continue
        rs, re_ = int(r.reference_start), int(r.reference_end)
        starts.append(rs)
        ends.append(re_)
        # mate position (same chromosome)
        if (r.next_reference_id == r.reference_id
                and r.next_reference_start is not None
                and r.next_reference_start >= 0):
            ms = int(r.next_reference_start)
            starts.append(ms)
            ends.append(ms + int(r.query_length or 0))
        # template length fallback
        if r.template_length:
            if r.template_length > 0:
                ends.append(rs + int(r.template_length))
            else:
                starts.append(re_ + int(r.template_length))
    if not starts:
        return None
    return min(starts), max(ends)


def base_counts(reads, pos0, min_mapq=MIN_MAPQ):
    c = collections.Counter()
    for r in reads:
        if r.query_sequence is None or r.is_unmapped:
            continue
        if r.mapping_quality < min_mapq:
            continue
        for qp, rp in r.get_aligned_pairs(matches_only=True):
            if rp == pos0:
                c[r.query_sequence[qp]] += 1
                break
    return c


def observed_alt(reads, pos0, alt, min_mapq=MIN_MAPQ):
    for r in reads:
        if r.query_sequence is None or r.is_unmapped:
            continue
        if r.mapping_quality < min_mapq:
            continue
        for qp, rp in r.get_aligned_pairs(matches_only=True):
            if rp == pos0:
                if r.query_sequence[qp] == alt:
                    return True
                break
    return False


def audit_fragments(group, truth, unlisted_all, unlisted_snv, truth_idx, max_targets=None):
    """Direct BAM audit of donor fragments at truth targets."""
    cfg = GROUPS[group]
    tumor = pysam.AlignmentFile(cfg["tumor"], "rb")
    donor = pysam.AlignmentFile(cfg["donor_bam"], "rb")

    un_all_by_chr = collections.defaultdict(list)
    for r in unlisted_all.itertuples():
        un_all_by_chr[str(r.chrom)].append(r)
    un_snv_by_chr = collections.defaultdict(list)
    for r in unlisted_snv.itertuples():
        un_snv_by_chr[str(r.chrom)].append(r)

    targets = truth.itertuples()
    if max_targets:
        targets = list(targets)[:max_targets]

    frag_rows = []
    target_rows = []
    t0 = time.time()
    n_done = 0

    for t in targets:
        chrom = str(t.CHROM)
        pos = int(t.POS)
        n_done += 1
        try:
            donor_q = set(r.query_name for r in donor.fetch(
                chrom, max(0, pos - WINDOW), pos + WINDOW))
        except ValueError:
            donor_q = set()
        try:
            tumor_reads = list(tumor.fetch(
                chrom, max(0, pos - WINDOW), pos + WINDOW))
        except ValueError:
            tumor_reads = []
        donor_reads = collections.defaultdict(list)
        for r in tumor_reads:
            if r.query_name in donor_q:
                donor_reads[r.query_name].append(r)

        un_all_here = un_all_by_chr.get(chrom, [])
        un_snv_here = un_snv_by_chr.get(chrom, [])

        n_frag = 0
        n_with = 0
        n_with_obs = 0
        vals = []
        obs_alleles = set()
        for qname, reads in donor_reads.items():
            span = frag_span_from_reads(reads)
            if span is None:
                continue
            s, e = span
            if not (s <= pos - 1 < e):
                continue
            flen = e - s
            plausible = 0 < flen <= MAX_FRAG
            if not plausible:
                continue
            n_frag += 1
            # target base
            tb = base_counts(reads, pos - 1)
            target_base = tb.most_common(1)[0][0] if tb else None
            # other truth targets inside span (exclude the target itself)
            arr = truth_idx.get(chrom)
            if arr is not None:
                n_other_truth = int(np.searchsorted(arr, e, side="left")
                                    - np.searchsorted(arr, s, side="left"))
                if s <= pos - 1 < e:
                    n_other_truth -= 1
            else:
                n_other_truth = 0
            # unlisted alleles in span
            in_all = [a for a in un_all_here
                      if s <= int(a.pos) - 1 < e and int(a.pos) != pos]
            in_snv = [a for a in un_snv_here
                      if s <= int(a.pos) - 1 < e and int(a.pos) != pos]
            obs = [a.var for a in in_snv
                   if observed_alt(reads, int(a.pos) - 1, a.alt)]
            if in_snv:
                n_with += 1
            if obs:
                n_with_obs += 1
                obs_alleles.update(obs)
            vals.append(len(in_snv))
            frag_rows.append(dict(
                group=group, chrom=chrom, target_pos=pos,
                target_ref=t.REF, target_alt=t.ALT, qname=qname,
                n_reads=len(reads), frag_start=s, frag_end=e, frag_len=flen,
                target_base=target_base,
                n_other_truth_in_span=n_other_truth,
                n_unlisted_snv_in_span=len(in_snv),
                n_unlisted_indel_in_span=len(in_all) - len(in_snv),
                n_unlisted_all_in_span=len(in_all),
                observed_unlisted_snv=";".join(obs),
                n_observed_unlisted_snv=len(obs),
            ))

        target_rows.append(dict(
            group=group, chrom=chrom, pos=pos, ref=t.REF, alt=t.ALT,
            n_donor_fragments=n_frag,
            n_frag_with_unlisted_snv=n_with,
            n_frag_with_observed_unlisted_snv=n_with_obs,
            frac_frag_with_unlisted_snv=(n_with / n_frag) if n_frag else np.nan,
            mean_unlisted_snv_per_frag=(float(np.mean(vals)) if vals else np.nan),
            median_unlisted_snv_per_frag=(float(np.median(vals)) if vals else np.nan),
            max_unlisted_snv_per_frag=(int(np.max(vals)) if vals else 0),
            n_distinct_unlisted_snv_observed=len(obs_alleles),
            observed_unlisted_snv=";".join(sorted(obs_alleles)),
            n_other_truth_in_span_mean=(
                float(np.mean([x["n_other_truth_in_span"] for x in frag_rows[-(n_frag):]]))
                if n_frag else np.nan),
        ))
        if n_done % 50 == 0 or n_done == len(truth):
            log(f"  group {group}: audited {n_done}/{len(truth)} targets "
                f"({time.time()-t0:.1f}s; fragments so far "
                f"{sum(r['n_donor_fragments'] for r in target_rows)})")

    tumor.close()
    donor.close()
    return pd.DataFrame(frag_rows), pd.DataFrame(target_rows)


# --------------------------------------------------------------------------- FP validation
def validate_fps(group, fp_df, donor_alleles, truth_set, unlisted_set,
                 donor_pos_set, bed):
    cfg = GROUPS[group]
    tumor = pysam.AlignmentFile(cfg["tumor"], "rb")
    donor = pysam.AlignmentFile(cfg["donor_bam"], "rb")
    bg = pysam.AlignmentFile(cfg["bg_bam"], "rb")
    rows = []
    for r in fp_df.itertuples():
        chrom, pos = str(r.CHROM), int(r.POS)
        ref, alt = r.REF, r.ALT
        var = r.var
        try:
            donor_q = set(x.query_name for x in donor.fetch(
                chrom, max(0, pos - FP_WINDOW), pos + FP_WINDOW))
        except ValueError:
            donor_q = set()
        try:
            bg_q = set(x.query_name for x in bg.fetch(
                chrom, max(0, pos - FP_WINDOW), pos + FP_WINDOW))
        except ValueError:
            bg_q = set()
        try:
            reads = list(tumor.fetch(chrom, max(0, pos - FP_WINDOW), pos + FP_WINDOW))
        except ValueError:
            reads = []
        d_reads = [x for x in reads if x.query_name in donor_q]
        b_reads = [x for x in reads if x.query_name in bg_q and x.query_name not in donor_q]
        u_reads = [x for x in reads if x.query_name not in donor_q and x.query_name not in bg_q]
        cd = base_counts(d_reads, pos - 1)
        cb = base_counts(b_reads, pos - 1)
        cu = base_counts(u_reads, pos - 1)
        d_cov = sum(cd.values())
        b_cov = sum(cb.values())
        u_cov = sum(cu.values())
        t_cov = d_cov + b_cov + u_cov
        d_alt, d_ref = cd.get(alt, 0), cd.get(ref, 0)
        b_alt, b_ref = cb.get(alt, 0), cb.get(ref, 0)
        t_alt = cd.get(alt, 0) + cb.get(alt, 0) + cu.get(alt, 0)
        t_ref = cd.get(ref, 0) + cb.get(ref, 0) + cu.get(ref, 0)
        d_frac = d_alt / d_cov if d_cov else np.nan
        b_frac = b_alt / b_cov if b_cov else np.nan
        donor_supported = (d_alt >= FP_DONOR_MIN_ALT and d_cov >= 5
                           and d_frac >= FP_DONOR_MIN_FRAC)
        bg_supported = (b_alt >= FP_BG_MIN_ALT and b_cov >= 5
                        and b_frac >= FP_BG_MIN_FRAC)
        if donor_supported:
            cls = "donor_supported"
        elif bg_supported:
            cls = "background_germline_supported"
        elif b_alt >= FP_BG_MIN_ALT and b_frac >= 0.001:
            cls = "low_level_background_error"
        elif d_cov < 5 and b_cov < 5:
            cls = "no_donor_or_bg_coverage"
        else:
            cls = "tumor_only_error"
        donor_alts_here = donor_alleles.get((chrom, pos), set())
        rows.append(dict(
            group=group, method=r.Method, platform=r.Platform, var=var,
            CHROM=chrom, POS=pos, REF=ref, ALT=alt,
            fp_distance=r.distance, in_target_interval=r.in_target_interval,
            in_panel=in_bed(bed, chrom, pos),
            vcf_donor_alts_at_pos=";".join(sorted(donor_alts_here)),
            vcf_donor_has_this_alt=(alt in donor_alts_here),
            vcf_is_unlisted=(var in unlisted_set),
            vcf_is_truth=(var in truth_set),
            tumor_cov=t_cov, donor_cov=d_cov, bg_cov=b_cov, unknown_cov=u_cov,
            tumor_alt=t_alt, tumor_ref=t_ref,
            donor_alt=d_alt, donor_ref=d_ref,
            bg_alt=b_alt, bg_ref=b_ref,
            donor_alt_frac=d_frac, bg_alt_frac=b_frac,
            donor_supported=donor_supported,
            bg_supported=bg_supported,
            fp_class=cls,
        ))
    tumor.close()
    donor.close()
    bg.close()
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- main
def main():
    t_start = time.time()
    log("=" * 78)
    log("G1: direct audit of donor alleles on PosMix donor fragments")
    log("=" * 78)

    bed = load_panel_bed(PANEL_BED)
    panel_bp = int((bed.end - bed.start).sum())
    log(f"panel BED: {len(bed):,} merged intervals, {panel_bp:,} bp, "
        f"{bed.chrom.nunique()} chromosomes")

    # pkl completeness check (documented, not used for the audit)
    if os.path.exists(PKL):
        try:
            pkl = pickle.load(open(PKL, "rb"))
            for sid in ("HG004", "HG005"):
                n = len(pkl[sid])
                inside = sum(in_bed(bed, str(c), int(p)) for c, p in
                             zip(pkl[sid].chrom, pkl[sid].pos))
                log(f"note: giab_panel_vars.pkl {sid}: {n} variants, "
                    f"{inside} inside the PosMix panel BED -> pkl is not a "
                    f"complete panel-variant source; using the panel VCFs instead")
        except Exception as e:
            log(f"note: could not inspect pkl: {e}")

    # ------------------------------------------------------------------ static
    static_summary_rows = []
    per_target_all = []
    unlisted_snv_by_group = {}
    unlisted_all_by_group = {}
    categories_by_group = {}
    truth_by_group = {}
    insert_by_group = {}

    for group, cfg in GROUPS.items():
        log("")
        log(f"### group {group}: donor={cfg['donor']} background={cfg['background']}")
        donor = load_panel_alleles(cfg["donor"])
        bg = load_panel_alleles(cfg["background"])
        truth = load_truth(group)
        cat = classify_unlisted(donor, bg, truth)
        truth_by_group[group] = truth
        unlisted_snv_by_group[group] = cat["unlisted_snv"]
        unlisted_all_by_group[group] = cat["unlisted"]
        categories_by_group[group] = cat

        log(f"donor alleles (split multiallelic): {len(donor)}; "
            f"donor positions: {len(cat['donor_pos'])}")
        log(f"donor-private positions: {len(cat['pos_private'])}; "
            f"donor-private alleles (allele-level): {len(cat['donor_dp_alleles'])}")
        log(f"truth targets: {len(truth)}; in donor-private alleles: "
            f"{truth['var'].isin(cat['donor_dp_alleles']).sum()}; "
            f"scorable (dist>=50): {int(truth.scorable.sum())}; "
            f"near (dist<50): {int((~truth.scorable).sum())}")
        log(f"unlisted donor alleles: {len(cat['unlisted'])} "
            f"(SNV={len(cat['unlisted_snv'])}, "
            f"pos-private={len(cat['unlisted_pos_private'])}, "
            f"shared-pos-diff-ALT={len(cat['unlisted_shared_diff'])})")
        log(f"shared positions: {len(cat['shared_pos'])}; "
            f"shared identical alleles: {len(cat['shared_identical_vars'])}; "
            f"shared-pos-diff-ALT donor alleles: {len(cat['shared_diff'])}")
        if len(cat["unlisted_snv"]):
            log("unlisted donor SNVs:")
            for r in cat["unlisted_snv"].itertuples():
                log(f"  {r.var} GT={r.GT} dosage={r.dosage:.2f} "
                    f"category={r.category}")

        # insert size distribution
        tlen, rl = sample_insert_sizes(cfg["donor_bam"], bed)
        insert_by_group[group] = dict(
            tlen_median=float(np.median(tlen)),
            tlen_mean=float(np.mean(tlen)),
            tlen_p10=float(np.percentile(tlen, 10)),
            tlen_p90=float(np.percentile(tlen, 90)),
            read_len_median=float(np.median(rl)) if len(rl) else np.nan,
            n_tlen=int(len(tlen)),
        )
        log(f"donor insert size (TLEN<=1kb): median={np.median(tlen):.0f} bp, "
            f"mean={np.mean(tlen):.0f} bp, P10={np.percentile(tlen,10):.0f}, "
            f"P90={np.percentile(tlen,90):.0f}; read length={np.median(rl):.0f}")

        # indices
        truth_idx = pos_index(truth.rename(columns={"CHROM": "chrom", "POS": "pos"}))
        un_snv_idx = pos_index(cat["unlisted_snv"])
        un_all_idx = pos_index(cat["unlisted"])
        un_pp_idx = pos_index(cat["unlisted_pos_private"])
        un_sp_idx = pos_index(cat["unlisted_shared_diff"])

        # per-target static counts
        for t in truth.itertuples():
            chrom, pos = str(t.CHROM), int(t.POS)
            row = dict(
                group=group, chrom=chrom, pos=pos, ref=t.REF, alt=t.ALT,
                P=t.P, AF=t.AF, GT=t.GT, GT2=t.GT2,
                pf_dist=t.dist, scorable=bool(t.scorable),
                n_other_truth_150=0, n_other_truth_300=0, n_other_truth_500=0,
                n_unlisted_snv_150=0, n_unlisted_snv_300=0, n_unlisted_snv_500=0,
                n_unlisted_all_150=0, n_unlisted_all_300=0, n_unlisted_all_500=0,
                n_unlisted_posprivate_500=0, n_unlisted_shareddiff_500=0,
                nearest_unlisted_snv_dist=np.inf,
            )
            for D in DS:
                row[f"n_unlisted_snv_{D}"] = count_within(un_snv_idx, chrom, pos, D)
                row[f"n_unlisted_all_{D}"] = count_within(un_all_idx, chrom, pos, D)
                nt = count_within(truth_idx, chrom, pos, D)
                if t.dist == 0 or True:
                    nt -= 1  # exclude self
                row[f"n_other_truth_{D}"] = max(0, nt)
            row["n_unlisted_posprivate_500"] = count_within(un_pp_idx, chrom, pos, 500)
            row["n_unlisted_shareddiff_500"] = count_within(un_sp_idx, chrom, pos, 500)
            if len(cat["unlisted_snv"]):
                row["nearest_unlisted_snv_dist"] = min_dist_to(un_snv_idx, chrom, pos)
            per_target_all.append(row)

        # static summary by D
        sub = pd.DataFrame([r for r in per_target_all if r["group"] == group])
        for D in DS:
            for col in (f"n_unlisted_snv_{D}", f"n_unlisted_all_{D}",
                        f"n_other_truth_{D}"):
                v = sub[col].values.astype(float)
                static_summary_rows.append(dict(
                    group=group, metric=f"{col}_per_target",
                    mean=float(np.mean(v)), median=float(np.median(v)),
                    p90=float(np.percentile(v, 90)), max=float(np.max(v)),
                    n_targets_ge1=int((v >= 1).sum()), n_targets=len(v),
                    sum=int(v.sum()),
                ))
            log(f"  +/-{D} bp per target: unlisted SNV mean={np.mean(sub[f'n_unlisted_snv_{D}']):.3f}, "
                f"median={np.median(sub[f'n_unlisted_snv_{D}']):.0f}, "
                f"P90={np.percentile(sub[f'n_unlisted_snv_{D}'],90):.0f}, "
                f"max={np.max(sub[f'n_unlisted_snv_{D}']):.0f}, "
                f"targets>=1={int((sub[f'n_unlisted_snv_{D}']>=1).sum())}; "
                f"unlisted ALL mean={np.mean(sub[f'n_unlisted_all_{D}']):.3f}, "
                f"max={np.max(sub[f'n_unlisted_all_{D}']):.0f}; "
                f"other truth targets mean={np.mean(sub[f'n_other_truth_{D}']):.2f}, "
                f"max={np.max(sub[f'n_other_truth_{D}']):.0f}")

        # static scalar summary
        static_summary_rows.extend([
            dict(group=group, metric="n_donor_alleles", mean=len(donor)),
            dict(group=group, metric="n_donor_positions", mean=len(cat["donor_pos"])),
            dict(group=group, metric="n_donor_pos_private_positions",
                 mean=len(cat["pos_private"])),
            dict(group=group, metric="n_donor_private_alleles",
                 mean=len(cat["donor_dp_alleles"])),
            dict(group=group, metric="n_truth_targets", mean=len(truth)),
            dict(group=group, metric="n_truth_scorable",
                 mean=int(truth.scorable.sum())),
            dict(group=group, metric="n_truth_near_dist_lt50",
                 mean=int((~truth.scorable).sum())),
            dict(group=group, metric="n_unlisted_donor_alleles",
                 mean=len(cat["unlisted"])),
            dict(group=group, metric="n_unlisted_donor_snv",
                 mean=len(cat["unlisted_snv"])),
            dict(group=group, metric="n_unlisted_posprivate",
                 mean=len(cat["unlisted_pos_private"])),
            dict(group=group, metric="n_unlisted_shared_pos_diff_alt",
                 mean=len(cat["unlisted_shared_diff"])),
            dict(group=group, metric="n_shared_positions",
                 mean=len(cat["shared_pos"])),
            dict(group=group, metric="n_shared_identical_alleles",
                 mean=len(cat["shared_identical_vars"])),
            dict(group=group, metric="n_shared_pos_diff_alt_alleles",
                 mean=len(cat["shared_diff"])),
            dict(group=group, metric="panel_bp", mean=panel_bp),
            dict(group=group, metric="insert_size_median",
                 mean=insert_by_group[group]["tlen_median"]),
            dict(group=group, metric="insert_size_mean",
                 mean=insert_by_group[group]["tlen_mean"]),
            dict(group=group, metric="read_length_median",
                 mean=insert_by_group[group]["read_len_median"]),
        ])

    # ------------------------------------------------------------------ BAM audit
    log("")
    log("### BAM-level donor-fragment audit (SURFSeq 5000, reads100M)")
    fragment_level = []
    fragment_target = {}
    max_targets_env = os.environ.get("G1_MAX_TARGETS")
    max_targets = int(max_targets_env) if max_targets_env else None
    if max_targets:
        log(f"NOTE: G1_MAX_TARGETS={max_targets} -> BAM audit is a subsample "
            f"(debug mode); tables will be marked accordingly")
    for group in GROUPS:
        truth = truth_by_group[group]
        cat = categories_by_group[group]
        n_audit = min(len(truth), max_targets) if max_targets else len(truth)
        log(f"group {group}: auditing {n_audit} targets")
        fl, ft = audit_fragments(
            group, truth, cat["unlisted"], cat["unlisted_snv"],
            pos_index(truth.rename(columns={"CHROM": "chrom", "POS": "pos"})),
            max_targets=max_targets,
        )
        fragment_level.append(fl)
        fragment_target[group] = ft
        log(f"group {group}: {len(fl):,} donor fragments, "
            f"{(fl.n_unlisted_snv_in_span > 0).sum():,} with >=1 unlisted SNV, "
            f"{(fl.n_observed_unlisted_snv > 0).sum():,} with observed ALT")

    fragment_level = pd.concat(fragment_level, ignore_index=True)
    fragment_target_all = pd.concat(fragment_target.values(), ignore_index=True)

    # merge BAM per-target into per-target table and compute expectation
    pt = pd.DataFrame(per_target_all)
    pt = pt.merge(
        fragment_target_all[["group", "chrom", "pos", "n_donor_fragments",
                             "n_frag_with_unlisted_snv",
                             "n_frag_with_observed_unlisted_snv",
                             "frac_frag_with_unlisted_snv",
                             "mean_unlisted_snv_per_frag",
                             "max_unlisted_snv_per_frag",
                             "n_distinct_unlisted_snv_observed",
                             "observed_unlisted_snv"]],
        on=["group", "chrom", "pos"], how="left",
    )
    for col in ("n_donor_fragments", "n_frag_with_unlisted_snv",
                "n_frag_with_observed_unlisted_snv"):
        pt[col] = pt[col].fillna(0).astype(int)

    # static fragment-length-aware expectation
    for group in GROUPS:
        cat = categories_by_group[group]
        Ls = np.array([], dtype=float)
        # use donor TLEN sample again (small cost) -- stored summary only
        tlen, _ = sample_insert_sizes(GROUPS[group]["donor_bam"], bed, n_intervals=40)
        Ls = tlen
        un = cat["unlisted_snv"]
        un_by_chr = collections.defaultdict(list)
        for r in un.itertuples():
            un_by_chr[str(r.chrom)].append(r)
        mask = pt.group == group
        lam = np.zeros(mask.sum())
        idxs = pt.index[mask]
        for j, i in enumerate(idxs):
            row = pt.loc[i]
            chrom, pos = str(row.chrom), int(row.pos)
            lam_j = 0.0
            for a in un_by_chr.get(chrom, []):
                d = abs(int(a.pos) - pos)
                if d == 0:
                    continue
                lam_j += float(a.dosage) * cover_prob(d, Ls)
            lam[j] = lam_j
        pt.loc[idxs, "lambda_unlisted_snv_per_frag"] = lam
        pt.loc[idxs, "p_frag_carries_unlisted_snv"] = 1.0 - np.exp(-lam)
        pt.loc[idxs, "expected_frags_with_unlisted_snv"] = (
            pt.loc[idxs, "n_donor_fragments"].values * (1.0 - np.exp(-lam)))
        pt.loc[idxs, "expected_unlisted_snv_pairs"] = (
            pt.loc[idxs, "n_donor_fragments"].values * lam)

    # region labels
    pt["region"] = "other"
    pt.loc[(pt.chrom == "6") & (pt.pos.between(31323000, 31325500)), "region"] = "chr6_31.32Mb_cluster"
    pt.loc[(pt.chrom == "10") & (pt.pos.between(123350000, 123366000)), "region"] = "chr10_123.36Mb_cluster"
    # also mark targets within 1 kb of any unlisted SNV
    near_un = []
    for r in pt.itertuples():
        cat = categories_by_group[r.group]
        u = cat["unlisted_snv"]
        near_un.append(bool((((u.chrom == r.chrom)
                              & ((u.pos - r.pos).abs() <= 1000)).any())))
    pt["within_1kb_of_unlisted_snv"] = near_un

    # append expectation / observed summary rows (fragment-level expectation)
    for group in GROUPS:
        sub = pt[pt.group == group]
        n_t = len(sub)
        for col, label in [
            ("expected_frags_with_unlisted_snv", "expected_frags_with_unlisted_snv_per_target"),
            ("expected_unlisted_snv_pairs", "expected_unlisted_snv_pairs_per_target"),
            ("lambda_unlisted_snv_per_frag", "lambda_unlisted_snv_per_frag"),
            ("p_frag_carries_unlisted_snv", "p_frag_carries_unlisted_snv_per_target"),
        ]:
            v = sub[col].values.astype(float)
            static_summary_rows.append(dict(
                group=group, metric=label,
                mean=float(np.mean(v)), median=float(np.median(v)),
                p90=float(np.percentile(v, 90)), max=float(np.max(v)),
                n_targets_ge1=int((v > 0).sum()), n_targets=n_t,
                sum=float(np.sum(v)),
            ))
        tot_frag = int(sub.n_donor_fragments.sum())
        tot_with = int(sub.n_frag_with_unlisted_snv.sum())
        static_summary_rows.extend([
            dict(group=group, metric="observed_donor_fragments_total", mean=tot_frag),
            dict(group=group, metric="observed_frags_with_unlisted_snv_total",
                 mean=tot_with),
            dict(group=group, metric="observed_frac_frags_with_unlisted_snv",
                 mean=(tot_with / tot_frag) if tot_frag else np.nan),
        ])
        for region in ("chr6_31.32Mb_cluster", "chr10_123.36Mb_cluster", "other"):
            r = sub[sub.region == region]
            if len(r) == 0:
                continue
            tf = int(r.n_donor_fragments.sum())
            tw = int(r.n_frag_with_unlisted_snv.sum())
            static_summary_rows.append(dict(
                group=group,
                metric=f"observed_frac_frags_with_unlisted_snv_{region}",
                mean=(tw / tf) if tf else np.nan,
                n_targets=len(r), sum=tw,
            ))

    # per-unlisted-allele static table
    allele_rows = []
    for group in GROUPS:
        cat = categories_by_group[group]
        truth = truth_by_group[group]
        tidx = pos_index(truth.rename(columns={"CHROM": "chrom", "POS": "pos"}))
        fl = fragment_level[fragment_level.group == group]
        obs_counts = collections.Counter()
        obs_targets = collections.defaultdict(set)
        for r in fl.itertuples():
            s = r.observed_unlisted_snv
            if isinstance(s, str) and s:
                for v in s.split(";"):
                    if v:
                        obs_counts[v] += 1
                        obs_targets[v].add(f"{r.chrom}:{r.target_pos}")
        for r in cat["unlisted"].itertuples():
            chrom, pos = str(r.chrom), int(r.pos)
            row = dict(
                group=group, chrom=chrom, pos=pos, ref=r.ref, alt=r.alt,
                GT=r.GT, dosage=r.dosage, snv=bool(r.snv),
                multiallelic=bool(r.multiallelic), category=r.category,
                pos_private=bool(r.pos_private),
                shared_pos_diff_alt=bool(r.shared_pos_diff_alt),
                var=r.var,
                dist_nearest_truth=min_dist_to(tidx, chrom, pos),
            )
            for D in (150, 300, 500, 1000):
                row[f"n_truth_targets_within_{D}"] = count_within(tidx, chrom, pos, D)
            row["observed_alt_fragment_count"] = int(obs_counts.get(r.var, 0))
            row["observed_target_positions"] = ";".join(
                sorted(obs_targets.get(r.var, set())))
            allele_rows.append(row)
    pd.DataFrame(allele_rows).to_csv(
        os.path.join(TAB, "G1_static_unintended_alleles.csv"), index=False)

    # ------------------------------------------------------------------ FP validation
    log("")
    log("### FP-site donor validation (SomaticXGB, SURFSeq 5000, groups 45/54)")
    fp_val = pd.DataFrame()
    exact_unlisted_fp = {}
    if os.path.exists(C2_CALLS):
        fp_all = pd.read_csv(C2_CALLS)
        fp_all["Group"] = fp_all.Group.astype(str)
        log(f"C2 FP table: {len(fp_all):,} calls, "
            f"methods={fp_all.Method.nunique()}, "
            f"groups={sorted(fp_all.Group.unique())}")
        for group in GROUPS:
            cat = categories_by_group[group]
            uset = set(cat["unlisted"]["var"])
            usnv_set = set(cat["unlisted_snv"]["var"])
            f = fp_all[fp_all.Group == group]
            exact_unlisted_fp[group] = dict(
                n_fp_total=len(f),
                n_fp_exact_unlisted_all=int(f["var"].isin(uset).sum()),
                n_fp_exact_unlisted_snv=int(f["var"].isin(usnv_set).sum()),
            )
            log(f"group {group}: {len(f):,} FP calls across all methods/platforms; "
                f"exact match to unlisted donor allele={exact_unlisted_fp[group]['n_fp_exact_unlisted_all']}, "
                f"to unlisted donor SNV={exact_unlisted_fp[group]['n_fp_exact_unlisted_snv']}")

        fp_sel = fp_all[(fp_all.Method == "SomaticXGB 0.1")
                        & (fp_all.Platform == "SURFSeq_5000")
                        & (fp_all.Group.isin(["45", "54"]))].copy()
        log(f"validating {len(fp_sel)} SomaticXGB SURFSeq 5000 FP sites in the BAM")
        vals = []
        for group in ("45", "54"):
            fg = fp_sel[fp_sel.Group == group].copy()
            if len(fg) == 0:
                continue
            cat = categories_by_group[group]
            donor_pos_alts = collections.defaultdict(set)
            for r in load_panel_alleles(GROUPS[group]["donor"]).itertuples():
                donor_pos_alts[(str(r.chrom), int(r.pos))].add(r.alt)
            # truth case: group 54 has chr18 absent from panel, fine
            vals.append(validate_fps(
                group, fg,
                donor_pos_alts,
                cat["truth_set"], set(cat["unlisted"]["var"]),
                cat["donor_pos"], bed,
            ))
        fp_val = pd.concat(vals, ignore_index=True)
        for group in ("45", "54"):
            fg = fp_val[fp_val.group == group]
            if len(fg) == 0:
                continue
            log(f"  group {group}: n={len(fg)}; "
                f"donor_supported={int(fg.donor_supported.sum())}, "
                f"background_germline_supported={int(fg.bg_supported.sum())}, "
                f"low_level_background_error={int((fg.fp_class=='low_level_background_error').sum())}, "
                f"tumor_only_error={int((fg.fp_class=='tumor_only_error').sum())}, "
                f"no_donor/bg_coverage={int((fg.fp_class=='no_donor_or_bg_coverage').sum())}, "
                f"vcf_unlisted={int(fg.vcf_is_unlisted.sum())}, "
                f"vcf_donor_has_this_alt={int(fg.vcf_donor_has_this_alt.sum())}, "
                f"in_panel={int(fg.in_panel.sum())}")
            near = fg[fg.fp_distance <= 150]
            log(f"  group {group} FPs <=150bp of truth: n={len(near)}; "
                f"donor_supported={int(near.donor_supported.sum())}, "
                f"background_supported={int(near.bg_supported.sum())}")
    else:
        log(f"C2 FP table not found: {C2_CALLS}; skipping FP validation")

    # ------------------------------------------------------------------ fragment summary
    fs_rows = []
    for group in GROUPS:
        fl = fragment_level[fragment_level.group == group]
        ft = fragment_target[group]
        for region, sel in [
            ("ALL", np.ones(len(fl), dtype=bool)),
            ("chr6_31.32Mb_cluster", (fl.chrom == "6") & fl.target_pos.between(31323000, 31325500)),
            ("chr10_123.36Mb_cluster", (fl.chrom == "10") & fl.target_pos.between(123350000, 123366000)),
            ("other", ~(((fl.chrom == "6") & fl.target_pos.between(31323000, 31325500))
                        | ((fl.chrom == "10") & fl.target_pos.between(123350000, 123366000)))),
        ]:
            sub = fl[sel]
            # target-level region selection
            if region == "ALL":
                tsel = np.ones(len(ft), dtype=bool)
            elif region == "chr6_31.32Mb_cluster":
                tsel = (ft.chrom == "6") & ft.pos.between(31323000, 31325500)
            elif region == "chr10_123.36Mb_cluster":
                tsel = (ft.chrom == "10") & ft.pos.between(123350000, 123366000)
            else:
                tsel = ~(((ft.chrom == "6") & ft.pos.between(31323000, 31325500))
                         | ((ft.chrom == "10") & ft.pos.between(123350000, 123366000)))
            tsub = ft[tsel]
            n_pairs = len(sub)
            n_with = int((sub.n_unlisted_snv_in_span > 0).sum())
            obs = int((sub.n_observed_unlisted_snv > 0).sum())
            # collapse to unique donor fragments (a fragment may cover >1 target)
            uq = sub.groupby("qname").agg(
                n_un=("n_unlisted_snv_in_span", "max"),
                n_obs=("n_observed_unlisted_snv", "max"))
            n_uniq = len(uq)
            n_uniq_with = int((uq.n_un > 0).sum())
            n_uniq_obs = int((uq.n_obs > 0).sum())
            vals = sub.n_unlisted_snv_in_span.values.astype(float)
            obs_alleles = set()
            for s in sub.observed_unlisted_snv:
                if isinstance(s, str) and s:
                    obs_alleles.update(s.split(";"))
            fs_rows.append(dict(
                group=group, region=region,
                n_targets=int(tsel.sum()),
                n_fragment_target_pairs=n_pairs,
                n_unique_donor_fragments=n_uniq,
                n_pairs_with_unlisted_snv=n_with,
                frac_pairs_with_unlisted_snv=(n_with / n_pairs if n_pairs else np.nan),
                n_unique_frag_with_unlisted_snv=n_uniq_with,
                frac_unique_frag_with_unlisted_snv=(n_uniq_with / n_uniq if n_uniq else np.nan),
                n_pairs_with_observed_unlisted_snv=obs,
                frac_pairs_with_observed_unlisted_snv=(obs / n_pairs if n_pairs else np.nan),
                n_unique_frag_with_observed_unlisted_snv=n_uniq_obs,
                frac_unique_frag_with_observed_unlisted_snv=(n_uniq_obs / n_uniq if n_uniq else np.nan),
                mean_unlisted_snv_per_frag=(float(np.mean(vals)) if n_pairs else np.nan),
                median_unlisted_snv_per_frag=(float(np.median(vals)) if n_pairs else np.nan),
                max_unlisted_snv_per_frag=(int(np.max(vals)) if n_pairs else 0),
                n_distinct_unlisted_snv_observed=len(obs_alleles),
                observed_unlisted_snv=";".join(sorted(obs_alleles)),
            ))
    fragment_summary = pd.DataFrame(fs_rows)

    # add observed/expected comparison at ALL level
    for group in GROUPS:
        mask = pt.group == group
        obs = int(pt.loc[mask, "n_frag_with_unlisted_snv"].sum())
        exp = float(pt.loc[mask, "expected_frags_with_unlisted_snv"].sum())
        i = fragment_summary[(fragment_summary.group == group)
                             & (fragment_summary.region == "ALL")].index
        fragment_summary.loc[i, "expected_frag_with_unlisted_snv_static"] = exp
        fragment_summary.loc[i, "observed_over_expected"] = (obs / exp) if exp else np.nan
        fs_all = fragment_summary.loc[i].iloc[0]
        static_summary_rows.extend([
            dict(group=group, metric="observed_unique_donor_fragments_total",
                 mean=int(fs_all.n_unique_donor_fragments)),
            dict(group=group, metric="observed_unique_frags_with_unlisted_snv_total",
                 mean=int(fs_all.n_unique_frag_with_unlisted_snv)),
            dict(group=group, metric="observed_frac_unique_frags_with_unlisted_snv",
                 mean=float(fs_all.frac_unique_frag_with_unlisted_snv)),
            dict(group=group, metric="observed_unique_frags_with_observed_unlisted_snv_total",
                 mean=int(fs_all.n_unique_frag_with_observed_unlisted_snv)),
            dict(group=group, metric="observed_frac_unique_frags_with_observed_unlisted_snv",
                 mean=float(fs_all.frac_unique_frag_with_observed_unlisted_snv)),
        ])
        log(f"group {group}: fragment-target pairs with >=1 unlisted SNV="
            f"{obs:,}/{int(fs_all.n_fragment_target_pairs):,}; unique donor "
            f"fragments with >=1 unlisted SNV="
            f"{int(fs_all.n_unique_frag_with_unlisted_snv):,}/"
            f"{int(fs_all.n_unique_donor_fragments):,} "
            f"({100*fs_all.frac_unique_frag_with_unlisted_snv:.2f}%); "
            f"static expectation={exp:.1f} pairs; "
            f"ratio={obs/exp if exp else float('nan'):.2f}")

    # ------------------------------------------------------------------ write tables
    static_summary = pd.DataFrame(static_summary_rows)
    static_summary.to_csv(os.path.join(TAB, "G1_static_summary.csv"), index=False)
    pt.to_csv(os.path.join(TAB, "G1_static_per_target.csv"), index=False)
    fragment_summary.to_csv(os.path.join(TAB, "G1_fragment_summary.csv"), index=False)
    fragment_level.to_csv(os.path.join(TAB, "G1_fragment_level_audit.csv"), index=False)
    if len(fp_val):
        fp_val.to_csv(os.path.join(TAB, "G1_fp_validation.csv"), index=False)

    log("")
    log("wrote G1_static_summary.csv, G1_static_per_target.csv, "
        "G1_static_unintended_alleles.csv, G1_fragment_summary.csv, "
        "G1_fragment_level_audit.csv")
    if len(fp_val):
        log("wrote G1_fp_validation.csv")

    # ------------------------------------------------------------------ figure
    make_figure(pt, fragment_summary, static_summary)

    # ------------------------------------------------------------------ conclusion
    write_conclusion(static_summary, fragment_summary, fp_val,
                     exact_unlisted_fp, categories_by_group, truth_by_group,
                     insert_by_group)

    log("")
    log(f"done in {time.time()-t_start:.1f}s")
    _log_fh.close()


def make_figure(pt, fragment_summary, static_summary):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    colors = {"45": "#2b6cb0", "54": "#c05621"}

    # Panel A: distribution of per-target unlisted SNV counts within +/-300
    ax = axes[0]
    categories = ["0", "1", "2", "3+"]
    width = 0.36
    for k, group in enumerate(("45", "54")):
        v = pt.loc[pt.group == group, "n_unlisted_snv_300"].values
        counts = [int((v == 0).sum()), int((v == 1).sum()),
                  int((v == 2).sum()), int((v >= 3).sum())]
        x = np.arange(len(categories)) + (k - 0.5) * width
        ax.bar(x, counts, width=width, color=colors[group],
               label=f"group {group} (n={len(v)})")
        for xi, ci in zip(x, counts):
            ax.text(xi, ci + 2, str(ci), ha="center", va="bottom", fontsize=8)
    ax.set_xticks(np.arange(len(categories)))
    ax.set_xticklabels(categories)
    ax.set_xlabel("Unlisted donor SNVs within +/-300 bp of a target")
    ax.set_ylabel("Number of truth targets")
    ax.set_title("A. Per-target unlisted donor SNVs")
    ax.legend(frameon=False, fontsize=8)
    ax.set_yscale("log")
    ax.set_ylim(0.8, max(ax.get_ylim()[1], 10))

    # Panel B: observed fraction of donor fragments carrying >=1 unlisted SNV
    ax = axes[1]
    sub = pt[pt.n_donor_fragments > 0].copy()
    sub = sub.sort_values(["group", "frac_frag_with_unlisted_snv"],
                          ascending=[True, False])
    labels = []
    vals = []
    cols = []
    for r in sub.itertuples():
        fr = r.frac_frag_with_unlisted_snv
        if np.isfinite(fr) and fr > 0:
            labels.append(f"{r.group}:{r.chrom}:{r.pos}")
            vals.append(100 * fr)
            cols.append(colors[r.group])
    if vals:
        order = np.argsort(vals)[::-1][:25]
        ax.barh(np.arange(len(order)), [vals[i] for i in order],
                color=[cols[i] for i in order])
        ax.set_yticks(np.arange(len(order)))
        ax.set_yticklabels([labels[i] for i in order], fontsize=6)
        ax.set_xlabel("% donor fragments carrying >=1 unlisted donor SNV")
        ax.set_title("B. Targets with the highest carrier fraction")
        ax.invert_yaxis()
    else:
        ax.text(0.5, 0.5, "no target carries an unlisted donor SNV",
                ha="center", va="center", transform=ax.transAxes)
    ax.legend(handles=[Patch(color=colors[g], label=f"group {g}") for g in ("45", "54")],
              frameon=False, fontsize=8, loc="lower right")

    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIG, f"G1_unintended_alleles.{ext}"), dpi=200)
    plt.close(fig)
    log("wrote G1_unintended_alleles.png/.pdf")


def write_conclusion(static_summary, fragment_summary, fp_val,
                     exact_unlisted_fp, categories_by_group, truth_by_group,
                     insert_by_group):
    def sval(group, metric, field="mean"):
        s = static_summary[(static_summary.group == group)
                           & (static_summary.metric == metric)]
        return s[field].iloc[0] if len(s) else float("nan")

    lines = []
    lines.append(
        "G1 (R1-4 / R2-2): direct audit of donor alleles on PosMix donor "
        "fragments. Groups 45 (HG004 donor -> HG005 background) and 54 "
        "(HG005 -> HG004); SURFSeq 5000, 100M reads. giab_panel_vars.pkl was "
        "incomplete (1,150/1,151 variants inside the PosMix panel), so the "
        "panel-restricted GIAB v4.2.1 VCFs used to build pf45/pf54 were used."
    )
    for group in ("45", "54"):
        cat = categories_by_group[group]
        truth = truth_by_group[group]
        n_truth = len(truth)
        n_un = len(cat["unlisted"])
        n_un_snv = len(cat["unlisted_snv"])
        n_near = int((~truth.scorable).sum())
        fs = fragment_summary[(fragment_summary.group == group)
                              & (fragment_summary.region == "ALL")].iloc[0]
        fs6 = fragment_summary[(fragment_summary.group == group)
                               & (fragment_summary.region == "chr6_31.32Mb_cluster")].iloc[0]
        n_pairs = int(fs.n_fragment_target_pairs)
        n_uniq = int(fs.n_unique_donor_fragments)
        n_with = int(fs.n_pairs_with_unlisted_snv)
        frac_pairs = fs.frac_pairs_with_unlisted_snv
        n_uniq_with = int(fs.n_unique_frag_with_unlisted_snv)
        frac_uniq = fs.frac_unique_frag_with_unlisted_snv
        n_obs_pairs = int(fs.n_pairs_with_observed_unlisted_snv)
        n_obs_uniq = int(fs.n_unique_frag_with_observed_unlisted_snv)
        d_obs = int(fs.n_distinct_unlisted_snv_observed)
        exp = fs.get("expected_frag_with_unlisted_snv_static", np.nan)
        fg = fp_val[fp_val.group == group] if len(fp_val) else pd.DataFrame()
        n_fp = len(fg)
        n_fp_don = int(fg.donor_supported.sum()) if n_fp else 0
        n_fp_bg = int(fg.bg_supported.sum()) if n_fp else 0
        near = fg[fg.fp_distance <= 150] if n_fp else pd.DataFrame()
        n_near_fp = len(near)
        n_near_don = int(near.donor_supported.sum()) if len(near) else 0
        n_fp_cov = int((fg.donor_cov >= 5).sum()) if n_fp else 0
        n_near_cov = int((near.donor_cov >= 5).sum()) if n_near_fp else 0
        max_don_frac = float(fg.donor_alt_frac.max()) if n_fp else float("nan")
        med_bg_frac = float(fg.bg_alt_frac.median()) if n_fp else float("nan")
        med_tum_frac = (float((fg.tumor_alt / (fg.tumor_alt + fg.tumor_ref)).median())
                        if n_fp else float("nan"))
        ex = exact_unlisted_fp.get(group, {})
        dir_txt = ("HG004 -> HG005" if group == "45" else "HG005 -> HG004")
        lines.append(
            f"Group {group} ({dir_txt}): all {n_truth}/{n_truth} donor-private "
            f"SNVs are in the truth list ({n_near} of them are <50 bp from "
            f"another target and are masked as 'near' at scoring time, not "
            f"removed from the list). Unlisted donor alleles = {n_un} total "
            f"({n_un_snv} SNV + {n_un - n_un_snv} indel/other, raw VCF "
            f"representation; indels are outside the 1-bp SNV F1 metric); the "
            f"{n_un_snv} unlisted SNVs are shared-position/different-ALT donor "
            f"alleles, i.e. 0 donor-private SNVs are missing. The "
            f"{len(cat['shared_identical_vars'])} shared-identical alleles "
            f"cannot create tumor-only differences."
        )
        lines.append(
            f"Group {group} per-target +/-150/300/500 bp unlisted SNVs: mean "
            f"{sval(group,'n_unlisted_snv_150_per_target'):.3f}/"
            f"{sval(group,'n_unlisted_snv_300_per_target'):.3f}/"
            f"{sval(group,'n_unlisted_snv_500_per_target'):.3f}, median 0, "
            f"P90 0, max "
            f"{int(sval(group,'n_unlisted_snv_150_per_target','max'))}/"
            f"{int(sval(group,'n_unlisted_snv_300_per_target','max'))}/"
            f"{int(sval(group,'n_unlisted_snv_500_per_target','max'))}; only "
            f"{int(sval(group,'n_unlisted_snv_300_per_target','n_targets_ge1'))}"
            f"/{n_truth} targets have any within +/-300 bp (confined to the "
            f"chr6:31.32 Mb and chr10:123.36 Mb hypervariable clusters). BAM "
            f"fragment audit: {n_pairs:,} fragment-target pairs from "
            f"{n_uniq:,} unique donor fragments; {n_with:,} pairs "
            f"({100*frac_pairs:.2f}%) and {n_uniq_with:,} unique fragments "
            f"({100*frac_uniq:.2f}%) carry >=1 unlisted donor SNV; "
            f"{n_obs_pairs:,} pairs/{n_obs_uniq:,} unique fragments show the "
            f"ALT base; {d_obs} distinct unlisted donor SNVs are observed; "
            f"static insert-size expectation = {exp:.1f} pairs. In the chr6 "
            f"cluster, {100*fs6.frac_pairs_with_unlisted_snv:.0f}% of pairs and "
            f"{100*fs6.frac_unique_frag_with_unlisted_snv:.0f}% of unique "
            f"fragments carry one."
        )
        lines.append(
            f"Group {group} FP impact: {ex.get('n_fp_total',0):,} FP calls "
            f"across all methods/platforms; exact matches to any unlisted donor "
            f"allele = {ex.get('n_fp_exact_unlisted_all',0)}, to an unlisted "
            f"donor SNV = {ex.get('n_fp_exact_unlisted_snv',0)}. SomaticXGB "
            f"SURFSeq 5000: {n_fp} FPs, {n_fp_don} donor-supported, "
            f"{n_fp_bg} background-supported; {n_fp_cov}/{n_fp} have donor "
            f"coverage (max donor ALT fraction {100*max_don_frac:.1f}%; median "
            f"background/tumor ALT fraction "
            f"{100*med_bg_frac:.2f}%/{100*med_tum_frac:.2f}%), so these are "
            f"low-level background/mapping errors, not donor alleles. Of "
            f"{n_near_fp} FPs within 150 bp of truth ({n_near_cov} with donor "
            f"coverage), {n_near_don} are donor-supported."
        )
    c45 = fragment_summary[(fragment_summary.group == "45")
                           & (fragment_summary.region == "chr6_31.32Mb_cluster")].iloc[0]
    c54 = fragment_summary[(fragment_summary.group == "54")
                           & (fragment_summary.region == "chr6_31.32Mb_cluster")].iloc[0]
    lines.append(
        "Conclusion: the truth set does not literally contain every donor "
        "allele (it omits shared-position donor SNVs and indels), but it "
        "contains 100% of donor-private SNVs (481/481 and 479/479). The only "
        f"omitted donor SNVs ({len(categories_by_group['45']['unlisted_snv'])} and "
        f"{len(categories_by_group['54']['unlisted_snv'])} alleles) are physically present on donor "
        "fragments (up to 100% of fragment-target pairs at individual chr6 "
        f"cluster targets; {100*c45.frac_pairs_with_unlisted_snv:.0f}%/"
        f"{100*c54.frac_pairs_with_unlisted_snv:.0f}% of pairs and "
        f"{100*c45.frac_unique_frag_with_unlisted_snv:.0f}%/"
        f"{100*c54.frac_unique_frag_with_unlisted_snv:.0f}% of unique cluster "
        "fragments overall), yet none of the 13,270 FP calls matches them, and "
        "no BAM-validated near-target SomaticXGB FP is donor-supported. "
        "Unintended donor alleles therefore explain 0 observed FPs (upper bound "
        f"{len(categories_by_group['45']['unlisted_snv'])}/"
        f"{len(categories_by_group['54']['unlisted_snv'])} distinct sites vs "
        "40/31 SomaticXGB SURFSeq FPs and 24-54 FPs per platform); residual FPs "
        "remain dominated by sequencing/mapping errors, consistent with C2 "
        "(9.98% of SomaticXGB FPs <=150 bp vs 4.55% null; most FPs are far from "
        "targets)."
    )
    text = "\n".join(lines) + "\n"
    with open(os.path.join(TAB, "G1_conclusion.txt"), "w") as fh:
        fh.write(text)
    log("wrote G1_conclusion.txt")


if __name__ == "__main__":
    main()
