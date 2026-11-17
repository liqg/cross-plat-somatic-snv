#!/usr/bin/env python3
"""C1 empty-target negative control: call counts, VAF distributions, and
comparison with false positives of the real PosMix simulation (4.test/45).

Definitions
-----------
Empty target (C1): tumor = 100M-read subsample of the raw matched normal,
no donor reads, normal = original 5M N.tidy. Every PASS call is a false
positive by construction.

Real simulation: original 4.test/45 PosMix calls. Truth = pf45_low loci with
inter-locus distance >= 50 bp (438); near = distance < 50 bp (43). Real FP =
PASS SNV calls that are neither truth nor near (same definition as
scripts/A1_metrics.py / make_fig3_data.py).

VAF: VarScan2 FREQ (or AD/(RD+AD)); Mutect2 AD; Strelka2 base counts
(AU/CU/GU/TU); SomaticXGB joined to VarScan2 raw FREQ by variant.

Outputs:
  tables/C1_empty_target_calls.csv
  tables/C1_empty_target_vs_real.csv
"""
import argparse
import gzip
import os
import sys
from bisect import bisect_right

import numpy as np
import pandas as pd

VAF_CUT = 0.005  # 0.5%
XGB_CUT = 0.1
PANEL_MB = 1.971248  # sum(end-start) of intersect_HG004_HG005_panel.sort.bed


def open_maybe_gz(path):
    return gzip.open(path, "rt") if path.endswith(".gz") else open(path, "rt")


def read_bed(path):
    bed = {}
    with open_maybe_gz(path) as fh:
        for line in fh:
            if not line.strip() or line.startswith(("#", "track", "browser")):
                continue
            f = line.rstrip("\n").split("\t")
            bed.setdefault(f[0].replace("chr", ""), []).append((int(f[1]), int(f[2])))
    for c in bed:
        bed[c].sort()
    return bed


def in_panel(bed, chrom, pos):
    chrom = chrom.replace("chr", "")
    ivs = bed.get(chrom)
    if not ivs:
        return False
    starts = [x[0] for x in ivs]
    i = bisect_right(starts, pos - 1) - 1
    if i < 0:
        return False
    s, e = ivs[i]
    return s <= (pos - 1) < e


def parse_freq(s):
    try:
        s = s.strip()
        if s.endswith("%"):
            return float(s[:-1]) / 100.0
        v = float(s)
        return v / 100.0 if v > 1 else v
    except Exception:
        return np.nan


def parse_varscan(fmt, sample, ref, alt):
    """Return VAF from a VarScan2 sample field."""
    keys = fmt.split(":")
    vals = sample.split(":")
    d = dict(zip(keys, vals))
    if "FREQ" in d:
        v = parse_freq(d["FREQ"])
        if np.isfinite(v):
            return v
    try:
        ad = float(d.get("AD", "nan"))
        rd = float(d.get("RD", "nan"))
        if np.isfinite(ad) and np.isfinite(rd) and ad + rd > 0:
            return ad / (ad + rd)
    except Exception:
        pass
    return np.nan


def parse_mutect(fmt, sample, ref, alt):
    keys = fmt.split(":")
    vals = sample.split(":")
    d = dict(zip(keys, vals))
    ad = d.get("AD", "")
    if ad and ad != ".":
        parts = ad.split(",")
        if len(parts) >= 2:
            try:
                refn = float(parts[0])
                altn = float(parts[1])
                if refn + altn > 0:
                    return altn / (refn + altn)
            except Exception:
                pass
    af = d.get("AF", "")
    if af and af != ".":
        try:
            return float(af)
        except Exception:
            pass
    return np.nan


BASES = ("A", "C", "G", "T")


def parse_strelka(fmt, sample, ref, alt):
    keys = fmt.split(":")
    vals = sample.split(":")
    d = dict(zip(keys, vals))
    out = {}
    for b in BASES:
        s = d.get(b + "U", "")
        if not s:
            continue
        try:
            out[b] = sum(float(x) for x in s.split(","))
        except Exception:
            pass
    if ref in out and alt in out and (out[ref] + out[alt]) > 0:
        return out[alt] / (out[ref] + out[alt])
    return np.nan


def get_vaf(caller, fmt, samples_fields, ref, alt):
    if caller in ("VarScan2_raw", "VarScan2_fpfilter", "SomaticXGB"):
        # sample order NORMAL, TUMOR; use TUMOR / last
        idx = None
        for i, nm in enumerate(samples_fields["names"]):
            if nm.upper() == "TUMOR":
                idx = i
        if idx is None:
            idx = len(samples_fields["names"]) - 1
        return parse_varscan(fmt, samples_fields["values"][idx], ref, alt)
    if caller == "Mutect2":
        idx = 0
        for i, nm in enumerate(samples_fields["names"]):
            if nm.lower() != "normal":
                idx = i
        return parse_mutect(fmt, samples_fields["values"][idx], ref, alt)
    if caller == "Strelka2":
        idx = None
        for i, nm in enumerate(samples_fields["names"]):
            if nm.upper() == "TUMOR":
                idx = i
        if idx is None:
            idx = len(samples_fields["names"]) - 1
        return parse_strelka(fmt, samples_fields["values"][idx], ref, alt)
    return np.nan


def read_vcf(path, caller, panel):
    """Return dict with total records, PASS SNV records, and per-pass-call list.

    Each call: (var, vaf, chrom, pos, ref, alt, filt)
    """
    total = 0
    total_snv = 0
    pass_calls = []
    names = []
    panel_total = 0
    outside = 0
    failed = 0
    if not path or not os.path.exists(path):
        return None
    with open_maybe_gz(path) as fh:
        for line in fh:
            if line.startswith("##"):
                continue
            if line.startswith("#CHROM"):
                names = line.rstrip("\n").split("\t")[9:]
                continue
            f = line.rstrip("\n").split("\t")
            if len(f) < 8:
                continue
            total += 1
            chrom, pos, _id, ref, alt, _qual, filt = f[0], int(f[1]), f[2], f[3], f[4], f[5], f[6]
            if len(ref) != 1 or len(alt) != 1 or "," in alt:
                continue
            total_snv += 1
            if not in_panel(panel, chrom, pos):
                outside += 1
                continue
            panel_total += 1
            is_pass = filt in ("PASS", ".") or "PASS" in filt.split(";")
            if not is_pass:
                failed += 1
                continue
            fmt = f[8] if len(f) > 8 else ""
            sf = {"names": names, "values": f[9:]}
            vaf = get_vaf(caller, fmt, sf, ref, alt)
            var = f"{chrom.replace('chr','')}:{pos}:{ref}:{alt}"
            pass_calls.append((var, vaf, chrom, pos, ref, alt, filt))
    return dict(path=path, total=total, total_snv=total_snv, panel_total=panel_total,
                outside=outside, failed=failed, pass_calls=pass_calls)


def read_pred(path, varscan_vaf):
    """SomaticXGB pred file: var, ..., pred. Return pass calls (pred>0.1) with VAF."""
    if not path or not os.path.exists(path):
        return None
    df = pd.read_csv(path, sep="\t", compression="gzip" if path.endswith(".gz") else None)
    df["pred"] = pd.to_numeric(df["pred"], errors="coerce")
    df = df.dropna(subset=["pred"])
    df["vaf"] = df["var"].map(varscan_vaf)
    calls = []
    for r in df.itertuples(index=False):
        if r.pred <= XGB_CUT:
            continue
        chrom, pos, ref, alt = r.var.split(":")
        calls.append((r.var, float(r.vaf) if np.isfinite(r.vaf) else np.nan,
                      chrom, int(pos), ref, alt, "PASS"))
    return dict(path=path, total=len(df), total_snv=len(df), pass_calls=calls,
                panel_total=len(df), outside=0,
                failed=int((df["pred"] <= XGB_CUT).sum()))


def summ(name, vafs):
    v = np.array([x for x in vafs if np.isfinite(x)], dtype=float)
    n = len(v)
    if n == 0:
        return dict(pass_snv=0, vaf_n=0, vaf_ge_0_5pct=0,
                    vaf_q25=np.nan, vaf_median=np.nan, vaf_q75=np.nan, vaf_max=np.nan,
                    vaf_lt_0_5pct=0, vaf_0_5_1pct=0, vaf_1_2pct=0, vaf_2_5pct=0, vaf_ge_5pct=0,
                    vaf_bins="lt0.5=0;0.5-1=0;1-2=0;2-5=0;ge5=0")
    bins = [int((v < 0.005).sum()), int(((v >= 0.005) & (v < 0.01)).sum()),
            int(((v >= 0.01) & (v < 0.02)).sum()), int(((v >= 0.02) & (v < 0.05)).sum()),
            int((v >= 0.05).sum())]
    return dict(pass_snv=n, vaf_n=int(np.isfinite(v).sum()),
                vaf_ge_0_5pct=int((v >= VAF_CUT).sum()),
                vaf_q25=float(np.quantile(v, 0.25)), vaf_median=float(np.median(v)),
                vaf_q75=float(np.quantile(v, 0.75)), vaf_max=float(v.max()),
                vaf_lt_0_5pct=bins[0], vaf_0_5_1pct=bins[1], vaf_1_2pct=bins[2],
                vaf_2_5pct=bins[3], vaf_ge_5pct=bins[4],
                vaf_bins=f"lt0.5={bins[0]};0.5-1={bins[1]};1-2={bins[2]};2-5={bins[3]};ge5={bins[4]}")


def add_dist(pf):
    pf = pf.copy()
    pf["CHROM"] = pf.CHROM.astype(str)
    pf["POS"] = pd.to_numeric(pf.POS)
    d = []
    for chrom, sub in pf.groupby("CHROM", sort=False):
        pos = np.sort(sub.POS.values)
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
    pf["dist"] = d
    return pf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--orig", required=True)
    ap.add_argument("--tables", required=True)
    ap.add_argument("--pf", required=True)
    ap.add_argument("--bed", default=str(common.REF_DIR / "panel" / "intersect_HG004_HG005_panel.sort.bed"))
    ap.add_argument("--sample", required=True)
    a = ap.parse_args()
    s = a.sample
    os.makedirs(a.tables, exist_ok=True)

    panel = read_bed(a.bed)
    pf = pd.read_csv(a.pf, sep="\t", header=None,
                     names=["CHROM", "POS", "P", "REF", "ALT", "GT", "f", "b", "GT2", "group", "AF"],
                     dtype={"CHROM": str})
    pf = add_dist(pf)
    truth = set(pf.loc[pf.dist >= 50].apply(lambda r: f"{r.CHROM}:{int(r.POS)}:{r.REF}:{r.ALT}", axis=1))
    near = set(pf.loc[pf.dist < 50].apply(lambda r: f"{r.CHROM}:{int(r.POS)}:{r.REF}:{r.ALT}", axis=1))
    print(f"truth={len(truth)} near={len(near)} panel_Mb={PANEL_MB}")

    empty_specs = {
        "VarScan2_raw": os.path.join(a.run, f"varscan/{s}.raw.snp.vcf.gz"),
        "VarScan2_fpfilter": os.path.join(a.run, f"varscan_novo/{s}.snp.pass.vcf.gz"),
        "Mutect2": os.path.join(a.run, f"mutect/{s}.filter.vcf.gz"),
        "Strelka2": os.path.join(a.run, f"strelka/{s}/results/variants/somatic.snvs.vcf.gz"),
    }
    real_specs = {
        "VarScan2_raw": os.path.join(a.orig, f"varscan/{s}.raw.snp.vcf.gz"),
        "VarScan2_fpfilter": os.path.join(a.orig, f"varscan_novo/{s}.snp.pass.vcf.gz"),
        "Mutect2": os.path.join(a.orig, f"mutect/{s}.filter.vcf.gz"),
        "Strelka2": os.path.join(a.orig, f"strelka/{s}/results/variants/somatic.snvs.vcf.gz"),
    }

    # VarScan VAF maps for SomaticXGB join
    def varmap_from_varscan(path):
        mp = {}
        if not path or not os.path.exists(path):
            return mp
        vr = read_vcf(path, "VarScan2_raw", panel)
        if vr:
            for var, vaf, *_ in vr["pass_calls"]:
                mp[var] = vaf
        return mp

    empty_varmap = varmap_from_varscan(empty_specs["VarScan2_raw"])
    real_varmap = varmap_from_varscan(real_specs["VarScan2_raw"])

    empty_pred_path = os.path.join(a.run, f"pred/{s}_pred.txt.gz")
    real_pred_path = os.path.join(a.orig, f"pred/{s}_pred.txt.gz")
    if os.path.exists(empty_pred_path) and empty_varmap:
        empty_specs["SomaticXGB"] = empty_pred_path
    if os.path.exists(real_pred_path) and real_varmap:
        real_specs["SomaticXGB"] = real_pred_path

    rows = []
    side = []
    for caller in ["VarScan2_raw", "VarScan2_fpfilter", "Mutect2", "Strelka2", "SomaticXGB"]:
        if caller not in empty_specs:
            print(f"SKIP {caller}: empty-target result unavailable")
            continue
        evr = read_vcf(empty_specs[caller], caller, panel) if caller != "SomaticXGB" else read_pred(empty_pred_path, empty_varmap)
        rvr = read_vcf(real_specs[caller], caller, panel) if caller != "SomaticXGB" else read_pred(real_pred_path, real_varmap)
        if evr is None:
            print(f"SKIP {caller}: missing empty VCF")
            continue
        # VarScan2 total calls for PASS column: raw = all raw calls; fpfilter = pass (+failed as total)
        if caller == "VarScan2_fpfilter":
            failed_file = os.path.join(a.run, f"varscan_novo/{s}.snp.failed.vcf.gz")
            fr = read_vcf(failed_file, caller, panel) if os.path.exists(failed_file) else None
            e_total_snv = evr["total_snv"] + (fr["total_snv"] if fr else 0)
            e_total_records = evr["total"] + (fr["total"] if fr else 0)
        else:
            e_total_snv = evr["total_snv"]
            e_total_records = evr["total"]
        ec = evr["pass_calls"]
        es = summ(caller, [c[1] for c in ec])
        es["total_records"] = e_total_records
        es["total_snv_records"] = e_total_snv
        es["outside_panel_snv"] = evr["outside"]
        es["filtered_snv"] = evr["failed"]

        # real FP at default PASS + VAF>=0.5%
        real_dict = dict(real_pass_snv=np.nan, real_truth_nan=np.nan,
                         real_FP_all=np.nan, real_FP_vaf_ge_0_5pct=np.nan,
                         real_near=np.nan)
        real_fp_vars = set()
        real_fp_vars_vaf = set()
        if rvr is not None:
            rcalls = rvr["pass_calls"]
            rs = summ(caller, [c[1] for c in rcalls])
            fpc = [c for c in rcalls if c[0] not in truth and c[0] not in near]
            fp_vaf = [c for c in fpc if np.isfinite(c[1]) and c[1] >= VAF_CUT]
            real_fp_vars = {c[0] for c in fpc}
            real_fp_vars_vaf = {c[0] for c in fp_vaf}
            real_dict = dict(real_pass_snv=rs["pass_snv"], real_truth=len(truth),
                             real_near=len(near), real_FP_all=len(fpc),
                             real_FP_vaf_ge_0_5pct=len(fp_vaf))
            # real VAF distribution for the FP set
            real_dict.update({
                "real_FP_vaf_q25": float(np.quantile([c[1] for c in fpc if np.isfinite(c[1])], .25)) if fpc else np.nan,
                "real_FP_vaf_median": float(np.median([c[1] for c in fpc if np.isfinite(c[1])])) if fpc else np.nan,
                "real_FP_vaf_q75": float(np.quantile([c[1] for c in fpc if np.isfinite(c[1])], .75)) if fpc else np.nan,
            })

        rfp = real_dict["real_FP_vaf_ge_0_5pct"]
        ratio = (es["vaf_ge_0_5pct"] / rfp
                 if np.isfinite(rfp) and rfp > 0 else np.nan)
        rows.append(dict(
            caller=caller,
            total_calls=es["total_records"],
            total_snv_calls=es["total_snv_records"],
            pass_snv_calls=es["pass_snv"],
            vaf_evaluable=es["vaf_n"],
            pass_snv_vaf_ge_0_5pct=es["vaf_ge_0_5pct"],
            vaf_q25=es["vaf_q25"], vaf_median=es["vaf_median"],
            vaf_q75=es["vaf_q75"], vaf_max=es["vaf_max"],
            vaf_bins=es["vaf_bins"],
            outside_panel_snv=es["outside_panel_snv"],
            real_pass_snv_calls=real_dict["real_pass_snv"],
            real_FP_all=real_dict["real_FP_all"],
            real_FP_vaf_ge_0_5pct=real_dict["real_FP_vaf_ge_0_5pct"],
            ratio_empty_vaf_ge_0_5pct_to_real_FP=ratio,
            empty_calls_overlapping_real_FP_all=len({c[0] for c in ec} & real_fp_vars),
            empty_calls_overlapping_real_FP_vaf_ge_0_5pct=len({c[0] for c in ec if np.isfinite(c[1]) and c[1] >= VAF_CUT} & real_fp_vars_vaf),
        ))
        side.append(dict(
            caller=caller,
            empty_pass_snv_calls=es["pass_snv"],
            empty_snv_vaf_ge_0_5pct=es["vaf_ge_0_5pct"],
            empty_calls_per_Mb=es["pass_snv"] / PANEL_MB,
            real_pass_snv_calls=real_dict["real_pass_snv"],
            real_truth=len(truth), real_near=len(near),
            real_FP_all=real_dict["real_FP_all"],
            real_FP_vaf_ge_0_5pct=real_dict["real_FP_vaf_ge_0_5pct"],
            empty_over_real_FP_all=(es["pass_snv"] / real_dict["real_FP_all"]
                                    if np.isfinite(real_dict["real_FP_all"]) and real_dict["real_FP_all"] > 0 else np.nan),
            empty_over_real_FP_vaf_ge_0_5pct=ratio,
            real_FP_vaf_median=real_dict.get("real_FP_vaf_median", np.nan),
        ))
        print(f"{caller}: empty pass SNV={es['pass_snv']} (VAF>=0.5%: {es['vaf_ge_0_5pct']}, "
              f"bins {es['vaf_bins']}); real FP all={real_dict['real_FP_all']} "
              f"(VAF>=0.5%: {real_dict['real_FP_vaf_ge_0_5pct']}); ratio={ratio:.3f}"
              if np.isfinite(ratio) else
              f"{caller}: empty pass SNV={es['pass_snv']} (VAF>=0.5%: {es['vaf_ge_0_5pct']}); "
              f"real FP all={real_dict['real_FP_all']} (VAF>=0.5%: {real_dict['real_FP_vaf_ge_0_5pct']})")

    dfc = pd.DataFrame(rows)
    dfs = pd.DataFrame(side)
    out1 = os.path.join(a.tables, "C1_empty_target_calls.csv")
    out2 = os.path.join(a.tables, "C1_empty_target_vs_real.csv")
    dfc.to_csv(out1, index=False)
    dfs.to_csv(out2, index=False)
    print(f"\nWrote {out1}\nWrote {out2}")
    print("\n" + dfc.to_string(index=False))


if __name__ == "__main__":
    main()
