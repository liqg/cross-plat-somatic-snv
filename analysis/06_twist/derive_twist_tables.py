#!/usr/bin/env python3
"""Derive the numerical tables behind Fig. 4 and Tables S9-S11.

Inputs
------
  --twist-vcf     Twist cfDNA Pan-Cancer Reference Standard v1 VCF (GRCh37)
  --panel-bed     capture-panel BED (T364V1.hg19.bed)
  --phys-root     work/6.twist_pair  (physical 1 % standards)
  --mix-root      work/8.twist_HG001_mix (Twist/HG001 PosMix simulation)
  --er-root  work/9.errorRates_bq30_twist_wt_PE15M/pipeline/bwamem

Outputs (under $CPSNV_RESULTS/tables)
-------------------------------------
  twist_truth_221.tsv
  supp_table_S9a_original_twist.csv
  supp_table_S9b_hybrid_twist.csv
  fig4b_fp_substitutions.csv
  fig4c_twist0_substitution_error.csv

The truth set is the panel-covered SNV subset of the Twist VCF after removing
the five sites excluded in the original analysis.
"""
from __future__ import annotations

import argparse
import bisect
import gzip
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common  # noqa: E402

EXCLUDED = {
    "2:29940529:A:G", "6:117725578:T:C", "7:55214348:C:A",
    "10:43613843:G:A", "20:57478807:C:A",
}
PREFIX = {"FA": "FASTASeq 300", "GL": "GenoLab M", "SF": "SURFSeq 5000",
          "NV": "NovaSeq 6000", "NX": "NovaSeq X", "T7": "DNBSEQ-T7",
          "MGISEQ-T7": "DNBSEQ-T7"}


def read_bed(path):
    df = pd.read_csv(path, sep="\t", header=None, comment="#",
                     names=["chrom", "start", "end"])
    df["chrom"] = df["chrom"].astype(str).str.replace("^chr", "", regex=True)
    out = {}
    for chrom, sub in df.groupby("chrom", sort=False):
        iv = sorted(zip(sub["start"].astype(int), sub["end"].astype(int)))
        merged = []
        for s, e in iv:
            if merged and s <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], e))
            else:
                merged.append((s, e))
        out[chrom] = merged
    return out


def overlaps(intervals, chrom, pos):
    iv = intervals.get(chrom)
    if not iv:
        return False
    p = int(pos) - 1  # BED is 0-based half-open
    i = bisect.bisect_right([s for s, _ in iv], p) - 1
    return i >= 0 and iv[i][0] <= p < iv[i][1]


def twist_truth(vcf, panel_bed, out_tsv):
    intervals = read_bed(panel_bed)
    cmd = ["bcftools", "query", "-f", "%CHROM\t%POS\t%REF\t%ALT\n", str(vcf)]
    res = subprocess.run(cmd, capture_output=True, text=True, check=True)
    rows = []
    for line in res.stdout.splitlines():
        if not line:
            continue
        chrom, pos, ref, alt = line.split("\t")
        chrom = chrom.replace("chr", "")
        if len(ref) != 1 or len(alt) != 1 or "," in alt:
            continue
        var = f"{chrom}:{pos}:{ref}:{alt}"
        if var in EXCLUDED or not overlaps(intervals, chrom, pos):
            continue
        rows.append({"chrom": chrom, "pos": int(pos), "ref": ref, "alt": alt, "var": var})
    df = pd.DataFrame(rows).drop_duplicates("var")
    df.to_csv(out_tsv, sep="\t", index=False)
    print(f"truth: {len(df)} panel-covered SNVs -> {out_tsv}")
    return set(df["var"])


def _sample_platform(sample):
    prefix = sample.split("_")[0]
    return PREFIX.get(prefix, common.platform_name(prefix))


def _replicate(sample):
    parts = sample.split("_")
    return int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 1


def physical_metrics(root, truth, out_csv):
    rows = []
    for pred_file in sorted((root / "pred").glob("*_pred.txt.gz")):
        sample = pred_file.name.replace("_pred.txt.gz", "")
        pred = pd.read_csv(pred_file, sep="\t", usecols=["var", "pred"])
        platform = _sample_platform(sample)
        for thr in (0.05, 0.1, 0.2):
            calls = set(pred.loc[pred["pred"] > thr, "var"])
            tp = len(calls & truth); fp = len(calls - truth); fn = len(truth - calls)
            rows.append(_metric_row(platform, _replicate(sample), sample, "SomaticXGB", thr, tp, fp, fn, truth))
        sets = {}
        sv = root / "strelka" / sample / "results" / "variants" / "somatic.snvs.vcf.gz"
        mv = root / "mutect" / f"{sample}.filter.vcf.gz"
        vv = root / "varscan_novo" / f"{sample}.snp.pass.vcf.gz"
        if sv.exists() and mv.exists():
            sets["Strelka2_Mutect2"] = common.vcf_variants(sv) | common.vcf_variants(mv)
        if vv.exists():
            sets["VarScan2_fpfilter"] = common.vcf_variants(vv)
        for method, calls in sets.items():
            tp = len(calls & truth); fp = len(calls - truth); fn = len(truth - calls)
            rows.append(_metric_row(platform, _replicate(sample), sample, method, None, tp, fp, fn, truth))
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    print("wrote", out_csv, len(df), "rows")
    return df


def hybrid_metrics(root, truth, out_csv):
    rows = []
    for pred_file in sorted((root / "pred").glob("*_pred.txt.gz")):
        name = pred_file.name.replace("_reads140M_pred.txt.gz", "").replace("_pred.txt.gz", "")
        # names look like FASTASeq_300-1
        platform = PREFIX.get(name.split("-")[0], common.platform_name(name.split("-")[0]))
        rep = int(name.split("-")[-1]) if "-" in name else 1
        pred = pd.read_csv(pred_file, sep="\t", usecols=["var", "pred"])
        calls = set(pred.loc[pred["pred"] > 0.1, "var"])
        tp = len(calls & truth); fp = len(calls - truth); fn = len(truth - calls)
        rows.append({"Platform": platform, "Replicate": rep, "Method": "SomaticXGB",
                     "Recall": tp / len(truth), "Precision": tp / (tp + fp) if tp + fp else 0.0,
                     "F1": 2 * tp / (2 * tp + fp + fn) if tp else 0.0})
        for method, pattern in (("Strelka2_Mutect2", "strelka"), ("VarScan2_fpfilter", "varscan_novo")):
            sid = name + "_reads140M"
            if method == "Strelka2_Mutect2":
                sv = root / "strelka" / sid / "results" / "variants" / "somatic.snvs.vcf.gz"
                mv = root / "mutect" / f"{sid}.filter.vcf.gz"
                if not (sv.exists() and mv.exists()):
                    continue
                calls = common.vcf_variants(sv) | common.vcf_variants(mv)
            else:
                vv = root / "varscan_novo" / f"{sid}.snp.pass.vcf.gz"
                if not vv.exists():
                    continue
                calls = common.vcf_variants(vv)
            tp = len(calls & truth); fp = len(calls - truth); fn = len(truth - calls)
            rows.append({"Platform": platform, "Replicate": rep, "Method": method,
                         "Recall": tp / len(truth), "Precision": tp / (tp + fp) if tp + fp else 0.0,
                         "F1": 2 * tp / (2 * tp + fp + fn) if tp else 0.0})
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    print("wrote", out_csv, len(df), "rows")
    return df


def _metric_row(platform, rep, sample, method, thr, tp, fp, fn, truth):
    return {"Platform": platform, "Replicate": rep, "Sample": sample, "Method": method,
            "Threshold": thr, "TP": tp, "FP": fp, "FN": fn,
            "Recall": tp / len(truth), "Precision": tp / (tp + fp) if tp + fp else 0.0,
            "F1": 2 * tp / (2 * tp + fp + fn) if tp else 0.0}


def fp_substitutions(root, truth, out_csv):
    rows = []
    for pred_file in sorted((root / "pred").glob("*_pred.txt.gz")):
        sample = pred_file.name.replace("_pred.txt.gz", "")
        platform = _sample_platform(sample)
        pred = pd.read_csv(pred_file, sep="\t", usecols=["var", "pred"])
        fp = pred[(pred["pred"] > 0.1) & (~pred["var"].isin(truth))]
        for var in fp["var"]:
            _, _, ref, alt = var.split(":")
            rows.append({"Method": "SomaticXGB (0.1)", "Platform_abbr": sample.split("_")[0],
                         "Substitution": ref + alt, "Platform": platform})
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    print("wrote", out_csv, len(df), "rows")
    return df


def twist0_error(root, out_csv):
    rows = []
    for sid_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        er = sid_dir / f"{sid_dir.name}.er.txt.gz"
        if not er.exists():
            continue
        parts = sid_dir.name.split("_")
        platform = PREFIX.get(parts[0], common.platform_name(parts[0]))
        agg = {}
        total = {}
        for chunk in pd.read_csv(er, sep="\t", usecols=["ref", "base", "qual", "count"],
                                 chunksize=2_000_000):
            chunk = chunk[chunk["qual"] >= 30]
            for (ref, base), grp in chunk.groupby(["ref", "base"]):
                k = (str(ref).upper(), str(base).upper())
                agg[k] = agg.get(k, 0) + int(grp["count"].sum())
            for ref, grp in chunk.groupby("ref"):
                total[str(ref).upper()] = total.get(str(ref).upper(), 0) + int(grp["count"].sum())
        for (ref, base), n in agg.items():
            if ref == base or ref not in total or total[ref] == 0:
                continue
            rows.append({"Platform": platform, "Substitution": ref + base,
                         "Mismatches": n, "Q30_bases": total[ref],
                         "Error_rate": n / total[ref]})
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    print("wrote", out_csv, len(df), "rows")
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--twist-vcf", default=str(common.TWIST_TRUTH_VCF))
    ap.add_argument("--panel-bed", default=str(common.PANEL_BED))
    ap.add_argument("--phys-root", default=str(common.TWIST_PHYS_ROOT))
    ap.add_argument("--mix-root", default=str(common.TWIST_MIX_ROOT))
    ap.add_argument("--er-root", default=str(common.ER_9_ROOT / "pipeline" / "bwamem"))
    ap.add_argument("--outdir", default=str(common.TABLES))
    a = ap.parse_args()
    out = Path(a.outdir); out.mkdir(parents=True, exist_ok=True)

    truth = twist_truth(a.twist_vcf, a.panel_bed, out / "twist_truth_221.tsv")
    physical_metrics(Path(a.phys_root), truth, out / "supp_table_S9a_original_twist.csv")
    hybrid_metrics(Path(a.mix_root), truth, out / "supp_table_S9b_hybrid_twist.csv")
    fp_substitutions(Path(a.phys_root), truth, out / "fig4b_fp_substitutions.csv")
    if Path(a.er_root).exists():
        twist0_error(Path(a.er_root), out / "fig4c_twist0_substitution_error.csv")
    else:
        print("skip 0% error table (errorRates directory not found):", a.er_root)


if __name__ == "__main__":
    main()
