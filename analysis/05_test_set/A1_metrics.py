#!/usr/bin/env python3
"""A1 metrics: exact replication of scripts/make_fig3_data.py for one condition.

Truth set: pf45_low loci with inter-variant distance >= 50 bp (near variants with
distance < 50 bp are excluded from both truth and calls, as in make_fig3_data.py).
Calls: PASS SNVs only (VCF FILTER PASS/.), SomaticXGB pred > 0.1.
Compared: original matched normal (4.test/45) vs non-overlapping normal rerun.
"""

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
import common  # noqa: E402
import argparse
import os
import subprocess

import numpy as np
import pandas as pd

BCFTOOLS = os.environ.get("BCFTOOLS", "bcftools")


def add_dist(df):
    df = df.copy()
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


def vcfvars(path):
    res = set()
    q = subprocess.run(
        [BCFTOOLS, "query", "-f", "%CHROM\t%POS\t%REF\t%ALT\t%FILTER\n", path],
        capture_output=True, text=True, check=True)
    for line in q.stdout.strip().split("\n"):
        if not line:
            continue
        c, p, r, a, flt = line.split("\t")
        if flt not in ("PASS", ".") and "PASS" not in flt:
            continue
        if len(r) == 1 and len(a) == 1 and "," not in a:
            res.add(c.replace("chr", "") + ":" + p + ":" + r + ":" + a)
    return res


def callsets(root, sample):
    s = vcfvars(f"{root}/strelka/{sample}/results/variants/somatic.snvs.vcf.gz")
    m = vcfvars(f"{root}/mutect/{sample}.filter.vcf.gz")
    v = vcfvars(f"{root}/varscan_novo/{sample}.snp.pass.vcf.gz")
    pred = pd.read_csv(f"{root}/pred/{sample}_pred.txt.gz", sep="\t", compression="gzip")
    x = set(pred.loc[pred.pred > 0.1, "var"])
    return {
        "SomaticXGB 0.1": x,
        "Strelka2": s,
        "Mutect2": m,
        "Strelka2_Mutect2": s | m,
        "VarScan2_fpfilter": v,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pf", required=True)
    ap.add_argument("--orig-root", required=True)
    ap.add_argument("--rerun-root", required=True)
    ap.add_argument("--sample", required=True)
    ap.add_argument("--overlap-csv", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    pf = pd.read_csv(a.pf, sep="\t", header=None,
                     names=["CHROM", "POS", "P", "REF", "ALT", "GT", "f", "b", "GT2", "group", "AF"])
    pf = add_dist(pf)
    pf["var"] = pf.CHROM + ":" + pf.POS.astype(str) + ":" + pf.REF + ":" + pf.ALT
    truth = set(pf.loc[pf.dist >= 50, "var"])
    near = set(pf.loc[pf.dist < 50, "var"])

    ov = pd.read_csv(a.overlap_csv)
    ov = ov.set_index("condition")

    conditions = [("original", a.orig_root), ("nonoverlap", a.rerun_root)]
    rows = []
    orig_sets = {}
    for cond, root in conditions:
        sets = callsets(root, a.sample)
        if cond == "original":
            orig_sets = sets
        for name in ["SomaticXGB 0.1", "Strelka2", "Mutect2", "Strelka2_Mutect2", "VarScan2_fpfilter"]:
            calls = set(sets[name]) - near
            tp = len(calls & truth)
            fp = len(calls - truth)
            fn = len(truth - calls)
            rec = tp / (tp + fn) if tp + fn else np.nan
            prec = tp / (tp + fp) if tp + fp else np.nan
            f1 = 2 * rec * prec / (rec + prec) if rec + prec else np.nan
            row = {
                "condition": cond, "method": name,
                "TP": tp, "FP": fp, "FN": fn,
                "recall": rec, "precision": prec, "F1": f1,
                "n_raw_pass_calls": len(sets[name]),
                "n_calls_after_near_filter": len(calls),
                "n_truth": len(truth),
                "normal_qname_overlap_pct": ov.loc[cond, "qname_overlap_pct"],
                "normal_read_overlap_pct": ov.loc[cond, "read_overlap_pct"],
            }
            if cond == "nonoverlap":
                o = orig_sets[name] - near
                c = calls
                row["added_vs_original"] = len(c - o)
                row["lost_vs_original"] = len(o - c)
                row["jaccard_vs_original"] = len(c & o) / len(c | o) if c | o else np.nan
            rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(a.out, index=False)
    print(df.to_string(index=False))
    print(f"\nWrote {a.out}")


if __name__ == "__main__":
    main()
