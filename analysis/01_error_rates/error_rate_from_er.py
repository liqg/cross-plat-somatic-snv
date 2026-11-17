#!/usr/bin/env python3
"""Compute targeted-panel error rates from errorRates .er.txt.gz tables.

The errorRates table is expected to contain the columns:
    ref, base, qual, count
where `ref` is the reference base, `base` is the observed base, `qual` is the
reported base quality, and `count` is the number of observations.

This script reports, for each requested quality threshold Q:
    - error rate  = mismatches / bases with qual >= Q
    - fraction of bases retained at qual >= Q
It also writes the per-quality error rate for calibration plots.

Example:
    python error_rate_from_er.py sample.er.txt.gz -o sample.error_rates.tsv
"""
import argparse
import os
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("er", help="errorRates .er.txt.gz file")
    ap.add_argument("-o", "--out", default=None, help="output TSV (default: stdout)")
    ap.add_argument("-q", "--thresholds", default="0,10,20,30,40",
                    help="comma-separated quality thresholds")
    args = ap.parse_args()

    df = pd.read_csv(args.er, sep="\t", usecols=["ref", "base", "qual", "count"])
    total = int(df["count"].sum())
    thresholds = [int(x) for x in args.thresholds.split(",")]

    rows = []
    for q in thresholds:
        sub = df[df["qual"] >= q]
        n = int(sub["count"].sum())
        err = int(sub.loc[sub["ref"] != sub["base"], "count"].sum())
        rows.append({
            "sample": os.path.basename(os.path.dirname(args.er)) or os.path.basename(args.er),
            "quality_threshold": q,
            "bases": n,
            "mismatches": err,
            "error_rate": err / n if n else float("nan"),
            "fraction_retained": n / total if total else float("nan"),
        })

    # per-quality calibration
    per_q = (df.groupby("qual")
               .apply(lambda x: pd.Series({
                   "bases": x["count"].sum(),
                   "mismatches": x.loc[x["ref"] != x["base"], "count"].sum(),
               }), include_groups=False)
               .reset_index())
    per_q["error_rate"] = per_q["mismatches"] / per_q["bases"]
    per_q["fraction_retained"] = per_q["bases"] / total
    per_q.insert(0, "sample", os.path.basename(os.path.dirname(args.er)) or os.path.basename(args.er))

    out = pd.DataFrame(rows)
    if args.out:
        base = args.out
        out.to_csv(base, sep="\t", index=False)
        per_q.to_csv(base.replace(".tsv", ".per_quality.tsv"), sep="\t", index=False)
        print(f"wrote {base} and {base.replace('.tsv', '.per_quality.tsv')}")
    else:
        print(out.to_string(index=False))


if __name__ == "__main__":
    main()
