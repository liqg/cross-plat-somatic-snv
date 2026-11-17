#!/usr/bin/env python3
"""Aggregate ``.er.txt.gz`` files into error_rates_by_q_clean.csv.

This reproduces the table used for Fig. 3d/e and the Q0/Q30 values in the
manuscript.  Each row is one sequencing library and one quality threshold.

Usage
-----
    python analysis/01_error_rates/build_error_rates_by_q.py \
        --root $CPSNV_WORK_ROOT/1.errorRates/pipeline/bwamem \
        --out $CPSNV_RESULTS/tables/error_rates_by_q_clean.csv
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common  # noqa: E402

THRESHOLDS = [0, 10, 20, 30, 40]


def parse_sample(sample):
    prefix = sample.split("_")[0]
    m = re.search(r"(HG\d{3})", sample)
    return prefix, (m.group(1) if m else "")


def one_file(path):
    df = pd.read_csv(path, sep="\t", usecols=["ref", "base", "qual", "count"])
    total = int(df["count"].sum())
    sample = path.parent.name
    platform, hg = parse_sample(sample)
    rows = []
    for q in THRESHOLDS:
        sub = df[df["qual"] >= q]
        n = int(sub["count"].sum())
        err = int(sub.loc[sub["ref"] != sub["base"], "count"].sum())
        rows.append({"sample": sample, "platform": platform, "hg": hg, "q": q,
                     "frac": n / total if total else float("nan"),
                     "err": err / n if n else float("nan"), "n": n})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(common.ER_1_ROOT / "pipeline" / "bwamem"))
    ap.add_argument("--out", default=str(common.TABLES / "error_rates_by_q_clean.csv"))
    a = ap.parse_args()
    root = Path(a.root)
    files = sorted(root.glob("*/*.er.txt.gz"))
    if not files:
        raise SystemExit(f"No .er.txt.gz files under {root}")
    rows = []
    for f in files:
        rows.extend(one_file(f))
    df = pd.DataFrame(rows)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.out, index=False)
    print("wrote", a.out, len(df), "rows from", len(files), "files")


if __name__ == "__main__":
    main()
