#!/usr/bin/env python3
"""Fig. 1b: specified versus calculated VAF after PosMix.

Runs ``posfeature`` on the donor, background and simulated-tumour BAMs, then
compares the observed alternate-allele fraction with the expectation

    expected = donor_AF * P + (1 - P) * background_AF

where P is the mixing proportion in the position file.  This is the Python
port of the R notebook code used for the manuscript.

Usage
-----
    python analysis/02_consistency/consistency_from_posfeature.py \
        --workdir $CPSNV_WORK_ROOT/2.consistency \
        --posfeature software/bin/posfeature \
        --genome $CPSNV_DATA_ROOT/reference/GRCh37/genome.fasta

The PosMix tumour BAMs must already exist under
``{workdir}/posmix/{id}_T_reads150M.bam`` (run workflow/posmix).
"""
from __future__ import annotations

import argparse
import gzip
import os
import shutil
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common  # noqa: E402

COMPLEMENT = str.maketrans("ACGTacgt", "TGCAtgca")
BASES = {"A", "C", "G", "T"}


def run_posfeature(posfeature, genome, posfile, bam, out):
    if out.exists() and out.stat().st_size > 0:
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [posfeature, "-g", genome, "-p", posfile, "-b", bam]
    print("RUN:", " ".join(cmd), "| bgzip >", out, flush=True)
    with gzip.open(out, "wt") as fh:
        p1 = subprocess.Popen(cmd, stdout=subprocess.PIPE)
        p2 = subprocess.Popen(["bgzip", "-c"], stdin=p1.stdout, stdout=fh)
        p1.stdout.close()
        rc1 = p1.wait()
        rc2 = p2.wait()
        if rc1 or rc2:
            raise RuntimeError(f"posfeature failed ({rc1}) or bgzip failed ({rc2})")


def count_alleles(path):
    """Return per-variant depth and alternate-allele fraction from posfeature."""
    usecols = ["type", "chrom", "pos", "isfwd", "ref", "alt", "qual"]
    frames = []
    for chunk in pd.read_csv(path, sep="\t", usecols=usecols, chunksize=2_000_000,
                             dtype={"chrom": str}):
        c = chunk[(chunk["type"].isin(["=", "X"])) & (chunk["qual"] >= 0)]
        c = c[c["ref"].isin(BASES) & c["alt"].isin(BASES)]
        rev = c["isfwd"] == 0
        c.loc[rev, "ref"] = c.loc[rev, "ref"].str.translate(COMPLEMENT).str.upper()
        c.loc[rev, "alt"] = c.loc[rev, "alt"].str.translate(COMPLEMENT).str.upper()
        c["var"] = (c["chrom"] + ":" + c["pos"].astype(str) + ":"
                    + c["ref"] + ":" + c["alt"])
        frames.append(c.groupby(["chrom", "pos", "ref", "alt", "var"]).size()
                      .rename("ad").reset_index())
    d = pd.concat(frames, ignore_index=True)
    d = d.groupby(["chrom", "pos", "ref", "alt", "var"], as_index=False)["ad"].sum()
    d["dp"] = d.groupby(["chrom", "pos"])["ad"].transform("sum")
    d["freq"] = d["ad"] / d["dp"]
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workdir", default=str(common.CONSISTENCY_ROOT))
    ap.add_argument("--posfeature", default=str(common.REPO_ROOT / "software" / "bin" / "posfeature"))
    ap.add_argument("--genome", default=str(common.GENOME_FASTA))
    ap.add_argument("--outdir", default=str(common.RESULTS_ROOT))
    ap.add_argument("--skip-run", action="store_true")
    a = ap.parse_args()

    work = Path(a.workdir)
    outdir = Path(a.outdir)
    posfeatures = work / "posmix"
    posfeatures.mkdir(parents=True, exist_ok=True)
    cfg_path = work / "config.json"
    if not cfg_path.exists():
        raise SystemExit(f"Missing {cfg_path}; run workflow/run.py configs --kind consistency")
    import json
    cfg = json.load(open(cfg_path))
    posfile = cfg["posfreq"] if "posfreq" in cfg else cfg["posfreq_file"]

    posfreq = common.read_position_table(posfile)
    posfreq = posfreq.rename(columns={"VAF": "freq_exp1"}) if "VAF" in posfreq else posfreq
    posfreq = posfreq.rename(columns={"P": "freq_replace"})
    posfreq["var"] = (posfreq["CHROM"].astype(str) + ":" + posfreq["POS"].astype(str)
                      + ":" + posfreq["REF"].astype(str) + ":" + posfreq["ALT"].astype(str))

    rows = []
    for entry in cfg["data"]:
        ident = entry["id"]
        tumor = posfeatures / f"{ident}_T_reads{cfg['tumor_reads'][0]}M.bam"
        donor = Path(entry["tumor"])
        background = Path(entry["normal"])
        f_t = posfeatures / f"{ident}_T_reads{cfg['tumor_reads'][0]}M.bam.pf.txt.gz"
        f_d = posfeatures / f"{ident}_donor.pf.txt.gz"
        f_b = posfeatures / f"{ident}_backgroud.pf.txt.gz"
        if not a.skip_run:
            for bam, out in ((tumor, f_t), (donor, f_d), (background, f_b)):
                if not bam.exists():
                    raise SystemExit(f"Missing BAM: {bam}")
                run_posfeature(a.posfeature, a.genome, posfile, str(bam), out)
        t = count_alleles(f_t)[["var", "freq", "dp"]].rename(columns={"freq": "freq_sim", "dp": "dp_sim"})
        d = count_alleles(f_d)[["var", "freq", "dp"]].rename(columns={"freq": "freq_donor", "dp": "dp_donor"})
        b = count_alleles(f_b)[["var", "freq", "dp"]].rename(columns={"freq": "freq_bg", "dp": "dp_bg"})
        df = (posfreq[["var", "freq_exp1", "freq_replace", "GT"]]
              .merge(t, on="var", how="inner")
              .merge(d, on="var", how="inner")
              .merge(b, on="var", how="inner"))
        df["freq_exp2"] = df["freq_donor"] * df["freq_replace"]
        df["freq_exp3"] = df["freq_donor"] * df["freq_replace"] + (1 - df["freq_replace"]) * df["freq_bg"]
        df["Platform"] = common.platform_name(ident)
        df["Sample"] = ident
        rows.append(df)

    all_df = pd.concat(rows, ignore_index=True)
    out_tsv = outdir / "tables" / "fig1b_consistency.tsv"
    out_tsv.parent.mkdir(parents=True, exist_ok=True)
    all_df.to_csv(out_tsv, sep="\t", index=False)
    print("wrote", out_tsv, len(all_df), "rows")

    platforms = [p for p in common.PLATFORM_ORDER if p in set(all_df["Platform"])]
    fig, axes = plt.subplots(1, len(platforms), figsize=(7.2, 1.6), sharex=True, sharey=True)
    if len(platforms) == 1:
        axes = [axes]
    for ax, plat in zip(axes, platforms):
        sub = all_df[all_df["Platform"] == plat]
        ax.scatter(sub["freq_exp3"] * 100, sub["freq_sim"] * 100, s=2,
                   c=common.PLATFORM_COLORS.get(plat), alpha=0.6, linewidths=0)
        lim = max(sub["freq_exp3"].max(), sub["freq_sim"].max()) * 100 * 1.05
        ax.plot([0, lim], [0, lim], color="black", lw=0.6)
        ax.set_title(plat, fontsize=7)
        ax.set_xlabel("Specified AF (%)", fontsize=6)
        ax.tick_params(labelsize=5.5)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Calculated AF (%)", fontsize=6)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(outdir / "figures" / f"Fig1b_consistency.{ext}", dpi=300, bbox_inches="tight")
    print("wrote", outdir / "figures" / "Fig1b_consistency.pdf")


if __name__ == "__main__":
    main()
