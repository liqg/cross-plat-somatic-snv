#!/usr/bin/env python3
"""Prepare the GIAB inputs used by the donor-fragment audit and negative audit.

Outputs
-------
  $CPSNV_DATA_ROOT/posfiles/donor_vcf/{sample}_bed45panel.vcf
  $CPSNV_RESULTS/intermediate/giab_panel_vars.pkl

The VCF is the GIAB v4.2.1 benchmark call set restricted to the capture panel,
with multi-allelic records left intact (G1 splits them when computing dosage).
The pickle maps each sample to a DataFrame with columns ``var``, ``chrom`` and
``pos`` for all panel-covered SNVs.

Usage
-----
    python analysis/05_test_set/prepare_giab_inputs.py \
        --giab-dir data/giab/v4.2.1 --panel-bed reference/panel/T364V1.hg19.bed
"""
from __future__ import annotations

import argparse
import pickle
import subprocess
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common  # noqa: E402

SAMPLES = ["HG001", "HG002", "HG003", "HG004", "HG005"]


def vcf_for(giab_dir, sample):
    for pattern in (f"{sample}.vcf.gz", f"{sample}*.vcf.gz", f"{sample}.vcf"):
        hits = sorted(Path(giab_dir).glob(pattern))
        if hits:
            return hits[0]
    raise FileNotFoundError(f"No GIAB VCF for {sample} under {giab_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--giab-dir", default=str(common.GIAB_DIR))
    ap.add_argument("--panel-bed", default=str(common.PANEL_BED))
    ap.add_argument("--vcf-out", default=str(common.DATA_ROOT / "posfiles" / "donor_vcf"))
    ap.add_argument("--pkl-out", default=str(common.GIAB_PANEL_VARS))
    a = ap.parse_args()

    vcf_out = Path(a.vcf_out); vcf_out.mkdir(parents=True, exist_ok=True)
    pkl = {}
    for sample in SAMPLES:
        src = vcf_for(a.giab_dir, sample)
        dst = vcf_out / f"{sample}_bed45panel.vcf"
        cmd = ["bcftools", "view", "-T", a.panel_bed, "-Ou", str(src),
               "-o", str(dst.with_suffix(".bcf"))]
        subprocess.run(cmd, check=True)
        with open(dst, "w") as fh:
            subprocess.run(["bcftools", "view", str(dst.with_suffix(".bcf"))],
                           check=True, stdout=fh)
        dst.with_suffix(".bcf").unlink(missing_ok=True)
        rows = []
        q = subprocess.run(["bcftools", "query", "-f", "%CHROM\t%POS\t%REF\t%ALT\n", str(dst)],
                           capture_output=True, text=True, check=True)
        for line in q.stdout.splitlines():
            if not line:
                continue
            chrom, pos, ref, alt = line.split("\t")
            if len(ref) != 1 or len(alt) != 1 or "," in alt:
                continue
            chrom = chrom.replace("chr", "")
            rows.append({"chrom": chrom, "pos": int(pos), "ref": ref, "alt": alt,
                         "var": f"{chrom}:{pos}:{ref}:{alt}"})
        pkl[sample] = pd.DataFrame(rows)
        print(sample, len(rows), "panel SNVs ->", dst)

    Path(a.pkl_out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.pkl_out, "wb") as fh:
        pickle.dump(pkl, fh)
    print("wrote", a.pkl_out)


if __name__ == "__main__":
    main()
