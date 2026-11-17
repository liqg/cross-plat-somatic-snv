# Provenance and licenses

## In-house components

| Component | Location | Notes |
|-----------|----------|-------|
| PosMix | `software/posmix/posmix`, `software/bin/posmix` | C++/HTSlib binary used to generate all simulations. Source: https://github.com/liqg/posmix |
| posfeature | `software/posmix/posfeature` | In-house per-read pileup used for Fig. 1b |
| bam-readcount (vendored build) | `software/bin/bam-readcount` | Used by the VarScan2 fpfilter step; a conda build is also installed |
| `reshapeRC` | `software/reshapeRC/reshapeRC.py` | Python implementation; output validated against 30 published `.rc.tsv.gz` files |
| `somaticfilter.pl` | `software/somaticfilter/somaticfilter.pl` | VarScan2 post-filter |
| SomaticXGB | `software/somaticxgb/` | Feature extraction R code, released model, NovaSeq X-only model, importance tables |
| `data.vcf` R package | `software/R/data.vcf_0.2.0.tar.gz` | VCF reader used by `SomaticXGB_features.R`; GPL (>= 2) |
| errorRates v1.3 | `software/errorRates/` | Mismatch-rate binary used for Figs. 3d/e and 4c and Tables S23/S24. The reporting pipeline and variant-calling components are not included |

## Third-party software

Versions used for the manuscript are pinned in `docker/environment.yml`.
Manta and Strelka are Python 2.7 packages and are pinned separately in
`docker/environment-legacy.yml`; the image creates both environments.

| Tool | Version |
|------|---------|
| fastp | 0.23.4 |
| BWA-MEM | 0.7.17-r1188 |
| Samtools / BCFtools / HTSlib | 1.17 |
| GATK (Mutect2) | 4.2.6.1 |
| Strelka2 | 2.9.10 |
| Manta | 1.6.0 |
| VarScan2 | 2.4.3 |
| bam-readcount | 1.0.1 |
| XGBoost (Python / R) | 1.7 |
| R | 4.2.3 |
| Python | 3.10 |

## Data

| Data | Source | License / terms |
|------|--------|-----------------|
| FASTQ | CNGBdb CNP0009527 | see CNGBdb project page |
| GIAB HG001-HG005 benchmark VCFs / high-confidence BEDs | NIST GIAB release FTP | public domain / GIAB terms |
| GRCh37 reference | UCSC / 1000 Genomes | public |
| Twist cfDNA Pan-Cancer Reference Standard v1 | Twist Bioscience | material terms; truth VCF is not redistributed |

## Code license

Unless stated otherwise, the code in this repository is released under the MIT
license.  Third-party components retain their upstream licenses:

- VarScan2 is released under the GNU LGPL (see the VarScan distribution).
- `data.vcf` is GPL (>= 2).
- GATK, Strelka2 and Manta are released under their respective licenses.
- PosMix is released with the manuscript; see https://github.com/liqg/posmix.

If you reuse the PosMix binary in a redistribution, confirm the license with the
corresponding author.
