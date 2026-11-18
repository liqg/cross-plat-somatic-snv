# cross-plat-somatic-snv

Code and workflows for the manuscript

> **Assessment and Development of Methods for Low-Frequency Somatic SNV
> Detection: A Multi-Platform High-Throughput Sequencing Perspective**

**Preprint (earlier version):** [bioRxiv 10.64898/2026.05.28.728367](https://doi.org/10.64898/2026.05.28.728367)

This repository reproduces the complete analysis chain:

```
FASTQ
  │  fastp + BWA-MEM + MarkDuplicates
  ▼
aligned BAMs
  │  PosMix position-specific fragment replacement
  ▼
simulated tumour / matched-normal BAM pairs
  │  Mutect2 · Strelka2 · VarScan2 (± fpfilter) · force-calling · bam-readcount
  ▼
candidate variants + 17 features
  │  SomaticXGB training / prediction
  ▼
benchmark metrics, Twist cfDNA validation, error-rate analyses
  │
  ▼
main figures (Fig. 1-4), Additional file 1 (Tables S1-S35),
Additional file 2 (Supplementary Figures S1-S12)
```

The repository covers the complete analysis chain, including the revision
analyses added in response to reviewer comments (A1-A4, B1-B1c, C1-C3,
D1-D1b, G1, R24, R27), a container and smoke tests that run without sequencing
data.

## Contents

| Path | What it contains |
|------|------------------|
| `workflow/align/` | `fastp` + BWA-MEM alignment (`align.smk`) |
| `workflow/posmix/` | PosMix + Mutect2 / Strelka2 / VarScan2 + force-calling (`somatic_snv_posmix.smk`) |
| `workflow/somaticxgb/` | Feature matrix builder, training grid search, prediction (R) |
| `software/errorRates/` | `errorRates-v1.3` binary and wrapper for mismatch-rate tables |
| `workflow/drivers/` | Generate per-pair configs from `config/*.tsv` |
| `workflow/pos_files/` | Position–frequency tables used for every simulation |
| `software/` | PosMix / posfeature binaries, SomaticXGB model and R code, `data.vcf`, `reshapeRC`, `somaticfilter.pl`, VarScan2 jar, errorRates binary |
| `analysis/` | Error rates, consistency, benchmark, model, test-set and Twist analyses; figure and table builders |
| `analysis/r_figures/` | Original R notebook code, vendored helpers, schematics and derived plotting inputs for the submitted main data panels |
| `analysis/supp_figure_data/` | Derived tables for Supplementary Figures S1-S12 |
| `reference/panel/` | Capture-panel BED and donor–background intersection BEDs |
| `released/` | The released Additional files for the manuscript |
| `tests/` | Fast smoke tests and synthetic fixtures |
| `docs/` | Data layout, figure-to-code map, reproduction guide, provenance |

## Quick start

```bash
# 1. Get the code
git clone https://github.com/liqg/cross-plat-somatic-snv.git
cd cross-plat-somatic-snv

# 2. Configure paths (reference genome, FASTQ root, work directory)
cp config/config.example.yaml config/config.yaml
$EDITOR config/config.yaml

# 3. Check tools and paths
python workflow/run.py doctor

# 4. Fast tests and submission preflight (no sequencing data needed)
python tests/test_smoke.py -v
python analysis/preflight.py
```

A container with all third-party tools is provided:

```bash
docker build -t cross-plat-somatic-snv .
docker run --rm -v /path/to/data:/data cross-plat-somatic-snv doctor
```

The image builds a Python 3.10 environment for the pipeline and a small
Python 2.7 side environment for Manta 1.6.0 and Strelka 2.9.10, whose conda
packages cannot coexist with Python 3.10.  In restricted networks, add
`--network=host` and optionally route mamba through a proxy:

```bash
docker build --network=host \
  --build-arg APT_MIRROR=mirrors.aliyun.com \
  --build-arg PROXY_HOST=<proxy-host> \
  --build-arg PROXY_PORT=1081 \
  -t cross-plat-somatic-snv .
```

The analysis-only image (figure and table scripts) is much smaller:

```bash
docker build -f docker/Dockerfile.analysis -t cpsnv-analysis .
docker run --rm cpsnv-analysis python analysis/07_figures_tables/make_all.py --step fig4 --dry-run
```

## Data

Raw FASTQ data: CNGBdb project **CNP0009527**
(DOI [10.26036/CNP0009527](https://doi.org/10.26036/CNP0009527)).

External references are not redistributed:

- GRCh37 reference FASTA (with BWA index, `.fai`, `.dict`)
- GIAB v4.2.1 benchmark VCFs and high-confidence BEDs for HG001-HG005
- Twist cfDNA Pan-Cancer Reference Standard v1 VCF and BAMs

The expected directory layout is in [`docs/data_layout.md`](docs/data_layout.md).
Small annotations (panel BED, intersection BEDs, position tables, released
supplementary files) are included.  The released SomaticXGB feature matrices
are in [`data/somaticxgb/`](data/somaticxgb/README.md): 493,937 training
candidates and 137,131 test candidates, with checksums and generation notes.

## Running the pipeline

The stages are independent Snakemake workflows driven by
`workflow/run.py`:

```bash
# generate JSON configs for every stage
python workflow/run.py configs --kind all

# FASTQ -> BAM
python workflow/run.py align --cores 64

# PosMix + callers for all six training pairs, two test pairs and the
# single-pair benchmark
python workflow/run.py simulate --role train --cores 100
python workflow/run.py simulate --role test  --cores 100
python workflow/run.py simulate --role benchmark --cores 100

# SomaticXGB feature matrix + training + prediction
Rscript workflow/somaticxgb/train_SomaticXGB.R \
    --pairs $CPSNV_WORK_ROOT/4.train/{12,13,21,23,31,32} \
    --outdir $CPSNV_WORK_ROOT/5.somaticxgb/train
python workflow/run.py predict --pairs 45 54
```

Use `--dry-run` to inspect every command first. Each PosMix pair runs in its
own working directory because `data[].id` is a platform name; this prevents
different donor–background pairs from overwriting each other.

Full commands, including errorRates and the Twist analyses, are in
[`docs/reproduction.md`](docs/reproduction.md).

## Figures and tables

Every main and supplementary panel is mapped to the script that produces it in
[`docs/figure_code_map.md`](docs/figure_code_map.md). In brief:

| Output | Script |
|--------|--------|
| Fig. 1b | `analysis/r_figures/01_fig1b_fig2d_fig3.R` (exact submitted panel); `analysis/02_consistency/consistency_from_posfeature.py` (consistency data preparation and optional preview) |
| Fig. 2d | `analysis/r_figures/01_fig1b_fig2d_fig3.R` (exact submitted panel) |
| Fig. 3a-e | `analysis/r_figures/01_fig1b_fig2d_fig3.R` (exact submitted panels) |
| Fig. 4 | `analysis/r_figures/02_fig4.R` (exact submitted panels) |
| Tables S1-S35 | `analysis/07_figures_tables/build_supplement_v4.py` (+ derived tables) |
| Figs. S1-S12 | `analysis/07_figures_tables/make_bmc_supp_figures.py` |

The schematics in Fig. 1a and Fig. 2a-c are vector artwork assembled in
Illustrator; they are not generated from data.  The R figure scripts output
only the data panels; the schematics are merged manually in the manuscript. The `released/` folder contains
the submitted Additional files.

## Compute requirements

- Alignment and simulation: ~100 cores per pair, 32-64 GB RAM; the full
  72-dataset alignment is the dominant cost.
- SomaticXGB training matrix: 493,937 rows x 17 features (66,972 positive /
  426,965 negative); 8-32 threads, ~8 GB RAM.
- Figure and table scripts: a laptop with 8 GB RAM.

## Citation

If you use this code or PosMix, please cite the manuscript (see
`CITATION.cff`); the PosMix software references the same article.

## License

MIT for the code in this repository. Third-party tools retain their own
licenses; see [`docs/provenance.md`](docs/provenance.md).
