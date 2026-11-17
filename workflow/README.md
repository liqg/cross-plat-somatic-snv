# Workflow

The stages are separate Snakemake workflows.  `workflow/run.py` generates the
per-stage configs, runs the workflows and provides a dry-run mode.

```
align/              fastp + BWA-MEM
posmix/             PosMix + Mutect2 / Strelka2 / VarScan2 + force-calling
somaticxgb/         feature matrix, training, prediction (R)
twist/              physical Twist calling notes and commands
drivers/            config generation
pos_files/          position-frequency tables
```

## Order

```
configs --kind align
    -> align
configs --kind simulation / benchmark / consistency / twist_hybrid
    -> simulate --role train
    -> simulate --role test
    -> simulate --role benchmark
    -> train_SomaticXGB.R
    -> predict_SomaticXGB.R
    -> analysis/*
```

## Config generation

`drivers/make_configs.py` reads `config/config.yaml`, `config/samples.tsv`,
`config/simulations.tsv`, `config/benchmark.tsv` and
`config/twist_hybrid.tsv`.  It writes JSON configs under `$CPSNV_WORK_ROOT`:

| Kind | Output |
|------|--------|
| `align` | `$CPSNV_WORK_ROOT/0.align/config.json` |
| `simulation` | `4.train/{pair}/config.json`, `4.test/{pair}/config.json` |
| `benchmark` | `3.performance/{vaf}/config.json` |
| `consistency` | `2.consistency/config.json` |
| `twist_hybrid` | `8.twist_HG001_mix/config.json` |

Each simulated pair gets six `data[]` entries (one per platform).  The tumour
and normal BAM paths are resolved from `config/samples.tsv`, so the same
physical library is used across the five TruSeq platforms and the converted
DNBSEQ library on DNBSEQ-T7.

## Per-pair execution

Run PosMix in the pair's own directory:

```bash
snakemake -s workflow/posmix/somatic_snv_posmix.smk \
  --configfile $CPSNV_WORK_ROOT/4.test/45/config.json \
  --directory $CPSNV_WORK_ROOT/4.test/45 --cores 100
```

The Snakefile resolves `reshapeRC` and `somaticfilter.pl` relative to its own
directory, so it can be invoked from any working directory.  `VarScan.v2.4.3.jar`
and `bam-readcount` are read from the config keys `varscan_jar` and
`bam_readcount` (defaults work inside the Docker image).

## errorRates

`errorRates-v1.3` is vendored at `software/errorRates/`.  Run it per BAM with
`software/errorRates/run_errorRates.sh`; the resulting `.er.txt.gz` files are
consumed by `analysis/01_error_rates/`.  No reporting pipeline is included.
