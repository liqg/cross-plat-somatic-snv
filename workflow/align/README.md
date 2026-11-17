# Alignment

`align.smk` performs:

1. merge multiple FASTQ lanes (or symlink a single pair)
2. `fastp` adapter trimming (`--detect_adapter_for_pe` for paired-end)
3. `bwa mem -K 100000000 -k 32 -M`
4. `samtools sort` + index

## Config

Generate it from `config/samples.tsv`:

```bash
python workflow/drivers/make_configs.py --config config/config.yaml --kind align
```

Then:

```bash
snakemake -s workflow/align/align.smk \
  --configfile $CPSNV_WORK_ROOT/0.align/config.json \
  --directory $CPSNV_WORK_ROOT/0.align --cores 64
```

Output: `pipeline/bwamem/{sample}/{sample}.bam` under the `--directory` root.
The manuscript reports raw (pre-deduplication) depth; duplicate marking is done
later by the PosMix pipeline (`MarkDuplicates` on the simulated BAM).

## Reference

`genome_fasta` must point to a GRCh37 FASTA with a BWA index, `.fai` and
`.dict`.  The `bed_file` is used for downstream target QC; the capture-panel
BED is in `reference/panel/`.
