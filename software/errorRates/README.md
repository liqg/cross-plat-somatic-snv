# errorRates v1.3

`errorRates-v1.3` counts reference/read base mismatches by quality, cycle and
strand, and writes the `.er.txt.gz` table consumed by the analysis scripts.
It does not call variants and is independent of Mutect2, Strelka2, VarScan2 and
SomaticXGB.

## Inputs

- an aligned, duplicate-marked BAM
- the GRCh37 FASTA used for alignment
- a BED of panel regions excluding germline-variant sites.  The manuscript used
  `reference/panel/T364V1.hg19.homoref.bed` (panel intersect high-confidence
  regions, homozygous-reference positions).

## Run

```bash
bash software/errorRates/run_errorRates.sh \
  -b $CPSNV_WORK_ROOT/0.align/pipeline/bwamem/SURFSeq_5000_HG001_GM231129011/SURFSeq_5000_HG001_GM231129011.bam \
  -f $CPSNV_DATA_ROOT/reference/GRCh37/genome.fasta \
  -l reference/panel/T364V1.hg19.homoref.bed \
  -o $CPSNV_WORK_ROOT/1.errorRates/pipeline/bwamem/SURFSeq_5000_HG001_GM231129011/SURFSeq_5000_HG001_GM231129011.er.txt.gz \
  -t 16
```

Output columns:

```
strand  b5  ref  b3  base  read12  cycle  qual  count
```

`analysis/01_error_rates/` converts it to Q-threshold and per-quality mismatch
rates.

## Provenance

The binary is the `errorRates-v1.3` build used for the manuscript (x86-64,
dynamically linked against the HTSlib dependencies provided by the Docker
image).  No reporting pipeline or variant-calling component is included.
