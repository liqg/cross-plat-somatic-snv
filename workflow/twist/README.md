# Twist cfDNA physical standards

The physical Twist data are not simulated.  Three libraries of the 1 % standard
and three of the 0 % wild-type standard were sequenced on five platforms.  The
analysis scores the 1 % tumour against the 0 % normal with the same callers and
SomaticXGB model used for the simulations.

## Inputs

- `config/twist_physical.tsv` - platform, replicate and library IDs
- aligned BAMs from `workflow/align` (`0.align/bam/...`)
- GRCh37 FASTA and `reference/panel/T364V1.hg19.bed`

The 1 % and 0 % libraries are separate materials, so the matched normal for a
1 % tumour is the corresponding 0 % library from the same platform; the
manuscript uses the 0 % wild-type standard as the matched normal.

## Calling

The caller settings are those in `workflow/posmix` and the manuscript Methods.
For each `(platform, replicate)` pair, with `T` the 1 % tumour BAM and `N` the
0 % normal BAM:

```bash
# Mutect2
gatk Mutect2 -R $GENOME -I $T -I $N -normal normal -O ${id}.raw.vcf.gz -L $PANEL
gatk FilterMutectCalls -V ${id}.raw.vcf.gz -R $GENOME -O mutect/${id}.filter.vcf.gz \
  --min-reads-per-strand 1 --min-median-base-quality 20 --normal-p-value-threshold 0.01

# Manta + Strelka2
python2.7 "$(command -v configManta.py)" --exome --normalBam $N --tumorBam $T \
  --referenceFasta $GENOME --runDir manta/${id}
python2.7 manta/${id}/runWorkflow.py -m local -j $THREADS
python2.7 "$(command -v configureStrelkaSomaticWorkflow.py)" --indelCandidates \
  manta/${id}/results/variants/candidateSmallIndels.vcf.gz \
  --exome --normalBam $N --tumorBam $T --referenceFasta $GENOME --runDir strelka/${id}
python2.7 strelka/${id}/runWorkflow.py -m local -j $THREADS

# VarScan2 + fpfilter
samtools mpileup -d 1000000 -q 30 -Q 30 -A -B -x --excl-flags UNMAP,SECONDARY,QCFAIL,DUP \
  -f $GENOME -l $PANEL $T | bgzip -c > varscan/${id}.T.pileup.gz
samtools mpileup -d 1000000 -q 30 -Q 30 -A -B -x --excl-flags UNMAP,SECONDARY,QCFAIL,DUP \
  -f $GENOME -l $PANEL $N | bgzip -c > varscan/${id}.N.pileup.gz
java -jar software/varscan/VarScan.v2.4.3.jar somatic \
  <(bgzip -dc varscan/${id}.N.pileup.gz) <(bgzip -dc varscan/${id}.T.pileup.gz) \
  varscan/${id} --min-var-freq 0.0025 --min-coverage-normal 10 --min-coverage-tumor 100 \
  --min-freq-for-hom 0.75 --normal-purity 1.00 --tumor-purity 1.00 --p-value 0.99 \
  --somatic-p-value 0.01 --strand-filter 0 --validation 1 --output-vcf 1
perl software/somaticfilter/somaticfilter.pl - < varscan/${id}.all.snp.vcf \
  | bgzip -c > varscan/${id}.raw.snp.vcf.gz
```

The exact commands used for the released `6.twist_pair` directory are the same
as the `varscan`, `varscan_novo`, `manta`, `strelka` and `mutect` rules in
`workflow/posmix/somatic_snv_posmix.smk`, with the PosMix tumour BAM replaced
by the physical 1 % BAM.

## SomaticXGB scoring

The force-calling and bam-readcount steps are the `features` rules in
`workflow/posmix`.  Run them on the same pair, then:

```bash
Rscript workflow/somaticxgb/predict_SomaticXGB.R \
  --pairs $CPSNV_WORK_ROOT/6.twist_pair \
  --outdir $CPSNV_WORK_ROOT/6.twist_pair/pred \
  --model software/somaticxgb/models/SomaticXGB
```

The released `6.twist_pair/pred/*_pred.txt.gz` files are the inputs to
`analysis/06_twist/derive_twist_tables.py`.

## 0 % error profile

The substitution-specific 0 % error rates (Fig. 4c) come from errorRates-v1.3 on
the 0 % wild-type libraries (`9.errorRates_bq30_twist_wt_PE15M`).  See
`software/errorRates/README.md`.

## Twist/HG001 simulation

The Twist/HG001 simulation (`8.twist_HG001_mix`) is a normal PosMix run:
generate its config with

```bash
python workflow/drivers/make_configs.py --kind twist_hybrid
```

and run it like any other pair with `workflow/posmix`.
