# `reshapeRC.py` validation

`reshapeRC.py` is the only implementation shipped in this repository.  It was
validated against the published `.rc.tsv.gz` files used in the manuscript.

## Published-file comparison

30 `.rc.gz` inputs from `6.twist_pair` and `8.twist_HG001_mix` were converted
and compared with the published `.rc.tsv.gz` files.  All 30 decompressed TSVs
were byte-identical.

Two representative pairs:

| Pair | Input `.rc.gz` | Expected `.rc.tsv.gz` |
|---|---|---|
| 1 | `$CPSNV_WORK_ROOT/8.twist_HG001_mix/varscan_novo/FASTASeq_300-1_reads140M.raw.snp.vcf.rc.gz` | `$CPSNV_WORK_ROOT/8.twist_HG001_mix/varscan_novo/FASTASeq_300-1_reads140M.raw.snp.vcf.rc.tsv.gz` |
| 2 | `$CPSNV_WORK_ROOT/6.twist_pair/varscan_novo/FA_1_GM23063001_GM23062701.raw.snp.vcf.rc.gz` | `$CPSNV_WORK_ROOT/6.twist_pair/varscan_novo/FA_1_GM23063001_GM23062701.raw.snp.vcf.rc.tsv.gz` |

```bash
python3 software/reshapeRC/reshapeRC.py \
  -i $CPSNV_WORK_ROOT/8.twist_HG001_mix/varscan_novo/FASTASeq_300-1_reads140M.raw.snp.vcf.rc.gz \
  -g $CPSNV_GENOME \
  -o /tmp/reshapeRC_pair1.tsv.gz

gzip -dc /tmp/reshapeRC_pair1.tsv.gz > /tmp/reshapeRC_pair1.tsv
gzip -dc $CPSNV_WORK_ROOT/8.twist_HG001_mix/varscan_novo/FASTASeq_300-1_reads140M.raw.snp.vcf.rc.tsv.gz > /tmp/expected_pair1.tsv
cmp /tmp/reshapeRC_pair1.tsv /tmp/expected_pair1.tsv
```

The decompressed TSV content is byte-identical; gzip headers differ because of
timestamps and compression settings.

## Row counts

| Pair | Input lines | Positions | Output rows | A/C/G/T/N rows | Indel rows | Exact match |
|---|---:|---:|---:|---:|---:|---|
| 1 | 804 | 804 | 4123 | 4020 | 103 | yes |
| 2 | 8230 | 8230 | 41969 | 41150 | 819 | yes |

The extra rows are indel entries such as `-G`, `-AA` and `+A`.  Use
`--canonical-only` to emit only A/C/G/T/N.

## Context columns

Columns 5 and 6 are the plus-strand FASTA bases at POS-1 and POS+1.  They were
checked against the reference for all 804 + 8230 positions: 0 mismatches.

Test positions include `1:9778780` (`GGC`), `1:9783211` (`CGG`),
`17:41205987` (`TAA`) and `X:152864516` (`CTC`).  No test position lies at a
contig boundary; at boundaries the script emits `N` for the missing base.

## stdin, gzip and FASTA backends

```bash
gzip -dc $CPSNV_WORK_ROOT/8.twist_HG001_mix/varscan_novo/FASTASeq_300-1_reads140M.raw.snp.vcf.rc.gz \
  | python3 software/reshapeRC/reshapeRC.py -i /dev/stdin -g $CPSNV_GENOME \
  > /tmp/reshapeRC_stdin.tsv
cmp /tmp/reshapeRC_stdin.tsv /tmp/expected_pair1.tsv
```

Also verified: `-i -` with plain and gzip-compressed stdin, gzip output via
`-o`, both FASTA backends (pysam and `.fai`), and `--canonical-only`.

## Automated tests

```bash
python3 -m unittest discover -s software/reshapeRC/tests -v
```

16 tests pass, including both published pairs (pysam and `.fai` backends),
stdin, gzip input/output, zero-row formatting, indel ordering, comments,
boundaries and error handling.  Tests that need the GRCh37 FASTA are skipped
when `CPSNV_GENOME` is not set.
