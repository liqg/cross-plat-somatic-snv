# Software

| Directory | Contents | License / source |
|-----------|----------|------------------|
| `bin/` | `posmix`, `posfeature`, `bam-readcount`, `libseqrt.so`, `libomp.so` | GeneMind / upstream |
| `posmix/` | copies of the PosMix and posfeature binaries | source: https://github.com/liqg/posmix |
| `somaticxgb/` | feature R code, released models, importance tables | manuscript |
| `R/` | `data.vcf_0.2.0.tar.gz` | GPL (>= 2) |
| `reshapeRC/` | Python replacement + tests + validation report | manuscript |
| `somaticfilter/` | VarScan2 post-filter Perl script | manuscript |
| `varscan/` | `VarScan.v2.4.3.jar` | LGPL |
| `errorRates/` | `errorRates-v1.3` binary + wrapper | in-house |

## Binaries

The PosMix binary is dynamically linked against HTSlib (`libz`, `libbz2`,
`liblzma`, `libcurl`).  `posfeature` additionally needs `libseqrt.so` and
`libomp.so`, which are shipped in `software/bin/`.  These libraries are
provided by the Docker image.  Set:

```bash
export PATH=/opt/cpsnv/software/bin:$PATH
export LD_LIBRARY_PATH=/opt/cpsnv/software/bin:$LD_LIBRARY_PATH
```

## `reshapeRC`

The Python implementation reproduces the 30 published `.rc.tsv.gz` files
byte-for-byte (see `reshapeRC/reshapeRC_validation.md`).  It is the only
implementation shipped here; no compiled version is needed.  Run the tests
with:

```bash
python -m unittest discover -s software/reshapeRC/tests -v
```

## `data.vcf`

Install the bundled package before running the R feature extractor:

```bash
R CMD INSTALL software/R/data.vcf_0.2.0.tar.gz
```

The Dockerfile does this automatically.
