# PosMix and posfeature

The Linux x86-64 binaries are in `software/bin/`:

- `posmix` - position-specific fragment replacement
- `posfeature` - per-read pileup used for the Fig. 1b consistency analysis

`posmix` is dynamically linked against HTSlib dependencies (`libz`, `libbz2`,
`liblzma`, `libcurl`); the Docker image provides them.  Source code:
https://github.com/liqg/posmix

`posfeature` also needs `libseqrt.so` and `libomp.so`, which are in the same
directory; set `LD_LIBRARY_PATH=software/bin`.
