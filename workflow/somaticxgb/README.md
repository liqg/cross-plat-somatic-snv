# SomaticXGB

Two command-line R scripts replace the original notebooks.

## Prerequisite

Run `workflow/posmix` with `"features"` in `callers` for every training / test
pair.  Each pair directory must contain:

```
config.json
mutect_a_vraw/{id}_reads{N}M.raw.vcf.gz
varscan/{id}_reads{N}M.raw.snp.sort.vcf.gz
varscan_novo/{id}_reads{N}M.raw.snp.vcf.rc.tsv.gz
```

The feature extractor is `software/somaticxgb/R/SomaticXGB_features.R`; the
matrix builder is `software/somaticxgb/R/build_feature_matrix.R`.

## Train

```bash
Rscript workflow/somaticxgb/train_SomaticXGB.R \
  --pairs $CPSNV_WORK_ROOT/4.train/{12,13,21,23,31,32} \
  --outdir $CPSNV_WORK_ROOT/5.somaticxgb/train \
  --nthread 16
```

Outputs: `feature_matrix.tsv.gz`, `cv_results.tsv`, `med_auc.tsv`,
`importance.tsv`, `SomaticXGB_retrained`.  The 5-fold grid is the 48-combination
search in Additional file 1: Tables S19-S20.  The final model uses
`max_depth=7`, `eta=0.1`, `nrounds=400` unless `--use-cv-best` is given.

## Predict

```bash
Rscript workflow/somaticxgb/predict_SomaticXGB.R \
  --pairs $CPSNV_WORK_ROOT/4.test/45 $CPSNV_WORK_ROOT/4.test/54 \
  --outdir $CPSNV_WORK_ROOT/5.somaticxgb/test \
  --model software/somaticxgb/models/SomaticXGB
```

Output: `predictions.tsv.gz` (the feature matrix plus a `pred` column).

## Released models

| Model | Path | Description |
|-------|------|-------------|
| SomaticXGB | `software/somaticxgb/models/SomaticXGB` | trained on all six platforms, 17 features |
| NovaSeq X-only | `software/somaticxgb/models/NovaSeqX_only` | trained on NovaSeq X rows only (Fig. 3b) |

`importance_SomaticXGB.txt` and `importance_NovaSeqX.txt` are the gain tables
plotted in Fig. 3d.

## Requirements

R 4.2.x with `data.table`, `xgboost` (1.7), `caret`, `PRROC`, `jsonlite`,
`magrittr`, `stringr`, `tidyr`, `tibble`, `Rcpp` and the bundled `data.vcf`
package (`R CMD INSTALL software/R/data.vcf_0.2.0.tar.gz`).
