# Analysis scripts

All scripts import `analysis/common.py`, which resolves `CPSNV_DATA_ROOT`,
`CPSNV_WORK_ROOT` and `CPSNV_RESULTS`.  Run them from anywhere:

```bash
python analysis/<dir>/<script>.py --help
```

## 01_error_rates

| Script | Purpose |
|--------|---------|
| `error_rate_from_er.py` | Q-threshold and per-quality mismatch rate for one `.er.txt.gz` file |
| `build_error_rates_by_q.py` | collect `1.errorRates` files into `error_rates_by_q_clean.csv` (Fig. 3d/e) |
| `A3_error_rates.py` | multi-threshold + per-quality aggregation, Tables S23/S24 and Fig. S3 |

## 02_consistency

| Script | Purpose |
|--------|---------|
| `consistency_from_posfeature.py` | run posfeature on donor/background/tumour BAMs and draw Fig. 1b |

## 03_benchmark

| Script | Purpose |
|--------|---------|
| `make_fig2d_data.py` | metrics from the single-pair `3.performance` call sets (input for the R figure script) |
| `B1_baseline_threshold_sweep.py` | caller score sweeps, Table S25 |
| `B1b_fair_tuning.py` | fair-tuned cut points, Table S25 |
| `B1c_all_depths_fair.py` | fair tuning at every depth |

## 04_model

| Script | Purpose |
|--------|---------|
| `A2_two_feature_baseline.py` | TLOD+DP and logistic baselines, Tables S21/S22, Fig. S2 |
| `A2_summary.py` | metrics / bootstrap for the A2 predictions |
| `C3_collinearity.py` | 17-feature VIF and correlations, Table S34, Fig. S1 |
| `make_fig3_data.py` | test-set metrics (input for the R figure script) |
| `make_fig3_roc.py` | ROC tables for the all-platform and NovaSeq X-only models |
| `group_cv.py` | leave-one-group-out CV |
| `lopo_table.py` | leave-one-platform-out CV |
| `negative_audit_retrain.py` | ambiguous-negative audit (Additional file 1) |

## 05_test_set

| Script | Purpose |
|--------|---------|
| `prepare_giab_inputs.py` | panel-restricted GIAB VCFs and `giab_panel_vars.pkl` |
| `A1_metrics.py` | non-overlapping-fragment sensitivity, Table S32 |
| `C1_empty_target_stats.py` | empty-target negative control, Table S33 |
| `C2_fp_distance.py` | false-positive distance enrichment, Table S26, Fig. S5 |
| `D1_matched_dedup_depth.py` | deduplicated-depth matching, Table S31 |
| `D1b_bam_depth_matching.py` | BAM-depth sensitivity, Table S31, Fig. S7 |
| `G1_donor_fragment_audit.py` | donor-fragment audit, Table S28, Fig. S6 |
| `R24_sharing_and_direction.py` | train/test allele sharing, Table S29 |
| `R27_mixed_models.py` | mixed models / bootstrap CIs, Table S30, Fig. S11 |

## 06_twist

| Script | Purpose |
|--------|---------|
| `derive_twist_tables.py` | truth set, physical/hybrid metrics, FP spectra, 0% error (input for the R figure script) |
| `A4_representative_calls_0pct.py` | representative-call 0% control table, Table S27 |

## 07_figures_tables

| Script | Purpose |
|--------|---------|
| `build_supplement_v4.py` | append S21-S35 and write Additional file 1 |
| `make_bmc_supp_figures.py` | redraw Figs. S1-S12 to BMC artwork specs from `../supp_figure_data/` |
| `make_all.py` | run the figure/table steps in order (`--dry-run` supported) |

## Notes

- `A1_metrics.py`, `C1_empty_target_stats.py` and
  `A4_representative_calls_0pct.py` take explicit paths; their `--help` lists
  the required inputs.
- Several revision scripts were written for the response letter.  They are
  included because they generate Tables S21-S35 and Figs. S1-S12.
- The scripts write only under `$CPSNV_RESULTS`; they never modify raw data.
