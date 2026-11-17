# Outputs

All analysis outputs are written under `$CPSNV_RESULTS`.

## Tables (`results/tables/`)

| File | Produced by | Content |
|------|-------------|---------|
| `error_rates_by_q_clean.csv` | `01_error_rates/build_error_rates_by_q.py` | per-library mismatch rate at Q0/Q10/Q20/Q30/Q40 |
| `A3_multithreshold_by_platform.csv` | `01_error_rates/A3_error_rates.py` | Table S23 |
| `A3_per_quality_by_platform.csv` | same | Table S24 |
| `A2_model_metrics_*.csv` | `04_model/A2_two_feature_baseline.py` | two-feature baseline, Tables S21/S22 |
| `A2_predictions_test.csv.gz` | same | per-candidate scores for every model |
| `C3_vif.csv`, `C3_correlation_*.csv` | `04_model/C3_collinearity.py` | Table S34 |
| `B1_*`, `B1b_*`, `B1c_*` | `03_benchmark/B1*.py` | Table S25 |
| `C2_fp_distance_*` | `05_test_set/C2_fp_distance.py` | Table S26 |
| `A4_representative_*` | `06_twist/A4_representative_calls_0pct.py` | Table S27 |
| `G1_*` | `05_test_set/G1_donor_fragment_audit.py` | Table S28 |
| `R24_*` | `05_test_set/R24_sharing_and_direction.py` | Table S29 |
| `R27_*` | `05_test_set/R27_mixed_models.py` | Table S30 |
| `D1_*`, `D1b_*` | `05_test_set/D1*.py` | Table S31 |
| `A1_*` | `05_test_set/A1_metrics.py` | Table S32 |
| `C1_*` | `05_test_set/C1_empty_target_stats.py` | Table S33 |
| `fig2d_performance_metrics.csv` | `03_benchmark/make_fig2d_data.py` | source data for Fig. 2d |
| `fig3a_test_100M_methods.csv`, `fig3_roc_curves.csv.gz`, `fig3_test_predictions.csv.gz` | `04_model/make_fig3_data.py`, `make_fig3_roc.py` | source data for Fig. 3 |
| `supp_table_S9a_original_twist.csv`, `supp_table_S9b_hybrid_twist.csv`, `fig4b_fp_substitutions.csv`, `fig4c_twist0_substitution_error.csv` | `06_twist/derive_twist_tables.py` | source data for Fig. 4 |
| `fig1b_consistency.tsv` | `02_consistency/consistency_from_posfeature.py` | source data for Fig. 1b |

The final workbook is
`$CPSNV_RESULTS/Additional_file_1_Supplementary_tables.xlsx`.

## Figures (`results/figures/`)

| File | Content |
|------|---------|
| `Fig1b_consistency.pdf/.png` | Fig. 1b |
| `fig2d_revised.png` | Fig. 2d |
| `fig3_revised.png` | Fig. 3a-e |
| `fig4_revised.png` | Fig. 4a-d |
| `supplementary/FigS1.pdf` ... `FigS12.pdf` | Additional file 2 pages |
| `Additional_file_2_Supplementary_figures.pdf` | combined A4 PDF (after `pdfunite`) |

## Intermediates (`results/intermediate/`)

| File | Content |
|------|---------|
| `twist_truth_221.tsv` | 221 panel-covered Twist SNVs |
| `giab_panel_vars.pkl` | panel-restricted GIAB variants for HG001-HG005 |

## Logs (`results/logs/`)

Each revision script writes a timestamped log; `make_all.py` prints the
commands it runs.  Keep these logs with the results when archiving.
