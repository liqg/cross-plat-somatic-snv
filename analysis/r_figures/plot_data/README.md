# Derived data used by the R figure scripts

These files are derived figure inputs copied from the original analysis tree.
They are not raw sequencing data; they are the tables that the original R
notebooks plot or use to construct plot data.

| File | Used by |
|---|---|
| `fig1b_df_sim.csv.gz` | Fig. 1b (derived with `export_fig1b_data.R`) |
| `fig3a_all_depths_metrics.csv` | Fig. 3a |
| `fig3c_feature_importance.csv` | Fig. 3c |
| `fig4a_original_twist_metrics.csv` | Fig. 4a |
| `fig4d_hybrid_metrics.csv` | Fig. 4d |
| `fig2d_performance_metrics.csv` | Fig. 2d |
| `fig3a_test_100M_methods.csv` | Fig. 3a |
| `fig3_roc_curves.csv.gz` | Fig. 3b |
| `fig3_test_predictions.csv.gz` | Fig. 3b (ROC and prediction summaries) |
| `error_rates_by_q_clean.csv` | Fig. 3d-e |
| `twist_original_metrics.csv` | Fig. 4a |
| `mixed_original_twist.csv` | Fig. 4a/4d summaries |
| `mixed_hybrid_twist.csv` | Fig. 4d |
| `twist_truth_221.tsv` | Fig. 4 truth set |
| `twist_fp_normal_af_FA1.csv` | Fig. 4 representative/control summaries |
| `fig4b_fp_substitutions.csv` | Fig. 4b |
| `fig4c_twist0_substitution_error.csv` | Fig. 4c (per-reference normalized error rate) |

Fig. 3c uses the released feature-importance tables in
`software/somaticxgb/importance_*.txt`.

Fig. 1b uses `fig1b_df_sim.csv.gz`, exported by
`analysis/r_figures/export_fig1b_data.R` from the original PosMix/posfeature
inputs.  The original R script can also recompute `df_sim` from the original
data tree.
