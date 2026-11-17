# Released artifacts

These are the files submitted with the manuscript:

| File | Content |
|------|---------|
| `Additional_file_1_Supplementary_tables.xlsx` | Tables S1-S35 (36 sheets) |
| `Additional_file_2_Supplementary_figures.pdf` | Supplementary Figures S1-S12, 13 A4 pages (legend page plus 12 figures) |

They can be regenerated with:

```bash
python analysis/07_figures_tables/build_supplement_v4.py
python analysis/07_figures_tables/make_bmc_supp_figures.py
bash analysis/07_figures_tables/combine_supp_figures.sh \
  "$CPSNV_RESULTS/figures/supplementary" \
  "$CPSNV_RESULTS/Additional_file_2_Supplementary_figures.pdf"
```

The workbook is built from the curated S1-S20 base workbook in
`reference/supplementary/` plus the S21-S35 tables produced by the analysis
scripts.
