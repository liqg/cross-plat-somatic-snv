# R figure scripts

These are the R plotting scripts for the submitted data panels.  The plotting
code follows the original analysis notebooks in `250120-paper-repeat`; the
input tables have been extracted into `plot_data/`, so no original absolute
data paths are needed.

## Scripts

| Script | Outputs |
|---|---|
| `01_fig1b_fig2d_fig3.R` | `fig_260610/fig1b.png`, `fig_260610/fig2.png`, `fig_260610/fig3.png` |
| `02_fig4.R` | `fig_261028/fig4.png` |
| `assemble_figures.py` | `complete_figures/Fig1-4.{png,tif,pdf}` |
| `export_fig1b_data.R` | provenance tool: rebuilds `plot_data/fig1b_df_sim.csv.gz` from the original PosMix/posfeature files |

## Inputs

All plotting inputs are under `plot_data/`; see `plot_data/README.md` for the
file-to-panel map.  The feature-importance input is
`plot_data/fig3c_feature_importance.csv`.

## Running

From the repository root, inside an environment with R, ggplot2, cowplot,
pROC, data.table and scales:

```bash
cd analysis/r_figures
Rscript 01_fig1b_fig2d_fig3.R
Rscript 02_fig4.R
```

Or run both inside the full Docker image:

```bash
docker run --rm \
  -v "$PWD/analysis/r_figures:/work" -w /work \
  --entrypoint Rscript cross-plat-somatic-snv:fig 01_fig1b_fig2d_fig3.R
docker run --rm \
  -v "$PWD/analysis/r_figures:/work" -w /work \
  --entrypoint Rscript cross-plat-somatic-snv:fig 02_fig4.R
```

## Complete figures

The schematic panels are in `schematics/`.  Assemble complete Fig. 1-4
automatically:

```bash
python analysis/r_figures/assemble_figures.py \
  --r-figures analysis/r_figures \
  --schematics analysis/r_figures/schematics \
  --out analysis/r_figures/complete_figures
```

Fig. 1 and Fig. 2 are vertical schematic + data-panel composites; Fig. 3 and
Fig. 4 are complete R-rendered data figures.
