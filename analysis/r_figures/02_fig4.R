# Generate Fig. 4 data panels from analysis/r_figures/plot_data.
# Plotting code follows the original notebook; only the input data source is repo-local.
suppressPackageStartupMessages({
  library(data.table)
  library(ggplot2)
  library(cowplot)
  library(scales)
})

PLOT_DATA <- "plot_data"
OUT <- "fig_261028"
dir.create(OUT, showWarnings = FALSE)

platcols <- data.table(
  abbr = c("FA", "GL", "SF", "NV", "NX", "T7"),
  name = c("FASTASeq 300", "GenoLab M", "SURFSeq 5000", "NovaSeq 6000", "NovaSeq X", "DNBSEQ-T7"),
  col = c("#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00", "#a65628")
)[, fac := factor(name, levels = name)]

mytheme <- theme_light() + theme(
  title = element_text(size = 6),
  plot.title = element_text(size = 6),
  legend.text = element_text(size = 5),
  axis.title = element_text(size = 6),
  strip.text = element_text(size = 6, color = "black"),
  axis.text = element_text(size = 5),
  panel.grid.major = element_blank(),
  panel.grid.minor = element_blank(),
  panel.background = element_blank(),
  strip.background = element_blank(),
  panel.spacing = unit(1, "lines"),
  legend.key.size = unit(6, "pt"),
  legend.spacing.y = unit(1, "mm"),
  legend.title = element_text(size = 6)
)
set_fig_size <- function(w = 4, h = 3) options(repr.plot.width = w, repr.plot.height = h)

method_levels <- c("Strelka2_Mutect2", "VarScan2_fpfilter", "Threshold_0.1")

# ---------------------------------------------------------------- Fig. 4a
d4a <- fread(file.path(PLOT_DATA, "fig4a_original_twist_metrics.csv"))
d4a[, Method := factor(Method, levels = method_levels)]
d4a[, Platform := factor(Platform, levels = platcols$name[platcols$name %in% Platform])]
d4a[, Metric := factor(Metric, levels = c("Recall", "Precision", "F1"))]

fig4_eval <- ggplot(d4a, aes(x = Platform, y = Value, color = Platform, shape = Method)) +
  geom_jitter(size = 0.5) +
  facet_wrap(~Metric, nrow = 1) +
  mytheme +
  theme(panel.spacing = unit(0.2, "lines")) +
  scale_y_continuous(breaks = seq(0, 1, by = 0.1)) +
  scale_color_manual(values = platcols[name %in% d4a$Platform, col]) +
  scale_shape_manual(values = c(1, 2, 3)) +
  theme(axis.text.x = element_text(size = 5, angle = 65, hjust = 1, vjust = 1)) +
  xlab("") + ylab("")

# ---------------------------------------------------------------- Fig. 4b
fp <- fread(file.path(PLOT_DATA, "fig4b_fp_substitutions.csv"))
fp[Method == "SomaticXGB (0.1)", Method := "Threshold_0.1"]
fp[, Method := factor(Method, levels = c("Strelka2_Mutect2", "VarScan2_fpfilter", "Threshold_0.1"))]
fp[, Platform := factor(Platform, levels = platcols$name[platcols$name %in% Platform])]
fp_count <- fp[, .N, by = .(Method, Platform, Substitution)]
fig4_fp <- ggplot(fp_count, aes(x = Substitution, y = N, fill = Platform)) +
  geom_col(position = "dodge") +
  facet_wrap(~Method, scales = "free_y", ncol = 1) +
  mytheme +
  theme(axis.text.x = element_text(angle = 90, hjust = 1, vjust = 1)) +
  scale_fill_manual(values = platcols$col[platcols$name %in% fp_count$Platform]) +
  xlab("") + ylab("Number of false positives") +
  theme(panel.spacing = unit(0, "mm"))

# ---------------------------------------------------------------- Fig. 4c
d4c <- fread(file.path(PLOT_DATA, "fig4c_twist0_substitution_error.csv"))
d4c[, Platform := factor(Platform, levels = platcols$name[platcols$name %in% Platform])]
fig4_k1_twist <- ggplot(d4c, aes(x = Substitution, y = Error_rate, fill = Platform)) +
  geom_col(width = 0.7, position = position_dodge2(padding = 0, preserve = "single")) +
  scale_fill_manual(values = platcols[name %in% d4c$Platform, col]) +
  labs(x = "", y = "Error rate") +
  mytheme +
  theme(axis.text.x = element_text(angle = 90, hjust = 1, vjust = 0.5))

# ---------------------------------------------------------------- Fig. 4d
d4d <- fread(file.path(PLOT_DATA, "fig4d_hybrid_metrics.csv"))
d4d[, Method := factor(Method, levels = method_levels)]
d4d[, Platform := factor(Platform, levels = platcols$name[platcols$name %in% Platform])]
d4d[, Metric := factor(Metric, levels = c("Recall", "Precision", "F1"))]
fig4_eval2 <- ggplot(d4d, aes(x = Platform, y = Value, color = Platform, shape = Method)) +
  geom_jitter(size = 0.5) +
  facet_wrap(~Metric, nrow = 1) +
  mytheme +
  theme(panel.spacing = unit(0.2, "lines")) +
  scale_y_continuous(breaks = seq(0, 1, by = 0.1), limits = c(0, 1)) +
  scale_color_manual(values = platcols[name %in% d4d$Platform, col]) +
  scale_shape_manual(values = c(1, 2, 3)) +
  theme(axis.text.x = element_text(size = 5, angle = 65, hjust = 1, vjust = 1)) +
  xlab("") + ylab("")

fig4 <- plot_grid(fig4_eval, fig4_fp, fig4_k1_twist, fig4_eval2,
                  ncol = 2, labels = c("a", "b", "c", "d"), label_size = 7)
set_fig_size(10, 5)
ggsave(file.path(OUT, "fig4.png"), fig4, width = 183, height = 100, units = "mm", dpi = 300, bg = "white")
print("wrote fig4")
