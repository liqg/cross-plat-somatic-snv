# Generate Fig. 1b, Fig. 2d and Fig. 3 data panels from analysis/r_figures/plot_data.
# Plotting code follows the original notebook; only the input data source is repo-local.
suppressPackageStartupMessages({
  library(data.table)
  library(ggplot2)
  library(cowplot)
  library(pROC)
  library(magrittr)
  library(scales)
})

PLOT_DATA <- "plot_data"
OUT <- "fig_260610"
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

# ---------------------------------------------------------------- Fig. 1b
df_sim <- fread(file.path(PLOT_DATA, "fig1b_df_sim.csv.gz"))
df_sim[, Platform := factor(Platform, levels = platcols$name)]

fig1b <- ggplot(df_sim) +
  geom_point(aes(x = freq_exp3, y = freq_sim, col = Platform), size = 0.1) +
  facet_grid(~Platform) +
  labs(x = "Specified VAFs (%)", y = "Calculated VAFs (%)") +
  mytheme +
  theme(legend.position = "none") +
  theme(panel.spacing = unit(0.2, "lines")) +
  scale_y_continuous(labels = scales::percent_format(scale = 100, accuracy = 0.1, suffix = "")) +
  scale_x_continuous(labels = scales::percent_format(scale = 100, accuracy = 0.1, suffix = "")) +
  scale_color_manual(values = platcols$col)

fig1 <- plot_grid(fig1b, labels = "b", label_size = 7)
set_fig_size(10, 2)
ggsave(file.path(OUT, "fig1b.png"), fig1, width = 180, height = 35, units = "mm", dpi = 300, bg = "white")
print("wrote fig1b")

# ---------------------------------------------------------------- Fig. 2d
d2 <- fread(file.path(PLOT_DATA, "fig2d_performance_metrics.csv"))
d2[, VAF := factor(VAF, levels = c("0.5%", "1%", "1.5%", "2%", "3%", "5%"))]
d2[, Reads := factor(Reads, levels = c(25, 50, 75, 100, 125),
                     labels = c("900X", "1800X", "2700X", "3600X", "4500X"))]
d2[, Method := factor(Method, levels = c("Strelka2_Mutect2", "VarScan2_fpfilter"))]
d2[, Platform := factor(Platform, levels = platcols$name)]
dd2 <- melt(d2, measure.vars = c("Recall", "Precision", "F1"),
            variable.name = "Metric", value.name = "value")
dd2[, Metric := factor(Metric, levels = c("Recall", "Precision", "F1"))]

f2_eval <- ggplot(dd2, aes(x = Reads, y = value, color = Platform, linetype = Method)) +
  theme_light() +
  geom_point(size = 0.3) +
  geom_line(aes(group = interaction(Platform, Method)), linewidth = 0.2) +
  facet_grid(Metric ~ VAF, switch = "y") +
  scale_linetype_manual(values = c("solid", "dashed")) +
  scale_y_continuous(breaks = seq(0, 1, by = 0.1)) +
  labs(x = "Mean depth") +
  mytheme +
  theme(axis.text.x = element_text(angle = 65, size = 5, hjust = 1, vjust = 1)) +
  ylab("") +
  scale_color_manual(values = platcols$col) +
  theme(strip.placement = "outside") +
  theme(legend.position = "bottom", legend.box.margin = margin(-10, 0, 0, 0))

fig2 <- plot_grid(f2_eval + theme(panel.spacing = unit(0.2, "lines")),
                  labels = c("d"), label_size = 7)
set_fig_size(15, 7)
ggsave(file.path(OUT, "fig2.png"), fig2, width = 183, height = 120, units = "mm", dpi = 300, bg = "white")
print("wrote fig2")

# ---------------------------------------------------------------- Fig. 3
d3 <- fread(file.path(PLOT_DATA, "fig3a_all_depths_metrics.csv"))
d3[, Depth := factor(Depth, levels = c("2100X", "2800X", "3500X", "4200X", "4900X"))]
d3[, Method := factor(Method, levels = c("Strelka2_Mutect2", "VarScan2_fpfilter",
                                         "Threshold_0.05", "Threshold_0.1", "Threshold_0.2"))]
d3[, Platform := factor(Platform, levels = platcols$name)]
dd3 <- melt(d3, measure.vars = c("Recall", "Precision", "F1"),
            variable.name = "Metric", value.name = "value")
dd3[, Metric := factor(Metric, levels = c("Recall", "Precision", "F1"))]

f3_eval <- ggplot(dd3, aes(x = Depth, y = value, color = Platform)) +
  theme_light() +
  geom_point(size = 0.3) +
  geom_line(aes(group = interaction(Platform, Method)), linewidth = 0.2) +
  facet_grid(Metric ~ Method, switch = "y", scales = "free_y") +
  scale_y_continuous(breaks = seq(0, 1, by = 0.1)) +
  labs(x = "Mean depth") +
  mytheme +
  theme(axis.text.x = element_text(angle = 65, size = 5, hjust = 1, vjust = 1)) +
  ylab("") +
  scale_color_manual(values = platcols$col) +
  theme(strip.placement = "outside")

# ROC / precision-recall curves from the released per-candidate predictions.
pred <- fread(file.path(PLOT_DATA, "fig3_test_predictions.csv.gz"))
make_pr <- function(score, training) {
  rbindlist(lapply(unique(pred$plat), function(x) {
    a <- pred[plat == x]
    b <- suppressMessages(pROC::roc(a$label, a[[score]]))
    pROC::coords(b, ret = "all", transpose = FALSE) %>%
      as.data.table() %>%
      .[, .(precision, recall)] %>%
      .[, Platform := x] %>%
      .[, auc := as.numeric(b$auc)] %>%
      .[, Training := training]
  }))
}
pr <- rbind(make_pr("pred", "All"), make_pr("pred_nx", "NovaSeq X"))
setnames(pr, c("precision", "recall"), c("Precision", "Recall"))
pr[, Platform := factor(Platform, levels = platcols$name)]

f3_roc1 <- ggplot(pr[Precision > 0.8 & Recall > 0.8]) +
  geom_line(aes(x = Recall, y = Precision, group = Platform, col = Platform), linewidth = 0.3) +
  facet_wrap(~Training) +
  scale_color_manual(values = platcols$col) +
  mytheme

imp <- fread(file.path(PLOT_DATA, "fig3c_feature_importance.csv"))
imp <- rbind(imp[Training == "All"][order(-Gain)][1:7],
             imp[Training == "NovaSeq X"][order(-Gain)][1:7])
imp[, Rank := factor(Feature, levels = unique(Feature))]
f3_rank <- ggplot(imp) +
  geom_col(aes(x = Rank, y = Gain)) +
  facet_wrap(~Training, nrow = 1) +
  mytheme +
  theme(axis.text.x = element_text(angle = 65, hjust = 1, vjust = 1)) +
  xlab("") +
  theme(panel.spacing = unit(0.5, "lines"))

er <- fread(file.path(PLOT_DATA, "error_rates_by_q_clean.csv"))
er30 <- er[q == 30, .(Sample = hg, Platform = platform, error_bq30 = err)]
er30[, Platform := platcols$name[match(Platform, platcols$abbr)]]
er30[, Sample := factor(Sample, levels = paste0("HG00", 1:5))]
er30[, Platform := factor(Platform, levels = platcols$name)]
f3_er <- ggplot(er30, aes(x = Sample, y = error_bq30, fill = Platform)) +
  geom_col(position = "dodge") +
  mytheme +
  scale_fill_manual(values = platcols$col) +
  scale_y_continuous(labels = scales::percent_format(scale = 1000, accuracy = 0.01, suffix = "")) +
  theme(legend.box.margin = margin(-10, 0, 0, 0), legend.position = "bottom") +
  ylab("Error rate (‰)") +
  xlab("")

errmean <- er30[, .(error_bq30 = mean(error_bq30)), by = Platform]
m3 <- merge(dd3[Depth == "2100X" & Method %in% c("Strelka2_Mutect2", "Threshold_0.1", "VarScan2_fpfilter")],
            errmean, by = "Platform")
m3[, Platform := factor(Platform, levels = platcols$name)]
f3_er_metric <- ggplot(m3) +
  geom_point(aes(x = error_bq30, y = value, col = Platform), size = 0.5) +
  facet_grid(Metric ~ Method, scales = "free_y", switch = "y") +
  scale_color_manual(values = platcols$col) +
  scale_x_continuous(labels = scales::percent_format(scale = 1000, accuracy = 0.01, suffix = "")) +
  theme_light() +
  theme(
    title = element_text(size = 6), plot.title = element_text(size = 6),
    legend.text = element_text(size = 5), axis.title = element_text(size = 6),
    strip.text = element_text(size = 6, color = "black"), axis.text = element_text(size = 5),
    panel.grid.minor = element_blank(), panel.background = element_blank(),
    strip.background = element_blank(), panel.spacing = unit(1, "lines"),
    legend.key.size = unit(6, "pt"), legend.spacing.y = unit(1, "mm"),
    legend.title = element_text(size = 6), strip.placement = "outside",
    legend.box.margin = margin(-10, 0, 0, 0)
  ) +
  xlab("Error rate (‰)") + ylab("") + ylim(c(NA, 1))

fig3 <- plot_grid(
  f3_eval + theme(panel.spacing = unit(0.2, "lines"), legend.margin = margin(0, 0, 0, 0),
                  legend.box.margin = margin(-5, 0, -5, -5)),
  plot_grid(f3_roc1 + theme(legend.position = "bottom", legend.title = element_blank(),
                            legend.margin = margin(0, 0, 0, 0), legend.box.margin = margin(-5, 0, -5, -5),
                            legend.text = element_text(size = 5)),
            f3_rank, nrow = 1, labels = c("b", "c"), label_size = 7, rel_widths = c(1, 1)),
  plot_grid(f3_er + theme(legend.title = element_blank()),
            f3_er_metric + theme(panel.spacing = unit(0.5, "lines"),
                                 legend.box.margin = margin(-5, 0, -5, -5)),
            labels = c("d", "e"), label_size = 7, rel_widths = c(1, 1.5)),
  ncol = 1, rel_heights = c(4, 2, 2.5), labels = c("a", ""), label_size = 7
)
set_fig_size(10, 7)
ggsave(file.path(OUT, "fig3.png"), fig3, width = 183, height = 190, units = "mm", dpi = 300, bg = "white")
print("wrote fig3")
