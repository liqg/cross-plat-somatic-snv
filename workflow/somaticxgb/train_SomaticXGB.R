#!/usr/bin/env Rscript
# Train SomaticXGB from per-pair PosMix force-calling outputs.
#
# Usage:
#   Rscript train_SomaticXGB.R --pairs DIR [DIR ...] --outdir DIR \
#       [--feature-matrix FILE] [--skip-cv] [--use-cv-best] [--nthread 8]
#
# Each pair directory must contain config.json and the `features` outputs of
# workflow/posmix (mutect_a_vraw, varscan, varscan_novo).  The script builds the
# feature matrix with software/somaticxgb/R/build_feature_matrix.R, runs the
# 5-fold grid search reported in Additional file 1: Table S19/S20, and fits the
# released model (max_depth 7, eta 0.1, nrounds 400).

suppressPackageStartupMessages({
  library(data.table)
  library(xgboost)
})

script_dir <- local({
  a <- commandArgs(trailingOnly = FALSE)
  f <- grep("^--file=", a, value = TRUE)
  if (length(f) == 1) dirname(normalizePath(sub("^--file=", "", f))) else getwd()
})
repo_root <- normalizePath(file.path(script_dir, "..", ".."))
build_script <- file.path(repo_root, "software", "somaticxgb", "R", "build_feature_matrix.R")
stopifnot(file.exists(build_script))

parse_args <- function(x) {
  out <- list(pairs = character(), outdir = "somaticxgb_run",
              feature_matrix = NULL, skip_cv = FALSE,
              use_cv_best = FALSE, nthread = 8L, folds = 5L)
  i <- 1
  while (i <= length(x)) {
    k <- x[i]
    if (k == "--pairs") {
      i <- i + 1
      while (i <= length(x) && !startsWith(x[i], "--")) {
        out$pairs <- c(out$pairs, x[i]); i <- i + 1
      }
    } else if (k == "--outdir") { i <- i + 1; out$outdir <- x[i]; i <- i + 1
    } else if (k == "--feature-matrix") { i <- i + 1; out$feature_matrix <- x[i]; i <- i + 1
    } else if (k == "--skip-cv") { out$skip_cv <- TRUE; i <- i + 1
    } else if (k == "--use-cv-best") { out$use_cv_best <- TRUE; i <- i + 1
    } else if (k == "--nthread") { i <- i + 1; out$nthread <- as.integer(x[i]); i <- i + 1
    } else if (k == "--folds") { i <- i + 1; out$folds <- as.integer(x[i]); i <- i + 1
    } else { stop("Unknown argument: ", k) }
  }
  out
}
a <- parse_args(commandArgs(trailingOnly = TRUE))
if (!length(a$pairs)) stop("Provide at least one --pairs directory")
dir.create(a$outdir, showWarnings = FALSE, recursive = TRUE)

features <- c(
  "DP", "ECNT", "MBQ_REF", "MBQ_ALT", "MFRL_REF", "MFRL_ALT", "MMQ_REF", "MMQ_ALT",
  "MPOS", "NALOD", "NLOD", "TLOD",
  "avg_mapping_quality", "avg_basequality", "avg_pos_as_fraction",
  "avg_num_mismatches_as_fraction", "avg_sum_mismatch_qualities"
)

feature_matrix <- if (is.null(a$feature_matrix)) file.path(a$outdir, "feature_matrix.tsv.gz") else a$feature_matrix

if (!file.exists(feature_matrix)) {
  cmd <- paste("Rscript", shQuote(build_script),
               paste(shQuote(a$pairs), collapse = " "), shQuote(feature_matrix))
  message("Running: ", cmd)
  status <- system(cmd)
  if (status != 0) stop("build_feature_matrix.R failed")
}
dtm <- fread(feature_matrix)
message("Feature matrix: ", nrow(dtm), " rows, ", sum(dtm$label == 1), " positives")

manuscript_params <- list(max_depth = 7L, eta = 0.1, nrounds = 400L)
if (!a$skip_cv) {
  suppressPackageStartupMessages(library(caret))
  set.seed(820)
  folds <- createFolds(dtm$label, k = a$folds, list = TRUE, returnTrain = FALSE)
  grid <- CJ(max_depth = c(3L, 5L, 7L, 9L),
             eta = c(0.01, 0.05, 0.1, 0.2),
             nrounds = c(100L, 200L, 400L))
  results <- data.table()
  for (g in seq_len(nrow(grid))) {
    p <- grid[g]
    for (k in seq_along(folds)) {
      te <- folds[[k]]
      dtrain <- xgb.DMatrix(data = data.matrix(dtm[-te, features, with = FALSE]),
                            label = dtm$label[-te])
      model <- xgboost(data = dtrain, max.depth = p$max_depth, eta = p$eta,
                       nthread = a$nthread, nrounds = p$nrounds,
                       objective = "binary:logistic", verbose = 0)
      pred <- predict(model, data.matrix(dtm[te, features, with = FALSE]))
      if (requireNamespace("PRROC", quietly = TRUE)) {
        pr <- PRROC::pr.curve(scores.class0 = pred, weights.class0 = dtm$label[te],
                              curve = FALSE)
        prauc <- pr$auc.integral
      } else {
        prauc <- NA_real_
      }
      results <- rbind(results, data.table(max_depth = p$max_depth, eta = p$eta,
                                           nrounds = p$nrounds, Fold = k,
                                           PR_AUC = prauc))
    }
  }
  fwrite(results, file.path(a$outdir, "cv_results.tsv"), sep = "\t")
  med <- results[, .(Median_PR_AUC = median(PR_AUC, na.rm = TRUE)),
                 by = .(max_depth, eta, nrounds)]
  setorder(med, -Median_PR_AUC)
  fwrite(med, file.path(a$outdir, "med_auc.tsv"), sep = "\t")
  if (a$use_cv_best && nrow(med)) {
    manuscript_params <- list(max_depth = med$max_depth[1], eta = med$eta[1],
                              nrounds = med$nrounds[1])
  }
}

dtrain <- xgb.DMatrix(data = data.matrix(dtm[, features, with = FALSE]),
                      label = dtm$label)
model <- xgboost(data = dtrain, max.depth = manuscript_params$max_depth,
                 eta = manuscript_params$eta, nthread = a$nthread,
                 nrounds = manuscript_params$nrounds,
                 objective = "binary:logistic", verbose = 0)
model_path <- file.path(a$outdir, "SomaticXGB_retrained")
xgb.save(model, model_path)
fwrite(xgb.importance(feature_names = features, model = model),
       file.path(a$outdir, "importance.tsv"), sep = "\t")
message("Saved model: ", model_path)
