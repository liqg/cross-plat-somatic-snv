#!/usr/bin/env Rscript
# Score candidate variants with a released or retrained SomaticXGB model.
#
# Usage:
#   Rscript predict_SomaticXGB.R --pairs DIR [DIR ...] --outdir DIR \
#       [--model PATH] [--feature-matrix FILE]
#
# The feature matrix is built with the same script as training
# (software/somaticxgb/R/build_feature_matrix.R).  Output:
#   {outdir}/predictions.tsv.gz   (feature matrix + `pred` column)

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
default_model <- file.path(repo_root, "software", "somaticxgb", "models", "SomaticXGB")
stopifnot(file.exists(build_script), file.exists(default_model))

parse_args <- function(x) {
  out <- list(pairs = character(), outdir = "somaticxgb_predict",
              model = default_model, feature_matrix = NULL)
  i <- 1
  while (i <= length(x)) {
    k <- x[i]
    if (k == "--pairs") {
      i <- i + 1
      while (i <= length(x) && !startsWith(x[i], "--")) {
        out$pairs <- c(out$pairs, x[i]); i <- i + 1
      }
    } else if (k == "--outdir") { i <- i + 1; out$outdir <- x[i]; i <- i + 1
    } else if (k == "--model") { i <- i + 1; out$model <- x[i]; i <- i + 1
    } else if (k == "--feature-matrix") { i <- i + 1; out$feature_matrix <- x[i]; i <- i + 1
    } else { stop("Unknown argument: ", k) }
  }
  out
}
a <- parse_args(commandArgs(trailingOnly = TRUE))
if (!length(a$pairs)) stop("Provide at least one --pairs directory")
dir.create(a$outdir, showWarnings = FALSE, recursive = TRUE)

feature_matrix <- if (is.null(a$feature_matrix)) file.path(a$outdir, "feature_matrix.tsv.gz") else a$feature_matrix
if (!file.exists(feature_matrix)) {
  cmd <- paste("Rscript", shQuote(build_script),
               paste(shQuote(a$pairs), collapse = " "), shQuote(feature_matrix))
  message("Running: ", cmd)
  status <- system(cmd)
  if (status != 0) stop("build_feature_matrix.R failed")
}
dtm <- fread(feature_matrix)

source(file.path(repo_root, "software", "somaticxgb", "R", "SomaticXGB_features.R"))
miss <- setdiff(SOMATICXGB_FEATURES, names(dtm))
if (length(miss)) stop("Feature matrix missing columns: ", paste(miss, collapse = ", "))
model <- xgb.load(a$model)
dtm$pred <- predict(model, data.matrix(dtm[, SOMATICXGB_FEATURES, with = FALSE]))
fwrite(dtm, file.path(a$outdir, "predictions.tsv.gz"), sep = "\t")
message("Wrote ", file.path(a$outdir, "predictions.tsv.gz"))
