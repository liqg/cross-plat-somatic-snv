library(data.table)
library(magrittr)
library(stringr)
library(jsonlite)
library(ggplot2)
#library(data.vcf)

source("R/readSamstats.R")
set_fig_size = function(w=4, h=3){
     options(repr.plot.width = w, repr.plot.height = h)
}
source("R/install.pkgs.R")
source("R/fsave.R")

library(cowplot)

config0_gmerror_panel = list(
genome=unbox("GRCh37"),
bed_file="/biobase/apps/GMError/PanelIGT/data/T364V1.hg19.homoref.bed",
fastp=unbox("--umi --umi_loc per_read --umi_len 6 --umi_skip 1 --umi_prefix UMI --detect_adapter_for_pe")
)

config0_gmerror_v6 = list(
genome=unbox("GRCh37"),
bed_file="/biobase/apps/GMError/WESV6/data/Exome-Agilent_V6.homoref.bed",
fastp=unbox("")
)

config0_gmerror_p3 = list(
genome=unbox("P3"),
bed_file="/biobase/apps/GMError/P3/data/P3.homoref.bed",
fastp=unbox("")
)

gmerror_config_insert = function(config, d, r1, id, fqget, head, r1_suffix="1.fq.gz", r2_suffix="2.fq.gz") {
    d = sub("\\/$", "", d)
    l = list(sampleid = unbox(id))
    if(!missingArg(head)) {l$head = head}
    if(!missingArg(fqget)) {l$fqget = fqget}

    fq1 = dir(d, r1, full.names = T, recursive=T)
    stopifnot(length(fq1)>0)
    stopifnot(all(grepl(r1_suffix, fq1)))
    l$read1 = fq1
    l$read2 = sub(r1_suffix, r2_suffix,fq1)
    config$data[[length(config$data)+1]] = l
    config
}
