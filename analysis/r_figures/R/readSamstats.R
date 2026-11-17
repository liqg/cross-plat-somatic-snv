library(data.table)
library(magrittr)

readSamstats <- function (x) 
{
    a = readLines(x)
    k = a[!grepl("^#", a) & grepl("\\t", a)] %>% sub("\t.+", "", .) %>% unique
    r = lapply(k, function(x) {
        desc = a[grep(paste0("^", x), a)[1] - 1]
        data = grep(paste0("^", x), a, value = T) %>% tstrsplit(split = "\t") %>% setDT %>% .[, -1, with = F]
        list(desc=desc, data=data)
    })
    names(r) = k
    if("IS" %in% names(r)) {
        r$IS$data = r$IS$data[, lapply(.SD, as.numeric)] %>% setnames(c("size","total","inward","outward","other"))
    }
    r
}
