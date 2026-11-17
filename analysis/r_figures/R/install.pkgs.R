options(Ncpus = 8)

setmirror = function(cran="https://mirrors.tuna.tsinghua.edu.cn/CRAN/", bioc="https://mirrors.tuna.tsinghua.edu.cn/bioconductor") {
    options(repos=cran)
    options(BioC_mirror=bioc)
}
setmirror()
install.pkgs <- function(x, lib=.libPaths()[1]) {
    method = "curl"
    extra = "--max-time 1111 --connect-timeout 10 --retry 5 --verbose --speed-time 10 --speed-limit 1 --no-keepalive  -H 'Connection: close'"

    y <- x[grepl(".tar.gz$", x, perl=T)]
    x <- setdiff(x, y)
    x <- setdiff(x, rownames(installed.packages()))
    
    if(length(x)) {
        if (!require(BiocManager)) install.packages("BiocManager", lib=lib, method=method, extra=extra)
        BiocManager::install(x, lib=lib, method=method, extra=extra)
    }

    # install from local
    a <- sapply(y, install.packages, repos=NULL)

    x <- c(x, sub("_.+.tar.gz$", "", basename(y), perl=T))
    x <- setdiff(x, rownames(installed.packages()))
    if(length(x)) {
        stop(paste0("Error: ", paste(x, collapse = ", "), " not be installed."))
    }
}

desc2pkgs <- function(desc_file, types=c("Depends","Imports","LinkingTo")) {
  library(data.table)
  a <- readLines(desc_file)
  a <- sub("^[\t ]+","", a)
  start <- grep("^[a-zA-Z]+:", a)
  end <- c(start[-1] - 1, length(a))
  a <- sapply(1:length(start), function(x) paste(a[start[x]:end[x]], collapse = ""))
  pkgs <- lapply(types, function(k){
    if(any(startsWith(a, k))){
      gsub(" ", "", strsplit(a[which(startsWith(a, k))], ":|,")[[1]][-1])
    }
  })
  pkgs <- data.table(type=rep(types, sapply(pkgs, length)), pkg0=unlist(pkgs))
  pkgs[, version:=sub(")$", "", sub(".+=", "", pkg0))]
  pkgs[!grepl("\\(",pkg0), version:=NA_character_]
  pkgs[, pkg:=sub("\\(.+", "", pkg0)]
  pkgs
}

test_web = function(file){
    r = TRUE
    a = try(suppressWarnings(readLines(file)))
    if(inherits(a, "try-error")) r =FALSE
    r
}

install.gitee <- function(url, subdir=NULL, cran=NULL){
    url = sub("/+$","",url)
    descfile <- paste0(url, "/raw/main", ifelse(is.null(subdir), "", paste0("/",subdir)), "/DESCRIPTION")
    if(!test_web(descfile)) {
        descfile <- paste0(url, "/raw/master", ifelse(is.null(subdir), "", paste0("/",subdir)), "/DESCRIPTION")
        if(!test_web(descfile)) {
            stop("not found DESCRIPTION file")
        }
    }
    a <- desc2pkgs(descfile)
    if(nrow(a)>0) {
        pkgs <- setdiff(a$pkg, "R")
        if (length(pkgs) > 0)  install.pkgs(pkgs)
    }
    remotes::install_git(url, subdir=subdir, upgrade="never")
}
