fsave <- function (
  ...,
  list = character(),
  file = stop("'file' must be specified"),
  envir = parent.frame(),
  ncores = 4,
  eval.promises = TRUE,
  precheck = TRUE)
{
  check_bgzip()
  if (!is.numeric(ncores))
    stop("'ncores' mut be numeric")
  names <- as.character(substitute(list(...)))[-1L]
  if (missing(list) && !length(names))
    warning("nothing specified to be save()d")
  list <- c(list, names)
  if (precheck) {
    ok <- vapply(list, exists, NA, envir = envir)
    if (!all(ok)) {
      n <- sum(!ok)
      stop(sprintf(
        ngettext(n, "object %s not found",
                 "objects %s not found"),
        paste(sQuote(list[!ok]),
              collapse = ", ")
      ), domain = NA)
    }
  }
  ncores.arg <- paste0(' -@ ', ncores)
  con <- pipe(paste0('bgzip ', ncores.arg, ' > ', file))
  on.exit(close(con))
  
  save(..., list=list, file=con, envir=envir, ascii=FALSE, version=2,
       eval.promises=eval.promises, precheck=precheck)
}

fsave.image <- function(
  file = ".frdata",
  ncores = 4,
  safe = TRUE) {
  ## Check arguments
  if (is.null(file)) {
    stop("file argument is required")
  }
  if (safe) {
    outfile <- paste0(file, "Tmp")
    i <- 0
    while (file.exists(outfile)) {
      i <- i + 1
      outfile <- paste0(file, "Tmp", i)
    }
  }
  else
    outfile <- file
  on.exit(file.remove(outfile))
  fsave(
    list = names(.GlobalEnv),
    file = outfile,
    envir = .GlobalEnv,
    ncores = ncores,
    precheck = F
  )
  if (safe)
    if (!file.rename(outfile, file)) {
      on.exit()
      stop(gettextf("image could not be renamed and is left in %s",
                    outfile),
           domain = NA)
    }
  on.exit()
}

fload <- function (file, envir = parent.frame(), verbose = FALSE, ncores = 4)
{
  if (!is.numeric(ncores))
    stop("'ncores' mut be numeric")
  if (!is.character(file))
    stop("bad 'file' argument")

  check_bgzip()
  
  con <- pipe(paste0('bgzip -dc -@ ', ncores, ' ', file))
  on.exit(close(con))
  
  load(con,envir,verbose)
}

check_bgzip <- function() {
  if (.Platform$OS.type == "unix") {
    if (system(
      'command -v bgzip',
      wait = T,
      ignore.stdout = TRUE,
      ignore.stderr = TRUE
    ) > 0) {
      stop('The bgzip command is not available on this system!')
    }
  } else {
    stop('Platform is not a unix system!')
  }
}

fsaveRDS <- function(
  object, 
  file = stop("'file' must be specified"),
  ncores = 4
){
  check_bgzip()
  if (!is.numeric(ncores))
    stop("'ncores' mut be numeric")
  
  ncores.arg <- paste0(' -@ ', ncores)
  con <- pipe(paste0('bgzip ', ncores.arg, ' > ', file))
  on.exit(close(con))
  saveRDS(object, file = con)
}

freadRDS <- function(file, ncores=4){
  check_bgzip()
  if (!is.numeric(ncores))
    stop("'ncores' mut be numeric")
  con <- pipe(paste0('bgzip -dc -@ ', ncores, ' ', file))
  on.exit(close(con))
  readRDS(con)
}
