# Extract the Fig. 1b plotting data (df_sim) from the original PosMix/posfeature files.
# This uses the exact data-preparation code from tumor-paper2-dnbseq-t7-260610.ipynb, cell 9.
library(data.table)
library(magrittr)
library(jsonlite)
library(stringr)

YXDIR <- "/mnt/nfsp2/projects/linyixin/240703-paper/"
js <- read_json(paste0(YXDIR, "240910-actual/241028-newposmix/config.json"))
BASES4 <- c(A="T", C="G", G="C", T="A")
platcols = data.table(
    abbr = c("FA","GL","SF","NV","NX","T7"),
    name = c("FASTASeq 300","GenoLab M","SURFSeq 5000","NovaSeq 6000","NovaSeq X","DNBSEQ-T7"),
    col = c("#e41a1c","#377eb8","#4daf4a","#984ea3","#ff7f00","#a65628")
) %>% .[, fac:=factor(name, levels = name)]

x = "FASTASeq_300"
f = function(x){
a = fread("/mnt/nfsp2/projects/linyixin/240703-paper/240910-actual/241028-newposmix/pf_actual_0910.txt") %>%
.[, var:=paste(V1,V2,V4,V5,sep=":")]

b = fread(paste0(YXDIR,"240910-actual/241028-newposmix/posmix/",x,"_T_reads150M.bam.pf")) %>%
.[ref %in% BASES4 & alt %in% BASES4 & type %in% c("=","X") & qual>=0] %>%
.[isfwd==0, ref:=BASES4[ref]] %>%
.[isfwd==0, alt:=BASES4[alt]] %>%
.[, var:=paste(chrom,pos,ref,alt,sep=":")] %>%
.[, .(ad=.N), by=c("chrom","pos","ref","alt","var")] %>%
.[, dp:=sum(ad), by=c("chrom","pos")] %>%
.[, freq:=ad/dp]
d = fread(cmd=paste("gzip -dc", paste0(YXDIR,"240910-actual/241028-newposmix/posmix/",x,"_donor.pf.txt.gz"))) %>%
.[ref %in% BASES4 & alt %in% BASES4 & type %in% c("=","X") & qual>=0] %>%
.[isfwd==0, ref:=BASES4[ref]] %>%
.[isfwd==0, alt:=BASES4[alt]] %>%
.[, var:=paste(chrom,pos,ref,alt,sep=":")] %>%
.[, .(ad=.N), by=c("chrom","pos","ref","alt","var")] %>%
.[, dp:=sum(ad), by=c("chrom","pos")] %>%
.[, freq:=ad/dp]

g = fread(cmd=paste("gzip -dc", paste0(YXDIR,"240910-actual/241028-newposmix/posmix/",x,"_backgroud.pf.txt.gz"))) %>%
.[ref %in% BASES4 & alt %in% BASES4 & type %in% c("=","X") & qual>=0] %>%
.[isfwd==0, ref:=BASES4[ref]] %>%
.[isfwd==0, alt:=BASES4[alt]] %>%
.[, var:=paste(chrom,pos,ref,alt,sep=":")] %>%
.[, .(ad=.N), by=c("chrom","pos","ref","alt","var")] %>%
.[, dp:=sum(ad), by=c("chrom","pos")] %>%
.[, freq:=ad/dp]    
    
df = merge(a[,.(var,freq_exp1=V11,gt=V6,freq_replace=V3)], 
                 b[,.(var,freq_sim=freq,dp_sim=dp)], by="var") %>%
           merge(d[,.(var,freq_donor=freq,dp_donor=dp)], by="var") %>%
           merge(g[,.(var,freq_bg=freq,dp_bg=dp)], by="var") %>%
.[, Platform:=sub("_"," ", x)]
df
}

df_sim = lapply(js$data, function(x){f(x$id)}) %>% rbindlist() %>%
.[, freq_exp2:=freq_donor*freq_replace] %>%
.[, freq_exp3:=freq_donor*freq_replace+(1-freq_replace)*freq_bg] %>%
.[gt=="1/0", gt:="0/1"] %>%
.[Platform=='MGISEQ-T7',Platform:='DNBSEQ-T7']%>%
.[, Platform:=factor(Platform, platcols$name)]
fwrite(df_sim, "plot_data/fig1b_df_sim.csv.gz", sep="\t", compress="gzip")
cat("wrote plot_data/fig1b_df_sim.csv.gz:", nrow(df_sim), "rows\n")
