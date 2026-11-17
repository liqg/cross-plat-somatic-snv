
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
import common  # noqa: E402
import pandas as pd, numpy as np, subprocess, glob, os, re
base=str(common.WORK_ROOT)
out=str(common.TABLES)
plat_map={'FA':'FASTASeq 300','GL':'GenoLab M','NV':'NovaSeq 6000','NX':'NovaSeq X','SF':'SURFSeq 5000','T7':'DNBSEQ-T7','MGISEQ-T7':'DNBSEQ-T7'}

def add_dist(df):
    df=df.copy(); df['CHROM']=df.CHROM.astype(str); df['POS']=pd.to_numeric(df.POS)
    d=[]
    for chrom in df.CHROM.unique():
        sub=df[df.CHROM==chrom]; pos=sub.POS.values; n=len(pos)
        if n==1: dd=[10000]
        elif n==2: dd=[pos[1]-pos[0]]*2
        else:
            d1=np.r_[10000,pos[1:]-pos[:-1]]; d2=np.r_[pos[1:]-pos[:-1],10000]; dd=np.minimum(d1,d2)
        d.extend(dd)
    df['dist']=d; return df

def vcfvars(path):
    res=set()
    q=subprocess.run(['bcftools','query','-f','%CHROM\t%POS\t%REF\t%ALT\t%FILTER\n',path],capture_output=True,text=True)
    for line in q.stdout.strip().split('\n'):
        if not line: continue
        c,p,r,a,flt=line.split('\t')
        if flt not in ('PASS','.') and 'PASS' not in flt: continue
        if len(r)==1 and len(a)==1 and ',' not in a:
            res.add(c.replace('chr','')+':'+p+':'+r+':'+a)
    return res

rows=[]
for g in ['45','54']:
    pf=pd.read_csv(f'{common.REPO_ROOT}/workflow/pos_files/test/{g}/pf{g}_low.txt',sep='\t',header=None,names=['CHROM','POS','P','REF','ALT','GT','f','b','GT2','group','AF'])
    pf=add_dist(pf); pf['var']=pf.CHROM+':'+pf.POS.astype(str)+':'+pf.REF+':'+pf.ALT
    truth=set(pf.loc[pf.dist>=50,'var']); near=set(pf.loc[pf.dist<50,'var'])
    for sid in sorted(os.listdir(f'{base}/4.test/{g}/strelka')):
        if not os.path.isdir(f'{base}/4.test/{g}/strelka/{sid}'): continue
        if not sid.endswith('reads100M'): continue
        plat=plat_map.get(sid.split('_reads')[0],sid.split('_reads')[0])
        s=vcfvars(f'{base}/4.test/{g}/strelka/{sid}/results/variants/somatic.snvs.vcf.gz')
        m=vcfvars(f'{base}/4.test/{g}/mutect/{sid}.filter.vcf.gz')
        v=vcfvars(f'{base}/4.test/{g}/varscan_novo/{sid}.snp.pass.vcf.gz')
        pred=pd.read_csv(f'{base}/4.test/{g}/pred/{sid}_pred.txt.gz',sep='\t',compression='gzip')
        x=set(pred.loc[pred.pred>0.1,'var'])
        for name,calls in [('SomaticXGB 0.1',x),('Strelka2',s),('Mutect2',m),('Strelka2_Mutect2',s|m),('VarScan2_fpfilter',v)]:
            calls=set(calls)-near
            tp=len(calls&truth); fp=len(calls-truth); fn=len(truth-calls)
            rec=tp/(tp+fn) if tp+fn else np.nan; prec=tp/(tp+fp) if tp+fp else np.nan; f1=2*rec*prec/(rec+prec) if rec+prec else np.nan
            rows.append({'Group':g,'Platform':plat,'Sample':sid,'Method':name,'TP':tp,'FP':fp,'FN':fn,'Recall':rec,'Precision':prec,'F1':f1})
df=pd.DataFrame(rows)
df.to_csv(f'{out}/fig3a_test_100M_methods.csv',index=False)
print(df.groupby(['Method'])[['Recall','Precision','F1']].mean().round(4).to_string())
print('\nF1 by platform/method:')
print(df.pivot_table(index='Platform',columns='Method',values='F1',aggfunc='mean').round(3).to_string())
