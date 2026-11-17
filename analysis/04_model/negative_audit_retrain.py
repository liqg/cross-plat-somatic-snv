
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
import common  # noqa: E402
import pandas as pd, numpy as np, glob, os, re, pickle, bisect
import xgboost as xgb
from sklearn.metrics import roc_auc_score, average_precision_score

out=str(common.RESULTS_ROOT)
features=['TLOD','DP','ECNT','MPOS','NALOD','NLOD','MBQ_REF','MBQ_ALT','MFRL_REF','MFRL_ALT','MMQ_REF','MMQ_ALT','avg_mapping_quality','avg_basequality','avg_pos_as_fraction','avg_num_mismatches_as_fraction','avg_sum_mismatch_qualities']
giab=pickle.load(open(str(common.GIAB_PANEL_VARS),'rb'))
# exact variants and positions per sample
gvars={s:set(df['var']) for s,df in giab.items()}
gpos={s:set(df.chrom.astype(str)+':'+df.pos.astype(str)) for s,df in giab.items()}

def spiked_positions(path):
    df=pd.read_csv(path,sep='\t')
    df['CHROM']=df.CHROM.astype(str); df['POS']=pd.to_numeric(df.POS)
    return {c:np.sort(g.POS.values) for c,g in df.groupby('CHROM')}

def near(posmap, chrom, pos, dist=150):
    arr=posmap.get(str(chrom))
    if arr is None: return False
    i=bisect.bisect_left(arr,pos)
    if i<len(arr) and arr[i]-pos<=dist: return True
    if i>0 and pos-arr[i-1]<=dist: return True
    return False

# training
train=pd.read_csv(str(common.TRAIN_MATRIX),sep='\t',low_memory=False)
train['g']=train.group.str.split('/').str[-1]
donor={'12':'HG001','13':'HG001','21':'HG002','23':'HG002','31':'HG003','32':'HG003'}
bg={'12':'HG002','13':'HG003','21':'HG001','23':'HG003','31':'HG001','32':'HG002'}
sp_pos={g:spiked_positions(f'{common.TRAIN_ONLY_DIR}/{g}/only{g}.txt') for g in donor}
amb=[]
for idx,r in train.iterrows():
    g=r.g; c=str(r.CHROM); p=int(r.POS); v=r['var']
    a=(v in gvars[donor[g]]) or (v in gvars[bg[g]]) or (f'{c}:{p}' in gpos[donor[g]]) or (f'{c}:{p}' in gpos[bg[g]]) or near(sp_pos[g],c,p,150)
    amb.append(a if r.label==0 else False)
train['ambiguous']=amb
print('train ambiguous',train.ambiguous.sum(),'of',len(train),'neg',train[(train.label==0)&train.ambiguous].shape[0])
tr=train[~train.ambiguous].copy()
print('train retained',len(tr),'pos',tr.label.sum(),'neg',(tr.label==0).sum())
# test
test=[]
for g in ['45','54']:
    for f in glob.glob(f'{common.TEST_ROOT}/{g}/pred/*_pred.txt.gz'):
        d=pd.read_csv(f,sep='\t',compression='gzip')
        d['group']=g; d['idfile']=os.path.basename(f); d['plat']=d.id.str.split('_reads').str[0]
        d['reads']=d.reads.astype(int)
        test.append(d[features+['label','var','pred','group','plat','reads','CHROM','POS']])
test=pd.concat(test,ignore_index=True)
tdon={'45':'HG004','54':'HG005'}; tbg={'45':'HG005','54':'HG004'}
# For test, pf path has header columns; use read with names
def spiked_positions_pf(path):
    df=pd.read_csv(path,sep='\t',header=None,names=['CHROM','POS','P','REF','ALT','GT','f','b','GT2','group','AF'])
    df['CHROM']=df.CHROM.astype(str); df['POS']=pd.to_numeric(df.POS)
    return {c:np.sort(g.POS.values) for c,g in df.groupby('CHROM')}
tsp={g:spiked_positions_pf(f'{common.REPO_ROOT}/workflow/pos_files/test/{g}/pf{g}_low.txt') for g in ['45','54']}
tamb=[]
for idx,r in test.iterrows():
    g=r.group; c=str(r.CHROM); p=int(r.POS); v=r['var']
    a=(v in gvars[tdon[g]]) or (v in gvars[tbg[g]]) or (f'{c}:{p}' in gpos[tdon[g]]) or (f'{c}:{p}' in gpos[tbg[g]]) or near(tsp[g],c,p,150)
    tamb.append(a if r.label==0 else False)
test['ambiguous']=tamb
te=test[~test.ambiguous].copy()
print('test ambiguous',test.ambiguous.sum(),'of',len(test),'retained',len(te),'pos',te.label.sum(),'neg',(te.label==0).sum())
# train two models: baseline on all training rows, audit on retained
params={'objective':'binary:logistic','eval_metric':'aucpr','max_depth':7,'eta':0.1,'nthread':8,'seed':42}
res=[]
for name,data in [('baseline_all',train),('audit_filtered',tr)]:
    dtr=xgb.DMatrix(data[features].values,label=data.label.values,feature_names=features)
    model=xgb.train(params,dtr,num_boost_round=300)
    p=model.predict(xgb.DMatrix(te[features].values,feature_names=features))
    te[name+'_pred']=p
    res.append((name,roc_auc_score(te.label,p),average_precision_score(te.label,p)))
print('metrics on audit-filtered test:',res)
# F1 at threshold 0.1
for name,_,_ in res:
    p=te[name+'_pred'].values; y=te.label.values
    calls=p>0.1; tp=((calls)&(y==1)).sum(); fp=((calls)&(y==0)).sum(); fn=((~calls)&(y==1)).sum()
    print(name,'F1@0.1',2*tp/(2*tp+fp+fn),'precision',tp/(tp+fp) if tp+fp else None,'recall',tp/(tp+fn))
# existing model on same filtered test
p=te.pred.values; y=te.label.values
print('existing model on filtered test: AUC',roc_auc_score(y,p),'PR',average_precision_score(y,p))
calls=p>0.1; tp=((calls)&(y==1)).sum(); fp=((calls)&(y==0)).sum(); fn=((~calls)&(y==1)).sum()
print('existing F1@0.1',2*tp/(2*tp+fp+fn),'precision',tp/(tp+fp),'recall',tp/(tp+fn))
te[['var','plat','reads','label','ambiguous','baseline_all_pred','audit_filtered_pred','pred']].to_csv(f'{out}/negative_audit_test_predictions.csv.gz',index=False,compression='gzip')
train[['var','group','plat','reads','label','ambiguous']].to_csv(f'{out}/negative_audit_train_labels.csv.gz',index=False,compression='gzip')
