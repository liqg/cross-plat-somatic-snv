
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
import common  # noqa: E402
import pandas as pd, numpy as np, glob, os, xgboost as xgb
from sklearn.metrics import roc_auc_score, average_precision_score
out=str(common.TABLES)
features=['TLOD','DP','ECNT','MPOS','NALOD','NLOD','MBQ_REF','MBQ_ALT','MFRL_REF','MFRL_ALT','MMQ_REF','MMQ_ALT','avg_mapping_quality','avg_basequality','avg_pos_as_fraction','avg_num_mismatches_as_fraction','avg_sum_mismatch_qualities']
train=pd.read_csv(str(common.TRAIN_MATRIX),sep='\t',low_memory=False)
train['plat']=train.plat.replace({'MGISEQ-T7':'DNBSEQ-T7','GenoLab_M':'GenoLab M','NovaSeq_6000':'NovaSeq 6000','NovaSeq_X':'NovaSeq X','SURFSeq_5000':'SURFSeq 5000','FASTASeq_300':'FASTASeq 300'})
test=[]
for g in ['45','54']:
    for f in glob.glob(f'{common.TEST_ROOT}/{g}/pred/*_pred.txt.gz'):
        d=pd.read_csv(f,sep='\t',compression='gzip')
        d['plat']=d.id.str.split('_reads').str[0].replace({'MGISEQ-T7':'DNBSEQ-T7','GenoLab_M':'GenoLab M','NovaSeq_6000':'NovaSeq 6000','NovaSeq_X':'NovaSeq X','SURFSeq_5000':'SURFSeq 5000','FASTASeq_300':'FASTASeq 300'})
        test.append(d[features+['plat','label','var']])
test=pd.concat(test,ignore_index=True)
params={'objective':'binary:logistic','eval_metric':'aucpr','max_depth':7,'eta':0.1,'nthread':8,'seed':42}
rows=[]
for plat in sorted(train.plat.unique()):
    tr=train[train.plat!=plat]; te=test[test.plat==plat]
    m=xgb.train(params,xgb.DMatrix(tr[features].values,label=tr.label.values,feature_names=features),num_boost_round=300)
    p=m.predict(xgb.DMatrix(te[features].values,feature_names=features))
    rows.append({'Held_out_platform':plat,'N_train':len(tr),'N_test':len(te),'AUC':roc_auc_score(te.label,p),'PR_AUC':average_precision_score(te.label,p)})
df=pd.DataFrame(rows); df.to_csv(f'{out}/lopo_results.csv',index=False)
print(df.round(4).to_string(index=False))
