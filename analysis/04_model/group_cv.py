
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
import common  # noqa: E402
import pandas as pd, numpy as np, xgboost as xgb
from sklearn.metrics import roc_auc_score, average_precision_score
out=str(common.TABLES)
features=['TLOD','DP','ECNT','MPOS','NALOD','NLOD','MBQ_REF','MBQ_ALT','MFRL_REF','MFRL_ALT','MMQ_REF','MMQ_ALT','avg_mapping_quality','avg_basequality','avg_pos_as_fraction','avg_num_mismatches_as_fraction','avg_sum_mismatch_qualities']
d=pd.read_csv(str(common.TRAIN_MATRIX),sep='\t',low_memory=False)
d['g']=d.group.str.split('/').str[-1]
params={'objective':'binary:logistic','eval_metric':'aucpr','max_depth':7,'eta':0.1,'nthread':8,'seed':42}
rows=[]
for g in sorted(d.g.unique()):
    tr=d[d.g!=g]; te=d[d.g==g]
    m=xgb.train(params,xgb.DMatrix(tr[features].values,label=tr.label.values,feature_names=features),num_boost_round=300)
    p=m.predict(xgb.DMatrix(te[features].values,feature_names=features))
    rows.append({'Held_out_group':g,'N_train':len(tr),'N_test':len(te),'AUC':roc_auc_score(te.label,p),'PR_AUC':average_precision_score(te.label,p)})
df=pd.DataFrame(rows); df.to_csv(f'{out}/group_cv_results.csv',index=False)
print(df.round(4).to_string(index=False))
print('mean AUC',df.AUC.mean(),'mean PR',df.PR_AUC.mean())
