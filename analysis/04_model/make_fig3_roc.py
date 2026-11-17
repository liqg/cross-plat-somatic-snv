
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
import common  # noqa: E402
import pandas as pd, numpy as np, glob, os, re, xgboost as xgb
from sklearn.metrics import roc_curve, auc, average_precision_score
base=str(common.WORK_ROOT); out=str(common.TABLES)
features=['TLOD','DP','ECNT','MPOS','NALOD','NLOD','MBQ_REF','MBQ_ALT','MFRL_REF','MFRL_ALT','MMQ_REF','MMQ_ALT','avg_mapping_quality','avg_basequality','avg_pos_as_fraction','avg_num_mismatches_as_fraction','avg_sum_mismatch_qualities']
train=pd.read_csv(str(common.TRAIN_MATRIX),sep='\t',low_memory=False)
train['plat']=train.plat.replace({'MGISEQ-T7':'DNBSEQ-T7','GenoLab_M':'GenoLab M','NovaSeq_6000':'NovaSeq 6000','NovaSeq_X':'NovaSeq X','SURFSeq_5000':'SURFSeq 5000','FASTASeq_300':'FASTASeq 300'})
# test features
test=[]
for g in ['45','54']:
    for f in glob.glob(f'{base}/4.test/{g}/pred/*_pred.txt.gz'):
        d=pd.read_csv(f,sep='\t',compression='gzip')
        d['plat']=d.id.str.split('_reads').str[0].replace({'MGISEQ-T7':'DNBSEQ-T7','GenoLab_M':'GenoLab M','NovaSeq_6000':'NovaSeq 6000','NovaSeq_X':'NovaSeq X','SURFSeq_5000':'SURFSeq 5000','FASTASeq_300':'FASTASeq 300'})
        d['reads']=d.reads.astype(int)
        test.append(d[features+['plat','reads','label','var','pred','id']])
test=pd.concat(test,ignore_index=True)
# all-platform model = existing pred; NovaSeq X-only model trained on NovaSeq X rows
params={'objective':'binary:logistic','eval_metric':'aucpr','max_depth':7,'eta':0.1,'nthread':8,'seed':42}
tr=train[train.plat=='NovaSeq X']
dtr=xgb.DMatrix(tr[features].values,label=tr.label.values,feature_names=features)
model=xgb.train(params,dtr,num_boost_round=300)
test['pred_nx']=model.predict(xgb.DMatrix(test[features].values,feature_names=features))
test[['plat','reads','label','var','pred','pred_nx','id']].to_csv(f'{out}/fig3_test_predictions.csv.gz',index=False,compression='gzip')
# ROC by platform
rows=[]
for plat,g in test.groupby('plat'):
    for col in ['pred','pred_nx']:
        fpr,tpr,_=roc_curve(g.label,g[col])
        rows.append(pd.DataFrame({'Platform':plat,'Model':'All' if col=='pred' else 'NovaSeq X','FPR':fpr,'TPR':tpr}))
    print(plat,'AUC all',auc(*roc_curve(g.label,g.pred)[:2]),'AUC nx',auc(*roc_curve(g.label,g.pred_nx)[:2]),'PR all',average_precision_score(g.label,g.pred),'PR nx',average_precision_score(g.label,g.pred_nx))
pd.concat(rows,ignore_index=True).to_csv(f'{out}/fig3_roc_curves.csv.gz',index=False,compression='gzip')
print('saved')
