#!/usr/bin/env python3
"""A2. Two-feature (TLOD + DP) baseline classifier.

Reviewer 3, Comment 4: "Would a simpler classifier that uses only TLOD and
depth perform similarly?"

Trains on the released/cached training matrix (dtm_0819.txt, 493,937 rows) and
evaluates on the HG004->HG005 and HG005->HG004 PosMix test matrix
(4.test/45|54, 60 condition files).

Models compared, all with identical XGBoost hyperparameters:
  full17   - the 17 features used by SomaticXGB
  tlod_dp  - TLOD + DP only (XGBoost)
  tlod     - TLOD only
  dp       - DP only
  logit2   - logistic regression on standardized TLOD + DP
  logit2q  - logistic regression on TLOD, DP, TLOD^2, DP^2, TLOD*DP
  threshold- simple rule: TLOD >= cut (cut selected on the training set for best F1@0.1-equivalent)
The released model's own scores (`pred`) are evaluated on the same rows for reference.

Outputs (tables/):
  A2_model_metrics_overall.csv
  A2_model_metrics_by_platform.csv
  A2_model_metrics_by_depth.csv
  A2_model_metrics_by_group.csv
  A2_threshold_curves.csv
  A2_predictions_test.csv.gz
  A2_training_feature_importance.csv
"""

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
import common  # noqa: E402
import os, glob, json
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve

BASE = str(common.WORK_ROOT)
TRAIN = str(common.TRAIN_MATRIX)
OUT = str(common.RESULTS_ROOT)

FEATURES = ['TLOD','DP','ECNT','MPOS','NALOD','NLOD','MBQ_REF','MBQ_ALT',
            'MFRL_REF','MFRL_ALT','MMQ_REF','MMQ_ALT','avg_mapping_quality',
            'avg_basequality','avg_pos_as_fraction',
            'avg_num_mismatches_as_fraction','avg_sum_mismatch_qualities']
TWO = ['TLOD','DP']

PLAT = {'FASTASeq_300':'FASTASeq 300','GenoLab_M':'GenoLab M',
        'NovaSeq_6000':'NovaSeq 6000','NovaSeq_X':'NovaSeq X',
        'SURFSeq_5000':'SURFSeq 5000','MGISEQ-T7':'DNBSEQ-T7'}

PARAMS = {'objective':'binary:logistic','eval_metric':'aucpr','max_depth':7,
          'eta':0.1,'nthread':32,'seed':42,'verbosity':0}
NROUND = 400


def load_train():
    tr = pd.read_csv(TRAIN, sep='\t', low_memory=False)
    tr['plat'] = tr['plat'].replace(PLAT)
    tr['reads'] = pd.to_numeric(tr['reads'], errors='coerce')
    # Guard against inf, which xgboost rejects in some versions.
    X = tr[FEATURES].copy()
    bad = ~np.isfinite(X.select_dtypes(include=[np.number]))
    nbad = int(bad.sum().sum())
    if nbad:
        print(f'[train] replacing {nbad} non-finite values with NaN')
        X = X.mask(bad)
    return tr, X.astype(np.float32)


def load_test():
    rows = []
    for g in ['45', '54']:
        for f in sorted(glob.glob(f'{BASE}/4.test/{g}/pred/*_pred.txt.gz')):
            d = pd.read_csv(f, sep='\t', compression='gzip')
            d['group'] = g
            d['file'] = os.path.basename(f)
            d['plat'] = d['id'].str.split('_reads').str[0].map(PLAT).fillna(d['id'].str.split('_reads').str[0])
            d['reads'] = d['reads'].astype(int)
            rows.append(d[FEATURES + ['label','var','CHROM','POS','REF','ALT','id','plat','reads','group','pred','file']])
    te = pd.concat(rows, ignore_index=True)
    X = te[FEATURES].copy().astype(np.float32)
    X = X.replace([np.inf, -np.inf], np.nan)
    return te, X


def best_f1(y, p):
    """Best achievable F1 over all thresholds and the threshold that attains it."""
    order = np.argsort(-p)
    pp = p[order]; yy = y[order]
    tp = np.cumsum(yy); fp = np.cumsum(1 - yy)
    fn = yy.sum() - tp
    f1 = 2 * tp / np.maximum(2 * tp + fp + fn, 1)
    i = int(np.argmax(f1))
    thr = pp[i]
    # threshold half-way to the next lower score, so the rule is reproducible
    thr = thr if i == len(pp) - 1 else (pp[i] + pp[i + 1]) / 2
    return float(f1[i]), float(thr)


def metrics(y, p, fixed=(0.05, 0.1, 0.2)):
    row = {'AUC': roc_auc_score(y, p), 'PR_AUC': average_precision_score(y, p)}
    for t in fixed:
        c = p > t
        tp = int((c & (y == 1)).sum()); fp = int((c & (y == 0)).sum()); fn = int(((~c) & (y == 1)).sum())
        row[f'F1@{t}'] = 2 * tp / max(2 * tp + fp + fn, 1)
        row[f'Precision@{t}'] = tp / max(tp + fp, 1)
        row[f'Recall@{t}'] = tp / max(tp + fn, 1)
        row[f'NCalls@{t}'] = tp + fp
    bf, bt = best_f1(y, p)
    row['F1_best'] = bf; row['Threshold_best'] = bt
    return row


def main():
    os.makedirs(f'{OUT}/tables', exist_ok=True)
    print('loading training matrix ...')
    tr, Xtr = load_train()
    ytr = tr['label'].astype(int).values
    print('train rows', len(tr), 'pos', int(ytr.sum()), 'neg', int((ytr == 0).sum()))
    print('train platforms', tr['plat'].value_counts().to_dict())

    print('loading test matrix ...')
    te, Xte = load_test()
    yte = te['label'].astype(int).values
    print('test rows', len(te), 'pos', int(yte.sum()), 'neg', int((yte == 0).sum()))
    print('test platforms', te['plat'].value_counts().to_dict())

    preds = {}
    # ---- XGBoost variants -------------------------------------------------
    for name, cols in [('full17', FEATURES), ('tlod_dp', TWO), ('tlod', ['TLOD']), ('dp', ['DP'])]:
        print(f'training xgb {name} ({len(cols)} features) ...')
        m = xgb.train(PARAMS, xgb.DMatrix(Xtr[cols].values, label=ytr, feature_names=cols), num_boost_round=NROUND)
        preds[name] = m.predict(xgb.DMatrix(Xte[cols].values, feature_names=cols))
        imp = m.get_score(importance_type='gain')
        pd.DataFrame({'model': name, 'feature': list(imp), 'gain': list(imp.values())}) \
          .sort_values('gain', ascending=False) \
          .to_csv(f'{OUT}/tables/A2_training_feature_importance_{name}.csv', index=False)

    # ---- logistic variants ------------------------------------------------
    print('training logistic TLOD+DP ...')
    sc = StandardScaler().fit(Xtr[TWO].values)
    lr = LogisticRegression(max_iter=2000, n_jobs=8, random_state=42)
    lr.fit(sc.transform(Xtr[TWO].values), ytr)
    preds['logit2'] = lr.predict_proba(sc.transform(Xte[TWO].values))[:, 1]

    # quadratic logistic: TLOD, DP, squares, interaction (still "only TLOD & depth")
    Xq = np.c_[Xtr[TWO].values, Xtr['TLOD'].values ** 2, Xtr['DP'].values ** 2,
               Xtr['TLOD'].values * Xtr['DP'].values]
    Xq = np.nan_to_num(Xq, nan=0.0, posinf=0.0, neginf=0.0)
    scq = StandardScaler().fit(Xq)
    lrq = LogisticRegression(max_iter=2000, n_jobs=8, random_state=42)
    lrq.fit(scq.transform(Xq), ytr)
    Xtq = np.c_[Xte[TWO].values, Xte['TLOD'].values ** 2, Xte['DP'].values ** 2,
                Xte['TLOD'].values * Xte['DP'].values]
    Xtq = np.nan_to_num(Xtq, nan=0.0, posinf=0.0, neginf=0.0)
    preds['logit2q'] = lrq.predict_proba(scq.transform(Xtq))[:, 1]

    # ---- simple TLOD threshold rule --------------------------------------
    # pick the TLOD cut maximising F1 on the training set
    m0 = metrics(ytr, tr['TLOD'].values.astype(float), fixed=())
    print('training TLOD-only best F1', m0['F1_best'], 'at TLOD >=', m0['Threshold_best'])
    preds['tlod_rule'] = te['TLOD'].values.astype(float)
    tl_cut = m0['Threshold_best']

    # ---- released model ---------------------------------------------------
    preds['released'] = te['pred'].values.astype(float)

    # ---- assemble predictions --------------------------------------------
    pcols = [f'{k}_pred' for k in preds]
    out = te[['var','CHROM','POS','REF','ALT','id','plat','reads','group','label']].copy()
    for k, v in preds.items():
        out[f'{k}_pred'] = v
    out.to_csv(f'{OUT}/tables/A2_predictions_test.csv.gz', index=False, compression='gzip')

    # ---- metrics ----------------------------------------------------------
    def summarize(keys):
        rows = []
        for k, v in preds.items():
            if keys is None:
                row = {'Model': k, **metrics(yte, v)}
                rows.append(row)
            else:
                for kv, idx in out.groupby(keys, sort=True).groups.items():
                    y = yte[idx]; p = v[idx]
                    row = {keys[0] if isinstance(keys, str) else ' / '.join(keys): kv, 'Model': k}
                    row.update(metrics(y, p))
                    rows.append(row)
        return pd.DataFrame(rows)

    overall = summarize(None)
    by_plat = summarize('plat')
    by_depth = summarize('reads')
    by_group = summarize('group')
    by_plat_depth = summarize(['plat','reads'])

    overall.to_csv(f'{OUT}/tables/A2_model_metrics_overall.csv', index=False)
    by_plat.to_csv(f'{OUT}/tables/A2_model_metrics_by_platform.csv', index=False)
    by_depth.to_csv(f'{OUT}/tables/A2_model_metrics_by_depth.csv', index=False)
    by_group.to_csv(f'{OUT}/tables/A2_model_metrics_by_group.csv', index=False)
    by_plat_depth.to_csv(f'{OUT}/tables/A2_model_metrics_by_platform_depth.csv', index=False)

    # threshold curves for figure / source data
    curves = []
    for k, v in preds.items():
        fpr, tpr, thr = roc_curve(yte, v)
        # precision-recall curve
        order = np.argsort(-v)
        yy = yte[order]
        tp = np.cumsum(yy); fp = np.cumsum(1 - yy)
        rec = tp / max(yy.sum(), 1); prec = tp / np.maximum(tp + fp, 1)
        for i in np.linspace(0, len(v) - 1, 500).astype(int):
            curves.append({'Model': k, 'Recall': float(rec[i]), 'Precision': float(prec[i]),
                           'Threshold': float(v[order][i])})
    pd.DataFrame(curves).to_csv(f'{OUT}/tables/A2_threshold_curves.csv', index=False)

    # verify the simple TLOD rule cut used above
    with open(f'{OUT}/tables/A2_model_config.json', 'w') as fh:
        json.dump({'xgb_params': PARAMS, 'n_rounds': NROUND,
                   'features_full': FEATURES, 'features_two': TWO,
                   'training_rows': int(len(tr)), 'training_pos': int(ytr.sum()),
                   'test_rows': int(len(te)), 'test_pos': int(yte.sum()),
                   'tlod_rule_cut_training_bestF1': tl_cut,
                   'logistic_coef_standardized': dict(zip(TWO, lr.coef_[0].tolist())),
                   'logistic_intercept': float(lr.intercept_[0])}, fh, indent=2)

    # ---- console summary --------------------------------------------------
    cols = ['Model','AUC','PR_AUC','F1@0.1','F1@0.2','F1_best','Threshold_best']
    print('\n=== overall ===')
    print(overall[cols].round(4).to_string(index=False))
    print('\n=== by platform, F1 @0.1 / F1_best ===')
    print(by_plat.pivot(index='plat', columns='Model', values=['F1@0.1','F1_best']).round(4).to_string())
    print('\n=== deltas full17 - tlod_dp ===')
    ov = overall.set_index('Model')
    for c in ['AUC','PR_AUC','F1@0.1','F1@0.2','F1_best']:
        print(c, round(ov.loc['full17', c] - ov.loc['tlod_dp', c], 5))
    print('\nwrote tables to', f'{OUT}/tables')


if __name__ == '__main__':
    main()
