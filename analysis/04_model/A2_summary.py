#!/usr/bin/env python3
"""A2 summary: recompute metrics from saved test predictions, including a
proper simple-TLOD-threshold rule, per-condition paired comparisons, and
bootstrap confidence intervals over the 60 test conditions.

Reads:  tables/A2_predictions_test.csv.gz, tables/A2_model_config.json
Writes: tables/A2_model_metrics_overall.csv (overwritten, with tlod_rule fixed)
        tables/A2_model_metrics_by_platform.csv
        tables/A2_model_metrics_by_depth.csv
        tables/A2_paired_by_condition.csv
        tables/A2_bootstrap_deltas.csv
        figures/A2_f1_by_depth.png
        figures/A2_pr_curves.png
"""

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
import common  # noqa: E402
import os, glob, json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

OUT = str(common.RESULTS_ROOT)
PRED = f'{OUT}/tables/A2_predictions_test.csv.gz'
CFG = f'{OUT}/tables/A2_model_config.json'
MODELS = ['full17','tlod_dp','tlod','dp','logit2','logit2q','tlod_rule','released']
LABEL = {'full17':'SomaticXGB 17 features','tlod_dp':'XGBoost TLOD+DP','tlod':'XGBoost TLOD only',
         'dp':'XGBoost DP only','logit2':'Logistic TLOD+DP','logit2q':'Logistic TLOD+DP (quadratic)',
         'tlod_rule':'TLOD threshold rule','released':'Released SomaticXGB model'}


def best_f1(y, p):
    order = np.argsort(-p)
    pp = p[order]; yy = y[order]
    tp = np.cumsum(yy); fp = np.cumsum(1 - yy); fn = yy.sum() - tp
    f1 = 2 * tp / np.maximum(2 * tp + fp + fn, 1)
    i = int(np.argmax(f1))
    thr = pp[i] if i == len(pp) - 1 else (pp[i] + pp[i + 1]) / 2
    return float(f1[i]), float(thr)


def metrics(y, p, fixed=(0.05, 0.1, 0.2)):
    row = {'N': len(y), 'NPos': int(y.sum()), 'AUC': roc_auc_score(y, p),
           'PR_AUC': average_precision_score(y, p)}
    for t in fixed:
        c = p > t
        tp = int((c & (y == 1)).sum()); fp = int((c & (y == 0)).sum()); fn = int(((~c) & (y == 1)).sum())
        row[f'F1@{t}'] = 2 * tp / max(2 * tp + fp + fn, 1)
        row[f'Precision@{t}'] = tp / max(tp + fp, 1)
        row[f'Recall@{t}'] = tp / max(tp + fn, 1)
        row[f'NCalls@{t}'] = tp + fp
    bf, bt = best_f1(y, p)
    row['F1_best'] = bf; row['Threshold_best'] = bt
    # precision at fixed recall levels
    for r in (0.90, 0.95):
        idx = np.where(np.cumsum(y[np.argsort(-p)]) >= r * y.sum())[0]
        if len(idx):
            k = idx[0]
            prec = np.cumsum(y[np.argsort(-p)])[k] / (k + 1)
            row[f'Precision@Recall{r:.2f}'] = float(prec)
    return row


def bootstrap_ci(y, p, n=200, seed=1):
    rng = np.random.default_rng(seed)
    vals = []
    idx = np.arange(len(y))
    for _ in range(n):
        b = rng.choice(idx, size=len(idx), replace=True)
        vals.append(average_precision_score(y[b], p[b]))
    return np.percentile(vals, [2.5, 97.5])


def main():
    os.makedirs(f'{OUT}/figures', exist_ok=True)
    d = pd.read_csv(PRED)
    cfg = json.load(open(CFG))
    cut = cfg['tlod_rule_cut_training_bestF1']
    # Replace the raw TLOD score with a 0/1 rule at the training-selected cut,
    # but keep the raw score for curve plotting.
    d['tlod_rule_pred_raw'] = d['tlod_rule_pred']
    d['tlod_rule_pred'] = (d['tlod_rule_pred'] >= cut).astype(float)
    y = d['label'].astype(int).values

    rows = []
    for m in MODELS:
        rows.append({'Model': m, **metrics(y, d[f'{m}_pred'].values)})
    overall = pd.DataFrame(rows)
    overall.to_csv(f'{OUT}/tables/A2_model_metrics_overall.csv', index=False)

    for key, fn in [('plat','by_platform'), ('reads','by_depth'), ('group','by_group')]:
        rr = []
        for kv, g in d.groupby(key):
            yy = g['label'].astype(int).values
            for m in MODELS:
                rr.append({key: kv, 'Model': m, **metrics(yy, g[f'{m}_pred'].values)})
        pd.DataFrame(rr).to_csv(f'{OUT}/tables/A2_model_metrics_{fn}.csv', index=False)

    # paired comparison across the 60 condition files (the independent-ish units)
    per_cond = []
    for (plat, reads, group), g in d.groupby(['plat','reads','group']):
        yy = g['label'].astype(int).values
        row = {'Platform': plat, 'Reads': reads, 'Group': group, 'N': len(g), 'NPos': int(yy.sum())}
        for m in MODELS:
            row[f'{m}_F1best'] = best_f1(yy, g[f'{m}_pred'].values)[0]
            row[f'{m}_F1@0.1'] = metrics(yy, g[f'{m}_pred'].values, fixed=(0.1,))['F1@0.1']
        per_cond.append(row)
    per_cond = pd.DataFrame(per_cond)
    per_cond.to_csv(f'{OUT}/tables/A2_paired_by_condition.csv', index=False)

    print('\n=== overall ===')
    print(overall[['Model','AUC','PR_AUC','F1@0.1','F1@0.2','F1_best','Threshold_best']].round(4).to_string(index=False))

    print('\n=== paired full17 - tlod_dp across 60 conditions ===')
    for metric in ['F1best','F1@0.1']:
        delta = per_cond[f'full17_{metric}'] - per_cond[f'tlod_dp_{metric}']
        print(metric, 'median', round(delta.median(),4), 'mean', round(delta.mean(),4),
              'full17 better in', int((delta > 0).sum()), 'tied', int((delta == 0).sum()),
              'tlod_dp better in', int((delta < 0).sum()), 'of', len(delta))
    print('\n=== paired full17 - tlod across 60 conditions ===')
    for metric in ['F1best','F1@0.1']:
        delta = per_cond[f'full17_{metric}'] - per_cond[f'tlod_{metric}']
        print(metric, 'median', round(delta.median(),4), 'full17 better in', int((delta > 0).sum()), 'of', len(delta))

    # bootstrap CI for PR-AUC overall
    boot = []
    for m in ['full17','tlod_dp','tlod','logit2']:
        lo, hi = bootstrap_ci(y, d[f'{m}_pred'].values)
        boot.append({'Model': m, 'PR_AUC': average_precision_score(y, d[f'{m}_pred'].values),
                     'PR_AUC_CI_low': lo, 'PR_AUC_CI_high': hi})
    pd.DataFrame(boot).to_csv(f'{OUT}/tables/A2_bootstrap_deltas.csv', index=False)
    print('\n=== PR-AUC bootstrap CI ===')
    print(pd.DataFrame(boot).round(4).to_string(index=False))

    # figures
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(4.2, 3.0))
        for m, style in [('full17','-o'), ('tlod_dp','-s'), ('tlod','-^'), ('logit2','-d'), ('released','--')]:
            sub = per_cond.groupby('Reads')[f'{m}_F1best'].mean()
            ax.plot(sub.index, sub.values, style, ms=3, lw=1.2, label=LABEL[m])
        ax.set_xlabel('Tumour reads (M)'); ax.set_ylabel('Best F1 on test set')
        ax.legend(fontsize=5, frameon=False); ax.grid(alpha=.25)
        fig.tight_layout(); fig.savefig(f'{OUT}/figures/A2_f1_by_depth.png', dpi=300)
        fig.savefig(f'{OUT}/figures/A2_f1_by_depth.pdf')
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(4.2, 3.0))
        for m in ['full17','tlod_dp','tlod','logit2']:
            p = d[f'{m}_pred'].values
            order = np.argsort(-p); yy = y[order]
            tp = np.cumsum(yy); fp = np.cumsum(1 - yy)
            rec = tp / yy.sum(); prec = tp / np.maximum(tp + fp, 1)
            k = np.linspace(0, len(p)-1, 400).astype(int)
            ax.plot(rec[k], prec[k], lw=1.2, label=LABEL[m])
        ax.set_xlabel('Recall'); ax.set_ylabel('Precision'); ax.set_xlim(0.8, 1.0)
        ax.legend(fontsize=5, frameon=False); ax.grid(alpha=.25)
        fig.tight_layout(); fig.savefig(f'{OUT}/figures/A2_pr_curves.png', dpi=300)
        fig.savefig(f'{OUT}/figures/A2_pr_curves.pdf')
        print('wrote A2 figures')
    except Exception as e:
        print('figure skipped:', e)

    print('\nwrote summary tables to', f'{OUT}/tables')


if __name__ == '__main__':
    main()
