#!/usr/bin/env python3
"""C3. Collinearity among the 17 SomaticXGB features.

Reviewer 4, minor comment 1: the paper lists the 17 features but does not
discuss whether they are collinear.

Computes, on the training matrix (dtm_0819.txt, 493,937 rows):
  - Pearson and Spearman correlation matrices;
  - variance inflation factors (VIF = 1/(1-R^2) from regressing each feature
    on the other 16);
  - all pairs with |Pearson r| >= 0.7;
  - the condition number of the standardized design matrix;
  - a descriptive by-platform table (mean of standardized feature) and the
    fraction of feature variance attributable to platform (eta^2), to check
    whether any feature acts as a platform proxy.

Outputs:
  C3_correlation_pearson.csv
  C3_correlation_spearman.csv
  C3_vif.csv
  C3_high_correlation_pairs.csv
  C3_platform_means.csv
  C3_platform_eta2.csv
  C3_correlation_heatmap.png / .pdf
"""

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
import common  # noqa: E402
import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from scipy import stats

BASE = str(common.TRAIN_MATRIX)
OUT = str(common.RESULTS_ROOT)
FEATURES = ['TLOD','DP','ECNT','MPOS','NALOD','NLOD','MBQ_REF','MBQ_ALT',
            'MFRL_REF','MFRL_ALT','MMQ_REF','MMQ_ALT','avg_mapping_quality',
            'avg_basequality','avg_pos_as_fraction',
            'avg_num_mismatches_as_fraction','avg_sum_mismatch_qualities']
PLAT = {'FASTASeq_300':'FASTASeq 300','GenoLab_M':'GenoLab M',
        'NovaSeq_6000':'NovaSeq 6000','NovaSeq_X':'NovaSeq X',
        'SURFSeq_5000':'SURFSeq 5000','MGISEQ-T7':'DNBSEQ-T7'}


def main():
    os.makedirs(f'{OUT}/tables', exist_ok=True)
    os.makedirs(f'{OUT}/figures', exist_ok=True)
    print('loading', BASE)
    tr = pd.read_csv(BASE, sep='\t', low_memory=False)
    tr['plat'] = tr['plat'].replace(PLAT)
    X = tr[FEATURES].astype(np.float64).replace([np.inf, -np.inf], np.nan)
    print('rows', len(X), 'missing per feature:', X.isna().sum().to_dict())

    # ---- correlations (pairwise complete observations) -------------------
    pear = X.corr(method='pearson')
    spear = X.corr(method='spearman')
    pear.to_csv(f'{OUT}/tables/C3_correlation_pearson.csv')
    spear.to_csv(f'{OUT}/tables/C3_correlation_spearman.csv')

    rows = []
    for i, a in enumerate(FEATURES):
        for j, b in enumerate(FEATURES):
            if j <= i:
                continue
            rows.append({'Feature_1': a, 'Feature_2': b,
                         'Pearson_r': pear.loc[a, b], 'Spearman_rho': spear.loc[a, b]})
    pairs = pd.DataFrame(rows)
    pairs['AbsPearson'] = pairs['Pearson_r'].abs()
    pairs.sort_values('AbsPearson', ascending=False).to_csv(
        f'{OUT}/tables/C3_high_correlation_pairs.csv', index=False)
    hi = pairs[(pairs.AbsPearson >= 0.7) | (pairs.Spearman_rho.abs() >= 0.7)]
    print(f'\n{len(hi)} pairs with |r| or |rho| >= 0.7')
    print(hi.sort_values('AbsPearson', ascending=False).round(3).to_string(index=False))

    # ---- VIF -------------------------------------------------------------
    # Standardize, then regress each feature on the other 16. Complete-case
    # rows are used; the training matrix has only a small number of NaNs.
    Xc = X.dropna()
    print('complete-case rows for VIF:', len(Xc))
    Z = (Xc - Xc.mean()) / Xc.std(ddof=0)
    Zv = Z.values
    vif = []
    for i, f in enumerate(FEATURES):
        y = Zv[:, i]
        Xo = np.delete(Zv, i, axis=1)
        lr = LinearRegression(n_jobs=8).fit(Xo, y)
        r2 = lr.score(Xo, y)
        vif.append({'Feature': f, 'R2_vs_others': r2,
                    'VIF': 1.0 / max(1e-12, 1.0 - r2)})
    vif = pd.DataFrame(vif).sort_values('VIF', ascending=False)
    vif.to_csv(f'{OUT}/tables/C3_vif.csv', index=False)
    print('\nVIF:')
    print(vif.round(3).to_string(index=False))

    # condition number of standardized matrix
    sv = np.linalg.svd(Zv[: min(200000, len(Zv))], compute_uv=False)
    cond = float(sv[0] / sv[-1])
    print('\ncondition number (first 200k complete rows):', cond)

    # ---- platform association (descriptive) ------------------------------
    # eta^2 from one-way ANOVA of each standardized feature across platforms.
    plat_rows = []
    eta_rows = []
    means = tr.groupby('plat')[FEATURES].mean()
    sds = tr[FEATURES].std()
    means_z = (means - tr[FEATURES].mean()) / sds
    means_z.to_csv(f'{OUT}/tables/C3_platform_means.csv')
    for f in FEATURES:
        sub = tr[['plat', f]].dropna()
        groups = [g[f].values for _, g in sub.groupby('plat')]
        # one-way ANOVA; eta^2 = SS_between / SS_total
        grand = sub[f].mean()
        ssb = sum(len(g) * (g.mean() - grand) ** 2 for g in groups)
        sst = ((sub[f] - grand) ** 2).sum()
        eta2 = ssb / sst if sst else np.nan
        H, p = stats.kruskal(*groups)
        eta_rows.append({'Feature': f, 'eta2_platform': eta2,
                         'Kruskal_H': H, 'p': p})
    eta = pd.DataFrame(eta_rows).sort_values('eta2_platform', ascending=False)
    eta.to_csv(f'{OUT}/tables/C3_platform_eta2.csv', index=False)
    print('\nplatform eta^2:')
    print(eta.round(4).to_string(index=False))

    # ---- heatmap ---------------------------------------------------------
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6.6, 5.8))
        im = ax.imshow(pear.values, cmap='RdBu_r', vmin=-1, vmax=1)
        ax.set_xticks(range(len(FEATURES)))
        ax.set_yticks(range(len(FEATURES)))
        ax.set_xticklabels(FEATURES, rotation=90, fontsize=6)
        ax.set_yticklabels(FEATURES, fontsize=6)
        for i in range(len(FEATURES)):
            for j in range(len(FEATURES)):
                v = pear.values[i, j]
                if i != j and abs(v) >= 0.7:
                    ax.text(j, i, f'{v:.2f}', ha='center', va='center', fontsize=4)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='Pearson r')
        fig.tight_layout()
        fig.savefig(f'{OUT}/figures/C3_correlation_heatmap.png', dpi=300)
        fig.savefig(f'{OUT}/figures/C3_correlation_heatmap.pdf')
        print('wrote heatmap')
    except Exception as e:
        print('heatmap skipped:', e)


if __name__ == '__main__':
    main()
