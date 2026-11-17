#!/usr/bin/env python3
"""
R24: Re-quantify training/test target sharing and compare the two test
directions (group 45 = HG004 donor into HG005 background; group 54 = HG005
donor into HG004 background), including shared-vs-novel detection and a
novel-only direction comparison.

Inputs
------
Training targets:
  $CPSNV_DATA_ROOT/posfiles/train/{12,13,21,23,31,32}/only{g}.txt
Test targets:
  workflow/pos_files/test/{45,54}/pf{g}_low.txt
Predictions:
  response/addendum-analyses/tables/A2_predictions_test.csv.gz

Outputs (all prefixed R24_):
  tables/R24_*.csv (plus R24_target_level_by_condition.csv.gz)
  figures/R24_f1_by_group.png/.pdf
  figures/R24_shared_vs_novel_recall.png/.pdf
  figures/R24_novel_only_recall.png/.pdf
  logs/R24.log

Notes
-----
* Alleles are CHROM:POS:REF:ALT after stripping a leading "chr" and uppercasing.
* Site-level F1 follows scripts/make_fig3_data.py: truth = pf variants with
  distance >= 50 bp to the nearest other pf variant; calls = candidates with
  score above the threshold, with candidates at the <50 bp "near" variants
  removed.
* Target-level detection (shared vs novel) uses all pf targets; a target not
  present in the candidate table is counted as not detected, as requested.
* Conditions are not independent (6 platforms x 5 depths, same targets);
  paired summaries are descriptive. Bootstrap intervals are condition-level
  and are labelled as descriptive.
"""

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
import common  # noqa: E402

import os
import sys
import glob
import time
import platform as _platform
import warnings

import numpy as np
import pandas as pd
import scipy
from scipy import stats
from sklearn.metrics import roc_auc_score, average_precision_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

OUT = str(common.RESULTS_ROOT)
TRAIN_BASE = str(common.TRAIN_ONLY_DIR)
TEST_BASE = str(common.TEST_ROOT)
PRED_PATH = f'{OUT}/tables/A2_predictions_test.csv.gz'

TRAIN_GROUPS = ['12', '13', '21', '23', '31', '32']
TEST_GROUPS = ['45', '54']
MODELS = ['full17', 'released']
THRESHOLDS = [0.1, 0.6]
GROUP_LABEL = {'45': '45 (HG004 donor / HG005 background)',
               '54': '54 (HG005 donor / HG004 background)'}
SHORT_LABEL = {'45': '45: HG004 into HG005', '54': '54: HG005 into HG004'}
# Region spanning the dense chr17:41.19-41.30 Mb target cluster (BRCA1).
# Used only as a sensitivity analysis; the primary analysis uses all targets.
HARD_REGION = ('17', 41190000, 41300000)
HARD_REGION_LABEL = 'chr17:41.19-41.30Mb'
AF_CALIPER = 0.001
N_BOOT = 2000
RNG_SEED = 42

os.makedirs(f'{OUT}/tables', exist_ok=True)
os.makedirs(f'{OUT}/figures', exist_ok=True)
os.makedirs(f'{OUT}/logs', exist_ok=True)

LOG_PATH = f'{OUT}/logs/R24.log'
_log_handle = open(LOG_PATH, 'w')


def log(*args):
    msg = ' '.join(str(a) for a in args)
    print(msg, flush=True)
    _log_handle.write(msg + '\n')
    _log_handle.flush()


def log_df(title, df, **kwargs):
    log('\n' + title)
    log(df.to_string(**kwargs))


def norm_chrom(s):
    return (s.astype(str)
            .str.replace(r'^chr', '', regex=True)
            .str.replace(r'^CHR', '', regex=True)
            .str.upper())


def call_mask(score, threshold):
    """Use > 0.1 as requested; >= 0.6 as requested."""
    return np.greater(score, threshold) if threshold == 0.1 else np.greater_equal(score, threshold)


def add_dist(df):
    """Copy of the distance rule in scripts/make_fig3_data.py."""
    df = df.copy()
    df['CHROM'] = df['CHROM'].astype(str)
    df['POS'] = pd.to_numeric(df['POS'])
    d = []
    for chrom in df['CHROM'].unique():
        sub = df[df['CHROM'] == chrom]
        pos = sub['POS'].values
        n = len(pos)
        if n == 1:
            dd = [10000]
        elif n == 2:
            dd = [pos[1] - pos[0]] * 2
        else:
            d1 = np.r_[10000, pos[1:] - pos[:-1]]
            d2 = np.r_[pos[1:] - pos[:-1], 10000]
            dd = np.minimum(d1, d2)
        d.extend(dd)
    df['dist'] = d
    return df


def boot_ci(x, stat=np.median, n=N_BOOT, seed=RNG_SEED):
    """Percentile bootstrap CI over the supplied condition-level values."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    vals = np.empty(n, dtype=float)
    for i in range(n):
        b = rng.choice(x, size=len(x), replace=True)
        vals[i] = stat(b)
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


# ---------------------------------------------------------------------------
# 1. Load training and test targets
# ---------------------------------------------------------------------------
def load_training():
    frames = []
    per_file = []
    for g in TRAIN_GROUPS:
        path = f'{TRAIN_BASE}/{g}/only{g}.txt'
        d = pd.read_csv(path, sep='\t', dtype={'CHROM': str})
        d['CHROM'] = norm_chrom(d['CHROM'])
        d['REF'] = d['REF'].astype(str).str.upper()
        d['ALT'] = d['ALT'].astype(str).str.upper()
        d['POS'] = pd.to_numeric(d['POS'], errors='coerce')
        d = d.dropna(subset=['POS']).copy()
        d['POS'] = d['POS'].astype(int)
        d['allele'] = d['CHROM'] + ':' + d['POS'].astype(str) + ':' + d['REF'] + ':' + d['ALT']
        d['locus'] = d['CHROM'] + ':' + d['POS'].astype(str)
        d['training_file'] = g
        frames.append(d[['training_file', 'CHROM', 'POS', 'REF', 'ALT', 'allele', 'locus']])
        per_file.append({
            'training_file': g,
            'rows': len(d),
            'distinct_alleles': d['allele'].nunique(),
            'distinct_loci': d['locus'].nunique(),
            'duplicate_allele_rows': int(d['allele'].duplicated().sum()),
        })
    tr = pd.concat(frames, ignore_index=True)
    per_file = pd.DataFrame(per_file)
    return tr, set(tr['allele']), set(tr['locus']), per_file


def load_test_targets():
    frames = {}
    for g in TEST_GROUPS:
        path = f'{common.REPO_ROOT}/workflow/pos_files/test/{g}/pf{g}_low.txt'
        d = pd.read_csv(path, sep='\t', header=None,
                        names=['CHROM', 'POS', 'P', 'REF', 'ALT', 'GT', 'f', 'b', 'GT2', 'idx', 'AF'],
                        dtype={'CHROM': str})
        d['CHROM'] = norm_chrom(d['CHROM'])
        d['REF'] = d['REF'].astype(str).str.upper()
        d['ALT'] = d['ALT'].astype(str).str.upper()
        d['POS'] = pd.to_numeric(d['POS'], errors='coerce')
        d = d.dropna(subset=['POS']).copy()
        d['POS'] = d['POS'].astype(int)
        d = add_dist(d)
        d['allele'] = d['CHROM'] + ':' + d['POS'].astype(str) + ':' + d['REF'] + ':' + d['ALT']
        d['locus'] = d['CHROM'] + ':' + d['POS'].astype(str)
        d['var'] = d['allele']  # alias, matches make_fig3_data.py naming
        d['group'] = g
        frames[g] = d
    return frames


def annotate_targets(df, train_alleles, train_loci):
    d = df.copy()
    d['exact_shared'] = d['allele'].isin(train_alleles)
    d['locus_shared'] = d['locus'].isin(train_loci)
    d['category'] = np.select(
        [d['exact_shared'], d['locus_shared']],
        ['exact', 'locus_only'],
        default='novel')
    d['any_shared'] = d['category'].isin(['exact', 'locus_only'])
    d['gt_class'] = np.where(d['GT'].astype(str).str.contains(r'1[/|]1', regex=True), 'hom', 'het')
    d['substitution'] = d['REF'] + '>' + d['ALT']
    d['hard_region'] = ((d['CHROM'] == HARD_REGION[0]) &
                        (d['POS'] >= HARD_REGION[1]) &
                        (d['POS'] <= HARD_REGION[2]))
    return d


def training_overlap_matrix(train_rows):
    """Distinct alleles shared between each pair of training target files."""
    sets = {}
    for g, sub in train_rows.groupby('training_file'):
        sets[g] = set(sub['allele'])
    rows = []
    for a in TRAIN_GROUPS:
        for b in TRAIN_GROUPS:
            rows.append({'file_a': a, 'file_b': b, 'shared_alleles': len(sets[a] & sets[b]),
                         'jaccard': len(sets[a] & sets[b]) / len(sets[a] | sets[b])})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 2. Metrics
# ---------------------------------------------------------------------------
def best_f1_candidate(y, p):
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    order = np.argsort(-p)
    pp = p[order]
    yy = y[order]
    tp = np.cumsum(yy)
    fp = np.cumsum(1 - yy)
    fn = yy.sum() - tp
    f1 = 2 * tp / np.maximum(2 * tp + fp + fn, 1)
    i = int(np.argmax(f1))
    thr = pp[i] if i == len(pp) - 1 else (pp[i] + pp[i + 1]) / 2
    return float(f1[i]), float(thr)


def candidate_metrics(df, model):
    y = df['label'].astype(int).values
    p = df[f'{model}_pred'].values
    out = {'N': len(df), 'NPos': int(y.sum()), 'AUC': roc_auc_score(y, p),
           'PR_AUC': average_precision_score(y, p)}
    for th in THRESHOLDS:
        c = call_mask(p, th)
        tp = int((c & (y == 1)).sum())
        fp = int((c & (y == 0)).sum())
        fn = int(((~c) & (y == 1)).sum())
        out[f'F1@{th}'] = 2 * tp / max(2 * tp + fp + fn, 1)
        out[f'Precision@{th}'] = tp / max(tp + fp, 1)
        out[f'Recall@{th}'] = tp / max(tp + fn, 1)
        out[f'NCalls@{th}'] = tp + fp
        out[f'TP@{th}'] = tp
        out[f'FP@{th}'] = fp
        out[f'FN@{th}'] = fn
    bf, bt = best_f1_candidate(y, p)
    c = p > bt
    tp = int((c & (y == 1)).sum())
    fp = int((c & (y == 0)).sum())
    fn = int(((~c) & (y == 1)).sum())
    out['F1_best'] = bf
    out['Threshold_best'] = bt
    out['Precision_best'] = tp / max(tp + fp, 1)
    out['Recall_best'] = tp / max(tp + fn, 1)
    out['NCalls_best'] = tp + fp
    out['TP_best'] = tp
    out['FP_best'] = fp
    out['FN_best'] = fn
    return out


def site_metrics(df, model, threshold, truth, near):
    p = df[f'{model}_pred'].values
    var = df['var'].values
    if threshold == 'best':
        # Select the threshold after removing calls at the <50 bp near
        # variants, matching make_fig3_data.py.  Near calls contribute
        # neither TP nor FP.
        y = np.isin(var, list(truth)).astype(int)
        is_near = np.isin(var, list(near))
        total_truth = len(truth)
        order = np.argsort(-p)
        pp = p[order]
        yy = y[order] * (~is_near[order])
        ff = (1 - y[order]) * (~is_near[order])
        tp = np.cumsum(yy)
        fp = np.cumsum(ff)
        fn = total_truth - tp
        f1 = 2 * tp / np.maximum(2 * tp + fp + fn, 1)
        i = int(np.argmax(f1))
        thr = pp[i] if i == len(pp) - 1 else (pp[i] + pp[i + 1]) / 2
        calls = set(var[p > thr]) - near
    else:
        calls = set(var[call_mask(p, threshold)]) - near
    tp = len(calls & truth)
    fp = len(calls - truth)
    fn = len(truth - calls)
    return {
        'N_truth': len(truth),
        'TP': tp, 'FP': fp, 'FN': fn, 'NCalls': tp + fp,
        'F1': 2 * tp / max(2 * tp + fp + fn, 1),
        'Precision': tp / max(tp + fp, 1),
        'Recall': tp / max(tp + fn, 1),
        'Threshold_value': (thr if threshold == 'best' else threshold),
    }


# ---------------------------------------------------------------------------
# 3. Main
# ---------------------------------------------------------------------------
def main():
    t0 = time.time()
    log('=' * 78)
    log('R24: training/test target sharing and test-direction comparison')
    log('Started:', time.strftime('%Y-%m-%d %H:%M:%S'))
    log('Python:', sys.version.replace('\n', ' '))
    log('Platform:', _platform.platform())
    log('pandas', pd.__version__, 'numpy', np.__version__, 'scipy', scipy.__version__)
    log('Command:', ' '.join(sys.argv))
    log('Training target files:', f'{TRAIN_BASE}/{{12,13,21,23,31,32}}/only{{g}}.txt')
    log('Test target files:', f'{TEST_BASE}/{{45,54}}/pf{{g}}_low.txt')
    log('Predictions:', PRED_PATH)
    log('Output directory:', OUT)
    log('=' * 78)

    # ---- 1. training set -------------------------------------------------
    train_rows, train_alleles, train_loci, train_per_file = load_training()
    log_df('Training target files', train_per_file)
    log(f'\nTraining rows: {len(train_rows):,}; distinct alleles: {len(train_alleles):,}; '
        f'distinct loci: {len(train_loci):,}; duplicate allele rows across files: '
        f'{len(train_rows) - len(train_alleles):,}')
    train_per_file.to_csv(f'{OUT}/tables/R24_training_set_summary.csv', index=False)
    overlap = training_overlap_matrix(train_rows)
    overlap.to_csv(f'{OUT}/tables/R24_training_overlap_matrix.csv', index=False)
    log_df('Training-file pairwise overlap (distinct alleles)', overlap.pivot(index='file_a', columns='file_b', values='shared_alleles'))

    # ---- 2. test targets and sharing ------------------------------------
    pfs = load_test_targets()
    for g in TEST_GROUPS:
        pfs[g] = annotate_targets(pfs[g], train_alleles, train_loci)
        log(f'\nTest group {g}: {len(pfs[g])} targets; '
            f'exact shared {int(pfs[g]["exact_shared"].sum())}, '
            f'locus shared {int(pfs[g]["locus_shared"].sum())}, '
            f'category counts {pfs[g]["category"].value_counts().to_dict()}')
        log(f'  P range {pfs[g]["P"].min():.4f}-{pfs[g]["P"].max():.4f}; '
            f'AF range {pfs[g]["AF"].min():.4f}-{pfs[g]["AF"].max():.4f}; '
            f'duplicate alleles {int(pfs[g]["allele"].duplicated().sum())}; '
            f'duplicate loci {int(pfs[g]["locus"].duplicated().sum())}')

    share_rows = []
    for g in TEST_GROUPS:
        d = pfs[g]
        n = len(d)
        es = int(d['exact_shared'].sum())
        ls = int(d['locus_shared'].sum())
        lo = int((d['category'] == 'locus_only').sum())
        nv = int((d['category'] == 'novel').sum())
        share_rows.append({
            'group': g, 'n_targets': n, 'n_loci': d['locus'].nunique(),
            'exact_shared': es, 'exact_shared_pct': 100 * es / n,
            'locus_shared': ls, 'locus_shared_pct': 100 * ls / n,
            'locus_only': lo, 'novel': nv, 'novel_pct': 100 * nv / n,
        })
    all_d = pd.concat([pfs[g] for g in TEST_GROUPS], ignore_index=True)
    n_all = len(all_d)
    es_all = int(all_d['exact_shared'].sum())
    ls_all = int(all_d['locus_shared'].sum())
    share_rows.append({
        'group': 'all', 'n_targets': n_all, 'n_loci': all_d['locus'].nunique(),
        'exact_shared': es_all, 'exact_shared_pct': 100 * es_all / n_all,
        'locus_shared': ls_all, 'locus_shared_pct': 100 * ls_all / n_all,
        'locus_only': int((all_d['category'] == 'locus_only').sum()),
        'novel': int((all_d['category'] == 'novel').sum()),
        'novel_pct': 100 * int((all_d['category'] == 'novel').sum()) / n_all,
    })
    sharing = pd.DataFrame(share_rows)
    sharing.to_csv(f'{OUT}/tables/R24_sharing_summary.csv', index=False)
    log_df('Training/test target sharing', sharing.round(2))

    # explicit consistency check against the manuscript numbers
    paper = pd.DataFrame([
        {'group': '45', 'quantity': 'exact_shared', 'recomputed': 303, 'manuscript': 303, 'denominator': 481},
        {'group': '45', 'quantity': 'locus_shared', 'recomputed': 303, 'manuscript': 303, 'denominator': 481},
        {'group': '54', 'quantity': 'exact_shared', 'recomputed': 208, 'manuscript': 208, 'denominator': 479},
        {'group': '54', 'quantity': 'locus_shared', 'recomputed': 210, 'manuscript': np.nan, 'denominator': 479},
        {'group': 'all', 'quantity': 'exact_shared', 'recomputed': 511, 'manuscript': 511, 'denominator': 960},
        {'group': 'all', 'quantity': 'locus_shared', 'recomputed': 513, 'manuscript': 513, 'denominator': 960},
    ])
    # fill recomputed from data, to make the check data-driven
    def get_share(g, col):
        if g == 'all':
            return int((all_d[col]).sum())
        return int(pfs[g][col].sum())
    paper['recomputed'] = [get_share(r['group'], 'exact_shared' if r['quantity'] == 'exact_shared' else 'locus_shared')
                           for _, r in paper.iterrows()]
    paper['pct'] = 100 * paper['recomputed'] / paper['denominator']
    paper['match'] = np.where(paper['manuscript'].isna(), 'not_reported', np.where(paper['recomputed'] == paper['manuscript'], 'match', 'mismatch'))
    paper.to_csv(f'{OUT}/tables/R24_sharing_consistency_with_manuscript.csv', index=False)
    log_df('Consistency with manuscript sharing numbers', paper)

    # ---- 3. predictions ------------------------------------------------
    pred = pd.read_csv(PRED_PATH)
    pred['group'] = pred['group'].astype(int)
    pred['reads'] = pred['reads'].astype(int)
    log(f'\nPredictions: {len(pred):,} candidate rows; '
        f'{pred.groupby(["plat", "reads", "group"]).ngroups} conditions')
    cond_counts = pred.groupby(['plat', 'reads', 'group']).agg(
        N=('var', 'size'), NPos=('label', 'sum')).reset_index()
    log_df('Conditions in A2 predictions (first 10 rows)', cond_counts.head(10))
    if pred.groupby(['plat', 'reads', 'group']).ngroups != 60:
        log('WARNING: expected 60 conditions, found',
            pred.groupby(['plat', 'reads', 'group']).ngroups)

    # verify candidate-level F1@0.1 against the existing A2 table
    try:
        a2 = pd.read_csv(f'{OUT}/tables/A2_model_metrics_by_group.csv')
        check_rows = []
        for g in [45, 54]:
            sub = pred[pred['group'] == g]
            for model in MODELS:
                cm = candidate_metrics(sub, model)
                ref = a2[(a2['group'] == g) & (a2['Model'] == model)].iloc[0]
                check_rows.append({'group': g, 'model': model,
                                   'recomputed_F1@0.1': cm['F1@0.1'],
                                   'A2_table_F1@0.1': ref['F1@0.1'],
                                   'recomputed_AUC': cm['AUC'], 'A2_table_AUC': ref['AUC'],
                                   'recomputed_PR_AUC': cm['PR_AUC'], 'A2_table_PR_AUC': ref['PR_AUC']})
        check = pd.DataFrame(check_rows)
        check['max_abs_delta'] = check[['recomputed_F1@0.1', 'recomputed_AUC', 'recomputed_PR_AUC']].sub(
            check[['A2_table_F1@0.1', 'A2_table_AUC', 'A2_table_PR_AUC']].values).abs().max(axis=1)
        log_df('Candidate-level metric check against A2_model_metrics_by_group.csv', check.round(6))
    except Exception as e:
        log('Could not check against A2_model_metrics_by_group.csv:', e)
        check = pd.DataFrame()

    # ---- 4. performance by condition -----------------------------------
    perf_rows = []
    site_sens_rows = []
    for (plat, reads, group), sub in pred.groupby(['plat', 'reads', 'group']):
        g = str(group)
        pf = pfs[g]
        truth = set(pf.loc[pf['dist'] >= 50, 'var'])
        near = set(pf.loc[pf['dist'] < 50, 'var'])
        # zero-candidate targets for this group (absent in all 30 conditions)
        # computed later; for now use all truth for site metrics
        for model in MODELS:
            cm = candidate_metrics(sub, model)
            for th in THRESHOLDS:
                perf_rows.append({
                    'group': g, 'platform': plat, 'reads': reads,
                    'model': model, 'level': 'candidate', 'threshold': th,
                    'N': cm['N'], 'NPos': cm['NPos'], 'N_truth': np.nan,
                    'AUC': cm['AUC'], 'PR_AUC': cm['PR_AUC'],
                    'F1': cm[f'F1@{th}'], 'Precision': cm[f'Precision@{th}'],
                    'Recall': cm[f'Recall@{th}'], 'NCalls': cm[f'NCalls@{th}'],
                    'TP': cm[f'TP@{th}'], 'FP': cm[f'FP@{th}'], 'FN': cm[f'FN@{th}'],
                    'Threshold_value': th,
                })
            perf_rows.append({
                'group': g, 'platform': plat, 'reads': reads,
                'model': model, 'level': 'candidate', 'threshold': 'best',
                'N': cm['N'], 'NPos': cm['NPos'], 'N_truth': np.nan,
                'AUC': cm['AUC'], 'PR_AUC': cm['PR_AUC'],
                'F1': cm['F1_best'], 'Precision': cm['Precision_best'],
                'Recall': cm['Recall_best'], 'NCalls': cm['NCalls_best'],
                'TP': cm['TP_best'], 'FP': cm['FP_best'], 'FN': cm['FN_best'],
                'Threshold_value': cm['Threshold_best'],
            })
            for th in THRESHOLDS + ['best']:
                sm = site_metrics(sub, model, th, truth, near)
                perf_rows.append({
                    'group': g, 'platform': plat, 'reads': reads,
                    'model': model, 'level': 'site', 'threshold': th,
                    'N': len(sub), 'NPos': int(sub['label'].sum()), 'N_truth': sm['N_truth'],
                    'AUC': np.nan, 'PR_AUC': np.nan,
                    'F1': sm['F1'], 'Precision': sm['Precision'], 'Recall': sm['Recall'],
                    'NCalls': sm['NCalls'], 'TP': sm['TP'], 'FP': sm['FP'], 'FN': sm['FN'],
                    'Threshold_value': sm['Threshold_value'],
                })
    perf = pd.DataFrame(perf_rows)
    perf.to_csv(f'{OUT}/tables/R24_performance_by_group.csv', index=False)

    # Validate the site-level rule against the published fig3a source data
    # (released SomaticXGB at 100M, threshold 0.1).  This is a QC check only.
    try:
        src_path = str(common.TABLES / 'fig3a_test_100M_methods.csv')
        src = pd.read_csv(src_path)
        src = src[src['Method'] == 'SomaticXGB 0.1'].copy()
        src['Platform'] = src['Platform'].str.replace('_', ' ', regex=False)
        src['Group'] = src['Group'].astype(str)
        ours = perf[(perf['level'] == 'site') & (perf['model'] == 'released') &
                    (perf['threshold'] == 0.1) & (perf['reads'] == 100)].copy()
        ours['group'] = ours['group'].astype(str)
        val = src.merge(ours, left_on=['Group', 'Platform'],
                        right_on=['group', 'platform'], suffixes=('_src', '_ours'))
        val['F1_abs_delta'] = (val['F1_src'] - val['F1_ours']).abs()
        val['Recall_abs_delta'] = (val['Recall_src'] - val['Recall_ours']).abs()
        val['Precision_abs_delta'] = (val['Precision_src'] - val['Precision_ours']).abs()
        val[['Group', 'Platform', 'F1_src', 'F1_ours', 'Recall_src', 'Recall_ours',
             'Precision_src', 'Precision_ours', 'TP_src', 'TP_ours', 'F1_abs_delta',
             'Recall_abs_delta', 'Precision_abs_delta']] \
            .to_csv(f'{OUT}/tables/R24_validation_vs_fig3a.csv', index=False)
        log(f'\nValidation against fig3a_test_100M_methods.csv (released model, 0.1): '
            f'n={len(val)}, max |dF1|={val["F1_abs_delta"].max():.2e}, '
            f'max |dRecall|={val["Recall_abs_delta"].max():.2e}, '
            f'max |dPrecision|={val["Precision_abs_delta"].max():.2e}')
        if len(val):
            log_df('Validation rows', val[['Group', 'Platform', 'F1_src', 'F1_ours',
                                           'Recall_src', 'Recall_ours', 'Precision_src',
                                           'Precision_ours']].round(5), index=False)
    except Exception as e:
        log('Could not validate against fig3a source data:', e)

    log_df('Performance by group (mean across 30 conditions)',
           perf.groupby(['group', 'level', 'model', 'threshold'])[['F1', 'Recall', 'Precision']].mean().round(4))

    # group-level pooled candidate metrics (A2 style) + mean across conditions
    group_summary_rows = []
    for g in ['45', '54']:
        sub = pred[pred['group'] == int(g)]
        for model in MODELS:
            cm = candidate_metrics(sub, model)
            row = {'group': g, 'model': model, 'aggregation': 'pooled_candidates',
                   'N': cm['N'], 'NPos': cm['NPos'], 'AUC': cm['AUC'], 'PR_AUC': cm['PR_AUC'],
                   'F1@0.1': cm['F1@0.1'], 'Precision@0.1': cm['Precision@0.1'],
                   'Recall@0.1': cm['Recall@0.1'],
                   'F1@0.6': cm['F1@0.6'], 'Precision@0.6': cm['Precision@0.6'],
                   'Recall@0.6': cm['Recall@0.6'],
                   'F1': cm['F1_best'], 'Precision': cm['Precision_best'],
                   'Recall': cm['Recall_best']}
            group_summary_rows.append(row)
        for level in ['candidate', 'site']:
            for model in MODELS:
                for th in THRESHOLDS + ['best']:
                    s = perf[(perf['group'] == g) & (perf['level'] == level) &
                             (perf['model'] == model) & (perf['threshold'] == th)]
                    group_summary_rows.append({
                        'group': g, 'model': model,
                        'aggregation': f'mean_conditions_{level}_{th}',
                        'N': s['N'].mean(), 'NPos': s['NPos'].mean(),
                        'AUC': s['AUC'].mean(), 'PR_AUC': s['PR_AUC'].mean(),
                        'F1@0.1': s['F1'].mean() if th == 0.1 else np.nan,
                        'Precision@0.1': s['Precision'].mean() if th == 0.1 else np.nan,
                        'Recall@0.1': s['Recall'].mean() if th == 0.1 else np.nan,
                        'F1': s['F1'].mean(), 'Precision': s['Precision'].mean(),
                        'Recall': s['Recall'].mean(),
                    })
    group_summary = pd.DataFrame(group_summary_rows)
    group_summary.to_csv(f'{OUT}/tables/R24_performance_group_summary.csv', index=False)

    # ---- 5. target-level detection table -------------------------------
    target_records = []
    for (plat, reads, group), sub in pred.groupby(['plat', 'reads', 'group']):
        g = str(group)
        pf = pfs[g]
        base = pf[['allele', 'category', 'P', 'AF', 'gt_class', 'substitution',
                   'hard_region', 'dist']].copy()
        for model in MODELS:
            pmap = pd.Series(sub[f'{model}_pred'].values, index=sub['var'].values)
            base[f'{model}_pred'] = base['allele'].map(pmap)
            base[f'{model}_present'] = base[f'{model}_pred'].notna()
            for th in THRESHOLDS:
                base[f'{model}_det_{th}'] = call_mask(base[f'{model}_pred'].values, th)
        for model in MODELS:
            for th in THRESHOLDS:
                tmp = base[['allele', 'category', 'P', 'AF', 'gt_class', 'substitution',
                            'hard_region', 'dist']].copy()
                tmp['platform'] = plat
                tmp['reads'] = reads
                tmp['group'] = g
                tmp['model'] = model
                tmp['threshold'] = th
                tmp['pred'] = base[f'{model}_pred'].values
                tmp['present'] = base[f'{model}_present'].values
                tmp['detected'] = base[f'{model}_det_{th}'].values
                target_records.append(tmp)
    target_long = pd.concat(target_records, ignore_index=True)
    target_long.to_csv(f'{OUT}/tables/R24_target_level_by_condition.csv.gz',
                       index=False, compression='gzip')
    log(f'\nTarget-level table: {len(target_long):,} rows; '
        f'present fraction {target_long["present"].mean():.4f}; '
        f'detected fraction {target_long["detected"].mean():.4f}')

    # zero-candidate target variants
    zero_rows = []
    for g in TEST_GROUPS:
        sub = target_long[(target_long['group'] == g) & (target_long['model'] == 'full17') &
                          (target_long['threshold'] == 0.1)]
        pres = sub.groupby('allele')['present'].sum()
        det = sub.groupby('allele')['detected'].sum()
        for _, r in pfs[g].iterrows():
            if pres.get(r['allele'], 0) == 0:
                zero_rows.append({
                    'group': g, 'allele': r['allele'], 'category': r['category'],
                    'P': r['P'], 'AF': r['AF'], 'GT': r['GT'],
                    'n_conditions_candidate_present': int(pres.get(r['allele'], 0)),
                    'n_conditions_detected': int(det.get(r['allele'], 0)),
                    'hard_region': bool(r['hard_region']),
                })
    zero_df = pd.DataFrame(zero_rows)
    zero_df.to_csv(f'{OUT}/tables/R24_zero_candidate_variants.csv', index=False)
    log_df('Target variants absent from all 30 candidate tables', zero_df)

    # variant-level detection summary
    var_det = target_long[(target_long['model'] == 'full17') & (target_long['threshold'] == 0.1)] \
        .groupby(['group', 'allele', 'category', 'P', 'AF', 'gt_class', 'substitution', 'hard_region']) \
        .agg(n_conditions=('present', 'size'),
             n_present=('present', 'sum'),
             n_detected=('detected', 'sum')).reset_index()
    var_det['present_rate'] = var_det['n_present'] / var_det['n_conditions']
    var_det['recall_site'] = var_det['n_detected'] / var_det['n_conditions']
    var_det['recall_present'] = np.where(var_det['n_present'] > 0,
                                        var_det['n_detected'] / var_det['n_present'], np.nan)
    var_det.to_csv(f'{OUT}/tables/R24_target_variant_detection.csv', index=False)
    log_df('Per-variant detection (full17, 0.1), mean across variants',
           var_det.groupby(['group', 'category']).agg(
               n=('allele', 'size'), recall_site=('recall_site', 'mean'),
               recall_present=('recall_present', 'mean'),
               present_rate=('present_rate', 'mean')).round(4))

    # Candidate-level score distribution for target variants: collapse each
    # variant across the 30 conditions (median score among candidate-present
    # rows), then summarise by sharing category.
    score_rows = []
    for g in TEST_GROUPS:
        for model in MODELS:
            z = target_long[(target_long['group'] == g) & (target_long['model'] == model) &
                            (target_long['threshold'] == 0.1)].copy()
            v = z.groupby(['allele', 'category']).agg(
                pred_median=('pred', 'median'),
                pred_mean=('pred', 'mean'),
                present_any=('present', 'max'),
                frac_present=('present', 'mean'),
                detected_rate=('detected', 'mean')).reset_index()
            v = v[v['present_any']]
            for cat in ['exact', 'locus_only', 'novel', 'any_shared']:
                s = v[v['category'] == cat] if cat != 'any_shared' else v[v['category'] != 'novel']
                zs = z[z['present']]
                zs = zs[zs['category'] == cat] if cat != 'any_shared' else zs[zs['category'] != 'novel']
                if len(s) == 0:
                    continue
                score_rows.append({
                    'group': g, 'model': model, 'category': cat,
                    'n_variants': len(s),
                    'frac_rows_pred_gt_0.1': (zs['pred'] > 0.1).mean() if len(zs) else np.nan,
                    'frac_rows_pred_ge_0.6': (zs['pred'] >= 0.6).mean() if len(zs) else np.nan,
                    'pred_median': s['pred_median'].median(),
                    'pred_q1': s['pred_median'].quantile(0.25),
                    'pred_q3': s['pred_median'].quantile(0.75),
                    'pred_IQR': s['pred_median'].quantile(0.75) - s['pred_median'].quantile(0.25),
                    'pred_mean': s['pred_median'].mean(),
                    'frac_variants_median_pred_gt_0.1': (s['pred_median'] > 0.1).mean(),
                    'frac_variants_median_pred_ge_0.6': (s['pred_median'] >= 0.6).mean(),
                    'mean_frac_conditions_present': s['frac_present'].mean(),
                    'mean_detected_rate_across_conditions': s['detected_rate'].mean(),
                })
    score_df = pd.DataFrame(score_rows)
    score_df.to_csv(f'{OUT}/tables/R24_candidate_score_by_category.csv', index=False)
    log_df('Candidate-level target score distribution (per-variant median across conditions)',
           score_df[(score_df['model'] == 'full17')].round(4), index=False)

    # ---- 6. shared vs novel by condition ------------------------------
    def shared_vs_novel_by_condition(x, present_only=False, exclude_hard=False, setting='site_all'):
        z = x.copy()
        if present_only:
            z = z[z['present']]
        if exclude_hard:
            z = z[~z['hard_region']]
        rows = []
        for keys, sub in z.groupby(['platform', 'reads', 'group', 'model', 'threshold']):
            plat, reads, group, model, th = keys
            row = {'platform': plat, 'reads': reads, 'group': group,
                   'model': model, 'threshold': th, 'setting': setting}
            for cat in ['exact', 'locus_only', 'novel']:
                s = sub[sub['category'] == cat]
                row[f'n_{cat}'] = len(s)
                row[f'recall_{cat}'] = s['detected'].mean() if len(s) else np.nan
            s_any = sub[sub['category'] != 'novel']
            row['n_any_shared'] = len(s_any)
            row['recall_any_shared'] = s_any['detected'].mean() if len(s_any) else np.nan
            row['diff_exact_minus_novel'] = row['recall_exact'] - row['recall_novel']
            row['diff_any_shared_minus_novel'] = row['recall_any_shared'] - row['recall_novel']
            row['n_pairs_exact_novel'] = min(row['n_exact'], row['n_novel'])
            rows.append(row)
        return pd.DataFrame(rows)

    svn_site = shared_vs_novel_by_condition(target_long, setting='site_all')
    svn_site.to_csv(f'{OUT}/tables/R24_shared_vs_novel_recall_by_condition.csv', index=False)
    svn_present = shared_vs_novel_by_condition(target_long, present_only=True, setting='candidate_present')
    svn_nohard = shared_vs_novel_by_condition(target_long, exclude_hard=True, setting='site_no_hard_region')

    # ---- 7. AF matching and stratified analyses ------------------------
    def greedy_af_match(left, right, left_name='exact', right_name='novel', caliper=AF_CALIPER):
        l = left[['allele', 'AF']].reset_index(drop=True)
        r = right[['allele', 'AF']].reset_index(drop=True)
        if len(l) == 0 or len(r) == 0:
            return pd.DataFrame(columns=[left_name, right_name, 'af_diff'])
        diff = np.abs(l['AF'].values[:, None] - r['AF'].values[None, :])
        order = np.dstack(np.unravel_index(np.argsort(diff, axis=None), diff.shape))[0]
        used_l, used_r = set(), set()
        pairs = []
        for i, j in order:
            if diff[i, j] > caliper:
                break
            if i in used_l or j in used_r:
                continue
            pairs.append({left_name: l['allele'].iloc[i], right_name: r['allele'].iloc[j],
                          'af_diff': diff[i, j]})
            used_l.add(i)
            used_r.add(j)
        return pd.DataFrame(pairs)

    match_info = []
    svn_match_rows = []
    for g in TEST_GROUPS:
        pf = pfs[g]
        a = pf[pf['category'] == 'exact']
        b = pf[pf['category'] == 'novel']
        pairs = greedy_af_match(a, b, left_name='exact', right_name='novel')
        match_info.append({'group': g, 'n_exact': len(a), 'n_novel': len(b),
                           'n_matched': len(pairs), 'caliper': AF_CALIPER})
        if len(pairs) == 0:
            continue
        for model in MODELS:
            for th in THRESHOLDS:
                for present_only in [False, True]:
                    z = target_long[(target_long['group'] == g) &
                                    (target_long['model'] == model) &
                                    (target_long['threshold'] == th)].copy()
                    if present_only:
                        z = z[z['present']]
                    e = z.merge(pairs[['exact', 'novel']], left_on='allele', right_on='exact')
                    n = z.merge(pairs[['exact', 'novel']], left_on='allele', right_on='novel')
                    if len(e) == 0 or len(n) == 0:
                        continue
                    m = e.merge(n, on=['platform', 'reads', 'model', 'threshold', 'exact', 'novel'],
                                suffixes=('_ex', '_nv'))
                    m['diff'] = m['detected_ex'].astype(float) - m['detected_nv'].astype(float)
                    per_cond = m.groupby(['platform', 'reads']).agg(
                        diff=('diff', 'mean'), n_pairs=('diff', 'size')).reset_index()
                    for _, r in per_cond.iterrows():
                        svn_match_rows.append({
                            'group': g, 'model': model, 'threshold': th,
                            'setting': 'af_matched_present' if present_only else 'af_matched',
                            'platform': r['platform'], 'reads': r['reads'],
                            'diff': r['diff'], 'n_pairs': r['n_pairs'],
                        })
    match_info = pd.DataFrame(match_info)
    log_df('Within-group 1:1 AF matching (exact vs novel)', match_info)
    svn_match = pd.DataFrame(svn_match_rows)

    # AF/P quintiles within group
    svn_strat_rows = []
    for g in TEST_GROUPS:
        pf = pfs[g].copy()
        for qcol in ['P', 'AF']:
            pf[f'{qcol}_q'] = pd.qcut(pf[qcol], 5, labels=False, duplicates='drop')
            qmap = pf.set_index('allele')[f'{qcol}_q'].to_dict()
            z = target_long[target_long['group'] == g].copy()
            z[f'{qcol}_q'] = z['allele'].map(qmap)
            for present_only in [False, True]:
                zz = z[z['present']] if present_only else z
                for (model, th, q), sub in zz.groupby(['model', 'threshold', f'{qcol}_q']):
                    ex = sub[sub['category'] == 'exact']['detected']
                    nv = sub[sub['category'] == 'novel']['detected']
                    if len(ex) == 0 or len(nv) == 0:
                        continue
                    # per-condition difference within this quintile
                    per = sub[sub['category'].isin(['exact', 'novel'])].groupby(
                        ['platform', 'reads', 'category'])['detected'].mean().unstack()
                    if 'exact' not in per.columns or 'novel' not in per.columns:
                        continue
                    per = per.dropna(subset=['exact', 'novel'])
                    diff = per['exact'] - per['novel']
                    if len(diff) == 0:
                        continue
                    svn_strat_rows.append({
                        'group': g, 'model': model, 'threshold': th,
                        'stratifier': qcol, 'quintile': int(q),
                        'setting': 'candidate_present' if present_only else 'site_all',
                        'n_exact_variants': sub[sub['category'] == 'exact']['allele'].nunique(),
                        'n_novel_variants': sub[sub['category'] == 'novel']['allele'].nunique(),
                        'mean_recall_exact': ex.mean(), 'mean_recall_novel': nv.mean(),
                        'median_diff': diff.median(), 'mean_diff': diff.mean(),
                        'q1_diff': np.percentile(diff, 25), 'q3_diff': np.percentile(diff, 75),
                        'n_conditions': len(diff),
                        'n_pos': int((diff > 0).sum()), 'n_ties': int((diff == 0).sum()),
                        'n_neg': int((diff < 0).sum()),
                    })
    svn_strat = pd.DataFrame(svn_strat_rows)
    svn_strat.to_csv(f'{OUT}/tables/R24_vaf_stratified_recall.csv', index=False)
    log_df('VAF-stratified exact-vs-novel recall (full17, 0.1, site_all)',
           svn_strat[(svn_strat['model'] == 'full17') & (svn_strat['threshold'] == 0.1) &
                     (svn_strat['setting'] == 'site_all')]
           .round(3), index=False)

    # ---- 8. shared vs novel summary -----------------------------------
    def summarize_svn(df, diff_col='diff_exact_minus_novel', shared_def='exact'):
        rows = []
        for keys, sub in df.groupby(['setting', 'group', 'model', 'threshold']):
            setting, group, model, th = keys
            x = sub[diff_col].dropna().values
            if len(x) == 0:
                continue
            med_lo, med_hi = boot_ci(x, np.median)
            mean_lo, mean_hi = boot_ci(x, np.mean)
            rows.append({
                'shared_definition': shared_def, 'setting': setting,
                'group': group, 'model': model, 'threshold': th,
                'n_conditions': len(x),
                'mean_recall_shared': sub['recall_exact' if shared_def == 'exact' else 'recall_any_shared'].mean(),
                'mean_recall_novel': sub['recall_novel'].mean(),
                'median_diff': np.median(x), 'mean_diff': np.mean(x),
                'q1_diff': np.percentile(x, 25), 'q3_diff': np.percentile(x, 75),
                'n_shared_higher': int((x > 0).sum()),
                'n_ties': int((x == 0).sum()),
                'n_novel_higher': int((x < 0).sum()),
                'boot_median_ci_low': med_lo, 'boot_median_ci_high': med_hi,
                'boot_mean_ci_low': mean_lo, 'boot_mean_ci_high': mean_hi,
            })
        return pd.DataFrame(rows)

    svn_summary = pd.concat([
        summarize_svn(svn_site, 'diff_exact_minus_novel', 'exact'),
        summarize_svn(svn_nohard, 'diff_exact_minus_novel', 'exact'),
        summarize_svn(svn_present, 'diff_exact_minus_novel', 'exact'),
        summarize_svn(svn_site, 'diff_any_shared_minus_novel', 'any_shared'),
        summarize_svn(svn_present, 'diff_any_shared_minus_novel', 'any_shared'),
    ], ignore_index=True)
    # add AF-matched summaries
    if len(svn_match) > 0:
        for keys, sub in svn_match.groupby(['setting', 'group', 'model', 'threshold']):
            setting, group, model, th = keys
            x = sub['diff'].dropna().values
            med_lo, med_hi = boot_ci(x, np.median)
            mean_lo, mean_hi = boot_ci(x, np.mean)
            svn_summary = pd.concat([svn_summary, pd.DataFrame([{
                'shared_definition': 'exact', 'setting': setting,
                'group': group, 'model': model, 'threshold': th,
                'n_conditions': len(x),
                'mean_recall_shared': np.nan, 'mean_recall_novel': np.nan,
                'median_diff': np.median(x), 'mean_diff': np.mean(x),
                'q1_diff': np.percentile(x, 25), 'q3_diff': np.percentile(x, 75),
                'n_shared_higher': int((x > 0).sum()), 'n_ties': int((x == 0).sum()),
                'n_novel_higher': int((x < 0).sum()),
                'boot_median_ci_low': med_lo, 'boot_median_ci_high': med_hi,
                'boot_mean_ci_low': mean_lo, 'boot_mean_ci_high': mean_hi,
            }])], ignore_index=True)
    svn_summary.to_csv(f'{OUT}/tables/R24_shared_vs_novel_summary.csv', index=False)
    log_df('Shared-vs-novel recall summary (full17, 0.1)',
           svn_summary[(svn_summary['shared_definition'] == 'exact') &
                       (svn_summary['model'] == 'full17') &
                       (svn_summary['threshold'] == 0.1)]
           .round(4), index=False)

    # ---- 9. novel-only direction comparison ---------------------------
    novel_long = target_long[target_long['category'] == 'novel'].copy()
    novel_rows = []
    for (plat, reads, model, th), sub in novel_long.groupby(['platform', 'reads', 'model', 'threshold']):
        row = {'platform': plat, 'reads': reads, 'model': model, 'threshold': th}
        for g in TEST_GROUPS:
            s = sub[sub['group'] == g]
            row[f'n_novel_{g}'] = len(s)
            row[f'recall_{g}'] = s['detected'].mean()
            sp = s[s['present']]
            row[f'n_present_{g}'] = len(sp)
            row[f'recall_present_{g}'] = sp['detected'].mean() if len(sp) else np.nan
        row['diff_45_minus_54'] = row['recall_45'] - row['recall_54']
        row['diff_present_45_minus_54'] = row['recall_present_45'] - row['recall_present_54']
        novel_rows.append(row)
    novel_wide = pd.DataFrame(novel_rows)
    novel_wide.to_csv(f'{OUT}/tables/R24_novel_only_comparison.csv', index=False)
    log_df('Novel-only site-level recall (full17, 0.1)',
           novel_wide[(novel_wide['model'] == 'full17') & (novel_wide['threshold'] == 0.1)],
           index=False)

    # novel-only AF matching between groups
    a45 = pfs['45'][pfs['45']['category'] == 'novel'][['allele', 'AF']]
    a54 = pfs['54'][pfs['54']['category'] == 'novel'][['allele', 'AF']]
    novel_pairs = greedy_af_match(a45, a54, left_name='a45', right_name='a54')
    novel_match_rows = []
    for model in MODELS:
        for th in THRESHOLDS:
            for present_only in [False, True]:
                z = novel_long[(novel_long['model'] == model) & (novel_long['threshold'] == th)].copy()
                if present_only:
                    z = z[z['present']]
                x45 = z[z['group'] == '45'].merge(novel_pairs[['a45', 'a54']],
                                                  left_on='allele', right_on='a45')
                x54 = z[z['group'] == '54'].merge(novel_pairs[['a45', 'a54']],
                                                  left_on='allele', right_on='a54')
                if len(x45) == 0 or len(x54) == 0:
                    continue
                m = x45.merge(x54, on=['platform', 'reads', 'model', 'threshold', 'a45', 'a54'],
                              suffixes=('_45', '_54'))
                m['diff'] = m['detected_45'].astype(float) - m['detected_54'].astype(float)
                per = m.groupby(['platform', 'reads']).agg(diff=('diff', 'mean'),
                                                           n_pairs=('diff', 'size')).reset_index()
                for _, r in per.iterrows():
                    novel_match_rows.append({
                        'platform': r['platform'], 'reads': r['reads'], 'model': model,
                        'threshold': th,
                        'setting': 'af_matched_present' if present_only else 'af_matched',
                        'diff_45_minus_54': r['diff'], 'n_pairs': r['n_pairs'],
                    })
    novel_match = pd.DataFrame(novel_match_rows)
    log(f'\nNovel-only 1:1 AF matching: {len(novel_pairs)} pairs of '
        f'{len(a45)} group-45 and {len(a54)} group-54 novel targets (caliper {AF_CALIPER})')

    # novel-only global AF/P quintile comparison
    novel_q_rows = []
    nb = pd.concat([pfs['45'][pfs['45']['category'] == 'novel'][['allele', 'group', 'P', 'AF']],
                    pfs['54'][pfs['54']['category'] == 'novel'][['allele', 'group', 'P', 'AF']]],
                   ignore_index=True)
    for qcol in ['P', 'AF']:
        nb[f'{qcol}_q'] = pd.qcut(nb[qcol], 5, labels=False, duplicates='drop')
        qmap = nb.set_index('allele')[f'{qcol}_q'].to_dict()
        z = novel_long.copy()
        z[f'{qcol}_q'] = z['allele'].map(qmap)
        for (model, th, q), sub in z.groupby(['model', 'threshold', f'{qcol}_q']):
            per = sub.groupby(['platform', 'reads', 'group'])['detected'].mean().unstack()
            if '45' not in per.columns or '54' not in per.columns:
                continue
            per = per.dropna(subset=['45', '54'])
            diff = per['45'] - per['54']
            novel_q_rows.append({
                'stratifier': qcol, 'quintile': int(q), 'model': model, 'threshold': th,
                'n_45': int((nb[(nb['group'] == '45') & (nb[f'{qcol}_q'] == q)]).shape[0]),
                'n_54': int((nb[(nb['group'] == '54') & (nb[f'{qcol}_q'] == q)]).shape[0]),
                'mean_recall_45': per['45'].mean(), 'mean_recall_54': per['54'].mean(),
                'median_diff': diff.median(), 'mean_diff': diff.mean(),
                'n_conditions': len(diff),
                'n_45_higher': int((diff > 0).sum()),
                'n_ties': int((diff == 0).sum()),
                'n_54_higher': int((diff < 0).sum()),
            })
    novel_q = pd.DataFrame(novel_q_rows)
    novel_q.to_csv(f'{OUT}/tables/R24_novel_only_vaf_stratified.csv', index=False)

    # novel-only summary
    novel_summary_rows = []
    for (model, th), sub in novel_wide.groupby(['model', 'threshold']):
        for metric_col, setting in [('diff_45_minus_54', 'site_all'),
                                    ('diff_present_45_minus_54', 'candidate_present')]:
            x = sub[metric_col].dropna().values
            med_lo, med_hi = boot_ci(x, np.median)
            mean_lo, mean_hi = boot_ci(x, np.mean)
            if setting == 'candidate_present':
                mr45 = sub['recall_present_45'].mean()
                mr54 = sub['recall_present_54'].mean()
            else:
                mr45 = sub['recall_45'].mean()
                mr54 = sub['recall_54'].mean()
            novel_summary_rows.append({
                'setting': setting, 'model': model, 'threshold': th,
                'n_conditions': len(x),
                'mean_recall_45': mr45,
                'mean_recall_54': mr54,
                'median_diff_45_minus_54': np.median(x), 'mean_diff_45_minus_54': np.mean(x),
                'q1_diff': np.percentile(x, 25), 'q3_diff': np.percentile(x, 75),
                'n_45_higher': int((x > 0).sum()), 'n_ties': int((x == 0).sum()),
                'n_54_higher': int((x < 0).sum()),
                'boot_median_ci_low': med_lo, 'boot_median_ci_high': med_hi,
                'boot_mean_ci_low': mean_lo, 'boot_mean_ci_high': mean_hi,
            })
    if len(novel_match) > 0:
        for (setting, model, th), sub in novel_match.groupby(['setting', 'model', 'threshold']):
            x = sub['diff_45_minus_54'].dropna().values
            med_lo, med_hi = boot_ci(x, np.median)
            mean_lo, mean_hi = boot_ci(x, np.mean)
            novel_summary_rows.append({
                'setting': setting, 'model': model, 'threshold': th,
                'n_conditions': len(x),
                'mean_recall_45': np.nan, 'mean_recall_54': np.nan,
                'median_diff_45_minus_54': np.median(x), 'mean_diff_45_minus_54': np.mean(x),
                'q1_diff': np.percentile(x, 25), 'q3_diff': np.percentile(x, 75),
                'n_45_higher': int((x > 0).sum()), 'n_ties': int((x == 0).sum()),
                'n_54_higher': int((x < 0).sum()),
                'boot_median_ci_low': med_lo, 'boot_median_ci_high': med_hi,
                'boot_mean_ci_low': mean_lo, 'boot_mean_ci_high': mean_hi,
            })
    novel_summary = pd.DataFrame(novel_summary_rows)
    novel_summary.to_csv(f'{OUT}/tables/R24_novel_only_summary.csv', index=False)
    log_df('Novel-only direction comparison summary (full17, 0.1)',
           novel_summary[(novel_summary['model'] == 'full17') &
                         (novel_summary['threshold'] == 0.1)].round(4), index=False)

    # ---- 10. group paired performance comparison ----------------------
    def paired_stats(df, value_col, keys):
        a = df.pivot_table(index=['platform', 'reads'], columns='group', values=value_col)
        if '45' not in a.columns or '54' not in a.columns:
            return None
        d = (a['45'] - a['54']).dropna()
        if len(d) == 0:
            return None
        med_lo, med_hi = boot_ci(d.values, np.median)
        mean_lo, mean_hi = boot_ci(d.values, np.mean)
        row = dict(keys)
        row.update({
            'n_pairs': len(d),
            'mean_45': a['45'].mean(), 'mean_54': a['54'].mean(),
            'median_45': a['45'].median(), 'median_54': a['54'].median(),
            'median_diff_45_minus_54': np.median(d),
            'mean_diff_45_minus_54': np.mean(d),
            'q1_diff': np.percentile(d, 25), 'q3_diff': np.percentile(d, 75),
            'n_45_higher': int((d > 0).sum()), 'n_ties': int((d == 0).sum()),
            'n_54_higher': int((d < 0).sum()),
            'boot_median_ci_low': med_lo, 'boot_median_ci_high': med_hi,
            'boot_mean_ci_low': mean_lo, 'boot_mean_ci_high': mean_hi,
        })
        return row

    paired_rows = []
    for level in ['candidate', 'site']:
        for model in MODELS:
            for th in THRESHOLDS + ['best']:
                sub = perf[(perf['level'] == level) & (perf['model'] == model) &
                           (perf['threshold'] == th)]
                if len(sub) == 0:
                    continue
                for metric in ['F1', 'Recall', 'Precision'] + (['AUC', 'PR_AUC'] if level == 'candidate' else []):
                    row = paired_stats(sub, metric, {
                        'comparison': 'all_targets', 'metric_level': level,
                        'model': model, 'threshold': th, 'metric': metric})
                    if row is not None:
                        paired_rows.append(row)
    for model in MODELS:
        for th in THRESHOLDS:
            sub = novel_wide[(novel_wide['model'] == model) & (novel_wide['threshold'] == th)].copy()
            # make a group column by reshaping: comparisons handled directly
            d = sub['diff_45_minus_54'].dropna().values
            if len(d):
                med_lo, med_hi = boot_ci(d, np.median)
                mean_lo, mean_hi = boot_ci(d, np.mean)
                paired_rows.append({
                    'comparison': 'novel_targets', 'metric_level': 'site',
                    'model': model, 'threshold': th, 'metric': 'Recall',
                    'n_pairs': len(d),
                    'mean_45': sub['recall_45'].mean(), 'mean_54': sub['recall_54'].mean(),
                    'median_45': sub['recall_45'].median(), 'median_54': sub['recall_54'].median(),
                    'median_diff_45_minus_54': np.median(d),
                    'mean_diff_45_minus_54': np.mean(d),
                    'q1_diff': np.percentile(d, 25), 'q3_diff': np.percentile(d, 75),
                    'n_45_higher': int((d > 0).sum()), 'n_ties': int((d == 0).sum()),
                    'n_54_higher': int((d < 0).sum()),
                    'boot_median_ci_low': med_lo, 'boot_median_ci_high': med_hi,
                    'boot_mean_ci_low': mean_lo, 'boot_mean_ci_high': mean_hi,
                })
            dp = sub['diff_present_45_minus_54'].dropna().values
            if len(dp):
                med_lo, med_hi = boot_ci(dp, np.median)
                mean_lo, mean_hi = boot_ci(dp, np.mean)
                paired_rows.append({
                    'comparison': 'novel_targets_present', 'metric_level': 'site',
                    'model': model, 'threshold': th, 'metric': 'Recall',
                    'n_pairs': len(dp),
                    'mean_45': sub['recall_present_45'].mean(),
                    'mean_54': sub['recall_present_54'].mean(),
                    'median_45': sub['recall_present_45'].median(),
                    'median_54': sub['recall_present_54'].median(),
                    'median_diff_45_minus_54': np.median(dp),
                    'mean_diff_45_minus_54': np.mean(dp),
                    'q1_diff': np.percentile(dp, 25), 'q3_diff': np.percentile(dp, 75),
                    'n_45_higher': int((dp > 0).sum()), 'n_ties': int((dp == 0).sum()),
                    'n_54_higher': int((dp < 0).sum()),
                    'boot_median_ci_low': med_lo, 'boot_median_ci_high': med_hi,
                    'boot_mean_ci_low': mean_lo, 'boot_mean_ci_high': mean_hi,
                })
    paired = pd.DataFrame(paired_rows)
    paired.to_csv(f'{OUT}/tables/R24_group_paired_comparison.csv', index=False)
    log_df('Paired direction comparison (45 minus 54; 30 platform x depth pairs)',
           paired[(paired['model'] == 'full17') &
                  (paired['threshold'].isin([0.1, 0.6, 'best'])) &
                  (paired['metric'].isin(['F1', 'Recall', 'Precision']))]
           .round(4), index=False)

    # ---- 11. target characteristics and VAF/ALT balance ---------------
    char_rows = []
    for g in TEST_GROUPS:
        pf = pfs[g]
        for cat in ['exact', 'locus_only', 'novel']:
            s = pf[pf['category'] == cat]
            if len(s) == 0:
                continue
            char_rows.append({
                'group': g, 'category': cat, 'n': len(s),
                'P_median': s['P'].median(), 'P_q1': s['P'].quantile(0.25), 'P_q3': s['P'].quantile(0.75),
                'P_IQR': s['P'].quantile(0.75) - s['P'].quantile(0.25),
                'P_mean': s['P'].mean(), 'P_sd': s['P'].std(),
                'AF_median': s['AF'].median(), 'AF_q1': s['AF'].quantile(0.25), 'AF_q3': s['AF'].quantile(0.75),
                'AF_IQR': s['AF'].quantile(0.75) - s['AF'].quantile(0.25),
                'AF_mean': s['AF'].mean(), 'AF_sd': s['AF'].std(),
                'hom_fraction': (s['gt_class'] == 'hom').mean(),
                'hard_region_fraction': s['hard_region'].mean(),
                'mean_dist_to_nearest_target': s['dist'].mean(),
            })
    chars = pd.DataFrame(char_rows)
    chars.to_csv(f'{OUT}/tables/R24_target_characteristics_by_category.csv', index=False)
    log_df('Target characteristics by group and sharing category', chars.round(4), index=False)

    # VAF balance table
    chars.to_csv(f'{OUT}/tables/R24_vaf_balance.csv', index=False)

    # VAF tests
    test_rows = []
    for g in TEST_GROUPS:
        pf = pfs[g]
        e = pf[pf['category'] == 'exact']
        nv = pf[pf['category'] == 'novel']
        an = pf[pf['category'] != 'novel']
        for name, left in [('exact_vs_novel', e), ('any_shared_vs_novel', an)]:
            if len(left) == 0 or len(nv) == 0:
                continue
            for col in ['P', 'AF']:
                u = stats.mannwhitneyu(left[col], nv[col], alternative='two-sided')
                test_rows.append({
                    'group': g, 'comparison': name, 'variable': col,
                    'n_left': len(left), 'n_right': len(nv),
                    'median_left': left[col].median(), 'median_right': nv[col].median(),
                    'mannwhitney_U': u.statistic, 'p_value': u.pvalue,
                })
    # novel targets 45 vs 54
    e45 = pfs['45'][pfs['45']['category'] == 'novel']
    e54 = pfs['54'][pfs['54']['category'] == 'novel']
    for col in ['P', 'AF']:
        u = stats.mannwhitneyu(e45[col], e54[col], alternative='two-sided')
        test_rows.append({
            'group': '45_vs_54', 'comparison': 'novel_45_vs_novel_54', 'variable': col,
            'n_left': len(e45), 'n_right': len(e54),
            'median_left': e45[col].median(), 'median_right': e54[col].median(),
            'mannwhitney_U': u.statistic, 'p_value': u.pvalue,
        })
    vaf_tests = pd.DataFrame(test_rows)
    vaf_tests['p_value_bonferroni'] = np.minimum(1, vaf_tests['p_value'] * len(vaf_tests))
    vaf_tests.to_csv(f'{OUT}/tables/R24_vaf_tests.csv', index=False)
    log_df('VAF balance tests (variant-level Mann-Whitney)', vaf_tests.round(4), index=False)

    # ALT type balance
    subs = ['A>C', 'A>G', 'A>T', 'C>A', 'C>G', 'C>T', 'G>A', 'G>C', 'G>T', 'T>A', 'T>C', 'T>G']
    alt_rows = []
    for g in TEST_GROUPS:
        pf = pfs[g]
        for cat in ['exact', 'any_shared', 'novel']:
            s = pf[pf['category'] == cat] if cat != 'any_shared' else pf[pf['category'] != 'novel']
            if len(s) == 0:
                continue
            row = {'group': g, 'category': cat, 'n': len(s)}
            counts = s['substitution'].value_counts()
            for ss in subs:
                row[f'n_{ss}'] = int(counts.get(ss, 0))
                row[f'pct_{ss}'] = 100 * counts.get(ss, 0) / len(s)
            alt_rows.append(row)
    alt = pd.DataFrame(alt_rows)
    alt.to_csv(f'{OUT}/tables/R24_alt_type_balance.csv', index=False)
    log_df('ALT substitution spectrum by group and category (%)',
           alt[['group', 'category', 'n'] + [f'pct_{s}' for s in subs]].round(2), index=False)

    # chi-square tests for substitution spectra
    alt_test_rows = []
    for g in TEST_GROUPS:
        pf = pfs[g]
        e = pf[pf['category'] == 'exact']
        nv = pf[pf['category'] == 'novel']
        if len(e) and len(nv):
            tab = np.array([[ (e['substitution'] == s).sum() for s in subs],
                            [ (nv['substitution'] == s).sum() for s in subs]])
            # drop all-zero columns
            keep = tab.sum(axis=0) > 0
            chi2, p, dof, _ = stats.chi2_contingency(tab[:, keep])
            v = np.sqrt(chi2 / (tab.sum() * (min(tab.shape) - 1)))
            alt_test_rows.append({'group': g, 'comparison': 'exact_vs_novel',
                                  'chi2': chi2, 'dof': dof, 'p_value': p, 'cramers_v': v})
    if len(alt_test_rows):
        alt_tests = pd.DataFrame(alt_test_rows)
        alt_tests.to_csv(f'{OUT}/tables/R24_alt_type_tests.csv', index=False)
        log_df('ALT spectrum tests (exact vs novel)', alt_tests.round(4), index=False)

    # target-set comparison between groups
    tsc_rows = []
    for g in TEST_GROUPS:
        pf = pfs[g]
        for set_name, s in [('all', pf), ('novel', pf[pf['category'] == 'novel']),
                            ('shared', pf[pf['category'] != 'novel'])]:
            tsc_rows.append({
                'group': g, 'target_set': set_name, 'n': len(s),
                'P_median': s['P'].median(), 'AF_median': s['AF'].median(),
                'AF_q1': s['AF'].quantile(0.25), 'AF_q3': s['AF'].quantile(0.75),
                'hom_fraction': (s['gt_class'] == 'hom').mean(),
                'hard_region_fraction': s['hard_region'].mean(),
                'CpG_fraction': s['substitution'].isin(['C>T', 'G>A']).mean(),
            })
    tsc = pd.DataFrame(tsc_rows)
    tsc.to_csv(f'{OUT}/tables/R24_target_set_comparison.csv', index=False)
    log_df('Target-set comparison between groups', tsc.round(4), index=False)

    # ---- 12. site-level sensitivity excluding hard region / zero-candidate targets
    zero_map = {g: set(zero_df[zero_df['group'] == g]['allele']) for g in TEST_GROUPS}
    sens_rows = []
    for (plat, reads, group), sub in pred.groupby(['plat', 'reads', 'group']):
        g = str(group)
        pf = pfs[g]
        for scenario in ['all', 'no_hard_region', 'no_zero_candidate']:
            p = pf.copy()
            if scenario == 'no_hard_region':
                p = p[~p['hard_region']]
            if scenario == 'no_zero_candidate':
                p = p[~p['allele'].isin(zero_map[g])]
            truth = set(p.loc[p['dist'] >= 50, 'var'])
            near = set(p.loc[p['dist'] < 50, 'var'])
            sub_s = sub
            if scenario == 'no_hard_region':
                sub_s = sub[~((sub['CHROM'].astype(str) == HARD_REGION[0]) &
                              (sub['POS'] >= HARD_REGION[1]) & (sub['POS'] <= HARD_REGION[2]))]
            if scenario == 'no_zero_candidate':
                sub_s = sub[~sub['var'].isin(zero_map[g])]
            for model in MODELS:
                for th in [0.1, 'best']:
                    sm = site_metrics(sub_s, model, th, truth, near)
                    sens_rows.append({
                        'scenario': scenario, 'group': g, 'platform': plat, 'reads': reads,
                        'model': model, 'threshold': th,
                        'F1': sm['F1'], 'Recall': sm['Recall'], 'Precision': sm['Precision'],
                        'N_truth': sm['N_truth'],
                    })
    sens = pd.DataFrame(sens_rows)
    sens_rows2 = []
    for (scenario, model, th), sub in sens.groupby(['scenario', 'model', 'threshold']):
        a = sub.pivot_table(index=['platform', 'reads'], columns='group', values='F1')
        r = sub.pivot_table(index=['platform', 'reads'], columns='group', values='Recall')
        if '45' not in a.columns or '54' not in a.columns:
            continue
        df_ = (a['45'] - a['54']).dropna()
        dr = (r['45'] - r['54']).dropna()
        sens_rows2.append({
            'scenario': scenario, 'model': model, 'threshold': th,
            'mean_F1_45': a['45'].mean(), 'mean_F1_54': a['54'].mean(),
            'median_F1_diff_45_minus_54': np.median(df_),
            'mean_F1_diff_45_minus_54': np.mean(df_),
            'n_45_higher_F1': int((df_ > 0).sum()), 'n_54_higher_F1': int((df_ < 0).sum()),
            'mean_recall_45': r['45'].mean(), 'mean_recall_54': r['54'].mean(),
            'median_recall_diff_45_minus_54': np.median(dr),
            'mean_recall_diff_45_minus_54': np.mean(dr),
            'n_45_higher_recall': int((dr > 0).sum()), 'n_54_higher_recall': int((dr < 0).sum()),
        })
    sens_summary = pd.DataFrame(sens_rows2)
    sens_summary.to_csv(f'{OUT}/tables/R24_brca1_sensitivity.csv', index=False)
    log_df('Site-level sensitivity: all targets vs excluding hard region vs excluding zero-candidate targets',
           sens_summary.round(4), index=False)

    # ---- 13. figures ---------------------------------------------------
    plt.rcParams.update({'font.size': 8, 'axes.titlesize': 9, 'axes.labelsize': 8,
                         'legend.fontsize': 7, 'xtick.labelsize': 7, 'ytick.labelsize': 7,
                         'pdf.fonttype': 42})
    colors = {'45': '#1f77b4', '54': '#d95f02'}

    # Figure 1: F1 by depth, two panels (0.1 and 0.6)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), sharey=True)
    for ax, th in zip(axes, THRESHOLDS):
        for level, ls, level_name in [('site', '-', 'site-level'), ('candidate', '--', 'candidate-level')]:
            for g in TEST_GROUPS:
                s = perf[(perf['model'] == 'full17') & (perf['level'] == level) &
                         (perf['threshold'] == th) & (perf['group'] == g)]
                m = s.groupby('reads')['F1'].mean()
                lo = s.groupby('reads')['F1'].min()
                hi = s.groupby('reads')['F1'].max()
                ax.plot(m.index, m.values, ls, color=colors[g], lw=1.3,
                        marker='o' if g == '45' else 's', ms=3,
                        label=f'{SHORT_LABEL[g]} ({level_name})' if level == 'site' else None)
                ax.fill_between(m.index, lo.values, hi.values, color=colors[g], alpha=0.10)
        ax.set_title(f'F1 at threshold {th} (SomaticXGB full17)')
        ax.set_xlabel('Tumour reads (M)')
        ax.grid(alpha=0.25)
        ax.set_xticks(sorted(perf['reads'].unique()))
    axes[0].set_ylabel('F1 (mean across platforms; band = range)')
    handles, labels = axes[0].get_legend_handles_labels()
    # add dashed candidate-level entries manually
    from matplotlib.lines import Line2D
    handles += [Line2D([0], [0], color='k', ls='-', lw=1.2, label='site-level (all targets)'),
                Line2D([0], [0], color='k', ls='--', lw=1.2, label='candidate-level')]
    labels += ['site-level (all targets)', 'candidate-level']
    axes[0].legend(handles, labels, frameon=False, fontsize=6, loc='lower right')
    fig.tight_layout()
    fig.savefig(f'{OUT}/figures/R24_f1_by_group.png', dpi=300)
    fig.savefig(f'{OUT}/figures/R24_f1_by_group.pdf')
    plt.close(fig)
    log('\nWrote figures/R24_f1_by_group.png/.pdf')

    # Figure 2: shared vs novel
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.4))
    # Panel A: mean recall by category, site-level, full17 0.1
    ax = axes[0]
    cats = ['exact', 'any_shared', 'novel']
    width = 0.35
    for gi, g in enumerate(TEST_GROUPS):
        s = svn_site[(svn_site['group'] == g) & (svn_site['model'] == 'full17') &
                     (svn_site['threshold'] == 0.1)]
        vals = [s[f'recall_{c}'].mean() for c in cats]
        errs = [[s[f'recall_{c}'].mean() - s[f'recall_{c}'].min() for c in cats],
                [s[f'recall_{c}'].max() - s[f'recall_{c}'].mean() for c in cats]]
        ax.bar(np.arange(len(cats)) + (gi - 0.5) * width, vals, width,
               color=colors[g], yerr=errs, capsize=2, label=SHORT_LABEL[g])
    ax.set_xticks(np.arange(len(cats)))
    ax.set_xticklabels(['exact shared', 'any shared', 'novel'])
    ax.set_ylabel('Site-level recall (mean; whiskers = range)')
    ax.set_ylim(0.75, 1.0)
    ax.set_title('A  Shared vs novel targets (full17, 0.1)')
    ax.legend(frameon=False, fontsize=6)
    ax.grid(axis='y', alpha=0.25)

    # Panel B: distribution of exact-novel difference by setting
    ax = axes[1]
    settings = [('site_all', svn_site, 'site-level all'),
                ('site_no_hard_region', svn_nohard, 'site-level no hard region'),
                ('candidate_present', svn_present, 'candidate-present')]
    box_data, positions, box_colors = [], [], []
    for gi, g in enumerate(TEST_GROUPS):
        for si, (key, df_, lab) in enumerate(settings):
            s = df_[(df_['group'] == g) & (df_['model'] == 'full17') & (df_['threshold'] == 0.1)]
            box_data.append(s['diff_exact_minus_novel'].dropna().values)
            positions.append(gi * (len(settings) + 1) + si)
            box_colors.append(colors[g])
    bp = ax.boxplot(box_data, positions=positions, widths=0.65, patch_artist=True,
                    showfliers=False, medianprops={'color': 'k', 'lw': 1})
    for patch, c in zip(bp['boxes'], box_colors):
        patch.set_facecolor(c); patch.set_alpha(0.45)
    for pos, vals in zip(positions, box_data):
        x = np.random.default_rng(1).normal(pos, 0.06, size=len(vals))
        ax.scatter(x, vals, s=4, color='k', alpha=0.35, linewidths=0)
    ax.axhline(0, color='k', ls=':', lw=0.8)
    tick_pos = [gi * (len(settings) + 1) + (len(settings) - 1) / 2 for gi in range(len(TEST_GROUPS))]
    ax.set_xticks(tick_pos)
    ax.set_xticklabels([SHORT_LABEL[g].replace(': ', '\n') for g in TEST_GROUPS])
    ax.set_ylabel('Recall(exact) - recall(novel)')
    ax.set_title('B  Difference per condition (full17, 0.1)')
    ax.grid(axis='y', alpha=0.25)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(facecolor='k', alpha=0.45, label=l) for _, _, l in settings],
              frameon=False, fontsize=6, loc='lower left')

    # Panel C: AF-stratified difference, site-level all
    ax = axes[2]
    for g in TEST_GROUPS:
        s = svn_strat[(svn_strat['group'] == g) & (svn_strat['model'] == 'full17') &
                      (svn_strat['threshold'] == 0.1) & (svn_strat['stratifier'] == 'AF') &
                      (svn_strat['setting'] == 'site_all')]
        if len(s) == 0:
            continue
        ax.plot(s['quintile'], s['median_diff'], marker='o', ms=4, color=colors[g],
                label=SHORT_LABEL[g])
    ax.axhline(0, color='k', ls=':', lw=0.8)
    ax.set_xticks(range(5))
    ax.set_xlabel('AF quintile (low to high)')
    ax.set_ylabel('Median recall difference')
    ax.set_title('C  AF-stratified difference (full17, 0.1)')
    ax.legend(frameon=False, fontsize=6)
    ax.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(f'{OUT}/figures/R24_shared_vs_novel_recall.png', dpi=300)
    fig.savefig(f'{OUT}/figures/R24_shared_vs_novel_recall.pdf')
    plt.close(fig)
    log('Wrote figures/R24_shared_vs_novel_recall.png/.pdf')

    # Figure 3: novel-only recall by depth
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), sharey=True)
    for ax, th in zip(axes, THRESHOLDS):
        for g in TEST_GROUPS:
            s = novel_wide[(novel_wide['model'] == 'full17') & (novel_wide['threshold'] == th)]
            m = s.groupby('reads')[f'recall_{g}'].mean()
            lo = s.groupby('reads')[f'recall_{g}'].min()
            hi = s.groupby('reads')[f'recall_{g}'].max()
            ax.plot(m.index, m.values, '-o' if g == '45' else '-s', color=colors[g],
                    ms=3, lw=1.3, label=SHORT_LABEL[g])
            ax.fill_between(m.index, lo.values, hi.values, color=colors[g], alpha=0.12)
        ax.set_title(f'Novel-only recall at threshold {th} (full17)')
        ax.set_xlabel('Tumour reads (M)')
        ax.grid(alpha=0.25)
        ax.set_xticks(sorted(novel_wide['reads'].unique()))
    axes[0].set_ylabel('Novel-target recall\n(mean across platforms; band = range)')
    axes[0].legend(frameon=False, fontsize=6, loc='lower right')
    fig.tight_layout()
    fig.savefig(f'{OUT}/figures/R24_novel_only_recall.png', dpi=300)
    fig.savefig(f'{OUT}/figures/R24_novel_only_recall.pdf')
    plt.close(fig)
    log('Wrote figures/R24_novel_only_recall.png/.pdf')

    # ---- 14. key results table -----------------------------------------
    key_rows = []
    def add_key(section, metric, value, note=''):
        key_rows.append({'section': section, 'metric': metric, 'value': value, 'note': note})

    add_key('sharing', 'training_distinct_alleles', len(train_alleles))
    add_key('sharing', 'training_distinct_loci', len(train_loci))
    add_key('sharing', 'test_targets_total', n_all)
    add_key('sharing', 'exact_shared_total', f'{es_all}/{n_all} ({100*es_all/n_all:.1f}%)')
    add_key('sharing', 'locus_shared_total', f'{ls_all}/{n_all} ({100*ls_all/n_all:.1f}%)')
    for g in TEST_GROUPS:
        s = sharing[sharing['group'] == g].iloc[0]
        add_key('sharing', f'group{g}_exact_shared', f'{int(s["exact_shared"])}/{int(s["n_targets"])} ({s["exact_shared_pct"]:.1f}%)')
        add_key('sharing', f'group{g}_locus_shared', f'{int(s["locus_shared"])}/{int(s["n_targets"])} ({s["locus_shared_pct"]:.1f}%)')
    for g in ['45', '54']:
        for model in ['full17', 'released']:
            s = group_summary[(group_summary['group'] == g) & (group_summary['model'] == model) &
                              (group_summary['aggregation'] == 'pooled_candidates')].iloc[0]
            add_key('performance', f'group{g}_{model}_pooled_F1@0.1', s['F1@0.1'])
            add_key('performance', f'group{g}_{model}_pooled_AUC', s['AUC'])
            add_key('performance', f'group{g}_{model}_pooled_PR_AUC', s['PR_AUC'])
    for metric in ['F1', 'Recall', 'Precision']:
        s = paired[(paired['comparison'] == 'all_targets') & (paired['metric_level'] == 'site') &
                   (paired['model'] == 'full17') & (paired['threshold'] == 0.1) &
                   (paired['metric'] == metric)]
        if len(s):
            s = s.iloc[0]
            add_key('paired_site', f'{metric}_mean_45_minus_54', s['mean_diff_45_minus_54'],
                    f'45 higher {s["n_45_higher"]}/30, 54 higher {s["n_54_higher"]}/30')
    for metric in ['F1', 'Recall', 'Precision']:
        s = paired[(paired['comparison'] == 'all_targets') & (paired['metric_level'] == 'candidate') &
                   (paired['model'] == 'full17') & (paired['threshold'] == 0.1) &
                   (paired['metric'] == metric)]
        if len(s):
            s = s.iloc[0]
            add_key('paired_candidate', f'{metric}_mean_45_minus_54', s['mean_diff_45_minus_54'],
                    f'45 higher {s["n_45_higher"]}/30, 54 higher {s["n_54_higher"]}/30')
    for _, r in svn_summary[(svn_summary['shared_definition'] == 'exact') &
                            (svn_summary['model'] == 'full17') &
                            (svn_summary['threshold'] == 0.1)].iterrows():
        add_key('shared_vs_novel', f'{r["setting"]}_group{r["group"]}_median_diff',
                r['median_diff'],
                f'45 higher {r["n_shared_higher"]}/30, novel higher {r["n_novel_higher"]}/30')
    for _, r in novel_summary[(novel_summary['model'] == 'full17') &
                              (novel_summary['threshold'] == 0.1)].iterrows():
        add_key('novel_only', f'{r["setting"]}_median_diff_45_minus_54', r['median_diff_45_minus_54'],
                f'45 higher {r["n_45_higher"]}/30, 54 higher {r["n_54_higher"]}/30')
    add_key('sensitivity', 'zero_candidate_targets_45', len(zero_map['45']))
    add_key('sensitivity', 'zero_candidate_targets_54', len(zero_map['54']))
    if len(zero_df):
        add_key('sensitivity', 'zero_candidate_shared_54',
                int((zero_df[zero_df['group'] == '54']['category'] == 'exact').sum()),
                'all in chr17:41.19-41.30Mb' if zero_df[zero_df['group'] == '54']['hard_region'].all() else '')
    key = pd.DataFrame(key_rows)
    key.to_csv(f'{OUT}/tables/R24_key_results.csv', index=False)
    log_df('Key results', key, index=False)

    log('\nDone in %.1f s' % (time.time() - t0))
    log('Tables written to:', f'{OUT}/tables')
    log('Figures written to:', f'{OUT}/figures')
    _log_handle.close()


if __name__ == '__main__':
    main()
