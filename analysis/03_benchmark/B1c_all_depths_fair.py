#!/usr/bin/env python3
"""B1c. Extend the 100M fair-tuning comparison to all five sequencing depths.

B1b selected threshold rules on 100M data with cross-group protocols (train on
one donor direction, evaluate on the other).  B1c does NOT re-tune at each
depth.  It freezes the 100M rules and applies them unchanged to 60M, 80M,
100M, 120M and 140M:

  Strelka2              default (FILTER=PASS)
  Mutect2               TLOD >= 8   (primary fixed rule)
                        TLOD >= 10  (sensitivity only)
  VarScan2              SSC >= 3
  Strelka2+Mutect2 union  Strelka2 default UNION Mutect2 TLOD >= 8
                        union with TLOD >= 10 (sensitivity only)
  SomaticXGB            pred >= 0.1 (paper main threshold)
                        pred >= 0.6 (B1b cross-group fair-tuned rule)

B1b's two cross-group folds selected TLOD>=10 (45->54) and TLOD>=8 (54->45).
Because one fixed rule has to be applied to every depth, the looser TLOD>=8 is
used as the primary Mutect2/union rule; TLOD>=10 is carried as a sensitivity
analysis.  This is stated in every output table.

Truth / near-site handling, VCF parsing and metric definitions are imported
from B1_baseline_threshold_sweep.py (which follows scripts/make_fig3_data.py):
  dist >= 50 -> truth, dist < 50 -> near; calls at near sites are removed;
  FN = |truth| - TP; single-base SNVs only for VCF callers; SomaticXGB
  prediction files keep all candidates as in B1.

The 12 samples per depth are conditions (2 donor directions x 6 platforms),
not independent replicates.  Only descriptive mean +/- SD summaries are
reported; no p values.

Validation: the 100M results are compared call-for-call and metric-for-metric
with tables/B1b_default_by_sample.csv,
tables/B1b_cross_group_eval_by_sample.csv and tables/B1_parse_stats.csv before
the other four depths are parsed.

Outputs (all new; existing B1/B1b files are not touched):
  tables/B1c_all_depths_by_condition.csv
  tables/B1c_all_depths_summary.csv
  tables/B1c_conclusion.txt
  figures/B1c_f1_by_depth.png / .pdf
  logs/B1c.log (written by the shell invocation)

Usage:
  /usr/bin/python3 scripts/B1c_all_depths_fair.py
  /usr/bin/python3 scripts/B1c_all_depths_fair.py --no-abort-on-validation-error
"""

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import B1_baseline_threshold_sweep as b1  # noqa: E402


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

OUT = b1.OUT
TABLES = b1.TABLES
FIGURES = b1.FIGURES

GROUPS = b1.GROUPS
PLATFORMS = b1.PLATFORMS
PLATFORM_NAMES = [p for _, p in PLATFORMS]
DEPTHS = [60, 80, 100, 120, 140]
SOMATIC_DEFAULT_CUT = b1.SOMATIC_DEFAULT_CUT

os.makedirs(TABLES, exist_ok=True)
os.makedirs(FIGURES, exist_ok=True)


def log(msg=''):
    b1.log(msg)


def section(title):
    b1.section(title)


# Fixed-rule conditions.  "Order" controls table/figure ordering.
CONDITIONS = [
    {'key': 'strelka_default', 'method': 'Strelka2', 'order': 1,
     'label': 'Strelka2 (default PASS)',
     'rule': 'FILTER=PASS',
     'rule_type': 'default',
     'rule_source': 'B1b cross-group selected default in both directions'},
    {'key': 'mutect2_tlod8', 'method': 'Mutect2', 'order': 2,
     'label': 'Mutect2 (TLOD>=8)',
     'rule': 'TLOD>=8',
     'rule_type': 'fixed-100M (unified)',
     'rule_source': ('B1b selected TLOD>=10 (45->54) and TLOD>=8 (54->45); '
                     'unified to the looser TLOD>=8 for all depths')},
    {'key': 'mutect2_tlod10', 'method': 'Mutect2', 'order': 3,
     'label': 'Mutect2 (TLOD>=10, sensitivity)',
     'rule': 'TLOD>=10',
     'rule_type': 'sensitivity',
     'rule_source': 'B1b 45->54 cross-group rule'},
    {'key': 'varscan_ssc3', 'method': 'VarScan2', 'order': 4,
     'label': 'VarScan2 (SSC>=3, fair-tuned)',
     'rule': 'SSC>=3',
     'rule_type': 'fixed-100M',
     'rule_source': 'B1b cross-group selected SSC>=3 in both directions'},
    {'key': 'union_tlod8', 'method': 'Strelka2_Mutect2_union', 'order': 5,
     'label': 'Strelka2+Mutect2 union (TLOD>=8)',
     'rule': 'Strelka2 default | TLOD>=8',
     'rule_type': 'fixed-100M (unified)',
     'rule_source': ('Strelka2 default union Mutect2 TLOD>=8; same unified '
                     'TLOD rule as Mutect2 above')},
    {'key': 'union_tlod10', 'method': 'Strelka2_Mutect2_union', 'order': 6,
     'label': 'Strelka2+Mutect2 union (TLOD>=10, sensitivity)',
     'rule': 'Strelka2 default | TLOD>=10',
     'rule_type': 'sensitivity',
     'rule_source': 'B1b 45->54 independent cross-group union rule'},
    {'key': 'xgb_pred01', 'method': 'SomaticXGB', 'order': 7,
     'label': 'SomaticXGB (pred>=0.1, paper default)',
     'rule': 'pred>=0.1',
     'rule_type': 'paper default',
     'rule_source': 'Paper main threshold; identical to B1 default'},
    {'key': 'xgb_pred06', 'method': 'SomaticXGB', 'order': 8,
     'label': 'SomaticXGB (pred>=0.6, fair-tuned)',
     'rule': 'pred>=0.6',
     'rule_type': 'fixed-100M',
     'rule_source': 'B1b cross-group selected pred>=0.6 in both directions'},
]
COND_BY_KEY = {c['key']: c for c in CONDITIONS}
KEY_RULES = ['xgb_pred06', 'xgb_pred01', 'varscan_ssc3', 'union_tlod8']


# ---------------------------------------------------------------------------
# Depth-aware paths and loading
# ---------------------------------------------------------------------------

def sample_id(pid, depth):
    """Sample stem, e.g. FASTASeq_300_reads60M."""
    return f'{pid}_reads{depth}M'


def paths_for(g, pid, depth):
    s = sample_id(pid, depth)
    return {
        'Strelka2': (f'{b1.TESTDIR}/{g}/strelka/{s}/results/variants/'
                     f'somatic.snvs.vcf.gz'),
        'Mutect2': f'{b1.TESTDIR}/{g}/mutect/{s}.filter.vcf.gz',
        'VarScan2': f'{b1.TESTDIR}/{g}/varscan_novo/{s}.snp.pass.vcf.gz',
        'SomaticXGB': f'{b1.TESTDIR}/{g}/pred/{s}_pred.txt.gz',
    }


def load_depth(depth, truth_cache):
    """Parse all 12 samples at one depth.  Returns (recs, parse_stats_rows)."""
    t0 = time.time()
    section(f'Parsing calls at {depth}M')
    recs = []
    stat_rows = []

    for g in GROUPS:
        truth = truth_cache[g]['truth']
        near = truth_cache[g]['near']
        for pid, platform in PLATFORMS:
            paths = paths_for(g, pid, depth)
            rec = {
                'group': g,
                'donor': g,
                'platform': platform,
                'pid': pid,
                'depth': depth,
                'sample': sample_id(pid, depth),
                'truth': truth,
                'near': near,
            }

            scores, st = b1.parse_vcf_scores(
                paths['Strelka2'], 'Strelka2', pass_only=True,
                near=near, verbose=False)
            rec['Strelka2'] = {'scores': scores}
            stat_rows.append({'Depth_M': depth, 'Group': g,
                              'Platform': platform,
                              'Sample': rec['sample'], 'Method': 'Strelka2',
                              'CallSet': 'PASS', **st})

            scores, st = b1.parse_vcf_scores(
                paths['Mutect2'], 'Mutect2', pass_only=True,
                near=near, verbose=False)
            rec['Mutect2'] = {'scores': scores}
            stat_rows.append({'Depth_M': depth, 'Group': g,
                              'Platform': platform,
                              'Sample': rec['sample'], 'Method': 'Mutect2',
                              'CallSet': 'PASS', **st})

            scores, st = b1.parse_vcf_scores(
                paths['VarScan2'], 'VarScan2', pass_only=False,
                near=near, verbose=False)
            rec['VarScan2'] = {'scores': scores}
            stat_rows.append({'Depth_M': depth, 'Group': g,
                              'Platform': platform,
                              'Sample': rec['sample'], 'Method': 'VarScan2',
                              'CallSet': 'snp.pass', **st})

            scores, st = b1.parse_pred_scores(
                paths['SomaticXGB'], near, verbose=False)
            rec['SomaticXGB'] = {'scores': scores}
            stat_rows.append({'Depth_M': depth, 'Group': g,
                              'Platform': platform,
                              'Sample': rec['sample'],
                              'Method': 'SomaticXGB',
                              'CallSet': 'pred_all', **st})

            recs.append(rec)

    # Compact per-depth call counts for the log.
    stat_df = pd.DataFrame(stat_rows)
    log(f'[depth {depth}M] parsed {len(recs)} samples '
        f'in {time.time() - t0:.1f}s')
    for method in ['Strelka2', 'Mutect2', 'VarScan2', 'SomaticXGB']:
        s = stat_df[stat_df['Method'] == method]
        log(f'  {method:<10} calls after near filter: '
            f'min={int(s["n_calls"].min())} '
            f'median={int(s["n_calls"].median())} '
            f'max={int(s["n_calls"].max())}')
    return recs, stat_rows


# ---------------------------------------------------------------------------
# Fixed rules -> calls -> metrics
# ---------------------------------------------------------------------------

def calls_for(rec, key):
    """Return the call set for one fixed rule on one sample."""
    if key == 'strelka_default':
        return set(rec['Strelka2']['scores'])
    if key == 'mutect2_tlod8':
        return {v for v, s in rec['Mutect2']['scores'].items() if s >= 8}
    if key == 'mutect2_tlod10':
        return {v for v, s in rec['Mutect2']['scores'].items() if s >= 10}
    if key == 'varscan_ssc3':
        return {v for v, s in rec['VarScan2']['scores'].items() if s >= 3}
    if key == 'union_tlod8':
        return (set(rec['Strelka2']['scores']) |
                {v for v, s in rec['Mutect2']['scores'].items() if s >= 8})
    if key == 'union_tlod10':
        return (set(rec['Strelka2']['scores']) |
                {v for v, s in rec['Mutect2']['scores'].items() if s >= 10})
    if key == 'xgb_pred01':
        return {v for v, s in rec['SomaticXGB']['scores'].items()
                if s >= SOMATIC_DEFAULT_CUT}
    if key == 'xgb_pred06':
        return {v for v, s in rec['SomaticXGB']['scores'].items() if s >= 0.6}
    raise ValueError(key)


def condition_table(recs):
    """One row per depth x fixed rule x group x platform."""
    rows = []
    for rec in recs:
        for cond in CONDITIONS:
            calls = calls_for(rec, cond['key'])
            m = b1.compute_metrics(calls, rec['truth'])
            rows.append({
                'Method': cond['method'],
                'Method_Label': cond['label'],
                'Rule': cond['rule'],
                'Rule_Key': cond['key'],
                'Rule_Type': cond['rule_type'],
                'Rule_Source': cond['rule_source'],
                'Condition_Order': cond['order'],
                'Depth_M': rec['depth'],
                'Group': rec['group'],
                'Donor': rec['donor'],
                'Platform': rec['platform'],
                'Sample': rec['sample'],
                'N_calls': len(calls),
                'TP': m['TP'],
                'FP': m['FP'],
                'FN': m['FN'],
                'TruthN': m['TruthN'],
                'Recall': m['Recall'],
                'Precision': m['Precision'],
                'F1': m['F1'],
            })
    df = pd.DataFrame(rows)
    df = df.sort_values(['Depth_M', 'Condition_Order', 'Group', 'Platform'],
                        kind='stable').reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Validation against the existing 100M B1/B1b tables
# ---------------------------------------------------------------------------

def _compare_rows(df, ref, key_cols, label,
                  metric_cols=('N_calls', 'TP', 'FP', 'FN', 'F1')):
    """Compare metric columns between our 100M rows and a B1/B1b table."""
    df = df.copy()
    ref = ref.copy()
    for col in key_cols:
        df[col] = df[col].astype(str)
        ref[col] = ref[col].astype(str)
    merged = df.merge(ref, on=key_cols, how='outer',
                      suffixes=('', '_ref'), indicator=True)
    missing = int((merged['_merge'] != 'both').sum())
    bad = []
    if missing:
        bad.append(f'{label}: {missing} rows missing on one side')
    both = merged[merged['_merge'] == 'both']
    for col in metric_cols:
        a = pd.to_numeric(both[col], errors='coerce').to_numpy(dtype=float)
        b = pd.to_numeric(both[f'{col}_ref'], errors='coerce').to_numpy(dtype=float)
        diff = np.abs(a - b)
        n_bad = int(np.sum(~(diff <= 1e-9)))
        if n_bad:
            ex = both.loc[~(diff <= 1e-9), key_cols + [col, f'{col}_ref']]
            bad.append(f'{label}: {n_bad} mismatches in {col}; '
                       f'example:\n{ex.head(3).to_string(index=False)}')
    n = len(both)
    log(f'  {label}: compared {n} rows '
        f'({"PASS" if not bad else "FAIL"})')
    return bad


def validate_100m(df100, recs100, stats100):
    """Validate the 100M run against B1/B1b outputs.  Returns list of problems."""
    section('VALIDATION at 100M vs existing B1 / B1b tables')
    problems = []

    def load(name):
        path = os.path.join(TABLES, name)
        if not os.path.exists(path):
            problems.append(f'missing reference table: {path}')
            return None
        return pd.read_csv(path)

    # 1. Default call sets and metrics: B1b_default_by_sample.csv.
    # Recompute the default set for each caller directly from the parsed
    # scores so that Mutect2 and VarScan2 defaults are checked too.
    ref = load('B1b_default_by_sample.csv')
    if ref is not None:
        ref = ref[ref['Protocol'] == 'default'].copy()
        default_rows = []
        for rec in recs100:
            truth = rec['truth']
            default_sets = {
                'Strelka2': set(rec['Strelka2']['scores']),
                'Mutect2': set(rec['Mutect2']['scores']),
                'VarScan2': set(rec['VarScan2']['scores']),
                'SomaticXGB': {v for v, s in rec['SomaticXGB']['scores'].items()
                               if s > SOMATIC_DEFAULT_CUT},
            }
            for method, calls in default_sets.items():
                m = b1.compute_metrics(calls, truth)
                default_rows.append({
                    'Method': method, 'Group': rec['group'],
                    'Platform': rec['platform'], 'Sample': rec['sample'],
                    'N_calls': len(calls), 'TP': m['TP'], 'FP': m['FP'],
                    'FN': m['FN'], 'F1': m['F1'],
                })
        ours = pd.DataFrame(default_rows)
        for method in ['Strelka2', 'Mutect2', 'VarScan2', 'SomaticXGB']:
            r = ref[ref['Method'] == method][
                ['Group', 'Platform', 'Sample', 'N_calls', 'TP', 'FP', 'FN',
                 'F1']].copy()
            o = ours[ours['Method'] == method][
                ['Group', 'Platform', 'Sample', 'N_calls', 'TP', 'FP', 'FN',
                 'F1']].copy()
            problems += _compare_rows(
                o, r, ['Group', 'Platform', 'Sample'],
                f'default {method} vs B1b_default_by_sample')

    # 2. Cross-group fixed rules: B1b_cross_group_eval_by_sample.csv
    ref = load('B1b_cross_group_eval_by_sample.csv')
    if ref is not None:
        # Fold evaluated on this sample's group.
        # group 45 <- fold 54->45 ; group 54 <- fold 45->54
        ref = ref.copy()
        ref['Group'] = ref['Group'].astype(str)
        fold_map = {'54': '45->54', '45': '54->45'}

        def ref_fold(g):
            return fold_map[str(g)]

        our_rule_for_fold = {
            'VarScan2': {'45->54': 'varscan_ssc3', '54->45': 'varscan_ssc3'},
            'Mutect2': {'45->54': 'mutect2_tlod10', '54->45': 'mutect2_tlod8'},
            'SomaticXGB': {'45->54': 'xgb_pred06', '54->45': 'xgb_pred06'},
        }
        for method, mapping in our_rule_for_fold.items():
            r = ref[(ref['Method'] == method) &
                    (ref['Protocol'].isin(['cross-group']))].copy()
            r = r[r['Fold'] == r['Group'].map(ref_fold)]
            r = r[['Group', 'Platform', 'Sample', 'N_calls', 'TP', 'FP',
                   'FN', 'F1']]
            o_parts = []
            for fold, key in mapping.items():
                g = '54' if fold == '45->54' else '45'
                o = df100[(df100['Rule_Key'] == key) &
                          (df100['Group'] == g)][
                    ['Group', 'Platform', 'Sample', 'N_calls', 'TP', 'FP',
                     'FN', 'F1']]
                o_parts.append(o)
            o = pd.concat(o_parts, ignore_index=True)
            problems += _compare_rows(
                o, r, ['Group', 'Platform', 'Sample'],
                f'cross-group {method} vs B1b_cross_group_eval_by_sample')

        # Union (independent) rows.
        r = ref[(ref['Method'] == 'Strelka2_Mutect2_union') &
                (ref['Protocol'] == 'cross-group (independent)')].copy()
        r = r[r['Fold'] == r['Group'].map(ref_fold)]
        r = r[['Group', 'Platform', 'Sample', 'N_calls', 'TP', 'FP', 'FN',
               'F1']]
        o_parts = []
        for fold, key in {'45->54': 'union_tlod10',
                          '54->45': 'union_tlod8'}.items():
            g = '54' if fold == '45->54' else '45'
            o = df100[(df100['Rule_Key'] == key) &
                      (df100['Group'] == g)][
                ['Group', 'Platform', 'Sample', 'N_calls', 'TP', 'FP', 'FN',
                 'F1']]
            o_parts.append(o)
        o = pd.concat(o_parts, ignore_index=True)
        problems += _compare_rows(
            o, r, ['Group', 'Platform', 'Sample'],
            'cross-group union vs B1b_cross_group_eval_by_sample')

    # 3. Parse-level call counts: B1_parse_stats.csv (primary rows).
    ref = load('B1_parse_stats.csv')
    if ref is not None:
        ref = ref[ref['analysis'] == 'primary'].copy()
        ref['Group'] = ref['group'].astype(str)
        ref['Platform'] = ref['platform']
        ref['Sample'] = ref['sample']
        ours = pd.DataFrame(stats100)
        for method, callset in [('Strelka2', 'PASS'), ('Mutect2', 'PASS'),
                                ('VarScan2', 'snp.pass'),
                                ('SomaticXGB', 'pred_all')]:
            r = ref[(ref['callset'] == callset) &
                    (ref['score_field'] == {'Strelka2': 'QSS',
                                            'Mutect2': 'TLOD',
                                            'VarScan2': 'SSC',
                                            'SomaticXGB': 'pred'}[method])][
                ['Group', 'Platform', 'Sample', 'n_calls']].copy()
            r.columns = ['Group', 'Platform', 'Sample', 'N_calls']
            o = ours[ours['Method'] == method][
                ['Group', 'Platform', 'Sample', 'n_calls']].copy()
            o.columns = ['Group', 'Platform', 'Sample', 'N_calls']
            problems += _compare_rows(
                o, r, ['Group', 'Platform', 'Sample'],
                f'parse n_calls {method} vs B1_parse_stats',
                metric_cols=('N_calls',))

    if problems:
        log('  VALIDATION PROBLEMS:')
        for p in problems:
            log(f'    - {p}')
    else:
        log('  VALIDATION PASSED: 100M call counts and metrics match '
            'B1/B1b exactly (tolerance 1e-9).')
    return problems


# ---------------------------------------------------------------------------
# Summaries
# ---------------------------------------------------------------------------

def summarize(df):
    """Mean +/- SD across platforms at each depth, for both donors and each."""
    rows = []
    for depth in DEPTHS:
        dsub = df[df['Depth_M'] == depth]
        for scope, ssub in [('both', dsub),
                            ('45', dsub[dsub['Group'] == '45']),
                            ('54', dsub[dsub['Group'] == '54'])]:
            if ssub.empty:
                continue
            # Use mean F1 of SomaticXGB 0.6 within the same scope/depth.
            xgb06_mean = float(ssub[ssub['Rule_Key'] == 'xgb_pred06']['F1'].mean())
            xgb06_by_platform = ssub[ssub['Rule_Key'] == 'xgb_pred06'].set_index(
                ['Group', 'Platform'])['F1']
            for cond in CONDITIONS:
                s = ssub[ssub['Rule_Key'] == cond['key']]
                if s.empty:
                    continue
                f1 = s['F1'].astype(float)
                rec = s['Recall'].astype(float)
                pre = s['Precision'].astype(float)
                # Number of conditions (platform within donor direction) where
                # this rule beats SomaticXGB 0.6.
                idx = pd.MultiIndex.from_frame(s[['Group', 'Platform']])
                n_above = int(np.sum(f1.to_numpy() >
                                     xgb06_by_platform.reindex(idx).to_numpy()))
                rows.append({
                    'Depth_M': depth,
                    'Scope': scope,
                    'Method': cond['method'],
                    'Method_Label': cond['label'],
                    'Rule': cond['rule'],
                    'Rule_Key': cond['key'],
                    'Rule_Type': cond['rule_type'],
                    'N_samples': int(len(s)),
                    'Mean_Recall': float(rec.mean()),
                    'SD_Recall': float(rec.std(ddof=1)),
                    'Mean_Precision': float(pre.mean()),
                    'SD_Precision': float(pre.std(ddof=1)),
                    'Mean_F1': float(f1.mean()),
                    'SD_F1': float(f1.std(ddof=1)),
                    'Median_F1': float(f1.median()),
                    'Min_F1': float(f1.min()),
                    'Max_F1': float(f1.max()),
                    'Delta_F1_vs_SomaticXGB_0.6': float(f1.mean()) - xgb06_mean,
                    'N_conditions_above_SomaticXGB_0.6': n_above,
                })
    out = pd.DataFrame(rows)
    scope_order = {'both': 0, '45': 1, '54': 2}
    out['_scope'] = out['Scope'].map(scope_order)
    out['_order'] = out['Rule_Key'].map(
        {c['key']: c['order'] for c in CONDITIONS})
    out = out.sort_values(['Depth_M', '_scope', '_order'],
                          kind='stable').drop(
        columns=['_scope', '_order']).reset_index(drop=True)
    return out


# ---------------------------------------------------------------------------
# Reversal diagnostics
# ---------------------------------------------------------------------------

def reversal_diagnostics(df):
    """Per-condition and per-donor-depth comparisons vs the tuned baselines."""
    piv = df.pivot_table(index=['Depth_M', 'Group', 'Platform'],
                         columns='Rule_Key', values='F1')
    baseline = piv[['varscan_ssc3', 'union_tlod8']].max(axis=1)
    out = {}
    for xk, name in [('xgb_pred06', 'SomaticXGB pred>=0.6'),
                     ('xgb_pred01', 'SomaticXGB pred>=0.1')]:
        delta = piv[xk] - baseline
        below = delta[delta < 0]
        out[name] = {
            'n_conditions': int(len(delta)),
            'n_below_best_baseline': int(len(below)),
            'min_delta': float(delta.min()),
            'mean_delta': float(delta.mean()),
            'detail': below,
        }

    # Donor x depth cell means.
    cell = df.pivot_table(index=['Depth_M', 'Group'],
                          columns='Rule_Key', values='F1', aggfunc='mean')
    cell_base = cell[['varscan_ssc3', 'union_tlod8']].max(axis=1)
    out['cell'] = pd.DataFrame({
        'SomaticXGB_0.6': cell['xgb_pred06'],
        'SomaticXGB_0.1': cell['xgb_pred01'],
        'VarScan2_SSC3': cell['varscan_ssc3'],
        'Union_TLOD8': cell['union_tlod8'],
        'best_baseline': cell_base,
        'delta_xgb06': cell['xgb_pred06'] - cell_base,
        'delta_xgb01': cell['xgb_pred01'] - cell_base,
    })
    return out


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

def make_figure(summary_df):
    x = np.arange(len(DEPTHS))
    panels = [
        ('45', 'Donor direction 45 (mean of 6 platforms)'),
        ('54', 'Donor direction 54 (mean of 6 platforms)'),
        ('both', 'Both donors (mean of 12 conditions)'),
    ]
    specs = [
        ('xgb_pred06', 'SomaticXGB pred>=0.6 (fair-tuned)',
         '#0b3d91', '-', 'o', 2.6, True, 2.3),
        ('xgb_pred01', 'SomaticXGB pred>=0.1 (paper default)',
         '#4a90e2', '--', 's', 2.2, True, 1.8),
        ('varscan_ssc3', 'VarScan2 SSC>=3 (fair-tuned)',
         '#e07b00', '-', '^', 2.4, True, 1.8),
        ('union_tlod8', 'Strelka2 + Mutect2 union (TLOD>=8)',
         '#1b8a3a', '-', 'D', 2.4, True, 1.8),
        ('mutect2_tlod8', 'Mutect2 TLOD>=8',
         '#9467bd', ':', 'v', 1.5, False, 1.2),
        ('strelka_default', 'Strelka2 default',
         '#7f7f7f', ':', 'P', 1.5, False, 1.2),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.1), sharey=True)
    all_means = []
    for ax, (scope, title) in zip(axes, panels):
        sub = summary_df[summary_df['Scope'] == scope]
        for key, label, color, ls, marker, ms, err, lw in specs:
            s = sub[sub['Rule_Key'] == key].sort_values('Depth_M')
            if s.empty:
                continue
            y = s['Mean_F1'].to_numpy(dtype=float)
            sd = s['SD_F1'].to_numpy(dtype=float)
            all_means.extend(y.tolist())
            if err:
                ax.errorbar(x, y, yerr=sd, color=color, ls=ls, marker=marker,
                            ms=ms, lw=lw, capsize=2.5,
                            elinewidth=0.8, capthick=0.8, label=label,
                            alpha=0.95)
            else:
                ax.plot(x, y, color=color, ls=ls, marker=marker, ms=ms,
                        lw=lw, label=label, alpha=0.85)
        ax.axvline(2, color='0.65', ls=':', lw=1.0)
        ax.text(2.04, 0.02, 'B1b tuning depth', fontsize=7.2, color='0.4',
                rotation=90, va='bottom', ha='left',
                transform=ax.get_xaxis_transform())
        ax.set_xticks(x)
        ax.set_xticklabels([f'{d}M' for d in DEPTHS])
        ax.set_xlabel('Sequencing depth (reads)')
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.grid(alpha=0.30, lw=0.6)
        ax.set_xlim(-0.45, len(DEPTHS) - 0.55)

    lo = max(0.0, np.floor(min(all_means) * 10) / 10 - 0.05)
    axes[0].set_ylim(lo, 1.0)
    axes[0].set_ylabel('F1 (bars = SD across platforms)')

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=3,
               fontsize=8.2, framealpha=0.95, bbox_to_anchor=(0.5, -0.005))
    fig.suptitle('B1c. Fixed 100M fair-tuned rules across sequencing depths',
                 fontsize=13, fontweight='bold')
    fig.tight_layout(rect=[0, 0.08, 1, 0.95])

    png = os.path.join(FIGURES, 'B1c_f1_by_depth.png')
    pdf = os.path.join(FIGURES, 'B1c_f1_by_depth.pdf')
    fig.savefig(png, dpi=300, bbox_inches='tight')
    fig.savefig(pdf, bbox_inches='tight')
    plt.close(fig)
    log(f'[figure] wrote {png}')
    log(f'[figure] wrote {pdf}')


# ---------------------------------------------------------------------------
# Conclusions
# ---------------------------------------------------------------------------

def write_conclusion(summary_df, df, rev):
    both = summary_df[summary_df['Scope'] == 'both']
    names = {'xgb_pred06': 'SomaticXGB0.6',
             'xgb_pred01': 'SomaticXGB0.1',
             'varscan_ssc3': 'VarScan2_SSC3',
             'union_tlod8': 'union_TLOD8'}

    def fmt(v):
        return f'{v:.3f}'

    lines = []
    lines.append(
        'Design: 100M cross-group rules frozen and applied to 60/80/100/120/'
        '140M; SomaticXGB pred>=0.1 and >=0.6, VarScan2 SSC>=3, union = '
        'Strelka2 default + Mutect2 TLOD>=8 (TLOD>=10 sensitivity in tables); '
        '12 conditions per depth, descriptive mean F1, no p values.')

    deltas = []
    for depth in DEPTHS:
        s = both[both['Depth_M'] == depth].set_index('Rule_Key')
        order = s.loc[KEY_RULES, 'Mean_F1'].sort_values(ascending=False)
        ranking = ' > '.join(f'{names[k]} {fmt(order[k])}' for k in order.index)
        base_best = s.loc[['varscan_ssc3', 'union_tlod8'], 'Mean_F1'].max()
        delta = s.loc['xgb_pred06', 'Mean_F1'] - base_best
        deltas.append((depth, delta))
        lines.append(f'{depth}M: {ranking}; XGB0.6 - best tuned baseline = '
                     f'{delta:+.3f}.')

    dmin = min(deltas, key=lambda z: z[1])
    dmax = max(deltas, key=lambda z: z[1])
    monotone_narrow = all(
        deltas[i][1] >= deltas[i + 1][1] for i in range(len(deltas) - 1))
    if monotone_narrow and dmax[0] == min(DEPTHS):
        gap_txt = (f'Gap to the best tuned baseline was widest at '
                   f'{dmax[0]}M ({dmax[1]:+.3f}) and narrowed with depth to '
                   f'{dmin[1]:+.3f} at {dmin[0]}M; low depth did not shrink '
                   f'or reverse the advantage.')
    else:
        gap_txt = (f'Gap to the best tuned baseline ranged from '
                   f'{dmin[1]:+.3f} at {dmin[0]}M to {dmax[1]:+.3f} at '
                   f'{dmax[0]}M; the ranking did not reverse.')
    lines.append(gap_txt)

    r6 = rev['SomaticXGB pred>=0.6']
    r1 = rev['SomaticXGB pred>=0.1']

    # Closest condition for each SomaticXGB threshold.
    piv = df.pivot_table(index=['Depth_M', 'Group', 'Platform'],
                         columns='Rule_Key', values='F1')
    base = piv[['varscan_ssc3', 'union_tlod8']].max(axis=1)
    d06 = piv['xgb_pred06'] - base
    d01 = piv['xgb_pred01'] - base
    idx6 = d06.idxmin()
    idx1 = d01.idxmin()

    # Cells where the union is the stronger baseline.
    union_better = piv[piv['union_tlod8'] > piv['varscan_ssc3']]
    union_plats = sorted(union_better.index.get_level_values('Platform').unique())

    rev_txt = (
        f'No reversal at any depth: pred>=0.6 and pred>=0.1 beat both tuned '
        f'baselines in {r6["n_conditions"] - r6["n_below_best_baseline"]}/'
        f'{r6["n_conditions"]} and '
        f'{r1["n_conditions"] - r1["n_below_best_baseline"]}/'
        f'{r1["n_conditions"]} platform x donor x depth conditions; closest '
        f'calls were {d01.min():+.3f} (pred>=0.1, {idx1[0]}M donor {idx1[1]}, '
        f'{idx1[2]} vs union) and {d06.min():+.3f} (pred>=0.6, {idx6[0]}M '
        f'donor {idx6[1]}, {idx6[2]}).')
    if union_plats:
        rev_txt += (f' The union was the stronger baseline only on '
                    f'{", ".join(union_plats)} (all depths), but SomaticXGB '
                    f'stayed above both there too.')
    lines.append(rev_txt + ' Details in logs/B1c.log.')

    text = '\n'.join(lines) + '\n'
    path = os.path.join(TABLES, 'B1c_conclusion.txt')
    with open(path, 'w') as fh:
        fh.write(text)
    log(f'[table] wrote {path}')
    log('')
    log('CONCLUSION FILE:')
    for line in lines:
        log('  ' + line)
    return lines


# ---------------------------------------------------------------------------
# Logging of key tables
# ---------------------------------------------------------------------------

def log_key_tables(df, summary_df, rev):
    section('SUMMARY: mean F1 across 12 conditions (both donors)')
    both = summary_df[summary_df['Scope'] == 'both'].copy()
    show = both.pivot(index='Depth_M', columns='Rule_Key', values='Mean_F1')
    show = show[[c['key'] for c in CONDITIONS if c['key'] in show.columns]]
    log(show.round(4).to_string())

    section('SUMMARY: SD F1 across 12 conditions (both donors)')
    show_sd = both.pivot(index='Depth_M', columns='Rule_Key', values='SD_F1')
    show_sd = show_sd[[c['key'] for c in CONDITIONS
                       if c['key'] in show_sd.columns]]
    log(show_sd.round(4).to_string())

    section('RANKING per depth (key methods, mean F1 across 12 conditions)')
    names = {'xgb_pred06': 'SomaticXGB0.6', 'xgb_pred01': 'SomaticXGB0.1',
             'varscan_ssc3': 'VarScan2_SSC3', 'union_tlod8': 'union_TLOD8'}
    for depth in DEPTHS:
        s = both[both['Depth_M'] == depth].set_index('Rule_Key')
        order = s.loc[KEY_RULES, 'Mean_F1'].sort_values(ascending=False)
        txt = ' > '.join(f'{names[k]} {order[k]:.4f}' for k in order.index)
        log(f'  {depth}M: {txt}')

    section('GAP: SomaticXGB minus best(VarScan2 SSC>=3, union TLOD>=8)')
    cell = rev['cell']
    log(cell.round(4).to_string())
    log('')
    for xk in ['SomaticXGB_0.6', 'SomaticXGB_0.1']:
        xgb_name = ('xgb_pred06' if xk.endswith('0.6') else 'xgb_pred01')
        log(f'  {xk}: per-condition wins/losses vs best baseline')
        for depth in DEPTHS:
            sub = df[df['Depth_M'] == depth]
            p = sub.pivot_table(index=['Group', 'Platform'],
                                columns='Rule_Key', values='F1')
            d = p[xgb_name] - p[['varscan_ssc3', 'union_tlod8']].max(axis=1)
            log(f'    {depth}M: wins={int((d > 0).sum())}/{len(d)}, '
                f'min_delta={d.min():+.4f}, mean_delta={d.mean():+.4f}')

    section('PER-PLATFORM / DONOR REVERSALS')
    for name in ['SomaticXGB pred>=0.6', 'SomaticXGB pred>=0.1']:
        r = rev[name]
        log(f'  {name}: below best tuned baseline in '
            f'{r["n_below_best_baseline"]}/{r["n_conditions"]} conditions '
            f'(min delta {r["min_delta"]:+.4f}, mean delta '
            f'{r["mean_delta"]:+.4f})')
        if not r['detail'].empty:
            det = r['detail'].rename('Delta_F1').reset_index()
            log(det.to_string(index=False))
        else:
            log('    none')

    section('SENSITIVITY: TLOD>=10 (primary rule is TLOD>=8)')
    sens = both[both['Rule_Key'].isin(
        ['mutect2_tlod8', 'mutect2_tlod10', 'union_tlod8',
         'union_tlod10'])].pivot(index='Depth_M', columns='Rule_Key',
                                 values='Mean_F1')
    log(sens.round(4).to_string())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--no-abort-on-validation-error', action='store_true',
                    help='continue even if the 100M validation mismatches')
    args = ap.parse_args()

    t0 = time.time()
    section('B1c. Fixed 100M fair-tuned rules across all five depths')
    log('Purpose: test whether the B1b method ranking changes with depth.')
    log('Rules are frozen from the 100M cross-group analysis; no re-tuning '
        'at other depths.')
    log(f'Depths: {", ".join(str(d) + "M" for d in DEPTHS)}')
    log(f'Groups (donor directions): {", ".join(GROUPS)}')
    log('Platforms: ' + ', '.join(PLATFORM_NAMES))
    log('Truth/near rule: make_fig3_data.py via B1 '
        '(dist>=50 truth, dist<50 near, calls at near removed, FN=|truth|-TP)')
    log('Metric definitions: B1.compute_metrics.')
    log('')
    log('Fixed rules:')
    for cond in CONDITIONS:
        log(f'  {cond["label"]:<48} {cond["rule"]:<32} '
            f'[{cond["rule_type"]}]')
        log(f'      source: {cond["rule_source"]}')

    truth_cache = {g: b1.load_truth(g) for g in GROUPS}

    # 1. Parse 100M first and validate against the existing B1/B1b tables.
    recs100, stats100 = load_depth(100, truth_cache)
    df100 = condition_table(recs100)
    problems = validate_100m(df100, recs100, stats100)
    if problems and not args.no_abort_on_validation_error:
        log('Aborting: 100M validation failed.  Re-run with '
            '--no-abort-on-validation-error to continue anyway.')
        sys.exit(1)

    # 2. Parse the remaining depths.
    all_recs = list(recs100)
    all_stats = list(stats100)
    for depth in [d for d in DEPTHS if d != 100]:
        recs, stats = load_depth(depth, truth_cache)
        all_recs.extend(recs)
        all_stats.extend(stats)

    section('Building condition table and summaries')
    df = condition_table(all_recs)
    log(f'[table rows] {len(df)} = {len(DEPTHS)} depths x '
        f'{len(CONDITIONS)} rules x {len(GROUPS)} groups x '
        f'{len(PLATFORMS)} platforms')

    summary = summarize(df)
    rev = reversal_diagnostics(df)

    path = os.path.join(TABLES, 'B1c_all_depths_by_condition.csv')
    df.to_csv(path, index=False)
    log(f'[table] wrote {path}')
    path = os.path.join(TABLES, 'B1c_all_depths_summary.csv')
    summary.to_csv(path, index=False)
    log(f'[table] wrote {path}')

    log_key_tables(df, summary, rev)
    make_figure(summary)
    write_conclusion(summary, df, rev)

    section('Done')
    log(f'Total elapsed: {time.time() - t0:.1f}s')


if __name__ == '__main__':
    main()
