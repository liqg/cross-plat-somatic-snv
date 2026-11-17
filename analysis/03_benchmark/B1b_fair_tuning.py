#!/usr/bin/env python3
"""B1b. Fair tuning controls for the baseline threshold sweep.

The B1 analysis reported two useful but optimistic quantities:
  * the per-sample "best" threshold, which is selected on the same test sample
    (oracle), and
  * a single global threshold chosen on all 12 test samples, which still uses
    labels from every sample that is later evaluated.

This script adds tuning protocols in which the evaluation-side labels play no
part in choosing the threshold:

  1. cross-group
       choose one global rule (absolute score cut, or top-fraction rule) on
       group 45 (6 samples), apply it unchanged to group 54, and vice versa.
       "Top-fraction" uses only the evaluation sample's score distribution,
       never its labels.

  2. leave-one-platform-out (LOPO)
       for each held-out platform, choose the rule on the other five platforms
       (10 samples, both groups) and apply it to the held-out platform.

  3. oracle upper
       per-sample best rule on that sample's own labels.  Reported only as an
       upper bound, clearly separated from the fair protocols.

Tuned union:
       Strelka2 and Mutect2 are each tuned on the training side, then their
       calls are unioned on the evaluation side.  Two fair variants are given:
       independent (each caller's rule chosen on its own F1) and joint (the
       QSS/TLOD pair chosen to maximise union F1 on the training side).  A
       per-sample oracle union is reported as an upper bound only.

SomaticXGB is put through the same cross-group, LOPO and oracle protocols so
that the comparison is like-for-like.  Its default is pred > 0.1.

Only new files are written; the B1 tables and figures are not modified.

Outputs (response/addendum-analyses/):
  tables/B1b_cross_group_tuning.csv
  tables/B1b_cross_group_eval_by_sample.csv
  tables/B1b_lopo_tuning.csv
  tables/B1b_lopo_eval_by_sample.csv
  tables/B1b_union_tuning.csv
  tables/B1b_union_oracle_by_sample.csv
  tables/B1b_fair_tuning_comparison.csv
  figures/B1b_fair_tuning_summary.png/.pdf
  figures/B1b_lopo_f1_by_platform.png/.pdf
  logs/B1b.log (written by the shell invocation)

Usage:
  /usr/bin/python3 scripts/B1b_fair_tuning.py
"""

import os
import sys

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

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

METHODS4 = ['Strelka2', 'Mutect2', 'VarScan2', 'SomaticXGB']
SUMMARY_METHODS = ['Strelka2', 'Mutect2', 'Strelka2_Mutect2_union',
                   'VarScan2', 'SomaticXGB']
METHOD_LABEL = {
    'Strelka2': 'Strelka2',
    'Mutect2': 'Mutect2',
    'Strelka2_Mutect2_union': 'Strelka2_Mutect2 union',
    'VarScan2': 'VarScan2',
    'SomaticXGB': 'SomaticXGB',
}

SCORE_LABEL = {
    'Strelka2': 'QSS',
    'Mutect2': 'TLOD',
    'VarScan2': 'SSC',
    'SomaticXGB': 'pred',
}

SOMATIC_DEFAULT_CUT = b1.SOMATIC_DEFAULT_CUT
PLAT_COLORS = b1.PLAT_COLORS

os.makedirs(TABLES, exist_ok=True)
os.makedirs(FIGURES, exist_ok=True)


def log(msg=''):
    b1.log(msg)


def section(title):
    b1.section(title)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_all_samples():
    """Parse all 12 samples once.  Returns a list of sample records."""
    section('Loading truth and parsing calls')
    truth_cache = {g: b1.load_truth(g) for g in GROUPS}
    recs = []

    for g in GROUPS:
        truth = truth_cache[g]['truth']
        near = truth_cache[g]['near']
        for pid, platform in PLATFORMS:
            sample = b1.sid(pid)
            rec = {
                'group': g,
                'platform': platform,
                'pid': pid,
                'sample': sample,
                'truth': truth,
                'near': near,
            }

            scores, st = b1.parse_vcf_scores(
                b1.path_strelka(g, pid), 'Strelka2', pass_only=True,
                near=near, verbose=False)
            rec['Strelka2'] = {
                'scores': scores, 'default': set(scores),
                'field': st['score_field'], 'n_calls': len(scores),
            }

            scores, st = b1.parse_vcf_scores(
                b1.path_mutect(g, pid), 'Mutect2', pass_only=True,
                near=near, verbose=False)
            rec['Mutect2'] = {
                'scores': scores, 'default': set(scores),
                'field': st['score_field'], 'n_calls': len(scores),
            }

            scores, st = b1.parse_vcf_scores(
                b1.path_varscan_pass(g, pid), 'VarScan2', pass_only=False,
                near=near, verbose=False)
            rec['VarScan2'] = {
                'scores': scores, 'default': set(scores),
                'field': st['score_field'], 'n_calls': len(scores),
            }

            scores, st = b1.parse_pred_scores(
                b1.path_pred(g, pid), near, verbose=False)
            rec['SomaticXGB'] = {
                'scores': scores,
                'default': {v for v, s in scores.items()
                            if s > SOMATIC_DEFAULT_CUT},
                'field': 'pred', 'n_calls': len(scores),
            }

            rec['Strelka2_Mutect2_union'] = {
                'default': (rec['Strelka2']['default'] |
                            rec['Mutect2']['default'])
            }
            recs.append(rec)

    log(f'Parsed {len(recs)} samples '
        f'({len(GROUPS)} groups x {len(PLATFORMS)} platforms).')
    return recs, truth_cache


def print_score_distributions(recs):
    section('Score distributions of the default call sets (12 samples)')
    rows = []
    for method in METHODS4:
        # Use the default call set (for SomaticXGB this is pred > 0.1, not
        # all candidates), so the distribution describes what is evaluated.
        vals = []
        for r in recs:
            sc = r[method]['scores']
            vals.append(np.array([sc[v] for v in r[method]['default']],
                                 dtype=float))
        rows.append({
            'Method': method,
            'ScoreField': SCORE_LABEL[method],
            'DefaultRule': (f'pred > {SOMATIC_DEFAULT_CUT}'
                            if method == 'SomaticXGB'
                            else 'all default calls'),
            'Calls_min': int(min(len(v) for v in vals)),
            'Calls_median': int(np.median([len(v) for v in vals])),
            'Calls_max': int(max(len(v) for v in vals)),
            'Score_min': float(min(v.min() for v in vals)),
            'Score_p10_median': float(np.median([np.quantile(v, 0.10)
                                                 for v in vals])),
            'Score_median_median': float(np.median([np.median(v)
                                                    for v in vals])),
            'Score_p90_median': float(np.median([np.quantile(v, 0.90)
                                                 for v in vals])),
            'Score_max': float(max(v.max() for v in vals)),
        })
    dist = pd.DataFrame(rows)
    log(dist.round(4).to_string(index=False))
    path = os.path.join(TABLES, 'B1b_score_distribution.csv')
    dist.to_csv(path, index=False)
    log(f'[table] wrote {path}')
    return dist


# ---------------------------------------------------------------------------
# Rules
# ---------------------------------------------------------------------------

def candidate_rules(method):
    """Default, every absolute cut, and every top-fraction rule."""
    rules = [{'type': 'default', 'value': None}]
    for cut in b1.ABS_CUTS[method]:
        rules.append({'type': 'absolute', 'value': float(cut)})
    for frac in b1.QUANTILE_FRACTIONS:
        rules.append({'type': 'quantile', 'value': float(frac)})
    return rules


def rule_str(rule, method):
    if rule['type'] == 'default':
        return 'default'
    if rule['type'] == 'absolute':
        return f"{SCORE_LABEL[method]}>={rule['value']:g}"
    return f"top {rule['value'] * 100:g}%"


def rule_priority(rule):
    return {'default': 2, 'absolute': 1, 'quantile': 0}[rule['type']]


def _rule_value(rule):
    v = rule.get('value')
    return float(v) if v is not None else -np.inf


def apply_rule(rec, method, rule):
    """Return the set of calls retained by a rule for one sample."""
    if rule['type'] == 'default':
        return set(rec[method]['default'])
    scores = rec[method]['scores']
    if not scores:
        return set()
    if rule['type'] == 'absolute':
        cut = rule['value']
        return {v for v, s in scores.items() if s >= cut}
    if rule['type'] == 'quantile':
        vals = np.fromiter(scores.values(), dtype=float, count=len(scores))
        cut = float(np.quantile(vals, 1.0 - rule['value']))
        return {v for v, s in scores.items() if s >= cut}
    raise ValueError(rule)


def eval_rule(rec, method, rule):
    calls = apply_rule(rec, method, rule)
    m = b1.compute_metrics(calls, rec['truth'])
    m['N_calls'] = len(calls)
    return m


def rule_changes_any(recs, method, rule):
    if rule['type'] == 'default':
        return False
    for r in recs:
        if apply_rule(r, method, rule) != r[method]['default']:
            return True
    return False


def select_rule(recs, method, rules, require_binding=False):
    """Choose the rule with the best mean F1 on the training samples.

    Ties prefer the default rule, then an absolute cut, then a higher value.
    Only training-side labels are used.
    """
    candidates = []
    for rule in rules:
        if require_binding and not rule_changes_any(recs, method, rule):
            continue
        ms = [eval_rule(r, method, rule) for r in recs]
        f1 = float(np.mean([m['F1'] for m in ms]))
        rec = float(np.mean([m['Recall'] for m in ms]))
        prec = float(np.mean([m['Precision'] for m in ms]))
        candidates.append({
            'rule': rule,
            'rule_str': rule_str(rule, method),
            'train_F1': f1,
            'train_Recall': rec,
            'train_Precision': prec,
            'per_sample': ms,
        })
    if not candidates:
        return None
    return max(candidates, key=lambda x: (
        x['train_F1'], x['train_Recall'], rule_priority(x['rule']),
        _rule_value(x['rule'])))


# ---------------------------------------------------------------------------
# Union helpers
# ---------------------------------------------------------------------------

def union_calls(rec, qss_rule, tlod_rule):
    return (apply_rule(rec, 'Strelka2', qss_rule) |
            apply_rule(rec, 'Mutect2', tlod_rule))


def union_eval(rec, qss_rule, tlod_rule):
    calls = union_calls(rec, qss_rule, tlod_rule)
    m = b1.compute_metrics(calls, rec['truth'])
    m['N_calls'] = len(calls)
    return m


def select_joint_pair(recs):
    """Choose (QSS cut, TLOD cut) maximising union F1 on training samples."""
    qcuts = b1.ABS_CUTS['Strelka2']
    tcuts = b1.ABS_CUTS['Mutect2']
    best = None
    best_key = None
    for qc in qcuts:
        qrule = {'type': 'absolute', 'value': float(qc)}
        for tc in tcuts:
            trule = {'type': 'absolute', 'value': float(tc)}
            ms = [union_eval(r, qrule, trule) for r in recs]
            f1 = float(np.mean([m['F1'] for m in ms]))
            rec = float(np.mean([m['Recall'] for m in ms]))
            prec = float(np.mean([m['Precision'] for m in ms]))
            key = (f1, rec, qc, tc)
            if best_key is None or key > best_key:
                best_key = key
                best = {
                    'qss_rule': qrule,
                    'tlod_rule': trule,
                    'rule_str': f'QSS>={qc:g} + TLOD>={tc:g}',
                    'train_F1': f1,
                    'train_Recall': rec,
                    'train_Precision': prec,
                    'per_sample': ms,
                }
    return best


def select_joint_pair_oracle(rec):
    """Per-sample oracle union pair (upper bound only)."""
    qcuts = b1.ABS_CUTS['Strelka2']
    tcuts = b1.ABS_CUTS['Mutect2']
    best = None
    best_key = None
    for qc in qcuts:
        qrule = {'type': 'absolute', 'value': float(qc)}
        for tc in tcuts:
            trule = {'type': 'absolute', 'value': float(tc)}
            m = union_eval(rec, qrule, trule)
            key = (m['F1'], m['Recall'], qc, tc)
            if best_key is None or key > best_key:
                best_key = key
                best = {
                    'qss_rule': qrule,
                    'tlod_rule': trule,
                    'rule_str': f'QSS>={qc:g} + TLOD>={tc:g}',
                    'metrics': m,
                }
    return best


# ---------------------------------------------------------------------------
# Cross-group protocol
# ---------------------------------------------------------------------------

def cross_group(recs):
    section('Cross-group tuning (train on one group, evaluate on the other)')
    tuning_rows = []
    sample_rows = []

    protocols = [
        (m, 'single') for m in METHODS4
    ] + [
        ('Strelka2_Mutect2_union', 'union_independent'),
        ('Strelka2_Mutect2_union', 'union_joint'),
    ]

    for method, kind in protocols:
        for train_g, eval_g in [('45', '54'), ('54', '45')]:
            train = [r for r in recs if r['group'] == train_g]
            ev = [r for r in recs if r['group'] == eval_g]
            fold = f'{train_g}->{eval_g}'

            if kind == 'single':
                rules = candidate_rules(method)
                best = select_rule(train, method, rules)
                best_binding = select_rule(
                    train, method,
                    [x for x in rules if x['type'] != 'default'],
                    require_binding=True)
                train_rule = best['rule_str']
                train_f1 = best['train_F1']
                train_rec = best['train_Recall']
                train_prec = best['train_Precision']
                eval_ms = [eval_rule(r, method, best['rule']) for r in ev]
                default_ms = [b1.compute_metrics(r[method]['default'],
                                                 r['truth']) for r in ev]
                for r, m, dm in zip(ev, eval_ms, default_ms):
                    sample_rows.append({
                        'Method': method,
                        'Protocol': 'cross-group',
                        'Fold': fold,
                        'TrainGroup': train_g,
                        'EvalGroup': eval_g,
                        'Group': r['group'],
                        'Platform': r['platform'],
                        'Sample': r['sample'],
                        'SelectedRule': train_rule,
                        'N_calls': m['N_calls'],
                        'TP': m['TP'], 'FP': m['FP'], 'FN': m['FN'],
                        'Recall': m['Recall'],
                        'Precision': m['Precision'],
                        'F1': m['F1'],
                        'Default_N_calls': len(r[method]['default']),
                        'Default_Recall': dm['Recall'],
                        'Default_Precision': dm['Precision'],
                        'Default_F1': dm['F1'],
                        'F1_Gain': m['F1'] - dm['F1'],
                    })
                tuning_rows.append({
                    'Method': method,
                    'Protocol': 'cross-group',
                    'Fold': fold,
                    'SelectedRule': train_rule,
                    'SelectedRule_Type': best['rule']['type'],
                    'SelectedRule_Value': best['rule']['value'],
                    'BestBindingRule': (best_binding['rule_str']
                                        if best_binding else ''),
                    'BestBindingRule_TrainF1': (best_binding['train_F1']
                                                if best_binding else np.nan),
                    'BestBindingRule_EvalF1': (
                        float(np.mean([eval_rule(r, method,
                                                 best_binding['rule'])['F1']
                                       for r in ev]))
                        if best_binding else np.nan),
                    'Train_Mean_F1': train_f1,
                    'Train_Mean_Recall': train_rec,
                    'Train_Mean_Precision': train_prec,
                    'Eval_Mean_F1': float(np.mean([m['F1'] for m in eval_ms])),
                    'Eval_Mean_Recall': float(np.mean([m['Recall'] for m in eval_ms])),
                    'Eval_Mean_Precision': float(np.mean([m['Precision'] for m in eval_ms])),
                    'Eval_Default_Mean_F1': float(np.mean([m['F1'] for m in default_ms])),
                    'Eval_Mean_F1_Gain': (float(np.mean([m['F1'] for m in eval_ms]))
                                          - float(np.mean([m['F1'] for m in default_ms]))),
                    'N_Eval': len(ev),
                })

            elif kind == 'union_independent':
                bs = select_rule(train, 'Strelka2', candidate_rules('Strelka2'))
                bm = select_rule(train, 'Mutect2', candidate_rules('Mutect2'))
                train_rule = f"{bs['rule_str']} | {bm['rule_str']}"
                eval_ms = [union_eval(r, bs['rule'], bm['rule']) for r in ev]
                default_ms = [b1.compute_metrics(
                    r['Strelka2_Mutect2_union']['default'], r['truth'])
                    for r in ev]
                for r, m, dm in zip(ev, eval_ms, default_ms):
                    sample_rows.append({
                        'Method': 'Strelka2_Mutect2_union',
                        'Protocol': 'cross-group (independent)',
                        'Fold': fold,
                        'TrainGroup': train_g,
                        'EvalGroup': eval_g,
                        'Group': r['group'],
                        'Platform': r['platform'],
                        'Sample': r['sample'],
                        'SelectedRule': train_rule,
                        'N_calls': m['N_calls'],
                        'TP': m['TP'], 'FP': m['FP'], 'FN': m['FN'],
                        'Recall': m['Recall'],
                        'Precision': m['Precision'],
                        'F1': m['F1'],
                        'Default_N_calls': len(r['Strelka2_Mutect2_union']['default']),
                        'Default_Recall': dm['Recall'],
                        'Default_Precision': dm['Precision'],
                        'Default_F1': dm['F1'],
                        'F1_Gain': m['F1'] - dm['F1'],
                    })
                tuning_rows.append({
                    'Method': 'Strelka2_Mutect2_union',
                    'Protocol': 'cross-group (independent)',
                    'Fold': fold,
                    'SelectedRule': train_rule,
                    'SelectedRule_Type': 'independent',
                    'SelectedRule_Value': np.nan,
                    'BestBindingRule': train_rule,
                    'BestBindingRule_TrainF1': (
                        float(np.mean([union_eval(r, bs['rule'], bm['rule'])['F1']
                                       for r in train]))),
                    'BestBindingRule_EvalF1': float(np.mean([m['F1'] for m in eval_ms])),
                    'Train_Mean_F1': float(np.mean(
                        [union_eval(r, bs['rule'], bm['rule'])['F1']
                         for r in train])),
                    'Train_Mean_Recall': float(np.mean(
                        [union_eval(r, bs['rule'], bm['rule'])['Recall']
                         for r in train])),
                    'Train_Mean_Precision': float(np.mean(
                        [union_eval(r, bs['rule'], bm['rule'])['Precision']
                         for r in train])),
                    'Eval_Mean_F1': float(np.mean([m['F1'] for m in eval_ms])),
                    'Eval_Mean_Recall': float(np.mean([m['Recall'] for m in eval_ms])),
                    'Eval_Mean_Precision': float(np.mean([m['Precision'] for m in eval_ms])),
                    'Eval_Default_Mean_F1': float(np.mean([m['F1'] for m in default_ms])),
                    'Eval_Mean_F1_Gain': (float(np.mean([m['F1'] for m in eval_ms]))
                                          - float(np.mean([m['F1'] for m in default_ms]))),
                    'N_Eval': len(ev),
                })

            elif kind == 'union_joint':
                bp = select_joint_pair(train)
                eval_ms = [union_eval(r, bp['qss_rule'], bp['tlod_rule'])
                           for r in ev]
                default_ms = [b1.compute_metrics(
                    r['Strelka2_Mutect2_union']['default'], r['truth'])
                    for r in ev]
                for r, m, dm in zip(ev, eval_ms, default_ms):
                    sample_rows.append({
                        'Method': 'Strelka2_Mutect2_union',
                        'Protocol': 'cross-group (joint)',
                        'Fold': fold,
                        'TrainGroup': train_g,
                        'EvalGroup': eval_g,
                        'Group': r['group'],
                        'Platform': r['platform'],
                        'Sample': r['sample'],
                        'SelectedRule': bp['rule_str'],
                        'N_calls': m['N_calls'],
                        'TP': m['TP'], 'FP': m['FP'], 'FN': m['FN'],
                        'Recall': m['Recall'],
                        'Precision': m['Precision'],
                        'F1': m['F1'],
                        'Default_N_calls': len(r['Strelka2_Mutect2_union']['default']),
                        'Default_Recall': dm['Recall'],
                        'Default_Precision': dm['Precision'],
                        'Default_F1': dm['F1'],
                        'F1_Gain': m['F1'] - dm['F1'],
                    })
                tuning_rows.append({
                    'Method': 'Strelka2_Mutect2_union',
                    'Protocol': 'cross-group (joint)',
                    'Fold': fold,
                    'SelectedRule': bp['rule_str'],
                    'SelectedRule_Type': 'joint',
                    'SelectedRule_Value': np.nan,
                    'BestBindingRule': bp['rule_str'],
                    'BestBindingRule_TrainF1': bp['train_F1'],
                    'BestBindingRule_EvalF1': float(np.mean([m['F1'] for m in eval_ms])),
                    'Train_Mean_F1': bp['train_F1'],
                    'Train_Mean_Recall': bp['train_Recall'],
                    'Train_Mean_Precision': bp['train_Precision'],
                    'Eval_Mean_F1': float(np.mean([m['F1'] for m in eval_ms])),
                    'Eval_Mean_Recall': float(np.mean([m['Recall'] for m in eval_ms])),
                    'Eval_Mean_Precision': float(np.mean([m['Precision'] for m in eval_ms])),
                    'Eval_Default_Mean_F1': float(np.mean([m['F1'] for m in default_ms])),
                    'Eval_Mean_F1_Gain': (float(np.mean([m['F1'] for m in eval_ms]))
                                          - float(np.mean([m['F1'] for m in default_ms]))),
                    'N_Eval': len(ev),
                })

    return pd.DataFrame(tuning_rows), pd.DataFrame(sample_rows)


# ---------------------------------------------------------------------------
# Leave-one-platform-out protocol
# ---------------------------------------------------------------------------

def lopo(recs):
    section('Leave-one-platform-out tuning (train on the other five platforms)')
    tuning_rows = []
    sample_rows = []

    protocols = [
        (m, 'single') for m in METHODS4
    ] + [
        ('Strelka2_Mutect2_union', 'union_independent'),
        ('Strelka2_Mutect2_union', 'union_joint'),
    ]

    for method, kind in protocols:
        for held_pid, held_platform in PLATFORMS:
            train = [r for r in recs if r['platform'] != held_platform]
            ev = [r for r in recs if r['platform'] == held_platform]
            fold = held_platform

            if kind == 'single':
                rules = candidate_rules(method)
                best = select_rule(train, method, rules)
                eval_ms = [eval_rule(r, method, best['rule']) for r in ev]
                default_ms = [b1.compute_metrics(r[method]['default'],
                                                 r['truth']) for r in ev]
                for r, m, dm in zip(ev, eval_ms, default_ms):
                    sample_rows.append({
                        'Method': method,
                        'Protocol': 'LOPO',
                        'HeldOutPlatform': held_platform,
                        'Group': r['group'],
                        'Platform': r['platform'],
                        'Sample': r['sample'],
                        'SelectedRule': best['rule_str'],
                        'N_calls': m['N_calls'],
                        'TP': m['TP'], 'FP': m['FP'], 'FN': m['FN'],
                        'Recall': m['Recall'],
                        'Precision': m['Precision'],
                        'F1': m['F1'],
                        'Default_N_calls': len(r[method]['default']),
                        'Default_Recall': dm['Recall'],
                        'Default_Precision': dm['Precision'],
                        'Default_F1': dm['F1'],
                        'F1_Gain': m['F1'] - dm['F1'],
                    })
                tuning_rows.append({
                    'Method': method,
                    'Protocol': 'LOPO',
                    'HeldOutPlatform': held_platform,
                    'SelectedRule': best['rule_str'],
                    'SelectedRule_Type': best['rule']['type'],
                    'SelectedRule_Value': best['rule']['value'],
                    'Train_Mean_F1': best['train_F1'],
                    'Train_Mean_Recall': best['train_Recall'],
                    'Train_Mean_Precision': best['train_Precision'],
                    'Eval_Mean_F1': float(np.mean([m['F1'] for m in eval_ms])),
                    'Eval_Mean_Recall': float(np.mean([m['Recall'] for m in eval_ms])),
                    'Eval_Mean_Precision': float(np.mean([m['Precision'] for m in eval_ms])),
                    'Eval_Default_Mean_F1': float(np.mean([m['F1'] for m in default_ms])),
                    'Eval_Mean_F1_Gain': (float(np.mean([m['F1'] for m in eval_ms]))
                                          - float(np.mean([m['F1'] for m in default_ms]))),
                    'N_Eval': len(ev),
                })

            elif kind == 'union_independent':
                bs = select_rule(train, 'Strelka2', candidate_rules('Strelka2'))
                bm = select_rule(train, 'Mutect2', candidate_rules('Mutect2'))
                train_rule = f"{bs['rule_str']} | {bm['rule_str']}"
                eval_ms = [union_eval(r, bs['rule'], bm['rule']) for r in ev]
                default_ms = [b1.compute_metrics(
                    r['Strelka2_Mutect2_union']['default'], r['truth'])
                    for r in ev]
                for r, m, dm in zip(ev, eval_ms, default_ms):
                    sample_rows.append({
                        'Method': 'Strelka2_Mutect2_union',
                        'Protocol': 'LOPO (independent)',
                        'HeldOutPlatform': held_platform,
                        'Group': r['group'],
                        'Platform': r['platform'],
                        'Sample': r['sample'],
                        'SelectedRule': train_rule,
                        'N_calls': m['N_calls'],
                        'TP': m['TP'], 'FP': m['FP'], 'FN': m['FN'],
                        'Recall': m['Recall'],
                        'Precision': m['Precision'],
                        'F1': m['F1'],
                        'Default_N_calls': len(r['Strelka2_Mutect2_union']['default']),
                        'Default_Recall': dm['Recall'],
                        'Default_Precision': dm['Precision'],
                        'Default_F1': dm['F1'],
                        'F1_Gain': m['F1'] - dm['F1'],
                    })
                tuning_rows.append({
                    'Method': 'Strelka2_Mutect2_union',
                    'Protocol': 'LOPO (independent)',
                    'HeldOutPlatform': held_platform,
                    'SelectedRule': train_rule,
                    'SelectedRule_Type': 'independent',
                    'SelectedRule_Value': np.nan,
                    'Train_Mean_F1': float(np.mean(
                        [union_eval(r, bs['rule'], bm['rule'])['F1']
                         for r in train])),
                    'Train_Mean_Recall': float(np.mean(
                        [union_eval(r, bs['rule'], bm['rule'])['Recall']
                         for r in train])),
                    'Train_Mean_Precision': float(np.mean(
                        [union_eval(r, bs['rule'], bm['rule'])['Precision']
                         for r in train])),
                    'Eval_Mean_F1': float(np.mean([m['F1'] for m in eval_ms])),
                    'Eval_Mean_Recall': float(np.mean([m['Recall'] for m in eval_ms])),
                    'Eval_Mean_Precision': float(np.mean([m['Precision'] for m in eval_ms])),
                    'Eval_Default_Mean_F1': float(np.mean([m['F1'] for m in default_ms])),
                    'Eval_Mean_F1_Gain': (float(np.mean([m['F1'] for m in eval_ms]))
                                          - float(np.mean([m['F1'] for m in default_ms]))),
                    'N_Eval': len(ev),
                })

            elif kind == 'union_joint':
                bp = select_joint_pair(train)
                eval_ms = [union_eval(r, bp['qss_rule'], bp['tlod_rule'])
                           for r in ev]
                default_ms = [b1.compute_metrics(
                    r['Strelka2_Mutect2_union']['default'], r['truth'])
                    for r in ev]
                for r, m, dm in zip(ev, eval_ms, default_ms):
                    sample_rows.append({
                        'Method': 'Strelka2_Mutect2_union',
                        'Protocol': 'LOPO (joint)',
                        'HeldOutPlatform': held_platform,
                        'Group': r['group'],
                        'Platform': r['platform'],
                        'Sample': r['sample'],
                        'SelectedRule': bp['rule_str'],
                        'N_calls': m['N_calls'],
                        'TP': m['TP'], 'FP': m['FP'], 'FN': m['FN'],
                        'Recall': m['Recall'],
                        'Precision': m['Precision'],
                        'F1': m['F1'],
                        'Default_N_calls': len(r['Strelka2_Mutect2_union']['default']),
                        'Default_Recall': dm['Recall'],
                        'Default_Precision': dm['Precision'],
                        'Default_F1': dm['F1'],
                        'F1_Gain': m['F1'] - dm['F1'],
                    })
                tuning_rows.append({
                    'Method': 'Strelka2_Mutect2_union',
                    'Protocol': 'LOPO (joint)',
                    'HeldOutPlatform': held_platform,
                    'SelectedRule': bp['rule_str'],
                    'SelectedRule_Type': 'joint',
                    'SelectedRule_Value': np.nan,
                    'Train_Mean_F1': bp['train_F1'],
                    'Train_Mean_Recall': bp['train_Recall'],
                    'Train_Mean_Precision': bp['train_Precision'],
                    'Eval_Mean_F1': float(np.mean([m['F1'] for m in eval_ms])),
                    'Eval_Mean_Recall': float(np.mean([m['Recall'] for m in eval_ms])),
                    'Eval_Mean_Precision': float(np.mean([m['Precision'] for m in eval_ms])),
                    'Eval_Default_Mean_F1': float(np.mean([m['F1'] for m in default_ms])),
                    'Eval_Mean_F1_Gain': (float(np.mean([m['F1'] for m in eval_ms]))
                                          - float(np.mean([m['F1'] for m in default_ms]))),
                    'N_Eval': len(ev),
                })

    return pd.DataFrame(tuning_rows), pd.DataFrame(sample_rows)


# ---------------------------------------------------------------------------
# Default and oracle rows
# ---------------------------------------------------------------------------

def default_rows(recs):
    rows = []
    for method in METHODS4:
        for r in recs:
            m = b1.compute_metrics(r[method]['default'], r['truth'])
            rows.append({
                'Method': method,
                'Protocol': 'default',
                'Group': r['group'],
                'Platform': r['platform'],
                'Sample': r['sample'],
                'N_calls': len(r[method]['default']),
                'TP': m['TP'], 'FP': m['FP'], 'FN': m['FN'],
                'Recall': m['Recall'],
                'Precision': m['Precision'],
                'F1': m['F1'],
            })
    for r in recs:
        m = b1.compute_metrics(r['Strelka2_Mutect2_union']['default'],
                               r['truth'])
        rows.append({
            'Method': 'Strelka2_Mutect2_union',
            'Protocol': 'default',
            'Group': r['group'],
            'Platform': r['platform'],
            'Sample': r['sample'],
            'N_calls': len(r['Strelka2_Mutect2_union']['default']),
            'TP': m['TP'], 'FP': m['FP'], 'FN': m['FN'],
            'Recall': m['Recall'],
            'Precision': m['Precision'],
            'F1': m['F1'],
        })
    return pd.DataFrame(rows)


def oracle_rows(recs):
    section('Oracle upper bounds (per-sample labels; upper bound only)')
    rows = []
    for method in METHODS4:
        rules = candidate_rules(method)
        for r in recs:
            best = select_rule([r], method, rules)
            m = best['per_sample'][0]
            rows.append({
                'Method': method,
                'Protocol': 'oracle (per-sample)',
                'Group': r['group'],
                'Platform': r['platform'],
                'Sample': r['sample'],
                'SelectedRule': best['rule_str'],
                'N_calls': m['N_calls'],
                'TP': m['TP'], 'FP': m['FP'], 'FN': m['FN'],
                'Recall': m['Recall'],
                'Precision': m['Precision'],
                'F1': m['F1'],
            })

    # Per-sample oracle union over the joint QSS/TLOD grid.
    for r in recs:
        b = select_joint_pair_oracle(r)
        m = b['metrics']
        rows.append({
            'Method': 'Strelka2_Mutect2_union',
            'Protocol': 'oracle (per-sample)',
            'Group': r['group'],
            'Platform': r['platform'],
            'Sample': r['sample'],
            'SelectedRule': b['rule_str'],
            'N_calls': m['N_calls'],
            'TP': m['TP'], 'FP': m['FP'], 'FN': m['FN'],
            'Recall': m['Recall'],
            'Precision': m['Precision'],
            'F1': m['F1'],
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def summarize_group(df, method, tuning, protocol):
    sub = df[(df['Method'] == method) & (df['Protocol'] == protocol)]
    if sub.empty:
        return None
    return {
        'Method': method,
        'Tuning': tuning,
        'Protocol': protocol,
        'N_samples': len(sub),
        'Mean_F1': sub['F1'].mean(),
        'SD_F1': sub['F1'].std(ddof=1),
        'Mean_Recall': sub['Recall'].mean(),
        'Mean_Precision': sub['Precision'].mean(),
    }


def build_summary(default_df, cross_sample_df, lopo_sample_df, oracle_df):
    """One row per method x tuning protocol, plus SomaticXGB reference columns."""
    rows = []

    # Defaults and oracle.
    for method in SUMMARY_METHODS:
        d = default_df[default_df['Method'] == method]
        o = oracle_df[oracle_df['Method'] == method]
        rows.append({
            'Method': method, 'Tuning': 'default', 'Protocol': 'none',
            'N_samples': len(d),
            'Mean_F1': d['F1'].mean(), 'SD_F1': d['F1'].std(ddof=1),
            'Mean_Recall': d['Recall'].mean(),
            'Mean_Precision': d['Precision'].mean(),
            'Selected_Rule_Summary': 'default',
        })
        rows.append({
            'Method': method, 'Tuning': 'oracle-upper',
            'Protocol': 'per-sample oracle',
            'N_samples': len(o),
            'Mean_F1': o['F1'].mean(), 'SD_F1': o['F1'].std(ddof=1),
            'Mean_Recall': o['Recall'].mean(),
            'Mean_Precision': o['Precision'].mean(),
            'Selected_Rule_Summary': 'per-sample best',
        })

    # Cross-group fair tuning, per method and union variant.
    cross_protocols = {
        'Strelka2': ['cross-group'],
        'Mutect2': ['cross-group'],
        'VarScan2': ['cross-group'],
        'SomaticXGB': ['cross-group'],
        'Strelka2_Mutect2_union': ['cross-group (independent)',
                                   'cross-group (joint)'],
    }
    for method, protocols in cross_protocols.items():
        for protocol in protocols:
            s = summarize_group(cross_sample_df, method, 'fair-tuned', protocol)
            if s is None:
                continue
            # Rule summary: collapse the two directions.
            rules = (cross_sample_df[(cross_sample_df['Method'] == method) &
                                     (cross_sample_df['Protocol'] == protocol)]
                     [['Fold', 'SelectedRule']]
                     .drop_duplicates().sort_values('Fold'))
            rule_str = '; '.join(f'{f}: {r}' for f, r in
                                 zip(rules['Fold'], rules['SelectedRule']))
            s['Selected_Rule_Summary'] = rule_str
            rows.append(s)

    # LOPO fair tuning.
    lopo_protocols = {
        'Strelka2': ['LOPO'],
        'Mutect2': ['LOPO'],
        'VarScan2': ['LOPO'],
        'SomaticXGB': ['LOPO'],
        'Strelka2_Mutect2_union': ['LOPO (independent)', 'LOPO (joint)'],
    }
    for method, protocols in lopo_protocols.items():
        for protocol in protocols:
            s = summarize_group(lopo_sample_df, method, 'fair-tuned', protocol)
            if s is None:
                continue
            rules = (lopo_sample_df[(lopo_sample_df['Method'] == method) &
                                    (lopo_sample_df['Protocol'] == protocol)]
                     [['HeldOutPlatform', 'SelectedRule']]
                     .drop_duplicates().sort_values('HeldOutPlatform'))
            rule_str = '; '.join(f'{p}: {r}' for p, r in
                                 zip(rules['HeldOutPlatform'],
                                     rules['SelectedRule']))
            s['Selected_Rule_Summary'] = rule_str
            rows.append(s)

    summary = pd.DataFrame(rows)

    # SomaticXGB reference values for each protocol.
    ref = {}
    for key, df, protocol in [
        ('default', default_df, None),
        ('oracle', oracle_df, 'oracle (per-sample)'),
        ('cross', cross_sample_df, 'cross-group'),
        ('lopo', lopo_sample_df, 'LOPO'),
    ]:
        sub = df[df['Method'] == 'SomaticXGB']
        if protocol is not None:
            sub = sub[sub['Protocol'] == protocol]
        ref[key] = float(sub['F1'].mean()) if not sub.empty else np.nan
    summary['SomaticXGB_Default_Mean_F1'] = ref['default']
    summary['SomaticXGB_CrossGroup_Mean_F1'] = ref['cross']
    summary['SomaticXGB_LOPO_Mean_F1'] = ref['lopo']
    summary['SomaticXGB_Oracle_Mean_F1'] = ref['oracle']

    # Own default reference.
    own_default = (summary[summary['Tuning'] == 'default']
                   .set_index('Method')['Mean_F1'].to_dict())
    summary['Own_Default_Mean_F1'] = summary['Method'].map(own_default)
    summary['F1_Gain_vs_Own_Default'] = (summary['Mean_F1'] -
                                         summary['Own_Default_Mean_F1'])
    summary['Delta_vs_SomaticXGB_default'] = (summary['Mean_F1'] -
                                              ref['default'])

    # Same-protocol SomaticXGB reference.
    same_protocol = []
    for _, row in summary.iterrows():
        p = row['Protocol']
        if p in ('none',):
            v = ref['default']
        elif p.startswith('per-sample oracle'):
            v = ref['oracle']
        elif p.startswith('cross-group'):
            v = ref['cross']
        elif p.startswith('LOPO'):
            v = ref['lopo']
        else:
            v = np.nan
        same_protocol.append(v)
    summary['SomaticXGB_SameProtocol_Mean_F1'] = same_protocol
    summary['Delta_vs_SomaticXGB_same_protocol'] = (
        summary['Mean_F1'] - summary['SomaticXGB_SameProtocol_Mean_F1'])
    summary['Still_Below_SomaticXGB_default'] = (
        summary['Mean_F1'] < ref['default'])
    summary['Still_Below_SomaticXGB_same_protocol'] = (
        summary['Mean_F1'] < summary['SomaticXGB_SameProtocol_Mean_F1'])

    # Order: method as requested, tuning in a logical order.
    method_order = {m: i for i, m in enumerate(SUMMARY_METHODS)}
    tuning_order = {'default': 0, 'fair-tuned': 1, 'oracle-upper': 2}
    protocol_order = {
        'none': 0,
        'cross-group': 1,
        'cross-group (independent)': 2,
        'cross-group (joint)': 3,
        'LOPO': 4,
        'LOPO (independent)': 5,
        'LOPO (joint)': 6,
        'per-sample oracle': 7,
    }
    summary['_mo'] = summary['Method'].map(method_order)
    summary['_to'] = summary['Tuning'].map(tuning_order)
    summary['_po'] = summary['Protocol'].map(protocol_order).fillna(99)
    summary = summary.sort_values(['_mo', '_to', '_po']).drop(
        columns=['_mo', '_to', '_po']).reset_index(drop=True)
    summary.insert(0, 'Method_Label', summary['Method'].map(METHOD_LABEL))
    return summary


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def make_summary_figure(summary_df):
    methods = SUMMARY_METHODS
    labels = ['Strelka2', 'Mutect2', 'Strelka2 +\nMutect2 union',
              'VarScan2', 'SomaticXGB']
    x = np.arange(len(methods))
    width = 0.20

    def get(method, protocol, col='Mean_F1'):
        r = summary_df[(summary_df['Method'] == method) &
                       (summary_df['Protocol'] == protocol)]
        if r.empty:
            return np.nan
        return float(r[col].iloc[0])

    # For the union use the joint fair-tuned variants in the bars.
    protocols = [
        ('default', 'none', 'default', '#7f7f7f'),
        ('cross-group', 'cross-group', 'fair-tuned, cross-group', '#1f77b4'),
        ('LOPO', 'LOPO', 'fair-tuned, LOPO', '#2ca02c'),
        ('oracle', 'per-sample oracle', 'oracle upper bound', '#ff7f0e'),
    ]
    # Map union to the joint protocol rows.
    def protocol_for(method, key):
        if method == 'Strelka2_Mutect2_union' and key == 'cross-group':
            return 'cross-group (joint)'
        if method == 'Strelka2_Mutect2_union' and key == 'LOPO':
            return 'LOPO (joint)'
        return key

    fig, ax = plt.subplots(figsize=(11, 5.5))
    for k, (key, proto, label, color) in enumerate(protocols):
        vals = [get(m, protocol_for(m, proto), 'Mean_F1') for m in methods]
        sds = [get(m, protocol_for(m, proto), 'SD_F1') for m in methods]
        pos = x + (k - 1.5) * width
        ax.bar(pos, vals, width, label=label, color=color, edgecolor='k',
               lw=0.5, yerr=sds, capsize=2.5,
               error_kw={'elinewidth': 0.8, 'ecolor': '#333333'})
        # Put the value inside the bar so that it cannot collide with the
        # error bar or with the next bar's label.
        for xi, v in zip(pos, vals):
            if np.isfinite(v):
                ax.text(xi, max(v - 0.025, 0.02), f'{v:.3f}', ha='center',
                        va='top', fontsize=7.5, rotation=0,
                        color='white' if k in (0, 1, 2) else 'black')
    # SomaticXGB default reference line.
    som_default = get('SomaticXGB', 'none', 'Mean_F1')
    ax.axhline(som_default, color='k', ls=':', lw=1.3,
               label=f'SomaticXGB default ({som_default:.3f})')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.10)
    ax.set_ylabel('Mean F1 across 12 samples')
    ax.set_title('B1b. Fair threshold tuning: training-side rule selection\n'
                 '(bars = mean, error bars = SD across samples)',
                 fontsize=12, fontweight='bold')
    ax.grid(axis='y', alpha=0.3, lw=0.6)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.0), ncol=5,
              fontsize=7.5, framealpha=0.95)
    fig.text(0.5, 0.005,
             'Cross-group: rule chosen on the other group. LOPO: rule chosen '
             'on the other five platforms. Oracle: per-sample best, shown as '
             'an upper bound only. Union uses the joint QSS/TLOD pair for '
             'cross-group and LOPO.',
             ha='center', va='bottom', fontsize=7.5, color='#333333')
    png = os.path.join(FIGURES, 'B1b_fair_tuning_summary.png')
    pdf = os.path.join(FIGURES, 'B1b_fair_tuning_summary.pdf')
    fig.savefig(png, dpi=300, bbox_inches='tight')
    fig.savefig(pdf, bbox_inches='tight')
    plt.close(fig)
    log(f'[figure] wrote {png}')
    log(f'[figure] wrote {pdf}')


def make_lopo_heatmap(lopo_sample_df, default_df):
    """Three heatmaps: default F1, LOPO-tuned F1, and gain."""
    methods = SUMMARY_METHODS
    method_labels = ['Strelka2', 'Mutect2', 'Strelka2 +\nMutect2 union',
                     'VarScan2', 'SomaticXGB']

    def value(method, platform, column, tuned):
        if tuned:
            sub = lopo_sample_df[(lopo_sample_df['Method'] == method) &
                                 (lopo_sample_df['Platform'] == platform)]
            if method == 'Strelka2_Mutect2_union':
                sub = sub[sub['Protocol'] == 'LOPO (joint)']
            else:
                sub = sub[sub['Protocol'] == 'LOPO']
        else:
            sub = default_df[(default_df['Method'] == method) &
                             (default_df['Platform'] == platform)]
        if sub.empty:
            return np.nan
        return float(sub[column].mean())

    mats = []
    for tuned, column in [(False, 'F1'), (True, 'F1'), (True, 'F1_Gain')]:
        mat = np.array([[value(m, p, column, tuned) for p in PLATFORM_NAMES]
                        for m in methods], dtype=float)
        mats.append(mat)

    fig, axes = plt.subplots(1, 3, figsize=(19, 4.6))
    titles = ['Default F1', 'LOPO-tuned F1', 'F1 gain vs default']
    for ax, mat, title in zip(axes, mats, titles):
        if title.startswith('F1 gain'):
            vmax = max(0.02, float(np.nanmax(np.abs(mat))))
            norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
            im = ax.imshow(mat, aspect='auto', cmap='RdBu_r', norm=norm)
        else:
            im = ax.imshow(mat, aspect='auto', cmap='viridis',
                           vmin=0, vmax=1)
        ax.set_xticks(np.arange(len(PLATFORM_NAMES)))
        ax.set_xticklabels(PLATFORM_NAMES, rotation=45, ha='right',
                           fontsize=9)
        ax.set_yticks(np.arange(len(methods)))
        ax.set_yticklabels(method_labels, fontsize=9)
        ax.set_title(title, fontsize=11, fontweight='bold')
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                v = mat[i, j]
                if np.isfinite(v):
                    if title.startswith('F1 gain'):
                        # Near-zero gains sit on a white background; use black
                        # text there and white text only on saturated cells.
                        text_color = ('white' if abs(v) >
                                      0.55 * np.nanmax(np.abs(mat))
                                      else 'black')
                    else:
                        text_color = 'white' if v < 0.55 else 'black'
                    ax.text(j, i, f'{v:.3f}', ha='center', va='center',
                            fontsize=8, color=text_color)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    fig.suptitle('B1b. Leave-one-platform-out tuning (mean of the two groups)',
                 fontsize=13, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    png = os.path.join(FIGURES, 'B1b_lopo_f1_by_platform.png')
    pdf = os.path.join(FIGURES, 'B1b_lopo_f1_by_platform.pdf')
    fig.savefig(png, dpi=300, bbox_inches='tight')
    fig.savefig(pdf, bbox_inches='tight')
    plt.close(fig)
    log(f'[figure] wrote {png}')
    log(f'[figure] wrote {pdf}')


# ---------------------------------------------------------------------------
# Conclusions
# ---------------------------------------------------------------------------

def print_conclusions(summary_df, cross_tune, lopo_tune, default_df,
                      oracle_df):
    section('B1b CONCLUSIONS')

    # SomaticXGB references.
    som = summary_df[(summary_df['Method'] == 'SomaticXGB') &
                     (summary_df['Tuning'] == 'default')].iloc[0]
    som_cross = summary_df[(summary_df['Method'] == 'SomaticXGB') &
                           (summary_df['Protocol'] == 'cross-group')].iloc[0]
    som_lopo = summary_df[(summary_df['Method'] == 'SomaticXGB') &
                          (summary_df['Protocol'] == 'LOPO')].iloc[0]
    log(f'SomaticXGB reference: default F1 {som["Mean_F1"]:.4f}, '
        f'cross-group fair-tuned {som_cross["Mean_F1"]:.4f}, '
        f'LOPO fair-tuned {som_lopo["Mean_F1"]:.4f}.')

    # VarScan2 before/after, fair protocols.
    vs_default = summary_df[(summary_df['Method'] == 'VarScan2') &
                            (summary_df['Tuning'] == 'default')].iloc[0]
    vs_cross = summary_df[(summary_df['Method'] == 'VarScan2') &
                          (summary_df['Protocol'] == 'cross-group')].iloc[0]
    vs_lopo = summary_df[(summary_df['Method'] == 'VarScan2') &
                         (summary_df['Protocol'] == 'LOPO')].iloc[0]
    log('')
    log('VarScan2 SSC tuning:')
    log(f'  default: F1 {vs_default["Mean_F1"]:.4f}, '
        f'recall {vs_default["Mean_Recall"]:.4f}, '
        f'precision {vs_default["Mean_Precision"]:.4f}')
    log(f'  cross-group fair-tuned: F1 {vs_cross["Mean_F1"]:.4f}, '
        f'recall {vs_cross["Mean_Recall"]:.4f}, '
        f'precision {vs_cross["Mean_Precision"]:.4f} '
        f'(gain {vs_cross["F1_Gain_vs_Own_Default"]:+.4f})')
    log(f'  LOPO fair-tuned: F1 {vs_lopo["Mean_F1"]:.4f}, '
        f'recall {vs_lopo["Mean_Recall"]:.4f}, '
        f'precision {vs_lopo["Mean_Precision"]:.4f} '
        f'(gain {vs_lopo["F1_Gain_vs_Own_Default"]:+.4f})')
    log(f'  gap to SomaticXGB default: default {vs_default["Delta_vs_SomaticXGB_default"]:+.4f}, '
        f'cross-group {vs_cross["Delta_vs_SomaticXGB_default"]:+.4f}, '
        f'LOPO {vs_lopo["Delta_vs_SomaticXGB_default"]:+.4f}')

    # Union fair-tuned.
    log('')
    log('Tuned Strelka2 + Mutect2 union (each caller tuned on training side):')
    for proto in ['cross-group (independent)', 'cross-group (joint)',
                  'LOPO (independent)', 'LOPO (joint)']:
        r = summary_df[(summary_df['Method'] == 'Strelka2_Mutect2_union') &
                       (summary_df['Protocol'] == proto)]
        if not r.empty:
            r = r.iloc[0]
            log(f'  {proto}: F1 {r["Mean_F1"]:.4f}, '
                f'recall {r["Mean_Recall"]:.4f}, '
                f'precision {r["Mean_Precision"]:.4f}, '
                f'delta vs SomaticXGB default '
                f'{r["Delta_vs_SomaticXGB_default"]:+.4f}')

    log('')
    log('Oracle upper bounds (per-sample labels; not deployable):')
    for method in SUMMARY_METHODS:
        r = summary_df[(summary_df['Method'] == method) &
                       (summary_df['Tuning'] == 'oracle-upper')]
        if not r.empty:
            r = r.iloc[0]
            log(f'  {method}: F1 {r["Mean_F1"]:.4f}')

    # Ranking under fair tuning.
    section('B1b ranking under fair tuning')
    for proto, label in [
        ('cross-group', 'cross-group'),
        ('LOPO', 'LOPO'),
    ]:
        rows = []
        for method in ['Strelka2', 'Mutect2', 'VarScan2', 'SomaticXGB']:
            r = summary_df[(summary_df['Method'] == method) &
                           (summary_df['Protocol'] == proto)]
            if not r.empty:
                rows.append((method, float(r['Mean_F1'].iloc[0])))
        rows.sort(key=lambda x: x[1], reverse=True)
        log(f'  {label}: ' + ' > '.join(f'{m} ({v:.3f})' for m, v in rows))
    # Union rows.
    for proto in ['cross-group (independent)', 'cross-group (joint)',
                  'LOPO (independent)', 'LOPO (joint)']:
        r = summary_df[(summary_df['Method'] == 'Strelka2_Mutect2_union') &
                       (summary_df['Protocol'] == proto)]
        if not r.empty:
            log(f'  union {proto}: F1 {r["Mean_F1"].iloc[0]:.3f}')

    # Explicit answer to the "gap" wording.
    section('B1b answer: does tuning close the VarScan2 precision gap?')
    log(f'VarScan2 default precision {vs_default["Mean_Precision"]:.3f}; '
        f'fair-tuned precision {vs_cross["Mean_Precision"]:.3f} '
        f'(cross-group) and {vs_lopo["Mean_Precision"]:.3f} (LOPO).')
    log(f'VarScan2 fair-tuned F1 {vs_cross["Mean_F1"]:.3f} vs '
        f'SomaticXGB default {som["Mean_F1"]:.3f}; the gap is '
        f'{vs_cross["Delta_vs_SomaticXGB_default"]:+.3f}.')
    log('The statement that "no threshold adjustment would close the gap" '
        'is too strong: a simple SSC threshold closes most of it '
        '(precision 0.31 -> about 0.89, F1 0.47 -> about 0.87), but does not '
        'overturn SomaticXGB.  The manuscript should report tuned VarScan2 '
        'as the fair comparator and say that SomaticXGB remains ahead by '
        'roughly 0.06-0.07 F1, rather than saying that threshold adjustment '
        'cannot help.')

    # Selected rules summary.
    section('B1b selected rules (fair protocols)')
    for method in ['Strelka2', 'Mutect2', 'VarScan2', 'SomaticXGB']:
        c = summary_df[(summary_df['Method'] == method) &
                       (summary_df['Protocol'] == 'cross-group')]
        l = summary_df[(summary_df['Method'] == method) &
                       (summary_df['Protocol'] == 'LOPO')]
        if not c.empty:
            log(f'  {method} cross-group: {c["Selected_Rule_Summary"].iloc[0]}')
        if not l.empty:
            log(f'  {method} LOPO: {l["Selected_Rule_Summary"].iloc[0]}')
    for proto in ['cross-group (independent)', 'cross-group (joint)',
                  'LOPO (independent)', 'LOPO (joint)']:
        r = summary_df[(summary_df['Method'] == 'Strelka2_Mutect2_union') &
                       (summary_df['Protocol'] == proto)]
        if not r.empty:
            log(f'  union {proto}: {r["Selected_Rule_Summary"].iloc[0]}')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    section('B1b. Fair tuning controls')
    log('Purpose: remove evaluation-side label leakage from threshold '
        'selection.')
    log('Cross-group: train on one group, evaluate on the other.')
    log('LOPO: train on five platforms, evaluate on the held-out platform.')
    log('Oracle rows use evaluation-side labels and are upper bounds only.')

    recs, truth_cache = load_all_samples()
    print_score_distributions(recs)

    # Default metrics.
    default_df = default_rows(recs)
    default_path = os.path.join(TABLES, 'B1b_default_by_sample.csv')
    default_df.to_csv(default_path, index=False)
    log(f'[table] wrote {default_path}')

    # Oracle upper bounds.
    oracle_df = oracle_rows(recs)
    oracle_path = os.path.join(TABLES, 'B1b_oracle_by_sample.csv')
    oracle_df.to_csv(oracle_path, index=False)
    log(f'[table] wrote {oracle_path}')

    # Cross-group.
    cross_tune, cross_sample = cross_group(recs)
    cross_tune_path = os.path.join(TABLES, 'B1b_cross_group_tuning.csv')
    cross_sample_path = os.path.join(TABLES,
                                     'B1b_cross_group_eval_by_sample.csv')
    cross_tune.to_csv(cross_tune_path, index=False)
    cross_sample.to_csv(cross_sample_path, index=False)
    log(f'[table] wrote {cross_tune_path}')
    log(f'[table] wrote {cross_sample_path}')

    # LOPO.
    lopo_tune, lopo_sample = lopo(recs)
    lopo_tune_path = os.path.join(TABLES, 'B1b_lopo_tuning.csv')
    lopo_sample_path = os.path.join(TABLES,
                                    'B1b_lopo_eval_by_sample.csv')
    lopo_tune.to_csv(lopo_tune_path, index=False)
    lopo_sample.to_csv(lopo_sample_path, index=False)
    log(f'[table] wrote {lopo_tune_path}')
    log(f'[table] wrote {lopo_sample_path}')

    # Union-specific table and oracle union per-sample table.
    union_rows = []
    for df, protocol_prefix in [(cross_tune, 'cross-group'),
                                (lopo_tune, 'LOPO')]:
        sub = df[df['Method'] == 'Strelka2_Mutect2_union'].copy()
        if not sub.empty:
            union_rows.append(sub)
    union_df = pd.concat(union_rows, ignore_index=True) if union_rows \
        else pd.DataFrame()
    union_path = os.path.join(TABLES, 'B1b_union_tuning.csv')
    union_df.to_csv(union_path, index=False)
    log(f'[table] wrote {union_path}')

    oracle_union = oracle_df[
        oracle_df['Method'] == 'Strelka2_Mutect2_union'].copy()
    oracle_union_path = os.path.join(TABLES,
                                     'B1b_union_oracle_by_sample.csv')
    oracle_union.to_csv(oracle_union_path, index=False)
    log(f'[table] wrote {oracle_union_path}')

    # Summary.
    summary = build_summary(default_df, cross_sample, lopo_sample, oracle_df)
    summary_path = os.path.join(TABLES,
                                'B1b_fair_tuning_comparison.csv')
    summary.to_csv(summary_path, index=False)
    log(f'[table] wrote {summary_path}')

    # Print key tables.
    section('Cross-group tuning (per direction)')
    log(cross_tune.round(4).to_string(index=False))
    section('LOPO tuning (per held-out platform)')
    log(lopo_tune.round(4).to_string(index=False))
    section('Fair tuning comparison')
    cols = ['Method', 'Tuning', 'Protocol', 'N_samples', 'Mean_F1', 'SD_F1',
            'Mean_Recall', 'Mean_Precision', 'F1_Gain_vs_Own_Default',
            'Delta_vs_SomaticXGB_default',
            'Still_Below_SomaticXGB_default']
    log(summary[cols].round(4).to_string(index=False))
    log('')
    log('Selected rules:')
    for _, r in summary.iterrows():
        log(f"  {r['Method']} | {r['Tuning']} | {r['Protocol']} | "
            f"{r['Selected_Rule_Summary']}")

    # Figures.
    make_summary_figure(summary)
    make_lopo_heatmap(lopo_sample, default_df)

    # Conclusions.
    print_conclusions(summary, cross_tune, lopo_tune, default_df, oracle_df)

    section('DONE')


if __name__ == '__main__':
    main()
