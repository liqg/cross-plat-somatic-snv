#!/usr/bin/env python3
"""B1. Threshold sweeps for the conventional baseline callers.

Reviewer 2, Comment 6: "say whether the baseline methods had a comparable
opportunity to be tuned".

This script gives Strelka2, Mutect2 and VarScan2 a score-threshold scan on the
same 100M-read test data used for SomaticXGB, and compares the best F1 of each
method with SomaticXGB's best F1.  Two analyses are produced:

  primary      threshold sweep inside the method's default call set
               (Strelka2 PASS, Mutect2 PASS, VarScan2 snp.pass, SomaticXGB all
               prediction candidates with pred > 0.1 as the default).
  sensitivity  score-only sweep over *all* VCF records, ignoring FILTER for
               Strelka2/Mutect2 (and pass+failed for VarScan2).  This is a
               deliberately generous upper bound: it asks what the best
               achievable F1 would be if the caller's hard filters were
               overridden and only the score were tuned.

Truth / near-site handling follows scripts/make_fig3_data.py exactly:
  var = CHROM:POS:REF:ALT
  dist = distance to the nearest other truth site on the same chromosome
  truth = sites with dist >= 50
  near  = sites with dist <  50
  every call is first removed if its var is in near, then compared with truth
  FN = |truth| - TP

Only single-base SNVs are used (REF and ALT each one base, ALT not
multi-allelic).  Chromosome names are normalised by stripping a leading "chr".

Score fields are discovered from the VCF header (bcftools view -h); the field
actually used is printed for every method.  For VarScan2, SSC is the somatic
score in the header of these files (SOMATICSCORE is not present).

Outputs (relative to response/addendum-analyses/):
  tables/B1_threshold_sweep.csv
  tables/B1_method_summary.csv
  tables/B1_best_f1_comparison.csv
  tables/B1_sensitivity_allrecords.csv
  figures/B1_f1_vs_threshold.png/.pdf
  logs/B1.log (written by the shell invocation, not by this script)

Usage:
  /usr/bin/python3 B1_baseline_threshold_sweep.py --validate
  /usr/bin/python3 B1_baseline_threshold_sweep.py
"""

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
import common  # noqa: E402

import argparse
import os
import re
import subprocess
import sys
from collections import Counter

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


# ---------------------------------------------------------------------------
# Paths and design constants
# ---------------------------------------------------------------------------

BASE = str(common.WORK_ROOT)
TESTDIR = f'{BASE}/4.test'
OUT = str(common.RESULTS_ROOT)
TABLES = os.path.join(OUT, 'tables')
FIGURES = os.path.join(OUT, 'figures')

GROUPS = ['45', '54']

# (directory id, display name), matching make_fig3_data.py's plat_map.
PLATFORMS = [
    ('FASTASeq_300', 'FASTASeq 300'),
    ('GenoLab_M', 'GenoLab M'),
    ('MGISEQ-T7', 'DNBSEQ-T7'),
    ('NovaSeq_6000', 'NovaSeq 6000'),
    ('NovaSeq_X', 'NovaSeq X'),
    ('SURFSeq_5000', 'SURFSeq 5000'),
]
PLAT_DISPLAY = dict(PLATFORMS)

METHODS = ['Strelka2', 'Mutect2', 'VarScan2', 'SomaticXGB']

# Quantile sweep: keep the top fraction of the sample's default calls.  1.0 is
# the default set.  The cut is the empirical quantile of the score; ties can
# make the actual retained fraction slightly different, which is recorded.
QUANTILE_FRACTIONS = [1.0, 0.95, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2,
                      0.1, 0.05, 0.02]

# Absolute score cuts.  The requested ranges are included; extra cuts cover
# the actual score distributions (Strelka2 PASS QSS starts at 33-51 in these
# files, Mutect2 PASS TLOD at 3-5, and VarScan2 SSC extends to 62).
ABS_CUTS = {
    'Strelka2': [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70,
                 75, 80, 90, 100, 110, 120, 130, 140],
    'Mutect2': [0, 2, 4, 6, 8, 10, 12, 15, 20, 25, 30, 35, 40, 50, 60, 70,
                80, 100, 120, 150, 200],
    'VarScan2': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20, 25, 30, 40,
                 50, 60],
    'SomaticXGB': [0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5,
                   0.6, 0.7, 0.8, 0.9, 0.95, 0.99],
}

# Candidate score fields, in order of preference.  INFO before FORMAT.
SCORE_CANDIDATES = {
    'Strelka2': [('INFO', ['QSS', 'SomaticEVS', 'QSS_NT'])],
    'Mutect2': [('INFO', ['TLOD']), ('FORMAT', ['TLOD'])],
    'VarScan2': [('INFO', ['SOMATICSCORE', 'SSC', 'SPV'])],
}

# Default (primary) call set for each method.
DEFAULT_CALLSET = {
    'Strelka2': 'PASS',
    'Mutect2': 'PASS',
    'VarScan2': 'snp.pass',
    'SomaticXGB': 'pred_all',
}
DEFAULT_RULE = {
    'Strelka2': 'FILTER=PASS',
    'Mutect2': 'FILTER=PASS',
    'VarScan2': 'all records in snp.pass',
    'SomaticXGB': 'pred > 0.1',
}
SOMATIC_DEFAULT_CUT = 0.1

# Figure colours, one per platform (consistent across panels).
PLAT_COLORS = {
    'FASTASeq 300': '#1f77b4',
    'GenoLab M': '#ff7f0e',
    'DNBSEQ-T7': '#2ca02c',
    'NovaSeq 6000': '#d62728',
    'NovaSeq X': '#9467bd',
    'SURFSeq 5000': '#8c564b',
}


# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------

def log(msg=''):
    print(msg, flush=True)


def section(title):
    log('')
    log('=' * 78)
    log(title)
    log('=' * 78)


# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------

def sid(platform_id):
    """Sample directory/file stem, e.g. FASTASeq_300_reads100M."""
    return f'{platform_id}_reads100M'


def path_strelka(g, pid):
    return (f'{TESTDIR}/{g}/strelka/{sid(pid)}/results/variants/'
            f'somatic.snvs.vcf.gz')


def path_mutect(g, pid):
    return f'{TESTDIR}/{g}/mutect/{sid(pid)}.filter.vcf.gz'


def path_varscan_pass(g, pid):
    return f'{TESTDIR}/{g}/varscan_novo/{sid(pid)}.snp.pass.vcf.gz'


def path_varscan_failed(g, pid):
    return f'{TESTDIR}/{g}/varscan_novo/{sid(pid)}.snp.failed.vcf.gz'


def path_pred(g, pid):
    return f'{TESTDIR}/{g}/pred/{sid(pid)}_pred.txt.gz'


def norm_chrom(x):
    """Strip a leading chr/Chr/CHR from a chromosome name."""
    return re.sub(r'^chr', '', str(x), flags=re.IGNORECASE)


def norm_var(chrom, pos, ref, alt):
    """Normalise a variant to CHROM:POS:REF:ALT (no chr prefix, upper case)."""
    return f'{norm_chrom(chrom)}:{int(float(pos))}:{str(ref).upper()}:{str(alt).upper()}'


def norm_var_string(v):
    """Normalise a pred-file var string.  Returns None if malformed."""
    parts = str(v).split(':')
    if len(parts) != 4:
        return None
    c, p, r, a = parts
    try:
        p = int(float(p))
    except ValueError:
        return None
    return f'{norm_chrom(c)}:{p}:{str(r).upper()}:{str(a).upper()}'


def run_cmd(cmd):
    """Run a command, return stdout.  bcftools warnings on stderr are ignored."""
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(
            f'Command failed ({p.returncode}): {" ".join(cmd)}\n'
            f'STDERR:\n{p.stderr[:4000]}'
        )
    return p.stdout


# ---------------------------------------------------------------------------
# Truth and near-site sets (exact replication of make_fig3_data.py)
# ---------------------------------------------------------------------------

TRUTH_COLUMNS = ['CHROM', 'POS', 'P', 'REF', 'ALT', 'GT', 'f', 'b', 'GT2',
                 'group', 'AF']


def nearest_distances(df):
    """Distance from each site to the nearest other site on the same chromosome.

    This is the same rule as add_dist() in scripts/make_fig3_data.py
    (10000 for a singleton; min(left gap, right gap) otherwise).  Assignment
    is by row index rather than by append order, which is equivalent for these
    chromosome-block-sorted truth files and safer if that ever changes.
    """
    dist = np.empty(len(df), dtype=float)
    for chrom in df['CHROM'].unique():
        idx = df.index[df['CHROM'] == chrom].to_numpy()
        pos = pd.to_numeric(df.loc[idx, 'POS']).to_numpy()
        n = len(pos)
        if n == 1:
            dd = np.array([10000.0])
        elif n == 2:
            dd = np.array([pos[1] - pos[0]] * 2, dtype=float)
        else:
            d1 = np.r_[10000.0, pos[1:] - pos[:-1]]
            d2 = np.r_[pos[1:] - pos[:-1], 10000.0]
            dd = np.minimum(d1, d2).astype(float)
        dist[idx] = dd
    return dist


def load_truth(g):
    path = f'{common.REPO_ROOT}/workflow/pos_files/test/{g}/pf{g}_low.txt'
    raw = pd.read_csv(path, sep='\t', header=None, dtype=str)
    if raw.shape[1] != len(TRUTH_COLUMNS):
        raise RuntimeError(
            f'{path}: expected {len(TRUTH_COLUMNS)} columns, got {raw.shape[1]}')
    pf = raw.copy()
    pf.columns = TRUTH_COLUMNS
    pf['CHROM'] = pf['CHROM'].map(norm_chrom)
    pf['POS'] = pd.to_numeric(pf['POS']).astype(int)
    pf['REF'] = pf['REF'].astype(str).str.upper()
    pf['ALT'] = pf['ALT'].astype(str).str.upper()
    pf['dist'] = nearest_distances(pf)
    pf['var'] = (pf['CHROM'] + ':' + pf['POS'].astype(str) + ':' +
                 pf['REF'] + ':' + pf['ALT'])
    truth = set(pf.loc[pf['dist'] >= 50, 'var'])
    near = set(pf.loc[pf['dist'] < 50, 'var'])
    n_dup = int(len(pf) - pf['var'].nunique())
    log(f'[truth g={g}] {path}')
    log(f'  rows={len(pf)}  truth(dist>=50)={len(truth)}  '
        f'near(dist<50)={len(near)}  duplicate_vars={n_dup}  '
        f'chromosomes={pf["CHROM"].nunique()}')
    if n_dup:
        log('  WARNING: duplicate truth vars were collapsed by set().')
    return {'truth': truth, 'near': near, 'pf': pf, 'path': path}


# ---------------------------------------------------------------------------
# VCF header inspection and score-field discovery
# ---------------------------------------------------------------------------

def get_header(path):
    return run_cmd(['bcftools', 'view', '-h', path])


def get_samples(path):
    out = run_cmd(['bcftools', 'query', '-l', path])
    return [x for x in out.strip().split('\n') if x]


def parse_header_fields(header):
    info, fmt = {}, {}
    for line in header.splitlines():
        m = re.match(r'##INFO=<ID=([^,>]+)', line)
        if m:
            info[m.group(1)] = line
            continue
        m = re.match(r'##FORMAT=<ID=([^,>]+)', line)
        if m:
            fmt[m.group(1)] = line
    return info, fmt


def choose_score_field(path, method):
    """Return (field, source, info_fields, format_fields, header)."""
    header = get_header(path)
    info, fmt = parse_header_fields(header)
    for source, candidates in SCORE_CANDIDATES[method]:
        available = info if source == 'INFO' else fmt
        for cand in candidates:
            if cand in available:
                return cand, source, info, fmt, header
    raise RuntimeError(
        f'No suitable score field for {method} in {path}; '
        f'INFO candidates={SCORE_CANDIDATES[method]}, '
        f'INFO fields={sorted(info)}, FORMAT fields={sorted(fmt)}')


def print_header_evidence(method, path):
    """Print the actual bcftools header lines for the score fields."""
    header = get_header(path)
    info, fmt = parse_header_fields(header)
    log(f'  --- {method} header evidence: {path}')
    wanted = set()
    for _, cands in SCORE_CANDIDATES.get(method, []):
        wanted.update(cands)
    shown = 0
    for line in header.splitlines():
        if line.startswith('##INFO=<ID=') or line.startswith('##FORMAT=<ID='):
            m = re.match(r'##(?:INFO|FORMAT)=<ID=([^,>]+)', line)
            if m and m.group(1) in wanted:
                log(f'    {line}')
                shown += 1
        elif line.startswith('##FILTER=<ID=PASS'):
            log(f'    {line}')
            shown += 1
    if not shown:
        log('    (no candidate score field lines found)')


# ---------------------------------------------------------------------------
# Parse calls from a VCF
# ---------------------------------------------------------------------------

FILTER_PASS = {'PASS', '.'}


def parse_vcf_scores(path, method, pass_only, near, sample_name=None,
                     verbose=True):
    """Parse SNV calls and their scores from a VCF.

    pass_only=True keeps FILTER PASS or '.'.  pass_only=False keeps every
    record (used only for the all-record sensitivity analysis).

    Returns (scores dict var->max score, stats dict).
    """
    field, source, info, fmt, header = choose_score_field(path, method)

    if source == 'INFO':
        query_fmt = ('%CHROM\\t%POS\\t%REF\\t%ALT\\t%FILTER\\t'
                     f'%INFO/{field}\\n')
        cmd = ['bcftools', 'query', '-f', query_fmt, path]
    else:
        # FORMAT fallback (not needed for the files in this study, but kept
        # generic).  Query one sample.
        samples = get_samples(path)
        if sample_name is None:
            preferred = ['tumor', 'TUMOR', 'Tumour', 'TUMOR_NORMAL']
            sample_name = next(
                (s for s in preferred if s in samples), samples[-1])
        if sample_name not in samples:
            sample_name = samples[-1]
        query_fmt = ('%CHROM\\t%POS\\t%REF\\t%ALT\\t%FILTER[\\t%'
                     f'{field}]\\n')
        cmd = ['bcftools', 'query', '-s', sample_name, '-f', query_fmt, path]

    out = run_cmd(cmd)

    scores = {}
    stats = {
        'score_field': field,
        'score_source': source,
        'path': path,
        'n_records': 0,
        'n_pass_filter': 0,
        'n_nonpass_skipped': 0,
        'n_multiallelic_skipped': 0,
        'n_indel_skipped': 0,
        'n_near_skipped': 0,
        'n_missing_score': 0,
        'n_duplicate_vars': 0,
        'n_calls': 0,
        'filter_counts': Counter(),
    }

    for line in out.splitlines():
        if not line:
            continue
        fields = line.split('\t')
        if len(fields) < 6:
            continue
        chrom, pos, ref, alt, flt, score_str = fields[:6]
        stats['n_records'] += 1
        stats['filter_counts'][flt] += 1

        is_pass = (flt in FILTER_PASS) or ('PASS' in flt)
        if is_pass:
            stats['n_pass_filter'] += 1
        if pass_only and not is_pass:
            stats['n_nonpass_skipped'] += 1
            continue

        if ',' in alt:
            stats['n_multiallelic_skipped'] += 1
            continue
        if len(ref) != 1 or len(alt) != 1 or alt in ('.', '*', '<*>'):
            stats['n_indel_skipped'] += 1
            continue

        var = norm_var(chrom, pos, ref, alt)
        if var in near:
            stats['n_near_skipped'] += 1
            continue

        try:
            score = float(score_str)
        except (TypeError, ValueError):
            stats['n_missing_score'] += 1
            continue
        if not np.isfinite(score):
            stats['n_missing_score'] += 1
            continue

        if var in scores:
            stats['n_duplicate_vars'] += 1
            if score > scores[var]:
                scores[var] = score
        else:
            scores[var] = score

    stats['n_calls'] = len(scores)
    if scores:
        vals = np.array(list(scores.values()), dtype=float)
        stats.update({
            'score_min': float(vals.min()),
            'score_p10': float(np.quantile(vals, 0.10)),
            'score_median': float(np.median(vals)),
            'score_p90': float(np.quantile(vals, 0.90)),
            'score_max': float(vals.max()),
        })
    else:
        stats.update({k: np.nan for k in
                      ('score_min', 'score_p10', 'score_median',
                       'score_p90', 'score_max')})
    if verbose:
        top_filters = ', '.join(
            f'{k}:{v}' for k, v in stats['filter_counts'].most_common(6))
        log(f'    {os.path.basename(path)} [{method}] '
            f'field={source}/{field} pass_only={pass_only} '
            f'records={stats["n_records"]} pass={stats["n_pass_filter"]} '
            f'calls={stats["n_calls"]} '
            f'(nonpass_skip={stats["n_nonpass_skipped"]}, '
            f'indel_skip={stats["n_indel_skipped"]}, '
            f'multi_skip={stats["n_multiallelic_skipped"]}, '
            f'near_skip={stats["n_near_skipped"]}, '
            f'missing_score={stats["n_missing_score"]}, '
            f'dup={stats["n_duplicate_vars"]})')
        if top_filters:
            log(f'      FILTER counts: {top_filters}')
    return scores, stats


def merge_score_dicts(a, b):
    """Merge two var->score dicts, keeping the maximum score."""
    out = dict(a)
    for k, v in b.items():
        if k in out:
            out[k] = max(out[k], v)
        else:
            out[k] = v
    return out


# ---------------------------------------------------------------------------
# Parse SomaticXGB predictions
# ---------------------------------------------------------------------------

def parse_pred_scores(path, near, snv_only=False, verbose=True):
    """Return (scores dict var->max pred, stats dict).

    By default all candidate rows are kept, matching make_fig3_data.py (the
    published SomaticXGB numbers include a handful of non-SNV candidates).
    snv_only=True restricts to single-base SNVs, as a sensitivity check.
    """
    p = pd.read_csv(path, sep='\t', compression='gzip')
    if 'var' not in p.columns or 'pred' not in p.columns:
        raise RuntimeError(f'{path}: expected columns var and pred, got '
                           f'{list(p.columns)}')

    stats = {
        'path': path,
        'score_field': 'pred',
        'score_source': 'pred file',
        'n_records': len(p),
        'n_non_snv': 0,
        'n_near_skipped': 0,
        'n_duplicate_vars': 0,
        'n_calls': 0,
        'n_missing_score': 0,
    }

    p = p.copy()
    p['var'] = p['var'].map(norm_var_string)
    bad_var = int(p['var'].isna().sum())
    p = p[p['var'].notna()].copy()

    # Count non-SNV candidates whether or not we filter them.  The published
    # fig3 numbers keep them; snv_only=True is the sensitivity analysis.
    ref = p['REF'].astype(str).str.upper() if 'REF' in p.columns else None
    alt = p['ALT'].astype(str).str.upper() if 'ALT' in p.columns else None
    if ref is None or alt is None:
        # Fall back to parsing the var string.
        parts = p['var'].str.split(':', expand=True)
        ref, alt = parts[2], parts[3]
    is_snv = (ref.str.len() == 1) & (alt.str.len() == 1) & (~alt.str.contains(','))
    stats['n_non_snv'] = int((~is_snv).sum())
    if snv_only:
        p = p[is_snv].copy()

    n_before_near = len(p)
    p = p[~p['var'].isin(near)].copy()
    stats['n_near_skipped'] = int(n_before_near - len(p))

    p['pred'] = pd.to_numeric(p['pred'], errors='coerce')
    stats['n_missing_score'] = int(p['pred'].isna().sum())
    p = p[p['pred'].notna() & np.isfinite(p['pred'])].copy()

    agg = p.groupby('var', as_index=False)['pred'].max()
    stats['n_duplicate_vars'] = int(len(p) - len(agg))
    stats['n_calls'] = int(len(agg))
    scores = dict(zip(agg['var'], agg['pred']))
    if scores:
        vals = np.array(list(scores.values()), dtype=float)
        stats.update({
            'score_min': float(vals.min()),
            'score_p10': float(np.quantile(vals, 0.10)),
            'score_median': float(np.median(vals)),
            'score_p90': float(np.quantile(vals, 0.90)),
            'score_max': float(vals.max()),
        })
    else:
        stats.update({k: np.nan for k in
                      ('score_min', 'score_p10', 'score_median',
                       'score_p90', 'score_max')})
    if verbose:
        log(f'    {os.path.basename(path)} [SomaticXGB] pred file '
            f'rows={stats["n_records"]} calls={stats["n_calls"]} '
            f'(non_snv={stats["n_non_snv"]}, '
            f'near_skip={stats["n_near_skipped"]}, '
            f'dup={stats["n_duplicate_vars"]}, '
            f'missing_score={stats["n_missing_score"]}, bad_var={bad_var})')
    return scores, stats


# ---------------------------------------------------------------------------
# Metrics and threshold sweeps
# ---------------------------------------------------------------------------

def compute_metrics(calls, truth):
    """TP/FP/FN, recall, precision, F1.  FN = |truth| - TP, as in fig3."""
    calls = set(calls)
    truth_n = len(truth)
    tp = len(calls & truth)
    fp = len(calls - truth)
    fn = truth_n - tp
    recall = tp / truth_n if truth_n else np.nan
    precision = tp / (tp + fp) if (tp + fp) else np.nan
    if np.isfinite(recall) and np.isfinite(precision) and (recall + precision) > 0:
        f1 = 2 * recall * precision / (recall + precision)
    else:
        f1 = np.nan
    return {'TP': tp, 'FP': fp, 'FN': fn, 'TruthN': truth_n,
            'Recall': recall, 'Precision': precision, 'F1': f1}


def _row_base(method, group, platform, sample, score_field, callset,
              scan_type, target_fraction, cut, min_retained, kept_n,
              kept_fraction, metrics):
    return {
        'Method': method,
        'ScoreField': score_field,
        'Group': group,
        'Platform': platform,
        'Sample': sample,
        'CallSet': callset,
        'ScanType': scan_type,
        'TargetFraction': target_fraction,
        'Cut': cut,
        'MinRetainedScore': min_retained,
        'KeptN': kept_n,
        'KeptFraction': kept_fraction,
        **metrics,
    }


def sweep_sample(method, group, platform, sample, scores, truth,
                 default_vars, callset, score_field, default_cut=None):
    """Build default + quantile + absolute sweep rows for one sample."""
    all_vars = set(scores)
    scores_arr = np.array([scores[v] for v in all_vars], dtype=float)
    vars_arr = np.array(sorted(all_vars), dtype=object)
    # Re-align: vars_arr sorted, so build the score array in the same order.
    score_by_var = scores
    scores_arr = np.array([score_by_var[v] for v in vars_arr], dtype=float)

    rows = []

    # ---- default row ------------------------------------------------------
    default_vars = set(default_vars)
    if default_vars:
        min_default = float(min(score_by_var[v] for v in default_vars))
    else:
        min_default = np.nan
    m = compute_metrics(default_vars, truth)
    rows.append(_row_base(
        method, group, platform, sample, score_field, callset,
        'default', np.nan,
        default_cut if default_cut is not None else np.nan,
        min_default, len(default_vars),
        len(default_vars) / len(all_vars) if all_vars else np.nan, m))

    if not all_vars:
        return rows

    # ---- quantile sweep ---------------------------------------------------
    for frac in QUANTILE_FRACTIONS:
        if len(scores_arr) == 1:
            cut = float(scores_arr[0])
        else:
            cut = float(np.quantile(scores_arr, 1.0 - frac))
        keep_mask = scores_arr >= cut
        kept = vars_arr[keep_mask]
        m = compute_metrics(kept, truth)
        min_ret = float(scores_arr[keep_mask].min()) if keep_mask.any() else np.nan
        rows.append(_row_base(
            method, group, platform, sample, score_field, callset,
            'quantile', frac, cut, min_ret, int(keep_mask.sum()),
            float(keep_mask.mean()), m))

    # ---- absolute sweep ---------------------------------------------------
    for cut in ABS_CUTS[method]:
        keep_mask = scores_arr >= cut
        kept = vars_arr[keep_mask]
        m = compute_metrics(kept, truth)
        min_ret = float(scores_arr[keep_mask].min()) if keep_mask.any() else np.nan
        rows.append(_row_base(
            method, group, platform, sample, score_field, callset,
            'absolute', np.nan, float(cut), min_ret, int(keep_mask.sum()),
            float(keep_mask.mean()), m))

    return rows


def _finite_cut(r):
    c = r['Cut']
    try:
        c = float(c)
    except (TypeError, ValueError):
        return -np.inf
    return c if np.isfinite(c) else -np.inf


def pick_best(rows):
    """Best row by F1; ties go to higher recall, then to the higher cut."""
    valid = [r for r in rows if np.isfinite(r.get('F1', np.nan))]
    if not valid:
        return None
    return max(valid, key=lambda r: (r['F1'], r['Recall'], _finite_cut(r)))


def global_selection(rows, scan_type, method):
    """Single threshold/fraction maximising mean F1 across samples.

    Ties go to higher mean recall, then to the higher threshold.
    """
    if scan_type == 'absolute':
        key_field = 'Cut'
        keys = sorted({float(r[key_field]) for r in rows
                       if r['ScanType'] == 'absolute'})
    elif scan_type == 'quantile':
        key_field = 'TargetFraction'
        keys = sorted({float(r[key_field]) for r in rows
                       if r['ScanType'] == 'quantile'})
    else:
        keys = [None]
        key_field = None

    candidates = []
    for k in keys:
        if key_field is None:
            sub = [r for r in rows if r['ScanType'] == scan_type]
        else:
            sub = [r for r in rows
                   if r['ScanType'] == scan_type
                   and np.isclose(float(r[key_field]), k)]
        if not sub:
            continue
        f1_vals = np.array([r['F1'] for r in sub], dtype=float)
        if not np.any(np.isfinite(f1_vals)):
            continue
        f1 = float(np.nanmean(f1_vals))
        rec = float(np.nanmean([r['Recall'] for r in sub]))
        prec = float(np.nanmean([r['Precision'] for r in sub]))
        candidates.append({
            'key': k, 'n': len(sub), 'mean_F1': f1,
            'mean_recall': rec, 'mean_precision': prec,
        })
    if not candidates:
        return None
    best = max(candidates, key=lambda x: (
        x['mean_F1'], x['mean_recall'],
        x['key'] if x['key'] is not None else -np.inf))
    best['method'] = method
    best['scan_type'] = scan_type
    return best


def global_best_overall(rows, method):
    """Best of the absolute and quantile global scans."""
    cands = []
    for st in ('absolute', 'quantile'):
        b = global_selection(rows, st, method)
        if b is not None:
            cands.append(b)
    if not cands:
        return None
    return max(cands, key=lambda x: (
        x['mean_F1'], x['mean_recall'],
        x['key'] if x['key'] is not None else -np.inf))


# ---------------------------------------------------------------------------
# Validation mode
# ---------------------------------------------------------------------------

def validate(truth_cache):
    section('VALIDATION: score fields, first calls, default metrics')
    log('Python: ' + sys.version.replace('\n', ' '))
    log(f'pandas {pd.__version__}, numpy {np.__version__}, '
        f'matplotlib {matplotlib.__version__}')
    try:
        log('bcftools: ' + run_cmd(['bcftools', '--version']).splitlines()[0])
    except Exception:
        pass

    g = '45'
    pid = 'FASTASeq_300'
    truth = truth_cache[g]['truth']
    near = truth_cache[g]['near']

    # 1. Header evidence for all three callers.
    section('1. VCF header evidence (bcftools view -h)')
    for method, path in [
        ('Strelka2', path_strelka(g, pid)),
        ('Mutect2', path_mutect(g, pid)),
        ('VarScan2', path_varscan_pass(g, pid)),
    ]:
        print_header_evidence(method, path)
        field, source, info, fmt, _ = choose_score_field(path, method)
        log(f'  CHOSEN for {method}: {source}/{field}')
        for _, cands in SCORE_CANDIDATES[method]:
            avail_info = [c for c in cands if c in info]
            avail_fmt = [c for c in cands if c in fmt]
            log(f'    candidates in INFO: {avail_info or "none"}; '
                f'in FORMAT: {avail_fmt or "none"}')

    # 2. Parse one sample per caller, print first calls and scores.
    section('2. Parsed calls and scores (first 8 per method)')
    for method, path, pass_only in [
        ('Strelka2', path_strelka(g, pid), True),
        ('Mutect2', path_mutect(g, pid), True),
        ('VarScan2', path_varscan_pass(g, pid), False),
    ]:
        scores, stats = parse_vcf_scores(
            path, method, pass_only=pass_only, near=near, verbose=True)
        log(f'  {method}: field={stats["score_source"]}/{stats["score_field"]}, '
            f'{stats["n_calls"]} calls after near filtering')
        items = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        log('    top-scoring calls:')
        for v, s in items[:8]:
            log(f'      {v}\t{s:.6g}\tshould_be_TP={v in truth}')
        log('    lowest-scoring calls:')
        for v, s in items[-8:]:
            log(f'      {v}\t{s:.6g}\tshould_be_TP={v in truth}')
        m = compute_metrics(set(scores), truth)
        log(f'    default metrics: TP={m["TP"]} FP={m["FP"]} '
            f'FN={m["FN"]} recall={m["Recall"]:.4f} '
            f'precision={m["Precision"]:.4f} F1={m["F1"]:.4f}')

    # 3. SomaticXGB pred file.
    section('3. SomaticXGB prediction file (first 8 rows)')
    p = pd.read_csv(path_pred(g, pid), sep='\t')
    log(f'  columns ({len(p.columns)}): {", ".join(p.columns)}')
    log(p[['var', 'pred', 'label']].head(8).to_string(index=False))
    scores, stats = parse_pred_scores(path_pred(g, pid), near, verbose=True)
    m = compute_metrics({v for v, s in scores.items() if s > SOMATIC_DEFAULT_CUT},
                        truth)
    log(f'  default (pred > {SOMATIC_DEFAULT_CUT}): TP={m["TP"]} FP={m["FP"]} '
        f'FN={m["FN"]} recall={m["Recall"]:.4f} '
        f'precision={m["Precision"]:.4f} F1={m["F1"]:.4f}')

    section('Validation done')


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def add_primary_and_sensitivity_rows(truth_cache):
    """Run all samples, return (primary_rows, sensitivity_rows, parse_stats)."""
    primary_rows = []
    sensitivity_rows = []
    parse_stats = []

    for g in GROUPS:
        truth = truth_cache[g]['truth']
        near = truth_cache[g]['near']
        log('')
        log(f'--- group {g}: truth={len(truth)}, near={len(near)} ---')

        for pid, platform in PLATFORMS:
            sample = sid(pid)
            log(f'  [{g} / {platform} / {sample}]')

            # ================= primary: Strelka2 PASS =================
            scores, st = parse_vcf_scores(
                path_strelka(g, pid), 'Strelka2', pass_only=True, near=near)
            st.update({'group': g, 'platform': platform, 'sample': sample,
                       'analysis': 'primary', 'callset': 'PASS'})
            parse_stats.append(st)
            primary_rows.extend(sweep_sample(
                'Strelka2', g, platform, sample, scores, truth,
                default_vars=set(scores), callset='PASS',
                score_field=st['score_field']))

            # ================= primary: Mutect2 PASS =================
            scores, st = parse_vcf_scores(
                path_mutect(g, pid), 'Mutect2', pass_only=True, near=near)
            st.update({'group': g, 'platform': platform, 'sample': sample,
                       'analysis': 'primary', 'callset': 'PASS'})
            parse_stats.append(st)
            primary_rows.extend(sweep_sample(
                'Mutect2', g, platform, sample, scores, truth,
                default_vars=set(scores), callset='PASS',
                score_field=st['score_field']))

            # ================= primary: VarScan2 snp.pass =================
            scores, st = parse_vcf_scores(
                path_varscan_pass(g, pid), 'VarScan2', pass_only=False,
                near=near)
            st.update({'group': g, 'platform': platform, 'sample': sample,
                       'analysis': 'primary', 'callset': 'snp.pass'})
            parse_stats.append(st)
            primary_rows.extend(sweep_sample(
                'VarScan2', g, platform, sample, scores, truth,
                default_vars=set(scores), callset='snp.pass',
                score_field=st['score_field']))

            # ================= primary: SomaticXGB =================
            scores, st = parse_pred_scores(path_pred(g, pid), near)
            st.update({'group': g, 'platform': platform, 'sample': sample,
                       'analysis': 'primary', 'callset': 'pred_all'})
            parse_stats.append(st)
            default_vars = {v for v, s in scores.items()
                            if s > SOMATIC_DEFAULT_CUT}
            primary_rows.extend(sweep_sample(
                'SomaticXGB', g, platform, sample, scores, truth,
                default_vars=default_vars, callset='pred_all',
                score_field='pred', default_cut=SOMATIC_DEFAULT_CUT))

            # ================= sensitivity: Strelka2 all records =========
            scores_all, st = parse_vcf_scores(
                path_strelka(g, pid), 'Strelka2', pass_only=False, near=near)
            st.update({'group': g, 'platform': platform, 'sample': sample,
                       'analysis': 'sensitivity', 'callset': 'ALL'})
            parse_stats.append(st)
            sensitivity_rows.extend(sweep_sample(
                'Strelka2', g, platform, sample, scores_all, truth,
                default_vars=set(scores_all), callset='ALL',
                score_field=st['score_field']))

            # ================= sensitivity: Mutect2 all records =========
            scores_all, st = parse_vcf_scores(
                path_mutect(g, pid), 'Mutect2', pass_only=False, near=near)
            st.update({'group': g, 'platform': platform, 'sample': sample,
                       'analysis': 'sensitivity', 'callset': 'ALL'})
            parse_stats.append(st)
            sensitivity_rows.extend(sweep_sample(
                'Mutect2', g, platform, sample, scores_all, truth,
                default_vars=set(scores_all), callset='ALL',
                score_field=st['score_field']))

            # ================= sensitivity: VarScan2 pass+failed ========
            sp, stp = parse_vcf_scores(
                path_varscan_pass(g, pid), 'VarScan2', pass_only=False,
                near=near)
            sf, stf = parse_vcf_scores(
                path_varscan_failed(g, pid), 'VarScan2', pass_only=False,
                near=near)
            scores_all = merge_score_dicts(sp, sf)
            st = dict(stp)
            st.update({
                'group': g, 'platform': platform, 'sample': sample,
                'analysis': 'sensitivity', 'callset': 'pass+failed',
                'n_records': stp['n_records'] + stf['n_records'],
                'n_calls': len(scores_all),
                'n_failed_records': stf['n_records'],
                'n_failed_calls': stf['n_calls'],
            })
            parse_stats.append(st)
            sensitivity_rows.extend(sweep_sample(
                'VarScan2', g, platform, sample, scores_all, truth,
                default_vars=set(scores_all), callset='pass+failed',
                score_field=stp['score_field']))

            # ================= sensitivity: SomaticXGB SNV-only ========
            scores_snv, st = parse_pred_scores(
                path_pred(g, pid), near, snv_only=True)
            st.update({'group': g, 'platform': platform, 'sample': sample,
                       'analysis': 'sensitivity', 'callset': 'pred_snv'})
            parse_stats.append(st)
            default_vars = {v for v, s in scores_snv.items()
                            if s > SOMATIC_DEFAULT_CUT}
            sensitivity_rows.extend(sweep_sample(
                'SomaticXGB', g, platform, sample, scores_snv, truth,
                default_vars=default_vars, callset='pred_snv',
                score_field='pred', default_cut=SOMATIC_DEFAULT_CUT))

    return primary_rows, sensitivity_rows, parse_stats


def score_distribution_table(rows):
    """Score distribution of the default call set and of the sweep rows."""
    dist_rows = []
    for method in METHODS:
        sub = [r for r in rows
               if r['Method'] == method and r['ScanType'] == 'default']
        if not sub:
            continue
        # Reconstruct the default scores from MinRetainedScore is not enough;
        # use the kept N and the min score across samples.  The actual score
        # vectors are summarised in parse_stats, but here we give the
        # threshold-relevant distribution: min retained score and cut range.
        mins = [r['MinRetainedScore'] for r in sub if np.isfinite(r['MinRetainedScore'])]
        kept = [r['KeptN'] for r in sub]
        dist_rows.append({
            'Method': method,
            'DefaultCallSet': DEFAULT_CALLSET[method],
            'N_samples': len(sub),
            'Calls_min': int(np.min(kept)),
            'Calls_median': int(np.median(kept)),
            'Calls_max': int(np.max(kept)),
            'MinScore_min': float(np.min(mins)),
            'MinScore_median': float(np.median(mins)),
            'MinScore_max': float(np.max(mins)),
        })
    return pd.DataFrame(dist_rows)


def raw_score_distribution(parse_stats):
    """Per-method score distribution from the parsed calls.

    parse_stats does not store the score vector (memory), so we report the
    number of calls and the per-sample minimum; the sweep table gives the
    distribution through the quantile cuts.  For a compact log table we use the
    quantile cuts in the sweep table instead.
    """
    return None


def build_summary(primary_df, sensitivity_df):
    summary_rows = []
    for method in METHODS:
        p = primary_df[primary_df['Method'] == method]
        d = p[p['ScanType'] == 'default'].copy()
        if d.empty:
            continue
        # per-sample best (oracle)
        best_rows = []
        for (g, plat, sample), sub in p.groupby(['Group', 'Platform', 'Sample']):
            b = pick_best(sub.to_dict('records'))
            if b is not None:
                best_rows.append(b)
        bdf = pd.DataFrame(best_rows)
        gains = bdf['F1'].to_numpy() - d.set_index(['Group', 'Platform', 'Sample']) \
            .loc[list(zip(bdf['Group'], bdf['Platform'], bdf['Sample'])), 'F1'] \
            .to_numpy()
        gl_abs = global_selection(p.to_dict('records'), 'absolute', method)
        gl_q = global_selection(p.to_dict('records'), 'quantile', method)
        gl_best = global_best_overall(p.to_dict('records'), method)

        row = {
            'Method': method,
            'ScoreField': d['ScoreField'].iloc[0],
            'DefaultCallSet': DEFAULT_CALLSET[method],
            'DefaultRule': DEFAULT_RULE[method],
            'N_samples': len(d),
            'Default_Mean_Recall': d['Recall'].mean(),
            'Default_Mean_Precision': d['Precision'].mean(),
            'Default_Mean_F1': d['F1'].mean(),
            'BestPerSample_Mean_Recall': bdf['Recall'].mean(),
            'BestPerSample_Mean_Precision': bdf['Precision'].mean(),
            'BestPerSample_Mean_F1': bdf['F1'].mean(),
            'BestPerSample_Mean_F1_Gain': float(np.mean(gains)),
            'BestPerSample_Max_F1_Gain': float(np.max(gains)),
            'BestPerSample_Min_F1_Gain': float(np.min(gains)),
            'BestPerSample_Median_Cut': bdf['Cut'].median(),
            'BestPerSample_Median_MinRetainedScore': bdf['MinRetainedScore'].median(),
            'GlobalAbs_Threshold': gl_abs['key'] if gl_abs else np.nan,
            'GlobalAbs_Mean_F1': gl_abs['mean_F1'] if gl_abs else np.nan,
            'GlobalAbs_Mean_Recall': gl_abs['mean_recall'] if gl_abs else np.nan,
            'GlobalAbs_Mean_Precision': gl_abs['mean_precision'] if gl_abs else np.nan,
            'GlobalAbs_Mean_F1_Gain': (gl_abs['mean_F1'] - d['F1'].mean())
                                      if gl_abs else np.nan,
            'GlobalQuantile_Fraction': gl_q['key'] if gl_q else np.nan,
            'GlobalQuantile_Mean_F1': gl_q['mean_F1'] if gl_q else np.nan,
            'GlobalQuantile_Mean_F1_Gain': (gl_q['mean_F1'] - d['F1'].mean())
                                           if gl_q else np.nan,
            'GlobalBest_Type': gl_best['scan_type'] if gl_best else np.nan,
            'GlobalBest_Threshold': gl_best['key'] if gl_best else np.nan,
            'GlobalBest_Mean_F1': gl_best['mean_F1'] if gl_best else np.nan,
            'GlobalBest_Mean_F1_Gain': (gl_best['mean_F1'] - d['F1'].mean())
                                       if gl_best else np.nan,
        }

        # Sensitivity analysis (all records / SNV-only candidates).
        s = sensitivity_df[sensitivity_df['Method'] == method]
        if not s.empty:
            s_vals = []
            for (g, plat, sample), sub in s.groupby(['Group', 'Platform', 'Sample']):
                b = pick_best(sub.to_dict('records'))
                if b is not None:
                    s_vals.append(b)
            sb = pd.DataFrame(s_vals)
            gl_s = global_selection(s.to_dict('records'), 'absolute', method)
            row.update({
                'Sensitivity_CallSet': s['CallSet'].iloc[0],
                'Sensitivity_BestPerSample_Mean_F1': sb['F1'].mean(),
                'Sensitivity_BestPerSample_Mean_F1_Gain': (
                    sb['F1'].mean() - d['F1'].mean()),
                'Sensitivity_GlobalAbs_Threshold': gl_s['key'] if gl_s else np.nan,
                'Sensitivity_GlobalAbs_Mean_F1': gl_s['mean_F1'] if gl_s else np.nan,
                'Sensitivity_GlobalAbs_Mean_F1_Gain': (
                    gl_s['mean_F1'] - d['F1'].mean()) if gl_s else np.nan,
            })
        summary_rows.append(row)
    return pd.DataFrame(summary_rows)


def build_comparison(primary_df, sensitivity_df):
    """Wide per group/platform table with default/best F1 for all methods."""
    rows = []
    for g in GROUPS:
        for pid, platform in PLATFORMS:
            sub = primary_df[(primary_df['Group'] == g) &
                             (primary_df['Platform'] == platform)]
            if sub.empty:
                continue
            sample = sid(pid)
            truth_n = int(sub['TruthN'].iloc[0])
            row = {'Group': g, 'Platform': platform, 'Sample': sample,
                   'TruthN': truth_n}
            best_by_method = {}
            default_by_method = {}
            for method in METHODS:
                msub = sub[sub['Method'] == method]
                d = msub[msub['ScanType'] == 'default']
                b = pick_best(msub.to_dict('records'))
                default_by_method[method] = d.iloc[0] if not d.empty else None
                best_by_method[method] = b
                if not d.empty:
                    row[f'{method}_default_F1'] = float(d['F1'].iloc[0])
                    row[f'{method}_default_Recall'] = float(d['Recall'].iloc[0])
                    row[f'{method}_default_Precision'] = float(d['Precision'].iloc[0])
                    row[f'{method}_default_calls'] = int(d['KeptN'].iloc[0])
                if b is not None:
                    row[f'{method}_best_F1'] = float(b['F1'])
                    row[f'{method}_best_Recall'] = float(b['Recall'])
                    row[f'{method}_best_Precision'] = float(b['Precision'])
                    row[f'{method}_best_cut'] = float(b['Cut']) if np.isfinite(b['Cut']) else np.nan
                    row[f'{method}_best_min_score'] = float(b['MinRetainedScore']) if np.isfinite(b['MinRetainedScore']) else np.nan
                    row[f'{method}_best_scan_type'] = b['ScanType']
                    row[f'{method}_best_calls'] = int(b['KeptN'])

            # All-record sensitivity best (not applicable to SomaticXGB's
            # primary all-candidate analysis; the SNV-only sensitivity is
            # reported separately in the summary table).
            sens = sensitivity_df[(sensitivity_df['Group'] == g) &
                                  (sensitivity_df['Platform'] == platform)]
            for method in ['Strelka2', 'Mutect2', 'VarScan2']:
                msub = sens[sens['Method'] == method]
                b = pick_best(msub.to_dict('records'))
                if b is not None:
                    row[f'{method}_allrecords_best_F1'] = float(b['F1'])
                    row[f'{method}_allrecords_best_cut'] = float(b['Cut']) if np.isfinite(b['Cut']) else np.nan
                    row[f'{method}_allrecords_best_scan_type'] = b['ScanType']

            # Differences vs SomaticXGB and ranks.
            som = best_by_method.get('SomaticXGB')
            for method in ['Strelka2', 'Mutect2', 'VarScan2']:
                b = best_by_method.get(method)
                if som is not None and b is not None:
                    row[f'SomaticXGB_minus_{method}_best_F1'] = float(som['F1'] - b['F1'])
            ranked = sorted(
                [(m, best_by_method[m]['F1']) for m in METHODS
                 if best_by_method.get(m) is not None],
                key=lambda x: x[1], reverse=True)
            for rank, (m, _) in enumerate(ranked, start=1):
                row[f'{m}_best_rank'] = rank
            rows.append(row)
    # Put the most useful columns first.
    front = ['Group', 'Platform', 'Sample', 'TruthN']
    cols = front + [c for c in rows[0].keys() if c not in front]
    return pd.DataFrame(rows)[cols]


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

def make_figure(primary_df, sensitivity_df, summary_df):
    """F1 vs threshold: quantile (top) and absolute (bottom), plus summary bars."""
    fig = plt.figure(figsize=(18, 11.5))
    gs = fig.add_gridspec(3, 4, height_ratios=[1, 1, 0.85],
                          hspace=0.50, wspace=0.30)
    axs = np.empty((2, 4), dtype=object)
    for i in range(2):
        for j in range(4):
            axs[i, j] = fig.add_subplot(gs[i, j])
    ax_sum = fig.add_subplot(gs[2, :])

    method_fields = {
        'Strelka2': 'QSS',
        'Mutect2': 'TLOD',
        'VarScan2': 'SSC',
        'SomaticXGB': 'pred',
    }
    xlims = {
        'Strelka2': (0, 110),
        'Mutect2': (0, 100),
        'VarScan2': (0, 30),
        'SomaticXGB': (0, 1),
    }

    # ---------- top row: quantile sweep ----------
    for j, method in enumerate(METHODS):
        ax = axs[0, j]
        sub = primary_df[(primary_df['Method'] == method) &
                         (primary_df['ScanType'] == 'quantile')]
        for pid, platform in PLATFORMS:
            s = sub[sub['Platform'] == platform]
            if s.empty:
                continue
            agg = s.groupby('TargetFraction')['F1'].mean().sort_index()
            ax.plot(agg.index, agg.values, marker='o', ms=3.5,
                    color=PLAT_COLORS[platform], lw=1.6, label=platform)
        ax.set_xlim(1.0, 0.0)
        ax.set_ylim(0, 1.0)
        ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_xlabel('Fraction of default calls retained\n(1.0 = default set; top-scoring kept)')
        ax.set_ylabel('F1 (mean of groups 45 + 54)')
        ax.set_title(f'{method}  |  score = {method_fields[method]}',
                     fontsize=11, fontweight='bold')
        ax.grid(alpha=0.30, lw=0.6)
        ax.axvline(1.0, color='k', ls=':', lw=1)

    # ---------- bottom row: absolute threshold sweep ----------
    for j, method in enumerate(METHODS):
        ax = axs[1, j]
        sub = primary_df[(primary_df['Method'] == method) &
                         (primary_df['ScanType'] == 'absolute')]
        for pid, platform in PLATFORMS:
            s = sub[sub['Platform'] == platform]
            if s.empty:
                continue
            agg = s.groupby('Cut')['F1'].mean().sort_index()
            ax.plot(agg.index, agg.values, marker='o', ms=3.0,
                    color=PLAT_COLORS[platform], lw=1.6, label=platform)

        # All-record score-only sensitivity for the three callers.
        if method in ('Strelka2', 'Mutect2', 'VarScan2'):
            sens = sensitivity_df[(sensitivity_df['Method'] == method) &
                                  (sensitivity_df['ScanType'] == 'absolute')]
            for pid, platform in PLATFORMS:
                s = sens[sens['Platform'] == platform]
                if s.empty:
                    continue
                agg = s.groupby('Cut')['F1'].mean().sort_index()
                ax.plot(agg.index, agg.values, ls='--', lw=1.0,
                        color=PLAT_COLORS[platform], alpha=0.65, marker='',
                        label=None)

        # Default mean F1 as a horizontal reference.
        d = primary_df[(primary_df['Method'] == method) &
                       (primary_df['ScanType'] == 'default')]
        if not d.empty:
            ax.axhline(d['F1'].mean(), color='k', ls=':', lw=1.2)
            ax.text(xlims[method][1] * 0.98, d['F1'].mean() + 0.012,
                    'default', ha='right', va='bottom', fontsize=8,
                    color='k')

        # Global best absolute threshold star.
        gl = global_selection(
            primary_df[primary_df['Method'] == method].to_dict('records'),
            'absolute', method)
        if gl is not None:
            ax.scatter([gl['key']], [gl['mean_F1']], marker='*', s=180,
                       color='k', zorder=6,
                       label='global best absolute cut')
            ax.annotate(f"cut={gl['key']:g}\nF1={gl['mean_F1']:.3f}",
                        xy=(gl['key'], gl['mean_F1']),
                        xytext=(8, -18), textcoords='offset points',
                        fontsize=8, color='k')

        ax.set_xlim(*xlims[method])
        ax.set_ylim(0, 1.0)
        ax.set_xlabel(f'Score threshold (keep {method_fields[method]} >= cut)')
        ax.set_ylabel('F1 (mean of groups 45 + 54)')
        ax.set_title(f'{method}  |  absolute threshold', fontsize=11)
        ax.grid(alpha=0.30, lw=0.6)

    # ---------- summary bars ----------
    methods = METHODS
    x = np.arange(len(methods))
    width = 0.26
    def vals(prefix):
        out = []
        for m in methods:
            row = summary_df[summary_df['Method'] == m]
            out.append(float(row[prefix].iloc[0]) if not row.empty else np.nan)
        return np.array(out)

    default_mean = vals('Default_Mean_F1')
    best_oracle = vals('BestPerSample_Mean_F1')
    best_global = vals('GlobalBest_Mean_F1')
    sens_mean = []
    for m in methods:
        row = summary_df[summary_df['Method'] == m]
        # For SomaticXGB the sensitivity column is the SNV-only candidate
        # check, which is not an all-record scan; do not draw it as one.
        if m == 'SomaticXGB':
            sens_mean.append(np.nan)
            continue
        if not row.empty and 'Sensitivity_BestPerSample_Mean_F1' in row.columns:
            v = row['Sensitivity_BestPerSample_Mean_F1'].iloc[0]
            sens_mean.append(float(v) if pd.notna(v) else np.nan)
        else:
            sens_mean.append(np.nan)
    sens_mean = np.array(sens_mean)

    ax_sum.bar(x - width, default_mean, width, label='default',
               color='#7f7f7f', edgecolor='k', lw=0.5)
    ax_sum.bar(x, best_global, width, label='best single threshold (global)',
               color='#1f77b4', edgecolor='k', lw=0.5)
    ax_sum.bar(x + width, best_oracle, width,
               label='best per-sample threshold (oracle)',
               color='#ff7f0e', edgecolor='k', lw=0.5)
    # Sensitivity marker for Strelka2/Mutect2/VarScan2.
    for i, m in enumerate(methods):
        if np.isfinite(sens_mean[i]):
            ax_sum.scatter([x[i] + width], [sens_mean[i]], marker='D', s=45,
                           color='#d62728', zorder=6,
                           label='all-record score-only best' if i == 0 else None)
    for i, m in enumerate(methods):
        ax_sum.text(x[i] - width, default_mean[i] + 0.012,
                    f'{default_mean[i]:.3f}', ha='center', va='bottom',
                    fontsize=8)
        ax_sum.text(x[i], best_global[i] + 0.012,
                    f'{best_global[i]:.3f}', ha='center', va='bottom',
                    fontsize=8)
        ax_sum.text(x[i] + width, best_oracle[i] + 0.012,
                    f'{best_oracle[i]:.3f}', ha='center', va='bottom',
                    fontsize=8)
        if np.isfinite(sens_mean[i]):
            ax_sum.text(x[i] + width, sens_mean[i] - 0.035,
                        f'{sens_mean[i]:.3f}', ha='center', va='top',
                        fontsize=7, color='#d62728')

    ax_sum.set_xticks(x)
    ax_sum.set_xticklabels([f'{m}\n({method_fields[m]})' for m in methods])
    ax_sum.set_ylim(0, 1.05)
    ax_sum.set_ylabel('Mean F1 across 12 samples')
    ax_sum.set_title('Default vs tuned F1 (100M reads, six platforms, two groups)',
                     fontsize=12, fontweight='bold')
    ax_sum.grid(axis='y', alpha=0.30, lw=0.6)
    ax_sum.legend(loc='upper left', ncol=2, fontsize=8, framealpha=0.9)
    ax_sum.text(0.5, -0.22,
                'Bars: default, best single global threshold, and best '
                'per-sample threshold (oracle). Red diamonds: best after '
                'ignoring the caller hard filters and thresholding the score '
                'only (Strelka2/Mutect2 all records; VarScan2 pass+failed).',
                transform=ax_sum.transAxes, ha='center', va='top', fontsize=7.5,
                color='#333333')

    # Shared legend for platform colours.
    handles = [Line2D([0], [0], color=PLAT_COLORS[p], lw=2, label=p)
               for _, p in PLATFORMS]
    handles.append(Line2D([0], [0], color='k', ls='--', lw=1.2,
                          label='all records, score-only (sensitivity)'))
    handles.append(Line2D([0], [0], color='k', ls=':', lw=1.2,
                          label='default mean F1'))
    fig.legend(handles=handles, loc='lower center', ncol=8, fontsize=9,
               frameon=False, bbox_to_anchor=(0.5, 0.005))
    fig.suptitle('B1. Baseline threshold sweeps on 100M-read test samples',
                 fontsize=14, fontweight='bold', y=0.995)

    png = os.path.join(FIGURES, 'B1_f1_vs_threshold.png')
    pdf = os.path.join(FIGURES, 'B1_f1_vs_threshold.pdf')
    fig.savefig(png, dpi=300, bbox_inches='tight')
    fig.savefig(pdf, bbox_inches='tight')
    plt.close(fig)
    log(f'[figure] wrote {png}')
    log(f'[figure] wrote {pdf}')


# ---------------------------------------------------------------------------
# Conclusions printed to the log
# ---------------------------------------------------------------------------

def print_conclusions(primary_df, sensitivity_df, summary_df, best_comparison):
    section('SUMMARY: default vs best')

    cols = ['Method', 'Default_Mean_F1', 'BestPerSample_Mean_F1',
            'BestPerSample_Mean_F1_Gain', 'BestPerSample_Max_F1_Gain',
            'GlobalBest_Type', 'GlobalBest_Threshold', 'GlobalBest_Mean_F1',
            'GlobalBest_Mean_F1_Gain']
    log(summary_df[cols].round(4).to_string(index=False))

    section('SUMMARY: recall / precision')
    cols2 = ['Method', 'Default_Mean_Recall', 'Default_Mean_Precision',
             'BestPerSample_Mean_Recall', 'BestPerSample_Mean_Precision',
             'GlobalAbs_Threshold', 'GlobalAbs_Mean_Recall',
             'GlobalAbs_Mean_Precision']
    log(summary_df[cols2].round(4).to_string(index=False))

    section('SUMMARY: all-record / SNV-only sensitivity')
    sens_cols = ['Method', 'Sensitivity_CallSet',
                 'Sensitivity_BestPerSample_Mean_F1',
                 'Sensitivity_BestPerSample_Mean_F1_Gain',
                 'Sensitivity_GlobalAbs_Threshold',
                 'Sensitivity_GlobalAbs_Mean_F1',
                 'Sensitivity_GlobalAbs_Mean_F1_Gain']
    have = [c for c in sens_cols if c in summary_df.columns]
    log(summary_df[have].round(4).to_string(index=False))

    section('SUMMARY: ranking by best F1 (mean across 12 samples)')
    rank = summary_df[['Method', 'Default_Mean_F1', 'BestPerSample_Mean_F1',
                       'GlobalBest_Mean_F1']].copy()
    rank['rank_default'] = rank['Default_Mean_F1'].rank(
        ascending=False, method='min').astype(int)
    rank['rank_best'] = rank['BestPerSample_Mean_F1'].rank(
        ascending=False, method='min').astype(int)
    rank = rank.sort_values('BestPerSample_Mean_F1', ascending=False)
    log(rank.round(4).to_string(index=False))

    # Per-sample win counts: SomaticXGB vs each caller at each one's best.
    section('SUMMARY: per-sample best-F1 wins (SomaticXGB vs each caller)')
    for method in ['Strelka2', 'Mutect2', 'VarScan2']:
        a = best_comparison[f'{method}_best_F1'].to_numpy(dtype=float)
        b = best_comparison['SomaticXGB_best_F1'].to_numpy(dtype=float)
        diff = b - a
        log(f'  SomaticXGB - {method}: mean={np.mean(diff):+.4f}, '
            f'median={np.median(diff):+.4f}, '
            f'SomaticXGB higher in {int((diff > 0).sum())}/{len(diff)}, '
            f'tied in {int((diff == 0).sum())}, lower in {int((diff < 0).sum())}')

    # VarScan2 precision before/after.
    section('SUMMARY: VarScan2 precision before and after SSC thresholding')
    vs = best_comparison[['Group', 'Platform',
                          'VarScan2_default_Precision',
                          'VarScan2_best_Precision',
                          'VarScan2_default_F1', 'VarScan2_best_F1',
                          'VarScan2_best_cut', 'VarScan2_best_scan_type']]
    log(vs.round(4).to_string(index=False))
    log('')
    log('  Default VarScan2 precision: mean %.4f (range %.4f-%.4f)'
        % (vs['VarScan2_default_Precision'].mean(),
           vs['VarScan2_default_Precision'].min(),
           vs['VarScan2_default_Precision'].max()))
    log('  Best-threshold VarScan2 precision: mean %.4f (range %.4f-%.4f)'
        % (vs['VarScan2_best_Precision'].mean(),
           vs['VarScan2_best_Precision'].min(),
           vs['VarScan2_best_Precision'].max()))

    # Key comparisons.
    section('CONCLUSIONS')
    s = summary_df.set_index('Method')
    def g(m, c):
        return float(s.loc[m, c])
    def global_text(m):
        typ = s.loc[m, 'GlobalBest_Type']
        thr = s.loc[m, 'GlobalBest_Threshold']
        val = s.loc[m, 'GlobalBest_Mean_F1']
        if typ == 'absolute':
            return f'global absolute cut {thr:g} gives {val:.4f}'
        if typ == 'quantile':
            return f'global quantile fraction {thr:g} gives {val:.4f}'
        return f'global best gives {val:.4f}'

    log(f'- Strelka2: default F1 {g("Strelka2","Default_Mean_F1"):.4f} -> '
        f'best per-sample {g("Strelka2","BestPerSample_Mean_F1"):.4f} '
        f'(gain {g("Strelka2","BestPerSample_Mean_F1_Gain"):+.4f}); '
        f'{global_text("Strelka2")}.')
    log(f'- Mutect2: default F1 {g("Mutect2","Default_Mean_F1"):.4f} -> '
        f'best per-sample {g("Mutect2","BestPerSample_Mean_F1"):.4f} '
        f'(gain {g("Mutect2","BestPerSample_Mean_F1_Gain"):+.4f}); '
        f'{global_text("Mutect2")}.')
    log(f'- VarScan2: default F1 {g("VarScan2","Default_Mean_F1"):.4f} -> '
        f'best per-sample {g("VarScan2","BestPerSample_Mean_F1"):.4f} '
        f'(gain {g("VarScan2","BestPerSample_Mean_F1_Gain"):+.4f}); '
        f'{global_text("VarScan2")}.')
    log(f'- SomaticXGB: default F1 {g("SomaticXGB","Default_Mean_F1"):.4f} -> '
        f'best per-sample {g("SomaticXGB","BestPerSample_Mean_F1"):.4f} '
        f'(gain {g("SomaticXGB","BestPerSample_Mean_F1_Gain"):+.4f}); '
        f'{global_text("SomaticXGB")}.')
    log('')
    log('Ranking by best per-sample F1: ' + ' > '.join(
        f'{m} ({g(m, "BestPerSample_Mean_F1"):.3f})'
        for m in s.sort_values('BestPerSample_Mean_F1', ascending=False).index))
    log('Ranking by default F1: ' + ' > '.join(
        f'{m} ({g(m, "Default_Mean_F1"):.3f})'
        for m in s.sort_values('Default_Mean_F1', ascending=False).index))
    log('')
    log('Caveats:')
    log('  * "Best per-sample" selects the threshold on the same test sample '
        '(oracle); it is an upper bound, not a deployable tuning procedure. '
        'The global threshold is the realistic single-choice comparison.')
    log('  * The all-record sensitivity deliberately ignores the caller hard '
        'filters and tunes the score only.  It is the most generous tuning '
        'opportunity the data allow.')
    log('  * SomaticXGB is evaluated on its own candidate set, which is much '
        'larger than any caller output (see KeptN in the tables); its '
        'advantage is not only a threshold effect.')
    log('  * SomaticXGB primary rows keep all pred candidates, matching '
        'make_fig3_data.py; the SNV-only sensitivity is in the sensitivity '
        'table and summary.')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--validate', action='store_true',
                    help='only validate parsing on one sample per method')
    args = ap.parse_args()

    os.makedirs(TABLES, exist_ok=True)
    os.makedirs(FIGURES, exist_ok=True)

    section('B1. Baseline threshold sweeps')
    log('Reviewer 2, Comment 6: comparable opportunity to tune the baselines')
    log(f'BASE = {BASE}')
    log(f'OUT  = {OUT}')
    log('Groups: ' + ', '.join(GROUPS))
    log('Platforms: ' + ', '.join(p for _, p in PLATFORMS))
    log('Truth/near rule: make_fig3_data.py (dist >= 50 truth; dist < 50 near; '
        'calls at near sites removed; FN = |truth| - TP)')
    log('SNV rule: len(REF)==1, len(ALT)==1, ALT not multiallelic; '
        'leading chr stripped.')

    truth_cache = {}
    section('Loading truth sets')
    for g in GROUPS:
        truth_cache[g] = load_truth(g)

    if args.validate:
        validate(truth_cache)
        return

    section('Parsing all samples and running sweeps')
    primary_rows, sensitivity_rows, parse_stats = \
        add_primary_and_sensitivity_rows(truth_cache)

    primary_df = pd.DataFrame(primary_rows)
    sensitivity_df = pd.DataFrame(sensitivity_rows)
    parse_df = pd.DataFrame(parse_stats)

    # Sort tables consistently: default, then quantile, then absolute.
    scan_order = {'default': 0, 'quantile': 1, 'absolute': 2}
    primary_df['_scan_order'] = primary_df['ScanType'].map(scan_order)
    primary_df = primary_df.sort_values(
        ['Method', 'Group', 'Platform', '_scan_order', 'TargetFraction', 'Cut'],
        na_position='first').drop(columns='_scan_order').reset_index(drop=True)
    sensitivity_df['_scan_order'] = sensitivity_df['ScanType'].map(scan_order)
    sensitivity_df = sensitivity_df.sort_values(
        ['Method', 'Group', 'Platform', 'CallSet', '_scan_order',
         'TargetFraction', 'Cut'],
        na_position='first').drop(columns='_scan_order').reset_index(drop=True)

    # ---- write the required tables ---------------------------------------
    sweep_path = os.path.join(TABLES, 'B1_threshold_sweep.csv')
    primary_df.to_csv(sweep_path, index=False)
    log(f'[table] wrote {sweep_path} ({len(primary_df)} rows)')

    sens_path = os.path.join(TABLES, 'B1_sensitivity_allrecords.csv')
    sensitivity_df.to_csv(sens_path, index=False)
    log(f'[table] wrote {sens_path} ({len(sensitivity_df)} rows)')

    summary_df = build_summary(primary_df, sensitivity_df)
    summary_path = os.path.join(TABLES, 'B1_method_summary.csv')
    summary_df.to_csv(summary_path, index=False)
    log(f'[table] wrote {summary_path} ({len(summary_df)} rows)')

    best_comparison = build_comparison(primary_df, sensitivity_df)
    comp_path = os.path.join(TABLES, 'B1_best_f1_comparison.csv')
    best_comparison.to_csv(comp_path, index=False)
    log(f'[table] wrote {comp_path} ({len(best_comparison)} rows)')

    # ---- parse stats table (not required, but useful for the log/audit) ---
    keep_cols = [c for c in [
        'analysis', 'callset', 'group', 'platform', 'sample', 'score_field',
        'score_source', 'n_records', 'n_pass_filter', 'n_calls',
        'n_nonpass_skipped', 'n_indel_skipped', 'n_multiallelic_skipped',
        'n_near_skipped', 'n_missing_score', 'n_duplicate_vars',
        'n_non_snv', 'n_failed_records', 'n_failed_calls',
        'score_min', 'score_p10', 'score_median', 'score_p90', 'score_max']
        if c in parse_df.columns]
    parse_df = parse_df[keep_cols]
    parse_path = os.path.join(TABLES, 'B1_parse_stats.csv')
    parse_df.to_csv(parse_path, index=False)
    log(f'[table] wrote {parse_path} ({len(parse_df)} rows)')

    # Compact score-distribution table, pooled over samples.
    dist_rows = []
    for (analysis, callset, method_field), grp in parse_df.groupby(
            ['analysis', 'callset', 'score_field']):
        dist_rows.append({
            'Analysis': analysis,
            'CallSet': callset,
            'ScoreField': method_field,
            'N_samples': len(grp),
            'Calls_min': int(grp['n_calls'].min()),
            'Calls_median': int(grp['n_calls'].median()),
            'Calls_max': int(grp['n_calls'].max()),
            'Score_min_min': grp['score_min'].min(),
            'Score_min_median': grp['score_min'].median(),
            'Score_p10_median': grp['score_p10'].median(),
            'Score_median_median': grp['score_median'].median(),
            'Score_p90_median': grp['score_p90'].median(),
            'Score_max_median': grp['score_max'].median(),
            'Score_max_max': grp['score_max'].max(),
        })
    dist_df = pd.DataFrame(dist_rows)
    dist_path = os.path.join(TABLES, 'B1_score_distribution.csv')
    dist_df.to_csv(dist_path, index=False)
    log(f'[table] wrote {dist_path} ({len(dist_df)} rows)')

    # ---- default F1 table in the log (fig3 replication check) -----------
    section('Default F1 by group/platform (fig3 replication check)')
    d = primary_df[primary_df['ScanType'] == 'default']
    pivot = d.pivot_table(index=['Group', 'Platform'], columns='Method',
                          values='F1', aggfunc='mean')
    log(pivot.round(4).to_string())
    log('')
    log('Mean default metrics across the 12 samples:')
    log(d.groupby('Method')[['Recall', 'Precision', 'F1']].mean().round(4)
        .to_string())

    # ---- score distributions from the sweep table ------------------------
    section('Score distributions of the default call sets')
    log(score_distribution_table(primary_rows).round(4).to_string(index=False))
    log('')
    log('Per-sample default score range (MinRetainedScore, i.e. the lowest '
        'score kept by the default rule):')
    for method in METHODS:
        sub = d[d['Method'] == method]
        log(f'  {method}: min={sub["MinRetainedScore"].min():.4g}, '
            f'median={sub["MinRetainedScore"].median():.4g}, '
            f'max={sub["MinRetainedScore"].max():.4g}; '
            f'calls={int(sub["KeptN"].min())}-{int(sub["KeptN"].max())}')

    # ---- best table in the log -------------------------------------------
    section('Best per-sample threshold (oracle) by method')
    rows = []
    for method in METHODS:
        sub = primary_df[primary_df['Method'] == method]
        vals = []
        for (gg, pp, ss), ssub in sub.groupby(['Group', 'Platform', 'Sample']):
            b = pick_best(ssub.to_dict('records'))
            if b is not None:
                vals.append(b)
        vdf = pd.DataFrame(vals)
        rows.append({
            'Method': method,
            'Best_F1_mean': vdf['F1'].mean(),
            'Best_F1_min': vdf['F1'].min(),
            'Best_F1_max': vdf['F1'].max(),
            'Best_Recall_mean': vdf['Recall'].mean(),
            'Best_Precision_mean': vdf['Precision'].mean(),
            'Median_Cut': vdf['Cut'].median(),
            'Median_MinRetainedScore': vdf['MinRetainedScore'].median(),
        })
    log(pd.DataFrame(rows).round(4).to_string(index=False))

    # ---- comparison preview ----------------------------------------------
    section('Best F1 comparison (per group/platform)')
    comp_cols = ['Group', 'Platform', 'TruthN',
                 'Strelka2_default_F1', 'Strelka2_best_F1',
                 'Mutect2_default_F1', 'Mutect2_best_F1',
                 'VarScan2_default_F1', 'VarScan2_best_F1',
                 'SomaticXGB_default_F1', 'SomaticXGB_best_F1']
    log(best_comparison[comp_cols].round(4).to_string(index=False))

    # ---- figure ----------------------------------------------------------
    section('Making figure')
    make_figure(primary_df, sensitivity_df, summary_df)

    # ---- conclusions -----------------------------------------------------
    print_conclusions(primary_df, sensitivity_df, summary_df, best_comparison)

    section('DONE')


if __name__ == '__main__':
    main()
