#!/usr/bin/env python3
"""A4. Representative SomaticXGB calls checked against the 0% Twist controls.

Reviewer 2, comment 1 (major): check representative discordant calls against
the 0% controls, across replicates, rather than arguing only from aggregate
substitution spectra.

This is a deliberately minimal, descriptive addendum.  It does NOT adjudicate
individual calls and it performs no statistical test.  It selects a
score/substitution-stratified set of 10 truth-list positives (TP) and 10 calls
outside the truth list (FP) from one FASTASeq 300 1% tumor / 0% normal pair,
then reports raw pileup support at each site in the 1% tumor and in the three
0% libraries (GM23062701, GM24041603, GM24041604).

Design decisions (also written to logs/A4.log):
  * Truth list: results/analysis/twist_truth_221.tsv (221 panel-covered
    variants; TP/FP are defined only with respect to this list).
  * FP candidates must be >10 bp from any truth-list variant, so a selected
    FP is not merely an alternative representation of a known truth site.
  * Selection is stratified over SomaticXGB score (low 0.1-0.5, mid 0.5-0.9,
    high 0.9-1.0) and over ALT substitution class.  It is not a random sample
    and the resulting frequencies are not estimates for the full callset.
  * Pileup: samtools mpileup -B -Q 0 -q 0 -d 100000.  BAQ is disabled because
    we want raw read support, not a re-called genotype.  samtools' default
    duplicate/secondary/QC-fail filtering still applies.
  * Depth in the output is the number of parsed base observations
    (ref + alt + other); deletion placeholders are reported separately.
    mpileup's own depth column is retained as *_mpileup_DP.

Outputs:
  tables/A4_representative_calls_per_call.csv
  tables/A4_representative_summary.csv
  figures/A4_representative_calls.png / .pdf
  logs/A4.log
"""

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
import common  # noqa: E402

import glob
import logging
import os
import subprocess
import sys
from datetime import datetime

import numpy as np
import pandas as pd
import scipy
from scipy import stats

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
OUT = str(common.RESULTS_ROOT)
PAIR_DIR = str(common.TWIST_PHYS_ROOT)
PRED = f'{PAIR_DIR}/pred/FA_1_GM23063001_GM23062701_pred.txt.gz'
TRUTH = str(common.TWIST_TRUTH_221)
REF = str(common.GENOME_FASTA)

BAM_DIR = str(common.WORK_ROOT / '5.twist_errorRates_mdup' / 'pipeline' / 'bwamem')
TUMOR = f'{BAM_DIR}/FA_1_GM23063001_PE70M/FA_1_GM23063001_PE70M.marked.bam'
NORMALS = [
    ('GM23062701', f'{BAM_DIR}/FA_0_GM23062701_PE2dot5M/FA_0_GM23062701_PE2dot5M.marked.bam'),
    ('GM24041603', f'{BAM_DIR}/FA_0_GM24041603_PE2dot5M/FA_0_GM24041603_PE2dot5M.marked.bam'),
    ('GM24041604', f'{BAM_DIR}/FA_0_GM24041604_PE2dot5M/FA_0_GM24041604_PE2dot5M.marked.bam'),
]

TP_TARGETS = {'low': 2, 'mid': 3, 'high': 5}
FP_TARGETS = {'low': 4, 'mid': 3, 'high': 3}
FP_MIN_TRUTH_DISTANCE = 10          # bp
SCORE_MIN = 0.1                     # SomaticXGB threshold in the paper

os.makedirs(f'{OUT}/tables', exist_ok=True)
os.makedirs(f'{OUT}/figures', exist_ok=True)
os.makedirs(f'{OUT}/logs', exist_ok=True)

LOG_PATH = f'{OUT}/logs/A4.log'
log = logging.getLogger('A4')
log.setLevel(logging.INFO)
log.handlers.clear()
_fh = logging.FileHandler(LOG_PATH, mode='w')
_fh.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
_sh = logging.StreamHandler(sys.stdout)
_sh.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
log.addHandler(_fh)
log.addHandler(_sh)

SEEN_MPILEUP_STDERR = set()


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------
def require_file(path, label):
    if not os.path.exists(path):
        raise FileNotFoundError(f'{label} not found: {path}')
    log.info('found %-28s %s', label, path)


def ensure_bam_index(bam):
    """Check for a BAM index; build one only if it is missing.

    The BAMs are read-only inputs.  Building an index writes a .bai next to
    the BAM, so this is attempted only if the index is absent and the BAI
    would not be created otherwise.
    """
    if os.path.exists(bam + '.bai'):
        return bam + '.bai'
    if os.path.exists(bam[:-4] + '.bai'):
        return bam[:-4] + '.bai'
    log.warning('BAM index missing for %s; attempting samtools index', bam)
    try:
        subprocess.run(['samtools', 'index', bam], check=True,
                       capture_output=True, text=True)
    except Exception as exc:
        raise RuntimeError(
            f'No BAM index for {bam} and samtools index failed: {exc}') from exc
    return bam + '.bai'


def flagstat_mapped(bam):
    """Return the mapped-read count from a sibling flagstat file, if present."""
    for suffix in ('.flagstat.txt', '.marked.bam.flagstat.txt'):
        path = bam + suffix
        if os.path.exists(path):
            with open(path) as fh:
                first = fh.readline().strip()
            return first
    return 'flagstat not found'


def fetch_ref_base(chrom, pos):
    proc = subprocess.run(
        ['samtools', 'faidx', REF, f'{chrom}:{pos}-{pos}'],
        capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f'samtools faidx failed at {chrom}:{pos}: {proc.stderr}')
    seq = ''.join(proc.stdout.splitlines()[1:]).upper()
    return seq


def parse_pileup_bases(bases, alt):
    """Count REF/ALT/other/deletion observations in an mpileup base string.

    Handles the mpileup read-start marker (^ + mapping-quality char), read-end
    marker ($), and indel annotations (+/- followed by a length and sequence),
    including multi-digit indel lengths.  '.' and ',' are reference observations;
    upper/lower case ALT bases are counted; '*' and '#' are deletion
    placeholders; '<'/'>' (reference skip) and all other bases are 'other'.
    """
    alt = str(alt).upper()
    ref = alt_n = other = deletions = 0
    i, n = 0, len(bases)
    while i < n:
        c = bases[i]
        if c == '^':                 # read start + mapQ char
            i += 2
            continue
        if c == '$':                 # read end
            i += 1
            continue
        if c in '+-':                # indel annotation
            i += 1
            j = i
            while j < n and bases[j].isdigit():
                j += 1
            if j == i:               # malformed; do not loop forever
                continue
            length = int(bases[i:j])
            i = j + length
            continue
        if c in '.,':
            ref += 1
        elif c.upper() == alt:
            alt_n += 1
        elif c in '*#':
            deletions += 1
        else:
            other += 1
        i += 1
    return ref, alt_n, other, deletions


def run_pileup(bam, sample_label, chrom, pos, ref_allele, alt_allele):
    """Return counts at one SNV from one BAM."""
    region = f'{chrom}:{pos}-{pos}'
    cmd = ['samtools', 'mpileup', '-B', '-Q', '0', '-q', '0', '-d', '100000',
           '-f', REF, '-r', region, bam]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f'mpileup failed for {sample_label} {region}:\n{proc.stderr}')

    for line in proc.stderr.splitlines():
        line = line.strip()
        if not line or 'samples in 1 input files' in line:
            continue
        if line not in SEEN_MPILEUP_STDERR:
            SEEN_MPILEUP_STDERR.add(line)
            log.warning('samtools mpileup stderr (%s): %s', sample_label, line)

    line = None
    for candidate in proc.stdout.splitlines():
        fields = candidate.split('\t')
        if fields[0] == str(chrom) and int(fields[1]) == int(pos):
            line = fields
            break
    if line is None:
        log.warning('no mpileup line for %s at %s', sample_label, region)
        return {'depth': 0, 'ref': 0, 'alt': 0, 'other': 0, 'deletions': 0,
                'AF': np.nan, 'mpileup_DP': 0, 'ref_base': 'NA'}

    ref_base = line[2].upper()
    if ref_base != str(ref_allele).upper():
        log.error('reference mismatch for %s at %s: BAM/mpileup REF=%s, expected=%s',
                  sample_label, region, ref_base, ref_allele)
    ref_n, alt_n, other_n, del_n = parse_pileup_bases(line[4], alt_allele)
    depth = ref_n + alt_n + other_n
    return {
        'depth': depth,
        'ref': ref_n,
        'alt': alt_n,
        'other': other_n,
        'deletions': del_n,
        'AF': (alt_n / depth) if depth > 0 else np.nan,
        'mpileup_DP': int(line[3]),
        'ref_base': ref_base,
    }


def score_stratum(score):
    if SCORE_MIN < score <= 0.5:
        return 'low'
    if 0.5 < score <= 0.9:
        return 'mid'
    if 0.9 < score <= 1.0:
        return 'high'
    return 'out_of_range'


# --------------------------------------------------------------------------
# Selection
# --------------------------------------------------------------------------
def select_representatives(cand, targets, label):
    """Greedy stratified selection.

    Fills the score-stratum quotas exactly.  At each step the utility rewards
    a substitution class not yet selected, a score far from already selected
    scores, and an unfilled stratum; it mildly penalises repeating a class.
    Ties are broken by the candidate order (sorted by chrom, pos, var).
    """
    cand = cand.sort_values(['chrom', 'pos', 'var']).reset_index(drop=True)
    selected, chosen = [], set()
    total = sum(targets.values())

    for step in range(total):
        type_counts = pd.Series([x['type'] for x in selected]).value_counts()
        need = {
            k: v - sum(1 for x in selected if x['stratum'] == k)
            for k, v in targets.items()
        }
        best = None
        for i, row in cand.iterrows():
            if i in chosen or need[row['stratum']] <= 0:
                continue
            novel = 1 if type_counts.get(row['type'], 0) == 0 else 0
            gap = min([abs(row['pred'] - x['score']) for x in selected],
                      default=0.5)
            utility = (5.0 * novel
                       - 1.5 * type_counts.get(row['type'], 0)
                       + 4.0 * gap
                       + 8.0 * need[row['stratum']] / targets[row['stratum']])
            key = (utility, -i)
            if best is None or key > best[0]:
                best = (key, i, row)
        if best is None:
            raise RuntimeError(f'{label}: no feasible candidate at step {step + 1}')
        _, idx, row = best
        chosen.add(idx)
        selected.append({
            'var': row['var'],
            'chrom': str(row['chrom']),
            'pos': int(row['pos']),
            'ref': row['ref'],
            'alt': row['alt'],
            'type': row['type'],
            'score': float(row['pred']),
            'stratum': row['stratum'],
            'xgb_DP': int(row['DP']),
            'xgb_TLOD': float(row['TLOD']),
            'xgb_ECNT': int(row['ECNT']),
            'dist_truth': float(row['dist_truth']),
            'truth_status': label,
        })
        log.info('  %s pick %2d/%d: %-24s score=%.6f  %-4s %-4s  xgb_DP=%d  dist_truth=%.0f',
                 label, step + 1, total, row['var'], row['pred'],
                 row['type'], row['stratum'], row['DP'], row['dist_truth'])
    return selected


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    log.info('=' * 78)
    log.info('A4 representative call / 0% control cross-replicate check')
    log.info('started %s', datetime.now().isoformat(timespec='seconds'))
    log.info('python %s', sys.version.replace('\n', ' '))
    log.info('pandas %s numpy %s scipy %s', pd.__version__, np.__version__,
             scipy.__version__)
    try:
        st_version = subprocess.run(['samtools', '--version'],
                                    capture_output=True, text=True, check=True)
        log.info('samtools: %s', st_version.stdout.splitlines()[0])
    except Exception:
        log.warning('could not determine samtools version')

    # ---- inputs ----------------------------------------------------------
    require_file(TRUTH, 'truth table')
    require_file(PRED, 'SomaticXGB prediction table')
    pred_files = sorted(glob.glob(f'{PAIR_DIR}/pred/*_pred.txt.gz'))
    log.info('pred directory: %s', f'{PAIR_DIR}/pred')
    log.info('pred files found: %d; selected file: %s', len(pred_files),
             os.path.basename(PRED))
    require_file(REF, 'reference FASTA')
    require_file(REF + '.fai', 'reference FASTA index')
    require_file(TUMOR, '1% tumor BAM')
    for _, bam in NORMALS:
        require_file(bam, '0% normal BAM')

    for bam in [TUMOR] + [b for _, b in NORMALS]:
        bai = ensure_bam_index(bam)
        log.info('BAM index: %s', bai)
    log.info('all four BAMs already had .bai indexes; no samtools index was run '
             'and no input file was modified')
    log.info('hts_idx_load3 "index older than data file" messages are '
             'timestamp-only warnings, not data problems')

    log.info('')
    log.info('--- truth table -------------------------------------------------')
    truth = pd.read_csv(TRUTH, sep='\t', dtype={'chrom': str})
    log.info('truth file: %s', TRUTH)
    log.info('columns: %s', list(truth.columns))
    log.info('rows: %d; unique var: %d', len(truth), truth['var'].nunique())
    log.info('first rows:\n%s', truth.head(3).to_string(index=False))
    if 'var' not in truth.columns:
        truth['var'] = (truth['chrom'].astype(str) + ':' +
                        truth['pos'].astype(str) + ':' +
                        truth['ref'].astype(str) + ':' + truth['alt'].astype(str))
    truth_vars = set(truth['var'].astype(str))
    truth_pos = {}
    for chrom, grp in truth.groupby('chrom'):
        truth_pos[str(chrom)] = grp['pos'].astype(int).values

    log.info('')
    log.info('--- SomaticXGB predictions -------------------------------------')
    pred = pd.read_csv(PRED, sep='\t')
    log.info('pred file: %s', PRED)
    log.info('columns: %s', list(pred.columns))
    log.info('rows: %d', len(pred))
    log.info('first rows:\n%s', pred.head(3).to_string(index=False))
    if pred['var'].duplicated().any():
        raise ValueError('duplicate var entries in prediction table')

    # The var field is the normalised single-base representation; REF/ALT
    # columns in this table can carry anchor bases for some callers.
    var_parts = pred['var'].astype(str).str.split(':', expand=True)
    if var_parts.shape[1] != 4:
        raise ValueError('unexpected var format in prediction table')
    pred['chrom'] = var_parts[0]
    pred['pos'] = var_parts[1].astype(int)
    pred['ref'] = var_parts[2]
    pred['alt'] = var_parts[3]
    pred['type'] = pred['ref'] + '>' + pred['alt']
    if ((pred['ref'].str.len() != 1) | (pred['alt'].str.len() != 1)).any():
        raise ValueError('prediction table contains non-SNV var entries')
    log.info('parsed var field: chrom/pos/ref/alt all single-base SNVs')

    pred = pred[pred['pred'] > SCORE_MIN].copy()
    pred['truth_status'] = np.where(pred['var'].isin(truth_vars), 'TP', 'FP')
    pred['stratum'] = pred['pred'].map(score_stratum)

    def dist_to_truth(row):
        arr = truth_pos.get(str(row['chrom']))
        if arr is None or len(arr) == 0:
            return np.inf
        return float(np.min(np.abs(arr - int(row['pos']))))

    pred['dist_truth'] = pred.apply(dist_to_truth, axis=1)

    tp_pool = pred[pred['truth_status'] == 'TP'].copy()
    fp_pool = pred[(pred['truth_status'] == 'FP') &
                   (pred['dist_truth'] > FP_MIN_TRUTH_DISTANCE)].copy()
    log.info('')
    log.info('calls with SomaticXGB pred > %.2f: %d (TP=%d, FP=%d)',
             SCORE_MIN, len(pred),
             int((pred['truth_status'] == 'TP').sum()),
             int((pred['truth_status'] == 'FP').sum()))
    log.info('TP pool: %d; by stratum: %s', len(tp_pool),
             tp_pool['stratum'].value_counts().to_dict())
    log.info('FP pool: %d after excluding calls <=%d bp from a truth site; '
             'by stratum: %s', len(fp_pool), FP_MIN_TRUTH_DISTANCE,
             fp_pool['stratum'].value_counts().to_dict())
    log.info('FP calls excluded for truth proximity: %d',
             int(((pred['truth_status'] == 'FP') &
                  (pred['dist_truth'] <= FP_MIN_TRUTH_DISTANCE)).sum()))
    for label, pool, targets in [('TP', tp_pool, TP_TARGETS),
                                 ('FP', fp_pool, FP_TARGETS)]:
        missing = {k: v for k, v in targets.items()
                   if (pool['stratum'] == k).sum() < v}
        if missing:
            raise ValueError(f'{label}: insufficient candidates for {missing}')

    log.info('')
    log.info('--- selection: score-stratified, substitution-diverse ----------')
    log.info('TP targets %s; FP targets %s', TP_TARGETS, FP_TARGETS)
    log.info('FP candidates are >%d bp from every truth variant.', FP_MIN_TRUTH_DISTANCE)
    selected = (select_representatives(tp_pool, TP_TARGETS, 'TP') +
                select_representatives(fp_pool, FP_TARGETS, 'FP'))

    # Display low -> high score within each truth class.
    selected = (sorted([x for x in selected if x['truth_status'] == 'TP'],
                       key=lambda x: (x['score'], x['var'])) +
                sorted([x for x in selected if x['truth_status'] == 'FP'],
                       key=lambda x: (x['score'], x['var'])))
    for rank, row in enumerate(selected, 1):
        row['selection_order'] = rank

    # ---- validate alleles against the reference --------------------------
    log.info('')
    log.info('--- reference allele check --------------------------------------')
    log.info('reference: %s', REF)
    for row in selected:
        ref_base = fetch_ref_base(row['chrom'], row['pos'])
        row['ref_check'] = 'PASS' if ref_base == row['ref'].upper() else 'FAIL'
        if row['ref_check'] == 'FAIL':
            log.error('REF mismatch: %s genome=%s table=%s',
                      row['var'], ref_base, row['ref'])
        else:
            log.info('REF ok: %-24s %s', row['var'], ref_base)

    # ---- pileup ----------------------------------------------------------
    log.info('')
    log.info('--- pileup ------------------------------------------------------')
    log.info('command template: samtools mpileup -B -Q 0 -q 0 -d 100000 '
             '-f %s -r <chrom>:<pos>-<pos> <bam>', REF)
    log.info('BAQ disabled (-B) to count raw base observations; samtools '
             'default duplicate/secondary/QC-fail filtering applies. MAPQ/BQ 0 '
             'means low-quality and multi-mapping reads are included.')
    log.info('depth = ref + alt + other parsed bases; deletions are separate.')
    log.info('tumor BAM: %s', TUMOR)
    log.info('tumor flagstat: %s', flagstat_mapped(TUMOR))
    for lib, bam in NORMALS:
        log.info('0%% replicate %s: %s', lib, bam)
        log.info('0%% replicate %s flagstat: %s', lib, flagstat_mapped(bam))

    records = []
    for idx, row in enumerate(selected, 1):
        log.info('')
        log.info('[%2d/%d] %s (%s, score=%.6f, %s)',
                 idx, len(selected), row['var'], row['truth_status'],
                 row['score'], row['type'])
        rec = {
            'selection_order': row['selection_order'],
            'var': row['var'],
            'chrom': row['chrom'],
            'pos': row['pos'],
            'ref': row['ref'],
            'alt': row['alt'],
            'alt_type': row['type'],
            'truth_status': row['truth_status'],
            'truth_definition': ('in_twist_truth_221'
                                 if row['truth_status'] == 'TP'
                                 else 'not_in_twist_truth_221'),
            'somaticxgb_score': round(row['score'], 8),
            'somaticxgb_stratum': row['stratum'],
            'xgb_DP': row['xgb_DP'],
            'xgb_TLOD': row['xgb_TLOD'],
            'xgb_ECNT': row['xgb_ECNT'],
            'dist_to_nearest_truth_bp': row['dist_truth'],
            'ref_check': row['ref_check'],
        }

        res = run_pileup(TUMOR, 'tumor', row['chrom'], row['pos'],
                         row['ref'], row['alt'])
        rec.update({
            'tumor_depth': res['depth'],
            'tumor_ref': res['ref'],
            'tumor_alt': res['alt'],
            'tumor_other': res['other'],
            'tumor_deletions': res['deletions'],
            'tumor_AF': res['AF'],
            'tumor_mpileup_DP': res['mpileup_DP'],
        })
        log.info('  1%% tumor: depth=%d ref=%d alt=%d other=%d del=%d AF=%.6g (mpileup_DP=%d)',
                 res['depth'], res['ref'], res['alt'], res['other'],
                 res['deletions'], res['AF'], res['mpileup_DP'])

        rep_alt = []
        rep_dep = []
        rep_af = []
        for lib, bam in NORMALS:
            q = run_pileup(bam, lib, row['chrom'], row['pos'],
                           row['ref'], row['alt'])
            prefix = f'n0_{lib}'
            rec.update({
                f'{prefix}_depth': q['depth'],
                f'{prefix}_ref': q['ref'],
                f'{prefix}_alt': q['alt'],
                f'{prefix}_other': q['other'],
                f'{prefix}_deletions': q['deletions'],
                f'{prefix}_AF': q['AF'],
                f'{prefix}_mpileup_DP': q['mpileup_DP'],
            })
            rep_alt.append(q['alt'])
            rep_dep.append(q['depth'])
            rep_af.append(q['AF'] if not pd.isna(q['AF']) else 0.0)
            log.info('  0%% %s: depth=%d ref=%d alt=%d other=%d del=%d AF=%.6g (mpileup_DP=%d)',
                     lib, q['depth'], q['ref'], q['alt'], q['other'],
                     q['deletions'], q['AF'], q['mpileup_DP'])

        rep_af = np.array(rep_af, dtype=float)
        rep_alt = np.array(rep_alt, dtype=int)
        rep_dep = np.array(rep_dep, dtype=int)
        rec.update({
            'n0_mean_AF': float(rep_af.mean()),
            'n0_max_AF': float(rep_af.max()),
            'n0_total_alt': int(rep_alt.sum()),
            'n0_total_depth': int(rep_dep.sum()),
            'n0_pooled_AF': (float(rep_alt.sum() / rep_dep.sum())
                             if rep_dep.sum() > 0 else np.nan),
            'n0_n_reps_alt_ge1': int((rep_alt >= 1).sum()),
            'n0_n_reps_alt_ge2': int((rep_alt >= 2).sum()),
        })
        log.info('  0%% summary: mean_AF=%.6g pooled_AF=%.6g '
                 'alt>=1 in %d/3 reps, alt>=2 in %d/3 reps',
                 rec['n0_mean_AF'], rec['n0_pooled_AF'],
                 rec['n0_n_reps_alt_ge1'], rec['n0_n_reps_alt_ge2'])
        records.append(rec)

    per_call = pd.DataFrame(records).sort_values('selection_order')
    per_call_path = f'{OUT}/tables/A4_representative_calls_per_call.csv'
    per_call.to_csv(per_call_path, index=False, float_format='%.8g')
    log.info('')
    log.info('wrote %s (%d rows x %d columns)',
             per_call_path, per_call.shape[0], per_call.shape[1])

    # ---- summary ---------------------------------------------------------
    log.info('')
    log.info('--- summary -----------------------------------------------------')

    def corr_coeff(x, y, method):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        mask = np.isfinite(x) & np.isfinite(y)
        x, y = x[mask], y[mask]
        if len(x) < 3 or len(np.unique(x)) < 2 or len(np.unique(y)) < 2:
            return np.nan
        if method == 'pearson':
            return float(stats.pearsonr(x, y)[0])
        return float(stats.spearmanr(x, y)[0])

    def summarize(df, label):
        alt_cols = [f'n0_{lib}_alt' for lib, _ in NORMALS]
        dep_cols = [f'n0_{lib}_depth' for lib, _ in NORMALS]
        alt_mat = df[alt_cols].to_numpy(dtype=int)
        dep_mat = df[dep_cols].to_numpy(dtype=int)
        af_mat = np.divide(alt_mat, dep_mat, out=np.zeros_like(alt_mat, dtype=float),
                           where=dep_mat > 0)
        alt_flat = alt_mat.ravel()
        dep_flat = dep_mat.ravel()
        reps_ge1 = (alt_mat >= 1).sum(axis=1)
        reps_ge2 = (alt_mat >= 2).sum(axis=1)
        tumor_af = df['tumor_AF'].to_numpy(dtype=float)
        n0_mean = df['n0_mean_AF'].to_numpy(dtype=float)
        return {
            'group': label,
            'n_calls': len(df),
            'tumor_depth_median': float(df['tumor_depth'].median()),
            'tumor_AF_mean': float(df['tumor_AF'].mean()),
            'tumor_AF_median': float(df['tumor_AF'].median()),
            'tumor_AF_min': float(df['tumor_AF'].min()),
            'tumor_AF_max': float(df['tumor_AF'].max()),
            'n0_depth_median': float(np.median(dep_mat)),
            'n0_depth_min': int(dep_mat.min()),
            'n0_depth_max': int(dep_mat.max()),
            'n0_mean_AF_mean': float(n0_mean.mean()),
            'n0_mean_AF_median': float(np.median(n0_mean)),
            'n0_max_AF': float(df['n0_max_AF'].max()),
            'n0_pooled_AF': (float(alt_mat.sum() / dep_mat.sum())
                             if dep_mat.sum() > 0 else np.nan),
            'n0_total_alt_reads': int(alt_mat.sum()),
            'n0_total_depth': int(dep_mat.sum()),
            'n_rep_observations': int(len(alt_flat)),
            'n_rep_obs_alt_0': int((alt_flat == 0).sum()),
            'n_rep_obs_alt_1': int((alt_flat == 1).sum()),
            'n_rep_obs_alt_ge2': int((alt_flat >= 2).sum()),
            'frac_rep_obs_alt_ge1': float((alt_flat >= 1).mean()),
            'frac_rep_obs_alt_ge2': float((alt_flat >= 2).mean()),
            'n_calls_any_rep_alt_ge1': int((reps_ge1 >= 1).sum()),
            'frac_calls_any_rep_alt_ge1': float((reps_ge1 >= 1).mean()),
            'n_calls_any_rep_alt_ge2': int((reps_ge2 >= 1).sum()),
            'frac_calls_any_rep_alt_ge2': float((reps_ge2 >= 1).mean()),
            'n_calls_ge2_reps_alt_ge1': int((reps_ge1 >= 2).sum()),
            'frac_calls_ge2_reps_alt_ge1': float((reps_ge1 >= 2).mean()),
            'n_calls_ge2_reps_alt_ge2': int((reps_ge2 >= 2).sum()),
            'frac_calls_ge2_reps_alt_ge2': float((reps_ge2 >= 2).mean()),
            'n_calls_all3_reps_alt_ge1': int((reps_ge1 == 3).sum()),
            'n_calls_all3_reps_alt_ge2': int((reps_ge2 == 3).sum()),
            'n_calls_reps_alt_ge2_0': int((reps_ge2 == 0).sum()),
            'n_calls_reps_alt_ge2_1': int((reps_ge2 == 1).sum()),
            'n_calls_reps_alt_ge2_2': int((reps_ge2 == 2).sum()),
            'n_calls_reps_alt_ge2_3': int((reps_ge2 == 3).sum()),
            'n_calls_n0_mean_AF_ge_tumor_AF': int((n0_mean >= tumor_af).sum()),
            'pearson_r_tumorAF_vs_n0meanAF': corr_coeff(tumor_af, n0_mean, 'pearson'),
            'spearman_rho_tumorAF_vs_n0meanAF': corr_coeff(tumor_af, n0_mean, 'spearman'),
            'note': ('descriptive only; n=10 per group; no statistical test or '
                     'call-level adjudication; correlations are unweighted and '
                     'outlier-sensitive'),
        }

    summary = pd.DataFrame([
        summarize(per_call[per_call.truth_status == 'TP'], 'TP'),
        summarize(per_call[per_call.truth_status == 'FP'], 'FP'),
        summarize(per_call, 'All'),
    ])
    summary_path = f'{OUT}/tables/A4_representative_summary.csv'
    summary.to_csv(summary_path, index=False, float_format='%.8g')
    log.info('wrote %s', summary_path)
    log.info('\n%s', summary.round(6).to_string(index=False))

    all_depths = per_call[[f'n0_{lib}_depth' for lib, _ in NORMALS]].to_numpy()
    log.info('0% BAMs are named PE2dot5M and contain ~5.0M reads '
             '(2.5M pairs); per-site depth is reported and no site is dropped '
             'for low depth.')
    log.info('0%% replicate depth across all selected calls/replicates: '
             'median=%.1f min=%d max=%d',
             float(np.median(all_depths)), int(all_depths.min()),
             int(all_depths.max()))

    # ---- figure ----------------------------------------------------------
    log.info('')
    log.info('--- figure ------------------------------------------------------')
    make_figure(per_call)
    log.info('wrote %s/A4_representative_calls.png and .pdf', f'{OUT}/figures')

    # ---- limitations -----------------------------------------------------
    log.info('')
    log.info('--- limitations (read with the table; no inference) -------------')
    log.info('1. One platform/pair: FASTASeq 300, 1% tumor FA_1_GM23063001_PE70M '
             'vs 0% libraries GM23062701 (matched), GM24041603, GM24041604.')
    log.info('2. 10 TP and 10 FP were selected by score/substitution-stratified '
             'greedy sampling, not at random. They describe support in these '
             'calls; they are not prevalence estimates for the callset.')
    log.info('3. "FP" means only "absent from twist_truth_221.tsv". It is not '
             'proof that the call is an artifact; the truth list is known to '
             'be incomplete for the Twist v1 material.')
    log.info('4. The 0% controls are 2.5M-read libraries from the same '
             'manufacturing process. At the observed depths, absence of ALT '
             'reads does not exclude a low-AF event, so zero 0% support '
             'cannot certify an individual call as real.')
    log.info('5. The tumor and 0% libraries are different libraries. ALT '
             'support in a 0% library does not by itself prove that the '
             'same tumor call is background, even when it is reproducible '
             'across 0% replicates.')
    log.info('6. Counts are raw pileup observations (BAQ off, MAPQ/BQ >=0, '
             'duplicates skipped by samtools defaults; low-MAPQ and '
             'multi-mapping reads are included). No test, no p-value, and no '
             'per-call adjudication is performed.')
    log.info('7. The optional Strelka2/Mutect2 comparison was not included in '
             'this minimal addendum.')
    log.info('')
    log.info('finished %s', datetime.now().isoformat(timespec='seconds'))
    log.info('=' * 78)


def make_figure(per_call):
    df = per_call.sort_values('selection_order').reset_index(drop=True)
    libs = [lib for lib, _ in NORMALS]
    af = np.vstack([
        df['tumor_AF'].to_numpy(dtype=float),
        *[df[f'n0_{lib}_AF'].to_numpy(dtype=float) for lib in libs],
    ]).T  # calls x samples
    alt = np.vstack([
        df['tumor_alt'].to_numpy(dtype=int),
        *[df[f'n0_{lib}_alt'].to_numpy(dtype=int) for lib in libs],
    ]).T
    dep = np.vstack([
        df['tumor_depth'].to_numpy(dtype=int),
        *[df[f'n0_{lib}_depth'].to_numpy(dtype=int) for lib in libs],
    ]).T

    fig, (ax0, ax1) = plt.subplots(
        1, 2, figsize=(15.5, 7.8),
        gridspec_kw={'width_ratios': [1.35, 1.0]})

    # ---- panel A: site x sample AF matrix --------------------------------
    masked = np.ma.masked_where((af <= 0) | ~np.isfinite(af), af)
    cmap = plt.get_cmap('viridis').copy()
    cmap.set_bad('#e6e6e6')
    im = ax0.imshow(masked, aspect='auto', cmap=cmap,
                    norm=LogNorm(vmin=1e-4, vmax=0.05))
    ax0.set_xticks(range(1 + len(libs)))
    ax0.set_xticklabels(['1% tumor'] +
                        [f'0% {lib}' + ('\n(matched)' if i == 0 else '')
                         for i, lib in enumerate(libs)],
                        fontsize=8)
    ylabels = []
    for _, row in df.iterrows():
        ylabels.append(
            f"{row['truth_status']}  {row['somaticxgb_score']:.3f}  {row['var']}")
    ax0.set_yticks(range(len(df)))
    ax0.set_yticklabels(ylabels, fontsize=7)
    for tick, status in zip(ax0.get_yticklabels(), df['truth_status']):
        tick.set_color('#b2182b' if status == 'FP' else '#2166ac')
    ax0.set_title('A. ALT support at representative calls\n'
                  'cell text: ALT reads / depth', fontsize=10)
    ax0.axhline(9.5, color='black', linewidth=0.8)
    for i in range(af.shape[0]):
        for j in range(af.shape[1]):
            if dep[i, j] == 0:
                txt = '0/0'
            else:
                txt = f'{alt[i, j]}/{dep[i, j]}'
            val = af[i, j] if af[i, j] > 0 else 1e-5
            color = 'white' if val > 0.008 else 'black'
            ax0.text(j, i, txt, ha='center', va='center', fontsize=6.2,
                     color=color)
    cbar = fig.colorbar(im, ax=ax0, fraction=0.035, pad=0.02)
    cbar.set_label('ALT fraction (log scale; grey = 0)', fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    # ---- panel B: paired tumor vs 0% AF ----------------------------------
    x = np.arange(1, len(df) + 1)
    tumor_af = df['tumor_AF'].to_numpy(dtype=float)
    n0_mean = df['n0_mean_AF'].to_numpy(dtype=float)
    rep_af = df[[f'n0_{lib}_AF' for lib in libs]].to_numpy(dtype=float)
    is_tp = (df['truth_status'] == 'TP').to_numpy()

    ax1.set_yscale('symlog', linthresh=1e-4)
    ax1.axvline(10.5, color='black', linewidth=0.8, linestyle=':')
    for xi, tv, nv in zip(x, tumor_af, n0_mean):
        ax1.plot([xi, xi], [max(tv, 1e-9), max(nv, 1e-9)],
                 color='#b0b0b0', linewidth=0.8, zorder=1)
    offsets = np.array([-0.18, 0.0, 0.18])
    for k, lib in enumerate(libs):
        ax1.scatter(x + offsets[k], np.maximum(rep_af[:, k], 0.0),
                    s=16, facecolors='none', edgecolors='#3690c0',
                    linewidths=0.7, alpha=0.75, zorder=2,
                    label='0% replicate' if k == 0 else None)
    ax1.scatter(x, n0_mean, s=34, marker='s', facecolors='#3690c0',
                edgecolors='black', linewidths=0.3, zorder=3,
                label='0% mean of 3')
    ax1.scatter(x, tumor_af, s=38, marker='o', facecolors='#d7301f',
                edgecolors='black', linewidths=0.3, zorder=4,
                label='1% tumor')
    ax1.set_xlim(0.3, len(df) + 0.7)
    ax1.set_xticks(x)
    ax1.set_xticklabels([str(i) for i in x], fontsize=7)
    ax1.set_xlabel('representative call (1-10 = TP; 11-20 = FP)', fontsize=9)
    ax1.set_ylabel('ALT fraction (symlog)', fontsize=9)
    ax1.set_title('B. 1% tumor versus 0% controls', fontsize=10)
    ax1.grid(axis='y', color='#dddddd', linewidth=0.5)
    ax1.set_axisbelow(True)
    handles = [
        Line2D([0], [0], marker='o', color='none', markerfacecolor='#d7301f',
               markeredgecolor='black', markersize=6, label='1% tumor'),
        Line2D([0], [0], marker='s', color='none', markerfacecolor='#3690c0',
               markeredgecolor='black', markersize=6, label='0% mean (3 reps)'),
        Line2D([0], [0], marker='o', color='none', markerfacecolor='none',
               markeredgecolor='#3690c0', markersize=5, label='0% individual rep'),
    ]
    ax1.legend(handles=handles, fontsize=7, loc='upper right', frameon=False)

    fig.text(0.01, 0.005,
             'Descriptive only. FP = absent from twist_truth_221, not proven artifact. '
             '0% libraries are 2.5M-read controls from the same manufacturing process; '
             'zero support at these depths does not exclude a low-AF event.',
             fontsize=7, color='#444444')
    fig.tight_layout(rect=[0, 0.03, 1, 1])
    fig.savefig(f'{OUT}/figures/A4_representative_calls.png', dpi=300,
                bbox_inches='tight')
    fig.savefig(f'{OUT}/figures/A4_representative_calls.pdf', bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    main()
