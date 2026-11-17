#!/usr/bin/env python3
"""Build sup_tables-submit-revised-v4.xlsx.

  * fixes the column swap and title typo in Table S11;
  * appends Tables S21-S35 from addendum-analyses/tables;
  * adds a sheet listing Supplementary Figures S1-S12.

Input : sup_tables-submit-revised-261112.xlsx
Output: sup_tables-submit-revised-v4.xlsx
"""

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
import common  # noqa: E402
import csv
import os

import openpyxl
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

T = str(common.TABLES)
SRC = str(common.REPO_ROOT / 'reference' / 'supplementary' / 'sup_tables-submit-revised-261112.xlsx')
OUT = str(common.RESULTS_ROOT / 'Additional_file_1_Supplementary_tables.xlsx')

HEAD_FILL = PatternFill('solid', fgColor='DCE6F1')
BOLD = Font(bold=True)
TITLE = Font(bold=True, size=12)
CAPTION = Font(italic=True, size=10)


def read_csv(path):
    with open(path, newline='', encoding='utf-8-sig') as f:
        return [row for row in csv.reader(f)]


def num(v):
    if v is None or v == '':
        return None
    try:
        if any(c in v for c in 'eE') and not any(c in v for c in '.') and v.replace('-', '').isdigit() is False:
            return float(v)
        return int(v) if v.lstrip('-').isdigit() else float(v)
    except (TypeError, ValueError):
        return v


def style_header(ws, row, ncol):
    for c in range(1, ncol + 1):
        cell = ws.cell(row=row, column=c)
        cell.font = BOLD
        cell.fill = HEAD_FILL
        cell.alignment = Alignment(vertical='top', wrap_text=True)
    ws.freeze_panes = ws.cell(row=row + 1, column=1)


def autosize(ws, maxw=48):
    for col in range(1, ws.max_column + 1):
        best = 8
        for row in range(1, ws.max_row + 1):
            v = ws.cell(row=row, column=col).value
            if v is not None:
                best = max(best, min(maxw, len(str(v))))
        ws.column_dimensions[get_column_letter(col)].width = best


def add_table(wb, number, title, caption, files):
    """files: list of (block_label or None, csv path)."""
    name = f'Table S{number}'
    if name in wb.sheetnames:
        del wb[name]
    ws = wb.create_sheet(name)
    ws['A1'] = title
    ws['A1'].font = TITLE
    ws['A2'] = caption
    ws['A2'].font = CAPTION
    ws['A2'].alignment = Alignment(wrap_text=True, vertical='top')
    ws.merge_cells(start_row=2, start_column=1, end_row=2, end_column=8)
    r = 4
    for label, path in files:
        rows = read_csv(path)
        if label:
            ws.cell(row=r, column=1, value=label).font = BOLD
            r += 1
        for i, row in enumerate(rows):
            for j, v in enumerate(row, start=1):
                ws.cell(row=r, column=j, value=num(v))
            if i == 0:
                style_header(ws, r, len(row))
            r += 1
        r += 2
    autosize(ws)
    return ws


S30_MAP = {
    'Hybrid method summary': 'Twist/HG001 descriptive',
    'GIAB platform MixedLM': 'GIAB platform: mixed model',
    'GIAB platform paired bootstrap': 'GIAB platform: paired bootstrap',
    'GIAB method MixedLM': 'GIAB method: mixed model',
    'GIAB method paired bootstrap': 'GIAB method: paired bootstrap',
    'Twist platform paired t': 'Twist platform: paired t',
    'Twist method paired t': 'Twist method: paired t',
    'Platform error rate vs F1': 'Error rate vs F1',
}
S30_SRC = {
    'Dataset': 'dataset', 'Analysis detail': 'analysis', 'Metric': 'metric',
    'Threshold': 'threshold', 'Term': 'term', 'Reference': 'reference',
    'n_units': 'n_units', 'n_replicates': 'n_replicates', 'n_platforms': 'n_platforms',
    'Estimate': 'estimate', 'SE': 'SE', 'CI95_low': 'CI95_low', 'CI95_high': 'CI95_high',
    'p': 'p', 'p_fdr': 'p_fdr', 'Inference': 'inference',
    'Value_rep1': 'value_rep1', 'Value_rep2': 'value_rep2', 'Value_rep3': 'value_rep3',
    'CI95_replicate_low': 'CI95_replicate_t_low', 'CI95_replicate_high': 'CI95_replicate_t_high',
    'Notes': 'notes',
}
S30_COLS = ['Analysis'] + list(S30_SRC)


def add_s30(wb):
    rows = read_csv(T + '/R27_supp_table_draft.csv')
    hdr = rows[0]
    idx = {h: i for i, h in enumerate(hdr)}
    ws = wb.create_sheet('Table S30')
    ws['A1'] = 'Table S30  Paired and hierarchical estimates with 95% confidence intervals'
    ws['A1'].font = TITLE
    ws['A2'] = ('Eight analysis blocks: Twist/HG001 descriptive means; GIAB platform and method mixed models and paired bootstraps; '
                'Twist platform and method paired t tests; and platform error-rate vs F1 correlation. Replicate values are shown where applicable. '
                'Raw model outputs are retained in the repository source data.')
    ws['A2'].font = CAPTION
    ws['A2'].alignment = Alignment(wrap_text=True, vertical='top')
    ws.merge_cells(start_row=2, start_column=1, end_row=2, end_column=len(S30_COLS))
    for j, h in enumerate(S30_COLS, 1):
        ws.cell(4, j, h)
    style_header(ws, 4, len(S30_COLS))
    for i, row in enumerate(rows[1:], 5):
        for j, h in enumerate(S30_COLS, 1):
            v = S30_MAP[row[idx['section']]] if h == 'Analysis' else row[idx[S30_SRC[h]]]
            ws.cell(i, j, num(v))
    ws.freeze_panes = 'A5'
    ws.auto_filter.ref = f'A4:{get_column_letter(len(S30_COLS))}{4 + len(rows) - 1}'
    autosize(ws, maxw=60)
    return ws


def main():
    wb = openpyxl.load_workbook(SRC)

    # Table S1: make the preparation hierarchy explicit for Reviewer 2.
    ws = wb['Table S1']
    if ws.max_column < 16:
        ws.cell(2, ws.max_column + 1, 'Preparation type')
        ws.cell(2, ws.max_column + 1, 'Independent preparation')
        platform = None
        for row in range(3, 75):
            if ws.cell(row, 1).value:
                platform = str(ws.cell(row, 1).value).strip()
            if not ws.cell(row, 3).value:
                continue
            if platform == 'DNBSEQ-T7':
                ws.cell(row, 15, 'DNBSEQ conversion')
                ws.cell(row, 16, 'No (converted from TruSeq)')
            else:
                ws.cell(row, 15, 'TruSeq aliquot')
                ws.cell(row, 16, 'Yes (shared library)')
    for row in range(1, ws.max_row + 1):
        v = ws.cell(row, 1).value
        if isinstance(v, str) and v.startswith('Note: Rows are grouped'):
            if 'Preparation type and independent preparation' not in v:
                ws.cell(row, 1).value = v + ' Preparation type and independent preparation record whether a row is a TruSeq library aliquot shared across the five TruSeq-compatible platforms or a DNBSEQ conversion of that library.'
            break

    # ---- fix Table S11 -------------------------------------------------
    ws = wb['Table S11']
    ws['A1'] = 'Supplementary Table 11  Performance of the Twist/HG001 simulation'
    for row in range(3, ws.max_row + 1):
        c, d = ws.cell(row=row, column=3).value, ws.cell(row=row, column=4).value
        ws.cell(row=row, column=3, value=d)
        ws.cell(row=row, column=4, value=c)
    # fix titles that do not match their sheet number
    wb['Table S4']['A1'] = 'Supplementary Table 4  Description and source of features used in our new method'
    wb['Table S6']['A1'] = 'Supplementary Table 6  Performance of the model trained only on NovaSeq X data'
    wb['Table S7']['A1'] = 'Supplementary Table 7  Feature importance metrics in the SomaticXGB model'
    # make the value column numeric where possible
    for row in range(3, ws.max_row + 1):
        cell = ws.cell(row=row, column=4)
        cell.value = num(cell.value)
    print('fixed S11')

    # ---- new tables ----------------------------------------------------
    T31 = [
        ('median candidate DP (primary analysis)',
         os.path.join(T, 'D1_pairwise_matched_depth.csv')),
        ('BAM-depth sensitivity',
         os.path.join(T, 'D1b_matched_BAMdepth_direction_counts.csv')),
        ('rank at matched depth',
         os.path.join(T, 'D1_rank_at_matched_depth.csv')),
    ]

    add_table(wb, 21,
              'Supplementary Table 21  Two-feature (TLOD + DP) baseline and the full 17-feature model',
              'All models were trained and evaluated on the same splits. PR-AUC, AUC and F1 are given at fixed thresholds 0.05, 0.1 and 0.2 and at the per-condition best threshold. The last two columns give the paired comparison against the full 17-feature model.',
              [(None, os.path.join(T, 'A2_supplement_table.csv'))])

    add_table(wb, 22,
              'Supplementary Table 22  Full model versus TLOD + DP model by platform and condition',
              'Block 1 gives the metrics by platform; block 2 gives the best F1 for each platform and condition. The full model is higher in all 60 test conditions.',
              [('by platform', os.path.join(T, 'A2_model_metrics_by_platform.csv')),
               ('best F1 by platform', os.path.join(T, 'A2_supplement_by_platform_F1best.csv'))])

    add_table(wb, 23,
              'Supplementary Table 23  Mismatch rate and fraction of bases retained at Q0-Q40',
              'Mean, SD and range across samples for each platform and quality threshold. These are descriptive application-level mismatch rates, not calibrated sequencing-error rates.',
              [(None, os.path.join(T, 'A3_multithreshold_by_platform.csv'))])

    add_table(wb, 24,
              'Supplementary Table 24  Mismatch rate stratified by the reported base quality',
              'The mismatch rate at each reported quality value. The curve is a description of the reported scores; it is not a calibration test.',
              [(None, os.path.join(T, 'A3_per_quality_by_platform.csv'))])

    add_table(wb, 25,
              'Supplementary Table 25  Threshold sweeps and fair-tuned comparison for the conventional callers',
              'Cut points were selected without evaluation-side labels, on the other donor group or on the other five platforms, and then applied to the held-out samples. Oracle per-sample upper bounds are given for reference. Tuning narrows but does not reverse the ranking.',
              [('method summary and oracle bounds', os.path.join(T, 'B1_method_summary.csv')),
               ('fair-tuned comparison', os.path.join(T, 'B1b_fair_tuning_comparison.csv'))])

    add_table(wb, 26,
              'Supplementary Table 26  Distance of false-positive calls to the nearest simulated target',
              'The null distribution is 200,000 length-weighted random positions within the panel BED. Proportions within 0, 50, 150, 300 and 1000 bp are compared with the null.',
              [(None, os.path.join(T, 'C2_fp_distance_summary.csv'))])

    add_table(wb, 27,
              'Supplementary Table 27  Representative calls with 0% control support',
              'Ten true-positive and ten false-positive SomaticXGB calls from the 1% Twist standard (FASTASeq 300), with REF/ALT support in the 1% tumour and three 0% control replicates. Descriptive only: the controls are shallow and are different libraries, so the table does not adjudicate individual calls.',
              [('per-call counts', os.path.join(T, 'A4_representative_calls_per_call.csv')),
               ('summary', os.path.join(T, 'A4_representative_summary.csv'))])

    add_table(wb, 28,
              'Supplementary Table 28  Donor-fragment audit of alleles absent from the truth set',
              'Static summary per target, fragment-level observations of unlisted donor SNVs, and matching of false-positive calls to unlisted donor alleles. No false-positive call matched an unlisted donor allele.',
              [('static summary', os.path.join(T, 'G1_static_summary.csv')),
               ('per target', os.path.join(T, 'G1_static_per_target.csv')),
               ('fragment level', os.path.join(T, 'G1_fragment_summary.csv')),
               ('false-positive validation', os.path.join(T, 'G1_fp_validation.csv'))])

    add_table(wb, 29,
              'Supplementary Table 29  Training-test allele sharing and comparison of donor directions',
              'Shared and novel targets are compared within conditions and after restricting to candidate-present targets, which separates candidate generation from model scoring.',
              [('shared versus novel', os.path.join(T, 'R24_shared_vs_novel_summary.csv')),
               ('novel only', os.path.join(T, 'R24_novel_only_summary.csv')),
               ('direction comparison', os.path.join(T, 'R24_group_paired_comparison.csv'))])

    add_s30(wb)

    add_table(wb, 31,
              'Supplementary Table 31  Platform comparison at matched duplicate-removed depth',
              'The primary analysis uses the median candidate DP as an on-target deduplicated-depth proxy; the BAM-depth analysis is a sensitivity check. Nearest-neighbour matching and linear/isotonic interpolation are both reported.',
              T31)

    add_table(wb, 32,
              'Supplementary Table 32  Non-overlapping-fragment sensitivity analysis',
              'Read-name overlap between the original normal and tumour, and F1 for every caller before and after removing shared fragments.',
              [(None, os.path.join(T, 'A1_nonoverlap_comparison.csv'))])

    add_table(wb, 33,
              'Supplementary Table 33  Empty-target negative control',
              'The tumour background was rebuilt with the same seed but without donor reads; with an empty target list every call is a false positive. Counts are compared with the real simulation.',
              [(None, os.path.join(T, 'C1_empty_target_vs_real.csv'))])

    add_table(wb, 34,
              'Supplementary Table 34  Collinearity of the 17 features',
              'Variance inflation factors (each standardized feature regressed on the other 16) and pairwise correlations. No VIF exceeds 5.',
              [('variance inflation factors', os.path.join(T, 'C3_vif.csv')),
               ('Pearson correlations', os.path.join(T, 'C3_correlation_pearson.csv'))])

    add_table(wb, 35,
              'Supplementary Table 35  Per-condition caller performance for Strelka2, Mutect2, their union and SomaticXGB',
              'Each row is one test condition. Default and sensitivity/fair-tuned rules are shown separately; metrics are computed against the donor-private truth set.',
              [(None, os.path.join(T, 'B1c_all_depths_by_condition.csv'))])

    # ---- figure list ---------------------------------------------------
    if 'Supplementary Figures' in wb.sheetnames:
        del wb['Supplementary Figures']
    ws = wb.create_sheet('Supplementary Figures')
    ws['A1'] = 'Supplementary figures (Additional file 2)'
    ws['A1'].font = TITLE
    ws.append(['Figure', 'Caption', 'File'])
    for c in range(1, 4):
        ws.cell(row=2, column=c).font = BOLD
        ws.cell(row=2, column=c).fill = HEAD_FILL
    figures = [
        ('Fig. S1', 'Pearson correlation heatmap of the 17 features.', 'FigS1.pdf'),
        ('Fig. S2', 'Precision-recall curves (a) and F1 by sequencing depth (b) for the full 17-feature and TLOD + DP models.', 'FigS2.pdf'),
        ('Fig. S3', 'Mismatch rate by quality threshold (a) and by reported base quality (b).', 'FigS3.pdf'),
        ('Fig. S4', 'Caller score sweeps (a) and default versus fair-tuned F1 (b).', 'FigS4.pdf'),
        ('Fig. S5', 'Fold enrichment of false-positive calls within 150 bp and 300 bp of a simulated target.', 'FigS5.pdf'),
        ('Fig. S6', 'Unlisted donor SNVs on donor fragments, for fragment-target pairs (a) and unique fragments (b).', 'FigS6.pdf'),
        ('Fig. S7', 'F1 at matched duplicate-removed depth from candidate DP (a, b) and from BAM depth (c, d), for the two donor directions.', 'FigS7.pdf'),
        ('Fig. S8', 'Site-level recall for shared versus novel targets (a) and novel-only targets before and after accounting for candidate generation (b).', 'FigS8.pdf'),
        ('Fig. S9', 'F1 before and after removing shared normal-tumour fragments.', 'FigS9.pdf'),
        ('Fig. S10', 'PASS SNV calls with an empty target list versus the real simulation.', 'FigS10.pdf'),
        ('Fig. S11', 'Forest plots of platform F1 differences (a) and method F1 differences (b) with 95% confidence intervals.', 'FigS11.pdf'),
        ('Fig. S12', 'Tumour and 0% control ALT fractions for representative true-positive and false-positive calls.', 'FigS12.pdf'),
    ]
    for row in figures:
        ws.append(row)
    autosize(ws, maxw=90)

    import re as _re
    for name in wb.sheetnames:
        m = _re.match(r'Table S(\d+)$', name)
        if m:
            ws_t = wb[name]
            a1 = ws_t['A1'].value or ''
            ws_t['A1'] = _re.sub(r'^Supplementary Table \d+', f'Table S{m.group(1)}', a1)
    wb.save(OUT)
    print('saved', OUT)
    print('sheets:', len(wb.sheetnames), wb.sheetnames[-15:])


if __name__ == '__main__':
    main()
