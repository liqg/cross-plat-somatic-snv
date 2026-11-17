#!/usr/bin/env python3
"""Redraw Supplementary Figures S1-S12 to BMC Bioinformatics artwork specs.

  * page width 170 mm (full) or 85 mm (half); height <= 225 mm incl. legend
  * sans-serif Arial-equivalent (Liberation Sans), 7 pt body, 6 pt ticks
  * panel letters a, b, c in 8 pt bold lowercase
  * lines >= 0.5 pt; vector PDF with embedded fonts; 300 dpi PNG preview
  * Okabe-Ito colour-blind-safe palette

Output: response/bmc-figures/FigS1.pdf ... FigS12.pdf (+ .png)
"""

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
import common  # noqa: E402
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams
import numpy as np
import pandas as pd

# Derived source tables for Supplementary Figures S1-S12 are vendored in
# analysis/supp_figure_data/ so a clean checkout does not need the original
# analysis tree.  CPSNV_SUPP_DATA can override the directory.
T = os.environ.get("CPSNV_SUPP_DATA", str(common.REPO_ROOT / "analysis" / "supp_figure_data"))
OUT = str(common.FIGURES / 'supplementary')
os.makedirs(OUT, exist_ok=True)

MM = 1 / 25.4

rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Liberation Sans', 'Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 7,
    'axes.labelsize': 7,
    'axes.titlesize': 7,
    'xtick.labelsize': 6,
    'ytick.labelsize': 6,
    'legend.fontsize': 6,
    'axes.linewidth': 0.5,
    'xtick.major.width': 0.5,
    'ytick.major.width': 0.5,
    'xtick.major.size': 2.5,
    'ytick.major.size': 2.5,
    'xtick.minor.width': 0.4,
    'ytick.minor.width': 0.4,
    'lines.linewidth': 1.0,
    'lines.markersize': 3,
    'errorbar.capsize': 1.5,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'savefig.pad_inches': 0.02,
})

# Okabe-Ito
OI = ['#0072B2', '#D55E00', '#009E73', '#CC79A7', '#E69F00',
      '#56B4E9', '#F0E442', '#000000', '#999999']

PLAT_COLORS = {
    'FASTASeq 300': '#0072B2',
    'GenoLab M': '#D55E00',
    'SURFSeq 5000': '#009E73',
    'NovaSeq 6000': '#CC79A7',
    'NovaSeq X': '#E69F00',
    'DNBSEQ-T7': '#56B4E9',
}
PLAT_ORDER = list(PLAT_COLORS)


def clean(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(direction='out', pad=1.5)


def panel(ax, letter, dx=-0.15, dy=1.03):
    ax.text(dx, dy, letter, transform=ax.transAxes, fontsize=8,
            fontweight='bold', va='bottom', ha='left')


CAPTIONS = {
    'FigS1': 'Pearson correlation heatmap of the 17 features in the training matrix.',
    'FigS2': 'Precision-recall curves (a) and F1 by sequencing depth (b) for the full 17-feature and TLOD + DP models.',
    'FigS3': 'Mismatch rate by quality threshold (a) and by reported base quality (b).',
    'FigS4': 'Caller score sweeps (a) and default versus fair-tuned F1 (b).',
    'FigS5': 'Fold enrichment of false-positive calls within 150 bp and 300 bp of a simulated target.',
    'FigS6': 'Unlisted donor SNVs on donor fragments, for fragment-target pairs (a) and unique fragments (b).',
    'FigS7': 'F1 at matched duplicate-removed depth.',
    'FigS8': 'Shared versus novel target recall.',
    'FigS9': 'Non-overlapping-fragment sensitivity analysis.',
    'FigS10': 'Empty-target negative control.',
    'FigS11': 'Forest plots of platform and method contrasts with 95% confidence intervals.',
    'FigS12': 'Representative calls in the 0% controls.',
}


def save(fig, name):
    n = int(name.replace('FigS', ''))
    fig.text(0.005, 0.995, f'Fig. S{n}', fontsize=8, fontweight='bold',
             va='top', ha='left', bbox=dict(facecolor='white', edgecolor='none', pad=1.0))
    fig.savefig(os.path.join(OUT, name + '.pdf'), format='pdf')
    fig.savefig(os.path.join(OUT, name + '.png'), dpi=300)
    plt.close(fig)
    print('wrote', name)


def read(name):
    return pd.read_csv(os.path.join(T, name))


# ------------------------------------------------------------------ S1
def s1():
    d = pd.read_csv(os.path.join(T, 'C3_correlation_pearson.csv'), index_col=0)
    fig, ax = plt.subplots(figsize=(170 * MM, 120 * MM))
    im = ax.imshow(d.values, cmap='RdBu_r', vmin=-1, vmax=1)
    ax.set_xticks(range(len(d)))
    ax.set_xticklabels(d.columns, rotation=45, ha='right', fontsize=5.5)
    ax.set_yticks(range(len(d)))
    ax.set_yticklabels(d.index, fontsize=5.5)
    for i in range(len(d)):
        for j in range(len(d)):
            v = d.values[i, j]
            if abs(v) >= 0.5:
                ax.text(j, i, f'{v:.2f}', ha='center', va='center', fontsize=5,
                        color='white' if abs(v) > 0.8 else 'black')
    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cb.set_label('Pearson r', fontsize=6)
    cb.ax.tick_params(labelsize=5.5, width=0.5, length=2)
    ax.tick_params(length=0)
    fig.subplots_adjust(left=0.30, right=0.80, top=0.97, bottom=0.27)
    save(fig, 'FigS1')


# ------------------------------------------------------------------ S2
def s2():
    curves = read('A2_threshold_curves.csv')
    depth = read('A2_model_metrics_by_depth.csv')
    fig, axs = plt.subplots(1, 2, figsize=(170 * MM, 70 * MM))
    styles = [('full17', 'Full 17-feature model', '#0072B2', 'o'),
              ('tlod_dp', 'TLOD + DP', '#D55E00', 's')]
    for m, lab, c, mk in styles:
        sub = curves[curves.Model == m].sort_values('Recall')
        axs[0].plot(sub.Recall, sub.Precision, color=c, label=lab)
    axs[0].set(xlabel='Recall', ylabel='Precision', xlim=(0, 1), ylim=(0, 1))
    axs[0].legend(frameon=False, loc='lower left', handlelength=1.2)
    clean(axs[0])
    panel(axs[0], 'a')

    for m, lab, c, mk in styles:
        sub = depth[depth.Model == m].sort_values('reads')
        axs[1].plot(sub.reads, sub['F1@0.1'], color=c, marker=mk, label=lab)
    axs[1].set(xlabel='Sequencing depth (million reads)',
               ylabel='F1 (threshold 0.1)')
    axs[1].legend(frameon=False, loc='lower right', handlelength=1.2)
    clean(axs[1])
    panel(axs[1], 'b')
    fig.tight_layout(w_pad=1.5)
    save(fig, 'FigS2')


# ------------------------------------------------------------------ S3
def s3():
    mt = read('A3_multithreshold_by_platform.csv')
    pq = read('A3_per_quality_by_platform.csv')
    fig, axs = plt.subplots(1, 2, figsize=(170 * MM, 75 * MM))
    for p in PLAT_ORDER:
        sub = mt[(mt.dataset == 'GIAB') & (mt.platform == p) & (mt.support_flag == 'ok')].sort_values('quality_threshold')
        if len(sub):
            axs[0].plot(sub.quality_threshold, sub.mean_error_rate, marker='o',
                        color=PLAT_COLORS[p], label=p)
    sub = mt[(mt.dataset == 'T7') & (mt.support_flag == 'ok')].sort_values('quality_threshold')
    if len(sub):
        axs[0].plot(sub.quality_threshold, sub.mean_error_rate, marker='o',
                    color=PLAT_COLORS['DNBSEQ-T7'], linestyle='--', label='DNBSEQ-T7')
    axs[0].set_yscale('log')
    axs[0].set(xlabel='Quality threshold', ylabel='Mismatch rate')
    axs[0].legend(frameon=False, ncol=2, handlelength=1.2, columnspacing=1)
    clean(axs[0])
    panel(axs[0], 'a', dx=-0.17)

    for p in PLAT_ORDER:
        sub = pq[(pq.dataset == 'GIAB') & (pq.platform == p) &
                 (pq.n_samples_with_bases > 0) & (pq.qual <= 40) & (pq.support_flag == 'ok')].sort_values('qual')
        if len(sub):
            axs[1].plot(sub.qual, sub.mean_error_rate, marker='o',
                        color=PLAT_COLORS[p], label=p)
    sub = pq[(pq.dataset == 'T7') & (pq.n_samples_with_bases > 0) & (pq.qual <= 40) & (pq.support_flag == 'ok')].sort_values('qual')
    if len(sub):
        axs[1].plot(sub.qual, sub.mean_error_rate, marker='o', linestyle='--',
                    color=PLAT_COLORS['DNBSEQ-T7'], label='DNBSEQ-T7')
    axs[1].set_yscale('log')
    axs[1].set(xlabel='Reported base quality', ylabel='Mismatch rate')
    clean(axs[1])
    panel(axs[1], 'b', dx=-0.17)
    fig.tight_layout(w_pad=2)
    save(fig, 'FigS3')


# ------------------------------------------------------------------ S4
def s4():
    sw = read('B1_threshold_sweep.csv')
    sw = sw[sw.ScanType == 'absolute']
    fig, axs = plt.subplots(1, 2, figsize=(170 * MM, 72 * MM))
    specs = [('Strelka2', 'PASS', 'Strelka2', '#0072B2'),
             ('Mutect2', 'PASS', 'Mutect2', '#D55E00'),
             ('VarScan2', 'snp.pass', 'VarScan2', '#009E73'),
             ('SomaticXGB', 'pred_all', 'SomaticXGB', '#CC79A7')]
    for method, callset, lab, c in specs:
        sub = sw[(sw.Method == method) & (sw.CallSet == callset)]
        g = sub.groupby('Cut').agg(f1=('F1', 'mean'), kept=('KeptFraction', 'mean')).reset_index()
        g['rel'] = g.Cut / g.Cut.max()
        axs[0].plot(g.rel, g.f1, marker='o', color=c, label=lab)
    axs[0].set(xlabel='Relative score threshold', ylabel='F1')
    axs[0].legend(frameon=False, loc='lower left', handlelength=1.2)
    clean(axs[0])
    panel(axs[0], 'a')

    ft = read('B1b_fair_tuning_comparison.csv')
    order = ['Strelka2', 'Mutect2', 'Strelka2_Mutect2 union', 'VarScan2', 'SomaticXGB']
    labels = ['Strelka2', 'Mutect2', 'Union', 'VarScan2', 'SomaticXGB']
    x = np.arange(len(order))
    default, fair = [], []
    for m in order:
        row = ft[(ft.Method_Label == m) & (ft.Tuning == 'default')]
        default.append(float(row.Own_Default_Mean_F1.iloc[0]) if len(row) else np.nan)
        row = ft[(ft.Method_Label == m) & (ft.Tuning == 'fair-tuned') &
                 (ft.Protocol.astype(str).str.startswith('cross-group'))]
        fair.append(float(row.Mean_F1.iloc[0]) if len(row) else np.nan)
    w = 0.38
    axs[1].bar(x - w / 2, default, width=w, color='#BDBDBD', label='Default')
    axs[1].bar(x + w / 2, fair, width=w, color='#0072B2', label='Fair-tuned')
    axs[1].axhline(float(ft[ft.Method_Label == 'SomaticXGB'].Mean_F1.iloc[0]),
                   color='#D55E00', linestyle='--', linewidth=0.8)
    axs[1].set(xticks=x, xticklabels=labels, ylabel='F1', ylim=(0, 1.05))
    axs[1].tick_params(axis='x', length=0)
    axs[1].legend(frameon=False, loc='upper left', handlelength=1.2)
    clean(axs[1])
    panel(axs[1], 'b')
    fig.tight_layout(w_pad=1.8)
    save(fig, 'FigS4')


# ------------------------------------------------------------------ S5
def s5():
    d = read('C2_fp_distance_summary.csv')
    agg = d.groupby('Method').agg(n150=('n_le150', 'sum'),
                                  n300=('n_le300', 'sum'),
                                  n=('n_fp_dist_valid', 'sum'),
                                  null150=('null_prop_le150', 'mean'),
                                  null300=('null_prop_le300', 'mean')).reset_index()
    name = {'SomaticXGB 0.1': 'SomaticXGB', 'Strelka2': 'Strelka2', 'Mutect2': 'Mutect2',
            'Strelka2_Mutect2': 'Strelka2+Mutect2', 'VarScan2_fpfilter': 'VarScan2'}
    order = ['SomaticXGB 0.1', 'Strelka2', 'Mutect2', 'Strelka2_Mutect2', 'VarScan2_fpfilter']
    agg = agg.set_index('Method').loc[order].reset_index()
    obs150 = agg.n150 / agg.n
    obs300 = agg.n300 / agg.n
    enr150 = obs150 / agg.null150
    enr300 = obs300 / agg.null300
    fig, ax = plt.subplots(figsize=(170 * MM, 65 * MM))
    x = np.arange(len(agg))
    w = 0.38
    ax.bar(x - w / 2, enr150, width=w, color='#0072B2', label='Within 150 bp')
    ax.bar(x + w / 2, enr300, width=w, color='#56B4E9', label='Within 300 bp')
    ax.axhline(1, color='#444444', linewidth=0.6, linestyle='--')
    ax.set(xticks=x, xticklabels=[name[m] for m in agg.Method],
           ylabel='Fold enrichment over panel null')
    ax.tick_params(axis='x', length=0)
    ax.legend(frameon=False, loc='upper right', handlelength=1.2)
    clean(ax)
    fig.tight_layout()
    save(fig, 'FigS5')


# ------------------------------------------------------------------ S6
def s6():
    d = read('G1_fragment_summary.csv')
    regions = ['ALL', 'chr6_31.32Mb_cluster', 'chr10_123.36Mb_cluster', 'other']
    rlab = ['All targets', 'chr6 cluster', 'chr10 cluster', 'Other']
    fig, axs = plt.subplots(1, 2, figsize=(170 * MM, 65 * MM))
    x = np.arange(len(regions))
    w = 0.38
    for k, (g, c) in enumerate([(45, '#0072B2'), (54, '#D55E00')]):
        sub = d[d.group == g].set_index('region').loc[regions]
        axs[0].bar(x + (k - 0.5) * w, sub.frac_pairs_with_unlisted_snv * 100,
                   width=w, color=c, label=f'HG004→HG005' if g == 45 else 'HG005→HG004')
        axs[1].bar(x + (k - 0.5) * w, sub.frac_unique_frag_with_unlisted_snv * 100,
                   width=w, color=c)
    for ax, lab, letter in [(axs[0], 'Fragment–target pairs (%)', 'a'),
                            (axs[1], 'Unique donor fragments (%)', 'b')]:
        ax.set(xticks=x, xticklabels=rlab, ylabel=lab)
        ax.tick_params(axis='x', length=0)
        clean(ax)
        panel(ax, letter)
    axs[0].legend(frameon=False, handlelength=1.2)
    fig.tight_layout(w_pad=2)
    save(fig, 'FigS6')


# ------------------------------------------------------------------ S7
def s7():
    d = read('D1_f1_vs_dedup_depth.csv')
    b = read('D1b_f1_vs_BAMdepth.csv')
    fig, axs = plt.subplots(2, 2, figsize=(170 * MM, 140 * MM))
    rows = [(45, 'HG004→HG005'), (54, 'HG005→HG004')]
    cols = [('MedianDP', 'linear_log10_DP', 'median candidate DP', d),
            ('BAM_est_depth', 'linear_log10_BAMdepth', 'BAM estimate', b)]
    for i, (g, glab) in enumerate(rows):
        for j, (xcol, src, xlab, df) in enumerate(cols):
            ax = axs[i, j]
            for p in PLAT_ORDER:
                sub = df[(df.Group == g) & (df.Source == src) &
                         (df.Platform == p) & (df.F1_metric == 'F1@0.1')]
                if 'Scope' in sub.columns:
                    sub = sub[sub.Scope == 'all6']
                sub = sub.sort_values(xcol)
                if len(sub):
                    ax.plot(sub[xcol], sub.F1, color=PLAT_COLORS[p], label=p, linewidth=0.9)
            ax.set(xlabel=f'Matched duplicate-removed depth\n({xlab})', ylabel='F1 (threshold 0.1)')
            clean(ax)
            panel(ax, 'abcd'[i * 2 + j], dx=-0.16)
            ax.set_title(glab, fontsize=7, pad=3, loc='left')
    handles, labels = axs[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=6, frameon=False,
               handlelength=1.2, columnspacing=1.2, bbox_to_anchor=(0.5, 0.005))
    fig.tight_layout(rect=(0, 0.05, 1, 1), h_pad=1.2, w_pad=1.5)
    save(fig, 'FigS7')


# ------------------------------------------------------------------ S8
def s8():
    sw = read('R24_shared_vs_novel_summary.csv')
    no = read('R24_novel_only_summary.csv')
    fig, axs = plt.subplots(1, 2, figsize=(170 * MM, 70 * MM))
    s = sw[(sw.shared_definition == 'exact') & (sw.setting == 'site_all') &
           (sw.model == 'full17') & (sw.threshold == 0.1)].set_index('group')
    x = np.arange(2)
    dx = 0.12
    for i, g in enumerate([45, 54]):
        a, b_ = s.loc[g, 'mean_recall_shared'], s.loc[g, 'mean_recall_novel']
        axs[0].plot([i - dx, i + dx], [a, b_], color='#777777', linewidth=0.6, zorder=1)
        axs[0].plot(i - dx, a, 'o', color='#0072B2', markersize=4, zorder=2, label='Shared' if i == 0 else None)
        axs[0].plot(i + dx, b_, 's', color='#D55E00', markersize=4, zorder=2, label='Novel' if i == 0 else None)
    axs[0].set(xticks=x, xticklabels=['HG004→HG005', 'HG005→HG004'],
               ylabel='Site-level recall', ylim=(0.8, 1.0))
    axs[0].tick_params(axis='x', length=0)
    axs[0].legend(frameon=False, loc='lower left', handlelength=1.0)
    clean(axs[0]); panel(axs[0], 'a')

    settings = ['site_all', 'candidate_present']
    slab = ['All targets', 'Candidate-present']
    n = no[(no.model == 'full17') & (no.threshold == 0.1)].set_index('setting')
    x = np.arange(2)
    for i, k in enumerate(settings):
        a, b_ = n.loc[k, 'mean_recall_45'], n.loc[k, 'mean_recall_54']
        axs[1].plot([i - dx, i + dx], [a, b_], color='#777777', linewidth=0.6, zorder=1)
        axs[1].plot(i - dx, a, 'o', color='#0072B2', markersize=4, zorder=2, label='HG004→HG005' if i == 0 else None)
        axs[1].plot(i + dx, b_, 's', color='#D55E00', markersize=4, zorder=2, label='HG005→HG004' if i == 0 else None)
    axs[1].set(xticks=x, xticklabels=slab, ylabel='Site-level recall', ylim=(0.8, 1.0))
    axs[1].tick_params(axis='x', length=0)
    axs[1].legend(frameon=False, loc='lower right', handlelength=1.0)
    clean(axs[1]); panel(axs[1], 'b')
    fig.tight_layout(w_pad=2)
    save(fig, 'FigS8')


# ------------------------------------------------------------------ S9
def s9():
    d = read('A1_nonoverlap_comparison.csv')
    piv = d.pivot(index='method', columns='condition', values='F1')
    order = ['SomaticXGB 0.1', 'Strelka2', 'Mutect2', 'Strelka2_Mutect2', 'VarScan2_fpfilter']
    piv = piv.loc[order]
    fig, ax = plt.subplots(figsize=(85 * MM, 62 * MM))
    y = np.arange(len(piv))
    h = 0.36
    ax.barh(y - h / 2, piv['original'], height=h, color='#BDBDBD', label='Original')
    ax.barh(y + h / 2, piv['nonoverlap'], height=h, color='#0072B2', label='Non-overlapping')
    ax.set(yticks=y, yticklabels=['SomaticXGB', 'Strelka2', 'Mutect2', 'Strelka2+Mutect2', 'VarScan2'],
           xlabel='F1', xlim=(0, 1))
    ax.tick_params(axis='y', length=0)
    ax.legend(frameon=False, loc='lower center', bbox_to_anchor=(0.5, -0.33), ncol=2, handlelength=1.2)
    clean(ax)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    save(fig, 'FigS9')


# ------------------------------------------------------------------ S10
def s10():
    d = read('C1_empty_target_vs_real.csv')
    fig, ax = plt.subplots(figsize=(85 * MM, 55 * MM))
    x = np.arange(len(d))
    w = 0.38
    ax.bar(x - w / 2, d.empty_pass_snv_calls, width=w, color='#009E73', label='Empty target list')
    ax.bar(x + w / 2, d.real_FP_all, width=w, color='#CC79A7', label='Real simulation')
    ax.set_yscale('log')
    ax.set(xticks=x, xticklabels=['VarScan2', 'VarScan2\nfpfilter', 'Mutect2', 'Strelka2', 'SomaticXGB'],
           ylabel='PASS SNV calls')
    ax.tick_params(axis='x', length=0)
    ax.legend(frameon=False, loc='upper right', handlelength=1.2)
    clean(ax)
    fig.tight_layout()
    save(fig, 'FigS10')


# ------------------------------------------------------------------ S11
def s11():
    p = read('R27_platform_mixed_models.csv')
    m = read('R27_method_mixed_models.csv')
    fig, axs = plt.subplots(1, 2, figsize=(170 * MM, 65 * MM))
    pd_ = p[(p.estimate_type == 'difference') & (p.metric == 'F1') &
            (p.threshold == 'SomaticXGB 0.1')]
    pd_ = pd_.iloc[::-1]
    y = np.arange(len(pd_))
    axs[0].errorbar(pd_.estimate, y,
                    xerr=[pd_.estimate - pd_.CI95_low, pd_.CI95_high - pd_.estimate],
                    fmt='o', color='#0072B2', markeredgewidth=0, markersize=3)
    axs[0].axvline(0, color='#444444', linewidth=0.6, linestyle='--')
    axs[0].set(yticks=y, yticklabels=pd_.term, xlabel='F1 difference vs SURFSeq 5000')
    axs[0].tick_params(axis='y', length=0)
    clean(axs[0]); panel(axs[0], 'a', dx=-0.42)

    md = m[(m.estimate_type == 'difference') & (m.metric == 'F1') &
           (m.comparison_set == 'primary (6 methods)')].iloc[::-1]
    y = np.arange(len(md))
    axs[1].errorbar(md.estimate, y,
                    xerr=[md.estimate - md.CI95_low, md.CI95_high - md.estimate],
                    fmt='o', color='#D55E00', markeredgewidth=0, markersize=3)
    axs[1].axvline(0, color='#444444', linewidth=0.6, linestyle='--')
    axs[1].set(yticks=y, yticklabels=md.method, xlabel='F1 difference vs SomaticXGB (0.1)')
    axs[1].tick_params(axis='y', length=0)
    clean(axs[1]); panel(axs[1], 'b', dx=-0.52)
    fig.tight_layout(w_pad=3)
    save(fig, 'FigS11')


# ------------------------------------------------------------------ S12
def s12():
    d = read('A4_representative_calls_per_call.csv')
    fig, ax = plt.subplots(figsize=(85 * MM, 65 * MM))
    colors = {'TP': '#0072B2', 'FP': '#D55E00'}
    markers = {'TP': 'o', 'FP': 's'}
    for status in ['TP', 'FP']:
        sub = d[d.truth_status == status]
        ax.scatter(sub.tumor_AF, sub.n0_pooled_AF, s=8, facecolors='none',
                   edgecolors=colors[status], marker=markers[status], linewidths=0.6,
                   label=status)
    lim = 0.095
    ax.plot([0, lim], [0, lim], color='#777777', linewidth=0.5, linestyle='--')
    ax.set(xlabel='Tumour ALT fraction (1%)', ylabel='Pooled ALT fraction (0% controls)',
           xlim=(0, 0.06), ylim=(0, 0.09))
    ax.legend(frameon=False, loc='upper left', handletextpad=0.2)
    clean(ax)
    fig.tight_layout()
    save(fig, 'FigS12')


def legends():
    fig = plt.figure(figsize=(170 * MM, 225 * MM))
    fig.text(0.02, 0.98, 'Supplementary Figures S1-S12', fontsize=10, fontweight='bold', va='top')
    y = 0.93
    for i in range(1, 13):
        key = f'FigS{i}'
        fig.text(0.02, y, f'Fig. S{i}', fontsize=8, fontweight='bold', va='top')
        fig.text(0.12, y, CAPTIONS[key], fontsize=8, va='top')
        y -= 0.065
    fig.savefig(os.path.join(OUT, 'FigS0.pdf'), format='pdf')
    fig.savefig(os.path.join(OUT, 'FigS0.png'), dpi=300)
    plt.close(fig)
    print('wrote FigS0')


if __name__ == '__main__':
    legends()
    for f in (s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12):
        f()
