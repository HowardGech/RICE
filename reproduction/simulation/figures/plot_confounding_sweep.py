"""Figure for the latent-confounding sweep: recovery metrics against the number of latent factors.
"""
import glob, os, re
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------------
# Bias-to-signal ratio shown under each x tick.
#
# One value per confounding setting, averaged over the 20 replicates. Each replicate file
# is the per-gene table written by confounder_effect_size_measured.py; the ratio is
# sd(z'u_j) / |v_j|, i.e. the latent contribution to a gene's linear predictor divided by
# the perturbation shift applied to it. Both are additive on the same scale, so the ratio
# is unit-free.
#
# TAG NAME: earlier settings are written with an `_um{d}_` tag and cova=20 with `_cova20_`.
# The glob therefore wildcards the tag and the regex accepts either spelling, so a run
# under either convention is picked up without editing this file again. Requiring digits
# after the tag is what keeps other tags (_hm_, _MOI3_) out.
# ---------------------------------------------------------------------------------
RATIO_DIR  = '/Volumes/gech/perturb_count/result/simu/simu_cova'
RATIO_GLOB = 'n10k_p100_s400_ER_*_repl*_confounder_effect_size.csv'
TAG_RE     = re.compile(r'_(?:um|cova)(\d+)_repl(\d+)_')
GENE_STAT  = 'mean'      # how per-gene ratios are reduced within a replicate
REPL_STAT  = 'mean'      # how replicate values are reduced within a setting


def load_bias_to_signal(directory=RATIO_DIR, pattern=RATIO_GLOB):
    files = sorted(glob.glob(os.path.join(directory, pattern)))
    if not files:
        raise FileNotFoundError(f'no files matching {pattern!r} in {directory}')
    rows = []
    for f in files:
        m = TAG_RE.search(os.path.basename(f))
        if m is None:
            continue
        t = pd.read_csv(f)
        if 'confounding_to_perturbation' in t.columns:
            v = t.confounding_to_perturbation
        elif {'sd_latent_eta', 'perturbation_eta'} <= set(t.columns):
            v = t.sd_latent_eta / t.perturbation_eta
        else:
            raise KeyError(
                f'{os.path.basename(f)} has no confounding_to_perturbation column and no '
                f'sd_latent_eta/perturbation_eta pair -- regenerate it with '
                f'confounder_effect_size_measured.py v3 or later. Columns: {list(t.columns)}')
        rows.append(dict(cova=int(m.group(1)), repl=int(m.group(2)),
                         ratio=float(getattr(v.dropna(), GENE_STAT)())))
    per_repl = pd.DataFrame(rows)
    if per_repl.empty:
        raise ValueError(f'{len(files)} files matched {pattern!r} but none carried an '
                         f'_um<d>_repl<r>_ or _cova<d>_repl<r>_ tag')
    out = (per_repl.groupby('cova')
           .agg(ratio=('ratio', REPL_STAT), sd=('ratio', 'std'), n_repl=('ratio', 'size'))
           .reset_index())
    print(f'bias-to-signal ratio from {len(per_repl)} files in {directory}')
    print(out.round(3).to_string(index=False))
    short = out[out.n_repl < 20]
    if len(short):
        print('WARNING: settings with fewer than 20 replicates:\n' +
              short.to_string(index=False))
    return out.set_index('cova')


BTS = load_bias_to_signal()

custom_palette = {
    "RICE": "#D55E00",
    "inspre": "#56B4E9",
    "IBCD": "#009E73",
    "dotears": "#CC79A7",
    "RICE(no RCF)": "#E69F00"
}

marker_map = {
    "RICE": "o",
    "inspre": "s",
    "IBCD": "D",
    "dotears": "^",
    "RICE(no RCF)": "P"
}

line_list = ['-', '--', ':', (0,(4,1,2,1)), (0,(3,1,1,1,1,1))]
data = data[data['measure'].isin(['precision', 'recall', 'mcc', 'shd'])]
data['method'] = pd.Categorical(data['method'], categories=['RICE', 'inspre', 'IBCD', 'dotears', 'RICE(no RCF)'], ordered=True)

# ---------------------------------------------------------------------------------
# EVEN X SPACING. The settings (2, 4, 6, 10, 20) are not evenly spaced, so plotting
# against the value itself compresses the low end and strands 20 far to the right. Each
# setting is therefore mapped to a consecutive integer position and the real value is
# restored as the tick LABEL. LEVELS is read off the data rather than hardcoded, so
# adding or dropping a setting needs no further edit here.
# ---------------------------------------------------------------------------------
LEVELS = sorted(data['cova'].unique())
POS = {c: i for i, c in enumerate(LEVELS)}
data = data.assign(cova_pos=data['cova'].map(POS))

_no_ratio = [c for c in LEVELS if c not in BTS.index]
_no_curve = [c for c in BTS.index if c not in LEVELS]
if _no_ratio:
    print(f'NOTE: no ratio file for setting(s) {_no_ratio}; their bracket reads (--)')
if _no_curve:
    print(f'NOTE: ratio files exist for setting(s) {_no_curve} but the summary table has '
          f'no rows for them, so they are not plotted -- rerun the per-method fits and '
          f'rebuild the summary if they should appear')

g = sns.FacetGrid(data, col="measure", row="intervention", palette=custom_palette,
                  aspect=1, sharey=False, height=3)

g.map_dataframe(sns.lineplot, x="cova_pos", errorbar=None, y="value",
                style='method', hue="method",
                markers=["o", "s", "D", "^", 'P'],
                palette=custom_palette, linewidth=1.3, markersize=8,
                markeredgewidth=.5, markeredgecolor="black", zorder=5)

def add_errorbars(data, **kwargs):
    ax = plt.gca()
    # observed=True: `method` is a Categorical, and pandas will switch this default in a
    # future release; being explicit also skips empty method groups within a facet
    for method, grp in data.groupby("method", observed=True):
        stats = grp.groupby("cova_pos")["value"].agg(["mean", "sem",'std']).reset_index()
        ax.errorbar(
            x=stats["cova_pos"],
            y=stats["mean"],
            yerr=stats["std"],   # 95% CI; use 1 for 1 SE
            fmt='none',                  # no line/marker, just error bars
            color=custom_palette[method],
            capsize=6,
            capthick=1.3,
            elinewidth=1.3,
            zorder=4
        )

g.map_dataframe(add_errorbars)

g.axes[0,3].set_yscale('log')
g.axes[1,3].set_yscale('log')
for ax in g.axes.flatten():
    ax.set_title('')
for ax in g.axes[-1,:]:
    ax.set_xlabel('')
fig = plt.gcf()

g.axes[0,0].set_title('PPV', fontsize=16)
g.axes[0,1].set_title('TPR', fontsize=16)
g.axes[0,2].set_title('MCC', fontsize=16)
g.axes[0,3].set_title('SHD', fontsize=16)
g.axes[0,0].set_ylabel('Soft Intervention', fontsize=16)
g.axes[1,0].set_ylabel('Hard Intervention', fontsize=16)
for ax in g.axes.flatten():
    ax.tick_params(labelsize=14)

g.add_legend(title='', ncol=5, fontsize=16, bbox_to_anchor=(.56, 1), loc='lower right')
# Centred on the AXES region rather than on figure x=0.5: add_legend's bbox_to_anchor
# widens the figure, so a figure-fraction x no longer sits under the panels. Pushed
# further down than the original y=-0.01 to clear the extra bracket row.
g.fig.canvas.draw()
_boxes = [ax.get_position() for ax in g.axes[-1, :]]
_xc = (min(b.x0 for b in _boxes) + max(b.x1 for b in _boxes)) / 2
g.fig.supxlabel("Degree of unmeasured confounding",
                fontsize=16, x=_xc, y=-0.07, ha='center')
for line in g._legend.get_lines():
    line.set_markersize(8)
g._legend.set_title('')

RATIO_FONTSIZE = 11      # smaller than the tick labels (14) so the brackets do not collide
for ax in g.axes.flatten():
    # positions, not values: one unit per setting, with the same 0.3-of-a-step padding
    # the original had at either end
    ax.set_xlim(-0.3, len(LEVELS) - 1 + 0.3)
    ax.set_xticks(range(len(LEVELS)))
    ax.set_xticklabels([str(c) for c in LEVELS])

# The bracket is drawn as an annotation rather than a second line of the tick label, so
# its font size can be set independently; at tick size the brackets run into each other.
# Only the bottom row carries x tick labels, so only that row is annotated. A setting with
# no ratio file is marked (--) rather than silently leaving a gap.
for ax in g.axes[-1, :]:
    for c in LEVELS:
        txt = f"({BTS.ratio[c]:.2f})" if c in BTS.index else "(--)"
        ax.annotate(txt, xy=(POS[c], .03), xycoords=('data', 'axes fraction'),
                    xytext=(0, -30), textcoords='offset points',
                    ha='center', va='top', fontsize=RATIO_FONTSIZE)

plt.savefig('../../plot/simu_cova_summary.pdf', bbox_inches="tight")
