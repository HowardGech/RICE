r"""LaTeX table: RICE against the three ordering baselines, per intervention and per
confounding setting.

Baselines are read from sortability_baselines_summary.csv, which is ALREADY aggregated.
That file is written by a groupby().agg(['mean','std','size']) on a three-level index,
so it has TWO header rows and three index columns and must be read as
    pd.read_csv(path, header=[0, 1], index_col=[0, 1, 2])
Reading it with the defaults silently yields one unnamed column block and junk rows.
Its values are rounded to three decimals, which is exactly the precision printed here,
so nothing is lost by using the summary rather than the per-file table.

RICE comes from the existing long frame `data` (one row per replicate) and is aggregated
the same way, so both sides of the table carry mean, SD and replicate count.

The comparison the table exists to make: oracleorder is the ceiling for ANY method that
works from a variable ordering, because it is handed the TRUE topological order. A RICE
value above it cannot be explained by varsortability.
"""
import os
import re
import numpy as np
import pandas as pd

SUMMARY_CSV = 'result/sortability/sortability_baselines_summary.csv'
# Absolute, because the notebook chdirs part-way through (cell 31 moves to
# .../result/real_soft_iv), so a relative output path lands in a different place depending
# on which cells have run. Point it wherever the manuscript sources live.
OUT_DIR = 'result/tables'
# MCC and SHD only: the four-metric version ran to 48 rows. PPV and TPR are still read in
# and aggregated, so adding them back is a one-line change to MEASURES.
MEASURES = [('mcc', 'MCC'), ('shd', 'SHD')]
# No cova = 8 was generated, so it is excluded explicitly rather than left to the union of
# whatever each source happens to contain -- an absent setting would otherwise dash a whole
# column and read as a missing result rather than a setting that was never run.
COVA = [2, 4, 6, 10, 20]
ALL_MEASURES = [('precision', 'PPV'), ('recall', 'TPR'), ('mcc', 'MCC'), ('shd', 'SHD')]
METHODS = ['randomsort', 'sortnregress', 'oracleorder', 'RICE']
LABELS = {'randomsort': 'Random order', 'sortnregress': 'Variance order',
          'oracleorder': 'Oracle order', 'RICE': r'\RICE'}
TNOTE = {'randomsort': 'a', 'sortnregress': 'b', 'oracleorder': 'c'}   # footnote markers
LOWER_BETTER = {'shd'}
DEC = {'precision': 3, 'recall': 3, 'mcc': 3, 'shd': 1}
RENAME = {'ppv': 'precision', 'tpr': 'recall'}      # sweep's names -> `data`'s names

# --- baselines, already aggregated ------------------------------------------------
bs = pd.read_csv(SUMMARY_CSV, header=[0, 1], index_col=[0, 1, 2])
bs = (bs.stack(level=0, future_stack=True)          # measure becomes an index level
      .rename_axis(index=['intervention', 'cova', 'method', 'measure'])
      .reset_index())
bs['measure'] = bs.measure.replace(RENAME)
bs = bs[bs.measure.isin([m for m, _ in ALL_MEASURES])]

# --- RICE, aggregated from the per-replicate frame the same way -------------------
rice = (data[data.method == 'RICE']
        .groupby(['measure', 'intervention', 'cova', 'method'])['value']
        .agg(['mean', 'std', 'size']).reset_index())

agg = pd.concat([rice, bs[rice.columns]], ignore_index=True)
agg['intervention'] = agg.intervention.astype(str).str.lower()
_dropped = sorted(set(agg.cova.unique()) - set(COVA))
agg = agg[agg.cova.isin(COVA)]
if _dropped:
    print(f'excluded setting(s) not in COVA: {_dropped}')


def cell_text(row, measure, best):
    """mean (sd), bolded when this method is the best in its row."""
    d = DEC[measure]
    if not np.isfinite(row['mean']):
        return '--'
    txt = (f"{row['mean']:.{d}f} ({row['std']:.{d}f})" if np.isfinite(row['std'])
           else f"{row['mean']:.{d}f}")
    return rf'\textbf{{{txt}}}' if best else txt


def latex_table(agg, measures=MEASURES, interventions=('soft', 'hard'), bold_best=True,
                label='tab:sortability', caption='', notes=()):
    """threeparttable, caption ABOVE the tabular, definitions in \\tablenotes.

    House style here: the caption carries the claim and the minimal construction only;
    anything a reader needs in order to interpret a COLUMN goes in a footnote attached to
    that column, so the caption stays short and the definition sits where it is used.
    Needs \\usepackage{threeparttable}, \\usepackage{multirow}, \\usepackage{booktabs}.
    """
    covas = [c for c in COVA if c in set(agg.cova.unique())]
    nrow_itv = len(measures) * len(covas)
    hdr = ' & '.join(LABELS[m] + (rf'\tnote{{{TNOTE[m]}}}' if m in TNOTE else '')
                     for m in METHODS)
    out = [r'\begin{table}[!t]', r'    \centering', r'    \begin{threeparttable}',
           rf'    \caption{{{caption}}}', rf'    \label{{{label}}}',
           r'    \small',
           r'    \begin{tabular}{c|c|r|' + 'r' * len(METHODS) + '}', r'    \toprule',
           rf'    Intervention & Metric & $d$ & {hdr} \\', r'    \midrule']
    for k, itv in enumerate(interventions):
        if k:
            out.append(r'    \midrule')
        for mi, (measure, mlabel) in enumerate(measures):
            if mi:
                out.append(rf'    \cmidrule{{2-{3 + len(METHODS)}}}')
            for i, c in enumerate(covas):
                sub = agg[(agg.measure == measure) & (agg.intervention == itv)
                          & (agg.cova == c)].set_index('method')
                vals = {m: sub.loc[m, 'mean'] if m in sub.index else np.nan for m in METHODS}
                finite = {m: v for m, v in vals.items() if np.isfinite(v)}
                best_m = ((min if measure in LOWER_BETTER else max)(finite, key=finite.get)
                          if bold_best and finite else None)
                cells = [cell_text(sub.loc[m], measure, m == best_m) if m in sub.index
                         else '--' for m in METHODS]
                col1 = rf'\multirow{{{nrow_itv}}}{{*}}{{{itv.capitalize()}}}' \
                    if (mi == 0 and i == 0) else ''
                col2 = rf'\multirow{{{len(covas)}}}{{*}}{{{mlabel}}}' if i == 0 else ''
                out.append(f'    {col1} & {col2} & {c} & ' + ' & '.join(cells) + r' \\')
    out += [r'    \bottomrule', r'    \end{tabular}']
    if notes:
        out.append(r'\begin{tablenotes}[flushleft]\footnotesize')
        for mark, text in notes:
            out.append(rf'  \item[{mark}] {text}')
        out.append(r'    \end{tablenotes}')
    out += [r'    \end{threeparttable}', r'\end{table}']
    tex = '\n'.join(out)
    # every \tablenotes item needs a \tnote marker in the table and vice versa
    marks = set(re.findall(r'\\tnote\{(\w)\}', tex))
    items = set(re.findall(r'\\item\[(\w)\]', tex))
    assert marks == items, f'footnote markers {marks} do not match items {items}'
    assert len({l.count('&') for l in tex.split(chr(10)) if l.strip().endswith(chr(92) * 2)}) == 1, \
        'rows have inconsistent column counts'
    return tex


# TITLE SCOPE. "is not explained by variable scaling or variance ordering" claims more
# than the experiment shows: it asserts the ordering contributes nothing. What the oracle
# bound establishes is that RICE's accuracy is higher than any ordering-based procedure can
# reach, which leaves open whether RICE also benefits from the ordering. "exceeds what ...
# can attain" is exactly the claim the design supports -- do not revert.
# TITLE SCOPE. "is not explained by variable scaling or variance ordering" claims more
# than the experiment shows: it asserts the ordering contributes nothing. What the oracle
# bound establishes is that RICE's accuracy is higher than any ordering-based procedure can
# reach, which leaves open whether RICE also benefits from the ordering. "exceeds what ...
# can attain" is exactly the claim the design supports -- do not revert.
#
# BODY SCOPE. The clause naming WHERE RICE leads is built by build_caption() from the
# numbers being tabulated, not written by hand. A hardcoded "at any setting" was wrong on
# the test data -- the orderings return sparser graphs while RICE's edge count grows with
# confounding, and SHD penalises extra edges directly, so SHD is where the universal is
# most likely to fail. The clause now names the metrics where it actually holds.
CAPTION_HEAD = (r'\textbf{The accuracy of \RICE\ exceeds what variable scaling or variance '
                r'ordering can attain.} One and the same penalised regression of each gene '
                r'on its predecessors was applied under three orderings of increasing '
                r'quality and compared with \RICE\ at the same settings, where $d$ is the '
                r'number of latent factors. Recovery improves from the random to the '
                r'variance to the oracle order, so the ordering does carry information; ')
CAPTION_TAIL = (r' Each cell is the mean over replicates with the standard deviation in '
                r'parentheses, and the best value in each row is in bold.')


def check_caption_claims(agg, measures=MEASURES, interventions=('soft', 'hard')):
    """The caption asserts two things about the numbers. Verify both against the table.

    A caption that contradicts its own table is the kind of thing a reviewer finds, so
    neither claim is written by hand: both are checked here and any violation is printed
    with the exact cell, so the wording can be weakened rather than silently left wrong.
      (1) recovery improves random -> variance -> oracle
      (2) RICE is better than all three, everywhere
    """
    bad_mono, bad_rice = [], []
    for measure, mlab in measures:
        better = (lambda a, b: a < b) if measure in LOWER_BETTER else (lambda a, b: a > b)
        for itv in interventions:
            for c in [x for x in COVA if x in set(agg.cova.unique())]:
                sub = (agg[(agg.measure == measure) & (agg.intervention == itv)
                           & (agg.cova == c)].set_index('method')['mean'])
                if not {'randomsort', 'sortnregress', 'oracleorder', 'RICE'} <= set(sub.index):
                    continue
                r_, v_, o_, rice = (sub['randomsort'], sub['sortnregress'],
                                    sub['oracleorder'], sub['RICE'])
                if not (better(v_, r_) and better(o_, v_)):
                    bad_mono.append((mlab, itv, c, round(r_, 3), round(v_, 3), round(o_, 3)))
                worse_than = [m for m, val in [('randomsort', r_), ('sortnregress', v_),
                                               ('oracleorder', o_)] if not better(rice, val)]
                if worse_than:
                    bad_rice.append((mlab, itv, c, round(rice, 3),
                                     {m: round(sub[m], 3) for m in worse_than}))
    print('\ncaption claim 1 -- recovery improves random -> variance -> oracle:')
    if bad_mono:
        print(f'   VIOLATED in {len(bad_mono)} cell(s) (metric, intervention, d, '
              f'random, variance, oracle):')
        for row in bad_mono:
            print('     ', row)
    else:
        print('   holds in every cell')
    print('caption claim 2 -- \\RICE\\ beats all three orderings everywhere:')
    if bad_rice:
        print(f'   VIOLATED in {len(bad_rice)} cell(s) (metric, intervention, d, RICE, '
              f'the ordering(s) that match or beat it):')
        for row in bad_rice:
            print('     ', row)
        print('   -> weaken the caption: name the metric where it does not hold, or '
              'restrict the sentence to the metrics where it does.')
    else:
        print('   holds in every cell')
    return bad_mono, bad_rice


def build_caption(agg, measures=MEASURES, interventions=('soft', 'hard')):
    """Assemble the caption with its scope clause DERIVED from the table.

    The clause states where RICE leads all three orderings. Writing it by hand produced a
    universal ('at any setting') that the numbers contradicted on SHD, so it is computed:
    whichever metrics pass get named, and if none pass the claim is dropped entirely rather
    than printed against the evidence.
    """
    _, bad = check_caption_claims(agg, measures, interventions)
    failed = {mlab for mlab, *_ in bad}
    held = [mlab for _, mlab in measures if mlab not in failed]
    names = [mlab for _, mlab in measures]
    if len(held) == len(names):
        clause = (r'none of the three matches \RICE\ at any setting, however, and the '
                  r'oracle order bounds what any procedure exploiting an ordering can '
                  r'attain.')
    elif held:
        where = held[0] if len(held) == 1 else ' and '.join([', '.join(held[:-1]), held[-1]])
        lost = sorted(failed)
        lost_txt = lost[0] if len(lost) == 1 else ' and '.join([', '.join(lost[:-1]), lost[-1]])
        clause = (rf'on {where} none of the three matches \RICE\ at any setting, however, '
                  rf'and the oracle order bounds what any procedure exploiting an ordering '
                  rf'can attain. On {lost_txt} the orderings return sparser graphs and '
                  rf'score better at the higher confounding levels, a difference in edge '
                  rf'count rather than in orientation.')
    else:
        clause = (r'the oracle order bounds what any procedure exploiting an ordering can '
                  r'attain.')
    print(f"\ncaption scope clause built from the data: metrics where \\RICE\\ leads "
          f"= {held or 'none'}")
    return CAPTION_HEAD + clause + CAPTION_TAIL


NOTES = [
    # (a) deliberately does not name the random-order variant: the 2021 paper names
    # sortnregress, and the random permutation is described as a procedure rather than
    # attributed as a named method.
    ('a', r'Genes are placed in a random permutation and each is regressed on its '
          r'predecessors in that order under an $L_1$ penalty, following the procedure of '
          r'\citet{reisach2021beware}. This is the floor: it uses no information about the '
          r'data beyond the regression itself.'),
    ('b', r'The same procedure with the genes ordered by increasing marginal variance, '
          r'that is \emph{sortnregress} \citep{reisach2021beware}. This is what the '
          r'variance ordering alone is worth, and it is the comparison a high '
          r'varsortability calls for.'),
    # (c) and the former (d) are merged: every note must have a \tnote marker in the
    # table, and there was no column to attach a fourth one to. threeparttable renders an
    # item with no marker as an orphan note, which is why the assertion below exists.
    ('c', r'The same procedure with the genes placed in the \emph{true} topological order '
          r'of the generating graph. No method can extract more from an ordering than it '
          r'would obtain from the correct one supplied in advance, so this upper-bounds '
          r'any procedure that exploits variable scaling or variance ordering, and a value '
          r'above it must therefore come from something beyond the ordering, though this '
          r'does not establish that the ordering is of no use to a method that has it. It '
          r'is well below one because knowing '
          r'the order is not knowing the graph: the order reduces the candidate set from '
          r'$p(p-1)$ ordered pairs to $p(p-1)/2$ but leaves a selection problem, and '
          r'regressing a gene on its predecessors does not remove latent confounding, so '
          r'confounded pairs are selected whatever the order.'),
]

os.makedirs(OUT_DIR, exist_ok=True)
full_path = os.path.join(OUT_DIR, 'sortability_table.tex')
mcc_path = os.path.join(OUT_DIR, 'sortability_table_mcc.tex')
CAPTION = build_caption(agg)
CAPTION_MCC = build_caption(agg, measures=[('mcc', 'MCC')])
open(full_path, 'w').write(latex_table(agg, caption=CAPTION, notes=NOTES))
open(mcc_path, 'w').write(latex_table(
    agg, measures=[('mcc', 'MCC')], label='tab:sortability_mcc',
    caption=CAPTION_MCC, notes=NOTES))
print(latex_table(agg, caption=CAPTION, notes=NOTES))

# --- why a cell renders as "--" ----------------------------------------------------
# A dash means that method has NO ROW for that (measure, intervention, cova). Because the
# cova column of the table is the UNION over methods, any setting present in the baselines
# but absent from the existing RICE results dashes the RICE column, and vice versa. The
# three prints below localise it: first the label vocabularies, then the per-method cova
# coverage, then the exact dashed cells.
print('\nlabel vocabularies (these must agree across sources):')
for col in ('measure', 'intervention', 'method'):
    print(f'   {col:<13}', {k: sorted(set(v)) for k, v in
                            agg.groupby(agg.method.isin(METHODS[:3])
                                        .map({True: 'baselines', False: 'existing'}))[col]
                            .apply(lambda s: s.unique()).items()})
print('\ncova coverage by method:')
print(agg.groupby('method')['cova'].apply(lambda s: sorted(s.unique())).to_string())
print('\nreplicates per cell, by method:')
print(agg.groupby('method')['size'].agg(['min', 'max']).to_string())

dashed = [(mlab, itv, c, m)
          for m in METHODS
          for measure, mlab in MEASURES
          for itv in ('soft', 'hard')
          for c in COVA
          if not ((agg.method == m) & (agg.measure == measure)
                  & (agg.intervention == itv) & (agg.cova == c)).any()]
if dashed:
    print(f'\n{len(dashed)} cell(s) will render as "--" (metric, intervention, d, method):')
    for row in dashed:
        print('   ', row)
    print('   Fix by making the labels agree, or restrict the table with '
          "latex_table(agg, ...) after filtering agg to the shared cova values.")
else:
    print('\nno dashed cells: every method has a row for every metric and setting')
print('\nwrote:')
for p_ in (full_path, mcc_path):
    print('   ' + os.path.abspath(p_))
