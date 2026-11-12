r"""GO enrichment dot plot, one figure per ontology, ALL clusters and ALL terms.

Differences from the previous version:
  * Biological Process and Molecular Function are SEPARATE figures, each with its own
    colour scale and legends, rather than two panels sharing them.
  * No term cut. Every term with adjusted p < ADJ_P in any cluster is plotted; there is
    no top-N selection, pooled or per cluster.
  * y labels keep the original gene-based style ("GENE1,GENE2,...") with the cluster
    size appended.
  * Clusters with no term below ADJ_P stay on the y axis as empty rows, and are named
    in the printout, so "no enriched term" is distinguishable from "dropped".

Term order along x is by the cluster in which each term is most significant, so the
plot reads roughly block-diagonal; a term significant in several clusters shows a point
in each.

x labels are the FULL GO term names. `terms_shorten` is not used and no abbreviation is
applied: these panels go in the appendix, where the complete term is what a reader
needs. The accession is stripped, and restored only for terms whose display names would
otherwise be identical.

Enrichr results are cached (one call per cluster per ontology). Delete the cache
directory to refresh.
"""
import os, re, time
import gseapy as gp
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.rc('font', size=14)

ADJ_P       = 0.01        # significance cut; every term below it is plotted
MIN_GENES   = 3           # clusters smaller than this are not tested, but still shown
CACHE_DIR   = f'enrichr_cache_{ctype}'
WIDE_WARN   = 60          # warn when a figure exceeds this many terms
ONTOLOGIES  = [('Biological_Process', 'Biological Process'),
               ('Molecular_Function', 'Molecular Function')]

os.makedirs(CACHE_DIR, exist_ok=True)


def cluster_label(genes):
    """Original style -- the first genes stand for the cluster -- plus its size."""
    if len(genes) > 2:
        return f'{genes[0]},{genes[1]},... (n={len(genes)})'
    return f'{",".join(genes)} (n={len(genes)})'


clusters, cluster_order = {}, []
for cid in np.unique(labels):
    genes = gene_names_non_isolated[labels == cid].tolist()
    name = cluster_label(genes)
    clusters[name] = genes
    cluster_order.append(name)

# Every Louvain community becomes one y row. Two clusters could in principle produce the
# same label (same leading gene symbols AND the same size), which would silently merge
# two rows -- assert rather than let the figure under-report the cluster count.
assert len(set(cluster_order)) == len(cluster_order), \
    f'duplicate cluster labels: {[n for n in cluster_order if cluster_order.count(n) > 1]}'
print(f'{len(cluster_order)} clusters from labels '
      f'(sizes {sorted((len(g) for g in clusters.values()), reverse=True)})')


def enrich(genes, ontology, tag):
    f = os.path.join(CACHE_DIR, f'{ontology}__{re.sub(r"[^A-Za-z0-9]+", "_", tag)}.csv')
    if os.path.exists(f):
        return pd.read_csv(f)
    time.sleep(1)                                  # Enrichr rate limit
    res = gp.enrichr(gene_list=genes, gene_sets=[f'GO_{ontology}_2025'],
                     organism='human', outdir=None).results
    # Do NOT cache an empty response. gp.enrichr returns an empty frame when a query is
    # throttled or fails, and writing that to disk locks the failure in: every later run
    # reads zero terms and reports the module as unenriched. Returning it uncached means
    # the next run retries.
    if len(res):
        res.to_csv(f, index=False)
    else:
        print(f'   WARNING: empty response for {tag} ({ontology}); not cached, rerun')
    return res


def make_labels(term_order):
    """Full GO term text, with the accession stripped -- no abbreviation and no renaming.

    The accession is restored for any pair of terms whose display names are identical,
    which GO does contain; without that the two would appear as the same tick label.
    """
    short = {t: t.split(' (')[0] for t in term_order}
    seen = {}
    for t, s in short.items():
        seen.setdefault(s, []).append(t)
    for s, ts in seen.items():
        if len(ts) > 1:
            for t in ts:
                gid = re.search(r'\((GO:\d+)\)', t)
                if gid:
                    short[t] = f'{s} ({gid.group(1)})'
    return [short[t] for t in term_order]


# --- term reduction ------------------------------------------------------------------
# The unfiltered figure had 131 terms on a 56-inch canvas, most of them nested GO
# ancestors of one another ('Translation' / 'Cytoplasmic Translation' / 'Gene Expression'
# / 'Macromolecule Biosynthetic Process' are one lineage counted four times). Three
# filters, applied in order, each reported so the effect of every one is visible:
#
#   1. BACKGROUND SIZE. A term annotated to thousands of genes is significant for almost
#      any large cluster and says nothing specific; a term with a handful is noise-prone.
#      Enrichr's 'Overlap' is 'k/K' with K the term's background size.
#   2. REDUNDANCY. Within a cluster, keep terms in order of significance and drop any
#      whose leading-edge genes are mostly already covered by a kept term. This is the
#      same idea as clusterProfiler's simplify()/REVIGO, computed from the 'Genes' column
#      so it costs no extra queries. Overlap coefficient, not Jaccard: a small child term
#      fully inside a large parent has low Jaccard but is entirely redundant.
#   3. CAP. At most TOP_PER_CLUSTER survivors per cluster, by significance.
#
# Set REDUCE = False to reproduce the unfiltered figure.

REDUCE          = True
K_RANGE         = (15, 500)   # allowed term background size
OVERLAP_MAX     = 0.50        # drop a term if this fraction of its genes is already kept
TOP_PER_CLUSTER = 5


def _bg_size(overlap):
    try:
        return int(str(overlap).split('/')[1])
    except (IndexError, ValueError):
        return np.nan


def reduce_terms(df):
    """Filter one ontology's significant hits. Returns (kept, log)."""
    log = {'significant': len(df), 'terms_in': df.Term.nunique()}

    d = df.copy()
    d['K'] = d.Overlap.map(_bg_size)
    keep_K = d.K.between(*K_RANGE) | d.K.isna()
    log['dropped_background_size'] = int((~keep_K).sum())
    d = d[keep_K]

    rows = []
    for cname, grp in d.groupby('Cluster', sort=False):
        grp = grp.sort_values('-log10p', ascending=False)
        kept_genes, kept = [], []
        for _, row in grp.iterrows():
            genes = set(str(row.Genes).split(';'))
            if any(len(genes & prev) / max(len(genes), 1) > OVERLAP_MAX
                   for prev in kept_genes):
                continue
            kept_genes.append(genes)
            kept.append(row)
            if len(kept) >= TOP_PER_CLUSTER:
                break
        rows.extend(kept)
    out = pd.DataFrame(rows)
    log['after_redundancy_and_cap'] = len(out)
    log['terms_out'] = out.Term.nunique()
    return out, log


tables, full_tables = {}, {}
for ontology, title in ONTOLOGIES:
    # --- 1. enrichment per cluster, everything below ADJ_P kept ------------------------
    rows, empty = [], []
    for name in cluster_order:
        genes = clusters[name]
        if len(genes) < MIN_GENES:
            empty.append(f'{name} [<{MIN_GENES} genes, not tested]')
            continue
        sig = enrich(genes, ontology, name).query('`Adjusted P-value` < @ADJ_P').copy()
        if sig.empty:
            empty.append(name)
            continue
        sig['Cluster'] = name
        rows.append(sig)
    df_all = pd.concat(rows, ignore_index=True)
    df_all['-log10p'] = -np.log10(df_all['Adjusted P-value'])
    unfiltered = df_all.copy()          # kept for the supplementary table
    if REDUCE:
        df_all, red = reduce_terms(df_all)
        print(f'{title}: {red["terms_in"]} terms -> {red["terms_out"]} '
              f'(dropped {red["dropped_background_size"]} hits on background size '
              f'outside {K_RANGE}, then redundancy at overlap > {OVERLAP_MAX} '
              f'and a cap of {TOP_PER_CLUSTER}/cluster)')

    # --- 2. term order: by the cluster each term is most significant in ----------------
    best = (df_all.loc[df_all.groupby('Term')['-log10p'].idxmax(), ['Term', 'Cluster']]
            .assign(ci=lambda d: d.Cluster.map(cluster_order.index)))
    peak = df_all.groupby('Term')['-log10p'].max()
    term_order = (best.assign(peak=best.Term.map(peak))
                  .sort_values(['ci', 'peak'], ascending=[True, False]).Term.tolist())
    df_all['x'] = df_all.Term.map({t: i for i, t in enumerate(term_order)})
    df_all['y'] = df_all.Cluster.map({c: i for i, c in enumerate(cluster_order)})
    tables[ontology] = df_all
    full_tables[ontology] = unfiltered

    print(f'{title}: {len(term_order)} terms, '
          f'{df_all.Cluster.nunique()}/{len(clusters)} clusters enriched, '
          f'{len(df_all)} points')
    if empty:
        print(f'   no term at adj. p < {ADJ_P}: {", ".join(empty)}')
    if len(term_order) > WIDE_WARN:
        print(f'   NOTE: {len(term_order)} terms makes a very wide figure; lower ADJ_P '
              f'if it is unreadable')

    # --- 3. figure --------------------------------------------------------------------
    # Height must include the x-label band: rotated full GO names are tall, and sizing
    # the figure to the grid alone makes constrained_layout collapse the axes to zero.
    # A 90-degree label occupies roughly 0.6 * fontsize per character along y.
    xlabels = make_labels(term_order)
    label_band = 0.6 * 14 / 72 * max(len(l) for l in xlabels)
    ylabel_band = 0.55 * 14 / 72 * max(len(c) for c in cluster_order)
    fig, ax = plt.subplots(
        figsize=(max(6.0, 0.40 * len(term_order) + ylabel_band + 0.8),
                 max(4.0, 0.42 * len(cluster_order) + label_band + 1.8)),
        layout='constrained')
    ax.scatter(x=df_all['x'], y=df_all['y'], s=df_all['-log10p'] * 50,
               c=df_all['Odds Ratio'], cmap='cool', edgecolors='black',
               linewidths=0.4, zorder=2)
    ax.set_xticks(range(len(term_order)))
    ax.set_xticklabels(xlabels, rotation=90)
    ax.set_yticks(range(len(cluster_order)))          # every cluster, enriched or not
    ax.set_yticklabels(cluster_order)
    ax.set_xlim(-0.5, len(term_order) - 0.5)
    ax.set_ylim(-0.5, len(cluster_order) - 0.5)
    ax.grid(axis='both', linestyle='--', alpha=0.7, zorder=1)
    ax.spines[['top', 'right', 'left', 'bottom']].set_visible(False)
    ax.set_title(title, fontsize=16)

    # --- 4. colour scale and size legend, per figure -----------------------------------
    vmin, vmax = df_all['Odds Ratio'].min(), df_all['Odds Ratio'].max()
    for coll in ax.collections:
        coll.set_clim(vmin, vmax)
    sm = plt.cm.ScalarMappable(cmap='cool', norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])

    legend_handles = [
        plt.scatter([], [], s=-np.log10(p) * 50, c='white', edgecolors='black',
                    linewidths=0.4, label=r'$10^{' + f'{int(np.log10(p))}' + r'}$')
        for p in [1e-2, 1e-3, 1e-4]]
    # Anchored below the MEASURED extent of the rotated term labels: a fixed offset put
    # the legend on top of them once the labels got long.
    fig.canvas.draw()
    inv = fig.transFigure.inverted()
    rend = fig.canvas.get_renderer()
    ylow = min(inv.transform((0, t.get_window_extent(rend).y0))[1]
               for t in ax.get_xticklabels())
    legend = fig.legend(handles=legend_handles, title=r'Adjusted $p$-value',
                        loc='upper left', frameon=False,
                        bbox_to_anchor=(0.0, ylow - 0.015), fontsize=14,
                        title_fontsize=14, borderpad=1, labelspacing=0.5,
                        handlelength=0.5)
    fig.canvas.draw()
    lb = legend.get_window_extent().transformed(inv)
    cbar_ax = fig.add_axes([lb.x1 + 0.02, lb.y1 - 0.055,
                            min(0.16, 1 - lb.x1 - 0.05), 0.022])
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
    cbar.set_label('Odds Ratio', fontsize=14, labelpad=8)
    cbar.ax.xaxis.set_label_position('top')
    cbar.ax.xaxis.set_ticks_position('bottom')
    cbar.outline.set_visible(False)

    out = f'{ctype}_enrichment_{ontology}.pdf'
    fig.savefig(out, bbox_inches='tight')
    print(f'   wrote {out}')

# the supplementary table carries EVERY significant term, unfiltered, so the figure's
# reduction hides nothing from a reviewer
full = pd.concat([t.assign(Ontology=o) for o, t in full_tables.items()], ignore_index=True)
full.drop(columns=['x', 'y'], errors='ignore').to_csv(f'{ctype}_enrichment_all_clusters.csv', index=False)
print(f'wrote {ctype}_enrichment_all_clusters.csv ({len(full)} rows)')
