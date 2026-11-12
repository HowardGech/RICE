r"""Full GO enrichment result for all modules, as one CSV -- queried the way the original
plotting loop queries, with no cache in the path.

Why this version exists: the cached variant reported 2 significant Molecular Function
terms for RPE1 where a fresh run of the original loop gives 7. The only step the two
pipelines did not share was the on-disk Enrichr cache, which could retain an empty
response from a throttled query and report the module as unenriched ever after. This
cell removes the cache entirely and retries an empty response instead.

Matching the original loop:
  * EVERY module is queried -- no minimum size. (Enrichr is called on two-gene modules
    too; whether to trust such a result is a separate question, and 'Module size' is in
    the output so those rows can be dropped at will.)
  * one gp.enrichr call per module per ontology, with a 1 s pause, and the same
    GO_*_2025 gene sets;
  * terms kept at adjusted p < ADJ_P.

Differing from the original loop, deliberately: modules are iterated directly rather
than collected into a dict keyed by their first two gene symbols. That dict silently
overwrites one module with another when two share leading symbols, so the original can
query fewer modules than exist. Iterating cannot lose a module.
"""
import time
import numpy as np
import pandas as pd
import gseapy as gp

ADJ_P      = 0.01
RETRIES    = 3            # an empty response is retried; Enrichr returns [] when throttled
PAUSE      = 1.0
OUT_CSV    = f'{ctype}_GO_enrichment_all_modules.csv'
DROP_COLS  = ['Old P-value', 'Old Adjusted P-value']
ONTOLOGIES = [('Biological_Process', 'Biological Process'),
              ('Molecular_Function', 'Molecular Function')]


def module_stem(genes):
    """Member genes only; 'Module size' is a separate column."""
    return f'{genes[0]},{genes[1]},...' if len(genes) > 2 else ','.join(genes)


modules = []
for cid in np.unique(labels):
    genes = gene_names_non_isolated[labels == cid].tolist()
    modules.append((module_stem(genes), genes))
_stems = [m[0] for m in modules]
_dupes = {s for s in _stems if _stems.count(s) > 1}
if _dupes:                      # repeated leading symbols: restore the size on those only
    modules = [(f'{s} (n={len(g)})' if s in _dupes else s, g) for s, g in modules]
    print(f'disambiguated modules with repeated leading genes: {sorted(_dupes)}')


def query(genes, ontology):
    """gp.enrichr, retrying while the response is empty. Returns (frame, n_attempts)."""
    for attempt in range(1, RETRIES + 1):
        time.sleep(PAUSE)
        res = gp.enrichr(gene_list=genes, gene_sets=[f'GO_{ontology}_2025'],
                         organism='human', outdir=None).results
        if len(res):
            return res, attempt
    return res, RETRIES


rows, report = [], []
for ontology, title in ONTOLOGIES:
    for name, genes in modules:
        res, attempts = query(genes, ontology)
        n_sig = int((res['Adjusted P-value'] < ADJ_P).sum()) if len(res) else 0
        report.append(dict(Ontology=title, Module=name, Size=len(genes),
                           Returned=len(res), Significant=n_sig, Attempts=attempts))
        print(f'   {title[:3]} {name:<32} {len(genes):>4} genes  '
              f'{len(res):>5} returned  {n_sig:>3} significant'
              + ('  [EMPTY after retries]' if not len(res) else
                 f'  [retried {attempts}x]' if attempts > 1 else ''))
        if not len(res):
            continue
        d = res[res['Adjusted P-value'] < ADJ_P].copy()
        if d.empty:
            continue
        d.insert(0, 'Ontology', title)
        d.insert(1, 'Module', name)
        d.insert(2, 'Module size', len(genes))
        # Enrichr's 'Overlap' is 'k/K': module genes annotated to the term / the term's
        # total annotation. Split out so both can be filtered on.
        kK = d.Overlap.astype(str).str.split('/', expand=True)
        d['Genes in term'] = pd.to_numeric(kK[0], errors='coerce')
        d['Term size'] = pd.to_numeric(kK[1], errors='coerce') if kK.shape[1] > 1 else np.nan
        d['GO accession'] = d.Term.str.extract(r'\((GO:\d+)\)')
        rows.append(d)

out = pd.concat(rows, ignore_index=True)
dropped = [c for c in DROP_COLS if c in out.columns]
out = out.drop(columns=dropped)
order = [m[0] for m in modules]
out['Module'] = pd.Categorical(out['Module'], order, ordered=True)
out = out.sort_values(['Ontology', 'Module', 'Adjusted P-value'])
out.to_csv(OUT_CSV, index=False)

report = pd.DataFrame(report)
print(f'\nwrote {OUT_CSV}: {len(out)} rows, {len(out.columns)} columns')
print(f'   dropped (Enrichr legacy, all zeros): {dropped or "none present"}')
for _, title in ONTOLOGIES:
    s = out[out.Ontology == title]
    print(f'   {title}: {len(s)} rows, {s.Term.nunique()} unique terms, '
          f'{s["Module"].nunique()}/{len(modules)} modules')
_empty = report[report.Returned == 0]
if len(_empty):
    print(f'   {len(_empty)} queries returned nothing after {RETRIES} attempts -- these '
          f'are NOT known to be unenriched:')
    for r in _empty.itertuples():
        print(f'      {r.Ontology}: {r.Module} ({r.Size} genes)')
report.to_csv(f'{ctype}_GO_enrichment_query_report.csv', index=False)
