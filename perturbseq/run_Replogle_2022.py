# %%
import os
import numpy as np
import pandas
import sys
sys.path.append('../source/')
from scRICE_CF import *
from utils import *
import cupy as cp
import anndata as ad

args = sys.argv[1:]
ctype = str(args[0])


# %%
if ctype == 'K562':
    adata = ad.read_h5ad('data/K562_essential_raw_singlecell_01.h5ad')
elif ctype == 'RPE1':
    adata = ad.read_h5ad('data/rpe1_raw_singlecell_01.h5ad')

# %%
adata.obs_keys()

# %%
if ctype == 'K562':
    gnum = 100
elif ctype == 'RPE1':
    gnum = 100
gene_id_inc = adata.obs.gene_id.value_counts()[adata.obs.gene_id.value_counts() >= gnum].index

def compute_gene_expression_means(adata, gene_ids):
    means = {}
    for gene_id in gene_ids:
        if gene_id == 'non-targeting':
            continue
        if gene_id not in adata.var.index.values:
            print(f"Gene {gene_id} not found in adata.var.index. Skipping.")
            continue
        gene_idx = np.where(adata.var.index == gene_id)[0][0]
        cell_gene_idx = adata.obs.gene_id == gene_id
        cell_control_idx = adata.obs.gene_id == 'non-targeting'
        gene_exp = adata.X[cell_gene_idx, gene_idx].mean()
        control_exp = adata.X[cell_control_idx, gene_idx].mean()
        means[gene_id] = {'gene_mean': gene_exp, 'control_mean': control_exp}
    return pandas.DataFrame(means).T
means_df = compute_gene_expression_means(adata, gene_id_inc)


means_df['ratio'] = means_df['gene_mean'] / means_df['control_mean']

gene_real_inc = list(means_df.index)


gene_targeting_filtered = []
z_scores = {}

adata.obs['UMI_size'] = adata.obs['core_adjusted_UMI_count']/adata.X.shape[1]

pbar = tqdm(gene_real_inc)
for gene_id in gene_real_inc:
    pbar.update(1)
    gene_idx = np.where(adata.var.index == gene_id)[0][0]
    cell_gene_idx = adata.obs.gene_id == gene_id
    cell_control_idx = adata.obs.gene_id == 'non-targeting'
    all_gene_values = adata.X[:,gene_idx].flatten() / adata.obs['UMI_size'].values.reshape(-1)
    all_gene_values = np.log(1 + all_gene_values)
    gene_values = all_gene_values[cell_gene_idx].flatten()
    control_values = all_gene_values[cell_control_idx].flatten()
    zs = (gene_values.mean() - control_values.mean()) / np.sqrt(gene_values.var()/len(gene_values) + control_values.var()/len(control_values))
    z_scores[gene_id] = zs
    if abs(zs) >= 5:
        mean_exp = adata.X[:,gene_idx].mean()
        if mean_exp >= .5:
            gene_targeting_filtered.append(gene_id)
pbar.close()
gene_real_inc = gene_targeting_filtered.copy()
gene_real_inc.append('non-targeting')
# %%
adata_real_inc = adata[adata.obs.gene_id.isin(gene_real_inc)].copy()

# %%
adata_real_inc.obs['library_size'] = adata_real_inc.X.mean(axis=1)
gene_counts = adata_real_inc.obs['gene_id'].value_counts()
adata_real_inc.obs['weights'] = adata_real_inc.X.shape[0]/len(gene_real_inc)/gene_counts[adata_real_inc.obs['gene_id']].values
gem_group_onehot = pandas.get_dummies(adata_real_inc.obs['gem_group'], prefix='gem_group', drop_first=True).astype(np.float32)
# %%
gene_targeting = gene_counts.index.categories[:-1]
gene_name_targeting = adata_real_inc.var.gene_name[gene_targeting].values

# %%
gene_columns = []
for gene in gene_targeting:
    gene_columns.append(np.where(adata_real_inc.var.index == gene)[0][0])
gene_columns = np.array(gene_columns)

# %%
data_X = adata_real_inc.X[:, gene_columns]
mask = np.zeros_like(data_X, dtype=bool)
for i in range(data_X.shape[0]):
    if adata_real_inc.obs['gene_id'].values[i] == 'non-targeting':
        continue
    j = np.where(adata_real_inc.obs['gene_id'].values[i] == gene_targeting)[0][0]
    if j.size > 0:
        mask[i, j] = True
weights = adata_real_inc.obs['weights'].values.reshape(-1,1)
Z = adata_real_inc.obs[['UMI_size','mitopercent']].values
Z[:,0] = np.log(Z[:,0])
Z = (Z - Z.mean(axis=0)) / Z.std(axis=0)
Z = np.hstack((Z, gem_group_onehot.values))

from tqdm.auto import tqdm
from sklearn.linear_model import LinearRegression
Z_front_size = Z.shape[1]

Offset = adata_real_inc.obs['UMI_size'].values.reshape(-1, 1)

normalized_X = data_X / Offset
normalized_X = np.log(1 + normalized_X)
normalized_X = normalized_X - normalized_X.mean(axis=0)
normalized_X = normalized_X / normalized_X.std(axis=0)

unique_genes = list(gene_targeting) + ['non-targeting']
np.random.seed(0)
train_cell_idx = np.array([])
test_cell_idx = np.array([])
train_gene_idx = dict()
test_gene_idx = dict()
train_cell_idx_dict = dict()
for gene in unique_genes:
    gene_cell_idx = np.where(adata_real_inc.obs['gene_id'].values == gene)[0]
    np.random.shuffle(gene_cell_idx)
    split_idx = int(0.8 * len(gene_cell_idx))
    train_cell_idx = np.concatenate((train_cell_idx, gene_cell_idx[:split_idx]))
    test_cell_idx = np.concatenate((test_cell_idx, gene_cell_idx[split_idx:]))
    train_cell_idx_dict[gene] = gene_cell_idx[:split_idx]
    train_gene_idx[gene] = (gene_cell_idx[:split_idx]).astype(int)
    test_gene_idx[gene] = (gene_cell_idx[split_idx:]).astype(int)
train_cell_idx = train_cell_idx.astype(int)
test_cell_idx = test_cell_idx.astype(int)


con_res = np.zeros_like(data_X)
for i in tqdm(range(data_X.shape[1])):
    lr = LinearRegression()
    predictors = np.hstack((Z, mask[:,i].reshape(-1,1)))
    lr.fit(predictors, normalized_X[:,i])
    con_res[:,i] = normalized_X[:,i] - lr.predict(predictors)
    

adata_not_inc = adata_real_inc[:,~adata_real_inc.var.index.isin(gene_targeting)].copy()
Offset = adata_real_inc.obs['UMI_size'].values.reshape(-1, 1)
other_gene_exp = np.log(1+adata_not_inc.X/Offset)
other_gene_exp = (other_gene_exp - other_gene_exp.mean(axis=0)) / other_gene_exp.std(axis=0)
for i in tqdm(range(other_gene_exp.shape[1])):
    lr = LinearRegression()
    predictors = np.hstack((Z,mask))
    lr.fit(predictors, other_gene_exp[:,i])
    other_gene_exp[:,i] = other_gene_exp[:,i] - lr.predict(predictors)
    
from sklearn.decomposition import PCA
other_gene_exp_pca = PCA(n_components=50).fit_transform(other_gene_exp)

np.savez(f'data/{ctype}_processed.npz', other_gene_exp_pca=other_gene_exp_pca, con_res=con_res, Z=Z, normalized_X=normalized_X, mask=mask, train_cell_idx=train_cell_idx, test_cell_idx=test_cell_idx)

Z = np.hstack((Z, other_gene_exp_pca))



cp.random.seed(0)
model = scRICE_CF(dtype=cp.float64)
model.prep(X=data_X[train_cell_idx], predictor = normalized_X[train_cell_idx], confounding_res = con_res[train_cell_idx], intervention_effect = mask[train_cell_idx], intervention_type='soft', add_intercept=True, Zm_g=Z[train_cell_idx], Zr_g=Z[train_cell_idx][:,:Z_front_size])

model.fit(regularizer='binsum', binsum_k=data_X.shape[1],pen1=1e-4, pen2=0, pen1_c=0, pen2_c=1e-2, lr=0.0001, lr_coef_multiplier = 2, max_iter=100, checkpoint=2000, h_tol=1e-4, loss_tol=1e-5,rho_init=1,verbose=True)

result = model.result()
result['gene_id'] = gene_targeting
result['gene_name'] = gene_name_targeting
np.savez(f'data/{ctype}_result.npz', **result)

