import random
import numpy as np
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity

def get_gene_trajectories(test_chunk, n_permutations=int(1e4), ceil=20, specific_genes=None, tissue_filter=None, cell_type_filter=None, disease_filter=None):
    gene_trajectories = {}
    ceil_ages = lambda ages, ceil=ceil: np.clip((np.ceil(ages) / ceil).astype(int) * ceil + ceil, ceil, 100)
    for key in tqdm(test_chunk.uns_keys()):
        if "status" not in key: continue
        
        gene = key.split('_status')[0]
        gene_emb_ages = test_chunk.obs['age'][ test_chunk.uns[f'{gene}_status_embeddings'] ]
        extra_mask = np.array([True]*len(gene_emb_ages))
        
        age_embeddings = test_chunk.uns['age_embeddings'][ test_chunk.uns[f'{gene}_status_embeddings'] ]
        if tissue_filter is not None:
            extra_mask = extra_mask * (test_chunk.obs['tissue_general'][ test_chunk.uns[f'{gene}_status_embeddings'] ] == tissue_filter).values
        
        if cell_type_filter is not None:
            extra_mask = extra_mask * (test_chunk.obs['cell_type'][ test_chunk.uns[f'{gene}_status_embeddings'] ] == cell_type_filter).values
        
        if disease_filter is not None:
            extra_mask = extra_mask * (test_chunk.obs['disease'][ test_chunk.uns[f'{gene}_status_embeddings'] ] == disease_filter).values

        if not gene_emb_ages.tolist(): continue
        valid_mask = ~gene_emb_ages.isna().values & extra_mask
        gene_emb_ages = gene_emb_ages[valid_mask].apply(lambda x: ceil_ages(x))

        if len(gene_emb_ages.value_counts().reset_index()) >= int(100/ceil):
            gene_emb = test_chunk.uns[f'{gene}_embeddings'][valid_mask]
            age_embeddings = age_embeddings[valid_mask]

            grouped_age_cosine = pd.DataFrame(np.diag(cosine_similarity(gene_emb, age_embeddings)), index=gene_emb_ages).groupby(level=0).mean().sort_index()
            similarities = {}
            ref_indices = gene_emb_ages == ceil
            if ref_indices.sum() < 2: continue
            for age in gene_emb_ages.drop_duplicates().sort_values(ascending=True):
                indices = gene_emb_ages == age
                if age != ceil:
                    similarities[age] = cosine_similarity(gene_emb[indices], gene_emb[ref_indices]).flatten().mean()
                else:
                    similarities[age] = cosine_similarity(gene_emb[indices], gene_emb[ref_indices])[np.triu_indices(indices.sum(), k=1)].mean()
            gene_trajectories[gene] = [list(similarities.keys()), list(similarities.values()), grouped_age_cosine.index.values, grouped_age_cosine.values.flatten()]
    return gene_trajectories

def get_null_distribution(test_chunk, n_permutations=int(1e4), ceil=20, tissue_filter=None, cell_type_filter=None, disease_filter=None):
    null_distribution = {}
    ceil_ages = lambda ages, ceil=ceil: np.clip((np.ceil(ages) / ceil).astype(int) * ceil + ceil, ceil, 100)
    for key in tqdm(test_chunk.uns_keys()):
        if "status" not in key: continue
        
        gene = key.split('_status')[0]
        gene_emb_ages = test_chunk.obs['age'][ test_chunk.uns[f'{gene}_status_embeddings'] ]
        extra_mask = np.array([True]*len(gene_emb_ages))
        
        age_embeddings = test_chunk.uns['age_embeddings'][ test_chunk.uns[f'{gene}_status_embeddings'] ]
        if tissue_filter is not None:
            extra_mask = extra_mask * (test_chunk.obs['tissue_general'][ test_chunk.uns[f'{gene}_status_embeddings'] ] == tissue_filter).values
        
        if cell_type_filter is not None:
            extra_mask = extra_mask * (test_chunk.obs['cell_type'][ test_chunk.uns[f'{gene}_status_embeddings'] ] == cell_type_filter).values
        
        if disease_filter is not None:
            extra_mask = extra_mask * (test_chunk.obs['disease'][ test_chunk.uns[f'{gene}_status_embeddings'] ] == disease_filter).values

        if not gene_emb_ages.tolist(): continue
        valid_mask = ~gene_emb_ages.isna().values & extra_mask
        gene_emb_ages = gene_emb_ages[valid_mask].apply(lambda x: ceil_ages(x))

        if len(gene_emb_ages.value_counts().reset_index()) >= int(100/ceil):
            gene_emb = test_chunk.uns[f'{gene}_embeddings'][valid_mask]
            age_embeddings = age_embeddings[valid_mask]
            gene_null = []
            for i in range(n_permutations):

                gene_emb_ages_arr = gene_emb_ages.values
                random.shuffle(gene_emb_ages_arr)
                similarities = {}
                ref_indices = gene_emb_ages_arr == ceil
                if ref_indices.sum() < 2: continue
                for age in gene_emb_ages.drop_duplicates().sort_values(ascending=True):
                    indices = gene_emb_ages_arr == age
                    num_expl = 20
                    if age != ceil:
                        similarities[age] = cosine_similarity(gene_emb[indices][:num_expl], gene_emb[ref_indices][:num_expl]).flatten().mean()
                    else:
                        similarities[age] = cosine_similarity(gene_emb[indices][:num_expl], gene_emb[ref_indices][:num_expl])[np.triu_indices(min(indices.sum(), num_expl), k=1)].mean()
                gene_null.append(np.abs(np.array(list(similarities.values())) - list(similarities.values())[0]).max())
            null_distribution[gene] = gene_null
    return null_distribution

import random
import pandas as pd
sample_k_avg = lambda lst, k: np.array([np.mean(random.choices(sublist, k=k)) for sublist in lst])

def hopf_metrics(time_values, state_values, raw_state_values):
    recurrence = state_values.std()
    
    recurrence_rate = lambda x, epsilon: np.mean(np.linalg.norm(x[:, None] - x[None, :], axis=-1) < epsilon)
    recurrence =  np.mean([recurrence_rate(sample_k_avg(raw_state_values, k=5), epsilon=0.4)  for _ in range(40)])
    
    divergence = state_values.std()# np.abs(np.diff(state_values - state_values[0])).sum()
    return pd.Series({'recurrence': recurrence, 'divergence': divergence})

import scanpy as sc

# A processed AnnData with age, predicted age, age embeddings, tissue embeddings, cell type embeddings
# and gene embeddings for the first 20k Genes ranked via CellxGene Seurat_v3 HVG
validation_set = sc.read_h5ad("/media/lleger/LaCie/age_cage/age_chunk3400.h5ad")

gene_trajectories_dict = get_gene_trajectories(validation_set, ceil=20, disease_filter="normal")

gene_trajectories = pd.DataFrame.from_dict(gene_trajectories_dict, orient="index", columns=['young_sim_x', 'young_sim_y', 'age_sim_x', 'age_sim_y'])
gene_trajectories['drift'] = gene_trajectories['young_sim_y'].apply(lambda x: np.abs(x - x[0]).max())

null_dist = get_null_distribution(validation_set, ceil=20, n_permutations=int(1e3), disease_filter="normal") #1h30 runtime

gene_trajectories['null_dist'] = null_dist

from scipy.stats import norm
from statsmodels.stats.multitest import multipletests

def compute_z_and_p(drift, null_dist):
    null_array = np.array(null_dist)
    mean = null_array.mean()
    std = null_array.std(ddof=1)
    z = (drift - mean) / std
    p = 1 - norm.cdf(z)
    return z, p

results = gene_trajectories.apply(lambda row: compute_z_and_p(row['drift'], row['null_dist']), axis=1)
gene_trajectories[['z', 'pval']] = pd.DataFrame(results.tolist(), index=gene_trajectories.index)
gene_trajectories['fdr'] = multipletests(gene_trajectories['pval'], method='fdr_bh')[1]

gene_trajectories['dissipative'] = gene_trajectories['drift'].apply(lambda x: "Dissipative" if x > np.percentile(gene_trajectories['drift'], 90) else "Conservative")
gene_trajectories['significant'] = gene_trajectories['fdr'] < 0.05
print("Percentage of significant genes per label:\n", gene_trajectories.groupby('dissipative').value_counts(subset=['significant'], normalize=True)) 

print('Z and FDR of significant dissipative genes:\n',
       gene_trajectories[(gene_trajectories['dissipative'] == 'Dissipative') &
                          (gene_trajectories['significant'] == True)][['z', 'fdr']].mean())

gene_trajectories[['recurrence', 'divergence']] = gene_trajectories.apply(
    lambda row: hopf_metrics(np.array(row['young_sim_x']), np.array(row['young_sim_y']), row['young_sim_y_raw']),
    axis=1
)

pd.to_pickle('saved_trajectories.pkl')