import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import functools, operator
from itertools import product
import torch.nn.functional as F
from braceexpand import braceexpand
from scanpy import read_h5ad, concat
from cellular_aging_map.eval import metrics
from cellular_aging_map.data_utils.tokenization import normalise_str

age_relabel = pd.read_csv('perturbgene/data_utils/age_relabeling.csv', index_col=0)
age_map = {k:v for k,v in zip(age_relabel['label'], age_relabel['age'])}
ceil_ages = lambda ages, ceil=10: np.clip((np.ceil(ages) / ceil).astype(int) * ceil + ceil, ceil, 100)

def get_phenotype_cells(paths: list, n: int = 1, match_other_phenotypes=False, get_info=False, **phenotypes):
    """
    Method to get N cells with a specific phenotype

    n: int number of cells to get
    paths: list of braceexpandable paths to h5ad files for reading
    phenotypes: dict type of sex=['M'], tissue=['blood'] etc
    """
    processed_paths = [path for bracepath in paths for path in braceexpand(bracepath)]
    cells = []

    combination_types = list(phenotypes)[:]
    required = {phenotype_combination: n for phenotype_combination in product(*phenotypes.values())}
    for pdx, path in enumerate(processed_paths):
        if not len(required):
            break
        anndata = read_h5ad(path)
        anndata.obs['age'] = anndata.obs['development_stage'].apply(lambda x: ceil_ages(age_map[normalise_str(x)]))
        anndata.obs.reset_index(drop=True, inplace=True)
        anndata.obs.index = anndata.obs.index.values.astype(str)
        if get_info:  print(anndata.obs[list(phenotypes.keys())].value_counts())
        # phenotypes is a dict where sex = M, tissue = blood etc
        mask = functools.reduce(operator.and_, (anndata.obs[phenotypic_type].isin(value) for phenotypic_type, value in phenotypes.items() if phenotypic_type in anndata.obs.columns))
        filtered_anndata = anndata[mask]

        if match_other_phenotypes and pdx == 0:
            index_of_sample = filtered_anndata.obs.value_counts().reset_index()[[phenotypic_type for phenotypic_type in ['age','tissue_general', 'cell_type', 'tissue', 'sex', 'development_stage', 'disease'] if phenotypic_type in filtered_anndata.obs.columns] + [0]].max().index[0]
            for col in [column for column in filtered_anndata.obs.iloc[index_of_sample:index_of_sample+1].columns if column not in phenotypes]:
                phenotypes[col] =  [filtered_anndata.obs.iloc[index_of_sample][col]]
            mask = functools.reduce(operator.and_, (anndata.obs[phenotypic_type].isin(value) for phenotypic_type, value in phenotypes.items()))
            filtered_anndata = anndata[mask]
        
        for combination in list(required.keys()): # select cells from specific phenotype combo
            mask = functools.reduce(operator.and_, (filtered_anndata.obs[phenotypic_type] == value for phenotypic_type, value in zip(combination_types,  combination)))
            selected_cells = filtered_anndata[mask][:required[combination]]
            cells.append(selected_cells)
            required[combination] -= len(selected_cells)
            if (required[combination] < 1): del required[combination]
    
    cells_total = concat(cells)
    print(f'Loaded {len(cells_total)} cells\n', cells_total.obs[list(phenotypes.keys())].value_counts().reset_index().sort_values(by=combination_types))
    
    return cells_total

def get_logits(shard, model, tokenizer):
    embeddings = {"tissue_prob":[], "cell_type_prob":[]}
    for cell in tqdm(shard, "Embeddings"):
        cell_dict = metrics.prepare_cell(cell, tokenizer) 
        
        cell_dict["input_ids"][1 + tokenizer.phenotypic_types.index('tissue')] = 2
        with torch.no_grad():
            output = model(**{key: val.to(model.device).unsqueeze(0) for key, val in cell_dict.items() if key != 'str_labels'})
        prob = F.softmax(output.logits, dim=-1)[0]
        embeddings["tissue_prob"].append(prob[1 + tokenizer.phenotypic_types.index("tissue")].cpu().detach().numpy())
        
        cell_dict["input_ids"][1 + tokenizer.phenotypic_types.index('cell_type')] = 2
        with torch.no_grad():
            output = model(**{key: val.to(model.device).unsqueeze(0) for key, val in cell_dict.items() if key != 'str_labels'})
        prob = F.softmax(output.logits, dim=-1)[0]
        embeddings["cell_type_prob"].append(prob[1 + tokenizer.phenotypic_types.index("cell_type")].cpu().detach().numpy())

    for key in embeddings:
        shard.uns[f'{key}_embeddings'] = np.stack(embeddings[key])
    del embeddings

def get_age_gap(shard, model, tokenizer):
    age_idx = 1 + tokenizer.phenotypic_types.index('development_stage')
    
    ages = []
    age_predictions = []
    for cell in tqdm(shard):
        prep = metrics.prepare_cell(cell, tokenizer)
        prep['input_ids'][age_idx] = 2
        y = prep['str_labels'][age_idx]
        del prep['str_labels']
        y_pred = model
        with torch.no_grad():
            output = model(**{key: val.to(model.device).unsqueeze(0) for key, val in prep.items()})
        probabilities = F.softmax(output.logits[:, age_idx] / 0.01, dim=-1)

        y_pred = [tokenizer.flattened_tokens[torch.distributions.Categorical(probability).sample().item()] for probability in probabilities][0]
        ages.append(age_map[y])
        age_predictions.append(age_map.get(y_pred, np.nan))
    
    shard.obs['age_pred'] = age_predictions
    return shard

# Ensure we get 100 cells from each tissue from each age group, this function can search within the first 2 million cells of the test set
cells = get_phenotype_cells(paths=['/media/rohola/ssd_storage/primary/cxg_chunk{3001..3200}.h5ad'], n=100, disease=['normal'], tissue_general=['lung', 'kidney', 'skin of body', 'brain',], age=[10, 30, 40, 50, 60, 70, 80, 90])

from cellular_aging_map.model.model import load_trained_model
model, tokenizer = load_trained_model("../../../saved/polygene-cam/", checkpoint_n=0)
get_logits(cells, model, tokenizer)
get_age_gap(cells, model, tokenizer)

from scipy.stats import entropy
for phenotype in ['tissue', 'cell_type']:
    cells.obs[f'{phenotype}_entropy'] = pd.Series(list(cells.uns[f'{phenotype}_prob_embeddings'])).apply(lambda p:  entropy(p, base=10)).tolist()

if cells.obs.index.name == 'soma_joinid': cells.obs.index.name = 'index_soma_joinid'
cells.write(f"/media/lleger/LaCie/age_cage/entropy_cells.h5ad")