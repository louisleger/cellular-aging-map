import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch.nn.functional as F
from cellular_aging_map.eval import metrics

age_relabel = pd.read_csv('age_relabeling.csv', index_col=0)
age_map = {k:v for k,v in zip(age_relabel['label'], age_relabel['age'])}

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
    
    shard.obs['age'] = ages
    shard.obs['age_pred'] = age_predictions
    return shard


def get_embeddings(shard, model, tokenizer, n_genes=10):
    embeddings = {"age":[], "tissue":[], "cell_type":[]}
    genes_of_interest = list(tokenizer.gene_type_id_map.keys())[:n_genes]
    for cell in tqdm(shard, "Embeddings"):
        cell_dict = metrics.prepare_cell(cell, tokenizer) 
        with torch.no_grad():
            output = model(**{key: val.to(model.device).unsqueeze(0) for key, val in cell_dict.items() if key != 'str_labels'})

        #cell_embeddings = F.softmax(output.logits, dim=-1)[0]
        cell_embeddings = output.hidden_states[0] # tensor shape (S, D)
        embeddings["age"].append(cell_embeddings[1 + tokenizer.phenotypic_types.index("development_stage")].cpu().detach().numpy())
        embeddings["tissue"].append(cell_embeddings[1 + tokenizer.phenotypic_types.index("tissue")].cpu().detach().numpy())
        embeddings["cell_type"].append(cell_embeddings[1 + tokenizer.phenotypic_types.index("cell_type")].cpu().detach().numpy())

        with torch.no_grad():
            output = model(**{key: val.to(model.device).unsqueeze(0) for key, val in cell_dict.items() if key != 'str_labels'})
        #embeddings.set_default("age_entropy", []).append(entropy(cell_embeddings[1 + tokenizer.phenotypic_types.index("development_stage")].cpu().detach().numpy()))
        cell_genes = set(cell_dict['token_type_ids'][tokenizer.gene_token_type_offset:-1].cpu().detach().numpy())
        cell_token_type_list = cell_dict['token_type_ids'].cpu().detach().tolist()
        for gene in genes_of_interest:
            gene_token_type_id = tokenizer.gene_type_id_map[gene] + tokenizer.gene_token_type_offset
            if gene_token_type_id in cell_genes:
                embeddings.setdefault(f"{shard.var.loc[gene, 'feature_name']}", []).append(
                    cell_embeddings[cell_token_type_list.index(gene_token_type_id)].cpu().detach().numpy()
                    )
                embeddings.setdefault(f"{shard.var.loc[gene, 'feature_name']}_status", []).append(
                    True
                    )
            else:
                embeddings.setdefault(f"{shard.var.loc[gene, 'feature_name']}_status", []).append(
                    False
                    )
    for key in embeddings:
        shard.uns[f'{key}_embeddings'] = np.stack(embeddings[key])
    del embeddings

DATA_PATH = f"/media/rohola/ssd_storage/primary/"
import scanpy as sc
 
from cellular_aging_map.model.model import load_trained_model
model, tokenizer = load_trained_model("../../../saved/polygene-cam/", checkpoint_n=0)

for chunk in range(100):
    test_chunk = sc.read_h5ad(f"/media/rohola/ssd_storage/primary/cxg_chunk{int(3400 + chunk)}.h5ad")
    get_age_gap(test_chunk, model, tokenizer)
    get_embeddings(test_chunk, model, tokenizer, n_genes=int(2e4))
    if test_chunk.obs.index.name == 'soma_joinid': test_chunk.obs.index.name = 'index_soma_joinid'
    test_chunk.write(f"/media/lleger/LaCie/age_cage/entropy{int(3400 + chunk)}.h5ad")