import os
import pandas as pd
import numpy as np
import torch
import argparse
import transformers
import openai
from collections import defaultdict
from utils import log_output, mkdir_path
from LLMModels import LLAMA2, ChatGLM2, GPT, DeepSeek

def load_entity_to_id_mapping(ent2ids_file_path):
    entity_to_id = {}
    with open(ent2ids_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            name, id_str = line.strip().split('\t')
            entity_to_id[name] = int(id_str)
    return entity_to_id

def load_id_to_entity_name_mapping(ent2ids_file_path):
    id_to_name = {}
    with open(ent2ids_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            name, id_str = line.strip().split('\t')
            id_to_name[int(id_str)] = name
    return id_to_name

def load_entid_to_readable_name_mapping(ent2name_file_path):
    entid_to_readable = {}
    with open(ent2name_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            entid, name = line.strip().split('\t')
            entid_to_readable[entid] = name
    return entid_to_readable

def load_entity_types(entity2type_file_path):
    entity_to_types = {}
    with open(entity2type_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                entity = parts[0]
                types = set(parts[1:])
                entity_to_types[entity] = types
    return entity_to_types

def load_transe_embedding(embedding_file):
    embedding_matrix = []
    with open(embedding_file, 'r', encoding='utf-8') as f:
        for line in f:
            embedding = list(map(float, line.strip().split()))
            embedding_matrix.append(embedding)
    return np.array(embedding_matrix)

def compute_relevance_score(entity, neighbors, transE_model, entity_freq,
                            entity_to_id, id_to_name, entid_to_readable,
                            entity_to_types, beta=0.2, topk=5):
    if entity not in entity_to_id:
        return [], {}

    entity_id = entity_to_id[entity]
    selected_neighbors = []
    candidates = []
    score_dict = {}
    type_usage_count = defaultdict(int)

    for nbr in neighbors:
        if nbr not in entity_to_id:
            continue
        nbr_id = entity_to_id[nbr]

        freq_score = 1.0 / (entity_freq.get(nbr, 1) + 1)
        sim_score = 1.0 / (np.linalg.norm(transE_model[entity_id] - transE_model[nbr_id]) + 1e-6)

        types = entity_to_types.get(nbr, set())
        if types:
            type_score_sum = sum(1.0 / (type_usage_count[t] + 1) for t in types)
            avg_type_score = type_score_sum / len(types)
        else:
            avg_type_score = 1.0

        final_score = freq_score + beta * sim_score / avg_type_score
        candidates.append((nbr, final_score, freq_score, sim_score, avg_type_score, types))

    entity_name = id_to_name.get(entity_id, entity)
    entity_readable = entid_to_readable.get(entity_name, entity_name)
    print(f'\nEntity: {entity_readable}')

    for _ in range(min(topk, len(candidates))):
        candidates.sort(key=lambda x: x[1], reverse=True)
        best_nbr, best_score, best_freq, best_sim, best_type_score, best_types = candidates.pop(0)
        selected_neighbors.append(best_nbr)
        score_dict[best_nbr] = (best_score, best_freq, best_sim)

        for t in best_types:
            type_usage_count[t] += 1

        nbr_name = id_to_name.get(entity_to_id[best_nbr], best_nbr)
        nbr_readable = entid_to_readable.get(nbr_name, nbr_name)

        print(f"  Neighbor: {nbr_readable}, Final Score: {best_score:.4f}, "
              f"Freq: {best_freq:.4f}, Sim: {best_sim:.4f}, TypePenalty: {best_type_score:.4f}")

    return selected_neighbors, score_dict

def expansion_cot_fb(args, llm):
    data_folder = args.LLMfold
    ent2ids_file = os.path.join(data_folder, 'ent2ids.txt')
    ent2name_file = os.path.join(data_folder, 'ent2name.txt')
    embedding_file = os.path.join(data_folder, 'entity2vec.TransE')
    graph_file = os.path.join(data_folder, 'processed_graph.tsv')
    entity2type_file = os.path.join(data_folder, 'entity2type.txt')

    entity_to_id = load_entity_to_id_mapping(ent2ids_file)
    id_to_name = load_id_to_entity_name_mapping(ent2ids_file)
    entid_to_readable = load_entid_to_readable_name_mapping(ent2name_file)
    entity_to_types = load_entity_types(entity2type_file)
    transE_model = load_transe_embedding(embedding_file)
    df = pd.read_csv(graph_file, sep='\t', header=None, names=['entity', 'neighbors'])
    entity_freq = df['neighbors'].str.split(',').explode().value_counts().to_dict()

    mkdir_path(data_folder)

    for topk in [1, 3, 5, 10]:
        prompt_file = os.path.join(data_folder, f'mycotdes_top{topk}.txt')
        nbr_file = os.path.join(data_folder, f'selected_neighbors_top{topk}.tsv')

        with open(prompt_file, 'w', encoding='utf-8') as pf, \
             open(nbr_file, 'w', encoding='utf-8') as nf:

            for _, row in df.iterrows():
                ent = row['entity']
                nbrs = row['neighbors'].split(',')

                selected, score_dict = compute_relevance_score(
                    ent, nbrs, transE_model, entity_freq,
                    entity_to_id, id_to_name, entid_to_readable,
                    entity_to_types,
                    beta=args.beta, topk=topk
                )

                selected_ids = [entity_to_id[n] for n in selected]
                selected_names = [id_to_name[i] for i in selected_ids]
                selected_readables = [entid_to_readable.get(name, name) for name in selected_names]

                readable_neighbors = [entid_to_readable.get(id_to_name[entity_to_id[n]], n) for n in selected]
                nf.write(f"{ent}\t{','.join(readable_neighbors)}\n")

                ent_readable = entid_to_readable.get(id_to_name[entity_to_id[ent]], ent)
                prompt = (
                    f"Please provide all information you know about the {ent_readable} based on its semantic relationships "
                    f"with the given neighbors {selected_readables}. "
                    f"Only output a concise description within 128 tokens."
                )

                ans = llm.qa(prompt)
                pf.write(f"{ent}\t{ans}\n")
                log_output(ans)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='FB15k-237-one Semantic Expansion')
    parser.add_argument('--LLMfold', type=str, required=True, help='Path to dataset folder')
    parser.add_argument('--LLMname', type=str, required=True, choices=['LLAMA2','ChatGLM2','GPT','DeepSeek'], help='Name of the LLM model')
    parser.add_argument('--cuda', type=str, default='0', help='CUDA device ID')
    parser.add_argument('--beta', type=float, default=0.5, help='Weight for similarity score')
    parser.add_argument('--max_length', type=int, default=256, help='Max generation length for LLM')
    parser.add_argument('--temperature', type=float, default=0.7, help='Sampling temperature for LLM')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda
    llm = {
        'LLAMA2': LLAMA2,
        'ChatGLM2': ChatGLM2,
        'GPT': GPT,
        'DeepSeek': DeepSeek
    }[args.LLMname](args)

    expansion_cot_fb(args, llm)
