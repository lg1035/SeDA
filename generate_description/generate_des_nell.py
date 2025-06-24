import os
import pandas as pd
import numpy as np
import torch
import argparse
import transformers
import openai
from utils import log_output, mkdir_path
from LLMModels import LLAMA2, ChatGLM2, GPT,DeepSeek


def load_entity_to_id_mapping(ent2ids_file_path):
    entity_to_id = {}
    with open(ent2ids_file_path, 'r', encoding='utf-8') as file:
        for line in file:
            entity, id_str = line.strip().split('\t')
            entity_to_id[entity] = int(id_str)
    return entity_to_id


def load_transe_embedding(embedding_file):
    embedding_matrix = []
    with open(embedding_file, 'r', encoding='utf-8') as f:
        for line in f:
            embedding = list(map(float, line.strip().split()))
            embedding_matrix.append(embedding)
    return np.array(embedding_matrix)


def get_entity_type(entity):
    return entity.split(':')[0] if ':' in entity else 'unknown'


def compute_relevance_score(entity, neighbors, transE_model, entity_freq, entity_to_id, beta=0.5, delta_w=1.0, topk=5):
    if entity not in entity_to_id:
        return []

    entity_id = entity_to_id[entity]
    type_weights = {}
    selected_neighbors = []
    all_scores = {}

    candidates = []
    for neighbor in neighbors:
        if neighbor not in entity_to_id:
            continue
        neighbor_id = entity_to_id[neighbor]
        freq_score = 1 / (entity_freq.get(neighbor, 1) + 1)
        sim_score = 1 / (np.linalg.norm(transE_model[entity_id] - transE_model[neighbor_id]) + 1e-6)
        neighbor_type = get_entity_type(neighbor)
        type_weight = type_weights.get(neighbor_type, 1.0)
        final_score = freq_score + beta * sim_score / type_weight
        candidates.append((neighbor, neighbor_type, final_score))

    for _ in range(min(topk, len(candidates))):
        if not candidates:
            break
        candidates.sort(key=lambda x: x[2], reverse=True)
        best_neighbor, best_type, best_score = candidates.pop(0)
        selected_neighbors.append(best_neighbor)
        all_scores[best_neighbor] = best_score
        type_weights[best_type] = type_weights.get(best_type, 1.0) + delta_w

        new_candidates = []
        for neighbor, neighbor_type, _ in candidates:
            type_weight = type_weights.get(neighbor_type, 1.0)
            neighbor_id = entity_to_id[neighbor]
            freq_score = 1 / (entity_freq.get(neighbor, 1) + 1)
            sim_score = 1 / (np.linalg.norm(transE_model[entity_id] - transE_model[neighbor_id]) + 1e-6)
            final_score = freq_score + beta * sim_score / type_weight
            new_candidates.append((neighbor, neighbor_type, final_score))
        candidates = new_candidates

    print(f'\nEntity: {entity}')
    for neighbor in selected_neighbors:
        print(f"  Neighbor: {neighbor}, Final Score: {all_scores[neighbor]:.4f}")

    return selected_neighbors


def expansion_cot_nell(args, llm, transE_model, entity_to_id):
    entity_use_file_path = f'{args.LLMfold}/processed_graph.tsv'
    entity_neighbors_df = pd.read_csv(entity_use_file_path, sep='\t', header=None, names=['entity', 'neighbors'])
    entity_freq = entity_neighbors_df['neighbors'].str.split(',').explode().value_counts().to_dict()

    topk_list = [5, 3]
    for topk in topk_list:
        save_path = f'{args.LLMfold}/mycotdes_top{topk}.txt'
        neighbors_save_path = f'{args.LLMfold}/selected_neighbors_top{topk}.tsv'

        with open(save_path, 'w', encoding='utf-8') as outfile, open(neighbors_save_path, 'w', encoding='utf-8') as neighbor_file:
            for _, row in entity_neighbors_df.iterrows():
                entity = row['entity']
                neighbors = row['neighbors'].split(',')
                selected_neighbors = compute_relevance_score(entity, neighbors, transE_model, entity_freq, entity_to_id, beta=0.5, delta_w=1.0, topk=topk)
                neighbors_str = ', '.join(selected_neighbors)

                neighbor_file.write(f'{entity}\t{neighbors_str}\n')

                prompt = (
                    f"Please provide all information you know about the {entity} based on its semantic relationships "
                    f"with the given neighbors {neighbors_str}. "
                    f"Only output a concise description within 128 tokens."
                )

                ans = llm.qa(prompt)
                outfile.write(f'{entity}\t{ans}\n')
                log_output(ans)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Enhanced Semantic Extraction')
    parser.add_argument('--LLMfold', type=str, required=True, help='The path of dataset')
    parser.add_argument('--LLMname', type=str, required=True, choices=['LLAMA2', 'ChatGLM2', 'GPT','DeepSeek'], help='The name of LLMs')
    parser.add_argument('--cuda', type=str, default='1', help='CUDA device ID (default: 0)')
    parser.add_argument('--ent2ids', type=str, default='ent2ids.txt', help='Entity to ID mapping file')
    parser.add_argument('--max_length', type=int, default=256, help='Maximum generation length for LLM')
    parser.add_argument('--temperature', type=float, default=0.7, help='Sampling temperature for LLM')

    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda

    llm = {'LLAMA2': LLAMA2, 'ChatGLM2': ChatGLM2, 'GPT': GPT,'DeepSeek':DeepSeek}[args.LLMname](args)
    transE_model = load_transe_embedding(f'{args.LLMfold}/entity2vec_TransE')
    entity_to_id = load_entity_to_id_mapping(f'{args.LLMfold}/{args.ent2ids}')

    expansion_cot_nell(args, llm, transE_model, entity_to_id)
