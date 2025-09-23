import os
import random
import math
import pickle
import torch
import time
from tqdm import tqdm
import copy
from transformers import BatchEncoding
from collections import defaultdict
import numpy as np
import json


class FKGCDataLoader(object):
    def __init__(self, in_paths, tokenizer, batch_size=2, max_desc_length=64,
                 model='bert', sp_num=5):
        """
        FKGC Data Loader
        Args:
            sp_num: support set size
        """
        self.datasetName = in_paths['dataset']
        self.batch_size = batch_size
        self.sp_num = sp_num

        self.train_set = self.load_dataset(in_paths['train'])
        self.valid_set = self.load_dataset(in_paths['valid'])
        self.test_set = self.load_dataset(in_paths['test'])

        self.load_entity_mapping_from_file(in_paths)
        
        self.load_relation_mapping_from_file(in_paths)
        
        self.tokenizer = tokenizer
        self.uid2text = {}
        self.uid2tokens = {}
        for p in in_paths['text']:
            self.load_text(p)

        self.build_fkgc_structures()

        self.structural_entity_embeddings = None
        self.transE_entity2id = None
        self.transE_loaded = False
        
        self.bert_entity_embeddings = None

        self.max_desc_length = max_desc_length
        self.model = model
        self.orig_vocab_size = len(tokenizer)
        self.neg_rate = 7

        self.n_ent = len(self.entity_set)
        self.n_rel = len(self.relation_set)


    def load_entity_mapping_from_file(self, in_paths):
        """Load entity mapping from file"""
        self.ent2id = {}
        self.id2ent = {}
        
        ent2ids_path = in_paths.get('ent2ids')
        if ent2ids_path and os.path.exists(ent2ids_path):
            print(f"Loading entity mapping from {ent2ids_path}")
            with open(ent2ids_path, 'r', encoding='utf-8') as f:
                for line in f:
                    ent, ent_id = line.strip().split('\t')
                    self.ent2id[ent] = int(ent_id)
                    self.id2ent[int(ent_id)] = ent
            print(f"Loaded {len(self.ent2id)} entities from {ent2ids_path}")
            self.entity_set = set(self.ent2id.keys())
            self.entity_list = sorted(self.entity_set)
        else:
            print("Entity mapping file not found, building from dataset...")
            self._build_entity_mapping_from_data()

        if 'fb15k237-one' in self.datasetName:
            ent2type_path = in_paths.get('ent2type')
            if ent2type_path and os.path.exists(ent2type_path):
                print(f"Loading entity type mapping from {ent2type_path}")
                self.entity_type_mapping = {}
                with open(ent2type_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        parts = line.strip().split('\t')
                        if len(parts) >= 2:
                            entity = parts[0]
                            types = parts[1:]
                            self.entity_type_mapping[entity] = types
                print(f"Loaded entity types for {len(self.entity_type_mapping)} entities")
            else:
                print("Entity type mapping file not found, using fallback method")
                self.entity_type_mapping = {}
        else:
            print(f"Dataset {self.datasetName} uses entity name format for type extraction")
            self.entity_type_mapping = {}

    def _build_entity_mapping_from_data(self):
        """Build entity mapping from training data"""
        self.entity_set = set([t[0] for t in (self.train_set + self.valid_set + self.test_set)] +
                              [t[-1] for t in (self.train_set + self.valid_set + self.test_set)])
        self.entity_list = sorted(self.entity_set)
        self.ent2id = {e: i for i, e in enumerate(self.entity_list)}
        self.id2ent = {i: e for i, e in enumerate(self.entity_list)}
        print(f"Built entity mapping from training data: {len(self.ent2id)} entities")

    def load_relation_mapping_from_file(self, in_paths):
        """Load background relation mapping from rel2ids.txt file"""
        rel2id_file = None
        if 'nell' in self.datasetName:
            rel2id_file = './data/nell/rel2ids.txt'
        elif 'fb15k237' in self.datasetName or 'fb15k237-one' in self.datasetName:
            rel2id_file = './data/fb15k237-one/rel2ids.txt'

        if rel2id_file and os.path.exists(rel2id_file):
            print(f"Loading background relation mapping from {rel2id_file}")
            self.rel2id = {}
            self.id2rel = {}

            try:
                with open(rel2id_file, 'r', encoding='utf8') as f:
                    for line_num, line in enumerate(f, 1):
                        line = line.strip()
                        if not line:
                            continue
                        parts = line.split('\t')
                        if len(parts) >= 2:
                            relation_name = parts[0]
                            try:
                                relation_id = int(parts[1])
                                self.rel2id[relation_name] = relation_id
                                self.id2rel[relation_id] = relation_name
                            except ValueError:
                                print(f"Warning: Invalid relation ID at line {line_num}: {line}")
                        else:
                            print(f"Warning: Invalid format at line {line_num}: {line}")

                if len(self.rel2id) == 0:
                    print(f"Warning: No background relations loaded from {rel2id_file}")
                    print("Falling back to building relation mapping from training data")
                    self._build_relation_mapping_from_data()
                else:
                    self.relation_set = set(self.rel2id.keys())
                    self.relation_list = sorted(self.relation_set)
                    print(f"Loaded {len(self.rel2id)} background relations from {rel2id_file}")
                    print("Note: Task relations (from train/valid/test sets) are not included in this mapping")
                    #print("Sample background relations:")
                    #for i, (relation_name, relation_id) in enumerate(list(self.rel2id.items())[:5]):
                        #print(f"  {relation_name}: {relation_id}")

            except Exception as e:
                print(f"Error loading background relation mapping from file: {e}")
                print("Falling back to building relation mapping from training data")
                self._build_relation_mapping_from_data()
        else:
            print(f"Background relation mapping file not found: {rel2id_file}")
            print("Building relation mapping from training data")
            self._build_relation_mapping_from_data()

    def _build_relation_mapping_from_data(self):
        """Build relation mapping from training data (fallback method)"""
        print("Building relation mapping from training data (this includes task relations)...")
        self.relation_set = set([t[1] for t in (self.train_set + self.valid_set + self.test_set)])
        self.relation_list = sorted(self.relation_set)
        self.rel2id = {r: i for i, r in enumerate(self.relation_list)}
        self.id2rel = {i: r for i, r in enumerate(self.relation_list)}
        print(f"Built relation mapping from training data: {len(self.rel2id)} relations (including task relations)")

    def build_fkgc_structures(self):
        """Build data structures required for FKGC"""
        self.rel2train_triples = defaultdict(list)
        for triple in self.train_set:
            h, r, t = triple
            self.rel2train_triples[r].append(triple)

        self.e1rel_e2 = self.load_e1rel_e2_from_json()

        self.rel2candidates = self.load_candidates_from_json()

        self.train_test_path_id = self.build_path_info()

        self.train_tasks = self.rel2train_triples
        self.valid_tasks = self.build_tasks(self.valid_set)
        self.test_tasks = self.build_tasks(self.test_set)

        self.classify_relations()

    def set_transE_structural_embeddings(self, embeddings, entity2id):
        """Set TransE structural embeddings (called by main)"""
        self.structural_entity_embeddings = embeddings
        self.transE_entity2id = entity2id
        self.transE_loaded = True
        print(f"TransE structural embeddings set: {len(entity2id)} entities, dim {embeddings.shape[1]}")
    
    def set_bert_entity_embeddings(self, embeddings):
        """Set BERT entity embeddings (for negative sampling and GAT)"""
        self.bert_entity_embeddings = embeddings
        print(f"BERT entity embeddings set: {embeddings.shape[0]} entities, dim {embeddings.shape[1]}")

    def load_e1rel_e2_from_json(self):
        """Load e1rel_e2 mapping from e1rel_e2.json file"""
        e1rel_e2_file = None
        if 'nell' in self.datasetName:
            e1rel_e2_file = './data/nell/e1rel_e2.json'
        elif 'fb15k237' in self.datasetName or 'fb15k237-one' in self.datasetName:
            e1rel_e2_file = './data/fb15k237-one/e1rel_e2.json'

        if not e1rel_e2_file:
            raise ValueError(f"Unknown dataset: {self.datasetName}")

        print(f"Loading e1rel_e2 from {e1rel_e2_file}")
        
        if not os.path.exists(e1rel_e2_file):
            raise FileNotFoundError(f"e1rel_e2.json file not found: {e1rel_e2_file}")
        
        try:
            with open(e1rel_e2_file, 'r', encoding='utf8') as f:
                e1rel_e2 = json.load(f)
            
            valid_e1rel_e2 = defaultdict(list)
            for key, tails in e1rel_e2.items():
                valid_tails = []
                for tail in tails:
                    if tail in self.ent2id:
                        valid_tails.append(tail)
                
                if len(valid_tails) > 0:
                    valid_e1rel_e2[key] = valid_tails
            
            print(f"Loaded e1rel_e2 with {len(valid_e1rel_e2)} head-relation pairs")
            return valid_e1rel_e2
            
        except Exception as e:
            raise RuntimeError(f"Error loading e1rel_e2 from JSON: {e}")


    def load_candidates_from_json(self):
        """Load candidate entities from rel2cands.json file"""
        rel2cands_file = None
        if 'nell' in self.datasetName:
            rel2cands_file = './data/nell/rel2cands.json'
        elif 'fb15k237' in self.datasetName or 'fb15k237-one' in self.datasetName:
            rel2cands_file = './data/fb15k237-one/rel2cands.json'

        if rel2cands_file and os.path.exists(rel2cands_file):
            print(f"Loading candidates from {rel2cands_file}")
            try:
                with open(rel2cands_file, 'r', encoding='utf8') as f:
                    rel2candidates = json.load(f)
                
                valid_candidates = {}
                for rel, candidates in rel2candidates.items():
                    valid_cands = []
                    for candidate in candidates:
                        if candidate in self.ent2id:
                            valid_cands.append(candidate)
                    
                    if len(valid_cands) > 0:
                        valid_candidates[rel] = valid_cands
                
                print(f"Loaded candidates for {len(valid_candidates)} relations")
                return valid_candidates
                
            except Exception as e:
                print(f"Error loading candidates from JSON: {e}")
                print("Falling back to building candidates from dataset")
                return self._build_candidates_from_dataset()
        else:
            print(f"Candidates file not found: {rel2cands_file}")
            print("Building candidates from dataset")
            return self._build_candidates_from_dataset()

    def _build_candidates_from_dataset(self):
        """Build candidate entities from dataset (fallback method)"""
        print("Building candidates from dataset...")
        rel2candidates = {}
        for rel in self.relation_set:
            candidates = set()
            for triple in self.train_set + self.valid_set + self.test_set:
                if triple[1] == rel:
                    candidates.add(triple[0])
                    candidates.add(triple[2])
            rel2candidates[rel] = list(candidates)
        print(f"Built candidates for {len(rel2candidates)} relations from dataset")
        return rel2candidates


    def build_tasks(self, dataset):
        """Group dataset by relations to build tasks"""
        tasks = defaultdict(list)
        for triple in dataset:
            h, r, t = triple
            tasks[r].append(triple)
        return tasks

    def build_path_info(self):
        """Build path information - read background relation paths from path_graph file"""
        path_info = {}

        # Try to read path_graph file
        path_graph_file = None
        if 'nell' in self.datasetName:
            path_graph_file = './data/nell/path_graph.txt'
        elif 'fb15k237' in self.datasetName or 'fb15k237-one' in self.datasetName:
            path_graph_file = './data/fb15k237-one/path_graph.txt'

        if path_graph_file and os.path.exists(path_graph_file):
            print(f"Loading background relation paths from {path_graph_file}")
            try:
                line_count = 0
                valid_count = 0
                format_mismatch_count = 0

                # First build entity pair to path mapping
                pair_to_paths = {}

                with open(path_graph_file, 'r', encoding='utf8') as f:
                    for line in f:
                        line_count += 1
                        parts = line.strip().split('\t')
                        if len(parts) >= 3:
                            h, r, t = parts[0], parts[1], parts[2]

                            # Handle concept: prefix - remove all concept: prefixes
                            h_clean = h.replace('concept:', '')
                            r_clean = r.replace('concept:', '')
                            t_clean = t.replace('concept:', '')

                            # Check if entities and background relations are in mapping
                            # Note: relations in path_graph should be background relations, should be in rel2id
                            if h_clean in self.ent2id and t_clean in self.ent2id and r_clean in self.rel2id:
                                # Use entity pair as key
                                pair = (h_clean, t_clean)
                                if pair not in pair_to_paths:
                                    pair_to_paths[pair] = []
                                pair_to_paths[pair].append([r_clean])
                                valid_count += 1
                            else:
                                format_mismatch_count += 1
                                if line_count <= 3:
                                    print(f"  Line {line_count} mismatch:")
                                    print(f"    h_clean in ent2id: {h_clean in self.ent2id}")
                                    print(f"    t_clean in ent2id: {t_clean in self.ent2id}")
                                    print(f"    r_clean in rel2id: {r_clean in self.rel2id}")
                                    if r_clean not in self.rel2id:
                                        print(f"    Background relation '{r_clean}' not found in rel2id")

                for pair, paths in pair_to_paths.items():
                    h_id = self.ent2id[pair[0]]
                    t_id = self.ent2id[pair[1]]
                    paths_id = []
                    for path in paths:
                        path_id = [self.rel2id[rel] for rel in path]
                        paths_id.append(path_id)
                    path_info[(h_id, t_id)] = paths_id

                print(f"Loaded background relation paths for {len(path_info)} entity pairs")
                print(f"  - Total lines: {line_count}")
                print(f"  - Valid background triples: {valid_count}")
                print(f"  - Format mismatches: {format_mismatch_count}")
                print(f"  - Entity pairs with background paths: {len(path_info)}")

            except Exception as e:
                print(f"Error loading background relation paths: {e}")
                print("Using simplified path information")
                return self._build_simple_path_info()
        else:
            print(f"Background relation path file not found: {path_graph_file}")
            print("Using simplified path information")
            return self._build_simple_path_info()

        return path_info

    def _build_simple_path_info(self):
        """Build simplified path information from training set triples"""
        path_info = {}
        print("Building simplified path information from training set...")

        for triple in self.train_set:
            h, r, t = triple
            if h in self.ent2id and t in self.ent2id and r in self.rel2id:
                pair = (self.ent2id[h], self.ent2id[t])
                if pair not in path_info:
                    path_info[pair] = []
                path_info[pair].append([self.rel2id[r]])

        print("Building 2-hop paths...")
        entity_to_relations = {}
        for triple in self.train_set:
            h, r, t = triple
            if h in self.ent2id and t in self.ent2id and r in self.rel2id:
                h_id, t_id, r_id = self.ent2id[h], self.ent2id[t], self.rel2id[r]

                if h_id not in entity_to_relations:
                    entity_to_relations[h_id] = []
                entity_to_relations[h_id].append((r_id, t_id))

        two_hop_count = 0
        for h_id in entity_to_relations:
            for r1_id, mid_id in entity_to_relations[h_id]:
                if mid_id in entity_to_relations:
                    for r2_id, t_id in entity_to_relations[mid_id]:
                        if h_id != t_id:
                            pair = (h_id, t_id)
                            if pair not in path_info:
                                path_info[pair] = []
                            path_info[pair].append([r1_id, r2_id])
                            two_hop_count += 1
                            if two_hop_count >= 10000:
                                break
            if two_hop_count >= 10000:
                break

        print(f"Built simplified path information:")
        print(f"  - Direct paths: {len([p for p in path_info.values() if len(p[0]) == 1])}")
        print(f"  - 2-hop paths: {two_hop_count}")
        print(f"  - Total entity pairs: {len(path_info)}")

        return path_info

    def load_dataset(self, in_path):
        """Load dataset"""
        dataset = []
        with open(in_path, 'r', encoding='utf8') as fil:
            for line in fil.readlines():
                if in_path[-3:] == 'txt':
                    h, t, r = line.strip('\n').split('\t')
                else:
                    h, r, t = line.strip('\n').split('\t')
                dataset.append((h, r, t))
        return dataset

    def load_text(self, in_path):
        """Load text descriptions"""
        uid2text = self.uid2text
        uid2tokens = self.uid2tokens
        tokenizer = self.tokenizer

        with open(in_path, 'r', encoding='utf8') as fil:
            print(f"Loading text from: {in_path}")
            for line in fil.readlines():
                uid, text = line.strip('\n').split('\t', 1)
                text = text.replace('@en', '').strip('"')

                if uid not in uid2text:
                    uid2text[uid] = text

                tokens = tokenizer.tokenize(text)
                if uid not in uid2tokens:
                    uid2tokens[uid] = tokens

        self.uid2text = uid2text
        self.uid2tokens = uid2tokens

    def train_generator(self):
        """Training data generator"""
        task_pool = list(self.train_tasks.keys())
        num_tasks = len(task_pool)
        rel_idx = 0

        while True:
            if rel_idx % num_tasks == 0:
                random.shuffle(task_pool)

            query_rel = task_pool[rel_idx % num_tasks]
            rel_idx += 1

            candidates = self.rel2candidates.get(query_rel, [])
            candidates_id = [self.ent2id[c] for c in candidates if c in self.ent2id]

            if len(candidates_id) <= 20:
                continue

            rel_triples = self.train_tasks[query_rel]
            random.shuffle(rel_triples)

            train_tri_id = [[self.ent2id[triple[0]], self.rel2id[triple[1]], self.ent2id[triple[2]]]
                            for triple in rel_triples]

            train_tri_id_fil = []
            for trip in train_tri_id:
                if trip[0] != trip[2]:
                    train_tri_id_fil.append(trip)
            train_tri_id = train_tri_id_fil

            if len(train_tri_id) < self.sp_num + self.batch_size:
                continue

            support_pair = train_tri_id[:self.sp_num]
            query_pair = train_tri_id[self.sp_num:]

            if len(support_pair) == 0 or len(query_pair) == 0:
                continue

            if len(query_pair) < self.batch_size:
                query_pair_pos = [random.choice(query_pair) for _ in range(self.batch_size)]
            else:
                query_pair_pos = random.sample(query_pair, self.batch_size)

            support_pair = [[pair[0], pair[2]] for pair in support_pair]
            query_pair_pos = [[pair[0], pair[2]] for pair in query_pair_pos]

            one_tomany_train = []
            for i in range(len(query_pair_pos)):
                key = self.id2ent[int(query_pair_pos[i][0])] + query_rel
                one2many = self.e1rel_e2.get(key, [])
                one2many2id = [self.ent2id[_] for _ in one2many if _ in self.ent2id]
                one_tomany_train.append(one2many2id)

            yield support_pair, query_pair_pos, one_tomany_train, candidates_id, query_rel

    def valid_generator(self):
        """Validation data generator"""
        task_pool = list(self.valid_tasks.keys())
        num_tasks = len(task_pool)
        rel_idx = 0

        while True:
            if rel_idx % num_tasks == 0:
                random.shuffle(task_pool)

            query_rel = task_pool[rel_idx % num_tasks]
            rel_idx += 1

            candidates = self.rel2candidates.get(query_rel, [])
            candidates_id = [self.ent2id[c] for c in candidates if c in self.ent2id]

            if len(candidates_id) <= 20:
                continue

            rel_triples = self.valid_tasks[query_rel]
            random.shuffle(rel_triples)

            valid_tri_id = [[self.ent2id[triple[0]], self.rel2id[triple[1]], self.ent2id[triple[2]]]
                            for triple in rel_triples]

            valid_tri_id_fil = []
            for trip in valid_tri_id:
                if trip[0] != trip[2]:
                    valid_tri_id_fil.append(trip)
            valid_tri_id = valid_tri_id_fil

            if len(valid_tri_id) < self.sp_num + self.batch_size:
                continue

            support_pair = valid_tri_id[:self.sp_num]
            query_pair = valid_tri_id[self.sp_num:]

            if len(support_pair) == 0 or len(query_pair) == 0:
                continue

            if len(query_pair) < self.batch_size:
                query_pair_pos = [random.choice(query_pair) for _ in range(self.batch_size)]
            else:
                query_pair_pos = random.sample(query_pair, self.batch_size)

            support_pair = [[pair[0], pair[2]] for pair in support_pair]
            query_pair_pos = [[pair[0], pair[2]] for pair in query_pair_pos]

            one_tomany_valid = []
            for i in range(len(query_pair_pos)):
                key = self.id2ent[int(query_pair_pos[i][0])] + query_rel
                one2many = self.e1rel_e2.get(key, [])
                one2many2id = [self.ent2id[_] for _ in one2many if _ in self.ent2id]
                one_tomany_valid.append(one2many2id)

            yield support_pair, query_pair_pos, one_tomany_valid, candidates_id, query_rel

    def test_generator(self):
        """Test data generator - improved version"""
        task_pool = list(self.test_tasks.keys())
        num_tasks = len(task_pool)
        rel_idx = 0

        while True:
            if rel_idx % num_tasks == 0:
                random.shuffle(task_pool)

            query_rel = task_pool[rel_idx % num_tasks]
            rel_idx += 1

            candidates = self.rel2candidates.get(query_rel, [])
            candidates_id = [self.ent2id[c] for c in candidates if c in self.ent2id]

            if len(candidates_id) <= 20:
                continue

            rel_triples = self.test_tasks[query_rel]
            random.shuffle(rel_triples)

            test_tri_id = [[self.ent2id[triple[0]], self.rel2id[triple[1]], self.ent2id[triple[2]]]
                           for triple in rel_triples]

            test_tri_id_fil = []
            for trip in test_tri_id:
                if trip[0] != trip[2]:
                    test_tri_id_fil.append(trip)
            test_tri_id = test_tri_id_fil

            if len(test_tri_id) < self.sp_num + self.batch_size:
                continue

            support_pair = test_tri_id[:self.sp_num]
            query_pair = test_tri_id[self.sp_num:]

            if len(support_pair) == 0 or len(query_pair) == 0:
                continue

            if len(query_pair) < self.batch_size:
                query_pair_pos = [random.choice(query_pair) for _ in range(self.batch_size)]
            else:
                query_pair_pos = random.sample(query_pair, self.batch_size)

            support_pair = [[pair[0], pair[2]] for pair in support_pair]
            query_pair_pos = [[pair[0], pair[2]] for pair in query_pair_pos]

            one_tomany_test = []
            for i in range(len(query_pair_pos)):
                key = self.id2ent[int(query_pair_pos[i][0])] + query_rel
                one2many = self.e1rel_e2.get(key, [])
                one2many2id = [self.ent2id[_] for _ in one2many if _ in self.ent2id]
                one_tomany_test.append(one2many2id)

            yield support_pair, query_pair_pos, one_tomany_test, candidates_id, query_rel

    def get_batch_data(self, support_pair, query_pair_pos, one_tomany_train, candidates_id):
        """Convert batch data to model input format"""
        support_texts = []
        support_tokens = []
        for pair in support_pair:
            h_id, t_id = pair
            h, t = self.id2ent[h_id], self.id2ent[t_id]
            text, tokens = self.triple_to_text((h, "relation", t), {'h': True, 'r': False, 't': True})
            support_texts.append(text)
            support_tokens.append(tokens)

        query_texts = []
        query_tokens = []
        for pair in query_pair_pos:
            h_id, t_id = pair
            h, t = self.id2ent[h_id], self.id2ent[t_id]
            text, tokens = self.triple_to_text((h, "relation", t), {'h': True, 'r': False, 't': True})
            query_texts.append(text)
            query_tokens.append(tokens)

        support_batch = self.my_tokenize(support_tokens, max_length=512, padding=True, model=self.model)
        query_batch = self.my_tokenize(query_tokens, max_length=512, padding=True, model=self.model)

        candidate_texts = []
        candidate_tokens = []
        for c_id in candidates_id:
            c = self.id2ent[c_id]
            text, tokens = self.element_to_text(c)
            candidate_texts.append(text)
            candidate_tokens.append(tokens)

        candidate_batch = self.my_tokenize(candidate_tokens, max_length=128, padding=True, model=self.model)

        return {
            'support': support_batch,
            'query': query_batch,
            'candidates': candidate_batch,
            'candidates_id': candidates_id,
            'one_tomany': one_tomany_train
        }

    def triple_to_text(self, triple, with_text):
        """Convert triple to text"""
        tokenizer = self.tokenizer
        ent2id = self.ent2id
        rel2id = self.rel2id

        h_n_tokens = min(228, self.max_desc_length)
        t_n_tokens = min(228, self.max_desc_length)
        r_n_tokens = min(51, self.max_desc_length)

        h, r, t = triple

        h_text_tokens = self.uid2tokens.get(h, [])[:h_n_tokens] if with_text['h'] else []
        r_text_tokens = self.uid2tokens.get(r, [])[:r_n_tokens] if with_text['r'] else []
        t_text_tokens = self.uid2tokens.get(t, [])[:t_n_tokens] if with_text['t'] else []

        h_token = [self.tokenizer.cls_token] if with_text['h'] else [tokenizer.mask_token]
        r_token = [self.tokenizer.cls_token] if with_text['r'] else [tokenizer.mask_token]
        t_token = [self.tokenizer.cls_token] if with_text['t'] else [tokenizer.mask_token]

        tokens = h_token + h_text_tokens + r_token + r_text_tokens + t_token + t_text_tokens
        text = tokenizer.convert_tokens_to_string(tokens)

        return text, tokens

    def element_to_text(self, target):
        """Convert single element to text"""
        tokenizer = self.tokenizer
        ent2id = self.ent2id
        rel2id = self.rel2id

        n_tokens = min(508, self.max_desc_length)
        text_tokens = self.uid2tokens.get(target, [])[:n_tokens]

        token = [self.tokenizer.cls_token]

        tokens = token + text_tokens
        text = tokenizer.convert_tokens_to_string(tokens)

        return text, tokens

    def my_tokenize(self, batch_tokens, max_length=1024, padding=True, model='roberta'):
        """Tokenize processing"""
        if model == 'roberta':
            start_tokens = [self.tokenizer.cls_token]
            end_tokens = [self.tokenizer.sep_token]
        elif model == 'bert':
            start_tokens = [self.tokenizer.cls_token]
            end_tokens = [self.tokenizer.sep_token]
        else:
            raise ValueError(f"Unsupported model type: {model}")

        batch_tokens = [start_tokens + tokens + end_tokens for tokens in batch_tokens]
        batch_size = len(batch_tokens)
        longest = min(max([len(tokens) for tokens in batch_tokens]), max_length)

        if model == 'bert':
            input_ids = torch.zeros((batch_size, longest)).long()
        elif model == 'roberta':
            input_ids = torch.ones((batch_size, longest)).long()

        token_type_ids = torch.zeros((batch_size, longest)).long()
        attention_mask = torch.zeros((batch_size, longest)).long()

        for i in range(batch_size):
            tokens = self.tokenizer.convert_tokens_to_ids(batch_tokens[i])
            input_ids[i, :len(tokens)] = torch.tensor(tokens).long()
            attention_mask[i, :len(tokens)] = 1

        if model in ['roberta']:
            return BatchEncoding(data={'input_ids': input_ids, 'attention_mask': attention_mask})
        elif model in ['bert']:
            return BatchEncoding(
                data={'input_ids': input_ids, 'attention_mask': attention_mask, 'token_type_ids': token_type_ids})


    def get_relation_tasks(self, split='train'):
        """Get relation tasks"""
        if split == 'train':
            return self.train_tasks
        elif split == 'valid':
            return self.valid_tasks
        elif split == 'test':
            return self.test_tasks
        else:
            raise ValueError(f"Unknown split: {split}")

    def get_candidates_for_relation(self, relation):
        return self.rel2candidates.get(relation, [])

    def get_path_info(self, head_id, tail_id):
        pair = (head_id, tail_id)
        return self.train_test_path_id.get(pair, [])

    def get_path_info_by_entity_names(self, head_name, tail_name):
        if head_name in self.ent2id and tail_name in self.ent2id:
            head_id = self.ent2id[head_name]
            tail_id = self.ent2id[tail_name]
            return self.get_path_info(head_id, tail_id)
        return []

    def generate_negative_samples(self, support_pair, query_pair_pos, relation, num_negatives=7):
        """Generate negative samples for FKGC task - multiple negative sampling strategies"""
        neg_samples = []

        if not hasattr(self, 'exclusive_relations'):
            self.classify_relations()

        for pair in query_pair_pos:
            h_id, t_id = pair
            h, t = self.id2ent[h_id], self.id2ent[t_id]

            for _ in range(num_negatives):
                neg_sample = None

                if self.is_exclusive_relation(relation):
                    neg_sample = self.type_aware_replacement(h, t, relation)
                elif self.is_inclusive_relation(relation):
                    neg_sample = self.semantic_contrastive_sampling(h, t, relation)

                if neg_sample is None:
                    neg_sample = self.random_replacement(h, t, relation)

                if neg_sample:
                    neg_samples.append(neg_sample)

        return neg_samples

    def random_replacement(self, h, t, relation):
        """Random replacement negative sample"""
        candidates = list(self.entity_set - {h, t})
        if not candidates:
            return None

        max_attempts = 50
        for attempt in range(max_attempts):
            
            replace_ent = random.choice(candidates)
            neg_triple = (h, relation, replace_ent)

            if neg_triple not in self.train_set and neg_triple not in self.valid_set and neg_triple not in self.test_set:
                return neg_triple
        
        return None

    def type_aware_replacement(self, h, t, relation):
        """Type-aware negative sample replacement"""
        if not self.is_exclusive_relation(relation):
            return None
        
        t_types = self.get_entity_types(t)

        real_triples = set()
        for triple in self.train_set + self.valid_set + self.test_set:
            if triple[0] == h and triple[1] == relation:
                real_triples.add(triple[2])

        type_similar_candidates = []
        count = 0
        for ent in self.entity_set:
            if count >= 1000:
                break
            if ent != t and ent not in real_triples:
                type_sim = self.compute_type_similarity(t, ent)
                if type_sim > 0.0:
                    type_similar_candidates.append((ent, type_sim))
                    if len(type_similar_candidates) >= 50:
                        break
            count += 1

        if not type_similar_candidates:
            return None

        type_similar_candidates.sort(key=lambda x: x[1], reverse=True)
        
        top_candidates = type_similar_candidates[:10]
        selected_ent, similarity = random.choice(top_candidates)
        
        return (h, relation, selected_ent)

    def relation_constrained_replacement(self, h, t, relation):
        """Relation-constrained negative sample replacement"""
        same_rel_entities = set()
        for triple in self.train_set:
            if triple[1] == relation:
                same_rel_entities.add(triple[0])
                same_rel_entities.add(triple[2])

        candidates = list(same_rel_entities - {h, t})
        if not candidates:
            return None

        replace_ent = random.choice(candidates)
        if random.random() < 0.5:
            neg_triple = (replace_ent, relation, t)
        else:
            neg_triple = (h, relation, replace_ent)

        if neg_triple not in self.train_set and neg_triple not in self.valid_set and neg_triple not in self.test_set:
            return neg_triple
        return None

    def extract_entity_type(self, entity):
        """Extract entity type"""
        if hasattr(self, 'entity_type_mapping') and entity in self.entity_type_mapping:
            return self.entity_type_mapping[entity][0]
        
        if 'nell' in self.datasetName:
            return entity.split(':')[0] if ':' in entity else 'unknown'

        return 'unknown'
    
    def get_entity_types(self, entity):
        if hasattr(self, 'entity_type_mapping') and entity in self.entity_type_mapping:
            return self.entity_type_mapping[entity]
        
        if 'nell' in self.datasetName:
            if ':' in entity:
                return [entity.split(':')[0]]
            else:
                return ['unknown']
        
        return ['unknown']
    
    def compute_type_similarity(self, entity1, entity2):
        """Compute type similarity between two entities"""
        types1 = set(self.get_entity_types(entity1))
        types2 = set(self.get_entity_types(entity2))
        
        if not types1 or not types2:
            return 0.0

        intersection = len(types1.intersection(types2))
        union = len(types1.union(types2))
        
        if union == 0:
            return 0.0
        
        return intersection / union

    def get_batch_with_negatives(self, support_pair, query_pair_pos, relation):
        """Get batch data with negative samples"""
        neg_samples = self.generate_negative_samples(support_pair, query_pair_pos, relation)

        pos_neg_pairs = []
        for i, pos_pair in enumerate(query_pair_pos):
            if i < len(neg_samples):
                pos_neg_pairs.append((pos_pair, neg_samples[i]))

        batch_data = self.convert_to_model_input(support_pair, pos_neg_pairs)

        return batch_data

    def convert_to_model_input(self, support_pair, pos_neg_pairs):
        """Convert to model input format"""
        support_texts = []
        pos_texts = []
        neg_texts = []

        for pair in support_pair:
            h_id, t_id = pair
            h, t = self.id2ent[h_id], self.id2ent[t_id]
            text, _ = self.triple_to_text((h, "relation", t), {'h': True, 'r': False, 't': True})
            support_texts.append(text)

        for pos_pair, neg_pair in pos_neg_pairs:
            h_id, t_id = pos_pair
            h, t = self.id2ent[h_id], self.id2ent[t_id]
            text, _ = self.triple_to_text((h, "relation", t), {'h': True, 'r': False, 't': True})
            pos_texts.append(text)

            h_neg, r_neg, t_neg = neg_pair
            text, _ = self.triple_to_text((h_neg, r_neg, t_neg), {'h': True, 'r': True, 't': True})
            neg_texts.append(text)

        support_batch = self.my_tokenize([self.tokenizer.tokenize(text) for text in support_texts],
                                         max_length=512, padding=True, model=self.model)
        pos_batch = self.my_tokenize([self.tokenizer.tokenize(text) for text in pos_texts],
                                     max_length=512, padding=True, model=self.model)
        neg_batch = self.my_tokenize([self.tokenizer.tokenize(text) for text in neg_texts],
                                     max_length=512, padding=True, model=self.model)

        return {
            'support': support_batch,
            'pos': pos_batch,
            'neg': neg_batch
        }

    def classify_relations(self):
        """Classify task relations into exclusive and inclusive types (only for relations in train/valid/test sets)"""
        self.exclusive_relations = set()
        self.inclusive_relations = set()

        task_relations = set()
        for triple in self.train_set + self.valid_set + self.test_set:
            task_relations.add(triple[1])

        print(f"Classifying {len(task_relations)} task relations...")

        for relation in task_relations:
            head_to_tails = {}
            
            all_triples = self.train_set + self.valid_set + self.test_set
            for triple in all_triples:
                if triple[1] == relation:
                    h, t = triple[0], triple[2]
                    if h not in head_to_tails:
                        head_to_tails[h] = set()
                    head_to_tails[h].add(t)

            if head_to_tails:
                avg_tails_per_head = sum(len(tails) for tails in head_to_tails.values()) / len(head_to_tails)

                if avg_tails_per_head < 2.0:
                    self.exclusive_relations.add(relation)
                else:
                    self.inclusive_relations.add(relation)
            else:
                self.inclusive_relations.add(relation)

        print(f"Exclusive task relations: {len(self.exclusive_relations)}")
        print(f"Inclusive task relations: {len(self.inclusive_relations)}")
        
        exclusive_examples = list(self.exclusive_relations)[:5]
        inclusive_examples = list(self.inclusive_relations)[:5]
        print(f"Exclusive task relation examples: {exclusive_examples}")
        print(f"Inclusive task relation examples: {inclusive_examples}")
        
        print("\n=== Task Relation Classification Results ===")
        print(f"{'Relation':<50} {'Type':<12} {'Train':<6} {'Valid':<6} {'Test':<6} {'Total':<6} {'Avg_Tails':<10}")
        print("-" * 100)
        
        task_relations = set()
        for triple in self.train_set + self.valid_set + self.test_set:
            task_relations.add(triple[1])
        
        sorted_task_relations = sorted(task_relations)
        
        print(f"{'Relation':<50} {'Type':<12} {'Train':<6} {'Valid':<6} {'Test':<6} {'Total':<6} {'Avg Tails':<10}")
        print("-" * 100)
        
        for relation in sorted_task_relations:
            train_count = sum(1 for triple in self.train_set if triple[1] == relation)
            valid_count = sum(1 for triple in self.valid_set if triple[1] == relation)
            test_count = sum(1 for triple in self.test_set if triple[1] == relation)
            total_count = train_count + valid_count + test_count
            
            head_to_tails = {}
            all_triples = self.train_set + self.valid_set + self.test_set
            for triple in all_triples:
                if triple[1] == relation:
                    h, t = triple[0], triple[2]
                    if h not in head_to_tails:
                        head_to_tails[h] = set()
                    head_to_tails[h].add(t)
            
            avg_tails = sum(len(tails) for tails in head_to_tails.values()) / len(head_to_tails) if head_to_tails else 0
            rel_type = "Exclusive" if relation in self.exclusive_relations else "Inclusive"
            
            print(f"{relation:<50} {rel_type:<12} {train_count:<6} {valid_count:<6} {test_count:<6} {total_count:<6} {avg_tails:<10.2f}")
        
        print("-" * 100)
        print(f"Total Task Relations: {len(task_relations)}")
        print(f"Exclusive Task Relations: {len(self.exclusive_relations)} ({len(self.exclusive_relations)/len(task_relations)*100:.1f}%)")
        print(f"Inclusive Task Relations: {len(self.inclusive_relations)} ({len(self.inclusive_relations)/len(task_relations)*100:.1f}%)")
        print("=" * 100)
        
        self.neg_sample_size = getattr(self, 'neg_sample_size', 7)

    def is_exclusive_relation(self, relation):
        return relation in self.exclusive_relations

    def is_inclusive_relation(self, relation):
        return relation in self.inclusive_relations

    def semantic_contrastive_sampling(self, h, t, relation):
        if not self.is_inclusive_relation(relation):
            return None
        
        real_triples = set()
        for triple in self.train_set + self.valid_set + self.test_set:
            if triple[0] == h and triple[1] == relation:
                real_triples.add(triple[2])
        
        candidates = []
        for ent in self.entity_set:
            if ent != t and ent not in real_triples:
                candidates.append(ent)
                if len(candidates) >= 200:
                    break

        if not candidates:
            return None

        if len(candidates) > 100:
            candidates = random.sample(candidates, 100)

        if len(candidates) >= 10:
            selected_candidate = random.choice(candidates)
            return (h, relation, selected_candidate)

        candidate_scores = []
        for i, candidate in enumerate(candidates[:20]):
            try:
                entity_sim = self.compute_entity_similarity(t, candidate)
                candidate_scores.append((candidate, entity_sim))
            except Exception as e:
                continue

        if not candidate_scores:
            selected_candidate = random.choice(candidates)
            return (h, relation, selected_candidate)

        best_candidate = max(candidate_scores, key=lambda x: x[1])[0]
        best_score = max(candidate_scores, key=lambda x: x[1])[1]
        return (h, relation, best_candidate)

    def compute_entity_similarity(self, ent1, ent2):
        if ent1 == ent2:
            return 1.0

        if ent1 not in self.ent2id or ent2 not in self.ent2id:
            return 0.5

        ent1_id = self.ent2id[ent1]
        ent2_id = self.ent2id[ent2]

        if hasattr(self, 'bert_entity_embeddings') and self.bert_entity_embeddings is not None:
            emb1 = self.bert_entity_embeddings[ent1_id]
            emb2 = self.bert_entity_embeddings[ent2_id]
            
            similarity = torch.cosine_similarity(emb1.unsqueeze(0), emb2.unsqueeze(0), dim=1).item()
            return similarity
        else:
            return random.uniform(0.3, 0.7)


    def load_transE_embeddings(self, ent2ids_file_path=None, embedding_file_path=None):
        if ent2ids_file_path is None:
            ent2id_path = './data/entity2id.txt'
        else:
            ent2id_path = ent2ids_file_path

        if embedding_file_path is None:
            embedding_path = './data/entity2vec.TransE'
        else:
            embedding_path = embedding_file_path

        self.entity_to_id = self.load_entity_to_id_mapping(ent2id_path)

        self.transE_embeddings = self.load_transe_embedding(embedding_path)

        print(f"Successfully loaded TransE embeddings: {len(self.transE_embeddings)} entities")
        print(f"Entity mapping: {len(self.entity_to_id)} entities")

    def load_entity_to_id_mapping(self, ent2ids_file_path):
        """Load entity to ID mapping"""
        entity_to_id = {}
        with open(ent2ids_file_path, 'r', encoding='utf-8') as file:
            for line in file:
                entity, id_str = line.strip().split('\t')
                entity_to_id[entity] = int(id_str)
        return entity_to_id

    def load_transe_embedding(self, embedding_file):
        """Load TransE embeddings"""
        embedding_matrix = []
        with open(embedding_file, 'r', encoding='utf-8') as f:
            for line in f:
                embedding = list(map(float, line.strip().split()))
                embedding_matrix.append(embedding)
        return np.array(embedding_matrix)

    def compute_relation_similarity(self, relation, entity):
        """Compute relation-entity similarity using pre-trained TransE embeddings"""
        if not hasattr(self, 'transE_embeddings') or not hasattr(self, 'entity_to_id'):
            ent2id_path = getattr(self, 'transE_entity2id_path', './data/entity2id.txt')
            embedding_path = getattr(self, 'transE_embedding_path', './data/entity2vec.TransE')
            self.load_transE_embeddings(ent2id_path, embedding_path)

        related_relations = set()
        entity_id = self.entity_to_id[entity]

        for triple in self.train_set:
            if triple[0] == entity or triple[2] == entity:
                related_relations.add(triple[1])

        if relation in related_relations:
            return 1.0

        if related_relations:
            max_similarity = 0.0
            entity_emb = self.transE_embeddings[entity_id]

            for rel in related_relations:
                string_sim = 0.0
                if relation.lower() in rel.lower() or rel.lower() in relation.lower():
                    string_sim = 0.3

                semantic_sim = self.compute_entity_relation_semantic_similarity(entity, rel)

                combined_sim = (string_sim + semantic_sim) / 2.0
                max_similarity = max(max_similarity, combined_sim)

            return max_similarity
        else:
            return 0.0

    def print_negative_sampling_stats(self):
        """Print negative sampling strategy statistics"""
        print(f"Exclusive relations: {len(self.exclusive_relations)}")
        print(f"Inclusive relations: {len(self.inclusive_relations)}")

    def compute_entity_relation_semantic_similarity(self, entity, relation):
        """Compute semantic similarity between entity and relation"""
        entity_relation_count = 0
        total_relation_count = 0

        for triple in self.train_set:
            if triple[1] == relation:
                total_relation_count += 1
                if triple[0] == entity or triple[2] == entity:
                    entity_relation_count += 1

        if total_relation_count > 0:
            return entity_relation_count / total_relation_count
        else:
            return 0.0
