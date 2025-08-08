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
                 add_tokens=False, p_tuning=False, model='bert', sp_num=5):
        """
        FKGC数据加载器
        Args:
            sp_num: 支持集大小 (support set size)
        """
        self.datasetName = in_paths['dataset']
        self.batch_size = batch_size
        self.sp_num = sp_num

        # 加载数据集
        self.train_set = self.load_dataset(in_paths['train'])
        self.valid_set = self.load_dataset(in_paths['valid'])
        self.test_set = self.load_dataset(in_paths['test'])

        # 从ent2id.txt文件读取实体映射
        self.load_entity_mapping_from_file(in_paths)
        
        # 从rel2ids.txt文件读取关系映射
        self.load_relation_mapping_from_file(in_paths)
        
        # 加载文本描述
        self.tokenizer = tokenizer
        self.uid2text = {}
        self.uid2tokens = {}
        for p in in_paths['text']:
            self.load_text(p)

        # 构建FKGC所需的数据结构
        self.build_fkgc_structures()

        # 其他参数
        self.max_desc_length = max_desc_length
        self.add_tokens = add_tokens
        self.p_tuning = p_tuning
        self.model = model
        self.orig_vocab_size = len(tokenizer)
        self.neg_rate = 7

        # 添加实体和关系数量属性
        self.n_ent = len(self.entity_set)
        self.n_rel = len(self.relation_set)

        # 添加特殊token
        if self.add_tokens:
            self.adding_tokens()

    def load_entity_mapping_from_file(self, in_paths):
        """从文件加载实体映射"""
        # 初始化字典
        self.ent2id = {}
        self.id2ent = {}
        
        # 加载实体映射
        ent2ids_path = in_paths.get('ent2ids')
        if ent2ids_path and os.path.exists(ent2ids_path):
            print(f"Loading entity mapping from {ent2ids_path}")
            with open(ent2ids_path, 'r', encoding='utf-8') as f:
                for line in f:
                    ent, ent_id = line.strip().split('\t')
                    self.ent2id[ent] = int(ent_id)
                    self.id2ent[int(ent_id)] = ent
            print(f"Loaded {len(self.ent2id)} entities from {ent2ids_path}")
            # 设置实体集合
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
                            types = parts[1:]  # 所有类型
                            self.entity_type_mapping[entity] = types
                print(f"Loaded entity types for {len(self.entity_type_mapping)} entities")
            else:
                print("Entity type mapping file not found, using fallback method")
                self.entity_type_mapping = {}
        else:
            print(f"Dataset {self.datasetName} uses entity name format for type extraction")
            self.entity_type_mapping = {}

    def _build_entity_mapping_from_data(self):
        """从训练数据构建实体映射"""
        self.entity_set = set([t[0] for t in (self.train_set + self.valid_set + self.test_set)] +
                              [t[-1] for t in (self.train_set + self.valid_set + self.test_set)])
        self.entity_list = sorted(self.entity_set)
        self.ent2id = {e: i for i, e in enumerate(self.entity_list)}
        self.id2ent = {i: e for i, e in enumerate(self.entity_list)}
        print(f"Built entity mapping from training data: {len(self.ent2id)} entities")

    def load_relation_mapping_from_file(self, in_paths):
        """从rel2ids.txt文件加载背景关系映射"""
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
                        if not line:  # 跳过空行
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
                    # 显示前几个背景关系作为验证
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
        """从训练数据构建关系映射（备用方法）"""
        print("Building relation mapping from training data (this includes task relations)...")
        self.relation_set = set([t[1] for t in (self.train_set + self.valid_set + self.test_set)])
        self.relation_list = sorted(self.relation_set)
        self.rel2id = {r: i for i, r in enumerate(self.relation_list)}
        self.id2rel = {i: r for i, r in enumerate(self.relation_list)}
        print(f"Built relation mapping from training data: {len(self.rel2id)} relations (including task relations)")

    def build_fkgc_structures(self):
        """构建FKGC所需的数据结构"""
        # 按关系分组训练数据
        self.rel2train_triples = defaultdict(list)
        for triple in self.train_set:
            h, r, t = triple
            self.rel2train_triples[r].append(triple)

        # 构建 e1rel_e2 映射 (头实体+关系 -> 尾实体列表)
        self.e1rel_e2 = defaultdict(list)
        for triple in self.train_set:
            h, r, t = triple
            key = h + r
            self.e1rel_e2[key].append(t)

        # 从rel2cands.json加载候选实体
        self.rel2candidates = self.load_candidates_from_json()

        self.train_test_path_id = self.build_path_info()

        # 构建任务池
        self.train_tasks = self.rel2train_triples
        self.valid_tasks = self.build_tasks(self.valid_set)
        self.test_tasks = self.build_tasks(self.test_set)

        # 关系分类
        self.classify_relations()

    def load_candidates_from_json(self):
        """从rel2cands.json文件加载候选实体"""
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
                
                # 验证候选实体是否在实体映射中
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
        """从数据集构建候选实体（备用方法）"""
        print("Building candidates from dataset...")
        rel2candidates = {}
        for rel in self.relation_set:
            candidates = set()
            # 从所有数据集（训练、验证、测试）中收集候选实体
            for triple in self.train_set + self.valid_set + self.test_set:
                if triple[1] == rel:
                    candidates.add(triple[0])  # 头实体
                    candidates.add(triple[2])  # 尾实体
            rel2candidates[rel] = list(candidates)
        print(f"Built candidates for {len(rel2candidates)} relations from dataset")
        return rel2candidates

    def build_tasks(self, dataset):
        """将数据集按关系分组构建任务"""
        tasks = defaultdict(list)
        for triple in dataset:
            h, r, t = triple
            tasks[r].append(triple)
        return tasks

    def build_path_info(self):
        """构建路径信息 - 从path_graph文件读取背景关系的路径信息"""
        path_info = {}

        # 尝试读取path_graph文件
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

                # 首先构建实体对到路径的映射
                pair_to_paths = {}

                with open(path_graph_file, 'r', encoding='utf8') as f:
                    for line in f:
                        line_count += 1
                        parts = line.strip().split('\t')
                        if len(parts) >= 3:
                            h, r, t = parts[0], parts[1], parts[2]

                            # 处理concept:前缀 - 移除所有concept:前缀
                            h_clean = h.replace('concept:', '')
                            r_clean = r.replace('concept:', '')
                            t_clean = t.replace('concept:', '')

                            # 检查实体和背景关系是否在映射中
                            # 注意：path_graph中的关系应该是背景关系，应该在rel2id中
                            if h_clean in self.ent2id and t_clean in self.ent2id and r_clean in self.rel2id:
                                # 使用实体对作为key
                                pair = (h_clean, t_clean)
                                if pair not in pair_to_paths:
                                    pair_to_paths[pair] = []
                                pair_to_paths[pair].append([r_clean])
                                valid_count += 1
                            else:
                                format_mismatch_count += 1
                                # 调试信息：显示不匹配的原因
                                if line_count <= 3:
                                    print(f"  Line {line_count} mismatch:")
                                    print(f"    h_clean in ent2id: {h_clean in self.ent2id}")
                                    print(f"    t_clean in ent2id: {t_clean in self.ent2id}")
                                    print(f"    r_clean in rel2id: {r_clean in self.rel2id}")
                                    if r_clean not in self.rel2id:
                                        print(f"    Background relation '{r_clean}' not found in rel2id")

                # 转换为ID格式
                for pair, paths in pair_to_paths.items():
                    h_id = self.ent2id[pair[0]]
                    t_id = self.ent2id[pair[1]]
                    # 转换路径中的关系为ID
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
        """构建简化的路径信息（从训练集三元组）"""
        path_info = {}
        print("Building simplified path information from training set...")

        # 构建实体对之间的直接关系
        for triple in self.train_set:
            h, r, t = triple
            if h in self.ent2id and t in self.ent2id and r in self.rel2id:
                pair = (self.ent2id[h], self.ent2id[t])
                if pair not in path_info:
                    path_info[pair] = []
                path_info[pair].append([self.rel2id[r]])

        # 构建多跳路径（2跳）
        print("Building 2-hop paths...")
        entity_to_relations = {}
        for triple in self.train_set:
            h, r, t = triple
            if h in self.ent2id and t in self.ent2id and r in self.rel2id:
                h_id, t_id, r_id = self.ent2id[h], self.ent2id[t], self.rel2id[r]

                # 构建实体到关系的映射
                if h_id not in entity_to_relations:
                    entity_to_relations[h_id] = []
                entity_to_relations[h_id].append((r_id, t_id))

        # 构建2跳路径
        two_hop_count = 0
        for h_id in entity_to_relations:
            for r1_id, mid_id in entity_to_relations[h_id]:
                if mid_id in entity_to_relations:
                    for r2_id, t_id in entity_to_relations[mid_id]:
                        if h_id != t_id:  # 避免自环
                            pair = (h_id, t_id)
                            if pair not in path_info:
                                path_info[pair] = []
                            path_info[pair].append([r1_id, r2_id])
                            two_hop_count += 1
                            if two_hop_count >= 10000:  # 限制数量
                                break
            if two_hop_count >= 10000:
                break

        print(f"Built simplified path information:")
        print(f"  - Direct paths: {len([p for p in path_info.values() if len(p[0]) == 1])}")
        print(f"  - 2-hop paths: {two_hop_count}")
        print(f"  - Total entity pairs: {len(path_info)}")

        return path_info

    def load_dataset(self, in_path):
        """加载数据集"""
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
        """加载文本描述"""
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
        """训练数据生成器"""
        task_pool = list(self.train_tasks.keys())
        num_tasks = len(task_pool)
        rel_idx = 0

        while True:
            if rel_idx % num_tasks == 0:
                random.shuffle(task_pool)

            query_rel = task_pool[rel_idx % num_tasks]
            rel_idx += 1

            # 获取候选实体
            candidates = self.rel2candidates.get(query_rel, [])
            candidates_id = [self.ent2id[c] for c in candidates if c in self.ent2id]

            # 候选实体数量检查
            if len(candidates_id) <= 20:
                continue

            # 获取该关系的训练三元组
            rel_triples = self.train_tasks[query_rel]
            random.shuffle(rel_triples)

            # 转换为ID形式
            train_tri_id = [[self.ent2id[triple[0]], self.rel2id[triple[1]], self.ent2id[triple[2]]]
                            for triple in rel_triples]

            train_tri_id_fil = []
            for trip in train_tri_id:
                if trip[0] != trip[2]:
                    train_tri_id_fil.append(trip)
            train_tri_id = train_tri_id_fil

            # 检查数据量是否足够
            if len(train_tri_id) < self.sp_num + self.batch_size:
                continue

            # 分割支持集和查询集
            support_pair = train_tri_id[:self.sp_num]
            query_pair = train_tri_id[self.sp_num:]

            if len(support_pair) == 0 or len(query_pair) == 0:
                continue

            # 查询集采样 - 参考代码逻辑
            if len(query_pair) < self.batch_size:
                query_pair_pos = [random.choice(query_pair) for _ in range(self.batch_size)]
            else:
                query_pair_pos = random.sample(query_pair, self.batch_size)

            # 转换为实体对形式
            support_pair = [[pair[0], pair[2]] for pair in support_pair]
            query_pair_pos = [[pair[0], pair[2]] for pair in query_pair_pos]

            # 构建one-to-many映射
            one_tomany_train = []
            for i in range(len(query_pair_pos)):
                key = self.id2ent[int(query_pair_pos[i][0])] + query_rel
                one2many = self.e1rel_e2.get(key, [])
                one2many2id = [self.ent2id[_] for _ in one2many if _ in self.ent2id]
                one_tomany_train.append(one2many2id)

            yield support_pair, query_pair_pos, one_tomany_train, candidates_id, query_rel

    def valid_generator(self):
        """验证数据生成器 """
        task_pool = list(self.valid_tasks.keys())
        num_tasks = len(task_pool)
        rel_idx = 0

        while True:
            if rel_idx % num_tasks == 0:
                random.shuffle(task_pool)

            query_rel = task_pool[rel_idx % num_tasks]
            rel_idx += 1

            # 获取候选实体
            candidates = self.rel2candidates.get(query_rel, [])
            candidates_id = [self.ent2id[c] for c in candidates if c in self.ent2id]

            # 候选实体数量检查
            if len(candidates_id) <= 20:
                continue

            # 获取该关系的验证三元组
            rel_triples = self.valid_tasks[query_rel]
            random.shuffle(rel_triples)

            # 转换为ID形式
            valid_tri_id = [[self.ent2id[triple[0]], self.rel2id[triple[1]], self.ent2id[triple[2]]]
                            for triple in rel_triples]

            # 过滤自环
            valid_tri_id_fil = []
            for trip in valid_tri_id:
                if trip[0] != trip[2]:
                    valid_tri_id_fil.append(trip)
            valid_tri_id = valid_tri_id_fil

            # 检查数据量是否足够
            if len(valid_tri_id) < self.sp_num + self.batch_size:
                continue

            # 分割支持集和查询集
            support_pair = valid_tri_id[:self.sp_num]
            query_pair = valid_tri_id[self.sp_num:]

            if len(support_pair) == 0 or len(query_pair) == 0:
                continue

            # 查询集采样
            if len(query_pair) < self.batch_size:
                query_pair_pos = [random.choice(query_pair) for _ in range(self.batch_size)]
            else:
                query_pair_pos = random.sample(query_pair, self.batch_size)

            # 转换为实体对形式
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
        """测试数据生成器 - 改进版本"""
        task_pool = list(self.test_tasks.keys())
        num_tasks = len(task_pool)
        rel_idx = 0

        while True:
            if rel_idx % num_tasks == 0:
                random.shuffle(task_pool)

            query_rel = task_pool[rel_idx % num_tasks]
            rel_idx += 1

            # 获取候选实体
            candidates = self.rel2candidates.get(query_rel, [])
            candidates_id = [self.ent2id[c] for c in candidates if c in self.ent2id]

            if len(candidates_id) <= 20:
                continue

            # 获取该关系的测试三元组
            rel_triples = self.test_tasks[query_rel]
            random.shuffle(rel_triples)

            # 转换为ID形式
            test_tri_id = [[self.ent2id[triple[0]], self.rel2id[triple[1]], self.ent2id[triple[2]]]
                           for triple in rel_triples]

            # 过滤自环
            test_tri_id_fil = []
            for trip in test_tri_id:
                if trip[0] != trip[2]:
                    test_tri_id_fil.append(trip)
            test_tri_id = test_tri_id_fil

            # 检查数据量是否足够
            if len(test_tri_id) < self.sp_num + self.batch_size:
                continue

            # 分割支持集和查询集
            support_pair = test_tri_id[:self.sp_num]
            query_pair = test_tri_id[self.sp_num:]

            if len(support_pair) == 0 or len(query_pair) == 0:
                continue

            # 查询集采样
            if len(query_pair) < self.batch_size:
                query_pair_pos = [random.choice(query_pair) for _ in range(self.batch_size)]
            else:
                query_pair_pos = random.sample(query_pair, self.batch_size)

            # 转换为实体对形式
            support_pair = [[pair[0], pair[2]] for pair in support_pair]
            query_pair_pos = [[pair[0], pair[2]] for pair in query_pair_pos]

            # 构建one-to-many映射
            one_tomany_test = []
            for i in range(len(query_pair_pos)):
                key = self.id2ent[int(query_pair_pos[i][0])] + query_rel
                one2many = self.e1rel_e2.get(key, [])
                one2many2id = [self.ent2id[_] for _ in one2many if _ in self.ent2id]
                one_tomany_test.append(one2many2id)

            yield support_pair, query_pair_pos, one_tomany_test, candidates_id, query_rel

    def get_batch_data(self, support_pair, query_pair_pos, one_tomany_train, candidates_id):
        """将批次数据转换为模型输入格式"""
        # 处理支持集
        support_texts = []
        support_tokens = []
        for pair in support_pair:
            h_id, t_id = pair
            h, t = self.id2ent[h_id], self.id2ent[t_id]
            # 这里需要根据具体任务添加关系信息
            text, tokens = self.triple_to_text((h, "relation", t), {'h': True, 'r': False, 't': True})
            support_texts.append(text)
            support_tokens.append(tokens)

        # 处理查询集
        query_texts = []
        query_tokens = []
        for pair in query_pair_pos:
            h_id, t_id = pair
            h, t = self.id2ent[h_id], self.id2ent[t_id]
            text, tokens = self.triple_to_text((h, "relation", t), {'h': True, 'r': False, 't': True})
            query_texts.append(text)
            query_tokens.append(tokens)

        # Tokenize
        support_batch = self.my_tokenize(support_tokens, max_length=512, padding=True, model=self.model)
        query_batch = self.my_tokenize(query_tokens, max_length=512, padding=True, model=self.model)

        # 处理候选实体
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
        """将三元组转换为文本"""
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

        if self.add_tokens:
            if self.p_tuning:
                h_token = ["[head_b1]", "[head_b2]"] + (
                    ['[ent_{}]'.format(ent2id[h])] if with_text['h'] else [tokenizer.mask_token]) + ["[head_a1]",
                                                                                                     "[head_a2]"]
                r_token = ["[rel_b1]", "[rel_b2]"] + (
                    ['[rel_{}]'.format(rel2id[r])] if with_text['r'] else [tokenizer.mask_token]) + ["[rel_a1]",
                                                                                                     "[rel_a2]"]
                t_token = ["[tail_b1]", "[tail_b2]"] + (
                    ['[ent_{}]'.format(ent2id[t])] if with_text['t'] else [tokenizer.mask_token]) + ["[tail_a1]",
                                                                                                     "[tail_a2]"]
            else:
                h_token = ['[ent_{}]'.format(ent2id[h])] if with_text['h'] else [tokenizer.mask_token]
                r_token = ['[rel_{}]'.format(rel2id[r])] if with_text['r'] else [tokenizer.mask_token]
                t_token = ['[ent_{}]'.format(ent2id[t])] if with_text['t'] else [tokenizer.mask_token]
        else:
            h_token = [self.tokenizer.cls_token] if with_text['h'] else [tokenizer.mask_token]
            r_token = [self.tokenizer.cls_token] if with_text['r'] else [tokenizer.mask_token]
            t_token = [self.tokenizer.cls_token] if with_text['t'] else [tokenizer.mask_token]

        tokens = h_token + h_text_tokens + r_token + r_text_tokens + t_token + t_text_tokens
        text = tokenizer.convert_tokens_to_string(tokens)

        return text, tokens

    def element_to_text(self, target):
        """将单个元素转换为文本"""
        tokenizer = self.tokenizer
        ent2id = self.ent2id
        rel2id = self.rel2id

        n_tokens = min(508, self.max_desc_length)
        text_tokens = self.uid2tokens.get(target, [])[:n_tokens]

        if self.add_tokens:
            if target in ent2id:
                token = ['[ent_{}]'.format(ent2id[target])]
            else:
                token = ['[rel_{}]'.format(rel2id[target])]
        else:
            token = [self.tokenizer.cls_token]

        tokens = token + text_tokens
        text = tokenizer.convert_tokens_to_string(tokens)

        return text, tokens

    def my_tokenize(self, batch_tokens, max_length=1024, padding=True, model='roberta'):
        """Tokenize处理"""
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

    def adding_tokens(self):
        """添加特殊token"""
        n_ent = len(self.ent2id)
        n_rel = len(self.rel2id)

        new_tokens = ["[ent_{}]".format(i) for i in range(n_ent)] + ["[rel_{}]".format(i) for i in range(n_rel)]

        if self.p_tuning:
            new_tokens += ["[head_b1]", "[head_b2]", "[head_a1]", "[head_a2]",
                           "[rel_b1]", "[rel_b2]", "[rel_a1]", "[rel_a2]",
                           "[tail_b1]", "[tail_b2]", "[tail_a1]", "[tail_a2]"]

        self.tokenizer.add_tokens(new_tokens)

    def get_relation_tasks(self, split='train'):
        """获取关系任务"""
        if split == 'train':
            return self.train_tasks
        elif split == 'valid':
            return self.valid_tasks
        elif split == 'test':
            return self.test_tasks
        else:
            raise ValueError(f"Unknown split: {split}")

    def get_candidates_for_relation(self, relation):
        """获取关系的候选实体"""
        return self.rel2candidates.get(relation, [])

    def get_path_info(self, head_id, tail_id):
        """获取实体对之间的路径信息"""
        pair = (head_id, tail_id)
        return self.train_test_path_id.get(pair, [])

    def get_path_info_by_entity_names(self, head_name, tail_name):
        """根据实体名称获取路径信息"""
        if head_name in self.ent2id and tail_name in self.ent2id:
            head_id = self.ent2id[head_name]
            tail_id = self.ent2id[tail_name]
            return self.get_path_info(head_id, tail_id)
        return []

    def generate_negative_samples(self, support_pair, query_pair_pos, relation, num_negatives=7):
        """为FKGC任务生成负样本 - 多重负采样策略"""
        neg_samples = []

        # 首先进行关系分类
        if not hasattr(self, 'exclusive_relations'):
            self.classify_relations()

        for pair in query_pair_pos:
            h_id, t_id = pair
            h, t = self.id2ent[h_id], self.id2ent[t_id]

            # 为每个正样本生成多个负样本
            for _ in range(num_negatives):
                neg_sample = None

                # 根据关系类型选择不同的负采样策略
                if self.is_exclusive_relation(relation):
                    # Exclusive关系：使用类型感知替换
                    neg_sample = self.type_aware_replacement(h, t, relation)
                elif self.is_inclusive_relation(relation):
                    # Inclusive关系：使用语义对比采样
                    neg_sample = self.semantic_contrastive_sampling(h, t, relation)

                if neg_sample is None:
                    neg_sample = self.random_replacement(h, t, relation)

                if neg_sample:
                    neg_samples.append(neg_sample)

        return neg_samples

    def random_replacement(self, h, t, relation):
        """随机替换负样本"""
        candidates = list(self.entity_set - {h, t})
        if not candidates:
            print(f"Warning: Random replacement failed - no candidates for ({h}, {relation}, {t})")
            print(f"  Entity set size: {len(self.entity_set)}")
            return None

        max_attempts = 50
        for attempt in range(max_attempts):
            
            replace_ent = random.choice(candidates)
            neg_triple = (h, relation, replace_ent)

            # 确保不是真实的三元组
            if neg_triple not in self.train_set and neg_triple not in self.valid_set and neg_triple not in self.test_set:
                return neg_triple
        
        print(f"Warning: Random replacement failed after {max_attempts} attempts for ({h}, {relation}, {t})")
        return None

    def type_aware_replacement(self, h, t, relation):
        """类型感知负样本替换"""
        if not self.is_exclusive_relation(relation):
            return None  # 只对exclusive关系使用类型感知替换

        # 获取原始尾实体的类型
        t_types = self.get_entity_types(t)

        # 预计算真实三元组，避免重复检查
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
                if type_sim > 0.0:  # 有类型重叠
                    type_similar_candidates.append((ent, type_sim))
                    if len(type_similar_candidates) >= 50:
                        break
            count += 1

        if not type_similar_candidates:
            return None

        # 按类型相似度排序，选择最相似的
        type_similar_candidates.sort(key=lambda x: x[1], reverse=True)
        
        # 从前10个最相似的候选者中随机选择
        top_candidates = type_similar_candidates[:10]
        selected_ent, similarity = random.choice(top_candidates)
        
        return (h, relation, selected_ent)

    def relation_constrained_replacement(self, h, t, relation):
        """关系约束负样本替换"""
        # 获取相同关系的其他三元组中的实体
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
        """提取实体类型"""
        # 对于fb15k237-one数据集，使用预加载的类型映射
        if hasattr(self, 'entity_type_mapping') and entity in self.entity_type_mapping:
            return self.entity_type_mapping[entity][0]  # 返回第一个类型作为主要类型
        
        # 对于nell数据集，使用type:ent_name格式
        if 'nell' in self.datasetName:
            return entity.split(':')[0] if ':' in entity else 'unknown'

        return 'unknown'
    
    def get_entity_types(self, entity):
        # 对于fb15k237-one数据集，使用预加载的类型映射
        if hasattr(self, 'entity_type_mapping') and entity in self.entity_type_mapping:
            return self.entity_type_mapping[entity]
        
        # 对于nell数据集，使用type:ent_name格式
        if 'nell' in self.datasetName:
            if ':' in entity:
                return [entity.split(':')[0]]
            else:
                return ['unknown']
        
        # 默认情况
        return ['unknown']
    
    def compute_type_similarity(self, entity1, entity2):
        """计算两个实体的类型相似度"""
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
        """获取包含负样本的批次数据"""
        # 生成负样本
        neg_samples = self.generate_negative_samples(support_pair, query_pair_pos, relation)

        pos_neg_pairs = []
        for i, pos_pair in enumerate(query_pair_pos):
            if i < len(neg_samples):
                pos_neg_pairs.append((pos_pair, neg_samples[i]))

        # 转换为模型输入格式
        batch_data = self.convert_to_model_input(support_pair, pos_neg_pairs)

        return batch_data

    def convert_to_model_input(self, support_pair, pos_neg_pairs):
        """转换为模型输入格式"""
        support_texts = []
        pos_texts = []
        neg_texts = []

        # 处理支持集
        for pair in support_pair:
            h_id, t_id = pair
            h, t = self.id2ent[h_id], self.id2ent[t_id]
            text, _ = self.triple_to_text((h, "relation", t), {'h': True, 'r': False, 't': True})
            support_texts.append(text)

        # 处理正负样本对
        for pos_pair, neg_pair in pos_neg_pairs:
            h_id, t_id = pos_pair
            h, t = self.id2ent[h_id], self.id2ent[t_id]
            text, _ = self.triple_to_text((h, "relation", t), {'h': True, 'r': False, 't': True})
            pos_texts.append(text)

            h_neg, r_neg, t_neg = neg_pair
            text, _ = self.triple_to_text((h_neg, r_neg, t_neg), {'h': True, 'r': True, 't': True})
            neg_texts.append(text)

        # Tokenize
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
        """将任务关系分类为exclusive和inclusive两种类型（只对训练集、验证集、测试集中的关系进行分类）"""
        self.exclusive_relations = set()
        self.inclusive_relations = set()

        # 收集所有任务关系
        task_relations = set()
        for triple in self.train_set + self.valid_set + self.test_set:
            task_relations.add(triple[1])

        print(f"Classifying {len(task_relations)} task relations...")

        for relation in task_relations:
            head_to_tails = {}
            
            # 从训练集、验证集、测试集中收集数据
            all_triples = self.train_set + self.valid_set + self.test_set
            for triple in all_triples:
                if triple[1] == relation:
                    h, t = triple[0], triple[2]
                    if h not in head_to_tails:
                        head_to_tails[h] = set()
                    head_to_tails[h].add(t)

            # 计算平均尾实体数量
            if head_to_tails:
                avg_tails_per_head = sum(len(tails) for tails in head_to_tails.values()) / len(head_to_tails)

                # 如果平均每个头实体对应少于2个尾实体，认为是exclusive关系
                if avg_tails_per_head < 2.0:
                    self.exclusive_relations.add(relation)
                else:
                    self.inclusive_relations.add(relation)
            else:
                # 默认分类为inclusive
                self.inclusive_relations.add(relation)

        print(f"Exclusive task relations: {len(self.exclusive_relations)}")
        print(f"Inclusive task relations: {len(self.inclusive_relations)}")
        
        # 打印一些任务关系示例及其分类
        exclusive_examples = list(self.exclusive_relations)[:5]
        inclusive_examples = list(self.inclusive_relations)[:5]
        print(f"Exclusive task relation examples: {exclusive_examples}")
        print(f"Inclusive task relation examples: {inclusive_examples}")
        
        # 打印任务关系的分类结果
        print("\n=== Task Relation Classification Results ===")
        print(f"{'Relation':<50} {'Type':<12} {'Train':<6} {'Valid':<6} {'Test':<6} {'Total':<6} {'Avg_Tails':<10}")
        print("-" * 100)
        
        # 收集所有任务关系
        task_relations = set()
        for triple in self.train_set + self.valid_set + self.test_set:
            task_relations.add(triple[1])
        
        # 按关系名称排序
        sorted_task_relations = sorted(task_relations)
        
        for relation in sorted_task_relations:
            train_count = sum(1 for triple in self.train_set if triple[1] == relation)
            valid_count = sum(1 for triple in self.valid_set if triple[1] == relation)
            test_count = sum(1 for triple in self.test_set if triple[1] == relation)
            total_count = train_count + valid_count + test_count
            
            # 计算平均尾实体数量
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

    def is_exclusive_relation(self, relation):
        """判断关系是否为exclusive类型"""
        return relation in self.exclusive_relations

    def is_inclusive_relation(self, relation):
        """判断关系是否为inclusive类型"""
        return relation in self.inclusive_relations

    def semantic_contrastive_sampling(self, h, t, relation):
        """语义对比采样 - Semantic Contrastive Sampling（简化版本）"""
        if not self.is_inclusive_relation(relation):
            return None  # 只对inclusive关系使用语义对比采样
        
        # 创建真实三元组的集合，用于快速查找
        real_triples = set()
        for triple in self.train_set + self.valid_set + self.test_set:
            if triple[0] == h and triple[1] == relation:
                real_triples.add(triple[2])
        
        # 快速获取候选实体
        candidates = []
        for ent in self.entity_set:
            if ent != t and ent not in real_triples:
                candidates.append(ent)
                if len(candidates) >= 200:  # 限制候选数量，避免过多
                    break

        if not candidates:
            return None

        # 大幅限制候选数量以避免计算过慢
        if len(candidates) > 100:  # 从1000减少到100
            candidates = random.sample(candidates, 100)

        if len(candidates) >= 10:
            selected_candidate = random.choice(candidates)
            return (h, relation, selected_candidate)

        candidate_scores = []
        for i, candidate in enumerate(candidates[:20]):  # 最多计算20个候选
            try:
                entity_sim = self.compute_entity_similarity(t, candidate)
                candidate_scores.append((candidate, entity_sim))
            except Exception as e:
                continue

        if not candidate_scores:
            return (h, relation, random.choice(candidates))

        # 选择相似度最高的候选作为负样本
        best_candidate = max(candidate_scores, key=lambda x: x[1])[0]
        return (h, relation, best_candidate)

    def compute_entity_similarity(self, ent1, ent2):
        """计算两个实体的语义相似度 - 使用预训练的TransE嵌入"""
        if ent1 == ent2:
            return 1.0

        # 检查是否已加载TransE嵌入
        if not hasattr(self, 'transE_embeddings') or not hasattr(self, 'entity_to_id'):
            # 加载TransE嵌入
            ent2id_path = getattr(self, 'transE_entity2id_path', './data/entity2id.txt')
            embedding_path = getattr(self, 'transE_embedding_path', './data/entity2vec.TransE')
            self.load_transE_embeddings(ent2id_path, embedding_path)

        # 检查实体是否在映射中
        if ent1 not in self.entity_to_id:
            print(f"Warning: Entity {ent1} not found in TransE mapping, using fallback similarity")
            return 0.5  # 返回默认相似度

        if ent2 not in self.entity_to_id:
            print(f"Warning: Entity {ent2} not found in TransE mapping, using fallback similarity")
            return 0.5  # 返回默认相似度

        # 获取实体的嵌入
        ent1_id = self.entity_to_id[ent1]
        ent2_id = self.entity_to_id[ent2]

        # 检查ID是否在有效范围内
        if ent1_id >= len(self.transE_embeddings) or ent2_id >= len(self.transE_embeddings):
            print(f"Warning: Entity ID out of range, using fallback similarity")
            return 0.5

        # 计算余弦相似度
        emb1 = self.transE_embeddings[ent1_id]
        emb2 = self.transE_embeddings[ent2_id]

        # 归一化嵌入
        emb1_norm = emb1 / (np.linalg.norm(emb1) + 1e-8)
        emb2_norm = emb2 / (np.linalg.norm(emb2) + 1e-8)

        # 计算余弦相似度
        similarity = np.dot(emb1_norm, emb2_norm)

        return float(similarity)

    def load_transE_embeddings(self, ent2ids_file_path=None, embedding_file_path=None):
        """加载TransE嵌入"""
        # 设置默认路径
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
        """加载实体到ID的映射"""
        entity_to_id = {}
        with open(ent2ids_file_path, 'r', encoding='utf-8') as file:
            for line in file:
                entity, id_str = line.strip().split('\t')
                entity_to_id[entity] = int(id_str)
        return entity_to_id

    def load_transe_embedding(self, embedding_file):
        """加载TransE"""
        embedding_matrix = []
        with open(embedding_file, 'r', encoding='utf-8') as f:
            for line in f:
                embedding = list(map(float, line.strip().split()))
                embedding_matrix.append(embedding)
        return np.array(embedding_matrix)

    def compute_relation_similarity(self, relation, entity):
        """计算关系与实体的相似度 - 使用预训练的TransE嵌入"""
        if not hasattr(self, 'transE_embeddings') or not hasattr(self, 'entity_to_id'):
            # 加载TransE嵌入
            ent2id_path = getattr(self, 'transE_entity2id_path', './data/entity2id.txt')
            embedding_path = getattr(self, 'transE_embedding_path', './data/entity2vec.TransE')
            self.load_transE_embeddings(ent2id_path, embedding_path)

        # 找到与entity相关的所有关系
        related_relations = set()
        entity_id = self.entity_to_id[entity]

        # 从训练集中找到与实体相关的关系
        for triple in self.train_set:
            if triple[0] == entity or triple[2] == entity:
                related_relations.add(triple[1])

        if relation in related_relations:
            return 1.0

        # 计算关系与实体的语义相似度
        if related_relations:
            max_similarity = 0.0
            entity_emb = self.transE_embeddings[entity_id]

            for rel in related_relations:
                # 计算关系名称的字符串相似度
                string_sim = 0.0
                if relation.lower() in rel.lower() or rel.lower() in relation.lower():
                    string_sim = 0.3

                # 计算实体与关系的语义相似度
                semantic_sim = self.compute_entity_relation_semantic_similarity(entity, rel)

                # 综合相似度
                combined_sim = (string_sim + semantic_sim) / 2.0
                max_similarity = max(max_similarity, combined_sim)

            return max_similarity
        else:
            return 0.0

    def compute_entity_relation_semantic_similarity(self, entity, relation):
        """计算实体与关系的语义相似度"""
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
