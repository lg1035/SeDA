import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from transformers import AutoModel, AutoTokenizer
from lora_utils import LoRAConfig, apply_lora_to_model, get_lora_parameters
from dataloader import FKGCDataLoader
from collections import OrderedDict



def get_model_device(model):
    if hasattr(model, 'module'):
        return next(model.module.parameters()).device
    else:
        return next(model.parameters()).device


class GATLayer(nn.Module):

    def __init__(self, in_dim, out_dim, num_heads=8, dropout=0.1):
        super(GATLayer, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.dropout = dropout

        self.W = nn.Linear(in_dim, out_dim * num_heads, bias=False)
        self.attention = nn.Parameter(torch.Tensor(1, num_heads, out_dim * 2))
        self.bias = nn.Parameter(torch.Tensor(out_dim))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.attention)
        nn.init.zeros_(self.bias)

    def forward(self, x, edge_index):
        N = x.size(0)
        E = edge_index.size(1)


        x = self.W(x).view(N, self.num_heads, -1)

        if E == 0:
            out = torch.zeros(N, self.num_heads, self.out_dim, device=x.device)
            return out.mean(dim=1) + self.bias

        row, col = edge_index
        x_i = x[row]
        x_j = x[col]

        alpha_input = torch.cat([x_i, x_j], dim=-1)
        alpha = torch.einsum('bhd,hd->bh', alpha_input, self.attention.squeeze(0))
        alpha = F.leaky_relu(alpha, negative_slope=0.2)

        alpha = alpha.view(-1, self.num_heads)
        
        try:
            from torch_scatter import scatter_softmax
            alpha_normalized = scatter_softmax(alpha, row, dim=0)
        except ImportError:
            alpha_normalized = torch.zeros_like(alpha)
            for i in range(N):
                mask = (row == i)
                if mask.any():
                    alpha_normalized[mask] = F.softmax(alpha[mask], dim=0)

        alpha = F.dropout(alpha_normalized, p=self.dropout, training=self.training)
        alpha = alpha.unsqueeze(-1)

        weighted_features = alpha * x_j
        out = torch.zeros(N, self.num_heads, self.out_dim, device=x.device)
        out.scatter_add_(0, row.unsqueeze(1).unsqueeze(2).expand(-1, self.num_heads, self.out_dim), weighted_features)

        final_output = out.mean(dim=1) + self.bias
        
        return final_output


class FKGCWithGAT(nn.Module):
    """FKGC model with BERT initialization and GAT improvement"""

    def __init__(self, bert_model_name, n_ent, n_rel, hidden_dim=768,
                 gat_heads=8, gat_layers=2, dropout=0.1,
                 use_lora=False, lora_config=None, data_loader=None, max_subgraph_edges=20000,
                 use_gat=True, debug_mode=False, fine_tune_bert=False,
                 scoring_paradigm='transe', fkgc_attention_temp=1.0, fkgc_prototype_shrinkage=0.1,
                 fkgc_attention_weight=0.7):
        """
        Initialize FKGC model
        Args:
            bert_model_name: BERT model name
            n_ent: number of entities
            n_rel: number of relations
            hidden_dim: hidden dimension
            gat_heads: GAT attention heads
            gat_layers: GAT layers
            dropout: dropout rate
            use_lora: whether to use LoRA
            lora_config: LoRA configuration
            data_loader: data loader
            max_subgraph_edges: max subgraph edges
            scoring_paradigm: scoring paradigm, 'transe' or 'fkgc'
            fkgc_attention_temp: FKGC attention temperature
            fkgc_prototype_shrinkage: FKGC prototype shrinkage
            fkgc_attention_weight: FKGC attention weight
        """
        super(FKGCWithGAT, self).__init__()

        self.bert_model_name = bert_model_name
        self.n_ent = n_ent
        self.n_rel = n_rel
        self.hidden_dim = hidden_dim
        self.gat_heads = gat_heads
        self.gat_layers = gat_layers
        self.dropout = dropout
        self.use_lora = use_lora
        self.lora_config = lora_config
        self.data_loader = data_loader
        self.max_subgraph_edges = max_subgraph_edges
        self.use_gat = use_gat
        self.debug_mode = False
        self.fine_tune_bert = fine_tune_bert

        self.scoring_paradigm = scoring_paradigm
        self.fkgc_attention_temp = fkgc_attention_temp
        self.fkgc_prototype_shrinkage = fkgc_prototype_shrinkage
        self.fkgc_attention_weight = fkgc_attention_weight
        self.debug_fkgc = False
        self.prototype_cache = {}
        self.cache_enabled = True
        self._safe_post_init(data_loader, bert_model_name, use_lora, lora_config, dropout)

    def clear_prototype_cache(self):
        self.prototype_cache.clear()

    def disable_prototype_cache(self):
        self.cache_enabled = False

    def enable_prototype_cache(self):
        self.cache_enabled = True

    def _safe_post_init(self, data_loader, bert_model_name, use_lora, lora_config, dropout):
        self.ent2id = getattr(data_loader, 'ent2id', {}) if data_loader is not None else {}
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(bert_model_name)
            self.bert = AutoModel.from_pretrained(bert_model_name)
            if use_lora and lora_config:
                apply_lora_to_model(self.bert, lora_config)
                print("Applied LoRA to BERT with provided configuration")
            print(f"BERT model loaded: {bert_model_name}")
        except Exception as e:
            print(f"Error loading BERT model: {e}")
            self.bert = None

        self.ent_embeddings = nn.Embedding(self.n_ent, self.hidden_dim)
        self.rel_embeddings = nn.Embedding(self.n_rel, self.hidden_dim)


        if self.use_gat:
            num_layers = int(self.gat_layers) if isinstance(self.gat_layers, int) else 2
            self.gat_layers = nn.ModuleList()
            for _ in range(num_layers):
                self.gat_layers.append(
                    GATLayer(self.hidden_dim, self.hidden_dim, self.gat_heads, dropout))
        else:
            num_layers = int(self.gat_layers) if isinstance(self.gat_layers, int) else 2
            self.mlp_layers = nn.ModuleList()
            for _ in range(num_layers):
                mlp_layer = nn.Sequential(
                    nn.Linear(self.hidden_dim, self.hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(self.hidden_dim, self.hidden_dim)
                )
                self.mlp_layers.append(mlp_layer)

        self.residual_weight = nn.Parameter(torch.tensor(0.1))
        self.dropout_rate = 0.3
        self.l2_reg_weight = 0.01
        self.embedding_norm_weight = 0.001
        self.score_margin = 0.5

        self.structural_gate = nn.Parameter(torch.tensor(0.5))
        self.semantic_gate = nn.Parameter(torch.tensor(0.5))
        
        self.adaptive_gate_net = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 2),
            nn.Sigmoid()
        )
        
        self.structural_entity_embeddings = nn.Embedding(self.n_ent, self.hidden_dim)
        self.structural_rel_embeddings = nn.Embedding(self.n_rel, self.hidden_dim)

        self.init_embeddings()

        if not hasattr(self, '_gat_config_printed'):
            print(f"\n=== GAT Model Configuration ===")
            print(f"GAT enabled: {self.use_gat}")
            if self.use_gat:
                print(f"Number of GAT layers: {len(self.gat_layers)}")
                print(f"GAT heads: {self.gat_heads}")
                print(f"Hidden dimension: {self.hidden_dim}")
                print(f"Max subgraph edges: {self.max_subgraph_edges}")
                print(f"Structural gate weight: {self.structural_gate.item():.3f}")
                print(f"Semantic gate weight: {self.semantic_gate.item():.3f}")
                print(f"Adaptive gating: Structural + Semantic")
                print(f"Adaptive gate network: {self.hidden_dim * 2} -> {self.hidden_dim} -> 2")
                print(f"Structural embeddings: TransE (always initialized)")
                print(f"Semantic embeddings: BERT + GAT")
            else:
                print(f"Number of MLP layers: {len(self.mlp_layers)}")
            print(f"=== End GAT Configuration ===\n")
            self._gat_config_printed = True

        self.relation_attn = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.Tanh(),
            nn.Linear(self.hidden_dim, 1)
        )
        self.proj = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.temperature = 0.07
        self.path_feat_linear = nn.Linear(3, 1)
        self.path_bias_scale = nn.Parameter(torch.tensor(0.2))



    def init_embeddings(self):
        nn.init.xavier_uniform_(self.ent_embeddings.weight)
        nn.init.xavier_uniform_(self.rel_embeddings.weight)

        nn.init.xavier_uniform_(self.structural_entity_embeddings.weight)
        nn.init.xavier_uniform_(self.structural_rel_embeddings.weight)

        self.load_structural_embeddings()



    def bert_encode_entity(self, entity_texts):
        try:
            inputs = self.tokenizer(entity_texts,
                                    padding=True,
                                    truncation=True,
                                    max_length=64,
                                    return_tensors='pt')

            device = get_model_device(self)
            inputs = {k: v.to(device) for k, v in inputs.items()}

            self.bert.eval()
            with torch.no_grad():
                if torch.cuda.is_available():
                    with torch.cuda.amp.autocast(dtype=torch.float16):
                        outputs = self.bert(**inputs)
                else:
                    outputs = self.bert(**inputs)

            entity_embeddings = outputs.last_hidden_state[:, 0, :]
            
            if hasattr(self, 'bert_projection') and self.bert_projection is not None:
                entity_embeddings = self.bert_projection(entity_embeddings)
            else:
                if not hasattr(self, '_bert_projection_fallback'):
                    self._bert_projection_fallback = nn.Linear(768, self.hidden_dim).to(device)
                    nn.init.xavier_uniform_(self._bert_projection_fallback.weight)
                    nn.init.zeros_(self._bert_projection_fallback.bias)
                entity_embeddings = self._bert_projection_fallback(entity_embeddings)

            del inputs, outputs

            return entity_embeddings

        except Exception as e:
            print(f"Error in BERT encoding: {e}")
            device = get_model_device(self)
            return torch.zeros(len(entity_texts), self.hidden_dim, device=device)

    def initialize_semantic_entity_embeddings(self, entity_texts):
        try:
            if not entity_texts:
                self.semantic_ent_emb = torch.empty(0, device=get_model_device(self))
                return
            entity_ids = []
            texts = []
            for ent_id, text in entity_texts.items():
                entity_ids.append(ent_id)
                texts.append(text)
            batch_size = 32
            emb_list = []
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                emb = self.bert_encode_entity(batch)
                emb_list.append(emb)
                if hasattr(torch.cuda, 'empty_cache'):
                    torch.cuda.empty_cache()
            if not emb_list:
                self.semantic_ent_emb = torch.empty(0, device=get_model_device(self))
                return
            all_emb = torch.cat(emb_list, dim=0)
            num_entities = self.ent_embeddings.weight.size(0)
            sem = torch.zeros((num_entities, self.hidden_dim), device=all_emb.device, dtype=all_emb.dtype)
            for i, ent_id in enumerate(entity_ids):
                if ent_id < num_entities:
                    sem[ent_id] = all_emb[i]
            self.semantic_ent_emb = sem.detach()
            print(f"Initialized separate semantic embeddings: shape={self.semantic_ent_emb.shape}")
        except Exception as e:
            print(f"Warning: initialize_semantic_entity_embeddings failed: {e}")
            self.semantic_ent_emb = torch.empty(0, device=get_model_device(self))

    def initialize_entity_embeddings(self, entity_texts):
        print("Initializing entity embeddings with BERT...")
        print(f"Model device: {next(self.parameters()).device}")
        print(f"BERT device: {next(self.bert.parameters()).device}")

        if not entity_texts:
            print("Warning: No entity texts provided, skipping BERT initialization")
            return

        entity_ids = []
        entity_texts_list = []

        for ent_id, text in entity_texts.items():
            entity_ids.append(ent_id)
            entity_texts_list.append(text)

        if not entity_ids:
            print("Warning: No entity IDs found, skipping BERT initialization")
            return

        batch_size = 320
        all_embeddings = []

        print(f"Initializing {len(entity_texts_list)} entity embeddings in batches of {batch_size}")

        for i in range(0, len(entity_texts_list), batch_size):
            batch_texts = entity_texts_list[i:i + batch_size]
            batch_embeddings = self.bert_encode_entity(batch_texts)
            all_embeddings.append(batch_embeddings)

            if hasattr(torch.cuda, 'empty_cache'):
                torch.cuda.empty_cache()

        if not all_embeddings:
            print("Warning: No embeddings generated, skipping BERT initialization")
            return

        all_embeddings = torch.cat(all_embeddings, dim=0)

        print(f"BERT embeddings shape: {all_embeddings.shape}")
        print(f"Expected entity embedding shape: {self.ent_embeddings.weight.shape}")

        for i, ent_id in enumerate(entity_ids):
            if ent_id < self.ent_embeddings.weight.size(0):
                self.ent_embeddings.weight.data[ent_id] = all_embeddings[i]

        print(f"Initialized {len(entity_ids)} entity embeddings")

    def initialize_relation_embeddings(self, support_triples, entity_embeddings):
        print("Initializing background relation embeddings from training data...")

        rel_to_triples = {}
        for h, r, t in support_triples:
            if r not in rel_to_triples:
                rel_to_triples[r] = []
            rel_to_triples[r].append((h, t))

        initialized_count = 0
        for rel_id, triples in rel_to_triples.items():
            rel_embeddings = []

            for h, t in triples:
                h_emb = entity_embeddings[h]
                t_emb = entity_embeddings[t]
                rel_emb = t_emb - h_emb
                rel_embeddings.append(rel_emb)

            if rel_embeddings:
                avg_rel_emb = torch.stack(rel_embeddings).mean(dim=0)
                self.rel_embeddings.weight.data[rel_id] = avg_rel_emb
                initialized_count += 1

        total_relations = self.rel_embeddings.weight.size(0)
        uninitialized_count = total_relations - initialized_count

        print(f"Initialized {initialized_count} background relation embeddings from training data")
        print(f"Remaining {uninitialized_count} background relations use Xavier initialization")
        print(f"Total background relations: {total_relations}")
        print("Note: Task relations will be dynamically computed during inference using support sets")

    def build_global_graph_edges(self, path_info=None, max_edges=None, task_entities=None, bert_embeddings=None):
        """
        Build high-relevance subgraph
        Args:
            path_info: path information
            max_edges: max edges, use self.max_subgraph_edges if None
            task_entities: task entity list
            bert_embeddings: BERT embeddings for similarity calculation
        """
        if max_edges is None:
            max_edges = self.max_subgraph_edges


        edges = []
        edge_types = []

        relation_to_temp_id = {}
        temp_id_counter = 0

        if path_info is not None:
            path_items = []
            for key, value in path_info.items():
                if key not in ['cached_entity_emb', 'cached_rel_emb']:
                    path_items.append((key, value))

            if task_entities is not None and len(task_entities) > 0:
                task_entity_set = set(task_entities)

                scored_paths = []
                for item in path_items:
                    try:
                        if len(item) == 2:
                            (h_id, t_id), paths = item
                        else:
                            print(f"Warning: Unexpected path_info format: {item}")
                            continue
                    except (ValueError, TypeError) as e:
                        print(f"Error unpacking path_info item: {item}, error: {e}")
                        continue

                    task_entities_in_pair = 0
                    if h_id in task_entity_set:
                        task_entities_in_pair += 1
                    if t_id in task_entity_set:
                        task_entities_in_pair += 1
                    base_relevance = task_entities_in_pair / 2.0

                    path_length_scores = []
                    for path in paths:
                        if len(path) == 1:
                            path_length_scores.append(1.0)
                        elif len(path) == 2:
                            path_length_scores.append(0.9)
                        else:
                            path_length_scores.append(0.6)
                    avg_path_score = sum(path_length_scores) / len(path_length_scores) if path_length_scores else 0.0

                    bert_similarity = 0.0
                    if bert_embeddings is not None:
                        try:
                            h_emb = bert_embeddings[h_id]
                            t_emb = bert_embeddings[t_id]
                            bert_similarity = torch.cosine_similarity(h_emb.unsqueeze(0), t_emb.unsqueeze(0), dim=1).item()
                        except:
                            bert_similarity = 0.0
                    
                    alpha = 0.2
                    beta = 0.6
                    gamma = 0.2
                    comprehensive_score = base_relevance * alpha + avg_path_score * beta + bert_similarity * gamma

                    scored_paths.append(((h_id, t_id), paths, comprehensive_score, base_relevance, avg_path_score))

                scored_paths.sort(key=lambda x: x[2], reverse=True)

                target_pairs = max_edges // 3
                if len(scored_paths) > 0:

                    two_hop_paths = []
                    other_paths = []

                    for (h_t, paths, comp_score, base_rel, path_score) in scored_paths:
                        has_two_hop = any(len(path) == 2 for path in paths)
                        if has_two_hop:
                            two_hop_paths.append((h_t, paths, comp_score, base_rel, path_score))
                        else:
                            other_paths.append((h_t, paths, comp_score, base_rel, path_score))

                    # 优先选择两跳路径，确保包含所有任务实体的两跳邻居
                    selected_paths = []

                    # 先添加所有两跳路径
                    if len(two_hop_paths) > 0:
                        selected_paths.extend(two_hop_paths[:target_pairs // 2])  
                        if self.debug_mode:
                            print(f"Selected {len(selected_paths)} two-hop paths for task entities")

                    # 再添加其他高质量路径
                    remaining_slots = target_pairs - len(selected_paths)
                    if remaining_slots > 0 and len(other_paths) > 0:
                        selected_paths.extend(other_paths[:remaining_slots])

                    if len(selected_paths) < target_pairs and len(other_paths) > 0:
                        remaining_slots = target_pairs - len(selected_paths)
                        selected_paths.extend(other_paths[len(selected_paths):len(selected_paths) + remaining_slots])

                    # 转换为path_items格式
                    path_items = [(h_t, paths) for h_t, paths, comp_score, base_rel, path_score in selected_paths]

                    if self.debug_mode:
                        two_hop_count = sum(
                            1 for _, paths, _, _, _ in selected_paths if any(len(path) == 2 for path in paths))
                        print(f"Selected {len(selected_paths)} entity pairs (including {two_hop_count} two-hop paths)")
                        print(
                            f"Average comprehensive score: {sum(comp_score for _, _, comp_score, _, _ in selected_paths) / len(selected_paths):.3f}")
                    # print(f"Selected {len(path_items)} highest-relevance entity pairs (avg relevance: {avg_relevance:.3f})")
                else:
                    if len(path_items) > target_pairs:
                        import random
                        random.seed(42)
                        path_items = random.sample(path_items, target_pairs)
                        # print(f"No task-relevant paths found, randomly sampled {len(path_items)} entity pairs")
            else:
                # 没有任务实体信息，使用简单的随机采样（适度放宽采样上限）
                target_pairs = max_edges // 3
                if len(path_items) > target_pairs:
                    import random
                    random.seed(42)
                    path_items = random.sample(path_items, target_pairs)
                    # print(f"Randomly sampled {len(path_items)} entity pairs from {len(path_info)} total pairs")

            for item in path_items:
                try:
                    if len(item) == 2:
                        (h_id, t_id), paths = item
                    else:
                        continue
                except (ValueError, TypeError):
                    continue
                for path in paths:
                    if len(path) >= 1:  
                        for rel_id in path:
                            if rel_id not in relation_to_temp_id:
                                relation_to_temp_id[rel_id] = temp_id_counter
                                temp_id_counter += 1

                        # 构建路径边
                        if len(path) == 1:
                            # 单跳路径：h -> r -> t
                            r_temp_id = relation_to_temp_id[path[0]]
                            edges.append([h_id, r_temp_id])
                            edge_types.append(0)
                            edges.append([r_temp_id, t_id])
                            edge_types.append(1)
                        else:
                            # 多跳路径：h -> r1 -> r2 -> ... -> t
                            prev_node = h_id
                            for i, rel_id in enumerate(path):
                                r_temp_id = relation_to_temp_id[rel_id]
                                if i == 0:
                                    edges.append([prev_node, r_temp_id])
                                    edge_types.append(0)
                                else:
                                    edges.append([prev_node, r_temp_id])
                                    edge_types.append(2)  # 路径连接

                                if i == len(path) - 1:
                                    edges.append([r_temp_id, t_id])
                                    edge_types.append(1)
                                else:
                                    prev_node = r_temp_id

                        if len(edges) >= max_edges:
                            # print(f"Reached edge limit ({max_edges}), stopping graph construction")
                            break
                if len(edges) >= max_edges:
                    break

        # 如果没有path_info或边数太少
        if len(edges) < max_edges // 2:  # 使用动态阈值而不是硬编码2000
            # print("Building simple global graph as fallback")
            for rel_id in range(min(self.n_rel, 200)):
                relation_to_temp_id[rel_id] = rel_id

            for i in range(0, min(10000, self.n_ent), 2):
                if i + 1 < self.n_ent:
                    rel_id = i % min(self.n_rel, 25)
                    edges.append([i, rel_id])
                    edge_types.append(0)
                    edges.append([rel_id, i + 1])
                    edge_types.append(1)

                    if len(edges) >= max_edges:
                        break

        if edges:
            edge_tensor = torch.tensor(edges, dtype=torch.long).t()
            edge_types_tensor = torch.tensor(edge_types, dtype=torch.long)
        else:
            # 如果没有边，创建空的张量
            edge_tensor = torch.empty((2, 0), dtype=torch.long)
            edge_types_tensor = torch.empty((0,), dtype=torch.long)

        # 输出图构建信息（只在第一次）
        if not hasattr(self, '_global_graph_debug_printed'):
            print(f"\n=== Graph Construction Info ===")
            print(f"Built task-relevant subgraph with {len(edges)} edges (limited to {max_edges})")
            print(f"Edge tensor shape: {edge_tensor.shape}")
            print(f"Number of edges: {edge_tensor.size(1)}")
            print(f"Edge types: {set(edge_types)}")
            print(f"Unique relations: {len(relation_to_temp_id)}")
            print(f"=== End Graph Construction Info ===\n")
            self._global_graph_debug_printed = True

        return edge_tensor, edge_types_tensor

    def forward_gat(self, entity_embeddings, relation_embeddings, edge_index, edge_types):
        """Improve embeddings through GAT layers - supports dynamic relation embeddings"""
        # print(f"\n=== GAT Debug Info ===")
        # print(f"GAT enabled: {self.use_gat}")
        # print(f"Number of GAT layers: {len(self.gat_layers) if self.use_gat else len(self.mlp_layers)}")
        # print(f"Edge index shape: {edge_index.shape}")
        # print(f"Number of edges: {edge_index.size(1)}")
        # print(f"Entity embeddings shape: {entity_embeddings.shape}")
        # print(f"Relation embeddings type: {type(relation_embeddings)}")
        
        if isinstance(relation_embeddings, dict):
            rel_emb_list = list(relation_embeddings.values())
            if rel_emb_list:
                max_rel_id = len(relation_embeddings)
                rel_emb_tensor = torch.zeros(max_rel_id, entity_embeddings.size(-1),
                                             device=entity_embeddings.device)
                for i, emb in enumerate(rel_emb_list):
                    if i < max_rel_id:
                        rel_emb_tensor[i] = emb
                relation_embeddings = rel_emb_tensor
            else:
                relation_embeddings = torch.zeros(self.n_rel, entity_embeddings.size(-1),
                                                  device=entity_embeddings.device)

        all_embeddings = torch.cat([entity_embeddings, relation_embeddings], dim=0)

        if self.training:
            all_embeddings = F.normalize(all_embeddings, p=2, dim=-1)

        if self.debug_mode and not hasattr(self, '_forward_gat_debug_printed'):
            print(
                f"Input embeddings: all_mean={all_embeddings.mean().item():.4f}, all_std={all_embeddings.std().item():.4f}")
            self._forward_gat_debug_printed = True

        x = all_embeddings

        if self.use_gat:
            # print(f"Using GAT layers for embedding improvement...")
            for i, gat_layer in enumerate(self.gat_layers):
                # print(f"GAT Layer {i + 1}/{len(self.gat_layers)}:")
                # print(f"  Input shape: {x.shape}")
                # print(f"  Input mean: {x.mean().item():.4f}, std: {x.std().item():.4f}")
                # print(f"  Edge index: {edge_index.shape} with {edge_index.size(1)} edges")

                x = gat_layer(x, edge_index)

                # print(f"  Output shape: {x.shape}")
                # print(f"  Output mean: {x.mean().item():.4f}, std: {x.std().item():.4f}")
                # print(f"  Output norm: {x.norm().item():.4f}")
        else:
            # print(f"Using MLP layers for embedding improvement...")
            for i, mlp_layer in enumerate(self.mlp_layers):
                # print(f"MLP Layer {i + 1}/{len(self.mlp_layers)}:")
                # print(f"  Input shape: {x.shape}")
                # print(f"  Input mean: {x.mean().item():.4f}, std: {x.std().item():.4f}")

                x = mlp_layer(x)

                # print(f"  Output shape: {x.shape}")
                # print(f"  Output mean: {x.mean().item():.4f}, std: {x.std().item():.4f}")
                # print(f"  Output norm: {x.norm().item():.4f}")

        n_entities = entity_embeddings.size(0)
        n_relations = relation_embeddings.size(0)
        improved_entity_emb = x[:n_entities]
        improved_rel_emb = x[n_entities:n_entities + n_relations]

        # print(f"Final GAT output:")
        # print(f"  Entity embeddings shape: {improved_entity_emb.shape}")
        # print(f"  Relation embeddings shape: {improved_rel_emb.shape}")
        # print(f"  Entity mean: {improved_entity_emb.mean().item():.4f}, std: {improved_entity_emb.std().item():.4f}")
        # print(f"  Relation mean: {improved_rel_emb.mean().item():.4f}, std: {improved_rel_emb.std().item():.4f}")

        structural_entity_emb = self.structural_entity_embeddings.weight
        
        adaptive_gates = self.compute_structural_semantic_gates(
            structural_entity_emb, improved_entity_emb, 
            relation_embeddings, improved_rel_emb)
        structural_gate_adaptive = adaptive_gates[0]
        semantic_gate_adaptive = adaptive_gates[1]
        
        structural_gate = torch.clamp(self.structural_gate + structural_gate_adaptive, 0.0, 1.0)
        semantic_gate = torch.clamp(self.semantic_gate + semantic_gate_adaptive, 0.0, 1.0)
        
        improved_entity_emb = structural_gate * structural_entity_emb + semantic_gate * improved_entity_emb
        
        if isinstance(relation_embeddings, dict):
            improved_rel_emb = semantic_gate * improved_rel_emb
        else:
            structural_rel_emb = self.structural_rel_embeddings.weight
            improved_rel_emb = structural_gate * structural_rel_emb + semantic_gate * improved_rel_emb

        # print(f"After gating (entity_gate={gate_e.item():.3f}, rel_gate={gate_r.item():.3f}):")
        # print(f"  Final entity mean: {improved_entity_emb.mean().item():.4f}, std: {improved_entity_emb.std().item():.4f}")
        # print(f"  Final relation mean: {improved_rel_emb.mean().item():.4f}, std: {improved_rel_emb.std().item():.4f}")
        # print(f"=== End GAT Debug Info ===\n")

        if self.debug_mode and not hasattr(self, '_improved_emb_debug_printed'):
            print(
                f"Improved embeddings stats: entity_mean={improved_entity_emb.mean().item():.4f}, rel_mean={improved_rel_emb.mean().item():.4f}")
            self._improved_emb_debug_printed = True

        return improved_entity_emb, improved_rel_emb

    def compute_structural_semantic_gates(self, structural_entity_emb, semantic_entity_emb, 
                                         relation_embeddings, semantic_rel_emb):
        """Compute structural-semantic adaptive gating weights"""
        entity_diff = torch.norm(semantic_entity_emb - structural_entity_emb, dim=-1, keepdim=True)
        entity_avg_diff = entity_diff.mean()
        
        if isinstance(relation_embeddings, dict) or isinstance(semantic_rel_emb, dict):
            rel_avg_diff = torch.tensor(0.0, device=structural_entity_emb.device)
        else:
            structural_rel_emb = self.structural_rel_embeddings.weight
            if semantic_rel_emb.size(0) == structural_rel_emb.size(0):
                rel_diff = torch.norm(semantic_rel_emb - structural_rel_emb, dim=-1, keepdim=True)
                rel_avg_diff = rel_diff.mean()
            else:
                rel_avg_diff = torch.tensor(0.0, device=structural_entity_emb.device)
        
        structural_mean = structural_entity_emb.mean(dim=0)
        semantic_mean = semantic_entity_emb.mean(dim=0)
        
        input_features = torch.cat([
            structural_mean,
            semantic_mean
        ], dim=-1)
        
        if input_features.size(-1) < self.hidden_dim * 2:
            padding = torch.zeros(self.hidden_dim * 2 - input_features.size(-1), 
                                device=structural_entity_emb.device)
            input_features = torch.cat([input_features, padding], dim=-1)
        elif input_features.size(-1) > self.hidden_dim * 2:
            input_features = input_features[:self.hidden_dim * 2]
        
        input_features = input_features.unsqueeze(0).expand(structural_entity_emb.size(0), -1)
        
        adaptive_weights = self.adaptive_gate_net(input_features)
        
        structural_gate_adaptive = adaptive_weights[:, 0].mean()
        semantic_gate_adaptive = adaptive_weights[:, 1].mean()
        
        return structural_gate_adaptive, semantic_gate_adaptive

    def load_structural_embeddings(self):
        try:
            if hasattr(self.data_loader, 'transE_loaded') and self.data_loader.transE_loaded:
                transE_embeddings = self.data_loader.structural_entity_embeddings
                transE_entity2id = self.data_loader.transE_entity2id
                
                if transE_embeddings is not None and transE_entity2id is not None:
                    embeddings = torch.from_numpy(transE_embeddings).float()
                    
                    loaded_count = 0
                    for entity, entity_id in transE_entity2id.items():
                        if entity in self.data_loader.ent2id:
                            model_entity_id = self.data_loader.ent2id[entity]
                            if model_entity_id < self.structural_entity_embeddings.weight.size(0):
                                if embeddings.size(-1) >= self.hidden_dim:
                                    self.structural_entity_embeddings.weight.data[model_entity_id] = embeddings[entity_id][:self.hidden_dim]
                                else:
                                    temp = torch.zeros(self.hidden_dim)
                                    temp[:embeddings.size(-1)] = embeddings[entity_id]
                                    self.structural_entity_embeddings.weight.data[model_entity_id] = temp
                                loaded_count += 1
                    
                    print(f"Loaded TransE structural embeddings: {loaded_count} entities")
                else:
                    print("TransE embeddings not available, using Xavier initialization")
            else:
                print("TransE not loaded, using Xavier initialization")
        except Exception as e:
            print(f"Error loading structural embeddings: {e}")
            print("Using Xavier initialization")


    def compute_improved_score(self, h_emb, r_emb, t_emb):
        """TransE scoring function"""
        return -torch.norm(h_emb + r_emb - t_emb, p=2)

    def forward(self, support_triples, query_triples, mode='training', path_info=None):
        """
        Forward pass - improved scoring strategy
        Args:
            support_triples: support triples [(h, r, t), ...] where r can be string or ID
            query_triples: query triples [(h, r, t), ...] where r can be string or ID
            mode: 'training' or 'inference'
            path_info: path information dict
        """
        device = next(self.parameters()).device

        cached_entity_emb = None
        cached_rel_emb = None
        if isinstance(path_info, dict):
            cached_entity_emb = path_info.get('cached_entity_emb', None)
            cached_rel_emb = path_info.get('cached_rel_emb', None)

        entity_embeddings = self.structural_entity_embeddings.weight

        relation_embeddings = self.compute_dynamic_relation_embeddings(support_triples, entity_embeddings)

        task_entities = set()
        for h, r, t in support_triples + query_triples:
            task_entities.add(h)
            task_entities.add(t)

        bert_entity_embeddings = self.ent_embeddings.weight
        edge_index, edge_types = self.build_global_graph_edges(
            path_info, max_edges=self.max_subgraph_edges, task_entities=list(task_entities), 
            bert_embeddings=bert_entity_embeddings)
        edge_index = edge_index.to(device)
        edge_types = edge_types.to(device)

        if cached_entity_emb is not None and cached_rel_emb is not None:
            improved_entity_emb, improved_rel_emb = cached_entity_emb, cached_rel_emb
        else:
            improved_entity_emb, improved_rel_emb = self.forward_gat(
                entity_embeddings, relation_embeddings, edge_index, edge_types)


        query_scores = []
        for h, r, t in query_triples:
            try:
                if isinstance(h, str):
                    h_int = getattr(self, 'ent2id', {}).get(h, 0)
                else:
                    h_int = int(h)

                if isinstance(t, str):
                    t_int = getattr(self, 'ent2id', {}).get(t, 0)
                else:
                    t_int = int(t)

                if h_int >= improved_entity_emb.size(0) or t_int >= improved_entity_emb.size(0):
                    h_emb = torch.zeros(improved_entity_emb.size(-1), device=improved_entity_emb.device,
                                        dtype=improved_entity_emb.dtype, requires_grad=True)
                    t_emb = torch.zeros(improved_entity_emb.size(-1), device=improved_entity_emb.device,
                                        dtype=improved_entity_emb.dtype, requires_grad=True)
                else:
                    h_emb = improved_entity_emb[h_int]
                    t_emb = improved_entity_emb[t_int]
            except (ValueError, TypeError, KeyError) as e:
                h_emb = torch.zeros(improved_entity_emb.size(-1), device=improved_entity_emb.device,
                                    dtype=improved_entity_emb.dtype, requires_grad=True)
                t_emb = torch.zeros(improved_entity_emb.size(-1), device=improved_entity_emb.device,
                                    dtype=improved_entity_emb.dtype, requires_grad=True)

            current_support = [(sh, st) for (sh, sr, st) in support_triples if sr == r]

            if getattr(self, 'debug_fkgc', False):
                try:
                    print(f"[FKGC] mode={mode}, relation={r}, support_size={len(current_support)}")
                except Exception:
                    pass

            if self.scoring_paradigm == 'fkgc':
                if len(current_support) == 0:
                    score = torch.tensor(0.0, device=h_emb.device, dtype=h_emb.dtype, requires_grad=True)
                else:
                    query_relation_rep = compute_query_relation_representation(h_emb, t_emb)

                    relation_prototype = compute_hybrid_prototype_cached(
                        self, support_triples, improved_entity_emb, r, query_relation_rep,
                        self.hidden_dim, self.fkgc_attention_temp, self.fkgc_prototype_shrinkage,
                        self.fkgc_attention_weight
                    )

                    if getattr(self, 'debug_fkgc', False):
                        try:
                            print(
                                f"[FKGC] ||proto||={relation_prototype.norm(p=2).item():.4f}, ||query||={query_relation_rep.norm(p=2).item():.4f}")
                        except Exception:
                            pass
                    score = compute_fkgc_score(query_relation_rep, relation_prototype, self.fkgc_attention_temp)
                    if getattr(self, 'debug_fkgc', False):
                        try:
                            print(f"[FKGC] score={score.item():.4f}")
                        except Exception:
                            pass
            else:
                if len(current_support) == 0:
                    score = self.compute_improved_score(h_emb, torch.zeros_like(h_emb), t_emb)
                else:
                    if isinstance(r, str):
                        try:
                            if r in relation_embeddings:
                                r_emb = relation_embeddings[r]
                            else:
                                r_emb = self.get_task_relation_embedding(r, support_triples, improved_entity_emb)
                        except Exception as rel_e:
                            r_emb = torch.zeros(improved_entity_emb.size(-1), device=improved_entity_emb.device)
                    else:
                        try:
                            if r >= improved_rel_emb.size(0):
                                r_emb = torch.zeros(improved_entity_emb.size(-1), device=improved_entity_emb.device)
                            else:
                                r_emb = improved_rel_emb[r]
                        except Exception as rel_e:
                            r_emb = torch.zeros(improved_entity_emb.size(-1), device=improved_entity_emb.device)
                    
                    score = self.compute_improved_score(h_emb, r_emb, t_emb)
            
            query_scores.append(score)

        if len(query_scores) == 0:
            print("Warning: No valid scores computed, returning empty tensor")
            return torch.empty(0, device=next(self.parameters()).device, dtype=improved_entity_emb.dtype,
                               requires_grad=True)

        query_scores = torch.stack(query_scores)

        if query_scores.dim() > 1:
            query_scores = query_scores.squeeze()

        if mode == 'training':
            return query_scores
        else:
            return improved_entity_emb, improved_rel_emb, query_scores

    def compute_margin_ranking_loss(self, pos_scores, neg_scores, margin=1.0):
        """
        Implement Margin-based Ranking Loss from paper formula 17
        ℒ = ∑_r [ γ + (1/m) ∑_{i=1}^m ϕ_- - (1/n) ∑_{j=1}^n ϕ_+ ]
        Args:
            pos_scores: positive sample scores [n]
            neg_scores: negative sample scores [m]
            margin: margin value γ
        Returns:
            Margin-based Ranking Loss
        """
        if not isinstance(pos_scores, torch.Tensor):
            pos_scores = torch.tensor(pos_scores, device=next(self.parameters()).device,
                                      dtype=next(self.parameters()).dtype, requires_grad=True)
        if not isinstance(neg_scores, torch.Tensor):
            neg_scores = torch.tensor(neg_scores, device=next(self.parameters()).device,
                                      dtype=next(self.parameters()).dtype, requires_grad=True)

        pos_mean = pos_scores.mean()
        neg_mean = neg_scores.mean()

        loss = margin + neg_mean - pos_mean

        loss = torch.relu(loss)

        return loss

    def compute_dynamic_relation_embeddings(self, support_triples, entity_embeddings, path_info=None):
        relations = set()
        for h, r, t in support_triples:
            relations.add(r)

        relation_embeddings = {}
        for r in relations:
            if isinstance(r, str):
                deltas = []
                path_bias_list = []
                for h_id, r_name, t_id in support_triples:
                    if r_name == r:
                        deltas.append(entity_embeddings[t_id] - entity_embeddings[h_id])
                        if path_info is not None and (h_id, t_id) in path_info:
                            paths = path_info.get((h_id, t_id), [])
                            has_len1 = 1.0 if any(len(p) == 1 for p in paths) else 0.0
                            has_len2 = 1.0 if any(len(p) == 2 for p in paths) else 0.0
                            count = float(len(paths))
                            feat = torch.tensor([has_len1, has_len2, torch.log1p(torch.tensor(count))],
                                                device=entity_embeddings.device, dtype=entity_embeddings.dtype)
                            path_bias_list.append(self.path_feat_linear(feat).squeeze(-1))
                        else:
                            path_bias_list.append(
                                torch.tensor(0.0, device=entity_embeddings.device, dtype=entity_embeddings.dtype))
                if len(deltas) > 0:
                    deltas_tensor = torch.stack(deltas, dim=0)
                    attn_logits = self.relation_attn(deltas_tensor).squeeze(-1)
                    if path_bias_list:
                        path_bias = torch.stack(path_bias_list, dim=0)
                        attn_logits = attn_logits + self.path_bias_scale * path_bias
                    attn = F.softmax(attn_logits, dim=0)
                    r_emb = torch.sum(attn.unsqueeze(-1) * deltas_tensor, dim=0)
                    r_emb = r_emb + 0.01 * torch.randn_like(r_emb)
                else:
                    r_emb = 0.1 * torch.randn(self.hidden_dim, device=entity_embeddings.device)
                relation_embeddings[r] = r_emb
            else:
                relation_embeddings[r] = self.rel_embeddings.weight[r]

        return relation_embeddings

    def compute_task_relation_embedding(self, relation_name, support_triples, entity_embeddings):
        relation_support = []
        for h, r, t in support_triples:
            if r == relation_name:
                relation_support.append((h, t))

        if not relation_support:
            return torch.randn(entity_embeddings.size(-1), device=entity_embeddings.device)

        rel_embeddings = []
        for h, t in relation_support:
            h_emb = entity_embeddings[h]
            t_emb = entity_embeddings[t]
            rel_emb = t_emb - h_emb
            rel_embeddings.append(rel_emb)

        return torch.stack(rel_embeddings).mean(dim=0)

    def get_task_relation_embedding(self, relation_name, support_triples, entity_embeddings):
        """Get task relation embedding (for query time)"""
        return self.compute_task_relation_embedding(relation_name, support_triples, entity_embeddings)


class FKGCDataLoaderWithText(FKGCDataLoader):
    """Extended FKGC data loader with text description support"""

    def __init__(self, in_paths, tokenizer, batch_size=2, max_desc_length=64,
                 model='bert', sp_num=5):
        super().__init__(in_paths, tokenizer, batch_size, max_desc_length,
                         model, sp_num)

        self.entity_texts = {}
        self.relation_texts = {}
        self.build_text_dicts()

    def build_text_dicts(self):
        """Build entity and relation text dictionaries"""
        for uid, text in self.uid2text.items():
            if uid in self.ent2id:
                self.entity_texts[self.ent2id[uid]] = text
            elif uid in self.rel2id:
                self.relation_texts[self.rel2id[uid]] = text

    def get_entity_texts(self):
        """Get all entity text descriptions"""
        return self.entity_texts

    def get_relation_texts(self):
        """Get all relation text descriptions"""
        return self.relation_texts


def train_fkgc_model(*args, **kwargs):
    raise NotImplementedError("train_fkgc_model is deprecated; use main_fkgc.py trainer instead.")


def train_classifier_only(*args, **kwargs):
    raise NotImplementedError("Classifier path removed; scoring uses relation attention now.")


# ==================== FKGC Paradigm Methods ====================

def compute_relation_prototype(support_triples, improved_entity_emb, relation_id, hidden_dim):
    """Compute relation prototype"""
    current_support = [(h, t) for (h, r, t) in support_triples if r == relation_id]
    if len(current_support) == 0:
        return torch.zeros(hidden_dim, device=improved_entity_emb.device)

    support_relations = []
    for h, t in current_support:
        if h < improved_entity_emb.size(0) and t < improved_entity_emb.size(0):
            h_emb = improved_entity_emb[h]
            t_emb = improved_entity_emb[t]
            rel_rep = t_emb - h_emb
            support_relations.append(rel_rep)

    if len(support_relations) == 0:
        return torch.zeros(hidden_dim, device=improved_entity_emb.device)

    return torch.stack(support_relations).mean(dim=0)


def compute_hybrid_prototype_cached(model, support_triples, improved_entity_emb, relation_id,
                                    query_relation_rep, hidden_dim, temperature=1.0, shrinkage=0.1,
                                    attention_weight=0.7):
    """Hybrid prototype computation"""
    cache_key = f"{relation_id}_{hash(str(support_triples))}_{hash(str(improved_entity_emb.data_ptr()))}"

    if hasattr(model, 'cache_enabled') and model.cache_enabled and cache_key in model.prototype_cache:
        cached_prototype, cached_query_rep = model.prototype_cache[cache_key]
        if torch.allclose(query_relation_rep, cached_query_rep, atol=1e-6):
            return cached_prototype

    simple_prototype = compute_relation_prototype(support_triples, improved_entity_emb, relation_id, hidden_dim)
    attention_prototype = compute_attention_weighted_prototype(
        support_triples, improved_entity_emb, relation_id, query_relation_rep, hidden_dim, temperature, shrinkage
    )

    hybrid_prototype = attention_weight * attention_prototype + (1 - attention_weight) * simple_prototype

    if hasattr(model, 'cache_enabled') and model.cache_enabled:
        model.prototype_cache[cache_key] = (hybrid_prototype.clone(), query_relation_rep.clone())
        if len(model.prototype_cache) > 1000:
            oldest_key = next(iter(model.prototype_cache))
            del model.prototype_cache[oldest_key]

    return hybrid_prototype


def compute_attention_weighted_prototype(support_triples, improved_entity_emb, relation_id,
                                         query_relation_rep, hidden_dim, temperature=1.0, shrinkage=0.1):
    """Attention-weighted prototype computation - optimized version"""
    current_support = [(h, t) for (h, r, t) in support_triples if r == relation_id]
    if len(current_support) == 0:
        return torch.zeros(hidden_dim, device=improved_entity_emb.device)

    valid_support = [(h, t) for h, t in current_support if
                     h < improved_entity_emb.size(0) and t < improved_entity_emb.size(0)]

    if len(valid_support) == 0:
        return torch.zeros(hidden_dim, device=improved_entity_emb.device)

    h_indices = torch.tensor([h for h, t in valid_support], device=improved_entity_emb.device)
    t_indices = torch.tensor([t for h, t in valid_support], device=improved_entity_emb.device)

    h_embs = improved_entity_emb[h_indices]
    t_embs = improved_entity_emb[t_indices]
    support_relations_tensor = t_embs - h_embs

    similarities = torch.cosine_similarity(query_relation_rep.unsqueeze(0), support_relations_tensor, dim=1)
    scaled_similarities = similarities / temperature
    attention_weights = F.softmax(scaled_similarities, dim=0)

    attention_prototype = torch.sum(attention_weights.unsqueeze(1) * support_relations_tensor, dim=0)
    return (1 - shrinkage) * attention_prototype


def compute_query_relation_representation(h_emb, t_emb):
    """Compute query relation representation"""
    return t_emb - h_emb


def compute_fkgc_score(query_relation_rep, relation_prototype, temperature=1.0):
    """FKGC scoring"""
    similarity = torch.cosine_similarity(query_relation_rep.unsqueeze(0), relation_prototype.unsqueeze(0), dim=1)
    return similarity / temperature