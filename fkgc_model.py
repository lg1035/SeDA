import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch_geometric.nn import GATConv
import math
import copy
from transformers import AutoModel, AutoTokenizer
from lora_utils import LoRAConfig, apply_lora_to_model, get_lora_parameters
from dataloader import FKGCDataLoader


def get_model_device(model):
    if hasattr(model, 'module'):
        # DataParallel
        return next(model.module.parameters()).device
    else:
        return next(model.parameters()).device


class GATLayer(nn.Module):
    """图注意力网络层"""
    def __init__(self, in_dim, out_dim, num_heads=8, dropout=0.1, concat=True):
        super(GATLayer, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.concat = concat
        
        # 线性变换层
        self.W = nn.Linear(in_dim, out_dim * num_heads, bias=False)
        
        # 注意力机制
        self.attention = nn.Parameter(torch.Tensor(1, num_heads, out_dim * 2))
        
        # 偏置项 - 由于使用平均池化，输出维度是out_dim
        self.bias = nn.Parameter(torch.Tensor(out_dim))
        
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.attention)
        nn.init.zeros_(self.bias)
    
    def forward(self, x, edge_index):
        """
        Args:
            x: 节点特征 [N, in_dim]
            edge_index: 边索引 [2, E]
        """
        N = x.size(0)
        E = edge_index.size(1)
        
        # 调试信息
        if not hasattr(self, '_gat_debug_printed'):
            print(f"GAT input: N={N}, E={E}, x_shape={x.shape}, x_mean={x.mean().item():.4f}, x_std={x.std().item():.4f}")
            print(f"Edge index shape: {edge_index.shape}")
            self._gat_debug_printed = True
        
        # 线性变换
        x = self.W(x)  # [N, out_dim * num_heads]
        x = x.view(N, self.num_heads, -1)  # [N, num_heads, out_dim]
        
        # 处理空的边索引
        if E == 0:
            if not hasattr(self, '_no_edges_warning'):
                print("Warning: No edges found, returning zero output")
                self._no_edges_warning = True
            # 如果没有边，返回零填充的输出
            out = torch.zeros(N, self.num_heads, self.out_dim, device=x.device)
            # 使用平均池化，输出维度是out_dim
            out = out.mean(dim=1)  # [N, out_dim]
            out = out + self.bias
            return out

        batch_size = 2000  # 每批处理的边数
        if E > batch_size:
            if not hasattr(self, '_batch_processing_debug_printed'):
                print(f"Large graph detected (E={E}), using batch processing with batch_size={batch_size}")
                self._batch_processing_debug_printed = True
            
            # 分批处理边，累积中间结果
            accumulated_out = torch.zeros(N, self.num_heads, self.out_dim, device=x.device)
            batch_count = 0
            
            for i in range(0, E, batch_size):
                end_idx = min(i + batch_size, E)
                batch_edge_index = edge_index[:, i:end_idx]
                
                # 处理当前批次，返回中间结果
                batch_out = self._process_edge_batch_intermediate(x, batch_edge_index)
                accumulated_out += batch_out
                batch_count += 1
            
            # 对所有批次结果进行平均池化
            out = accumulated_out.mean(dim=1)  # [N, out_dim]
            out = out + self.bias
            return out
        else:
            # 小图直接处理
            return self._process_edge_batch(x, edge_index)
    
    def _process_edge_batch_intermediate(self, x, edge_index):
        """处理一批边，返回中间结果（不进行平均池化）"""
        N = x.size(0)
        E = edge_index.size(1)
        
        # 准备源节点和目标节点特征
        row, col = edge_index
        x_i = x[row]  # [E, num_heads, out_dim]
        x_j = x[col]  # [E, num_heads, out_dim]
        
        # 计算注意力分数
        alpha_input = torch.cat([x_i, x_j], dim=-1)  # [E, num_heads, out_dim * 2]
        alpha = (alpha_input * self.attention).sum(dim=-1)  # [E, num_heads]
        alpha = F.leaky_relu(alpha, negative_slope=0.2)
        
        # 应用softmax归一化（按目标节点分组）
        alpha = alpha.view(-1, self.num_heads)  # [E, num_heads]
        
        # 按目标节点分组进行softmax
        unique_targets = torch.unique(row)
        alpha_normalized = torch.zeros_like(alpha)
        
        for target in unique_targets:
            mask = (row == target)
            if mask.sum() > 0:
                alpha_group = alpha[mask]
                alpha_normalized[mask] = F.softmax(alpha_group, dim=0)
        
        alpha = alpha_normalized
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        
        # 重新组织alpha以匹配特征维度
        alpha = alpha.unsqueeze(-1)  # [E, num_heads, 1]
        
        # 聚合邻居信息
        weighted_features = alpha * x_j  # [E, num_heads, out_dim]

        out = torch.zeros(N, self.num_heads, self.out_dim, device=x.device)
        out.index_add_(0, row, weighted_features)
        
        # 返回中间结果，不进行平均池化
        return out
    
    def _process_edge_batch(self, x, edge_index):
        """处理一批边"""
        N = x.size(0)
        E = edge_index.size(1)

        row, col = edge_index
        x_i = x[row]  # [E, num_heads, out_dim]
        x_j = x[col]  # [E, num_heads, out_dim]
        
        # 计算注意力分数
        alpha_input = torch.cat([x_i, x_j], dim=-1)  # [E, num_heads, out_dim * 2]
        alpha = (alpha_input * self.attention).sum(dim=-1)  # [E, num_heads]
        alpha = F.leaky_relu(alpha, negative_slope=0.2)

        alpha = alpha.view(-1, self.num_heads)  # [E, num_heads]
        
        # 按目标节点分组进行softmax
        unique_targets = torch.unique(row)
        alpha_normalized = torch.zeros_like(alpha)
        
        for target in unique_targets:
            mask = (row == target)
            if mask.sum() > 0:
                alpha_group = alpha[mask]
                alpha_normalized[mask] = F.softmax(alpha_group, dim=0)
        
        alpha = alpha_normalized
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        
        # 重新组织alpha以匹配特征维度
        alpha = alpha.unsqueeze(-1)  # [E, num_heads, 1]
        
        # 聚合邻居信息
        weighted_features = alpha * x_j  # [E, num_heads, out_dim]
        
        # 聚合到目标节点
        out = torch.zeros(N, self.num_heads, self.out_dim, device=x.device)
        out.index_add_(0, row, weighted_features)
        
        # 使用平均池化
        out = out.mean(dim=1)  # [N, out_dim]
        out = out + self.bias
        
        return out

class FKGCWithGAT(nn.Module):
    """基于BERT初始化和GAT改进的FKGC模型"""
    
    def __init__(self, bert_model_name, n_ent, n_rel, hidden_dim=768, 
                 gat_heads=8, gat_layers=2, dropout=0.1, add_tokens=False,
                 use_lora=False, lora_config=None, data_loader=None, max_subgraph_edges=2000):
        """
        初始化FKGC模型
        Args:
            bert_model_name: BERT模型名称
            n_ent: 实体数量
            n_rel: 关系数量
            hidden_dim: 隐藏层维度
            gat_heads: GAT注意力头数
            gat_layers: GAT层数
            dropout: Dropout率
            add_tokens: 是否添加特殊token
            use_lora: 是否使用LoRA
            lora_config: LoRA配置
            data_loader: 数据加载器
            max_subgraph_edges: 子图最大边数，默认2000
        """
        super(FKGCWithGAT, self).__init__()
        
        # 保存配置参数
        self.bert_model_name = bert_model_name
        self.n_ent = n_ent
        self.n_rel = n_rel
        self.hidden_dim = hidden_dim
        self.gat_heads = gat_heads
        self.gat_layers = gat_layers
        self.dropout = dropout
        self.add_tokens = add_tokens
        self.use_lora = use_lora
        self.lora_config = lora_config
        self.data_loader = data_loader
        self.max_subgraph_edges = max_subgraph_edges  #子图大小配置
        
        # print(f"FKGC Model initialized with max_subgraph_edges={max_subgraph_edges}")
        
        # 初始化BERT tokenizer和模型
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(bert_model_name)
            if use_lora and lora_config:
                # 初始化BERT
                self.bert = AutoModel.from_pretrained(bert_model_name)
                # LoRA的配置
                print("Using LoRA configuration")
            else:
                self.bert = AutoModel.from_pretrained(bert_model_name)
            print(f"BERT model loaded: {bert_model_name}")
        except Exception as e:
            print(f"Error loading BERT model: {e}")
            raise
        
        # 实体和关系嵌入层
        self.ent_embeddings = nn.Embedding(n_ent, hidden_dim)
        self.rel_embeddings = nn.Embedding(n_rel, hidden_dim)

        if add_tokens:
            self.ent_tokens = nn.Embedding(n_ent, hidden_dim)
            self.rel_tokens = nn.Embedding(n_rel, hidden_dim)
        
        # GAT层
        self.gat_layers = nn.ModuleList()
        for i in range(gat_layers):
            if i == 0:
                in_dim = hidden_dim
                out_dim = hidden_dim
            else:
                in_dim = hidden_dim
                out_dim = hidden_dim
            self.gat_layers.append(GATLayer(in_dim, out_dim, gat_heads, dropout, concat=False))
        
        # 输出投影层 - 从GAT输出维度投影到hidden_dim
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # 添加残差连接的权重
        self.residual_weight = nn.Parameter(torch.tensor(0.1))  # 可学习的残差权重
        
        # 分类器
        self.classifier = nn.Linear(hidden_dim, 2)  # 二分类：正确/错误
        
        # 相似度计算层
        self.similarity_layer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # 初始化参数
        self.init_embeddings()
    
    def init_embeddings(self):
        """初始化嵌入"""
        # 实体嵌入使用Xavier初始化
        nn.init.xavier_uniform_(self.ent_embeddings.weight)
        nn.init.xavier_uniform_(self.rel_embeddings.weight)
        
        if self.add_tokens:
            nn.init.xavier_uniform_(self.ent_tokens.weight)
            nn.init.xavier_uniform_(self.rel_tokens.weight)
    
    def bert_encode_entity(self, entity_texts):
        """使用BERT编码实体文本"""
        try:
            # 对实体文本进行tokenize
            inputs = self.tokenizer(entity_texts, 
                                   padding=True, 
                                   truncation=True, 
                                   max_length=64,
                                   return_tensors='pt')
            
            # 获取正确的设备（支持DataParallel）
            device = get_model_device(self)
            
            # 移动到正确的设备
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # BERT编码
            if self.use_lora:
                # LoRA模式下允许梯度计算
                outputs = self.bert(**inputs)
            else:
                # 非LoRA模式下冻结梯度
                with torch.no_grad():
                    outputs = self.bert(**inputs)
            
            # 使用[CLS] token的表示作为实体表示
            entity_embeddings = outputs.last_hidden_state[:, 0, :]  # [batch_size, hidden_dim]
            
            # 清理中间变量以节省内存
            del inputs, outputs
            
            return entity_embeddings
            
        except Exception as e:
            print(f"Error in BERT encoding: {e}")
            # 返回零嵌入作为后备
            device = get_model_device(self)
            return torch.zeros(len(entity_texts), self.hidden_dim, device=device)
    
    def initialize_entity_embeddings(self, entity_texts):
        """使用BERT初始化实体嵌入"""
        print("Initializing entity embeddings with BERT...")
        
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
        
        # 分批处理以避免内存问题
        batch_size = 4  # 多卡时进一步减少批处理大小
        all_embeddings = []
        
        print(f"Initializing {len(entity_texts_list)} entity embeddings in batches of {batch_size}")
        
        for i in range(0, len(entity_texts_list), batch_size):
            batch_texts = entity_texts_list[i:i+batch_size]
            batch_embeddings = self.bert_encode_entity(batch_texts)
            all_embeddings.append(batch_embeddings)
            
            # 清理GPU内存
            if hasattr(torch.cuda, 'empty_cache'):
                torch.cuda.empty_cache()
        
        if not all_embeddings:
            print("Warning: No embeddings generated, skipping BERT initialization")
            return
        
        # 合并所有嵌入
        all_embeddings = torch.cat(all_embeddings, dim=0)
        
        # 更新实体嵌入层
        for i, ent_id in enumerate(entity_ids):
            if ent_id < self.ent_embeddings.weight.size(0):  # 确保ID在范围内
                self.ent_embeddings.weight.data[ent_id] = all_embeddings[i]
        
        print(f"Initialized {len(entity_ids)} entity embeddings")
    
    def initialize_relation_embeddings(self, support_triples, entity_embeddings):
        """使用训练数据初始化背景关系嵌入（任务关系在运行时动态处理）"""
        print("Initializing background relation embeddings from training data...")
        
        # 按关系分组支持集三元组
        rel_to_triples = {}
        for h, r, t in support_triples:
            if r not in rel_to_triples:
                rel_to_triples[r] = []
            rel_to_triples[r].append((h, t))
        
        # 为每个背景关系计算平均嵌入
        initialized_count = 0
        for rel_id, triples in rel_to_triples.items():
            rel_embeddings = []
            
            for h, t in triples:
                # 计算尾实体 - 头实体的差值
                h_emb = entity_embeddings[h]
                t_emb = entity_embeddings[t]
                rel_emb = t_emb - h_emb
                rel_embeddings.append(rel_emb)
            
            # 计算平均值
            if rel_embeddings:
                avg_rel_emb = torch.stack(rel_embeddings).mean(dim=0)
                self.rel_embeddings.weight.data[rel_id] = avg_rel_emb
                initialized_count += 1
        
        # 对于没有样本的背景关系，使用随机初始化（保持原有的Xavier初始化）
        total_relations = self.rel_embeddings.weight.size(0)
        uninitialized_count = total_relations - initialized_count
        
        print(f"Initialized {initialized_count} background relation embeddings from training data")
        print(f"Remaining {uninitialized_count} background relations use Xavier initialization")
        print(f"Total background relations: {total_relations}")
        print("Note: Task relations will be dynamically computed during inference using support sets")
    
    def build_global_graph_edges(self, path_info=None, max_edges=None, task_entities=None):
        """
        构建高相关度子图 - 使用配置的子图大小
        Args:
            path_info: 路径信息
            max_edges: 最大边数，如果为None则使用self.max_subgraph_edges
            task_entities: 任务实体列表
        """
        # 使用配置的子图大小，如果没有传入max_edges参数
        if max_edges is None:
            max_edges = self.max_subgraph_edges
        
        # print(f"Building subgraph with max_edges={max_edges}")
        
        edges = []
        edge_types = []  # 0: 头实体->关系, 1: 关系->尾实体, 2: 路径连接
        
        # 为每个关系分配临时ID（用于图构建）
        relation_to_temp_id = {}
        temp_id_counter = 0
        
        # 从path_info构建高相关度子图
        if path_info is not None:
            path_items = list(path_info.items())
            
            # 如果有任务实体信息，计算相关度并排序
            if task_entities is not None and len(task_entities) > 0:
                # 将任务实体转换为集合以便快速查找
                task_entity_set = set(task_entities)
                
                # 计算每个实体对的相关度分数
                scored_paths = []
                for (h_id, t_id), paths in path_items:
                    # 计算相关度：任务实体在实体对中的比例
                    task_entities_in_pair = 0
                    if h_id in task_entity_set:
                        task_entities_in_pair += 1
                    if t_id in task_entity_set:
                        task_entities_in_pair += 1
                    
                    # 相关度分数 = 任务实体数量 / 2
                    relevance_score = task_entities_in_pair / 2.0
                    
                    # 只保留相关度大于0的路径
                    if relevance_score > 0:
                        scored_paths.append(((h_id, t_id), paths, relevance_score))
                
                # 按相关度降序排序
                scored_paths.sort(key=lambda x: x[2], reverse=True)
                
                # 选择相关度最高的路径
                target_pairs = max_edges // 4  # 约500个实体对
                if len(scored_paths) > 0:
                    # print(f"Found {len(scored_paths)} task-relevant entity pairs out of {len(path_items)} total pairs")

                    selected_paths = scored_paths[:target_pairs]
                    path_items = [(h_t, paths) for h_t, paths, score in selected_paths]

                    avg_relevance = sum(score for _, _, score in selected_paths) / len(selected_paths)
                    # print(f"Selected {len(path_items)} highest-relevance entity pairs (avg relevance: {avg_relevance:.3f})")
                else:
                    # 如果没有找到相关路径，使用简单的随机采样
                    if len(path_items) > target_pairs:
                        import random
                        random.seed(42)
                        path_items = random.sample(path_items, target_pairs)
                        # print(f"No task-relevant paths found, randomly sampled {len(path_items)} entity pairs")
            else:
                # 没有任务实体信息，使用简单的随机采样
                target_pairs = max_edges // 4
                if len(path_items) > target_pairs:
                    import random
                    random.seed(42)
                    path_items = random.sample(path_items, target_pairs)
                    # print(f"Randomly sampled {len(path_items)} entity pairs from {len(path_info)} total pairs")
            
            for (h_id, t_id), paths in path_items:
                for path in paths:
                    if len(path) >= 1:  # 至少有一个关系
                        # 为路径中的每个关系分配临时ID
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
                                    # 第一个关系：h -> r1
                                    edges.append([prev_node, r_temp_id])
                                    edge_types.append(0)
                                else:
                                    # 中间关系：r_{i-1} -> r_i
                                    edges.append([prev_node, r_temp_id])
                                    edge_types.append(2)  # 路径连接
                                
                                if i == len(path) - 1:
                                    # 最后一个关系：r_n -> t
                                    edges.append([r_temp_id, t_id])
                                    edge_types.append(1)
                                else:
                                    prev_node = r_temp_id
                        
                        # 检查边数限制
                        if len(edges) >= max_edges:
                            # print(f"Reached edge limit ({max_edges}), stopping graph construction")
                            break
                
                # 检查边数限制
                if len(edges) >= max_edges:
                    break
        
        # 如果没有path_info或边数太少
        if len(edges) < 200:  # 进一步降低阈值
            # print("Building simple global graph as fallback")
            # 为所有关系分配ID
            for rel_id in range(min(self.n_rel, 25)):  # 进一步限制关系数量
                relation_to_temp_id[rel_id] = rel_id

            for i in range(0, min(1000, self.n_ent), 2):  # 进一步限制实体数量
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
        
        # 调试信息
        if not hasattr(self, '_global_graph_debug_printed'):
            # print(f"Built task-relevant subgraph with {len(edges)} edges (limited to {max_edges})")
            # print(f"Subgraph edge tensor shape: {edge_tensor.shape}")
            # print(f"Subgraph edge types: {set(edge_types)}")
            # print(f"Subgraph unique relations: {len(relation_to_temp_id)}")
            self._global_graph_debug_printed = True
        
        return edge_tensor, edge_types_tensor

    def build_graph_edges(self, triples, path_info=None):
        """构建图的边 - 支持路径信息增强和动态关系处理"""
        edges = []
        edge_types = []  # 0: 头实体->关系, 1: 关系->尾实体, 2: 路径连接
        
        # 为每个关系分配临时ID（用于图构建）
        relation_to_temp_id = {}
        temp_id_counter = 0
        
        # 添加直接的三元组边
        for h, r, t in triples:
            # 为关系分配临时ID
            if r not in relation_to_temp_id:
                relation_to_temp_id[r] = temp_id_counter
                temp_id_counter += 1
            
            r_temp_id = relation_to_temp_id[r]
            
            # 头实体到关系
            edges.append([h, r_temp_id])
            edge_types.append(0)
            # 关系到尾实体
            edges.append([r_temp_id, t])
            edge_types.append(1)

        if path_info is not None:
            for h, r, t in triples:
                h_ent, t_ent = h, t
                
                # 查找实体对之间的路径
                if hasattr(self, 'data_loader') and hasattr(self.data_loader, 'get_path_info'):
                    paths = self.data_loader.get_path_info(h_ent, t_ent)
                    
                    # 添加路径边
                    for path in paths:
                        if len(path) > 1:  # 多跳路径
                            # 为路径中的每个关系添加边
                            for rel_id in path:
                                if rel_id != r:  # 避免重复
                                    edges.append([h_ent, rel_id])
                                    edge_types.append(2)  # 路径边类型
                                    edges.append([rel_id, t_ent])
                                    edge_types.append(2)
        
        if edges:
            edge_tensor = torch.tensor(edges, dtype=torch.long).t()
            edge_types_tensor = torch.tensor(edge_types, dtype=torch.long)
        else:
            # 如果没有边，创建空的张量
            edge_tensor = torch.empty((2, 0), dtype=torch.long)
            edge_types_tensor = torch.empty((0,), dtype=torch.long)
        
        # 调试信息
        if not hasattr(self, '_debug_printed'):
            print(f"Built {len(edges)} edges from {len(triples)} triples")
            print(f"Edge tensor shape: {edge_tensor.shape}")
            print(f"Edge types: {set(edge_types)}")
            print(f"Unique relations: {len(relation_to_temp_id)}")
            self._debug_printed = True
        
        return edge_tensor, edge_types_tensor
    
    def forward_gat(self, entity_embeddings, relation_embeddings, edge_index, edge_types):
        """通过GAT层进行嵌入改进 - 支持动态关系嵌入"""
        # 处理动态关系嵌入
        if isinstance(relation_embeddings, dict):
            # 动态关系嵌入：将字典转换为张量
            rel_emb_list = list(relation_embeddings.values())
            if rel_emb_list:
                # 计算最大关系数量
                max_rel_id = max(len(relation_embeddings), entity_embeddings.size(0))

                rel_emb_tensor = torch.zeros(max_rel_id, entity_embeddings.size(-1), 
                                           device=entity_embeddings.device)
                # 填充已知的关系嵌入
                for i, emb in enumerate(rel_emb_list):
                    if i < max_rel_id:
                        rel_emb_tensor[i] = emb
                relation_embeddings = rel_emb_tensor
            else:
                relation_embeddings = torch.zeros(entity_embeddings.size(0), entity_embeddings.size(-1),
                                                device=entity_embeddings.device)
        
        # 合并实体和关系嵌入
        all_embeddings = torch.cat([entity_embeddings, relation_embeddings], dim=0)
        
        # 调试信息
        if not hasattr(self, '_forward_gat_debug_printed'):
            print(f"Input embeddings: all_mean={all_embeddings.mean().item():.4f}, all_std={all_embeddings.std().item():.4f}")
            self._forward_gat_debug_printed = True
        
        # 通过GAT层进行嵌入改进
        x = all_embeddings
        for i, gat_layer in enumerate(self.gat_layers):
            # 调试信息
            if not hasattr(self, '_gat_debug_printed'):
                print(f"GAT input: N={x.size(0)}, x_shape={x.shape}, x_mean={x.mean().item():.4f}, x_std={x.std().item():.4f}")
                print(f"Edge index shape: {edge_index.shape}")
                self._gat_debug_printed = True
            
            x = gat_layer(x, edge_index)
            
            # 调试信息
            if not hasattr(self, '_gat_output_debug_printed'):
                print(f"GAT Layer {i+1} output stats: mean={x.mean().item():.4f}, std={x.std().item():.4f}")
                self._gat_output_debug_printed = True
        
        # 分离实体和关系嵌入
        n_entities = entity_embeddings.size(0)
        improved_entity_emb = x[:n_entities]
        improved_rel_emb = x[n_entities:]
        
        # 应用残差连接
        improved_entity_emb = improved_entity_emb + self.residual_weight * entity_embeddings
        improved_rel_emb = improved_rel_emb + self.residual_weight * relation_embeddings
        
        # 调试信息
        if not hasattr(self, '_improved_emb_debug_printed'):
            print(f"Improved embeddings stats: entity_mean={improved_entity_emb.mean().item():.4f}, rel_mean={improved_rel_emb.mean().item():.4f}")
            self._improved_emb_debug_printed = True
        
        return improved_entity_emb, improved_rel_emb
    
    def forward(self, support_triples, query_triples, mode='training', path_info=None):
        """
        前向传播 - 改进的评分策略
        Args:
            support_triples: 支持集三元组 [(h, r, t), ...] 其中r可以是字符串或ID
            query_triples: 查询集三元组 [(h, r, t), ...] 其中r可以是字符串或ID
            mode: 'training' 或 'inference'
            path_info: 路径信息字典
        """
        device = next(self.parameters()).device
        
        # 获取当前嵌入
        entity_embeddings = self.ent_embeddings.weight

        relation_embeddings = self.compute_dynamic_relation_embeddings(support_triples, entity_embeddings)
        
        # 提取当前任务相关的实体
        task_entities = set()
        for h, r, t in support_triples + query_triples:
            task_entities.add(h)
            task_entities.add(t)
        
        # 构建任务相关的子图
        global_edge_index, global_edge_types = self.build_global_graph_edges(
            path_info, max_edges=self.max_subgraph_edges, task_entities=list(task_entities))
        global_edge_index = global_edge_index.to(device)
        global_edge_types = global_edge_types.to(device)
        
        # 通过GAT改进嵌入
        improved_entity_emb, improved_rel_emb = self.forward_gat(
            entity_embeddings, relation_embeddings, global_edge_index, global_edge_types)
        
        # 计算查询三元组的分数
        query_scores = []
        for h, r, t in query_triples:
            try:
                # 检查实体ID是否在有效范围内
                if h >= improved_entity_emb.size(0) or t >= improved_entity_emb.size(0):

                    h_emb = torch.zeros(improved_entity_emb.size(-1), device=improved_entity_emb.device)
                    t_emb = torch.zeros(improved_entity_emb.size(-1), device=improved_entity_emb.device)
                else:
                    h_emb = improved_entity_emb[h]
                    t_emb = improved_entity_emb[t]
                
                # 获取关系嵌入
                if isinstance(r, str):
                    # 使用动态计算的关系嵌入
                    try:
                        r_emb = self.get_task_relation_embedding(r, support_triples, improved_entity_emb)
                    except Exception as rel_e:
                        print(f"Error computing relation embedding for {r}: {rel_e}")

                        r_emb = torch.zeros(improved_entity_emb.size(-1), device=improved_entity_emb.device)
                else:
                    # 使用预训练的关系嵌入
                    try:
                        if r >= improved_rel_emb.size(0):
                            print(f"Warning: Relation ID out of range: r={r}, max_rel_id={improved_rel_emb.size(0)-1}")
                            r_emb = torch.zeros(improved_entity_emb.size(-1), device=improved_entity_emb.device)
                        else:
                            r_emb = improved_rel_emb[r]
                    except Exception as rel_e:
                        print(f"Error accessing relation embedding for {r}: {rel_e}")
                        r_emb = torch.zeros(improved_entity_emb.size(-1), device=improved_entity_emb.device)

                score = self.compute_improved_score(h_emb, r_emb, t_emb)
                query_scores.append(score)
            except Exception as e:
                print(f"Error computing score for triple ({h}, {r}, {t}): {e}")
                # 使用零分数作为默认值
                default_score = torch.tensor(0.0, device=next(self.parameters()).device)
                query_scores.append(default_score)
        
        # 检查是否有有效的分数
        if len(query_scores) == 0:
            print("Warning: No valid scores computed, returning empty tensor")
            return torch.empty(0, device=next(self.parameters()).device)
        
        query_scores = torch.stack(query_scores)

        if query_scores.dim() > 1:
            query_scores = query_scores.squeeze()
        
        if mode == 'training':
            return query_scores
        else:
            # 推理模式：返回改进后的嵌入
            return improved_entity_emb, improved_rel_emb, query_scores

    def compute_improved_score(self, h_emb, r_emb, t_emb):
        """TransE"""
        # TransE评分：h + r ≈ t，使用负的L2范数作为分数
        # 注意：这里保持与trainer.py中TransE评分的一致性
        score = -torch.norm(h_emb + r_emb - t_emb, p=2, dim=-1)
        return score

    def compute_transe_score(self, h_emb, r_emb, t_emb):
        """TransE三元组评分函数"""
        # 计算三元组的评分，使用的是TransE评分方式：h + r ≈ t
        # 返回原始距离分数，用于后续的margin和sigmoid转换
        scores = torch.norm(h_emb + r_emb - t_emb, p=2, dim=-1)
        return scores

    def compute_dynamic_relation_embeddings(self, support_triples, entity_embeddings):
        """动态计算关系嵌入，支持任务关系和背景关系"""
        # 获取所有唯一的关系
        relations = set()
        for h, r, t in support_triples:
            relations.add(r)
        
        # 为每个关系计算嵌入
        relation_embeddings = {}
        for r in relations:
            if isinstance(r, str):
                # 任务关系：基于支持集计算
                r_emb = self.compute_task_relation_embedding(r, support_triples, entity_embeddings)
                relation_embeddings[r] = r_emb
            else:
                # 背景关系：使用预训练嵌入
                relation_embeddings[r] = self.rel_embeddings.weight[r]
        
        return relation_embeddings

    def compute_task_relation_embedding(self, relation_name, support_triples, entity_embeddings):
        """基于支持集计算任务关系的嵌入"""
        # 找到该关系的支持三元组
        relation_support = []
        for h, r, t in support_triples:
            if r == relation_name:
                relation_support.append((h, t))
        
        if not relation_support:
            # 如果没有支持样本，使用随机初始化
            return torch.randn(entity_embeddings.size(-1), device=entity_embeddings.device)
        
        # 计算关系嵌入：平均所有支持样本的 (t - h)
        rel_embeddings = []
        for h, t in relation_support:
            h_emb = entity_embeddings[h]
            t_emb = entity_embeddings[t]
            rel_emb = t_emb - h_emb
            rel_embeddings.append(rel_emb)
        
        # 返回平均嵌入
        return torch.stack(rel_embeddings).mean(dim=0)

    def get_task_relation_embedding(self, relation_name, support_triples, entity_embeddings):
        """获取任务关系的嵌入"""
        return self.compute_task_relation_embedding(relation_name, support_triples, entity_embeddings)
    
    def compute_similarity(self, query_emb, candidate_embs):
        """计算查询嵌入与候选嵌入的相似度"""
        # 扩展查询嵌入以匹配候选数量
        query_emb = query_emb.unsqueeze(0).expand(candidate_embs.size(0), -1)
        
        # 计算相似度
        similarity = self.similarity_layer(torch.cat([query_emb, candidate_embs], dim=-1))
        
        return similarity.squeeze(-1)
    
    def predict_triple(self, h, r, t):
        """预测三元组的正确性"""
        h_emb = self.ent_embeddings(h)
        r_emb = self.rel_embeddings(r)
        t_emb = self.ent_embeddings(t)
        
        # 拼接嵌入
        triple_emb = torch.cat([h_emb, r_emb, t_emb], dim=-1)
        
        # 分类
        logits = self.classifier(triple_emb)
        probs = F.softmax(logits, dim=-1)
        
        return probs[:, 1]  # 返回正确的概率

class FKGCDataLoaderWithText(FKGCDataLoader):
    """扩展的FKGC数据加载器"""
    
    def __init__(self, in_paths, tokenizer, batch_size=2, max_desc_length=64, 
                 add_tokens=False, p_tuning=False, model='bert', sp_num=5):
        super().__init__(in_paths, tokenizer, batch_size, max_desc_length, 
                        add_tokens, p_tuning, model, sp_num)
        
        # 构建实体文本字典
        self.entity_texts = {}
        self.relation_texts = {}
        self.build_text_dicts()
    
    def build_text_dicts(self):
        """构建实体和关系的文本字典"""
        for uid, text in self.uid2text.items():
            if uid in self.ent2id:
                self.entity_texts[self.ent2id[uid]] = text
            elif uid in self.rel2id:
                self.relation_texts[self.rel2id[uid]] = text
    
    def get_entity_texts(self):
        """获取所有实体的文本描述"""
        return self.entity_texts
    
    def get_relation_texts(self):
        """获取所有关系的文本描述"""
        return self.relation_texts

def train_fkgc_model(model, dataloader, num_epochs=100, lr=1e-4, device='cuda'):
    """训练FKGC模型"""
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    # 初始化嵌入
    entity_texts = dataloader.get_entity_texts()
    model.initialize_entity_embeddings(entity_texts)
    
    # 获取训练数据用于关系初始化
    train_triples = dataloader.train_set[:1000]  # 使用部分训练数据
    entity_embeddings = model.ent_embeddings.weight.data
    model.initialize_relation_embeddings(train_triples, entity_embeddings)
    
    print("Starting training...")
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        num_batches = 0
        
        for support_pair, query_pair_pos, one_tomany_train, candidates_id in dataloader.train_generator():
            # 转换为三元组格式
            support_triples = []
            for h_id, t_id in support_pair:
                # 这里需要根据具体任务确定关系
                r_id = 0  # 简化处理
                support_triples.append((h_id, r_id, t_id))
            
            query_triples = []
            for h_id, t_id in query_pair_pos:
                r_id = 0  # 简化处理
                query_triples.append((h_id, r_id, t_id))
            
            # 前向传播
            scores = model(support_triples, query_triples, mode='training')
            
            # 计算损失（这里需要根据具体任务调整）
            target_scores = torch.zeros_like(scores)
            loss = criterion(scores, target_scores)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            if num_batches % 100 == 0:
                print(f"Epoch {epoch}, Batch {num_batches}, Loss: {loss.item():.4f}")
        
        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch}, Average Loss: {avg_loss:.4f}")
    
    return model 