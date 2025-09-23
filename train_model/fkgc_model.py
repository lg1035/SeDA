import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from transformers import AutoModel, AutoTokenizer
from lora_utils import LoRAConfig, apply_lora_to_model, get_lora_parameters
from dataloader import FKGCDataLoader
from collections import OrderedDict


# ==================== 原有组件 ====================

def get_model_device(model):
    """安全地获取模型设备，支持DataParallel"""
    if hasattr(model, 'module'):
        # DataParallel包装的模型
        return next(model.module.parameters()).device
    else:
        return next(model.parameters()).device


class GATLayer(nn.Module):
    """简化的图注意力网络层"""

    def __init__(self, in_dim, out_dim, num_heads=8, dropout=0.1):
        super(GATLayer, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.dropout = dropout

        # 线性变换层
        self.W = nn.Linear(in_dim, out_dim * num_heads, bias=False)
        # 注意力机制
        self.attention = nn.Parameter(torch.Tensor(1, num_heads, out_dim * 2))
        # 偏置项
        self.bias = nn.Parameter(torch.Tensor(out_dim))
        
        # 移除层归一化和Dropout
        # self.layer_norm = nn.LayerNorm(out_dim)
        # self.dropout_layer = nn.Dropout(dropout)
        
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.attention)
        nn.init.zeros_(self.bias)

    def forward(self, x, edge_index):
        N = x.size(0)
        E = edge_index.size(1)
        
        # print(f"    GATLayer forward: N={N}, E={E}, input_shape={x.shape}")

        # 线性变换
        x = self.W(x).view(N, self.num_heads, -1)

        # 处理空的边索引
        if E == 0:
            # print(f"    Warning: No edges found in GATLayer, returning zero output")
            out = torch.zeros(N, self.num_heads, self.out_dim, device=x.device)
            return out.mean(dim=1) + self.bias

        # 准备源节点和目标节点特征
        row, col = edge_index
        x_i = x[row]
        x_j = x[col]

        # 计算注意力分数 - 使用更高效的实现
        alpha_input = torch.cat([x_i, x_j], dim=-1)
        alpha = torch.einsum('bhd,hd->bh', alpha_input, self.attention.squeeze(0))
        alpha = F.leaky_relu(alpha, negative_slope=0.2)

        # 使用最简单的softmax归一化方法
        alpha = alpha.view(-1, self.num_heads)
        
        # 直接使用PyTorch的scatter_softmax（如果可用）
        try:
            from torch_scatter import scatter_softmax
            alpha_normalized = scatter_softmax(alpha, row, dim=0)
        except ImportError:
            # 回退到循环方法
            alpha_normalized = torch.zeros_like(alpha)
            for i in range(N):
                mask = (row == i)
                if mask.any():
                    alpha_normalized[mask] = F.softmax(alpha[mask], dim=0)

        alpha = F.dropout(alpha_normalized, p=self.dropout, training=self.training)
        alpha = alpha.unsqueeze(-1)

        # 聚合邻居信息 - 使用更高效的实现
        weighted_features = alpha * x_j
        out = torch.zeros(N, self.num_heads, self.out_dim, device=x.device)
        out.scatter_add_(0, row.unsqueeze(1).unsqueeze(2).expand(-1, self.num_heads, self.out_dim), weighted_features)

        final_output = out.mean(dim=1) + self.bias
        
        # 移除残差连接、层归一化和Dropout
        # if x.size(-1) == final_output.size(-1):
        #     x_single = x.mean(dim=1) if x.dim() == 3 else x
        #     final_output = final_output + 0.5 * x_single
        
        # final_output = self.layer_norm(final_output)
        # final_output = self.dropout_layer(final_output)
        
        # print(f"    GATLayer output: shape={final_output.shape}, mean={final_output.mean().item():.4f}, std={final_output.std().item():.4f}")
        return final_output


class FKGCWithGAT(nn.Module):
    """基于BERT初始化和GAT改进的FKGC模型"""

    def __init__(self, bert_model_name, n_ent, n_rel, hidden_dim=768,
                 gat_heads=8, gat_layers=2, dropout=0.1,
                 use_lora=False, lora_config=None, data_loader=None, max_subgraph_edges=20000,
                 use_gat=True, debug_mode=False, fine_tune_bert=False,
                 scoring_paradigm='transe', fkgc_attention_temp=1.0, fkgc_prototype_shrinkage=0.1,
                 fkgc_attention_weight=0.7):
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
            use_lora: 是否使用LoRA
            lora_config: LoRA配置
            data_loader: 数据加载器
            max_subgraph_edges: 子图最大边数，默认2000
            scoring_paradigm: 评分范式，'transe' 或 'fkgc'，默认为 'transe'
            fkgc_attention_temp: FKGC注意力温度参数
            fkgc_prototype_shrinkage: FKGC原型收缩因子
            fkgc_attention_weight: FKGC注意力权重（混合原型）
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
        self.use_lora = use_lora
        self.lora_config = lora_config
        self.data_loader = data_loader
        self.max_subgraph_edges = max_subgraph_edges  # 新增：子图大小配置
        self.use_gat = use_gat  # 新增：是否使用GAT
        self.debug_mode = False  # 新增：调试模式（默认关闭）
        self.fine_tune_bert = fine_tune_bert  # 新增：是否全量微调BERT

        # 评分范式参数
        self.scoring_paradigm = scoring_paradigm  # 评分范式：'transe' 或 'fkgc'
        self.fkgc_attention_temp = fkgc_attention_temp  # FKGC注意力温度参数
        self.fkgc_prototype_shrinkage = fkgc_prototype_shrinkage  # FKGC原型收缩因子
        self.fkgc_attention_weight = fkgc_attention_weight  # FKGC注意力权重（混合原型）
        # 调试总开关（可在运行时修改）
        self.debug_fkgc = False
        # 原型缓存（性能优化）
        self.prototype_cache = {}
        self.cache_enabled = True
        # 继续完成初始化
        self._safe_post_init(data_loader, bert_model_name, use_lora, lora_config, dropout)

    def clear_prototype_cache(self):
        """清理原型缓存"""
        self.prototype_cache.clear()

    def disable_prototype_cache(self):
        """禁用原型缓存"""
        self.cache_enabled = False

    def enable_prototype_cache(self):
        """启用原型缓存"""
        self.cache_enabled = True
        # print(f"FKGC Paradigm enabled: {self.use_fkgc_paradigm}")

    def _safe_post_init(self, data_loader, bert_model_name, use_lora, lora_config, dropout):
        """在__init__中调用的后续初始化，确保 ent_embeddings 总被创建。"""
        # 实体ID映射
        self.ent2id = getattr(data_loader, 'ent2id', {}) if data_loader is not None else {}

        # BERT
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(bert_model_name)
            self.bert = AutoModel.from_pretrained(bert_model_name)
            if use_lora and lora_config:
                apply_lora_to_model(self.bert, lora_config)
                print("Applied LoRA to BERT with provided configuration")
            print(f"BERT model loaded: {bert_model_name}")
        except Exception as e:
            print(f"Error loading BERT model: {e}")
            # 继续构造其余组件，避免属性缺失
            self.bert = None

        # 嵌入层
        self.ent_embeddings = nn.Embedding(self.n_ent, self.hidden_dim)
        self.rel_embeddings = nn.Embedding(self.n_rel, self.hidden_dim)


        # GAT/MLP
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

        # 其余组件
        self.residual_weight = nn.Parameter(torch.tensor(0.1))
        self.dropout_rate = 0.3
        self.l2_reg_weight = 0.01
        self.embedding_norm_weight = 0.001
        self.score_margin = 0.5

        # 结构-语义门控机制
        self.structural_gate = nn.Parameter(torch.tensor(0.5))  # 结构信息门控权重
        self.semantic_gate = nn.Parameter(torch.tensor(0.5))    # 语义信息门控权重
        
        # 自适应门控网络：融合结构信息和语义信息
        self.adaptive_gate_net = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),  # 输入：结构嵌入 + 语义嵌入
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 2),  # 输出：结构门控权重 + 语义门控权重
            nn.Sigmoid()  # 确保输出在[0,1]范围内
        )
        
        # TransE结构嵌入层
        self.structural_entity_embeddings = nn.Embedding(self.n_ent, self.hidden_dim)
        self.structural_rel_embeddings = nn.Embedding(self.n_rel, self.hidden_dim)

        self.init_embeddings()        

        # 输出GAT参数信息
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


        # 完成

    def init_embeddings(self):
        """初始化嵌入"""
        # 实体嵌入使用Xavier初始化
        nn.init.xavier_uniform_(self.ent_embeddings.weight)
        nn.init.xavier_uniform_(self.rel_embeddings.weight)
        
        # 结构嵌入（TransE）使用Xavier初始化
        nn.init.xavier_uniform_(self.structural_entity_embeddings.weight)
        nn.init.xavier_uniform_(self.structural_rel_embeddings.weight)
        
        # 总是需要初始化结构嵌入（TransE）
        self.load_structural_embeddings()


    # 已移除：三元组分类器初始化

    def bert_encode_entity(self, entity_texts):
        """使用BERT编码实体文本"""
        try:
            # print(f"BERT encoding {len(entity_texts)} entities...")
            # 对实体文本进行tokenize
            inputs = self.tokenizer(entity_texts,
                                    padding=True,
                                    truncation=True,
                                    max_length=64,
                                    return_tensors='pt')

            # 获取正确的设备（支持DataParallel）
            device = get_model_device(self)
            # print(f"Moving inputs to device: {device}")

            # 移动到正确的设备
            inputs = {k: v.to(device) for k, v in inputs.items()}

            # BERT编码（无论是否LoRA，一律不保留计算图；此路径仅用于生成静态实体嵌入）
            self.bert.eval()
            with torch.no_grad():
                # 使用半精度可大幅降低显存占用（在CUDA上）
                if torch.cuda.is_available():
                    with torch.cuda.amp.autocast(dtype=torch.float16):
                        outputs = self.bert(**inputs)
                else:
                    outputs = self.bert(**inputs)

            # 使用[CLS] token的表示作为实体表示
            entity_embeddings = outputs.last_hidden_state[:, 0, :]  # [batch_size, 768]

            # 降维：从BERT的768维降到模型的100维
            if hasattr(self, 'bert_projection') and self.bert_projection is not None:
                entity_embeddings = self.bert_projection(entity_embeddings)  # [batch_size, 100]
            else:
                # 如果没有投影层，使用简单的线性变换
                if not hasattr(self, '_bert_projection_fallback'):
                    self._bert_projection_fallback = nn.Linear(768, self.hidden_dim).to(device)
                    # 使用Xavier初始化
                    nn.init.xavier_uniform_(self._bert_projection_fallback.weight)
                    nn.init.zeros_(self._bert_projection_fallback.bias)
                entity_embeddings = self._bert_projection_fallback(entity_embeddings)  # [batch_size, 100]

            # 清理中间变量以节省内存
            del inputs, outputs

            return entity_embeddings

        except Exception as e:
            print(f"Error in BERT encoding: {e}")
            # 返回零嵌入作为后备
            device = get_model_device(self)
            return torch.zeros(len(entity_texts), self.hidden_dim, device=device)

    def initialize_semantic_entity_embeddings(self, entity_texts):
        """生成并保存一份独立的语义实体嵌入矩阵（BERT编码实体文本）。"""
        try:
            if not entity_texts:
                self.semantic_ent_emb = torch.empty(0, device=get_model_device(self))
                return
            entity_ids = []
            texts = []
            for ent_id, text in entity_texts.items():
                entity_ids.append(ent_id)
                texts.append(text)
            # 分批BERT编码
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
            # 写入按ID对齐的矩阵
            num_entities = self.ent_embeddings.weight.size(0)
            sem = torch.zeros((num_entities, self.hidden_dim), device=all_emb.device, dtype=all_emb.dtype)
            for i, ent_id in enumerate(entity_ids):
                if ent_id < num_entities:
                    sem[ent_id] = all_emb[i]
            # 保存为buffer（不参与梯度）
            self.semantic_ent_emb = sem.detach()
            print(f"Initialized separate semantic embeddings: shape={self.semantic_ent_emb.shape}")
        except Exception as e:
            print(f"Warning: initialize_semantic_entity_embeddings failed: {e}")
            self.semantic_ent_emb = torch.empty(0, device=get_model_device(self))

    def initialize_entity_embeddings(self, entity_texts):
        """使用BERT初始化实体嵌入"""
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

        # 分批处理以避免内存问题
        batch_size = 320  # 使用更大的批次大小，提高初始化效率
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

        # 合并所有嵌入
        all_embeddings = torch.cat(all_embeddings, dim=0)

        # 检查嵌入维度
        print(f"BERT embeddings shape: {all_embeddings.shape}")
        print(f"Expected entity embedding shape: {self.ent_embeddings.weight.shape}")

        # 更新实体嵌入层
        for i, ent_id in enumerate(entity_ids):
            if ent_id < self.ent_embeddings.weight.size(0):
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

    def build_global_graph_edges(self, path_info=None, max_edges=None, task_entities=None, bert_embeddings=None):
        """
        构建高相关度子图 - 使用配置的子图大小
        Args:
            path_info: 路径信息
            max_edges: 最大边数，如果为None则使用self.max_subgraph_edges
            task_entities: 任务实体列表
            bert_embeddings: BERT嵌入，用于子图构建时的相似度计算
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
            # 过滤掉缓存信息，只保留路径信息
            path_items = []
            for key, value in path_info.items():
                if key not in ['cached_entity_emb', 'cached_rel_emb']:
                    path_items.append((key, value))

            # 如果有任务实体信息，计算相关度并排序（允许不直接相邻但高分路径进入）
            if task_entities is not None and len(task_entities) > 0:
                # 将任务实体转换为集合以便快速查找
                task_entity_set = set(task_entities)

                # 计算每个实体对的综合评分
                scored_paths = []
                for item in path_items:
                    try:
                        # 处理不同的path_info格式
                        if len(item) == 2:
                            (h_id, t_id), paths = item
                        else:
                            # 如果格式不匹配，跳过这个项目
                            print(f"Warning: Unexpected path_info format: {item}")
                            continue
                    except (ValueError, TypeError) as e:
                        print(f"Error unpacking path_info item: {item}, error: {e}")
                        continue

                    # 1) 基础相关度：任务实体在实体对中的比例（0, 0.5, 1.0）
                    task_entities_in_pair = 0
                    if h_id in task_entity_set:
                        task_entities_in_pair += 1
                    if t_id in task_entity_set:
                        task_entities_in_pair += 1
                    base_relevance = task_entities_in_pair / 2.0

                    # 2) 路径长度偏好：短路径得分高
                    path_length_scores = []
                    for path in paths:
                        if len(path) == 1:
                            path_length_scores.append(1.0)
                        elif len(path) == 2:
                            path_length_scores.append(0.9)
                        else:
                            path_length_scores.append(0.6)
                    avg_path_score = sum(path_length_scores) / len(path_length_scores) if path_length_scores else 0.0

                    # 3) BERT嵌入相似度：如果提供了BERT嵌入，计算实体间的语义相似度
                    bert_similarity = 0.0
                    if bert_embeddings is not None:
                        try:
                            h_emb = bert_embeddings[h_id]
                            t_emb = bert_embeddings[t_id]
                            # 计算余弦相似度
                            bert_similarity = torch.cosine_similarity(h_emb.unsqueeze(0), t_emb.unsqueeze(0), dim=1).item()
                        except:
                            bert_similarity = 0.0
                    
                    # 4) 综合评分：结合基础相关度、路径质量和BERT相似度
                    alpha = 0.2  # 基础相关度权重
                    beta = 0.6   # 路径质量权重
                    gamma = 0.2  # BERT相似度权重
                    comprehensive_score = base_relevance * alpha + avg_path_score * beta + bert_similarity * gamma

                    scored_paths.append(((h_id, t_id), paths, comprehensive_score, base_relevance, avg_path_score))

                # 按综合评分降序排序
                scored_paths.sort(key=lambda x: x[2], reverse=True)

                # 优先选择两跳邻居：先选择所有两跳路径，再选择其他路径
                target_pairs = max_edges // 3  # 适度放宽实体对上限以扩大边数
                if len(scored_paths) > 0:
                    # print(f"Found {len(scored_paths)} task-relevant entity pairs out of {len(path_items)} total pairs")

                    # 分离两跳路径和其他路径
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
                        selected_paths.extend(two_hop_paths[:target_pairs // 2])  # 预留一半给两跳路径
                        if self.debug_mode:
                            print(f"Selected {len(selected_paths)} two-hop paths for task entities")

                    # 再添加其他高质量路径
                    remaining_slots = target_pairs - len(selected_paths)
                    if remaining_slots > 0 and len(other_paths) > 0:
                        selected_paths.extend(other_paths[:remaining_slots])

                    # 如果两跳路径不够，从其他路径中补充
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
                    # 如果没有找到相关路径，使用简单的随机采样
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

        # 如果没有path_info或边数太少，构建一个简单的全局图（放宽触发与规模）
        if len(edges) < max_edges // 2:  # 使用动态阈值而不是硬编码2000
            # print("Building simple global graph as fallback")
            # 为所有关系分配ID
            for rel_id in range(min(self.n_rel, 200)):
                relation_to_temp_id[rel_id] = rel_id

            # 构建一些基本的连接
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
        """通过GAT层进行嵌入改进 - 支持动态关系嵌入"""
        # GAT调试信息已关闭
        # print(f"\n=== GAT Debug Info ===")
        # print(f"GAT enabled: {self.use_gat}")
        # print(f"Number of GAT layers: {len(self.gat_layers) if self.use_gat else len(self.mlp_layers)}")
        # print(f"Edge index shape: {edge_index.shape}")
        # print(f"Number of edges: {edge_index.size(1)}")
        # print(f"Entity embeddings shape: {entity_embeddings.shape}")
        # print(f"Relation embeddings type: {type(relation_embeddings)}")
        
        # 处理动态关系嵌入
        if isinstance(relation_embeddings, dict):
            # 动态关系嵌入：将字典转换为张量
            rel_emb_list = list(relation_embeddings.values())
            if rel_emb_list:
                # 使用实际的关系数量，而不是与实体数量比较
                max_rel_id = len(relation_embeddings)
                # 创建关系嵌入张量
                rel_emb_tensor = torch.zeros(max_rel_id, entity_embeddings.size(-1),
                                             device=entity_embeddings.device)
                # 填充已知的关系嵌入
                for i, emb in enumerate(rel_emb_list):
                    if i < max_rel_id:
                        rel_emb_tensor[i] = emb
                relation_embeddings = rel_emb_tensor
            else:
                # 如果没有关系嵌入，使用零张量（正确的维度）
                relation_embeddings = torch.zeros(self.n_rel, entity_embeddings.size(-1),
                                                  device=entity_embeddings.device)

        # 合并实体和关系嵌入
        all_embeddings = torch.cat([entity_embeddings, relation_embeddings], dim=0)

        # 训练时进行嵌入归一化，防止嵌入范数过大
        if self.training:
            # 对嵌入进行L2归一化，但保持一定的范数变化
            all_embeddings = F.normalize(all_embeddings, p=2, dim=-1)

        # 调试信息：监控输入（只在第一个batch打印）
        if self.debug_mode and not hasattr(self, '_forward_gat_debug_printed'):
            print(
                f"Input embeddings: all_mean={all_embeddings.mean().item():.4f}, all_std={all_embeddings.std().item():.4f}")
            self._forward_gat_debug_printed = True

        # 通过GAT层或MLP进行嵌入改进
        x = all_embeddings

        if self.use_gat:
            # 使用GAT层
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
            # 使用MLP层
            # print(f"Using MLP layers for embedding improvement...")
            for i, mlp_layer in enumerate(self.mlp_layers):
                # print(f"MLP Layer {i + 1}/{len(self.mlp_layers)}:")
                # print(f"  Input shape: {x.shape}")
                # print(f"  Input mean: {x.mean().item():.4f}, std: {x.std().item():.4f}")
                
                x = mlp_layer(x)
                
                # print(f"  Output shape: {x.shape}")
                # print(f"  Output mean: {x.mean().item():.4f}, std: {x.std().item():.4f}")
                # print(f"  Output norm: {x.norm().item():.4f}")

        # 分离实体和关系嵌入
        n_entities = entity_embeddings.size(0)
        n_relations = relation_embeddings.size(0)
        improved_entity_emb = x[:n_entities]
        improved_rel_emb = x[n_entities:n_entities + n_relations]

        # print(f"Final GAT output:")
        # print(f"  Entity embeddings shape: {improved_entity_emb.shape}")
        # print(f"  Relation embeddings shape: {improved_rel_emb.shape}")
        # print(f"  Entity mean: {improved_entity_emb.mean().item():.4f}, std: {improved_entity_emb.std().item():.4f}")
        # print(f"  Relation mean: {improved_rel_emb.mean().item():.4f}, std: {improved_rel_emb.std().item():.4f}")

        # 结构-语义门控融合：融合TransE结构信息和BERT+GAT语义信息
        # 获取结构嵌入（TransE）
        structural_entity_emb = self.structural_entity_embeddings.weight
        
        # 计算自适应门控权重
        adaptive_gates = self.compute_structural_semantic_gates(
            structural_entity_emb, improved_entity_emb, 
            relation_embeddings, improved_rel_emb)
        structural_gate_adaptive = adaptive_gates[0]
        semantic_gate_adaptive = adaptive_gates[1]
        
        # 结合基础门控权重和自适应权重
        structural_gate = torch.clamp(self.structural_gate + structural_gate_adaptive, 0.0, 1.0)
        semantic_gate = torch.clamp(self.semantic_gate + semantic_gate_adaptive, 0.0, 1.0)
        
        # 应用结构-语义门控融合
        # 实体：结构嵌入 + 语义嵌入（BERT+GAT）
        improved_entity_emb = structural_gate * structural_entity_emb + semantic_gate * improved_entity_emb
        
        # 关系：结构嵌入 + 语义嵌入
        if isinstance(relation_embeddings, dict):
            # 对于动态关系，使用语义嵌入
            improved_rel_emb = semantic_gate * improved_rel_emb
        else:
            # 对于预训练关系，融合结构嵌入
            structural_rel_emb = self.structural_rel_embeddings.weight
            improved_rel_emb = structural_gate * structural_rel_emb + semantic_gate * improved_rel_emb

        # print(f"After gating (entity_gate={gate_e.item():.3f}, rel_gate={gate_r.item():.3f}):")
        # print(f"  Final entity mean: {improved_entity_emb.mean().item():.4f}, std: {improved_entity_emb.std().item():.4f}")
        # print(f"  Final relation mean: {improved_rel_emb.mean().item():.4f}, std: {improved_rel_emb.std().item():.4f}")
        # print(f"=== End GAT Debug Info ===\n")

        # 调试信息：监控改进后的嵌入
        if self.debug_mode and not hasattr(self, '_improved_emb_debug_printed'):
            print(
                f"Improved embeddings stats: entity_mean={improved_entity_emb.mean().item():.4f}, rel_mean={improved_rel_emb.mean().item():.4f}")
            self._improved_emb_debug_printed = True

        return improved_entity_emb, improved_rel_emb

    def compute_structural_semantic_gates(self, structural_entity_emb, semantic_entity_emb, 
                                         relation_embeddings, semantic_rel_emb):
        """计算结构-语义自适应门控权重"""
        # 计算结构嵌入和语义嵌入的差异
        entity_diff = torch.norm(semantic_entity_emb - structural_entity_emb, dim=-1, keepdim=True)
        entity_avg_diff = entity_diff.mean()
        
        # 计算关系嵌入的差异
        if isinstance(relation_embeddings, dict) or isinstance(semantic_rel_emb, dict):
            # 对于动态关系，只使用语义嵌入
            rel_avg_diff = torch.tensor(0.0, device=structural_entity_emb.device)
        else:
            # 对于预训练关系，计算结构嵌入和语义嵌入的差异
            structural_rel_emb = self.structural_rel_embeddings.weight
            # 确保维度匹配
            if semantic_rel_emb.size(0) == structural_rel_emb.size(0):
                rel_diff = torch.norm(semantic_rel_emb - structural_rel_emb, dim=-1, keepdim=True)
                rel_avg_diff = rel_diff.mean()
            else:
                # 如果维度不匹配，使用零差异
                rel_avg_diff = torch.tensor(0.0, device=structural_entity_emb.device)
        
        # 构建输入特征：结构嵌入 + 语义嵌入的统计信息
        # 使用嵌入的均值和标准差作为特征
        structural_mean = structural_entity_emb.mean(dim=0)
        semantic_mean = semantic_entity_emb.mean(dim=0)
        
        # 组合特征
        input_features = torch.cat([
            structural_mean,
            semantic_mean
        ], dim=-1)
        
        # 如果特征维度不匹配，使用零填充
        if input_features.size(-1) < self.hidden_dim * 2:
            padding = torch.zeros(self.hidden_dim * 2 - input_features.size(-1), 
                                device=structural_entity_emb.device)
            input_features = torch.cat([input_features, padding], dim=-1)
        elif input_features.size(-1) > self.hidden_dim * 2:
            input_features = input_features[:self.hidden_dim * 2]
        
        # 扩展到正确的批次大小
        input_features = input_features.unsqueeze(0).expand(structural_entity_emb.size(0), -1)
        
        # 通过自适应门控网络计算权重
        adaptive_weights = self.adaptive_gate_net(input_features)
        
        # 返回结构和语义的自适应权重
        structural_gate_adaptive = adaptive_weights[:, 0].mean()  # 结构门控权重
        semantic_gate_adaptive = adaptive_weights[:, 1].mean()    # 语义门控权重
        
        return structural_gate_adaptive, semantic_gate_adaptive

    def load_structural_embeddings(self):
        """从dataloader加载TransE结构嵌入"""
        try:
            # 从数据加载器获取TransE结构嵌入
            if hasattr(self.data_loader, 'transE_loaded') and self.data_loader.transE_loaded:
                # 获取TransE嵌入和映射
                transE_embeddings = self.data_loader.structural_entity_embeddings
                transE_entity2id = self.data_loader.transE_entity2id
                
                if transE_embeddings is not None and transE_entity2id is not None:
                    # 转换为tensor
                    embeddings = torch.from_numpy(transE_embeddings).float()
                    
                    # 更新结构嵌入
                    loaded_count = 0
                    for entity, entity_id in transE_entity2id.items():
                        if entity in self.data_loader.ent2id:
                            model_entity_id = self.data_loader.ent2id[entity]
                            if model_entity_id < self.structural_entity_embeddings.weight.size(0):
                                # 维度对齐
                                if embeddings.size(-1) >= self.hidden_dim:
                                    self.structural_entity_embeddings.weight.data[model_entity_id] = embeddings[entity_id][:self.hidden_dim]
                                else:
                                    # 零填充
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

    # 已移除：语义关系辅助路径（当前FKGC核心未使用）

    def compute_improved_score(self, h_emb, r_emb, t_emb):
        """TransE评分函数"""
        return -torch.norm(h_emb + r_emb - t_emb, p=2)

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

        # 读取评估缓存（若提供）
        cached_entity_emb = None
        cached_rel_emb = None
        if isinstance(path_info, dict):
            cached_entity_emb = path_info.get('cached_entity_emb', None)
            cached_rel_emb = path_info.get('cached_rel_emb', None)

        # 获取当前嵌入 - 使用BERT初始化的嵌入进行GAT计算
        entity_embeddings = self.structural_entity_embeddings.weight

        # 动态处理关系嵌入
        # 对于任务关系（字符串），基于支持集计算关系嵌入
        # 对于背景关系（ID），使用预训练的关系嵌入
        relation_embeddings = self.compute_dynamic_relation_embeddings(support_triples, entity_embeddings)

        # 提取当前任务相关的实体
        task_entities = set()
        for h, r, t in support_triples + query_triples:
            task_entities.add(h)
            task_entities.add(t)

        # 使用原来的全局图构建策略 - 使用BERT嵌入构造子图
        bert_entity_embeddings = self.ent_embeddings.weight  # BERT初始化的嵌入用于子图构建
        edge_index, edge_types = self.build_global_graph_edges(
            path_info, max_edges=self.max_subgraph_edges, task_entities=list(task_entities), 
            bert_embeddings=bert_entity_embeddings)
        edge_index = edge_index.to(device)
        edge_types = edge_types.to(device)

        # 通过GAT改进嵌入（使用局部子图）；若提供缓存，则跳过GAT（训练/评估均可复用）
        if cached_entity_emb is not None and cached_rel_emb is not None:
            improved_entity_emb, improved_rel_emb = cached_entity_emb, cached_rel_emb
        else:
            improved_entity_emb, improved_rel_emb = self.forward_gat(
                entity_embeddings, relation_embeddings, edge_index, edge_types)


        # 计算查询三元组的分数
        query_scores = []
        for h, r, t in query_triples:
            try:
                # 取实体嵌入 - 处理字符串实体ID
                # 如果实体ID是字符串，需要映射到整数索引
                if isinstance(h, str):
                    # 尝试从数据加载器的实体映射中获取索引
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
                # 如果无法转换，使用零向量
                h_emb = torch.zeros(improved_entity_emb.size(-1), device=improved_entity_emb.device,
                                    dtype=improved_entity_emb.dtype, requires_grad=True)
                t_emb = torch.zeros(improved_entity_emb.size(-1), device=improved_entity_emb.device,
                                    dtype=improved_entity_emb.dtype, requires_grad=True)

            # 支持集中与当前关系 r 匹配的样本
            current_support = [(sh, st) for (sh, sr, st) in support_triples if sr == r]

            if getattr(self, 'debug_fkgc', False):
                try:
                    print(f"[FKGC] mode={mode}, relation={r}, support_size={len(current_support)}")
                except Exception:
                    pass

            # 根据评分范式选择评分方法
            if self.scoring_paradigm == 'fkgc':
                # FKGC范式：基于关系原型的评分
                if len(current_support) == 0:
                    # 如果没有支持样本，使用零分数
                    score = torch.tensor(0.0, device=h_emb.device, dtype=h_emb.dtype, requires_grad=True)
                else:
                    # 计算查询关系表示
                    query_relation_rep = compute_query_relation_representation(h_emb, t_emb)

                    # 使用混合原型（结合简单均值和注意力加权，提高泛化能力）
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
                    # 计算FKGC分数（带温度缩放）
                    score = compute_fkgc_score(query_relation_rep, relation_prototype, self.fkgc_attention_temp)
                    if getattr(self, 'debug_fkgc', False):
                        try:
                            print(f"[FKGC] score={score.item():.4f}")
                        except Exception:
                            pass
            else:
                # TransE范式：简单的距离评分
                if len(current_support) == 0:
                    score = self.compute_improved_score(h_emb, torch.zeros_like(h_emb), t_emb)
                else:
                    # 获取关系嵌入
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

        # 检查是否有有效的分数
        if len(query_scores) == 0:
            print("Warning: No valid scores computed, returning empty tensor")
            return torch.empty(0, device=next(self.parameters()).device, dtype=improved_entity_emb.dtype,
                               requires_grad=True)

        query_scores = torch.stack(query_scores)

        # 确保返回1D张量
        if query_scores.dim() > 1:
            query_scores = query_scores.squeeze()

        if mode == 'training':
            # 训练模式：仅返回分数
            return query_scores
        else:
            # 推理模式：返回改进后的嵌入
            return improved_entity_emb, improved_rel_emb, query_scores

    def compute_margin_ranking_loss(self, pos_scores, neg_scores, margin=1.0):
        """
        实现论文公式17的Margin-based Ranking Loss
        ℒ = ∑_r [ γ + (1/m) ∑_{i=1}^m ϕ_- - (1/n) ∑_{j=1}^n ϕ_+ ]
        Args:
            pos_scores: 正样本分数 [n]
            neg_scores: 负样本分数 [m]
            margin: 边界值 γ
        Returns:
            Margin-based Ranking Loss
        """
        # 确保输入是张量并保持梯度连接和数据类型一致
        if not isinstance(pos_scores, torch.Tensor):
            pos_scores = torch.tensor(pos_scores, device=next(self.parameters()).device,
                                      dtype=next(self.parameters()).dtype, requires_grad=True)
        if not isinstance(neg_scores, torch.Tensor):
            neg_scores = torch.tensor(neg_scores, device=next(self.parameters()).device,
                                      dtype=next(self.parameters()).dtype, requires_grad=True)

        # 计算正样本和负样本的平均分数
        pos_mean = pos_scores.mean()  # (1/n) ∑_{j=1}^n ϕ_+
        neg_mean = neg_scores.mean()  # (1/m) ∑_{i=1}^m ϕ_-

        # 实现公式17：γ + (1/m) ∑_{i=1}^m ϕ_- - (1/n) ∑_{j=1}^n ϕ_+
        loss = margin + neg_mean - pos_mean

        # 使用ReLU确保损失非负
        loss = torch.relu(loss)

        return loss

    def compute_dynamic_relation_embeddings(self, support_triples, entity_embeddings, path_info=None):
        """动态计算关系嵌入，支持任务关系和背景关系，使用缓存机制"""
        # 获取所有唯一的关系
        relations = set()
        for h, r, t in support_triples:
            relations.add(r)

        # 为每个关系计算嵌入（注意力聚合替代简单平均）
        relation_embeddings = {}
        for r in relations:
            if isinstance(r, str):
                # 任务关系：注意力聚合 (t - h) + 路径感知偏置
                deltas = []
                path_bias_list = []
                for h_id, r_name, t_id in support_triples:
                    if r_name == r:
                        deltas.append(entity_embeddings[t_id] - entity_embeddings[h_id])
                        # 计算路径特征
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
                    # 路径偏置
                    if path_bias_list:
                        path_bias = torch.stack(path_bias_list, dim=0)
                        attn_logits = attn_logits + self.path_bias_scale * path_bias
                    attn = F.softmax(attn_logits, dim=0)
                    r_emb = torch.sum(attn.unsqueeze(-1) * deltas_tensor, dim=0)
                    # 恢复原始噪声水平
                    r_emb = r_emb + 0.01 * torch.randn_like(r_emb)
                else:
                    # 使用随机初始化而不是零初始化
                    r_emb = 0.1 * torch.randn(self.hidden_dim, device=entity_embeddings.device)
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
        """获取任务关系的嵌入（用于查询时）"""
        return self.compute_task_relation_embedding(relation_name, support_triples, entity_embeddings)


class FKGCDataLoaderWithText(FKGCDataLoader):
    """扩展的FKGC数据加载器，支持文本描述"""

    def __init__(self, in_paths, tokenizer, batch_size=2, max_desc_length=64,
                 model='bert', sp_num=5):
        super().__init__(in_paths, tokenizer, batch_size, max_desc_length,
                         model, sp_num)

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


def train_fkgc_model(*args, **kwargs):
    raise NotImplementedError("train_fkgc_model is deprecated; use main_fkgc.py trainer instead.")


def train_classifier_only(*args, **kwargs):
    raise NotImplementedError("Classifier path removed; scoring uses relation attention now.")


# ==================== FKGC范式方法 ====================

def compute_relation_prototype(support_triples, improved_entity_emb, relation_id, hidden_dim):
    """计算关系原型"""
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
    """混合原型计算"""
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
    """注意力加权原型计算 - 优化版本"""
    current_support = [(h, t) for (h, r, t) in support_triples if r == relation_id]
    if len(current_support) == 0:
        return torch.zeros(hidden_dim, device=improved_entity_emb.device)

    # 使用向量化操作替代循环
    valid_support = [(h, t) for h, t in current_support if
                     h < improved_entity_emb.size(0) and t < improved_entity_emb.size(0)]

    if len(valid_support) == 0:
        return torch.zeros(hidden_dim, device=improved_entity_emb.device)

    h_indices = torch.tensor([h for h, t in valid_support], device=improved_entity_emb.device)
    t_indices = torch.tensor([t for h, t in valid_support], device=improved_entity_emb.device)

    # 批量计算关系表示
    h_embs = improved_entity_emb[h_indices]
    t_embs = improved_entity_emb[t_indices]
    support_relations_tensor = t_embs - h_embs

    # 计算相似度和注意力权重
    similarities = torch.cosine_similarity(query_relation_rep.unsqueeze(0), support_relations_tensor, dim=1)
    scaled_similarities = similarities / temperature
    attention_weights = F.softmax(scaled_similarities, dim=0)

    # 计算加权原型
    attention_prototype = torch.sum(attention_weights.unsqueeze(1) * support_relations_tensor, dim=0)
    return (1 - shrinkage) * attention_prototype


def compute_query_relation_representation(h_emb, t_emb):
    """计算查询的关系表示"""
    return t_emb - h_emb


def compute_fkgc_score(query_relation_rep, relation_prototype, temperature=1.0):
    """FKGC评分"""
    similarity = torch.cosine_similarity(query_relation_rep.unsqueeze(0), relation_prototype.unsqueeze(0), dim=1)
    return similarity / temperature