from collections import defaultdict
import math
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ===================== 读取三元组数据 =====================
# 从三元组文件 path_graph.txt 中读取数据
# 每行格式为：head \t relation \t tail
triples_df = pd.read_csv("path_graph.txt", sep='\t', names=['head', 'relation', 'tail'])

# ===================== 统计每个关系下尾实体的频数 =====================
relation_tail_counts = defaultdict(lambda: defaultdict(int))
for _, row in triples_df.iterrows():
    r = row['relation']
    t = row['tail']
    relation_tail_counts[r][t] += 1

# ===================== 计算每个关系的熵 =====================
relation_entropy = {}
for r, tail_counter in relation_tail_counts.items():
    total = sum(tail_counter.values())
    entropy = 0.0
    for count in tail_counter.values():
        p = count / total
        entropy -= p * math.log(p + 1e-12)  # 避免 log(0)
    if abs(entropy) < 1e-8:
        entropy = 0.0  # 修正 -0.0 为 0.0
    relation_entropy[r] = entropy

# ===================== 归一化熵（Beta_r） =====================
max_entropy = max(relation_entropy.values()) if relation_entropy else 1.0
beta_r_map = {
    r: (entropy / max_entropy)
    for r, entropy in relation_entropy.items()
}

# ===================== 打印熵和 Beta_r =====================
print("Relation\tEntropy\t\tBeta_r (scaled)")
print("=" * 50)
for r in sorted(relation_entropy.keys()):
    entropy = relation_entropy[r]
    beta = beta_r_map[r]
    print(f"{r}\t{entropy:.4f}\t\t{beta:.4f}")

# ===================== 绘制散点图表示关系熵 =====================
# 构建 DataFrame
entropy_df = pd.DataFrame({
    'Relation': list(relation_entropy.keys()),
    'Entropy': list(relation_entropy.values())
})

# 排序并添加索引作为 x 轴
entropy_df = entropy_df.sort_values(by='Entropy', ascending=False).reset_index(drop=True)
entropy_df['Index'] = entropy_df.index

# 设置绘图风格
sns.set(style="whitegrid")
plt.figure(figsize=(12, 6))

# 画散点图
plt.scatter(entropy_df['Index'], entropy_df['Entropy'], color='teal', alpha=0.7, s=50)

# 设置坐标轴和标题
plt.title('Relation Entropy Distribution (Scatter Plot)', fontsize=14)
plt.xlabel('Relation Index (sorted by entropy)', fontsize=12)
plt.ylabel('Entropy', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.5)

# 紧凑布局显示
plt.tight_layout()

# 显示图像
plt.show()
