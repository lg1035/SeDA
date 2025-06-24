from collections import defaultdict
import math
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ==== 计算单个图的关系熵函数 ====
def compute_relation_entropy(filepath):
    triples_df = pd.read_csv(filepath, sep='\t', names=['head', 'relation', 'tail'])
    relation_tail_counts = defaultdict(lambda: defaultdict(int))

    for _, row in triples_df.iterrows():
        r = row['relation']
        t = row['tail']
        relation_tail_counts[r][t] += 1

    relation_entropy = {}
    for r, tail_counter in relation_tail_counts.items():
        total = sum(tail_counter.values())
        entropy = 0.0
        for count in tail_counter.values():
            p = count / total
            entropy -= p * math.log(p + 1e-12)
        if abs(entropy) < 1e-8:
            entropy = 0.0
        relation_entropy[r] = entropy

    return relation_entropy

# ==== 分别读取两个图的关系熵 ====
entropy_1 = compute_relation_entropy("path_graph.txt")
entropy_2 = compute_relation_entropy("../FB15K237-one/path_graph.txt")

# ==== 构建 DataFrame ====
df1 = pd.DataFrame({
    'Relation': list(entropy_1.keys()),
    'Entropy': list(entropy_1.values()),
    'Source': 'Graph1'
})
df2 = pd.DataFrame({
    'Relation': list(entropy_2.keys()),
    'Entropy': list(entropy_2.values()),
    'Source': 'Graph2'
})

# 合并两个图的数据
combined_df = pd.concat([df1, df2], ignore_index=True)

# 为了可视化效果添加排序索引
combined_df = combined_df.sort_values(by='Entropy', ascending=False).reset_index(drop=True)
combined_df['Index'] = combined_df.index

# ==== 绘制散点图 ====
sns.set(style="whitegrid")
plt.figure(figsize=(14, 6))

# 使用颜色区分两个图
sns.scatterplot(data=combined_df, x='Index', y='Entropy', hue='Source', palette='Set1', s=60, alpha=0.7)

plt.title('Relation Entropy Comparison: Graph1 vs Graph2', fontsize=14)
plt.xlabel('Relation Index (sorted by entropy)', fontsize=12)
plt.ylabel('Entropy', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.legend(title='Source')
plt.show()
