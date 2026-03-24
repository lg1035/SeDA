import json

# 假设您的 JSON 数据存储在名为 'ent2ids.json' 的文件中
with open('ent2ids', 'r', encoding='utf-8') as json_file:
    ent2ids = json.load(json_file)

# 创建一个新的字典，用于存储去除 'concept:' 前缀的键和值
cleaned_ent2ids = {key.replace('concept:', '', 1): value for key, value in ent2ids.items()}

# 根据编号（ID）对字典进行升序排序
sorted_ent2ids = dict(sorted(cleaned_ent2ids.items(), key=lambda item: item[1]))

# 将排序后的键值对写入文本文件
with open('ent2ids.txt', 'w', encoding='utf-8') as txt_file:
    for key, value in sorted_ent2ids.items():
        txt_file.write(f"{key}\t{value}\n")
