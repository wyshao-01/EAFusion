import json

# 定义文件路径
file1 = 'D:/text_fusion/CrossFuse-main/test_imgs/msrs/ir_entity_msrs.json'
file2 = 'D:/text_fusion/CrossFuse-main/test_imgs/msrs/vi_entity_msrs.json'
output_file = 'D:/text_fusion/CrossFuse-main/entity_msrs_test.json'

# 读取第一个JSON文件
with open(file1, 'r', encoding='utf-8') as f:
    data1 = json.load(f)

# 读取第二个JSON文件
with open(file2, 'r', encoding='utf-8') as f:
    data2 = json.load(f)

# 合并两个字典，如果有重复的键，合并它们的值
merged_data = {}

# 获取两个字典的键，并保持原始顺序
keys1 = list(data1.keys())
keys2 = list(data2.keys())

# 合并键，并去除重复项
all_keys = keys1 + [key for key in keys2 if key not in keys1]

for key in all_keys:
    # 获取两个文件中该键对应的值
    value1 = data1.get(key, [])
    value2 = data2.get(key, [])

    # 合并值列表，并去除重复项
    merged_values = list(set(value1 + value2))

    # 将合并后的值保存到merged_data字典中
    merged_data[key] = merged_values

# 保存合并后的数据到新的JSON文件
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(merged_data, f, ensure_ascii=False, indent=4)

print(f"Merged JSON has been saved to {output_file}")