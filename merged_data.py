import json

# 定义两个JSON文件的路径
captions_json_path = 'F:/MSRS-main/train/cropped_vi_captions.json'
entities_json_path = 'F:/MSRS-main/train/cropped_vi_captions_entity.json'

# 读取两个JSON文件
with open(captions_json_path, 'r', encoding='utf-8') as file:
    captions = json.load(file)

with open(entities_json_path, 'r', encoding='utf-8') as file:
    entities = json.load(file)

# 创建一个空字典来存储合并后的数据
merged_data = {}

# 遍历captions字典
for key in captions:
    # 如果键在两个字典中都存在
    if key in entities:
        # 将caption和entity合并为一个新字典
        merged_data[key] = {
            'caption': captions[key],
            'entity': entities[key]
        }

# 打印合并后的新字典
for key, value in merged_data.items():
    print(f"{key}: {value}")  # 打印键和合并后的值

# 将合并后的新字典保存为新的JSON文件
merged_json_path = 'F:/MSRS-main/train/cropped_vi_captions_merged.json'
with open(merged_json_path, 'w', encoding='utf-8') as file:
    json.dump(merged_data, file, ensure_ascii=False, indent=4)

# import json
#
# # 定义JSON文件路径
# json_file_path = 'D:/text_fusion\CrossFuse-main/cropped_vi_captions_merged.json'
#
# # 读取JSON文件
# with open(json_file_path, 'r', encoding='utf-8') as file:
#     data = json.load(file)
#
# # 获取字典中键值对的数量
# number_of_items = len(data)
#
# # 打印键值对数量
# print(f"The dictionary has {number_of_items} items.")