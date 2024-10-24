import json
from factual_scene_graph.parser.scene_graph_parser import SceneGraphParser

# 创建SceneGraphParser实例，指定使用CUDA（如果可用）
parser = SceneGraphParser('D:/text_fusion/CrossFuse-main/lizhuang144flan-t5-base-VG-factual-sg', device='cuda')

# 读取JSON文件中的字典
with open('F:/MSRS-main/train/cropped_vi_captions.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

# 创建一个空字典来存储新的键值对
new_data = {}

# 遍历字典中的每个键值对
for key, text in data.items():
    # 使用SceneGraphParser解析文本
    graph_obj = parser.parse([text], beam_size=5, return_text=False, max_output_len=128)

    # 提取只包含'head'部分的entities列表
    heads_only = [entity['head'] for entity in graph_obj[0]['entities']]

    # 将包含'head'部分的列表与原始键组成新的键值对
    new_data[key] = heads_only


# 打印新的键值对
for key, heads in new_data.items():
    print(f"{key}: {heads}")  # 打印键和实体的'head'部分

# 将新字典保存为新的JSON文件
with open('F:/MSRS-main/train/cropped_vi_captions_new.json', 'w', encoding='utf-8') as file:
    json.dump(new_data, file, ensure_ascii=False, indent=4)