import json
from factual_scene_graph.parser.scene_graph_parser import SceneGraphParser

# 创建SceneGraphParser实例，指定使用CUDA（如果可用）
parser = SceneGraphParser('D:/text_fusion/CrossFuse-main/lizhuang144flan-t5-base-VG-factual-sg', device='cuda')

# 定义两个JSON文件的路径
vi_captions_path = 'D:/text_fusion/CrossFuse-main/test_imgs/msrs/vi_captions_msrs.json'
ir_captions_path = 'D:/text_fusion/CrossFuse-main/test_imgs/msrs/ir_captions_msrs.json'

# 读取JSON文件中的字典
with open(vi_captions_path, 'r', encoding='utf-8') as file:
    vi_data = json.load(file)

with open(ir_captions_path, 'r', encoding='utf-8') as file:
    ir_data = json.load(file)

# 创建一个空字典来存储新的键值对
new_vi_data = {}
new_ir_data = {}

# 定义一个函数来处理数据
def process_data(data):
    new_data = {}
    for key, text in data.items():
        # 使用SceneGraphParser解析文本
        graph_obj = parser.parse([text], beam_size=5, return_text=False, max_output_len=128)

        # 提取只包含'head'部分的entities列表
        heads_only = [entity['head'] for entity in graph_obj[0]['entities']]

        # 将包含'head'部分的列表与原始键组成新的键值对
        new_data[key] = heads_only
    return new_data

# 处理可见光图像数据
new_vi_data = process_data(vi_data)

# 处理红外图像数据
new_ir_data = process_data(ir_data)

# 打印新的键值对
for key, heads in new_vi_data.items():
    print(f"{key}: {heads}")  # 打印键和实体的'head'部分

for key, heads in new_ir_data.items():
    print(f"{key}: {heads}")  # 打印键和实体的'head'部分

# 将新字典保存为新的JSON文件
with open('D:/text_fusion/CrossFuse-main/test_imgs/msrs/vi_entity_msrs.json', 'w', encoding='utf-8') as file:
    json.dump(new_vi_data, file, ensure_ascii=False, indent=4)

with open('D:/text_fusion/CrossFuse-main/test_imgs/msrs/ir_entity_msrs.json', 'w', encoding='utf-8') as file:
    json.dump(new_ir_data, file, ensure_ascii=False, indent=4)

print("可见光图像描述已保存到 JSON 文件。")
print("红外图像描述已保存到 JSON 文件。")