# import json
#
# # 读取类别名称列表
# with open('num_classes.json', 'r') as f:
#     num_classes_data = json.load(f)
# classes = num_classes_data['classes']
#
# # 创建类别到索引的映射
# class_to_idx = {class_name: idx for idx, class_name in enumerate(classes)}
#
# # 读取测试数据
# with open('entity_msrs_test.json', 'r') as f:
#     entity_msrs_test_data = json.load(f)
#
# # 初始化结果字典
# one_hot_results = {}
#
# # 转换实体标签为多热编码
# for image_name, entities in entity_msrs_test_data.items():
#     # 初始化多热编码向量，所有类别初始为0
#     one_hot_vector = [0] * len(classes)
#
#     for entity in entities:
#         # 处理可能的类别名称不一致问题，例如多余的空格或大小写不一致
#         normalized_entity = entity.strip().lower()
#         if normalized_entity in class_to_idx:
#             # 如果实体在类别映射中，将对应索引的位置设置为1
#             one_hot_vector[class_to_idx[normalized_entity]] = 1
#         else:
#             # 如果实体不在类别映射中，可以选择跳过或记录
#             print(f"Entity '{entity}' in image '{image_name}' not found in num_classes.json")
#
#     # 将多热编码向量保存到结果字典
#     one_hot_results[image_name] = one_hot_vector
#
# # 将结果写回一个新的JSON文件
# with open('entity_msrs_test_one_hot.json', 'w') as f:
#     json.dump(one_hot_results, f, indent=4)

import json

# 读取all_results.json文件
with open(r'D:/text_fusion/CrossFuse-main/results/all_results.json', 'r') as f:
    all_results = json.load(f)

# 读取num_classes.json文件
with open(r'D:/text_fusion/CrossFuse-main/num_classes.json', 'r') as f:
    num_classes = json.load(f)
    classes = num_classes['classes']

# 创建一个新的字典，用于存储图像名和对应的类别列表
results_with_classes = {}

# 遍历all_results中的每个图像结果
for image_name, hot_encodings in all_results.items():
    # 由于hot_encodings是一个列表，我们取出第一个元素（也是唯一的元素）
    hot_encoding = hot_encodings[0]
    # 将多热编码转换为类别列表
    classes_list = [classes[i] for i, v in enumerate(hot_encoding) if v == 1]
    # 将类别列表添加到新字典中
    results_with_classes[image_name] = classes_list

# 将新的字典保存为新的json文件
with open(r'D:/text_fusion/CrossFuse-main/results/all_results_with_classes.json', 'w') as f:
    json.dump(results_with_classes, f, indent=4)

print("Done. The new dictionary has been saved as 'all_results_with_classes.json'.")