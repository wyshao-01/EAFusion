# import json
# from pathlib import Path
# from collections import Counter
#
# # 定义 JSON 文件路径
# json_file_path = Path("D:/text_fusion/CrossFuse-main/cropped_fused_entity.json")
# output_file_path = Path("D:/text_fusion/CrossFuse-main/train_classes.json")
#
# # 读取 JSON 文件
# def read_json_file(file_path):
#     with open(file_path, 'r', encoding='utf-8') as file:
#         data = json.load(file)
#     return data
#
# # 统计所有值出现的次数
# def count_values(data):
#     value_counts = Counter()
#     for value_list in data.values():
#         value_counts.update(value_list)
#     return value_counts
#
# # 过滤每个键的值，只保留排名前N的值
# def filter_values(data, top_n_values):
#     filtered_data = {}
#     for key, values in data.items():
#         filtered_values = [value for value in values if value in top_n_values]
#         filtered_data[key] = filtered_values
#     return filtered_data
#
# # 主函数
# def main(top_n=38):
#     # 读取 JSON 数据
#     data = read_json_file(json_file_path)
#
#     # 获取所有值的出现次数统计
#     value_counts = count_values(data)
#
#     # 获取前N个最频繁出现的值
#     top_n_values = set([value for value, _ in value_counts.most_common(top_n)])
#
#     # 过滤每个键的值，只保留前N名的值
#     filtered_data = filter_values(data, top_n_values)
#
#     # 保存过滤后的数据到新的 JSON 文件
#     with open(output_file_path, 'w', encoding='utf-8') as output_file:
#         json.dump(filtered_data, output_file, indent=4)
#
#     # 打印信息
#     print(f"Filtered data saved to {output_file_path}. Top {top_n} values by occurrence:")
#     for value, count in value_counts.most_common(top_n):
#         print(f"Value: {value}, Count: {count}")
#
# if __name__ == '__main__':
#     # 调用主函数并传递想要的排名数，例如前10、前20等
#     main(top_n=38)  # 修改此数值即可获得任意排名的结果

import json
from pathlib import Path
from collections import Counter

# 定义 JSON 文件路径
json_file_path = Path("D:/text_fusion/CrossFuse-main/cropped_fused_entity.json")
output_file_path = Path("D:/text_fusion/CrossFuse-main/num_classes.json")

# 读取 JSON 文件
def read_json_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data

# 统计所有值出现的次数
def count_values(data):
    value_counts = Counter()
    for value_list in data.values():
        value_counts.update(value_list)
    return value_counts

# 主函数
def main(top_n=38):
    # 读取 JSON 数据
    data = read_json_file(json_file_path)

    # 获取所有值的出现次数统计
    value_counts = count_values(data)

    # 获取前N个最频繁出现的值
    top_n_values = set([value for value, _ in value_counts.most_common(top_n)])

    # 保存前N个最频繁出现的值到 num_classes.json 文件
    with open(output_file_path, 'w', encoding='utf-8') as output_file:
        json.dump({'classes': list(top_n_values)}, output_file, indent=4)

    # 打印信息
    print(f"Top {top_n} classes saved to {output_file_path}.")

if __name__ == '__main__':
    # 调用主函数并传递想要的排名数，例如前10、前20等
    main(top_n=38)  # 修改此数值即可获得任意排名的结果