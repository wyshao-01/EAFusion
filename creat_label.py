import pickle
from pathlib import Path
import json

# 定义 JSON 文件路径
num_classes_json_path = Path("D:/text_fusion/CrossFuse-main/num_classes.json")
train_classes_json_path = Path("D:/text_fusion/CrossFuse-main/train_classes.json")

# 读取类别列表
def read_classes(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data['classes']

# 创建多热编码标签向量
def create_one_hot_vector(image_labels, classes):
    one_hot_vector = [0] * len(classes)
    for label in image_labels:
        if label in classes:
            index = classes.index(label)
            one_hot_vector[index] = 1
    return one_hot_vector

# 主函数
def main():
    # 读取所有类别标签
    classes = read_classes(num_classes_json_path)
    print(f"Classes: {classes}")

    # 读取每个图像的标签
    with open(train_classes_json_path, 'r', encoding='utf-8') as file:
        train_data = json.load(file)

    # 创建训练标签字典
    train_label = {}
    for image_name, labels in train_data.items():
        one_hot_vector = create_one_hot_vector(labels, classes)
        train_label[image_name] = one_hot_vector

    # 打印结果
    print("Train label dictionary created:")
    for image_name, label_vector in train_label.items():
        print(f"{image_name}: {label_vector}")

    # 保存训练标签字典到 PKL 文件
    with open('D:/text_fusion/CrossFuse-main/train_label.pkl', 'wb') as f:
        pickle.dump(train_label, f)

if __name__ == '__main__':
    main()