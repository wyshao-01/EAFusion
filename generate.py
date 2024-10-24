from args import Args
import utils
from torch.autograd import Variable
from Models import Generator
import torch
import os
from utils import make_floor
from imageio import imsave
import numpy as np
import pickle
import torch.nn.functional as F
import json
from PIL import Image

args = Args()
args_dict = args.get_args()
def _generate_fusion_image(G_model, ir_img, vis_img,text):

    f= G_model(ir_img, vis_img,text)



    return f

def load_model(model_path):
    G_model = Generator()
    G_model.load_state_dict(torch.load(model_path))
    print('# generator parameters:', sum(param.numel() for param in G_model.parameters()))
    G_model.eval()
    G_model.cuda()
    return G_model

def load_text_features(text_path, image_name):
    # 加载.pkl文件中的文本特征
    with open(text_path, 'rb') as f:
        text_features = pickle.load(f)
    return text_features.get(image_name, None)  # 使用get方法来安全地获取特征向量

def generate(model, ir_path, vis_path, text_path, result, mode):
    result_path = "results"
    ir_img = utils.get_test_images(ir_path, mode=mode)
    vis_img = utils.get_test_images(vis_path, mode=mode)

    out = utils.get_image(vis_path, height=None, width=None)
    ir_img = ir_img.cuda()
    vis_img = vis_img.cuda()
    ir_img = Variable(ir_img, requires_grad=False)
    vis_img = Variable(vis_img, requires_grad=False)

    # 检查 text_path 是否为字符串路径
    if isinstance(text_path, str):
        # 如果是字符串路径，则加载文本特征
        text = torch.from_numpy(np.load(text_path)).cuda()
    else:
        # 如果 text_path 已经是一个 NumPy 数组，则直接转换为张量
        text = torch.from_numpy(text_path).cuda()

    # 找到所需的最大长度
    max_length = max(tensor.size(0) for tensor in text)  # 假设的最大长度

    # 对张量进行填充，以确保它具有所需的长度
    padded_text = torch.nn.functional.pad(text, (0, 0, 0, max_length - text.size(0))) if text.size(0) < max_length else text

    # 将填充后的文本特征张量移动到GPU（如果尚未移动）
    padded_text = padded_text.cuda()

    # 确保结果文件夹存在
    os.makedirs(result_path, exist_ok=True)

    ir_filename = os.path.basename(ir_path)
    ir_filename_without_ext = os.path.splitext(ir_filename)[0]  # 移除文件扩展名

    img_fusion, classes = _generate_fusion_image(model, ir_img, vis_img, padded_text)

    probabilities = torch.sigmoid(classes)
    # 设置阈值，这里以0.5为例
    threshold = 0.5
    # 根据概率值确定类别
    predicted_classes = (probabilities > threshold).float()
    print(predicted_classes)

    predicted_classes_np = predicted_classes.cpu().numpy()

    # 构造 JSON 文件名，使用红外图像名（不包括.png后缀）加上 _predicted_classes 作为文件名
    json_filename = f"{ir_filename_without_ext}_predicted_classes.json"

    # 构造输出文件的完整路径
    json_output_path = os.path.join(result_path, json_filename)

    # 创建一个字典，键是图像名（不包括.png后缀），值是对应的多热编码
    all_results = {}  # 如果这是第一次运行，初始化字典
    if os.path.exists(os.path.join(result_path, 'all_results.json')):
        with open(os.path.join(result_path, 'all_results.json'), 'r') as f:
            all_results = json.load(f)

    all_results[ir_filename_without_ext] = predicted_classes_np.tolist()

    # 将所有结果保存为一个大的 JSON 文件
    with open(os.path.join(result_path, 'all_results.json'), 'w') as f:
        json.dump(all_results, f, indent=4)

    img_fusion = img_fusion * 255

    img_fusion = img_fusion.squeeze()
    img_fusion = img_fusion

    # 保存融合后的图像
    output_path = os.path.join(result_path, ir_filename)
    img_fusion = img_fusion.clamp(0, 255).cpu().data.numpy().astype(np.uint8)
    Image.fromarray(img_fusion).save(output_path)

    utils.save_images(output_path, img_fusion, out)
# 确保在调用generate函数时传递正确的参数
# generate(model, ir_path, vis_path, ir_text_path, vi_text_path, model_path, index, mode='L')



