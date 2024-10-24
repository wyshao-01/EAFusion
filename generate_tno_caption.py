from transformers import Blip2Processor, Blip2ForConditionalGeneration
from PIL import Image
import torch
import os
import json
from glob import glob

# 确定设备
device = "cuda" if torch.cuda.is_available() else "cpu"

# 加载模型和处理器
processor = Blip2Processor.from_pretrained('D:/text_fusion/CrossFuse-main/BLIP2/2/blip2-opt-6.7b')
model = Blip2ForConditionalGeneration.from_pretrained("D:/text_fusion/CrossFuse-main/BLIP2/2/blip2-opt-6.7b", torch_dtype=torch.float16).to(device)

ir_root_path = 'D:/text_fusion/CrossFuse-main/test_imgs/msrs/ir'
vi_root_path = 'D:/text_fusion/CrossFuse-main/test_imgs/msrs/rgb'

# 定义一个函数来处理图像并生成描述
def generate_descriptions(root_path, descriptions_dict):
    # 获取所有图像路径
    image_paths = glob(os.path.join(root_path, '*.png'))

    for image_path in image_paths:
        # 加载图像
        image = Image.open(image_path).convert('RGB')

        # 使用处理器处理图像
        inputs = processor(images=image, return_tensors="pt").to(device, torch.float16)

        # 生成图像描述
        generated_ids = model.generate(**inputs, max_new_tokens=50)
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        print(generated_text)

        # 获取图像文件名（不含扩展名）
        image_name = os.path.basename(image_path).split('.')[0]

        # 将图像名和生成的描述保存到字典中
        descriptions_dict[image_name] = generated_text

# 存储图像名和生成的描述
image_descriptions_ir = {}
image_descriptions_vi = {}

# 生成红外图像的描述
generate_descriptions(ir_root_path, image_descriptions_ir)
# 生成可见光图像的描述
generate_descriptions(vi_root_path, image_descriptions_vi)

# 保存为 JSON 文件
with open('D:/text_fusion/CrossFuse-main/test_imgs/msrs/ir_captions_msrs.json', 'w', encoding='utf-8') as f:
    json.dump(image_descriptions_ir, f, ensure_ascii=False, indent=4)

with open('D:/text_fusion/CrossFuse-main/test_imgs/msrs/vi_captions_msrs.json', 'w', encoding='utf-8') as f:
    json.dump(image_descriptions_vi, f, ensure_ascii=False, indent=4)

print("图像描述已保存到 JSON 文件。")