import os
import shutil
import os
from PIL import Image

from transformers import Blip2Processor, Blip2ForConditionalGeneration
import torch
from PIL import Image
import requests
import os
from PIL import Image


from transformers import Blip2Processor, Blip2ForConditionalGeneration
from PIL import Image
import torch
import os
import json
from glob import glob

import os

subset = 'train'
class_ = 'ir'

device = "cuda" if torch.cuda.is_available() else "cpu"

# 加载模型和处理器
processor = Blip2Processor.from_pretrained('D:/text_fusion\CrossFuse-main\BLIP2/2/blip2-opt-6.7b')
model = Blip2ForConditionalGeneration.from_pretrained("D:/text_fusion\CrossFuse-main\BLIP2/2/blip2-opt-6.7b", torch_dtype=torch.float16).to(device)

root_path = f'F:\MSRS-main/{subset}/cropped_{class_}'

with open(f'F:\MSRS-main/{subset}/{subset}_patch_list.json', 'rb') as f:
    patch_list = json.load(f)

# 存储图像名和生成的描述
image_descriptions = {}

# 遍历图像路径
for image_path in patch_list:
    # 加载图像
    image = Image.open(os.path.join(root_path,image_path)+'.png').convert('RGB')

    # 使用处理器处理图像
    inputs = processor(images=image, return_tensors="pt").to(device, torch.float16)
    # 生成图像描述
    generated_ids = model.generate(**inputs, max_new_tokens=50)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    print(generated_text)
    # 获取图像文件名（不含扩展名）
    image_name = os.path.basename(image_path).split('.')[0]

    # 将图像名和生成的描述保存到字典中
    image_descriptions[image_name] = generated_text

# 保存为 JSON 文件
with open(f'F:\MSRS-main/{subset}/cropped_{class_}_captions.json', 'w') as f:
    json.dump(image_descriptions, f)

print("图像描述已保存到 JSON 文件。")

