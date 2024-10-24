from PIL import Image
import os


def resize_images(input_folder, output_folder, size=(256, 256)):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            img_path = os.path.join(input_folder, filename)
            img = Image.open(img_path)
            img = img.resize(size, Image.LANCZOS )  # 使用ANTIALIAS滤波器来保持图像质量
            img.save(os.path.join(output_folder, filename))


# 指定输入和输出文件夹
input_folder = 'D:\mmasia\CrossFuse-main/test_imgs/tno'
output_folder = 'D:\mmasia\CrossFuse-main/test_imgs/1'

# 调用函数
resize_images(input_folder, output_folder)