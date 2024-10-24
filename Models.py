import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch
import utils
import math
from Vit import Token
from Channel_Vit import Cross_Channel
from args import Args
import torch
from sklearn.metrics import accuracy_score
import  json

args = Args()
args_dict = args.get_args()


import torch
import torch.nn.functional as F


def process_tensor(FUSE, n):
    # 获取FUSE张量的形状
    B, current_n, current_dim = FUSE.size()

    # 创建一个与FUSE相同批次大小的张量，用于存储处理后的张量
    processed_FUSE = torch.zeros(B, 2048, 2048, device=FUSE.device)

    for i in range(B):
        # 对每个批次中的张量进行处理
        current_fuse = FUSE[i]
        if current_fuse.size(0) == n:
            # 如果n等于当前张量的第一个维度，不需要改动
            current_fuse = current_fuse[:, :2048]  # 截断到2048
        elif current_fuse.size(0) > n:
            # 如果n小于当前张量的第一个维度，去掉超过n的部分
            current_fuse = current_fuse[:n, :]
        else:
            # 如果n大于当前张量的第一个维度，用0进行padding
            # 计算需要填充的行数
            padding_size = n - current_fuse.size(0)
            # 创建一个形状为[padding_size, current_dim]的0张量
            padding_tensor = torch.zeros(padding_size, current_dim, device=FUSE.device)
            # 将padding_tensor附加到当前张量的末尾
            current_fuse = torch.cat((current_fuse, padding_tensor), dim=0)

        # 如果第二维度超过2048，则进行平均池化
        if current_fuse.size(1) > 2048:
            # 使用adaptive_avg_pool1d进行平均池化，输出大小设置为2048
            current_fuse = F.adaptive_max_pool1d(current_fuse, 2048)

        # 确保处理后的张量维度为[2048, 2048]
        processed_FUSE[i] = current_fuse[:, :2048]

    return processed_FUSE


# Convolution operation 填充卷积激活
class ConvLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, use_Relu=True):
        super(ConvLayer, self).__init__()
        reflection_padding = int(np.floor(kernel_size / 2))
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)
        self.use_Relu = use_Relu
        self.ReLU = nn.LeakyReLU(0.1)

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        if self.use_Relu is True:
            out = self.ReLU(out)
        return out


class Image_Reconstruction(nn.Module):
    def __init__(self):
        super(Image_Reconstruction, self).__init__()
        self.conv_1 = ConvLayer(64, 32, 3, stride=1, use_Relu=True)
        self.conv_2 = ConvLayer(32, 16, 3, stride=1, use_Relu=True)
        self.conv_3 = ConvLayer(16, 1, 1, stride=1, use_Relu=True)

    def forward(self, x):
        x = self.conv_1(x)
        x = self.conv_2(x)
        x = self.conv_3(x)
        return x


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        kernel_size_2 = 3
        self.save = utils.save_feat
        # encoder_convlayer
        self.SFB_ir = nn.Sequential(ConvLayer(1,args_dict['ini_channel'],3,1,True),ConvLayer(args_dict['ini_channel'], args_dict['ini_channel'], 3, 1, True))
        self.SFB_vis = nn.Sequential(ConvLayer(1, args_dict['ini_channel'], 3, 1, True), ConvLayer(args_dict['ini_channel'], args_dict['ini_channel'], 3, 1, True))
        self.T_vit = Token(embed_dim=args_dict['s_vit_embed_dim'],patch_size=args_dict['s_vit_patch_size'],channel=args_dict['ini_channel'])      #64   8
        self.cro_cha_vit = Cross_Channel(embed_dim=128,patch_size=16,channel=args_dict['ini_channel'])
        self.Image_Reconstruction = Image_Reconstruction()

        self.fc1 = nn.Linear(1024*2048, 38)

        self.sigmod = nn.Sigmoid()
        self.relu =nn.ReLU()




    def forward(self,input_ir,input_vis, input_text):


        ir1 = self.SFB_ir(input_ir)# 16    256
        B = ir1.size(0)
        vis1 = self.SFB_vis(input_vis)

        cross_ir = self.cro_cha_vit(ir1, vis1) + ir1
        cross_vis = self.cro_cha_vit(vis1, ir1) + vis1

        cross_text_ir_, out_cla_ir = self.T_vit(cross_ir, input_text)
        cross_text_vis_, out_cla_vis = self.T_vit(cross_vis, input_text)

        cross_text_ir = cross_text_ir_ + cross_ir

        cross_text_vis = cross_text_vis_ + cross_vis

        fuse = torch.cat([cross_text_ir,cross_text_vis], dim=1)

        fuse_calss = torch.cat([out_cla_ir,out_cla_vis], dim=-1)

        compressed_fuse = fuse_calss.view(B,-1)

        compressed_fuse = F.adaptive_max_pool1d(compressed_fuse,1024*2048 )

        classes= self.fc1(compressed_fuse)
        # classes = torch.sigmoid(classes)


        final_out = self.Image_Reconstruction(fuse)

        return final_out,classes



