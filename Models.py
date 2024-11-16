import torch.nn as nn
import numpy as np
import utils
from Vit import Token
from Channel_Vit import Cross_Channel
from args import Args
import torch
import torch.nn.functional as F

args = Args()
args_dict = args.get_args()


class ConvLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, use_Relu=True):
        super(ConvLayer, self).__init__()
        reflection_padding = int(np.floor(kernel_size / 2))
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)
        self.ReLU = nn.LeakyReLU(0.1)
        self.use_Relu = use_Relu

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        if self.use_Relu is True:
            out = self.ReLU(out)
        return out


class Image_Reconstruction(nn.Module):
    def __init__(self):
        super(Image_Reconstruction, self).__init__()
        self.conv_1 = ConvLayer(64, 32, 1, stride=1, use_Relu=True)
        self.conv_2 = ConvLayer(32, 16, 1, stride=1, use_Relu=True)
        self.conv_3 = ConvLayer(16, 1, 1, stride=1, use_Relu=True)

    def forward(self, x):
        x = self.conv_1(x)
        x = self.conv_2(x)
        x = self.conv_3(x)
        return x


class EGNet(nn.Module):
    def __init__(self):
        super(EGNet, self).__init__()
        kernel_size_2 = 3
        self.save = utils.save_feat
        # encoder_convlayer
        self.SFB_ir = nn.Sequential(ConvLayer(1,args_dict['ini_channel'],3,1,True),ConvLayer(args_dict['ini_channel'], args_dict['ini_channel'], 3, 1, True))
        self.SFB_vis = nn.Sequential(ConvLayer(1, args_dict['ini_channel'], 3, 1, True), ConvLayer(args_dict['ini_channel'], args_dict['ini_channel'], 3, 1, True))
        self.T_vit = Token(embed_dim=args_dict['s_vit_embed_dim'],patch_size=args_dict['s_vit_patch_size'],channel=args_dict['ini_channel'])      #64   8
        self.cro_cha_vit = Cross_Channel(embed_dim=128,patch_size=16,channel=args_dict['ini_channel'])
        self.Image_Reconstruction = Image_Reconstruction()

        self.fc1 = nn.Linear(4194304//16, 9)

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

        cross_text_ir = (cross_text_ir_ + cross_ir)
        cross_text_vis = (cross_text_vis_ + cross_vis)

        fuse = torch.cat([cross_text_ir,cross_text_vis], dim=1)

        compressed_fuse = fuse.view(B,-1)

        cross_fuse = F.adaptive_max_pool1d(compressed_fuse,4194304//16)

        classes= self.fc1(cross_fuse)

        final_out = self.Image_Reconstruction(fuse)

        return final_out,classes





