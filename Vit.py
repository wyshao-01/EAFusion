import numpy as np
from einops import rearrange
from einops.layers.torch import Rearrange
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from args import Args
args = Args()
args_dict = args.get_args()
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
class Cross_modal_gui_Attention(nn.Module):
    def __init__(self, dim_1, dim_2, dim_3, num_heads):
        super(Cross_modal_gui_Attention, self).__init__()
        self.num_heads = num_heads
        self.dim_1 = dim_1
        self.dim_2 = dim_2
        self.dim_3 = dim_3  # 新增的第三个输入的维度

        self.query_linear_1 = nn.Linear(dim_1, dim_1)  # 第一对输入的query
        self.key_linear_1 = nn.Linear(dim_2, dim_1)    # 第一对输入的key
        self.value_linear_1 = nn.Linear(dim_2, dim_1)  # 第一对输入的value

        self.query_linear_2 = nn.Linear(dim_1, dim_1)  # 第二对输入的query
        self.key_linear_2 = nn.Linear(dim_3, dim_1)    # 第二对输入的key
        self.value_linear_2 = nn.Linear(dim_3, dim_1)  # 第二对输入的value

        # Feed Forward Network
        self.linear1 = nn.Linear(dim_1, dim_1 * 2)
        self.dropout = nn.Dropout(0.1)
        self.linear2 = nn.Linear(dim_1 * 2, dim_1)

    def forward(self, x, y, z):
        batch_size = x.size(0)
        num_pixels = x.size(1)

        # 第一对输入的交叉注意力
        query_1 = self.query_linear_1(x)  # [batch_size, num_pixels, dim_1]
        key_1 = self.key_linear_1(y)      # [batch_size, num_texts, dim_1]
        value_1 = self.value_linear_1(y) # [batch_size, num_texts, dim_1]

        query_1 = query_1.view(batch_size, -1, self.num_heads, self.dim_1 // self.num_heads).transpose(1, 2)
        key_1 = key_1.view(batch_size, -1, self.num_heads, self.dim_1 // self.num_heads).transpose(1, 2)
        value_1 = value_1.view(batch_size, -1, self.num_heads, self.dim_1 // self.num_heads).transpose(1, 2)

        scores_1 = torch.matmul(query_1, key_1.transpose(-2, -1)) / math.sqrt(self.dim_1 // self.num_heads)
        attn_1 = F.softmax(scores_1, dim=-1)

        out_1 = torch.matmul(attn_1, value_1)
        out_1 = out_1.transpose(1, 2).contiguous().view(batch_size, num_pixels, self.dim_1)


        out_1 = self.linear2(self.dropout(F.relu(self.linear1(out_1))))

        # 第二对输入的交叉注意力，使用前一对交叉注意力的结果作为query
        query_2 = self.query_linear_2(out_1)  # [batch_size, num_pixels, dim_1]
        key_2 = self.key_linear_2(z)         # [batch_size, num_texts, dim_1]
        value_2 = self.value_linear_2(z)     # [batch_size, num_texts, dim_1]

        query_2 = query_2.view(batch_size, -1, self.num_heads, self.dim_1 // self.num_heads).transpose(1, 2)
        key_2 = key_2.view(batch_size, -1, self.num_heads, self.dim_1 // self.num_heads).transpose(1, 2)
        value_2 = value_2.view(batch_size, -1, self.num_heads, self.dim_1 // self.num_heads).transpose(1, 2)

        scores_2 = torch.matmul(query_2, key_2.transpose(-2, -1)) / math.sqrt(self.dim_1 // self.num_heads)
        attn_2 = F.softmax(scores_2, dim=-1)

        out_2 = torch.matmul(attn_2, value_2)
        out_2 = out_2.transpose(1, 2).contiguous().view(batch_size, num_pixels, self.dim_1)


        out = self.linear2(self.dropout(F.relu(self.linear1(out_2))))

        return out

class SelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(SelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        assert self.head_dim * num_heads == embed_dim

        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)

        # Feed Forward Network
        self.linear1 = nn.Linear(embed_dim, embed_dim * 1)
        self.dropout = nn.Dropout(0.1)
        self.linear2 = nn.Linear(embed_dim * 1, embed_dim)

    def forward(self, x):
        batch_size = x.size(0)
        seq_len = x.size(1)

        # 计算query, key, value
        query = self.query(x)  # [batch, seq_len, embed_dim]
        key = self.key(x)  # [batch, seq_len, embed_dim]
        value = self.value(x)  # [batch, seq_len, embed_dim]

        # 多头处理
        query = query.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # 注意力分数
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = F.softmax(scores, dim=-1)

        # 应用注意力权重
        out = torch.matmul(attn, value)
        out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.embed_dim)

        # 应用dropout
        out = self.dropout(out)

        # 应用FFN
        out = self.linear2(self.dropout(F.relu(self.linear1(out))))

        return out


class ConvLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, is_last=False):
        super(ConvLayer, self).__init__()
        reflection_padding = int(np.floor(kernel_size / 2))
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)
        self.dropout = nn.Dropout2d(p=0.5)
        self.is_last = is_last

    def forward(self, x):

        out = self.reflection_pad(x)
        out = self.conv2d(out)

        if self.is_last is False:
            out = F.relu(out, inplace=True)
        return out



class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class C_DePatch(nn.Module):
    def __init__(self, channel=3, embed_dim=128, patch_size=16):
        self.patch_size = patch_size
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(embed_dim, patch_size ** 2),
        )

    def forward(self, x, ori):
        b, c, h, w = ori
        h_ = h // self.patch_size
        w_ = w // self.patch_size
        x = self.projection(x)
        x = rearrange(x, '(b h w) c (p1 p2) -> b c (h p1) (w p2)', h=h_, w=w_, p1=self.patch_size, p2=self.patch_size)
        return x


class T_DePatch(nn.Module):
    def __init__(self, channel=32, embed_dim=128, patch_size=16):
        self.patch_size = patch_size
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(embed_dim, patch_size ** 2 * channel),
        )
        #self.conv1 = ConvLayer(1, 96, 1, 1)

    def forward(self, x, ori):
        b, c, h, w = ori
        h_ = h // self.patch_size
        w_ = w // self.patch_size
        x = self.projection(x)
        x = rearrange(x, 'b (h w) (p1 p2 c) -> b c (h p1) (w p2)', h=h_, w=w_, p1=self.patch_size, p2=self.patch_size)
        return x


class encoder(nn.Module):
    def __init__(self, embed_dim, depth,
                 num_heads, mlp_ratio=2., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models

        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        return x


class Token(nn.Module):
    def __init__(self, size=256, embed_dim=args_dict['s_vit_embed_dim'], depth=4, channel=32,
                 num_heads=4, mlp_ratio=2., patch_size=args_dict['s_vit_patch_size'], qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm): #depth = 4
        super().__init__()

        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size),
            nn.Linear(patch_size ** 2 * channel, embed_dim),
        )

        self.pos_drop = nn.Dropout(p=drop_rate)


        self.en = encoder(embed_dim, depth,
                          num_heads, mlp_ratio, qkv_bias, qk_scale, drop_rate, attn_drop_rate,
                          drop_path_rate, norm_layer)

        self.depatch = T_DePatch(channel=channel, embed_dim=embed_dim, patch_size=patch_size)
        self.fc = nn.Linear(768, 1024)
        self.corss_image_entity = Cross_modal_gui_Attention(1024, 1024,1024, 8)

        self.norm = nn.LayerNorm(1024)
        self.self_attention = SelfAttention(1024,8)

    def forward(self, x, input_text):

        ori = x.shape
        x2_t = self.embedding(x)
        x2_t = self.pos_drop(x2_t)
        x2_t = self.en(x2_t)

        text_features = input_text
        text_features = self.fc(text_features)
        b=x2_t.shape[1]


        concatenated = torch.cat([ text_features,x2_t], dim=1)

        output = self.self_attention(concatenated)

        output = output[:, -b:, :]



        #output = self.corss_image_entity(x2_t, text_features,x2_t)


        out = self.depatch(output, ori)

        return out, output