
from torch import nn
import torch
import torch.nn.functional as F

def gaussian_window(size, sigma):
    coords = torch.arange(size, dtype=torch.float)
    coords -= size // 2

    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g /= g.sum()

    return g.view(1, 1, 1, -1)


def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(1)
    window = gaussian_window(window_size, 1.5).to(img1.device)
    window = window.expand(channel, 1, 1, window_size)

    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

def histogram_equalization(tensor_image, clip_limit=0.1, gain=1):
    """
    在Tensor上进行直方图均衡化。

    参数:
    tensor_image (torch.Tensor): 一个单通道的灰度图像Tensor，值的范围在[0, 1]。
    clip_limit (float): 对比度限制参数，用于防止剪切直方图时过度增强噪声。默认为0.01。
    gain (float): 对比度增强因子，用于控制对比度增强的程度。默认为1.0。

    返回:
    torch.Tensor: 直方图均衡化后的图像Tensor，值的范围在[0, 1]。
    """
    # 将Tensor的值缩放到[0, 255]并转换为整数
    tensor_image_int = (tensor_image * 255).to(torch.int32)

    # 计算直方图
    hist = torch.histc(tensor_image_int, bins=256, min=0, max=255)

    # 计算直方图的累积分布函数（CDF）
    cdf = torch.cumsum(hist, dim=0)

    # 归一化CDF
    cdf_normalized = cdf / cdf[-1]

    # 应用对比度限制和增益
    cdf_normalized = torch.clamp(cdf_normalized, min=clip_limit, max=1 - clip_limit)
    cdf_normalized = torch.sigmoid(gain * (cdf_normalized - 0.5))

    # 将CDF映射到[0, 255]的范围
    cdf_normalized = (cdf_normalized * 255).to(torch.int32)

    # 创建一个包含所有灰度级映射到新灰度级的映射表
    mapping = cdf_normalized[tensor_image_int]

    # 将结果缩放回[0, 1]范围
    equalized_tensor_image = mapping.float() / 255.0

    return equalized_tensor_image


class g_content_loss(nn.Module):
    def __init__(self):
        super(g_content_loss, self).__init__()
        self.L2_loss = nn.MSELoss()
        self.L1_loss = torch.nn.L1Loss(reduction="mean")

        self.gradient=gradient()




    def forward(self, img_ir, img_vi, img_fusion):
        lambda_2=1
        lambda_3=10
        image_vi_grad = self.gradient(img_vi)
        image_ir_grad = self.gradient(img_ir)
        image_fusion_grad = self.gradient(img_fusion)
        image_max_grad = torch.round((image_vi_grad + image_ir_grad) // (
                torch.abs(image_vi_grad + image_ir_grad) + 0.0000000001)) * torch.max(
            torch.abs(image_vi_grad), torch.abs(image_ir_grad))

        image_ir = img_ir

        image_vi = img_vi


        image_max_int = torch.round((image_vi + image_ir) // (
                torch.abs(image_vi + image_ir) + 0.0000000001)) * torch.max(
            torch.abs(image_vi), torch.abs(image_ir))


        intensity_loss = 1*self.L1_loss(img_fusion, image_max_int)
        grad_loss = 15 * self.L1_loss(image_fusion_grad, image_max_grad)
        ssim_loss= 5*((1-ssim(img_fusion,img_vi))+ (1-ssim(img_fusion,img_ir)))

        content_loss = intensity_loss + grad_loss + ssim_loss
        return content_loss,  intensity_loss, grad_loss




class gradient(nn.Module):
    def __init__(self,channels=1):
        super(gradient, self).__init__()
        laplacian_kernel = torch.tensor([[1/8,1/8,1/8],[1/8,-1,1/8],[1/8,1/8,1/8]]).float()

        laplacian_kernel = laplacian_kernel.view(1, 1, 3, 3)
        laplacian_kernel = laplacian_kernel.repeat(channels, 1, 1, 1)
        self.laplacian_filter = nn.Conv2d(in_channels=channels, out_channels=channels,
                            kernel_size=3, groups=channels, bias=False)

        self.laplacian_filter.weight.data = laplacian_kernel
        self.laplacian_filter.weight.requires_grad = False
    def forward(self,x):
        return self.laplacian_filter(x) ** 2


