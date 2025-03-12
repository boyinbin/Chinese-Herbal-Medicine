import torch
import torch.nn as nn
from torchvision.ops import SqueezeExcitation
import torch.nn.functional as F
from timm.models.layers import DropPath
from functools import partial


class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x



class GRN(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=(1, 2), keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x



class multi_representation(nn.Module):
    def __init__(self, in_channel, depth):
        super(multi_representation, self).__init__()
        self.atrous_block1 = nn.Sequential(
            nn.Conv2d(in_channel, depth, 1, 1),
            nn.BatchNorm2d(depth),
            nn.ReLU(inplace=True)
        )
        self.atrous_block2 = nn.Sequential(
            nn.Conv2d(in_channel, depth, 3, 1, padding=1),
            nn.BatchNorm2d(depth),
            nn.ReLU(inplace=True)
        )
        self.atrous_block3 = nn.Sequential(
            nn.Conv2d(in_channel, depth, 3, 1, padding=2, dilation=2),
            nn.BatchNorm2d(depth),
            nn.ReLU(inplace=True)
        )
        self.conv_1x1_output = nn.Sequential(
            nn.Conv2d(depth * 4, depth, 1, 1),
            nn.BatchNorm2d(depth),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        atrous_block1 = self.atrous_block1(x)
        atrous_block2 = self.atrous_block2(x)
        atrous_block3 = self.atrous_block3(x)

        net = self.conv_1x1_output(torch.cat([
            x, atrous_block1, atrous_block2, atrous_block3
        ], dim=1))
        return net



class Block(nn.Module):

    def __init__(self, dim, drop_path=0.):
        super().__init__()
        self.norm = LayerNorm(dim, eps=1e-6)
        # self.pwconv1 = nn.Linear(dim, 4 * dim)  # pointwise/1x1 convs, implemented with linear layers
        self.se = SE_Block(dim)
        self.act = nn.GELU()
        self.grn = GRN(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.mr = multi_representation(dim, dim)  # 2
        self.conv = nn.Conv2d(dim, dim, kernel_size=1, stride=1, padding=0)  # 3

    def forward(self, x):
        input = x  # (N, C, H, W)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)  # LN层归一化
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
        x = (self.mr(x))
        x = self.act(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.grn(x)  # GRN全局响应归一化
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
        x = self.conv(x)  # 1*1卷积
        x = input + self.drop_path(x)  # 残差
        return x



class MSPyraNet(nn.Module):
    def __init__(self, in_chans=3,
                 depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024],
                 drop_path_rate=0.,
                 ):
        super().__init__()
        self.depths = depths
        # 创建下采样列表
        self.downsample_layers = nn.ModuleList()  # stem and 3 intermediate downsampling conv layers
        # stem层
        stem = nn.Sequential(
            nn.Conv2d(3, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)
        # 创建3个降采样层
        for i in range(3):
            downsample_layer = nn.Sequential(
                LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)
        # stages,存储四个stage阶段（特征提取阶段）
        self.stages = nn.ModuleList()  # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[Block(dim=dims[i], drop_path=dp_rates[cur + j]) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)  # final norm layer
        self.classifier = nn.Sequential(
            nn.Linear(1024*1*1, 128),
            nn.ReLU(),
            nn.Linear(128, 20) # 16为分类的个数
        )

    def forward_features(self, x):
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        return self.norm(x.mean([-2, -1]))  # global average pooling, (N, C, H, W) -> (N, C)

    def forward(self, x):  # 32*3*224*224
        x = self.forward_features(x)  # 经过编码器进行特征提取, 32*1024*1*1
        x = x.view(x.size(0), -1)  # 32*1024
        x = self.classifier(x)  # 32*16
        return x
