import torch
import torch.nn as nn
import numpy

from functools import partial
from typing import Optional

class ConvNormAct(nn.Sequential):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        kernel_size: int,
        bn: nn.Module = nn.BatchNorm2d,
        act: nn.Module = nn.ReLU,
        stride = (1, 1),
        **kwargs
    ):

        super().__init__(
            nn.Conv2d(
                in_features,
                out_features,
                kernel_size = kernel_size,
                padding = kernel_size // 2,
                stride=stride,
                **kwargs
            ),
            bn(out_features),
            act(),
        )

class DepthWiseSeparableConv(nn.Sequential):
    def __init__(self, in_features: int, out_features: int, kernel_size=3, bn: nn.Module = nn.BatchNorm2d, act: nn.Module = nn.ReLU):
        super().__init__(
            nn.Conv2d(in_features, in_features, kernel_size=kernel_size, padding=kernel_size // 2, groups=in_features),
            nn.Conv2d(in_features, out_features, kernel_size=1),
            bn(out_features),
            act()
        )

        
class ResidualAdd(nn.Module):
    def __init__(self, block: nn.Module, shortcut: Optional[nn.Module] = None):
        super().__init__()
        self.block = block
        self.shortcut = shortcut
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = x
        x = self.block(x)
        if self.shortcut:
            res = self.shortcut(res)
        x += res
        return x


class FusedMBConv(nn.Sequential):
    def __init__(self, in_features: int, out_features: int, act=nn.ReLU, expansion: int = 4):
        residual = ResidualAdd if in_features == out_features else nn.Sequential
        expanded_features = in_features * expansion
        Conv1X1BnReLU = partial(ConvNormAct, kernel_size=1)
        Conv3X3BnReLU = partial(DepthWiseSeparableConv, kernel_size=3)
        super().__init__(
            nn.Sequential(
                residual(
                    nn.Sequential(
                        Conv3X3BnReLU(in_features, 
                                      expanded_features, 
                                      act=nn.ReLU6
                                     ),
                        # wide -> narrow
                        Conv1X1BnReLU(expanded_features, out_features, act=nn.Identity),
                    ),
                ),
                act(),
            )
        )
        

class Mobile_like_block(nn.Sequential):
    def __init__(self, in_features: int, out_features: int, expansion: int = 4):
        super().__init__(
            nn.Sequential(
                FusedMBConv(in_features, in_features, expansion=expansion),
                FusedMBConv(in_features, out_features, nn.Hardswish, expansion)
            )
        )

class DownsamplingNormAct(nn.Sequential):
    def __init__(self, in_features : int, scale_factor: int = 2, bn: nn.Module = nn.BatchNorm2d, act: nn.Module = nn.ReLU):
        # rho only divisors of 2 and 3!
        ksize = 2 if scale_factor % 2 == 0 else 3
        ConvDownsample = partial(nn.Conv2d, kernel_size=ksize, stride=scale_factor, groups=in_features)
        super().__init__(
            nn.Sequential(
                ConvDownsample(in_features, in_features),
                bn(in_features),
                act()
            )
        )
        
class UpsamplingNormAct(nn.Sequential):
    def __init__(self, in_features : int, scale_factor: int = 2, mode='bilinear', bn: nn.Module = nn.BatchNorm2d, act: nn.Module = nn.ReLU):
        super().__init__(
            nn.Sequential(
                nn.Upsample(scale_factor=scale_factor, mode=mode),
                bn(in_features),
                act()
            )
        )
    
class Mobile_like_encoder(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.features = ConvNormAct(3, 16, kernel_size=3, stride=(2, 2), act=nn.Hardswish, bias=False)
        self.block1_1 = Mobile_like_block(16, 24)
        self.downsample1 = DownsamplingNormAct(24, 2)
        
        self.block2_1 = Mobile_like_block(24, 32)
        self.block2_2 = Mobile_like_block(32, 40)
        self.downsample2 = DownsamplingNormAct(40, 2)
        
        self.block3_1 = Mobile_like_block(40, 48)
        self.block3_2 = Mobile_like_block(48, 96)
        self.block3_3 = Mobile_like_block(96, 576)
        self.downsample3 = DownsamplingNormAct(576, 2)
        
        self.final_features = ConvNormAct(576, 1152, kernel_size=1, stride=(1, 1), act=nn.Hardswish, bias=False)
        
    def forward(self, x):
        res = self.features(x)
        self.residual1 = res
        
        print(res.shape)
        
        res = self.block1_1(res)
        self.residual2 = res
        res = self.downsample1(res)
        
        print(res.shape)
        
        res = self.block2_1(res)
        res = self.block2_2(res)
        self.residual3 = res
        res = self.downsample2(res)
        
        print(res.shape)
        
        res = self.block3_1(res)
        res = self.block3_2(res)
        res = self.block3_3(res)
        self.residual4 = res
        res = self.downsample3(res)
        
        print(res.shape)
        
        res = self.final_features(res)
        print(res.shape)
        
        return res
    
class Mobile_like_decoder(nn.Module):
    def __init__(self, encoder_instance):
        super().__init__()
        
        self.encoder_instance = encoder_instance
        
        self.features = ConvNormAct(1152, 576, kernel_size=1, stride=(1, 1), act=nn.Hardswish, bias=False)
        
        self.upsample1 = nn.Upsample(scale_factor=2, mode='bilinear')# UpsamplingNormAct(576)
        self.block1_1 = Mobile_like_block(1152, 512)
        
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.block2_1 = Mobile_like_block(552, 256)
        self.block2_2 = Mobile_like_block(256, 128)
        
        self.upsample3 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.block3_1 = Mobile_like_block(152, 96)
        self.block3_2 = Mobile_like_block(96, 40)
        self.block3_3 = Mobile_like_block(40, 16)
        
        self.restore = nn.ConvTranspose2d(32, 3, 3, stride=2, padding=1)
        
    def forward(self, x):
        
        res = self.features(x)
        res = self.upsample1(res)
        
        print(res.shape)
        
        res = torch.cat((self.encoder_instance.residual4, res), 1)
        res = self.block1_1(res)
        res = self.upsample2(res)
        
        print(res.shape)
        
        res = torch.cat((self.encoder_instance.residual3, res), 1)
        res = self.block2_1(res)
        res = self.block2_2(res)
        res = self.upsample3(res)
        
        print(res.shape)
        
        res = torch.cat((self.encoder_instance.residual2, res), 1)
        res = self.block3_1(res)
        res = self.block3_2(res)
        res = self.block3_3(res)
        
        print(res.shape)
        
        res = torch.cat((self.encoder_instance.residual1, res), 1)
        res = self.restore(res) #все сводится к тому, надо или не надо тут добавлять upsample
        
        print(res.shape)
        return res