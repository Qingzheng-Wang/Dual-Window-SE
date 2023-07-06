#!/usr/bin/env python
# -*- coding: utf-8 -*- #
# ==================================================
# File Name:        Blocks.py
# Author:           Qingzheng WANG
# Time:             2023/7/3 14:02
# Description:      Blocks in DNN, including ResBlock,
#                   Deconv2D+PReLU+BN, Conv2D+PReLU+BN,
#                   Conv2D+cauLN, Deconv2D
# Function List:    
# ===================================================

import torch
from complexPyTorch.complexLayers import ComplexConv2d, ComplexBatchNorm2d, ComplexConvTranspose2d, ComplexReLU

class ResBlock(torch.nn.Module):
    def __init__(self, iFM, oFM, T, F, opt, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.iFM = iFM # input Feature Maps
        self.oFM = oFM # output Feature Maps
        self.T = T # time steps
        self.F = F # frequency bins
        self.opt = opt # options, opt[self.type]["resblock"]

        self.depth_conv1 = ComplexConv2d(in_channels=self.iFM, out_channels=self.oFM,
                                           kernel_size=opt["depth_conv1"]["kernel_size"],
                                           stride=opt["depth_conv1"]["stride"],
                                           padding=opt["depth_conv1"]["padding"],
                                           dilation=opt["depth_conv1"]["dilation"],
                                           groups=self.iFM,
                                           bias=False)
        self.point_conv1 = ComplexConv2d(in_channels=self.iFM, out_channels=self.oFM,
                                           kernel_size=opt["point_conv1"]["kernel_size"],
                                           bias=False)
        self.prelu1 = ComplexReLU()
        self.bn1 = ComplexBatchNorm2d(num_features=self.oFM)
        self.ds_conv1 = torch.nn.Sequential(self.depth_conv1, self.point_conv1, self.prelu1, self.bn1)  # depthwise separable convolution

        self.depth_conv2 = ComplexConv2d(in_channels=self.oFM, out_channels=self.oFM,
                                           kernel_size=opt["depth_conv2"]["kernel_size"],
                                           stride=opt["depth_conv2"]["stride"],
                                           padding=opt["depth_conv2"]["padding"],
                                           dilation=opt["depth_conv2"]["dilation"],
                                           groups=self.oFM,
                                           bias=False)
        self.point_conv2 = ComplexConv2d(in_channels=self.oFM, out_channels=self.oFM,
                                           kernel_size=opt["point_conv2"]["kernel_size"],
                                           bias=False)
        self.prelu2 = ComplexReLU()
        self.bn2 = ComplexBatchNorm2d(num_features=self.oFM)
        self.ds_conv2 = torch.nn.Sequential(self.depth_conv2, self.point_conv2, self.prelu2, self.bn2)

        self.depth_conv3 = ComplexConv2d(in_channels=self.oFM, out_channels=self.oFM,
                                           kernel_size=opt["depth_conv3"]["kernel_size"],
                                           stride=opt["depth_conv3"]["stride"],
                                           padding=opt["depth_conv3"]["padding"],
                                           dilation=opt["depth_conv3"]["dilation"],
                                           groups=self.oFM,
                                           bias=False)
        self.point_conv3 = ComplexConv2d(in_channels=self.oFM, out_channels=self.oFM,
                                           kernel_size=opt["point_conv3"]["kernel_size"],
                                           bias=False)
        self.prelu3 = ComplexReLU()
        self.bn3 = ComplexBatchNorm2d(num_features=self.oFM)
        self.ds_conv3 = torch.nn.Sequential(self.depth_conv3, self.point_conv3, self.prelu3, self.bn3)

        self.depth_conv4 = ComplexConv2d(in_channels=self.oFM, out_channels=self.oFM,
                                           kernel_size=opt["depth_conv4"]["kernel_size"],
                                           stride=opt["depth_conv4"]["stride"],
                                           padding=opt["depth_conv4"]["padding"],
                                           dilation=opt["depth_conv4"]["dilation"],
                                           groups=self.oFM,
                                           bias=False)
        self.point_conv4 = ComplexConv2d(in_channels=self.oFM, out_channels=self.oFM,
                                           kernel_size=opt["point_conv4"]["kernel_size"],
                                           bias=False)
        self.prelu4 = ComplexReLU()
        self.bn4 = ComplexBatchNorm2d(num_features=self.oFM)
        self.ds_conv4 = torch.nn.Sequential(self.depth_conv4, self.point_conv4, self.prelu4, self.bn4)

        self.depth_conv5 = ComplexConv2d(in_channels=self.oFM, out_channels=self.oFM,
                                           kernel_size=opt["depth_conv5"]["kernel_size"],
                                           stride=opt["depth_conv5"]["stride"],
                                           padding=opt["depth_conv5"]["padding"],
                                           dilation=opt["depth_conv5"]["dilation"],
                                           groups=self.oFM,
                                           bias=False)
        self.point_conv5 = ComplexConv2d(in_channels=self.oFM, out_channels=self.oFM,
                                           kernel_size=opt["point_conv5"]["kernel_size"],
                                           bias=False)
        self.prelu5 = ComplexReLU()
        self.bn5 = ComplexBatchNorm2d(num_features=self.oFM)
        self.ds_conv5 = torch.nn.Sequential(self.depth_conv5, self.point_conv5, self.prelu5, self.bn5)

    def forward(self, x):
        x = x + self.ds_conv1(x)
        x = x + self.ds_conv2(x)
        x = x + self.ds_conv3(x)
        x = x + self.ds_conv4(x)
        x = x + self.ds_conv5(x)
        return x

class Deconv2DPReLUBN(torch.nn.Module):
    def __init__(self, iFM, oFM, T, F, kernel_time, kernel_freq, stride_time, stride_freq, pad_time, pad_freq,
                 dilation_time, dilation_freq, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.iFM = iFM  # input Feature Maps
        self.oFM = oFM  # output Feature Maps
        self.T = T  # time steps
        self.F = F  # frequency bins
        self.kernel_time = kernel_time
        self.kernel_freq = kernel_freq
        self.stride_time = stride_time
        self.stride_freq = stride_freq
        self.pad_time = pad_time
        self.pad_freq = pad_freq
        self.dilation_time = dilation_time
        self.dilation_freq = dilation_freq

        self.deconv = ComplexConvTranspose2d(in_channels=self.iFM, out_channels=self.oFM,
                                               kernel_size=(self.kernel_time, self.kernel_freq),
                                               stride=(self.stride_time, self.stride_freq),
                                               padding=(self.pad_time, self.pad_freq),
                                               dilation=(self.dilation_time, self.dilation_freq))
        self.prelu = ComplexReLU()
        self.bn = ComplexBatchNorm2d(num_features=self.oFM)

    def forward(self, x):
        x = self.deconv(x)
        x = self.prelu(x)
        x = self.bn(x)
        return x

class Conv2DPReLUBN(torch.nn.Module):
    def __init__(self, iFM, oFM, T, F, kernel_time, kernel_freq, stride_time, stride_freq, pad_time, pad_freq,
                 dilation_time, dilation_freq, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.iFM = iFM  # input Feature Maps
        self.oFM = oFM  # output Feature Maps
        self.T = T  # time steps
        self.F = F  # frequency bins
        self.kernel_time = kernel_time
        self.kernel_freq = kernel_freq
        self.stride_time = stride_time
        self.stride_freq = stride_freq
        self.pad_time = pad_time
        self.pad_freq = pad_freq
        self.dilation_time = dilation_time
        self.dilation_freq = dilation_freq

        self.conv = ComplexConv2d(in_channels=self.iFM, out_channels=self.oFM,
                                    kernel_size=(self.kernel_time, self.kernel_freq),
                                    stride=(self.stride_time, self.stride_freq),
                                    padding=(self.pad_time, self.pad_freq),
                                    dilation=(self.dilation_time, self.dilation_freq),
                                    bias=False)
        self.prelu = ComplexReLU()
        self.bn = ComplexBatchNorm2d(num_features=self.oFM)

    def forward(self, x):
        x = self.conv(x)
        x = self.prelu(x)
        x = self.bn(x)
        return x

class Conv2DCauLN(torch.nn.Module):
    def __init__(self, iFM, oFM, T, F, kernel_time, kernel_freq, stride_time, stride_freq, pad_time, pad_freq,
                 dilation_time, dilation_freq, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.iFM = iFM  # input Feature Maps
        self.oFM = oFM  # output Feature Maps
        self.T = T  # time steps
        self.F = F  # frequency bins
        self.kernel_time = kernel_time
        self.kernel_freq = kernel_freq
        self.stride_time = stride_time
        self.stride_freq = stride_freq
        self.pad_time = pad_time
        self.pad_freq = pad_freq
        self.dilation_time = dilation_time
        self.dilation_freq = dilation_freq

        self.conv = ComplexConv2d(in_channels=self.iFM, out_channels=self.oFM,
                                    kernel_size=(self.kernel_time, self.kernel_freq),
                                    stride=(self.stride_time, self.stride_freq),
                                    padding=(self.pad_time, self.pad_freq),
                                    dilation=(self.dilation_time, self.dilation_freq),
                                    bias=False)
        # self.ln = torch.nn.LayerNorm(normalized_shape=[self.oFM, self.T, self.F-2])

    def forward(self, x):
        x = self.conv(x)
        # x = self.ln(x)
        return x

class Deconv2D(torch.nn.Module):
    def __init__(self, iFM, oFM, T, F, kernel_time, kernel_freq, stride_time, stride_freq, pad_time, pad_freq,
                 dilation_time, dilation_freq, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.iFM = iFM  # input Feature Maps
        self.oFM = oFM  # output Feature Maps
        self.T = T  # time steps
        self.F = F  # frequency bins
        self.kernel_time = kernel_time
        self.kernel_freq = kernel_freq
        self.stride_time = stride_time
        self.stride_freq = stride_freq
        self.pad_time = pad_time
        self.pad_freq = pad_freq
        self.dilation_time = dilation_time
        self.dilation_freq = dilation_freq

        self.deconv = ComplexConvTranspose2d(in_channels=self.iFM, out_channels=self.oFM,
                                               kernel_size=(self.kernel_time, self.kernel_freq),
                                               stride=(self.stride_time, self.stride_freq),
                                               padding=(self.pad_time, self.pad_freq),
                                               dilation=(self.dilation_time, self.dilation_freq),
                                               bias=False)

    def forward(self, x):
        x = self.deconv(x)
        return x
