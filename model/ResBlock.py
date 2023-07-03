#!/usr/bin/env python
# -*- coding: utf-8 -*- #
# ==================================================
# File Name:        ResBlock.py
# Author:           Qingzheng WANG
# Time:             2023/7/3 10:46
# Description:                       
# Function List:    
# ===================================================

import torch

class ResBlock:
    def __init__(self, iC, oC, T, F, kernel_time, kernel_freq, stride_time, stride_freq,
                 pad_time, pad_freq, dilation_time, dilation_freq, padding_mode):
        self.iC = iC # input channel
        self.oC = oC # output channel
        self.T = T # time steps
        self.F = F # frequency bins
        self.kernel_time = kernel_time
        self.kernel_freq = kernel_freq
        self.stride_time = stride_time
        self.stride_freq = stride_freq
        self.pad_time = pad_time
        self.pad_freq = pad_freq
        self.dilation_time = dilation_time
        self.dilation_freq = dilation_freq
        self.padding_mode = padding_mode

        self.depth_conv = torch.nn.Conv2d(in_channels=self.iC, out_channels=self.oC,
                                          kernel_size=(self.kernel_time, self.kernel_freq),
                                          groups=self.iC)
        self.point_conv = torch.nn.Conv2d(in_channels=self.iC, out_channels=self.oC, kernel_size=(1, 1))
        self.ds_conv = torch.nn.Sequential(self.depth_conv, self.point_conv) # depthwise separable convolution


    def forward(self, x):
        for i in range(5):
            x = self.ds_conv(x) + x
        y = x
        return y
