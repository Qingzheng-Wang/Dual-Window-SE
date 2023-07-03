#!/usr/bin/env python
# -*- coding: utf-8 -*- #
# ==================================================
# File Name:        DNN.py
# Author:           Qingzheng WANG
# Time:             2023/7/1 15:46
# Description:      implementation of DNN1
# Function List:    
# ===================================================

import torch
from Blocks import *

class DNN(torch.nn.Module):
    def __init__(self, iFM, oFM, T, F, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.iFM = iFM # input Feature Maps
        self.T = T # time steps
        self.F = F # frequency bins
        self.oFM = oFM # output Feature Maps

        self.conv2d_cauln = Conv2DCauLN(iFM=self.iFM, oFM=30, T=self.T, F=self.F,
                                        kernel_time=1, kernel_freq=3, stride_time=1, stride_freq=1,
                                        pad_time=0, pad_freq=0, dilation_time=1, dilation_freq=1)

        self.resblock = ResBlock(iFM=30, oFM=30, T=self.T, F=self.F)

        self.conv2d_prelu_bn30 = Conv2DPReLUBN(iFM=30, oFM=30, T=self.T, F=self.F,
                                              kernel_time=1, kernel_freq=3, stride_time=1, stride_freq=2,
                                              pad_time=0, pad_freq=0, dilation_time=1, dilation_freq=1)
        self.conv2d_prelu_bn64 = Conv2DPReLUBN(iFM=30, oFM=64, T=self.T, F=self.F,
                                               kernel_time=1, kernel_freq=3, stride_time=1, stride_freq=2,
                                               pad_time=0, pad_freq=0, dilation_time=1, dilation_freq=1)
        self.conv2d_prelu_bn192 = Conv2DPReLUBN(iFM=64, oFM=192, T=self.T, F=self.F,
                                                kernel_time=1, kernel_freq=3, stride_time=1, stride_freq=1,
                                                pad_time=0, pad_freq=0, dilation_time=1, dilation_freq=1)

        self.deconv2d_prelu_bn6430 = Deconv2DPReLUBN(iFM=64, oFM=30, T=self.T, F=self.F,
                                                   kernel_time=1, kernel_freq=3, stride_time=1, stride_freq=2,
                                                   pad_time=0, pad_freq=0, dilation_time=1, dilation_freq=1)
        self.deconv2d_prelu_bn30 = Deconv2DPReLUBN(iFM=30, oFM=30, T=self.T, F=self.F,
                                                   kernel_time=1, kernel_freq=3, stride_time=1, stride_freq=2,
                                                   pad_time=0, pad_freq=0, dilation_time=1, dilation_freq=1)
        self.deconv2d_prelu_bn64 = Deconv2DPReLUBN(iFM=192, oFM=64, T=self.T, F=self.F,
                                                   kernel_time=1, kernel_freq=3, stride_time=1, stride_freq=1,
                                                   pad_time=0, pad_freq=0, dilation_time=1, dilation_freq=1)

        self.skip_connection30 = Conv2DPReLUBN(iFM=30, oFM=30, T=self.T, F=self.F,
                                               kernel_time=1, kernel_freq=3, stride_time=1, stride_freq=1,
                                               pad_time=0, pad_freq=0, dilation_time=1, dilation_freq=1)
        self.skip_connection64 = Conv2DPReLUBN(iFM=64, oFM=64, T=self.T, F=self.F,
                                               kernel_time=1, kernel_freq=1, stride_time=1, stride_freq=1,
                                               pad_time=0, pad_freq=0, dilation_time=1, dilation_freq=1)
        self.skip_connection192 = Conv2DPReLUBN(iFM=192, oFM=192, T=self.T, F=self.F,
                                                kernel_time=1, kernel_freq=1, stride_time=1, stride_freq=1,
                                                pad_time=0, pad_freq=0, dilation_time=1, dilation_freq=1)

        self.deconv2d = Deconv2D(iFM=30, oFM=self.oFM, T=self.T, F=self.F,
                                 kernel_time=1, kernel_freq=3, stride_time=1, stride_freq=1,
                                 pad_time=0, pad_freq=0, dilation_time=1, dilation_freq=1)

        self.lstm = torch.nn.LSTM(input_size=self.T, hidden_size=300, num_layers=3, batch_first=True, proj_size=self.T)


    def forward(self, x):
        s = []
        x = self.conv2d_cauln(x)
        x = self.resblock(x)
        s.append(x)
        for i in range(4):
            x = self.conv2d_prelu_bn30(x)
            x = self.resblock(x)
            s.append(x)
        x = self.conv2d_prelu_bn64(x)
        s.append(x)
        x = self.conv2d_prelu_bn192(x)
        s.append(x)
        x = x.reshape(x.shape[0], x.shape[1], -1) # (batch, channel, time*freq)
        x, _ = self.lstm(x)
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2], 1)
        x += s[-1]
        x = self.deconv2d_prelu_bn64(x)
        x += s[-2]
        x = self.deconv2d_prelu_bn6430(x)
        t = -2
        for i in range(4):
            t -= 1
            x += s[t]
            x = self.resblock(x)
            x = self.deconv2d_prelu_bn30(x)
        x += s[0]
        x = self.resblock(x)
        x = self.deconv2d(x)
        return x




if __name__ == '__main__':
    device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
    ino = torch.randn(1, 1, 256, 129).to(device)
    n = DNN(1, 1, 256, 127).to(device)
    y = n(ino)




