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
    def __init__(self, iFM, oFM, T, F, type, opt, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.iFM = iFM # input Feature Maps
        self.T = T # time steps
        self.F = F # frequency bins
        self.oFM = oFM # output Feature Maps
        self.type = type # "DNN1" or "DNN2"
        self.opt = opt # options

        opt_conv2d_cauln = opt[self.type]["conv2d_cauln"]
        self.conv2d_cauln = Conv2DCauLN(iFM=self.iFM, oFM=opt_conv2d_cauln["oFM"], T=self.T, F=self.F,
                                        kernel_time=opt_conv2d_cauln["kernel_time"],
                                        kernel_freq=opt_conv2d_cauln["kernel_freq"],
                                        stride_time=opt_conv2d_cauln["stride_time"],
                                        stride_freq=opt_conv2d_cauln["stride_freq"],
                                        pad_time=opt_conv2d_cauln["pad_time"],
                                        pad_freq=opt_conv2d_cauln["pad_freq"],
                                        dilation_time=opt_conv2d_cauln["dilation_time"],
                                        dilation_freq=opt_conv2d_cauln["dilation_freq"])

        opt_resblock = opt[self.type]["resblock"]
        self.resblock = ResBlock(iFM=opt_resblock["iFM"],
                                 oFM=opt_resblock["oFM"],
                                 T=self.T, F=self.F, opt=opt_resblock)

        opt_conv2d_prelu_bn30 = opt[self.type]["conv2d_prelu_bn30"]
        self.conv2d_prelu_bn30 = Conv2DPReLUBN(iFM=opt_conv2d_prelu_bn30["iFM"],
                                               oFM=opt_conv2d_prelu_bn30["oFM"],
                                               T=self.T, F=self.F,
                                               kernel_time=opt_conv2d_prelu_bn30["kernel_time"],
                                               kernel_freq=opt_conv2d_prelu_bn30["kernel_freq"],
                                               stride_time=opt_conv2d_prelu_bn30["stride_time"],
                                               stride_freq=opt_conv2d_prelu_bn30["stride_freq"],
                                               pad_time=opt_conv2d_prelu_bn30["pad_time"],
                                               pad_freq=opt_conv2d_prelu_bn30["pad_freq"],
                                               dilation_time=opt_conv2d_prelu_bn30["dilation_time"],
                                               dilation_freq=opt_conv2d_prelu_bn30["dilation_freq"])

        opt_conv2d_prelu_bn64 = opt[self.type]["conv2d_prelu_bn64"]
        self.conv2d_prelu_bn64 = Conv2DPReLUBN(iFM=opt_conv2d_prelu_bn64["iFM"],
                                               oFM=opt_conv2d_prelu_bn64["oFM"],
                                               T=self.T,
                                               F=self.F,
                                               kernel_time=opt_conv2d_prelu_bn64["kernel_time"],
                                               kernel_freq=opt_conv2d_prelu_bn64["kernel_freq"],
                                               stride_time=opt_conv2d_prelu_bn64["stride_time"],
                                               stride_freq=opt_conv2d_prelu_bn64["stride_freq"],
                                               pad_time=opt_conv2d_prelu_bn64["pad_time"],
                                               pad_freq=opt_conv2d_prelu_bn64["pad_freq"],
                                               dilation_time=opt_conv2d_prelu_bn64["dilation_time"],
                                               dilation_freq=opt_conv2d_prelu_bn64["dilation_freq"])

        opt_conv2d_prelu_bn192 = opt[self.type]["conv2d_prelu_bn192"]
        self.conv2d_prelu_bn192 = Conv2DPReLUBN(iFM=opt_conv2d_prelu_bn192["iFM"],
                                                oFM=opt_conv2d_prelu_bn192["oFM"],
                                                T=self.T, F=self.F,
                                                kernel_time=opt_conv2d_prelu_bn192["kernel_time"],
                                                kernel_freq=opt_conv2d_prelu_bn192["kernel_freq"],
                                                stride_time=opt_conv2d_prelu_bn192["stride_time"],
                                                stride_freq=opt_conv2d_prelu_bn192["stride_freq"],
                                                pad_time=opt_conv2d_prelu_bn192["pad_time"],
                                                pad_freq=opt_conv2d_prelu_bn192["pad_freq"],
                                                dilation_time=opt_conv2d_prelu_bn192["dilation_time"],
                                                dilation_freq=opt_conv2d_prelu_bn192["dilation_freq"])

        opt_deconv2d_prelu_bn6430 = opt[self.type]["deconv2d_prelu_bn6430"]
        self.deconv2d_prelu_bn6430 = Deconv2DPReLUBN(iFM=opt_deconv2d_prelu_bn6430["iFM"],
                                                     oFM=opt_deconv2d_prelu_bn6430["oFM"],
                                                     T=self.T, F=self.F,
                                                     kernel_time=opt_deconv2d_prelu_bn6430["kernel_time"],
                                                     kernel_freq=opt_deconv2d_prelu_bn6430["kernel_freq"],
                                                     stride_time=opt_deconv2d_prelu_bn6430["stride_time"],
                                                     stride_freq=opt_deconv2d_prelu_bn6430["stride_freq"],
                                                     pad_time=opt_deconv2d_prelu_bn6430["pad_time"],
                                                     pad_freq=opt_deconv2d_prelu_bn6430["pad_freq"],
                                                     dilation_time=opt_deconv2d_prelu_bn6430["dilation_time"],
                                                     dilation_freq=opt_deconv2d_prelu_bn6430["dilation_freq"])

        opt_deconv2d_prelu_bn30 = opt[self.type]["deconv2d_prelu_bn30"]
        self.deconv2d_prelu_bn30 = Deconv2DPReLUBN(iFM=opt_deconv2d_prelu_bn30["iFM"],
                                                   oFM=opt_deconv2d_prelu_bn30["oFM"],
                                                   T=self.T, F=self.F,
                                                   kernel_time=opt_deconv2d_prelu_bn30["kernel_time"],
                                                   kernel_freq=opt_deconv2d_prelu_bn30["kernel_freq"],
                                                   stride_time=opt_deconv2d_prelu_bn30["stride_time"],
                                                   stride_freq=opt_deconv2d_prelu_bn30["stride_freq"],
                                                   pad_time=opt_deconv2d_prelu_bn30["pad_time"],
                                                   pad_freq=opt_deconv2d_prelu_bn30["pad_freq"],
                                                   dilation_time=opt_deconv2d_prelu_bn30["dilation_time"],
                                                   dilation_freq=opt_deconv2d_prelu_bn30["dilation_freq"])

        opt_deconv2d_prelu_bn64 = opt[self.type]["deconv2d_prelu_bn64"]
        self.deconv2d_prelu_bn64 = Deconv2DPReLUBN(iFM=opt_deconv2d_prelu_bn64["iFM"],
                                                   oFM=opt_deconv2d_prelu_bn64["oFM"],
                                                   T=self.T, F=self.F,
                                                   kernel_time=opt_deconv2d_prelu_bn64["kernel_time"],
                                                   kernel_freq=opt_deconv2d_prelu_bn64["kernel_freq"],
                                                   stride_time=opt_deconv2d_prelu_bn64["stride_time"],
                                                   stride_freq=opt_deconv2d_prelu_bn64["stride_freq"],
                                                   pad_time=opt_deconv2d_prelu_bn64["pad_time"],
                                                   pad_freq=opt_deconv2d_prelu_bn64["pad_freq"],
                                                   dilation_time=opt_deconv2d_prelu_bn64["dilation_time"],
                                                   dilation_freq=opt_deconv2d_prelu_bn64["dilation_freq"])

        opt_skip_connection30 = opt[self.type]["skip_connection30"]
        self.skip_connection30 = Conv2DPReLUBN(iFM=opt_skip_connection30["iFM"],
                                               oFM=opt_skip_connection30["oFM"],
                                               T=self.T,
                                               F=self.F,
                                               kernel_time=opt_skip_connection30["kernel_time"],
                                               kernel_freq=opt_skip_connection30["kernel_freq"],
                                               stride_time=opt_skip_connection30["stride_time"],
                                               stride_freq=opt_skip_connection30["stride_freq"],
                                               pad_time=opt_skip_connection30["pad_time"],
                                               pad_freq=opt_skip_connection30["pad_freq"],
                                               dilation_time=opt_skip_connection30["dilation_time"],
                                               dilation_freq=opt_skip_connection30["dilation_freq"])

        opt_skip_connection64 = opt[self.type]["skip_connection64"]
        self.skip_connection64 = Conv2DPReLUBN(iFM=opt_skip_connection64["iFM"],
                                               oFM=opt_skip_connection64["oFM"],
                                               T=self.T,
                                               F=self.F,
                                               kernel_time=opt_skip_connection64["kernel_time"],
                                               kernel_freq=opt_skip_connection64["kernel_freq"],
                                               stride_time=opt_skip_connection64["stride_time"],
                                               stride_freq=opt_skip_connection64["stride_freq"],
                                               pad_time=opt_skip_connection64["pad_time"],
                                               pad_freq=opt_skip_connection64["pad_freq"],
                                               dilation_time=opt_skip_connection64["dilation_time"],
                                               dilation_freq=opt_skip_connection64["dilation_freq"])

        opt_skip_connection192 = opt[self.type]["skip_connection192"]
        self.skip_connection192 = Conv2DPReLUBN(iFM=opt_skip_connection192["iFM"],
                                                oFM=opt_skip_connection192["oFM"],
                                                T=self.T,
                                                F=self.F,
                                                kernel_time=opt_skip_connection192["kernel_time"],
                                                kernel_freq=opt_skip_connection192["kernel_freq"],
                                                stride_time=opt_skip_connection192["stride_time"],
                                                stride_freq=opt_skip_connection192["stride_freq"],
                                                pad_time=opt_skip_connection192["pad_time"],
                                                pad_freq=opt_skip_connection192["pad_freq"],
                                                dilation_time=opt_skip_connection192["dilation_time"],
                                                dilation_freq=opt_skip_connection192["dilation_freq"])

        opt_deconv2d = opt[self.type]["deconv2d"]
        self.deconv2d = Deconv2D(iFM=opt_deconv2d["iFM"],
                                 oFM=self.oFM,
                                 T=self.T,
                                 F=self.F,
                                 kernel_time=opt_deconv2d["kernel_time"],
                                 kernel_freq=opt_deconv2d["kernel_freq"],
                                 stride_time=opt_deconv2d["stride_time"],
                                 stride_freq=opt_deconv2d["stride_freq"],
                                 pad_time=opt_deconv2d["pad_time"],
                                 pad_freq=opt_deconv2d["pad_freq"],
                                 dilation_time=opt_deconv2d["dilation_time"],
                                 dilation_freq=opt_deconv2d["dilation_freq"])

        opt_lstm = opt[self.type]["lstm"]
        self.lstm = torch.nn.LSTM(input_size=self.T,
                                  hidden_size=opt_lstm["hidden_size"],
                                  num_layers=opt_lstm["num_layers"],
                                  batch_first=opt_lstm["batch_first"],
                                  proj_size=self.T)

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
    from config import option
    opt = option.parse("/data/home/wangqingzheng/data/home/wangqingzheng/Dual-Window-SE/config/train.yaml")
    device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
    print(device)
    ino = torch.randn(1, 1, 256, 129).to(device)
    n = DNN(1, 1, 256, 127, "DNN1", opt).to(device)
    y = n(ino)




