#!/usr/bin/env python
# -*- coding: utf-8 -*- #
# ==================================================
# File Name:        ComplexLSTM.py
# Author:           Qingzheng WANG
# Time:             2023/7/5 15:56
# Description:                       
# Function List:    
# ===================================================
import torch
import torch.nn as nn
from torch import stack

class ComplexLSTM(nn.Module):
  def __init__(self, in_channels, hidden_size, num_layers):
    super(ComplexLSTM, self).__init__()
    self.in_channels = in_channels
    self.hidden_size = hidden_size
    self.num_layers = num_layers


    self.re_LSTM = nn.LSTM(self.in_channels, self.hidden_size,
                           self.num_layers, batch_first=True, proj_size=1)
    self.im_LSTM = nn.LSTM(self.in_channels, self.hidden_size,
                           self.num_layers, batch_first=True, proj_size=1)


  def forward(self, x):
        x_re = x[..., 0].real
        x_re = torch.unsqueeze(x_re, -1)
        x_im = x[..., 0].imag
        x_im = torch.unsqueeze(x_im, -1)

        out_re1, (hn_re1, cn_re1) =  self.re_LSTM(x_re)
        out_re2, (hn_re2, cn_re2) =  self.im_LSTM(x_im)
        out_re = out_re1 - out_re2
        # hn_re  = hn_re1  - hn_re2
        # cn_re  = cn_re1  - cn_re2

        out_im1, (hn_im1, cn_im1) =  self.re_LSTM(x_re)
        out_im2, (hn_im2, cn_im2) =  self.im_LSTM(x_im)
        out_im = out_im1 + out_im2
        # hn_im  = hn_im1  + hn_im2
        # cn_im  = cn_im1  + cn_im2

        out = torch.complex(out_re, out_im)
        # hn = stack([hn_re, hn_im], -1)
        # cn = stack([cn_re, cn_im], -1)

        return out