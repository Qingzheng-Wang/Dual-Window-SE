#!/usr/bin/env python
# -*- coding: utf-8 -*- #
# ==================================================
# File Name:        loss.py
# Author:           Qingzheng WANG
# Time:             2023/7/4 11:50
# Description:      define loss functions
# Function List:    
# ===================================================

import torch

def l_ri_mag(y16_pred, y16_true, opt):
    """
    loss function for every 16ms frames
    :param y16_pred: 16ms spectrum, predicted current frame
    :param y16_true: 16ms spectrum, current frame
    :param opt: train.yaml, configure
    :return: loss
    """
    a = torch.norm(y16_pred.real-y16_true.real, p=1, dim=0)
    b = torch.norm(y16_pred.imag-y16_true.imag, p=1, dim=0)
    c = torch.norm(torch.sqrt(torch.square(y16_pred.real)+torch.square(y16_pred.imag))
                   -torch.abs(y16_true), p=1, dim=0)
    return (a + b + c).to(opt["device"])

def l_wav_mag(y4_oa_pred, y4_oa_true, opt):
    """
    loss function for the current 4ms frame
    overlap-added with the predicted future 4ms frame(s)
    :param y4_oa_pred: 4ms current wav overlap-add with the predicted future 4ms wav(s)
    :param y4_oa_true: the original place wav
    :param opt: train.yaml, configure
    :return: loss
    """
    a = torch.norm(y4_oa_pred - y4_oa_true, p=1, dim=0)
    b = torch.norm(torch.stft(y4_oa_pred, n_fft=64, hop_length=32,
                              win_length=64, window=torch.hann_window(64).cuda(),
                              return_complex=True).abs()-
                   torch.stft(y4_oa_true, n_fft=64, hop_length=32,
                              win_length=64, window=torch.hann_window(64).cuda(),
                              return_complex=True).abs(), p=1, dim=0)
    return (a + b).to(opt["device"])