#!/usr/bin/env python
# -*- coding: utf-8 -*- #
# ==================================================
# File Name:        test.py
# Author:           Qingzheng WANG
# Time:             2023/7/7 22:17
# Description:                       
# Function List:    
# ===================================================

import torch
from model.DNN import DNN
from config.option import parse
import librosa
from tqdm import tqdm
from scipy.io import wavfile

if __name__ == '__main__':
    opt_train = parse("config/test.yaml")
    md = DNN1 = DNN(iFM=1, oFM=1, T=1, F=129, type="DNN1", opt=opt_train)
    DNN1.load_state_dict(torch.load("/data/home/wangqingzheng/data/home/wangqingzheng/Dual-Window-SE/model/DNN1_0_2000.pth"))
    x, sr = librosa.load("/data/home/wangqingzheng/Edinburgh-Dataset/noisy_testset_dev/p232_001.wav", sr=16000)
    x = torch.stft(torch.tensor(x), n_fft=256, hop_length=32, win_length=256, window=torch.hann_window(256), return_complex=True)
    x = torch.tensor(x).transpose(1, 0)
    # reshape x to [shpe[0], 1, 1, shape[1]]
    x = x.reshape((x.shape[0], 1, 1, x.shape[1]))
    # stack x in y on shape[0]
    y = torch.complex(torch.zeros((x.shape[0], 1, 1, x.shape[3])), torch.zeros((x.shape[0], 1, 1, x.shape[3])))
    # 加上进度条
    for i in tqdm(range(x.shape[0])):
        y[i] = md(x[i].reshape(1, x[i].shape[0], x[i].shape[1], x[i].shape[2]))
    y = y.reshape((y.shape[0], y.shape[3]))
    y = y.transpose(1, 0)
    y = torch.istft(y, n_fft=256, hop_length=32, win_length=256, window=torch.hann_window(256))
    wavfile.write("test_dev_232_001_2000.wav", 16000, torch.detach(y).numpy())