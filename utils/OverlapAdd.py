#!/usr/bin/env python
# -*- coding: utf-8 -*- #
# ==================================================
# File Name:        OverlapAdd.py
# Author:           Qingzheng WANG
# Time:             2023/7/6 16:32
# Description:                       
# Function List:    
# ===================================================
import torch

def overlap_add(frames, n_fft, hop_length, window, win_length):
    """
    istft(frame) -> overlap-add waveform
    :param frames: list of frames, like [frame1, frame2, ...]
    :param n_fft: number of fft points
    :param hop_length: hop size
    :param window: window type
    :param win_length: window length
    :return: overlap-added waveform
    """
    wavs = []
    for f in frames:
        wavs.append(torch.istft(f, n_fft=n_fft, hop_length=hop_length,
                                win_length=win_length, window=window, return_complex=False))
    # 对每一个wav，加上window对应的窗，例如hanning窗，就加上hanning窗
    wavs = torch.tensor([w*window for w in wavs])
    # 重叠相加
    y = wavs.sum(dim=0) / window.pow(2).sum(dim=0)
    return y

if __name__ == '__main__':
    # import librosa
    # ns, sr = librosa.load("/data/home/wangqingzheng/Edinburgh-Dataset/noisy_trainset_dev/p226_001.wav", sr=16000)
    # frames = librosa.stft(ns, n_fft=64, hop_length=32, win_length=64, window='hann')
    # frames = torch.tensor(frames).transpose(1, 0)
    # y = overlap_add(frames, 64, 32, torch.hann_window(64), 64)
    # librosa.output.write_wav("test.wav", y, sr=sr)
    n = torch.randn((33, 2))
    i = torch.randn((33, 2))
    u = torch.complex(n, i)
    y = torch.istft(u, n_fft=64, hop_length=32, win_length=64, window=torch.hann_window(64), return_complex=False)
