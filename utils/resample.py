#!/usr/bin/env python
# -*- coding: utf-8 -*- #
# ==================================================
# File Name:        resample.py
# Author:           Qingzheng WANG
# Time:             2023/7/3 10:03
# Description:      resample data from 48kHz to 16kHz
# Function List:    
# ===================================================

import librosa
import os
from tqdm import tqdm
from scipy.io import wavfile

def resample(dirs):
    dir_list = os.listdir(dirs)
    for _dir in tqdm(dir_list):
        data_list = os.listdir(os.path.join(dirs, _dir))
        for data in tqdm(data_list):
            data_path = os.path.join(dirs, _dir, data)
            y, sr = librosa.load(data_path, dtype='int16')
            t = librosa.resample(y, orig_sr=sr, target_sr=16000., )
            wavfile.write(data_path, int(sr), t)

if __name__ == '__main__':
    data = "/data/home/wangqingzheng/Edinburgh-Dataset"
    resample(data)
