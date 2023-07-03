#!/usr/bin/env python
# -*- coding: utf-8 -*- #
# ==================================================
# File Name:        dataset.py
# Author:           Qingzheng WANG
# Time:             2023/7/4 10:44
# Description:                       
# Function List:    
# ===================================================
from torch.utils.data import Dataset
import os


class EdinburghDataset(Dataset):
    def __init__(self, noisy_speech, clean_speech):
        """

        :param noisy_speech: the path of noisy speech directory
        :param clean_speech: the path of clean speech directory
        """
        super(EdinburghDataset).__init__()
        self.noisy_speech = noisy_speech
        self.clean_speech = clean_speech
        self.keys = os.listdir(self.noisy_speech)

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):
        key = self.keys[index]
        if key not in self.keys:
            raise ValueError
        return self.noisy_speech[key], self.clean_speech[key]