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
import librosa


class EdinburghTrainDataset(Dataset):
    """
    Edinburgh dataset for training,
    each noisy speech epoch(source) is corresponding to a clean speech epoch(target),
    when trainer use the dataloader to get the data as source or target,
    it gets the batch of frames(spectrum) of noisy speech and clean speech
    """

    def __init__(self, noisy_speech_dir, clean_speech_dir):
        """

        :param noisy_speech_dir: the path of noisy speech directory
        :param clean_speech_dir: the path of clean speech directory
        """
        super(EdinburghTrainDataset).__init__()
        self.noisy_speech_dir = noisy_speech_dir
        self.clean_speech_dir = clean_speech_dir
        self.uttrs = os.listdir(self.noisy_speech_dir) # get the utterance list
        self.noisy_speech_list = [os.path.join(self.noisy_speech_dir, uttr) for uttr in self.uttrs]
        self.clean_speech_list = [os.path.join(self.clean_speech_dir, uttr) for uttr in self.uttrs]
        self.frames = self.get_frames()

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, index):
        return self.frames[index]

    def get_frames(self):
        frames = []
        for ns in self.noisy_speech_list:
            ns, sr = librosa.load(ns, sr=16000)
            ns_stft = librosa.stft(ns, n_fft=256, hop_length=32, win_length=256, window='tukey')
            ns_stft = ns_stft.transpose((1, 0)) # the original is (freq, time), transpose to (time, freq)
            frames.append(ns_stft)
        return frames