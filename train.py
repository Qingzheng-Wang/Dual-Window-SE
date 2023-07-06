#!/usr/bin/env python
# -*- coding: utf-8 -*- #
# ==================================================
# File Name:        train.py
# Author:           Qingzheng WANG
# Time:             2023/7/4 16:08
# Description:                       
# Function List:    
# ===================================================

import torch
from torch.utils.data import DataLoader
from data.dataset import EdinburghTrainDataset
from model.DNN import DNN
from config.option import parse
from model.loss import l_ri_mag, l_wav_mag

opt_train = parse("config/train.yaml")
DNN1 = DNN(iFM=1, oFM=1, T=1, F=129, type="DNN1", opt=opt_train)
DNN1 = DNN1.to(opt_train["device"])
optim = torch.optim.Adam(DNN1.parameters(), lr=opt_train["optim"]["lr"])

def trainer(dataloader, model, loss, optimizer, opt):
    with torch.autograd.set_detect_anomaly(True):
        for b, (x, y) in enumerate(dataloader):
            x = x.to(opt["device"])
            y = y.to(opt["device"])
            y_pred = model(x)
            optimizer.zero_grad()
            l = loss(y_pred, y).reshape(-1).norm()
            l.backward()
            optimizer.step()
            print("batch: {}, loss: {}".format(b, l.item()))

def train():
    print(opt_train["device"])
    for t in range(opt_train["train"]["epoch"]):
        dataset = EdinburghTrainDataset(noisy_speech_dir=opt_train["dataset"]["train"]["source"],
                                        clean_speech_dir=opt_train["dataset"]["train"]["target"])
        dataloader = DataLoader(dataset, batch_size=opt_train["train"]["batch_size"],
                                shuffle=opt_train["dataset"]["dataloader"]["shuffle"])
        trainer(dataloader, DNN1, l_ri_mag, optim, opt_train)
        torch.save(DNN1.state_dict(), "model/DNN1_{}.pth".format(t))

if __name__ == '__main__':
    train()