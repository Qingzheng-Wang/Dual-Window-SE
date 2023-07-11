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
import tqdm
from torch.utils.data import DataLoader
from data.dataset import EdinburghTrainDataset
from model.DNN import DNN
from config.option import parse
from model.loss import l_ri_mag, l_wav_mag
from torchmetrics.functional import scale_invariant_signal_distortion_ratio as sisdr

opt_train = parse("config/train.yaml")
DNN1 = DNN(iFM=1, oFM=1, T=1, F=129, type="DNN1", opt=opt_train)
DNN1 = DNN1.to(opt_train["device"])
DNN2 = DNN(iFM=1, oFM=1, T=1, F=129, type="DNN2", opt=opt_train)
DNN2 = DNN2.to(opt_train["device"])
optim1 = torch.optim.Adam(DNN1.parameters(), lr=opt_train["optim"]["lr"])
optim2 = torch.optim.Adam(DNN2.parameters(), lr=opt_train["optim"]["lr"])

def trainer(dataloader, model1, model2, loss1, loss2, optimizer1, optimizer2, opt, t):
    with torch.autograd.set_detect_anomaly(True):
        for b, (x, y1, y2) in enumerate(dataloader):
            x = x.to(opt["device"])
            y1 = y1.to(opt["device"])
            y2 = y2.to(opt["device"])
            y_pred1 = model1(x)
            optimizer1.zero_grad()
            l1 = loss1(y_pred1, y1).reshape(-1).mean()
            l1.backward(retain_graph=True)
            optimizer1.step()
            # batch总数
            print("Epoch{}>>>DNN1 batch: {}/{}, loss: {}".format(t, b, len(dataloader), l1.item()))

            y_pred2 = model2(y_pred1)
            optimizer2.zero_grad()
            l2 = loss2(y_pred2, y2).reshape(-1).mean()
            l2.backward()
            optimizer2.step()
            print("Epoch{}>>>DNN2 batch: {}/{}, loss: {}".format(t, b, len(dataloader), l2.item()))

            if b % 1000 == 0:
                torch.save(model1.state_dict(), "model/DNN1_{}_{}.pth".format(t, b))
                torch.save(model2.state_dict(), "model/DNN2_{}_{}.pth".format(t, b))

def tester(dataloader, model1, model2, loss1, loss2, opt):
    l1 = 0.
    l2 = 0.
    sisdr_cur = 0.
    sisdr_fut = 0.
    with torch.no_grad():
        for b, (x, y1, y2) in enumerate(dataloader):
            x = x.to(opt["device"])
            y1 = y1.to(opt["device"])
            y2 = y2.to(opt["device"])
            y_pred1 = model1(x)
            l1 += loss1(y_pred1, y1).reshape(-1).mean()
            sisdr_cur += sisdr(y_pred1, y1)
            print("DNN1 batch: {}, loss: {}".format(b, l1.item()))


            y_pred2 = model2(y_pred1)
            l2 += loss2(y_pred2, y2).reshape(-1).mean()
            print("DNN2 batch: {}, loss: {}".format(b, l2.item()))

def train():
    print(opt_train["device"])
    DNN1.load_state_dict(torch.load("/data/home/wangqingzheng/data/home/wangqingzheng/Dual-Window-SE/model/DNN1_0_14000.pth")) # load the pre-trained model
    DNN2.load_state_dict(torch.load("/data/home/wangqingzheng/data/home/wangqingzheng/Dual-Window-SE/model/DNN2_0_14000.pth")) # load the pre-trained model
    for t in range(opt_train["train"]["epoch"]):
        print("epoch: {}".format(t))
        dataset = EdinburghTrainDataset(noisy_speech_dir=opt_train["dataset"]["train"]["source"],
                                        clean_speech_dir=opt_train["dataset"]["train"]["target"])
        dataloader = DataLoader(dataset, batch_size=opt_train["train"]["batch_size"],
                                shuffle=opt_train["dataset"]["dataloader"]["shuffle"])
        trainer(dataloader, DNN1, DNN2, l_ri_mag, l_ri_mag, optim1, optim2, opt_train, t)
        torch.save(DNN1.state_dict(), "model/params/DNN1_{}.pth".format(t))
        torch.save(DNN2.state_dict(), "model/params/DNN2_{}.pth".format(t))

if __name__ == '__main__':
    train()