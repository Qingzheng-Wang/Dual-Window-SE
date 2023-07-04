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

def trainer(dataloader, model, loss, optimizer, opt):
    for b, (x, y) in enumerate(dataloader):
        y_pred = model(x)
        y_pred = y_pred.to(opt["device"])
        y = y.to(opt["device"])
        loss = loss(y_pred, y, opt)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if b % 100 == 0:
            print("batch: {}, loss: {}".format(b, loss.item()))

