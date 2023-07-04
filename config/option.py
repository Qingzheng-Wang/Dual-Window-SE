#!/usr/bin/env python
# -*- coding: utf-8 -*- #
# ==================================================
# File Name:        option.py
# Author:           Qingzheng WANG
# Time:             2023/7/4 18:25
# Description:                       
# Function List:    
# ===================================================

import yaml


def parse(opt_path):
    with open(opt_path, mode='r', encoding='utf-8') as f:
        opt = yaml.load(f, Loader=yaml.FullLoader)

    return opt