#!/usr/bin/env python
# -*- coding: utf-8 -*- #
# ==================================================
# File Name:        trashbin.py
# Author:           Qingzheng WANG
# Time:             2023/7/4 17:47
# Description:                       
# Function List:    
# ===================================================

# class ResBlock(torch.nn.Module):
#     def __init__(self, iFM, oFM, T, F, opt, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.iFM = iFM # input Feature Maps
#         self.oFM = oFM # output Feature Maps
#         self.T = T # time steps
#         self.F = F # frequency bins
#         self.opt = opt # options, opt[self.type]["resblock"]
#
#         self.depth_conv1 = torch.nn.Conv2d(in_channels=self.iFM, out_channels=self.oFM, kernel_size=opt["depth_conv1"]["kernel_size"],
#                                            stride=(1, 1), padding=(1, 1), dilation=(1, 1),
#                                            groups=self.iFM)
#         self.point_conv1 = torch.nn.Conv2d(in_channels=self.iFM, out_channels=self.oFM, kernel_size=(1, 1))
#         self.prelu1 = torch.nn.PReLU()
#         self.bn1 = torch.nn.BatchNorm2d(num_features=self.oFM)
#         self.ds_conv1 = torch.nn.Sequential(self.depth_conv1, self.point_conv1, self.prelu1, self.bn1)  # depthwise separable convolution
#
#         self.depth_conv2 = torch.nn.Conv2d(in_channels=self.oFM, out_channels=self.oFM, kernel_size=(3, 3),
#                                            stride=(1, 1), padding=(2, 1), dilation=(2, 1),
#                                            groups=self.oFM)
#         self.point_conv2 = torch.nn.Conv2d(in_channels=self.oFM, out_channels=self.oFM, kernel_size=(1, 1))
#         self.prelu2 = torch.nn.PReLU()
#         self.bn2 = torch.nn.BatchNorm2d(num_features=self.oFM)
#         self.ds_conv2 = torch.nn.Sequential(self.depth_conv2, self.point_conv2, self.prelu2, self.bn2)
#
#         self.depth_conv3 = torch.nn.Conv2d(in_channels=self.oFM, out_channels=self.oFM, kernel_size=(3, 3),
#                                            stride=(1, 1), padding=(4, 1), dilation=(4, 1),
#                                            groups=self.oFM)
#         self.point_conv3 = torch.nn.Conv2d(in_channels=self.oFM, out_channels=self.oFM, kernel_size=(1, 1))
#         self.prelu3 = torch.nn.PReLU()
#         self.bn3 = torch.nn.BatchNorm2d(num_features=self.oFM)
#         self.ds_conv3 = torch.nn.Sequential(self.depth_conv3, self.point_conv3, self.prelu3, self.bn3)
#
#         self.depth_conv4 = torch.nn.Conv2d(in_channels=self.oFM, out_channels=self.oFM, kernel_size=(3, 3),
#                                            stride=(1, 1), padding=(8, 1), dilation=(8, 1),
#                                            groups=self.oFM)
#         self.point_conv4 = torch.nn.Conv2d(in_channels=self.oFM, out_channels=self.oFM, kernel_size=(1, 1))
#         self.prelu4 = torch.nn.PReLU()
#         self.bn4 = torch.nn.BatchNorm2d(num_features=self.oFM)
#         self.ds_conv4 = torch.nn.Sequential(self.depth_conv4, self.point_conv4, self.prelu4, self.bn4)
#
#         self.depth_conv5 = torch.nn.Conv2d(in_channels=self.oFM, out_channels=self.oFM, kernel_size=(3, 3),
#                                            stride=(1, 1), padding=(16, 1), dilation=(16, 1),
#                                            groups=self.oFM)
#         self.point_conv5 = torch.nn.Conv2d(in_channels=self.oFM, out_channels=self.oFM, kernel_size=(1, 1))
#         self.prelu5 = torch.nn.PReLU()
#         self.bn5 = torch.nn.BatchNorm2d(num_features=self.oFM)
#         self.ds_conv5 = torch.nn.Sequential(self.depth_conv5, self.point_conv5, self.prelu5, self.bn5)