#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  2 19:42:46 2022

@author: dldou
"""

import torch
import torch.nn as nn

class Conv_x2(nn.Module):

    def __init__(self, in_ch, out_ch, ker_size):

        super(Conv_x2, self).__init__()

        self.conv_x2 = nn.Sequential
        (
            nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=ker_size, bias=True),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=ker_size, bias=True),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.conv_x2(x)



class Block_down_unet(nn.Module):

    def __init__(self, in_ch, out_ch, ker_size=3):

        super(Block_down_unet, self).__init__()

        self.block = nn.Sequential
        (
            Conv_x2(in_ch, out_ch, ker_size),
            nn.MaxPool2d(ker_size=2, stride=2),
        )

    def forward(self, x):
        return self.block(x)



class Block_up_unet(nn.Module):

    def __init__(self, in_ch, out_ch, ker_size=3):

        super(Block_up_unet, self).__init__()

        self.block = nn.Sequential
        (
            nn.ConvTranspose2d(in_channels=in_ch, out_channels=(in_ch//2), kernel_size=2, stride=2),
            Conv_x2(in_ch, out_ch, 3),
        )

    def forward(self, concat_tensor, processed_tensor):
        
        #/!\ /!\ NEED TO ADD PADDING TO CONCAT /!\ /!\
        
        #Concatenation path
        x = torch.cat([concat_tensor, processed_tensor], dim=1)
        #Then we apply the double convolution
        x = self.block(x)
        return x


class UNet(nn.Module):

    def __init__(self, in_channels, nof_classes):

        super(UNet, self).__init__()

        self.nof_channels = in_channels
        self.nof_classes  = nof_classes

        #Contracting path
        self.down1 = Block_down_unet(self.nof_channels, 64, 3)
        self.down2 = Block_down_unet(64, 128, 3)
        self.down3 = Block_down_unet(128, 256, 3)
        self.down4 = Block_down_unet(256, 512, 3)

        #Expanding path
        self.up1 = Block_up_unet(512, 1024, 3)
        self.up2 = Block_up_unet(1024, 512, 3)
        self.up3 = Block_up_unet(512, 256, 3)
        self.up4 = Block_up_unet(256, 128, 3)

        #Last block with 1x1 convolution
        self.last_block = nn.Sequential
        (
            Conv_x2(128, 64, 3),
            nn.Conv2d(in_channels=64, out_channels=2, kernel_size=1),
        )


    def forward(self, x):

        #Encoder (contracting path)
        x1 = self.down1(x)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)

        #Bottom layer
        x5 = self.up1(x1, x4)

        #Decode (expanding path + concatenating path)
        x = self.up1(x1, x5)
        x = self.up2(x2, x)
        x = self.up3(x3, x)
        x = self.up4(x4, x)

        #Last layer
        x = self.last_block(x)

        return x
