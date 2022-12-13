#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 12 20:55:01 2022

@author: dldou
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from PIL import Image


#Preparing the Dataset class
class TrayDataset(Dataset):

    def __init__(self, annotations_file_path, 
                 _img_dir_path, _label_dir_path,
                 _transforms = None, _target_transforms = None):
        super(TrayDataset, self).__init__()

        self.labels            = pd.read_csv(annotations_file_path)
        self.img_dir_path      = _img_dir_path
        self.label_dir_path    = _label_dir_path
        self.transforms        = _transforms
        self.target_transforms = _target_transforms


    def __len__(self):
        return len(os.listdir(self.img_dir_path))


    def __getitem__(self, img_idx):

        #/!\ NOT THAT GOOD BECAUSE DATA LIST ARE SORTED EVERY TIME WE ACCESS /!\#
        img_name   = sorted(os.listdir(self.img_dir_path))[img_idx]
        label_name = sorted(os.listdir(self.label_dir_path))[img_idx]

        #Get the image
        image_path = self.img_dir_path + img_name
        image      = Image.open(image_path)
        if self.transforms:
            image  = self.transforms(image)
        #Get the label
        label_path = self.label_dir_path + label_name
        label      = Image.open(label_path)
        if self.target_transforms:
            label  = self.target_transforms(label)

        return image, label


