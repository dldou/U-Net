#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 12 20:55:52 2022

@author: dldou
"""


import torch
from torchsummary import summary
import torchvision.datasets as datasets 
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt



if __name__ == "__main__":

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.cuda.empty_cache()

    #Data
    transform        = transforms.Compose([transforms.Pad((78,158), fill=0, padding_mode='constant'),
                                            transforms.ToTensor(),
                                            transforms.Normalize(mean=0.5, std=0.5)])
    
    target_transform = transforms.Compose([transforms.Pad((78,158), fill=0, padding_mode='constant'),
                                           transforms.ToTensor()])
    
    annotations_file_path    = '/content/classes.csv'
    train_dataset_img_path   = '/content/TrayDataset/TrayDataset/XTrain/'
    train_dataset_label_path = '/content/TrayDataset/TrayDataset/yTrain/'
        
    train_dataset = TrayDataset(annotations_file_path, 
                                train_dataset_img_path, train_dataset_label_path,
                                transform, target_transform
                                )

    annotations_file_path    = '/content/classes.csv'
    test_dataset_img_path   = '/content/TrayDataset/TrayDataset/XTest/'
    test_dataset_label_path = '/content/TrayDataset/TrayDataset/yTest/'

    transform        = transforms.Compose([transforms.Pad((78,158), fill=0, padding_mode='constant'),
                                        transforms.ToTensor()])
    target_transform = transforms.Compose([transforms.Pad((78,158), fill=0, padding_mode='constant'), 
                                           transforms.ToTensor()])

    test_dataset = TrayDataset(annotations_file_path, 
                                test_dataset_img_path, test_dataset_label_path,
                                transform, target_transform
                                )
    #Test Dataloader
    _batch_size = 1

    train_dataloader = DataLoader(train_dataset, batch_size=_batch_size)
    test_dataloader = DataLoader(test_dataset, batch_size=_batch_size)

    #Model instanciation
    in_channels = 3 
    nof_classes = len(train_dataset) #(has the background class)

    Unet = UNet(in_channels, nof_classes)
    Unet.to(device)

    #summary(Unet, (3, 572, 572))

    #Training
    model = Unet
    train_loader = train_dataloader
    test_loader  = test_dataloader
    nof_epochs = 5
    learning_rate = 0.001
    optimizer = torch.optim.Adam(Unet.parameters(), lr = 0.001)
    #/!\ Wrong criterion /!\#
    criterion = torch.nn.MSELoss()
    file_path_save_model = '/content/checkpoint.pth'

    train_model(model, train_loader, test_loader, 
                nof_epochs, optimizer, learning_rate, criterion, 
                file_path_save_model)

    #Display results
    plot_results(Unet, test_loader, device)

