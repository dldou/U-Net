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

    in_channels = 1 
    nof_classes = 2

    Unet = UNet(in_channels, nof_classes)
    Unet.to(device)

    #summary(Unet, (1, 572, 572))

    #Data
    transform = transforms.Compose([transforms.Pad((272,272), padding_mode='symmetric'),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=0.5, std=0.5)
                               ])

    #Train dataset
    mnist_trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    #Dataloader used to shuffle and create batch
    train_loader   = torch.utils.data.DataLoader(mnist_trainset, batch_size=1, shuffle=True)
    #Test dataset
    mnist_testset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    test_loader   = torch.utils.data.DataLoader(mnist_testset, batch_size=1, shuffle=True)

    #Training
    model = Unet
    train_loader = train_loader
    test_loader  = test_loader
    nof_epochs = 5
    learning_rate = 0.001
    optimizer = torch.optim.Adam(Unet.parameters(), lr = 0.001)
    criterion = torch.nn.MSELoss()
    file_path_save_model = '/content/checkpoint.pth'

    train_model(model, train_loader, test_loader, 
                nof_epochs, optimizer, learning_rate, criterion, 
                file_path_save_model)

    #Display results
    plot_results(Unet, test_loader, device)



