#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 12 20:53:03 2022

@author: dldou
"""

import matplotlib.pyplot as plt
from torch.autograd import Variable
import random

def saveModel(model, file_path):
    """
        Function to save model's parameters
    """
    torch.save(model.state_dict(), file_path)


def loadModel(model, file_path, device):
    """
        Function to load function when only the params have been saved
    """
    params = torch.load(file_path)
    model.load_state_dict(params)


def checkPoint_model(model, 
                     optimizer, loss, epoch,
                     file_path):
    """
        Function to save model's checkpoints
    """
    
    torch.save({'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss}, 
                file_path)

def load_checkPoint_model(model, optimizer, file_path, device):

    checkpoint = torch.load(file_path)

    #Loading
    model.load_state_dict(checkpoint['model_state_dict'], map_location=device)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'], map_location=device)
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']

    return epoch, loss


def train_epoch(model, optimizer, criterion, 
                train_loader, 
                device):

    loss = 0.0

    for i, (images, labels) in enumerate(train_loader, 0):
        
            #Each time create the label (not optimal /!\ /!\ /!\)
            labels = image_labeling(images, -0.5)

            #Data send to device + requires_grad=True
            images, labels = Variable(images.to(device)), Variable(labels.to(device))
            #Zero the gradient 
            optimizer.zero_grad()
            #Predictions 
            predicted_images = model(images)
            predicted_images.to(device)
            #Cropping the image
            crop_image = transforms.CenterCrop(predicted_images.size()[3])
            images     = crop_image(images)
            images     = images.to(device)
            #Loss
            loss = criterion(predicted_images, images)
            #print("training loss: ", loss)
            #Upgrade the gradients (backpropagate) and the optimizer
            loss.backward()
            optimizer.step()

    return loss


def validate_epoch(model, optimizer, criterion, 
                   test_loader, epoch,
                   device):
    
    losses          = []
    loss            = 0.0
    accuracy        = 0.0
    nof_predictions = 0.0

    #Fasten the inference by setting every requires_grad to False
    with torch.no_grad():
        for data in test_loader:
            #Get data and send them to the device
            images, _ = data
            images    = images.to(device)
            #Run the model on the test set
            outputs = model(images)
            outputs.to(device)
            #Cropping the image
            crop_image = transforms.CenterCrop((outputs.size()[2], outputs.size()[3]))
            images     = crop_image(images)
            images     = images.to(device)
            #Compute the loss on the batch
            loss = criterion(image_labeling(images, -0.5), outputs)
            #print("validation loss: ", loss)
            losses.append(loss)
    
    ###############################
    #  Definition of loss ratio r #
    ###############################
    #
    # loss ratio = loss / maximum loss theoritically possible
    # 
    # accuracy = 100*(1 - r)
    #
    # For further details on the calcul of the accuracy, one may refer to the ReadMe.me file 
    ###############################

    #With nomalization, max is 1 for tensors' values
    max_theorical_loss = (2*1)**2
    #Number of pixels in each images of dataset is test_loader.dataset[0][0].squeeze().numel()
    loss_ratio = sum(losses)/(max_theorical_loss * test_loader.dataset[0][0].squeeze().numel()) 
    accuracy = 100*(1 - loss_ratio)

    return accuracy


def train_model(model, train_loader, test_loader, 
                nof_epochs, optimizer, learning_rate, criterion, 
                file_path_save_model):
    
    #Which device + sending model to its memory
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("The model will be running on", device, "device")
    model.to(device)

    best_accuracy = 0.0

    for epoch in range(nof_epochs):

        epoch_accuracy = 0.0
        epoch_loss     = 0.0

        #Training
        model.train()
        epoch_loss = train_epoch(model, optimizer, criterion, 
                                 train_loader, 
                                 device)
        
        #Validation
        model.eval()
        epoch_accuracy = validate_epoch(model, optimizer, criterion, 
                                        test_loader, epoch,
                                        device)
        
        print('Epoch', epoch+1,', accuracy: {:.4f} % \n'.format(epoch_accuracy))
        
        #Save model when best accuracy is beaten
        if epoch_accuracy > best_accuracy:
            #load_checkPoint_model(model, optimizer, file_path_save_model, device)
            saveModel(model, file_path_save_model)
            best_accuracy = epoch_accuracy

    return model


def plot_results(model, test_loader,
                 device):
    """
    Display few images and outputs to compare them 
    """

    fig = plt.figure(figsize=(10,10))

    #Display 4*4=16 images
    for i in range(int(16/2)):

        #Select a random image in the dataset
        nof_images = len(test_loader.dataset)
        idx = random.randrange(nof_images)
        #Inference
        image = test_loader.dataset[idx][0].unsqueeze(0).to(device)
        with torch.no_grad():
            model.eval()
            segmented_image = model(image)
            #print(torch.max(model(image), 1).indices[0].item())
        #Get the label
        image = image_labeling(image, -0.5)
        #Sent back the image to the CPU
        image = image.squeeze().to('cpu')
        segmented_image = segmented_image.squeeze().to('cpu')

        #Plot
        ax_image = plt.subplot(4,4, 2*i+1 )
        ax_image.set_title("Real image")
        plt.imshow(image, cmap='gray_r')
        plt.axis('off')

        ax_segmented_image = plt.subplot(4,4, 2*(i+1) )
        ax_segmented_image.set_title("Segmented image")
        plt.imshow(segmented_image, cmap='gray_r')
        plt.axis('off')
    
    #fig.suptitle("{} on few examples\n Reached accuracy with xxx epochs: xxx".format(model.__class__.__name__))

    #Save
    #plt.savefig(str(model.__class__.__name__) + "_accuraccy_xxx" + ".pdf")

    #Show
    plt.show()

