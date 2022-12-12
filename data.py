#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 12 20:55:01 2022

@author: dldou
"""

import torch



def image_labeling(image, threshold):
    """
        Apply a threshold to tag the ink 1 and the paper 0
    """

    #Copying tensor
    label = image.detach().clone()
    #Tagging
    label = torch.where(label > threshold, 1, 0)

    return label