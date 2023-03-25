#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 31 19:29:18 2022

@author: bartelsaa

This is where it all comes together! 
"""

# Import the dataloader and models
from load_data import prepare_data, LiddedDataset
from models import  BoxFlowNet
from train import train_model
import torch
from torch import nn
from utils import generate_csv_catalog, load_model, navier_calc, navier_mse, read_datafile
import pandas as pd
import sys
import os
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

logdir = "./runs/"
# Writer will output to ./runs/ directory by default

# =============================================================================
# Begin Inputs to training harness
# =============================================================================

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")


input_dim = 4 # from the grid size
output_dim = 3

layers = [(20, nn.Tanh()), (20, nn.Tanh()), (20, nn.Tanh()), (20, nn.Tanh()),
          (20, nn.Tanh()), (20, nn.Tanh()), (20, nn.Tanh()), (20, nn.Tanh()),
          (output_dim, nn.Tanh())]

# tried, smaller layers too
# layers = [(20, nn.Tanh()), (20, nn.Tanh()), (20, nn.Tanh()), (20, nn.Tanh()),
#           (output_dim, nn.Tanh())]

catalog_path = "../data/catalog.csv"
maxEpoch = 1000
batch_size = 1
load_saved_model = False # need to do!
generate_catalog = True
get_preview = False
#  generate csv if not already there or generated new data
if generate_catalog: generate_csv_catalog() #  --> places catalog.csv in default ../data/ folder
# =============================================================================
# End Inputs to training harness
# =============================================================================

# Prep data
train_DataLoader, test_DataLoader  = prepare_data(catalog_path, num_workers=0,
                                                 test_train_split=0.2,
                                                 train_batch_size=batch_size,
                                                 test_batch_size=batch_size)

# call and load the model
if load_saved_model:
    BoxFlowNet_gpu = BoxFlowNet(input_dim, layers).to(device)
    BoxFlowNet_gpu = BoxFlowNet_gpu.load_state_dict(torch.load('model.pth'))
    print("loaded model from model.pth")
else:    
    BoxFlowNet_gpu = BoxFlowNet(input_dim, layers).to(device)
    
if get_preview:
    train_features = next(iter(train_DataLoader))
    print(f"Feature batch shape: {train_features['input_array'].size()}")
    
    from torchsummary import summary
    
    # get summary before training to see how much it fills up GPU etc.
    summary(BoxFlowNet_gpu, train_features['input_array'].size())

# =============================================================================
# Train the model
# =============================================================================
epoch_list, loss_list = train_model(train_DataLoader, BoxFlowNet_gpu, maxEpoch, device, layers,
            optimizer="Adam", save_state=True, writer=True)

# =============================================================================
# Test the model
# =============================================================================
# Initialize the saved model
# BoxFlowNet_model_run = BoxFlowNet(input_dim, layers)
# checkpoint = torch.load('./model.pth')
# BoxFlowNet_model_run.load_state_dict(checkpoint)


