#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 31 19:18:58 2022

@author: bartelsaa
"""
from numpy import vstack
from numpy import argmax
import numpy as np
from pandas import read_csv
from sklearn.metrics import accuracy_score
from torchvision.datasets import MNIST
from torchvision.transforms import Compose
from torchvision.transforms import ToTensor
from torchvision.transforms import Normalize
import torch
from torch.utils.data import DataLoader
from torch.nn import Conv2d, Sigmoid
from torch.nn import MaxPool2d
from torch.nn import Linear
from torch.nn import ReLU
from torch.nn import Softmax
from torch.nn import Module

from torch.nn.init import kaiming_uniform_
from torch.nn.init import xavier_uniform_
import torch.nn as nn
import torch.nn.functional as F
import time as t

# for autodifferentiation (the special sauce)

import torch
from torch import nn, optim
from torch.nn.modules import Module
from typing import Literal


class BoxFlowNet(nn.Module):
    def __init__(self, input_size, layers_data: list):
        super().__init__()

        self.layers = nn.ModuleList()
        self.input_size = input_size  # Can be useful later ...

        # save lambdas to be updated here:
        self.lambda1 = torch.empty(1, requires_grad=True, device="cuda")
        self.lambda2 = torch.empty(1, requires_grad=True, device="cuda")

        # iteratively build the MLP with a tuple of size and activation type
        for size, activation in layers_data:
            self.layers.append(nn.Linear(input_size, size))
            input_size = size  # For the next layer
            if activation is not None:
                assert isinstance(
                    activation, Module
                ), "Each tuples should contain a size (int) and a torch.nn.modules.Module."
                self.layers.append(activation)

    def forward(self, input_data):
        for layer in self.layers:
            input_data = layer(input_data)
        return input_data


# =============================================================================
# Playing with different models
# =============================================================================
class CNN(Module):
    # define model elements
    def __init__(self, n_channels):
        super(CNN, self).__init__()
        # input to first hidden layer
        self.hidden1 = Conv2d(n_channels, 32, (3, 3))
        kaiming_uniform_(self.hidden1.weight, nonlinearity="relu")
        self.act1 = ReLU()
        # first pooling layer
        self.pool1 = MaxPool2d((2, 2), stride=(2, 2))
        # second hidden layer
        self.hidden2 = Conv2d(32, 32, (3, 3))
        kaiming_uniform_(self.hidden2.weight, nonlinearity="relu")
        self.act2 = ReLU()
        # second pooling layer
        self.pool2 = MaxPool2d((2, 2), stride=(2, 2))
        # fully connected layer
        self.hidden3 = Linear(5 * 5 * 32, 100)
        kaiming_uniform_(self.hidden3.weight, nonlinearity="relu")
        self.act3 = ReLU()
        # output layer
        self.hidden4 = Linear(100, 10)
        xavier_uniform_(self.hidden4.weight)
        self.act4 = Softmax(dim=1)

    # forward propagate input
    def forward(self, X):
        # input to first hidden layer
        X = self.hidden1(X)
        X = self.act1(X)
        X = self.pool1(X)
        # second hidden layer
        X = self.hidden2(X)
        X = self.act2(X)
        X = self.pool2(X)
        # flatten
        X = X.view(-1, 4 * 4 * 50)
        # third hidden layer
        X = self.hidden3(X)
        X = self.act3(X)
        # output layer
        X = self.hidden4(X)
        X = self.act4(X)
        return X


# model definition


class BoxFlowNet_MLP(Module):
    # define model elements
    def __init__(self, n_inputs):
        super(BoxFlowNet_MLP, self).__init__()
        # input to first hidden layer
        self.hidden1 = Linear(n_inputs, 10)
        xavier_uniform_(self.hidden1.weight)
        self.act1 = Sigmoid()
        # second hidden layer
        self.hidden2 = Linear(10, 8)
        xavier_uniform_(self.hidden2.weight)
        self.act2 = Sigmoid()
        # third hidden layer and output
        self.hidden3 = Linear(8, 1)
        xavier_uniform_(self.hidden3.weight)

    # forward propagate input
    def forward(self, X):
        # input to first hidden layer
        X = self.hidden1(X)
        X = self.act1(X)
        # second hidden layer
        X = self.hidden2(X)
        X = self.act2(X)
        # third hidden layer and output
        X = self.hidden3(X)
        return X


# =============================================================================
# Write these tomorrow
# =============================================================================
# # evaluate the model
# def evaluate_model(test_dl, model):
#     predictions, actuals = list(), list()
#     for i, (inputs, targets) in enumerate(test_dl):
#         # evaluate the model on the test set
#         yhat = model(inputs)
#         # retrieve numpy array
#         yhat = yhat.detach().numpy()
#         actual = targets.numpy()
#         actual = actual.reshape((len(actual), 1))
#         # store
#         predictions.append(yhat)
#         actuals.append(actual)
#     predictions, actuals = np.vstack(predictions), np.vstack(actuals)
#     # calculate mse
#     mse = mean_squared_error(actuals, predictions)
#     return mse

# # make a class prediction for one row of data
# def predict(row, model):
#     # convert row to data
#     row = Tensor([row])
#     # make prediction
#     yhat = model(row)
#     # retrieve numpy array
#     yhat = yhat.detach().numpy()
#     return yhat
