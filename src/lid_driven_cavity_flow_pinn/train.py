#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 25 16:51:05 2022

@author: bartelsaa
"""
import time as t
import os
import numpy as np
from numpy import vstack
from typing import Literal
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import accuracy_score, mean_squared_error

# pytorch imports
import torch
from torch import Tensor
from torch.optim import SGD, LBFGS, Adam


# import util functions

from utils import navier_calc, navier_mse, save_model, load_model

# # train the model
start_time = t.time()

# TODO: Tasks pending completion -@bartelsaa at 3/25/2023, 7:02:48 PM
# Stub in the correct way to do this here:
# https://github.com/jayroxis/PINNs/blob/master/Burgers%20Equation/Burgers%20Identification%20(PyTorch).ipynb


def train_model(
    train_dl,
    model,
    max_epochs,
    device,
    layers,
    optimizer: str = Literal["Adam", "SGD", "LBFGS"],
    save_state: bool = True,
    writer: bool = True,
    state_dict_path="model.pth",
):
    tb = SummaryWriter()
    sample = next(iter(train_dl))
    tb.add_graph(model.to(device), sample["input_array"].to(device).float())
    # initialize some training stuff
    global start_time
    train_loss, epoch_list = [], []

    size = len(train_dl.dataset)

    # define the optimization
    # emulating net_NS with LBFGS loss function, adding in the lambda parameters!
    optim_param = list(model.parameters())

    if optimizer == "Adam":
        optimizer = Adam(optim_param, lr=0.01)
    elif optimizer == "SGD":
        optimizer = SGD(optim_param, lr=0.00001, momentum=0.9)
    elif optimizer == "LBFGS":
        # Use LBFGS optimizer similar to prof
        optimizer = LBFGS(
            optim_param, lr=0.01, tolerance_grad=1.0 * np.finfo(float).eps
        )
    else:
        raise TypeError(
            """optimizer *must* be one of the following:\n
                           [Adam, SGD, LBFGS]"""
        )

    # super important to update the lambdas in the optimizer!
    # the best way to do this is to store them IN the model as seen in
    # BoxFlowNet above
    optimizer.add_param_group(
        {"params": [model.lambda1, model.lambda2], "lr": 0.01, "name": "lambdas"}
    )

    # send model to GPU
    model = model.to(device)
    print("\n------------ beggining model training ------------ \n")
    print(f"\n set model on gpu, running {max_epochs} epochs\n")
    # enumerate epochs
    for epoch in range(max_epochs):
        total_loss = 0
        print("Epoch {}/{}".format(epoch, max_epochs - 1))
        print("-" * 25)
        # enumerate mini batches
        for i, sample in enumerate(train_dl):
            # Move tensors to the configured device and make sure they float
            input_array, UVP_answer = (
                sample["input_array"].to(device).float().requires_grad_(),
                sample["UVP_answer"].to(device).float().requires_grad_(),
            )

            # unpack input
            x, y, time, Re = (
                input_array[:, :, 0],
                input_array[:, :, 1].requires_grad_(),
                input_array[:, :, 2].requires_grad_(),
                input_array[:, :, 3].requires_grad_(),
            )  # unpack for readability

            # unpack output
            u_answer, v_answer, pressure_answer = (
                UVP_answer[:, :, 0].requires_grad_(),
                UVP_answer[:, :, 1].requires_grad_(),
                UVP_answer[:, :, 2].requires_grad_(),
            )

            # -------- do physics --------
            #
            u_pred, v_pred, pressure_pred, f_u_pred, f_v_pred = navier_calc(
                x, y, time, model.lambda1, model.lambda2, device, Re, model, layers
            )

            # calculate loss (the learning part)
            loss = navier_mse(u_answer, u_pred, v_answer, v_pred, f_u_pred, f_v_pred)

            # running_loss += loss.item()
            # train_loss=running_loss/len(trainloader)
            if i % 100 == 0:
                elapsed = t.time() - start_time

                train_loss.append(loss.item() / len(train_dl))
                total_loss += loss.item()
                # print(train_loss)
                print(
                    f"loss: {(loss.item()/len(train_dl)):>7f}\n",
                    "time: {%.2f}" % (elapsed),
                )
                # print(f"""lambda1 value: {model.lambda1.data[0]}, lambda2 value: {model.lambda2.data[0]} loss: {float(losss)}""")
                # reset time
                start_time = t.time()
                epoch_list.append(epoch)

            # -------- backward + optimize --------
            # credit assignment
            loss.backward()
            optimizer.step()

        tb.add_scalar("Loss", total_loss, epoch)
        for l, layer in enumerate(model.layers):
            # print(l, layer)
            if (l % 2) == 0:
                tb.add_histogram(f"Linear-{l}.bias", model.layers[l].bias, epoch)
                tb.add_histogram(f"Linear-{l}.weight", model.layers[l].weight, epoch)
        # save at the end of every epoch!
        print(
            "saving current pytorch model to: ",
            os.path.join(os.getcwd(), state_dict_path),
        )
        save_model(model, os.path.join(os.getcwd(), state_dict_path))
    return epoch_list, train_loss


# TODO: create legit way to evaluate model
# evaluate the model
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
#     predictions, actuals = vstack(predictions), vstack(actuals)
#     # calculate mse
#     mse = mean_squared_error(actuals, predictions)
#     return mse


# make a class prediction for one row of data
def predict(row, model):
    # convert row to data
    row = Tensor([row])
    # make prediction
    yhat = model(row)
    # retrieve numpy array
    yhat = yhat.detach().numpy()
    return yhat
