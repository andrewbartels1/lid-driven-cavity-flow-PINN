#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 25 16:51:05 2022

@author: bartelsaa
"""
import glob
import os
import pandas as pd
import numpy as np
import torch
from torch.autograd import grad


# =============================================================================
# Utility Functions
# =============================================================================
def reshape_data(file_contents: tuple) -> tuple:
    """
    A bit clunky but a way to flatten arrays for ML applications

    Parameters
    ----------
    file_contents : tuple
        ``FlowPy`` file content output.

    Returns
    -------
    tuple
        flatten arrays or untouched int type values.

    """
    flatten_contents = []  # initiate new tuple

    for item in file_contents:
        if isinstance(item, (np.ndarray, np.generic)):
            flatten_contents.append(item.flatten())
        else:
            flatten_contents.append(item)

    return tuple(flatten_contents)


def read_datafile(filepath: str) -> tuple[np.array, np.array, np.array, np.array, int]:
    """Reads .txt file from ``FlowPy`` and outputs the pressure, u velocity
    v velocity, time delta, and reynolds number for a given iteration


    Parameters
    ----------
    filepath : str
        path to data file.

        Example: "../data/Re300/PUV10000.txt"

    Returns
    -------
    p_p : np.array
        DESCRIPTION.
    u_p : np.array
        DESCRIPTION.
    v_p : np.array
        DESCRIPTION.
    timestep : np.array
        time delta at that iteration, i.e. if the filename is PUV108800.txt at the
        108,800 iteration, the FlowPy was at a time delta of 0.006947693968062454
    Re : int
        Reynolds number for that datafile

    """

    # get Reynolds from folder name
    Re = get_Reynolds_from_filename(filepath)

    arr = np.loadtxt(filepath, delimiter="\t")
    rows, cols = arr.shape[0], arr.shape[1] - 1

    # this makes the assumption the grid is square!!!!
    rowpts, colpts = (
        round(np.sqrt(rows)),
        round(np.sqrt(rows)),
    )
    p_p = np.zeros((rowpts, colpts))
    u_p = np.zeros((rowpts, colpts))
    v_p = np.zeros((rowpts, colpts))
    p_arr = arr[:, 0]
    u_arr = arr[:, 1]
    v_arr = arr[:, 2]

    time = arr[:, 3]

    p_p = p_arr.reshape((rowpts, colpts))
    u_p = u_arr.reshape((rowpts, colpts))
    v_p = v_arr.reshape((rowpts, colpts))
    time = time.reshape((rowpts, colpts))

    return p_p, u_p, v_p, time, Re


def get_boundary_samples(array: np.array) -> np.array:
    """Gets all of the Boundary Conditions of an Array to sample"""
    return np.concatenate(
        [array[0, :-1], array[:-1, -1], array[-1, ::-1], array[-2:0:-1, 0]]
    )


def get_Reynolds_from_filename(filepath: str) -> int:
    """
    Gets the Reynolds from filename alone.

    Parameters
    ----------
    filepath : str
        filepath where the `FlowPy` data was outputted in text file form.

    Returns
    -------
    int
        Reynolds number (integer).

    """
    for file_part in filepath.split(os.sep):
        if file_part.find("Re") != -1:
            return int(file_part[2:])


def get_data_files(folder_path: str = "../data/", file_ending: str = ".mat") -> list:
    """Gets list of data file to process


    Parameters
    ----------
    folder_path : str, optional
        file location to folder with bunch of data in it. The default is "../data/".
    file_ending : str, optional
        file type of data files. The default is '.mat'.

    Raises
    ------

        Error folder_path isn't a folder!.
        Error if file_list comes back empty

    Returns
    -------
    list
        list of files.

    """
    # make sure folder exists
    if not os.path.isdir(folder_path):
        raise TypeError("folder_path give isn't a folder or doesn't exist")

    # make sure the filesep is in there for glob
    folder_path = (
        folder_path + os.path.sep
        if (folder_path[-1] is not os.path.sep)
        else folder_path
    )

    file_list = glob.glob(folder_path + "*" + file_ending)

    if file_list:
        return file_list
    else:
        raise "file_list came back empty! Make sure there's data in there and the path and file ending is correct!"


def generate_csv_catalog(
    filename: str = "catalog.csv", dataDir: str = "../data/", file_ending: str = ".txt"
) -> pd.DataFrame:
    """Generate catalog.csv for Dataset/DataLoader in pytorch.

    Always run this from the `src` directory to get everything to work properly
    by default, if not you're on your own to figure out file path issues!

    Parameters
    ----------
    filename : str, optional
        name of csv file to make with all the filepaths and details.
        The default is 'catalog.csv'.
    dataDir : str, optional
        DESCRIPTION. The default is "../data/".
    file_ending : str, optional
        DESCRIPTION. The default is '.txt'.

    Raises
    ------
    TypeError
        If dataDir isn't a folder will raise type error.

        tells the user this function should be run from `src/` of the repo

    Returns
    -------
    bool
        binary descriptor that the function wrote the csv successfully.

    """
    # make sure folder exists
    if not os.path.isdir(dataDir):
        raise TypeError(
            "dataDir give isn't a folder or doesn't exist \n maybe try changing current working directory to 'src'"
        )

    # make sure the filesep is in there for glob
    folder_path = dataDir + os.path.sep if (dataDir[-1] is not os.path.sep) else dataDir

    file_list = glob.glob("".join([folder_path + "**", os.sep, "**" + file_ending]))

    file_list.sort()  # sort the file list

    if file_list:
        pass
    else:
        raise "file_list came back empty! Make sure there's data in there and the path and file ending is correct!"

    # Generate csv catalog if file_list isn't empty!
    columns = ["filepath", "Re", "xsize", "ysize"]

    data = []

    # get grid size
    p_p, u_p, v_p, timestep, Re = read_datafile(file_list[0])

    # this assumes all the Res are the same grid size!
    x_size, y_size = p_p.shape
    for file in file_list:
        Re = get_Reynolds_from_filename(file)

        data.append(tuple([file, Re, x_size, y_size]))

    # create DataFrame object
    df = pd.DataFrame.from_records(data, columns=columns).sort_values(
        by=["Re", "filepath"]
    )

    # write out csv.
    df.to_csv(os.path.join(dataDir, filename), index=False)

    print(df.head(), f"\n number of files cataloged: {len(df)}\n")

    return df


# =============================================================================
# train.py helper functions
# =============================================================================
def get_all_model_weights(model):
    all_model_weights = []

    for layer_num, _ in enumerate(model.layers):
        # print("this is layer number ", layer_num)
        if (layer_num % 2) == 0:
            # print(model.layers[layer_num].weight, "<-- layer num")
            # print(model.layers[layer_num].weight.shape, "<-- layer shape")
            if (model.layers[layer_num].weight.shape[0] == 20) and (
                model.layers[layer_num].weight.shape[1] == 20
            ):
                # print("storing weight!")
                all_model_weights.append(model.layers[layer_num].weight.flatten())

    return torch.stack(all_model_weights).flatten()


def shape_weights_to_xygrid_flattened_size(
    input_tensor: torch.Tensor, desired_output_size: tuple, length_difference: int
):
    """
    takes the input, which is something like 2800 which are the weights
    of the models, and reshapes it to be the desired_output_size which is
    the [batch_size, X], where X is the grid size, so something like

    grid size: 151 x 151
    flattened size: 22801
    input_size: 2800

    input_size -> repeat to flattened_size

    then repeat that 1-d array by batch_size

    Parameters
    ----------
    input_tensor : torch.Tensor
        DESCRIPTION.
    desired_output_size : tuple
        DESCRIPTION.
    length_difference : int
        DESCRIPTION.

    Returns
    -------
    shaped_tensor : TYPE
        DESCRIPTION.

    """
    # expand in the first dimension, then repeat in the 0th dim
    # right_tensor_len = F.pad(input_tensor, (0, int(length_difference)))
    try:
        length = desired_output_size[1]
    except:
        print(desired_output_size, "length allocation failed!")
        length = desired_output_size[0]

    right_tensor_len = (
        input_tensor.unsqueeze(1)
        .repeat(1, int(np.ceil(length / input_tensor.shape[0])), 1)
        .squeeze(2)
    )

    shaped_tensor = right_tensor_len[:, 0:length].repeat(desired_output_size[0], 1)

    # print(shaped_tensor)
    # print("output shape", shaped_tensor.shape)

    return shaped_tensor


def navier_mse(u, u_pred, v, v_pred, f_u_pred, f_v_pred):
    return (
        torch.sum(torch.square(u - u_pred))
        + torch.sum(torch.square(v - v_pred))
        + torch.sum(torch.square(f_u_pred))
        + torch.sum(torch.square(f_v_pred))
    )


def navier_calc(x, y, t, lambda1, lambda2, device, Re, model, layers):
    """Do Physics here.

    See Physics Informed Deep Learning (Part II): Data-driven_2017.pdf
    equation 9 (page 7) for incompressible fluid flow.

    Parameters
    ----------
    psi : TYPE
        DESCRIPTION.
    pressure : TYPE
        DESCRIPTION.
    x : TYPE
        DESCRIPTION.
    y : TYPE
        DESCRIPTION.
    t : TYPE
        DESCRIPTION.
    lambda1 : TYPE
        DESCRIPTION.
    lambda2 : TYPE
        DESCRIPTION.

    Returns
    -------
    u : TYPE
        DESCRIPTION.
    v : TYPE
        DESCRIPTION.
    p : TYPE
        DESCRIPTION.
    f_u : TYPE
        DESCRIPTION.
    f_v : TYPE
        DESCRIPTION.

    """
    lambda1 = lambda1.to(device)
    lambda2 = lambda2.to(device)
    # This extra equation is the continuity equation for incompressible
    # fluids that
    # describes the conservation of mass of the fluid. We make the assumption
    # that:
    # print("y shape", y.shape)

    all_model_weights = get_all_model_weights(model)

    realtime_model_weights_of_linear_layers = all_model_weights.to(
        device
    )  # also known as psi in NavierStokes.py code

    #  pad the weights of psi and p to x,y,t,Re
    weights_shape = list(realtime_model_weights_of_linear_layers.size())

    #  this is really the model weigthts mushed in to the same size as the
    #  input array size so predicting elements are using the network weights
    #  and the predictions are the size of the input array, not dependant on
    #  the weights and total weight numbers
    psi = shape_weights_to_xygrid_flattened_size(
        realtime_model_weights_of_linear_layers, y.shape, y.shape[-1] - weights_shape[0]
    ).to(device)
    pressure = shape_weights_to_xygrid_flattened_size(
        realtime_model_weights_of_linear_layers, y.shape, y.shape[-1] - weights_shape[0]
    ).to(device)

    # the grad needs a function of the tensors, this is a
    # HACK: super hack, find a better way, NS.py makes a network mid function call
    # and it's super opaque as to how it's stored etc.

    # this outputs a tuple of 2 -by- [batch_size x length], taking the first one for now
    # TODO: see if averaging them or something is better?
    u = grad(
        (psi * y),
        (psi, y),
        create_graph=True,
        grad_outputs=torch.ones_like(y),
        allow_unused=True,
    )[0]
    v = -grad(
        (psi * x),
        (psi, x),
        create_graph=True,
        grad_outputs=torch.ones_like(x),
        allow_unused=True,
    )[0]

    u_t = grad(
        (u * t),
        (u, t),
        create_graph=True,
        grad_outputs=torch.ones_like(t),
        allow_unused=True,
    )[0]
    u_x = grad(
        (u * x),
        (u, x),
        create_graph=True,
        grad_outputs=torch.ones_like(x),
        allow_unused=True,
    )[0]
    u_y = grad(
        (u * y),
        (u, y),
        create_graph=True,
        grad_outputs=torch.ones_like(y),
        allow_unused=True,
    )[0]
    u_xx = grad(
        (u_x * x),
        (u_x, x),
        create_graph=True,
        grad_outputs=torch.ones_like(x),
        allow_unused=True,
    )[0]
    u_yy = grad(
        (u_y * y),
        (u_y, y),
        create_graph=True,
        grad_outputs=torch.ones_like(y),
        allow_unused=True,
    )[0]

    v_t = grad(
        (v * t),
        (v, t),
        create_graph=True,
        grad_outputs=torch.ones_like(t),
        allow_unused=True,
    )[0]
    v_x = grad(
        (v * x),
        (v, x),
        create_graph=True,
        grad_outputs=torch.ones_like(x),
        allow_unused=True,
    )[0]
    v_y = grad(
        (v * y),
        (v, y),
        create_graph=True,
        grad_outputs=torch.ones_like(y),
        allow_unused=True,
    )[0]
    v_xx = grad(
        (v_x * x),
        (v_x, x),
        create_graph=True,
        grad_outputs=torch.ones_like(x),
        allow_unused=True,
    )[0]
    v_yy = grad(
        (v_y * y),
        (v_y, y),
        create_graph=True,
        grad_outputs=torch.ones_like(y),
        allow_unused=True,
    )[0]

    p_x = grad(
        (pressure * x),
        (pressure, x),
        create_graph=True,
        grad_outputs=torch.ones_like(x),
        allow_unused=True,
    )[1]
    p_y = grad(
        (pressure * y),
        (pressure, y),
        create_graph=True,
        grad_outputs=torch.ones_like(y),
        allow_unused=True,
    )[1]

    f_u = u_t + lambda1 * (u * u_x + v * u_y) + p_x - lambda2 * (u_xx + u_yy)
    f_v = v_t + lambda1 * (u * v_x + v * v_y) + p_y - lambda2 * (v_xx + v_yy)

    return u, v, pressure, f_u, f_v


def save_model(model, state_dict_path):
    """Saves model to file"""
    return torch.save(model.state_dict(), state_dict_path)


def load_model(state_dict_path="model.pth"):
    """Load model from file"""
    return torch.load(state_dict_path)
