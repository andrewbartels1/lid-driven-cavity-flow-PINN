#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 29 12:18:55 2022

@author: bartelsaa
"""

import torch
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
import os
import numpy as np
from utils import read_datafile, reshape_data

# dataset definition

# TODO: finish type hinting and docstringing everything


class LiddedDataset(Dataset):
    # load the dataset
    def __init__(self, path, Re_to_omit=[]):
        # load the csv file as a dataframe
        self.dataCatalog = pd.read_csv(path)

    # number of rows in the dataset

    def __len__(self):
        return len(self.dataCatalog)

    # get a row at an index 5375
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # get indexs from pandas
        file_contents = read_datafile(self.dataCatalog.filepath[idx])  # tuple of

        # reshape according to grid size
        p_p, u_p, v_p, time, Re = reshape_data(file_contents)

        #  make the x-grid and y-grid linear array that matches up with the
        #  flattened ones above
        x_grid_flattened = np.tile(np.linspace(
            0, Re/100, self.dataCatalog.xsize[idx]),
            self.dataCatalog.xsize[idx])

        y_grid_flattened = np.tile(np.linspace(
            0, Re/100, self.dataCatalog.ysize[idx]),
            self.dataCatalog.ysize[idx])

        # store the inputs and outputs
        # input in to model: x_grid, y_grid, timestep, u_correction,
        #                    v_correction, time, Reynolds number
        #
        # output from the model: predicts the UVP arrays (3-by x size x ysize
        #                        given an x,y grid and time (might include
        #                        Re later)
        #
        #
        input_array = np.vstack([x_grid_flattened,
                                 y_grid_flattened,
                                 time.flatten(), 
                                 np.repeat(Re, x_grid_flattened.shape)])

        uvp_output = np.vstack([u_p.flatten(), v_p.flatten(), p_p.flatten()])

        # ensure target has the right shape
        sample = {'input_array': torch.from_numpy(input_array.T), 
            'UVP_answer': torch.from_numpy(uvp_output.T)}

        return sample

    # get indexes for train and test rows
    def get_splits(self, n_test=0.2, random=False):
        """
        This function can randomly or orderdly (either lower/upper Reynolds number)
        generate the test train split along with the amount of data to hold out for test

        Parameters
        ----------
        n_test : float, optional
            _description_, by default 0.2
        random : bool, optional
            _description_, by default False

        Returns
        -------
        _type_
            _description_
        """
        # determine sizes
        test_size = round(n_test * len(self.dataCatalog))
        train_size = len(self.dataCatalog) - test_size
        # calculate the split
        return random_split(self, [train_size, test_size])

# prepare the dataset


def prepare_data(path: str,
                 num_workers: int = 0,
                 test_train_split: int = 0.2,
                 train_batch_size: int = 128,
                 test_batch_size: int = 128) -> tuple[torch.utils.data.dataloader.DataLoader,
                                                      torch.utils.data.dataloader.DataLoader]:
    """
    Function to prepare the data into test, train outputting the dataloader
    
    Credit: https://machinelearningmastery.com/pytorch-tutorial-develop-deep-learning-models/

    Parameters
    ----------
    path : str
        path to the csv catalog with all the filepaths and Reynolds numbers.
    num_workers: int
        number of parallel workers to call for a  DataLoader.

    Returns
    -------
    train_dl : torch.utils.data.dataloader.DataLoader
        Dataloader used for training.
    test_dl : torch.utils.data.dataloader.DataLoader
        Dataloader used for testing.

    """
    # load the dataset
    dataset = LiddedDataset(path)
    # calculate split
    train, test = dataset.get_splits(test_train_split)

    # prepare data loaders
    train_dl = DataLoader(train, batch_size=train_batch_size,
                          shuffle=True, num_workers=num_workers)
    test_dl = DataLoader(test, batch_size=test_batch_size,
                         shuffle=False, num_workers=num_workers)
    return train_dl, test_dl
