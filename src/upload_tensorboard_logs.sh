#!/bin/bash

conda activate torch

tensorboard dev upload --logdir \
    'runs'
