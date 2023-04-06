from pathlib import Path
from time import time
from torch.utils.data import RandomSampler, DataLoader
from typing import List, Tuple
import torch
import random

import json


def dump_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=4)


def make_text_data_fits_it_sits(
    P_boundary_list: List,
    U_boundary_list: List,
    V_boundary_list: List,
    time_list: List,
    Re_list: List,
) -> List[List[float]]:
    output_list = []
    header = ["x", "y", "time", "P", "U", "V", "Re"]
    output_list.append(header)
    Re_list = [[Re] * len(P_boundary_list[0]) for Re in Re_list]
    shortened_time_list = time_list[: len(P_boundary_list)]
    # this is probably a stupid way to do it but it works for now
    for time_step in zip(
        P_boundary_list, U_boundary_list, V_boundary_list, shortened_time_list, Re_list
    ):
        
        P, U, V, time, Re = time_step # unpack quick
        
        # setup the x & y
        x, y  = list(range(0,len(P))), list(range(0,len(P)))
        for x_single, y_single, time_single, P_single, U_single, V_single, Re_single,  in zip(
            x, y, time, P, U, V, Re
        ):
            output_list.append([x_single, y_single, time_single, P_single, U_single, V_single, Re_single])

    return output_list


def load_jsonl(path):
    return json.load(path)


def dump_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f)


def write_data_file_to_jsonl(path_to_json: Path, list_to_jsonify: List):
    """
    should mimic the this format
    ["t", "u", "v", "pressure", "Re"]
    [2, 0.001, 0.0, 0.001698680166, 0.0, 0.0]
    """
    dump_json(path_to_json, list_to_jsonify)