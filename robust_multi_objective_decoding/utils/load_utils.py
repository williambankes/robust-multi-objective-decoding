# -*- coding: utf-8 -*-
"""
Created on Tue Sep  3 13:45:59 2024

@author: William
"""

import os
import torch

from typing import List, Union
from datasets import load_from_disk, concatenate_datasets, DatasetDict
from random import randrange
from robust_multi_objective_decoding.oracles.shield_gemma import HarmType


def process_labelled_dataset_locally(
    data_path: str,
) -> List[DatasetDict]:
    """
    Download a batched dataset from a high level directory.

    Parameters
    ----------
    data_path : str
        Path to high level directory containing data.

    Returns
    -------
    DatasetDict
        The concat'd dataset.

    """

    print("Loading Dataset")
    data = load_all_datasets_in_directory(data_path)
    data = concatenate_datasets(data)

    return data


def check_directory_exists(directory_path: str) -> bool:
    if os.path.exists(directory_path) and os.path.isdir(directory_path):
        print(f"Directory '{directory_path}' exists.")
        return True
    else:
        print(f"Directory '{directory_path}' does not exist.")
        return False


def find_directories_in_directory(
    directory_path: str, verbose: bool = True
) -> List[str]:
    directories = [
        d
        for d in os.listdir(directory_path)
        if os.path.isdir(os.path.join(directory_path, d))
    ]

    if verbose:
        print(f"Loading PKUSafeRLFHDataset. Found directories: {directories}")

    return directories


def load_all_datasets_in_directory(parent_directory: str, verbose: bool = True):
    directories = find_directories_in_directory(parent_directory)

    datasets = list()
    for dir_name in directories:
        dataset_path = os.path.join(parent_directory, dir_name)

        if verbose:
            print(f"Loading dataset from directory: {dataset_path}")

        dataset = load_from_disk(dataset_path)
        datasets.append(dataset)

    return datasets


def assign_int_label(example, idx):
    example["idx"] = idx
    return example


def assign_random_cut(example, idx):
    length = len(example["tokens_shieldgemma_dangerous_probs"])
    example["split_idx"] = randrange(length)
    return example


def load_base_vf_module_state_dict_from_checkpoint(
    checkpoint_path: str, devices: Union[None, List[int]] = None
) -> dict[str, torch.Tensor]:
    """Loads the state dict of a BaseValueFunctionModule subclass from a checkpoint path

    Input: Checkpoint path generated using `train_safety_cbf.py`
    Return: State dict of a BaseValueFunctionModule subclass
    """

    if devices is None:
        map_loc = "cpu"
    else:
        map_loc = f"cuda:{devices[0]}"

    state_dict = torch.load(checkpoint_path, weights_only=True, map_location=map_loc)[
        "state_dict"
    ]
    # NOTE: Some very hacky stuff to change the module names
    # This is because the original state dict was saved for a ValueFunctionLearner
    # But here we just want the ValueFunctionModule
    state_dict = {
        k.replace("base_value_function_module.", ""): v for k, v in state_dict.items()
    }
    return state_dict


def oracle_harm_type_lookup(harm_types: List[str]) -> HarmType:
    """
    Lookup for a list of Harmtypes based on string input.

    Parameters
    ----------
    harm_types : List[str]
        A list of string representations of the desired HarmTypes.

    Returns
    -------
    HarmType
        A list of HarmTypes corresponding to specific oracle harm policies.
    """

    oracle_harm_type_lookup = {
        "DANGEROUS": HarmType.DANGEROUS,
        "HATE": HarmType.HATE,
        "SEXUAL": HarmType.SEXUAL,
        "HARASSMENT": HarmType.HARASSMENT,
    }

    return [
        oracle_harm_type_lookup.get(harm_type, None) for harm_type in harm_types
    ]  # Return None if harm type not found
