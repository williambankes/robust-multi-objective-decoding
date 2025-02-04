# -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 10:15:19 2024

@author: William
"""

import os
import argparse
import torch
from typing import List
from huggingface_hub import login
from omegaconf import DictConfig, OmegaConf
import rich.syntax
import rich.tree

def parse_int_or_list(value:str) -> int | List[int]:
    """
    Parse int or list inputs from command line

    Parameters
    ----------
    value : str
        A string input in the form of a list or int.

    Raises
    ------
    argparse
        If the input is not in the form of an int or list raises error.

    Returns
    -------
    int|List[int]
        Returns the string converted to either an int or List[int].
    """
    
    #Try converting to int:
    try:
        return int(value)
    
    except (ValueError):
        
        # If it's not an int, assume it's a list
        try:
            strip_brackets = value.strip('[]')
            
            #Deal with the case where the list has length 1
            if len(strip_brackets.split(',')) == 1:
                return [int(strip_brackets)]
            #Deal with the case where the list is length > 1
            else:
                return list(map(int, value.strip('[]').split(',')))
        except (TypeError, AttributeError):
            
            raise argparse.ArgumentTypeError(f"Value: {value} should be an integer or a list of integers (e.g., '3' or '[1,2,3]')")
        

def setup_huggingface_auth(env_var_name:str='HUGGINGFACE_HUB_TOKEN'):

    # Setup huggingface hub login
        huggingface_hub_token = os.getenv(env_var_name)

        if huggingface_hub_token:
            # Log in to the Hugging Face Hub using the token
            login(token=huggingface_hub_token)
            print("Successfully logged in to Hugging Face Hub.")
        else:
            print("HUGGINGFACE_HUB_TOKEN environment variable not set.")


def setup_checkpoint_dir(checkpoint_dir:str='checkpoints'):
    """
    Create checkpoint directory

    Parameters
    ----------
    checkpoint_dir : str, optional
        The name of the checkpoint directory. The default is 'checkpoints'.

    Returns
    -------
    None.

    """
    # Create checkpoint directory:
    curr_dir = os.getcwd()
    checkpoint_dir = os.path.join(curr_dir, "checkpoints")
    
    if not os.path.exists(checkpoint_dir):
        print(f'Creating checkpoint dir: {checkpoint_dir}')
        os.mkdir(checkpoint_dir)
    else: 
        print(f'Checkpoint dir: {checkpoint_dir} exists')

    return checkpoint_dir

def torch_dtype_lookup(dtype:str) -> torch.dtype:
    """
    Lookup for torch dtype based on string input.

    Parameters
    ----------
    dtype : str
        The string representation of the desired torch dtype.

    Returns
    -------
    torch.dtype
        The corresponding torch dtype.
    """
    torch_dtype_lookup = {
        'bfloat16': torch.bfloat16,
        'float16': torch.float16,
        'float32': torch.float32,
        'float64': torch.float64,
        'int8': torch.int8,
        'int16': torch.int16,
        'int32': torch.int32
    }
    
    return torch_dtype_lookup.get(dtype, None)  # Return None if dtype not found

def print_config(
    config: DictConfig,
    fields: list[str] = (
        "trainer",
        "learner",
        "model",
        "dataset",
        "dataloader",
        "collate_fn",
        "tokenizer",
        "callbacks",
        "logger",
    ),
    resolve: bool = True,
) -> None:
    """Prints content of DictConfig using Rich library and its tree structure.

    Args:
        config (DictConfig): Configuration composed by Hydra.
        fields (Sequence[str], optional): Determines which main fields from config will
        be printed and in what order.
        resolve (bool, optional): Whether to resolve reference fields of DictConfig.
    """

    style = "dim"
    tree = rich.tree.Tree("CONFIG", style=style, guide_style=style)

    for field in fields:
        branch = tree.add(field, style=style, guide_style=style)

        config_section = config.get(field)
        branch_content = str(config_section)
        if isinstance(config_section, DictConfig):
            branch_content = OmegaConf.to_yaml(config_section, resolve=resolve)

        branch.add(rich.syntax.Syntax(branch_content, "yaml"))

    rich.print(tree)

    with open("config_tree.txt", "w") as fp:
        rich.print(tree, file=fp)