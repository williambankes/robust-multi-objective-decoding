"""Utilities for managing torch device"""

import torch
from contextlib import contextmanager
from typing import Generator


def _init_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        # Parse the PyTorch version to check if it's below version 2.0
        major_version = int(torch.__version__.split(".")[0])
        if major_version >= 2:
            return "mps"
        else:
            return "cpu"
    else:
        return "cpu"


_DEFAULT_DEVICE = _init_device()


def get_device() -> str:
    global _DEFAULT_DEVICE
    return _DEFAULT_DEVICE


def set_device(device: str) -> None:
    global _DEFAULT_DEVICE
    _DEFAULT_DEVICE = device


@contextmanager
def use_device(device: str) -> Generator:
    global _DEFAULT_DEVICE
    old_device = _DEFAULT_DEVICE
    _DEFAULT_DEVICE = device
    yield
    _DEFAULT_DEVICE = old_device
