# -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 10:28:09 2024

@author: William
"""

from unittest.mock import patch
from robust_multi_objective_decoding.utils.utils import parse_int_or_list
from robust_multi_objective_decoding.utils.load_utils import (
    load_base_vf_module_state_dict_from_checkpoint,
)

############# FIXTURES #############

# TODO: put a better proxy of a checkpoint here for testing
proxy_checkpoint = {"state_dict": {"test": 1}}


############# TESTS #############


def test_parse_int_or_list():
    inputs = ["3", "[3]", "[1,2,3]"]

    for input_ in inputs:
        parse_int_or_list(input_)


def test_load_base_vf_module_state_dict_from_checkpoint():
    with patch(
        "robust_multi_objective_decoding.utils.load_utils.torch.load",
        return_value=proxy_checkpoint,
    ) as m:
        output = load_base_vf_module_state_dict_from_checkpoint(".")

    # Write assertion tests on this based on the output:
    assert output.get("test", None) == 1
