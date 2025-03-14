import pytest

from robust_multi_objective_decoding.dataset_builder import (
    PromptDataset,
    TinystoriesPromptDatasetBuilder,
)
from robust_multi_objective_decoding.model_handler import ModelHandler
from robust_multi_objective_decoding.oracles.oracle import CompositeOracle

from robust_multi_objective_decoding.oracles.simple import (
    CharacterLengthOracle,
    PunctuationOracle,
)


@pytest.fixture(scope="module")
def base_model():
    return ModelHandler.from_pretrained("EleutherAI/pythia-70m")


@pytest.fixture(scope="module")
def model_with_decoder():
    # TODO
    return ModelHandler.from_pretrained("EleutherAI/pythia-70m")


@pytest.fixture(scope="module")
def model_with_cbf_decoder():
    # TODO
    return ModelHandler.from_pretrained("EleutherAI/pythia-70m")


@pytest.fixture(scope="module")
def dataset():
    full_dataset = TinystoriesPromptDatasetBuilder(split="train").build()
    return PromptDataset(full_dataset.name, full_dataset.prompts[:3])


@pytest.fixture(scope="module")
def oracle():
    return CompositeOracle([CharacterLengthOracle(max_length=25), PunctuationOracle()])
