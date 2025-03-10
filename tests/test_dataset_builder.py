import pytest
import tempfile
import pathlib
from robust_multi_objective_decoding.dataset_builder import (
    CompletionDataset,
    Completion,
    PromptDataset,
    TinystoriesPromptDatasetBuilder,
)
from robust_multi_objective_decoding.model_handler import GenerationConfig


@pytest.fixture
def mock_completion_dataset():
    return CompletionDataset(
        dataset_name="test_dataset",
        model_name="test_model",
        generation_config=GenerationConfig(
            max_new_tokens=10, do_sample=True, temperature=0.7
        ),
        completions=[
            Completion(prompt="Hello", response="World"),
            Completion(prompt="Test", response="Passed"),
        ],
    )


@pytest.mark.parametrize(
    "split",
    [
        "train",
        "validation",
    ],
)
def test_tinystories_dataset_builder(split):
    builder = TinystoriesPromptDatasetBuilder(split=split)
    dataset = builder.build()
    assert isinstance(dataset, PromptDataset)
    prompts = dataset.prompts
    assert isinstance(prompts, list)
    assert isinstance(prompts[0], str)
    assert len(prompts) > 0
