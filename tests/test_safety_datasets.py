import pytest
import pandas as pd
import numpy as np

from unittest.mock import patch

from robust_multi_objective_decoding.data.safety_datasets import PKUSafeRLHFDataset
from datasets import Dataset


################## Fixtures ##################


@pytest.fixture(scope="module")
def dataset_one_harmtype():
    # Here we patch the load_from_disk method in the PKUSafeRLHFDataset class such that it returns a small dataset
    with patch(
        "robust_multi_objective_decoding.data.safety_datasets.load_from_disk",
        return_value=Dataset.from_pandas(df_data),
    ) as m:
        yield PKUSafeRLHFDataset(
            data_path="path/to/data",
            split="train",
            train_test_val_split=[0.34, 0.34, 0.32],
        )


@pytest.fixture(scope="module")
def dataset_all_harmtypes():
    with patch(
        "robust_multi_objective_decoding.data.safety_datasets.load_from_disk",
        return_value=Dataset.from_pandas(df_data),
    ) as m:
        yield PKUSafeRLHFDataset(
            data_path="path/to/data",
            split="train",
            harm_labels=["dangerous", "hate", "sexual", "harassment"],
            train_test_val_split=[0.34, 0.34, 0.32],
        )


@pytest.fixture(scope="module")
def dataset_prompt_template():
    with patch(
        "robust_multi_objective_decoding.data.safety_datasets.load_from_disk",
        return_value=Dataset.from_pandas(df_data),
    ) as m:
        yield PKUSafeRLHFDataset(
            data_path="path/to/data",
            split="train",
            harm_labels=["dangerous", "hate", "sexual", "harassment"],
            apply_prompt_template=True,
            train_test_val_split=[0.34, 0.34, 0.32],
        )


@pytest.fixture(scope="module")
def dataset_split_prompts():
    with patch(
        "robust_multi_objective_decoding.data.safety_datasets.load_from_disk",
        return_value=Dataset.from_pandas(df_split_repeated_prompt_data),
    ) as m:
        yield PKUSafeRLHFDataset(
            data_path="path/to/data",
            split="train",
            harm_labels=["dangerous", "hate", "sexual", "harassment"],
            train_test_val_split=[0.6, 0.2, 0.2],
        )


@pytest.fixture(scope="module")
def dataset_split_prompts_test():
    with patch(
        "robust_multi_objective_decoding.data.safety_datasets.load_from_disk",
        return_value=Dataset.from_pandas(df_split_repeated_prompt_data),
    ) as m:
        yield PKUSafeRLHFDataset(
            data_path="path/to/data",
            split="test",
            harm_labels=["dangerous", "hate", "sexual", "harassment"],
            train_test_val_split=[0.6, 0.2, 0.2],
        )


df_data = pd.DataFrame(
    {
        "tokens_shieldgemma_DANGEROUS_probs": [[0.6, 0.4]] * 15,
        "tokens_shieldgemma_HATE_probs": [[0.2, 0.8]] * 15,
        "tokens_shieldgemma_SEXUAL_probs": [[0.1, 0.3]] * 15,
        "tokens_shieldgemma_HARASSMENT_probs": [[0.0, 0.1]] * 15,
        "response_0": ["Once upon"] * 5 + ["Long ago"] * 5 + ["Life passed"] * 5,
        "prompt": ["Tell me a story"] * 5
        + ["Speak to me of the past"] * 5
        + ["Tell me things"] * 5,
    }
)

df_bad_data_wrong_length = pd.DataFrame(
    {
        "tokens_shieldgemma_DANGEROUS_probs": [[0.6, 0.4]] * 15,
        "tokens_shieldgemma_HATE_probs": [[0.2]] * 15,
        "tokens_shieldgemma_SEXUAL_probs": [[0.1, 0.3]] * 15,
        "tokens_shieldgemma_HARASSMENT_probs": [[0.0, 0.1]] * 15,
        "response_0": ["Once upon"] * 5 + ["Long ago"] * 5 + ["Life passed"] * 5,
        "prompt": ["Tell me a story"] * 5
        + ["Speak to me of the past"] * 5
        + ["Tell me things"] * 5,
    }
)

df_bad_missing_response = pd.DataFrame(
    {
        "tokens_shieldgemma_DANGEROUS_probs": [[0.6, 0.4]] * 5,
        "tokens_shieldgemma_HATE_probs": [[0.2, 0.2]] * 5,
        "tokens_shieldgemma_SEXUAL_probs": [[0.1, 0.3]] * 5,
        "tokens_shieldgemma_HARASSMENT_probs": [[0.0, 0.1]] * 5,
        "response_0": [""] + ["Once upon"] * 4,
        "prompt": ["Tell me a story"] * 5,
    }
)

df_filter_human_disagreements = pd.DataFrame(
    {
        "combined_harm_label": [[1] * 5 + [0], [1] * 6, [1] * 6, [1] * 6],
        "is_response_0_safe": [False, False, False, False],
    }
)

df_terminal_state_data = pd.DataFrame(
    {"combined_harm_label": [[1, 0, 1, 1, 0, 1, 0], [1, 1, 1, 1], [0, 0, 0, 0]]}
)

df_split_repeated_prompt_data = pd.DataFrame(
    {
        "tokens_shieldgemma_DANGEROUS_probs": [[0.6, 0.4]] * 15,
        "tokens_shieldgemma_HATE_probs": [[0.2, 0.2]] * 15,
        "tokens_shieldgemma_SEXUAL_probs": [[0.1, 0.3]] * 15,
        "tokens_shieldgemma_HARASSMENT_probs": [[0.0, 0.1]] * 15,
        "response_0": ["Long ago", "Once upon", "Sometime before"] * 5,
        "prompt": [
            "Tell me a story",
            "Hear this story now",
            "Think on this",
            "Weave me a tale",
            "Sit down and hear",
        ]
        * 3,
    }
)


################## Tests ##################


def test_safety_dataset_load_data(dataset_one_harmtype):
    data = dataset_one_harmtype.load_data("path/to/data")

    assert data is not None
    assert isinstance(data, Dataset)


def test_get_train_test_val_indices(dataset_all_harmtypes):
    train_upper, val_upper, test_upper = (
        dataset_all_harmtypes.get_train_test_val_indices(100, [0.8, 0.1, 0.1])
    )

    assert train_upper == 80
    assert val_upper == 90
    assert test_upper == 100


def test_safety_dataset_process_labels_cutoff(dataset_all_harmtypes):
    data = Dataset.from_pandas(df_data)
    out = dataset_all_harmtypes.process_labels_cutoff(data)

    assert out[0]["tokens_shieldgemma_DANGEROUS_probs"] == [1, 0]
    assert out[0]["tokens_shieldgemma_HATE_probs"] == [0, 1]
    assert out[0]["tokens_shieldgemma_SEXUAL_probs"] == [0, 0]
    assert out[0]["tokens_shieldgemma_HARASSMENT_probs"] == [0, 0]


def test_safety_dataset_check_labels(dataset_all_harmtypes):
    data = Dataset.from_pandas(df_bad_data_wrong_length)

    print(len(data))

    # This should raise an AssertionError
    with pytest.raises(AssertionError):
        dataset_all_harmtypes.check_labels(data)


def test_filter_responses_check_labels(dataset_all_harmtypes):
    data = Dataset.from_pandas(df_bad_missing_response)

    out = dataset_all_harmtypes.filter_responses(data)

    assert len(out) == 4


def test_combine_harm_labels(dataset_all_harmtypes):
    data = Dataset.from_pandas(df_data)

    # Manually calculate the harm labels:
    data = dataset_all_harmtypes.process_labels_cutoff(data)
    out = dataset_all_harmtypes.combine_harm_labels(data)

    assert out[0]["combined_harm_label"] == [0, 0]


def test_apply_terminal_state_to_combined_labels(dataset_all_harmtypes):
    data = Dataset.from_pandas(df_terminal_state_data)

    out = dataset_all_harmtypes.apply_terminal_state_to_combined_labels(data)

    assert out[0]["combined_harm_label"] == [
        1,
        1,
        1,
        1,
        1,
        1,
        0,
    ]  # Expecting all labels to be 1 except the last one
    assert out[1]["combined_harm_label"] == [1, 1, 1, 1]  # All labels should be 1
    assert out[2]["combined_harm_label"] == [0, 0, 0, 0]  # No change expected


def test_apply_prompt_template(dataset_prompt_template):
    """
    Check that the output has the prompt template applied.
    """

    out = dataset_prompt_template[0]
    assert out[1].splitlines()[0].strip() == "Prompt:"
    assert out[1].splitlines()[-2].strip() == "Answer:"


def test_filter_disagreement_with_human_labelers(dataset_all_harmtypes):
    data = Dataset.from_pandas(df_filter_human_disagreements)

    out = dataset_all_harmtypes.filter_disagreement_with_human_labelers(data)

    assert len(out) == 1
    assert out["combined_harm_label"][0][-1] == 0


def test_filter_safe_responses_to_size(dataset_all_harmtypes):
    data = Dataset.from_pandas(df_filter_human_disagreements)

    out = dataset_all_harmtypes.filter_safe_responses_to_size(data, 2)

    assert len(out) == 3
    assert (
        sum([label[-1] for label in out["combined_harm_label"]]) == 2
    )  # there should be two safe responses

    # Edge case where size is less than the number of safe responses
    out = dataset_all_harmtypes.filter_safe_responses_to_size(data, 0)

    assert len(out) == 1
    assert sum([label[-1] for label in out["combined_harm_label"]]) == 0

    # Edge case where size is greater than the number of safe responses
    with pytest.raises(ValueError):
        out = dataset_all_harmtypes.filter_safe_responses_to_size(data, 10)


def test_split_prompts(dataset_split_prompts, dataset_split_prompts_test):
    """
    Test that prompts are split correctly into train and test sets
    """

    out = dataset_split_prompts.data

    assert len(out) == 9
    assert len(np.unique(out["prompt"])) == 3

    out = dataset_split_prompts_test.data

    assert len(out) == 3
    assert len(np.unique(out["prompt"])) == 1
