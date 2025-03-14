import pytest
import torch

from typing import List
from robust_multi_objective_decoding.data.collate_functions import (
    adjust_label_to_token_length,
    create_word_break_tokenizer_collate_fn,
    create_word_break_tokenizer_rand_len_collate_fn,
    create_eos_reward_tokenizer_collate_fn,
    create_eos_reward_tokenizer_rand_len_collate_fn,
    create_classifier_collate_fn,
    create_classifier_rand_len_collate_fn,
    pad_safety_labels,
)
from transformers import AutoTokenizer


################## FIXTURES ##################


class ProxyTokenizer:
    i = 0

    # Default Gemma 2 settings
    padding_side = "left"
    truncation_side = "right"

    def __call__(self, data: List[str], *args, **kwargs):
        batch_size = len(data)

        # Find max sequence length:
        max_len = 0
        for d in data:
            max_len = max(max_len, len(d.split()))

        return {
            "input_ids": torch.ones(batch_size, max_len),
            "attention_mask": torch.ones(batch_size, max_len),
        }


@pytest.fixture
def mock_tokenizer():
    return ProxyTokenizer()


################### TESTS ####################


def test_create_eos_reward_tokenizer_collate_fn():
    tokenizer = ProxyTokenizer()

    # The test will be the following, we have some sentences of varying length
    collate_fn = create_eos_reward_tokenizer_collate_fn(tokenizer, max_length=100)

    # idx, prompt, response, labels
    batch = [
        (
            0,
            "Tell me a story",
            "Once upon a time there was a big bad wolf",
            torch.tensor([1, 1, 1, 1, 1, 1, 1, 0, 0, 0]),
        ),
        (
            1,
            "Tell me a story",
            "Once upon a time there was a happy bunny",
            torch.tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
        ),
    ]

    output = collate_fn(batch)

    assert (output["tokenwise_safety_labels"][:, -1] == torch.tensor([-1, 1])).all()


def test_create_eos_reward_tokenizer_rand_len_collate_fn():
    tokenizer = AutoTokenizer.from_pretrained("openaccess-ai-collective/tiny-mistral")
    tokenizer.pad_token_id = 0

    # The test will be the following, we have some sentences of varying length
    collate_fn = create_eos_reward_tokenizer_rand_len_collate_fn(
        tokenizer, max_length=400
    )

    story1 = "Once upon a time there was a big bad wolf "
    story2 = "Once upon a time there was a happy bunny "

    # idx, prompt, response, labels
    batch = [
        (0, "Tell me a story", story1, torch.tensor([1] * 100 + [0] * 28)),
        (1, "Tell me a story", story2, torch.tensor([1] * 128)),
    ]

    output = collate_fn(batch)

    assert (output["tokenwise_safety_labels"][:, -1] == torch.tensor([-1, 1])).all()
    assert output["input_ids"].shape == output["tokenwise_safety_labels"].shape


def test_create_work_break_tokenizer_rand_len_collate_fn():
    tokenizer = AutoTokenizer.from_pretrained("openaccess-ai-collective/tiny-mistral")
    tokenizer.pad_token_id = 0

    # The test will be the following, we have some sentences of varying length
    collate_fn = create_word_break_tokenizer_rand_len_collate_fn(
        tokenizer, max_length=400
    )

    story = ("Once upon " * 64).strip()

    # idx, prompt, response, labels
    batch = [
        (0, "Tell me a story", story, torch.tensor([1] * 70 + [0] * 58)),
        (1, "Tell me a story", story, torch.tensor([1] * 128)),
    ]

    output = collate_fn(batch)

    assert (output["tokenwise_safety_labels"][:, -1] == torch.tensor([0, 1])).all()
    assert (output["tokenwise_safety_labels"][:, 0] == torch.tensor([-100, -100])).all()

    # Ensure the input_ids and the tokenwise_safety_labels are the correct length:
    len_tok_prompt = tokenizer(["Tell me a story"], return_tensors="pt")[
        "input_ids"
    ].shape[1]

    # Find the number of tokens in the response:
    safe_labels = output["tokenwise_safety_labels"]
    len_resp = (safe_labels != -100).sum(dim=1)

    # Find the number of tokens in the prompt response encoding:
    input_ids = output["input_ids"]
    len_prompt_resp = (input_ids != 0).sum(dim=1)

    # Ensure the safe labels and inputs are consistent:
    assert ((len_resp + len_tok_prompt) == len_prompt_resp).all()


def test_create_word_break_tokenizer_collate_fn():
    tokenizer = AutoTokenizer.from_pretrained("openaccess-ai-collective/tiny-mistral")
    tokenizer.pad_token_id = 0

    idx = [0, 1]

    prompt = ["Tell me a story", "How should we treat people?"]

    data = [
        "Once upon a time there was a big bad wolf",
        "I think we should be really nasty to everyone and everything",
    ]

    labels = [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0]]

    batch = list()
    for i, label in enumerate(labels):
        assert len(label) == len(data[i].split())
        batch.append((idx[i], prompt[i], data[i], label))

    collate_fn = create_word_break_tokenizer_collate_fn(tokenizer, max_length=100)

    output = collate_fn(batch)

    # Check the input _ids and attention mask are correct
    assert output.get("input_ids", None) is not None
    assert output["input_ids"].shape == (2, 18)
    assert output.get("attention_mask", None) is not None
    assert output["attention_mask"].shape == output["input_ids"].shape

    assert output.get("tokenwise_safety_labels", None) is not None
    assert output["tokenwise_safety_labels"].shape == output["input_ids"].shape

    # We pad the beginning of the shorter first sequence
    assert output["tokenwise_safety_labels"][0, -1] == 1

    # The second sequence is longer, so we check the last token is unsafe
    assert output["tokenwise_safety_labels"][1, -1] == 0

    # Check the first tokens
    assert output["tokenwise_safety_labels"][0, 0] == -100
    assert output["tokenwise_safety_labels"][1, 0] == -100


def test_adjust_label_to_token_length(mock_tokenizer):
    # Test with varying safety input:
    data = "Once upon a time there was a big bad wolf"
    label = [1, 1, 1, 1, 1, 1, 1, 0, 0, 0]

    adjusted_label = adjust_label_to_token_length(data, label, mock_tokenizer)
    tokenized_data = mock_tokenizer([data], return_tensors="pt")

    assert len(adjusted_label) == (tokenized_data["input_ids"].shape[1] - 1)

    # Test with varying safety input:
    data = "Once upon a time there was a big bad wolf"
    label = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

    adjusted_label = adjust_label_to_token_length(data, label, mock_tokenizer)
    tokenized_data = mock_tokenizer([data], return_tensors="pt")

    assert len(adjusted_label) == (tokenized_data["input_ids"].shape[1] - 1)

    # Test with varying safety input:
    data = "Once upon a time there was a big bad wolf"
    label = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    adjusted_label = adjust_label_to_token_length(data, label, mock_tokenizer)
    tokenized_data = mock_tokenizer([data], return_tensors="pt")

    assert len(adjusted_label) == (tokenized_data["input_ids"].shape[1] - 1)


def test_pad_safety_labels():
    len_tok_prompt = 5
    len_tok_response = (
        10  # we assume the labels are the same length as the tokenized response
    )
    labels = [1] * len_tok_response  # The input labels don't matter as we just pad them

    # For too big check padding both sides:
    max_shape = 100
    padding_side = "left"
    truncation_side = "left"
    padded_labels = pad_safety_labels(
        labels,
        len_tok_response,
        len_tok_prompt,
        padding_side,
        truncation_side,
        max_shape,
    )

    assert len(padded_labels) == max_shape
    assert padded_labels[0] == -100  # Check the max shape is padded
    assert padded_labels[-1] == 1  # Check the response is non-padded
    assert (
        padded_labels[len_tok_response - len_tok_prompt] == -100
    )  # Check the prompt is padded

    padding_side = "right"
    padded_labels = pad_safety_labels(
        labels,
        len_tok_response,
        len_tok_prompt,
        padding_side,
        truncation_side,
        max_shape,
    )

    assert len(padded_labels) == max_shape
    assert padded_labels[0] == -100  # Check the prompt is padded
    assert padded_labels[-1] == -100  # Check the max shape is non-padded
    assert padded_labels[len_tok_prompt + 1] == 1  # Check the response is non-padded

    # For too small check trunction:
    max_shape = 12
    truncation_side = "right"
    padded_labels = pad_safety_labels(
        labels,
        len_tok_response,
        len_tok_prompt,
        padding_side,
        truncation_side,
        max_shape,
    )

    assert len(padded_labels) == max_shape
    assert padded_labels[0] == -100  # Check the prompt is padded
    assert padded_labels[-1] == 1  # Check the response is non-padded

    truncation_side = "left"
    padded_labels = pad_safety_labels(
        labels,
        len_tok_response,
        len_tok_prompt,
        padding_side,
        truncation_side,
        max_shape,
    )

    assert len(padded_labels) == max_shape
    assert padded_labels[0] == -100  # Check the prompt is padded
    assert padded_labels[-1] == 1  # Check the response is non-padded

    # Test small setup:
    len_tok_prompt = 2
    len_tok_response = (
        2  # we assume the labels are the same length as the tokenized response
    )
    labels = [1] * len_tok_response
    max_shape = 12
    padding_side = "left"
    truncation_side = "left"
    padded_labels = pad_safety_labels(
        labels,
        len_tok_response,
        len_tok_prompt,
        padding_side,
        truncation_side,
        max_shape,
    )

    assert len(padded_labels) == max_shape
    assert padded_labels[0] == -100  # Check the max shape is padded
    assert padded_labels[-1] == 1  # Check the response is non-padded
    assert (
        padded_labels[len_tok_response - len_tok_prompt] == -100
    )  # Check the prompt is padded

    padding_side = "right"
    padded_labels = pad_safety_labels(
        labels,
        len_tok_response,
        len_tok_prompt,
        padding_side,
        truncation_side,
        max_shape,
    )

    assert len(padded_labels) == max_shape
    assert padded_labels[0] == -100  # Check the prompt is padded
    assert padded_labels[-1] == -100  # Check the max shape is non-padded
    assert padded_labels[len_tok_prompt + 1] == 1  # Check the response is non-padded


def test_classifier_collate_fn():
    tokenizer = AutoTokenizer.from_pretrained("openaccess-ai-collective/tiny-mistral")
    tokenizer.pad_token_id = 0

    print(tokenizer.padding_side)
    print(tokenizer.truncation_side)

    idx = [0, 1]

    prompt = ["Tell me a story", "How should we treat people?"]

    data = [
        "Once upon a time there was a big bad wolf",
        "I think we should be really nasty to everyone and everything",
    ]

    labels = [[1, 1, 1, 1, 1, 1, 1, 0, 0, 1], [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0]]

    batch = list()
    for i, label in enumerate(labels):
        assert len(label) == len(data[i].split())
        batch.append((idx[i], prompt[i], data[i], label))

    collate_fn = create_classifier_collate_fn(tokenizer, max_length=100)

    output = collate_fn(batch)

    # Check the input _ids and attention mask are correct
    assert output.get("input_ids", None) is not None
    assert output["input_ids"].shape == (2, 18)
    assert output.get("attention_mask", None) is not None
    assert output["attention_mask"].shape == output["input_ids"].shape

    assert output.get("safety_labels", None) is not None
    assert tuple(output["safety_labels"].shape) == (2,)

    # Check the safety labels are returned correctly
    assert output["safety_labels"][0] == 1
    assert output["safety_labels"][1] == 0


def test_classifier_rand_len_collate_fn():
    tokenizer = AutoTokenizer.from_pretrained("openaccess-ai-collective/tiny-mistral")
    tokenizer.pad_token_id = 0

    story = ("Once upon " * 64).strip()

    # idx, prompt, response, labels
    batch = [
        (0, "Tell me a story", story, torch.tensor([1] * 70 + [0] * 58)),
        (1, "Tell me a story", story, torch.tensor([1] * 128)),
    ]

    collate_fn = create_classifier_rand_len_collate_fn(tokenizer, max_length=100)

    output = collate_fn(batch)

    # Check the input _ids and attention mask are correct
    assert output.get("input_ids", None) is not None
    assert output.get("attention_mask", None) is not None
    assert output["attention_mask"].shape == output["input_ids"].shape

    assert output.get("safety_labels", None) is not None
    assert tuple(output["safety_labels"].shape) == (2,)

    # Check the safety labels are returned correctly
    assert output["safety_labels"][0] == 0
    assert output["safety_labels"][1] == 1

    len_tok_resp = tokenizer([story], return_tensors="pt")["input_ids"].shape[1]
    len_tok_prompt = tokenizer(["Tell me a story"], return_tensors="pt")[
        "input_ids"
    ].shape[1]

    assert output["input_ids"].shape[1] <= len_tok_resp + len_tok_prompt
