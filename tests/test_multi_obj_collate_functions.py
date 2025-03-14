import pytest
import torch
from robust_multi_objective_decoding.data.multi_obj_collate_functions import (
    create_collate_functions,
)
from transformers import AutoTokenizer
import os

################## FIXTURES ##################

IN_GITHUB_ACTIONS = os.getenv("GITHUB_ACTIONS") == "true"
NO_HUB_AUTH_ID = os.getenv("HUGGINGFACE_HUB_TOKEN") is None


@pytest.fixture
def mock_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained("openaccess-ai-collective/tiny-mistral")
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


@pytest.fixture
def mock_it_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b-it")
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


################### TESTS ####################
# TODO: simplify the tests -> a lot of repeated code


# 1. Test the eos reward collate function
def test_create_eos_reward(mock_tokenizer):
    # The test will be the following, we have some sentences of varying length
    collate_fn = create_collate_functions(
        tokenizer=mock_tokenizer,
        reward_collations=["eos"],
        max_length=100,
        rand_len=False,
    )

    # idx, prompt, response, labels
    batch = [
        (0, "Tell me a story", "Once upon a time there was a big bad wolf", [-1.0]),
        (1, "Tell me a story", "Once upon a time there was a happy bunny", [1.0]),
    ]

    output = collate_fn(batch)

    assert (output["rewards"][..., -1] == torch.tensor([[-1], [1]])).all()
    assert output["input_ids"].shape[0] == output["rewards"].shape[0]
    assert output["input_ids"].shape[1] == output["rewards"].shape[2]
    assert (output["rewards"][..., 0] == torch.tensor([[-100], [-100]])).all()

    # Test the random length version:
    collate_fn = create_collate_functions(
        tokenizer=mock_tokenizer,
        reward_collations=["eos"],
        max_length=100,
        rand_len=True,
        rand_len_range=[8, 10],
    )

    output = collate_fn(batch)

    # Assert the output is correctly formatted:
    assert (output["rewards"][..., -1] == torch.tensor([[-1], [1]])).all()
    assert output["input_ids"].shape[0] == output["rewards"].shape[0]
    assert output["input_ids"].shape[1] == output["rewards"].shape[2]
    assert (output["rewards"][..., 0] == torch.tensor([[-100], [-100]])).all()

    # Check the random lengths are correct:
    len_tok_prompt = mock_tokenizer(["Tell me a story"], return_tensors="pt")[
        "input_ids"
    ].shape[1]

    # Find the number of tokens in the response:
    safe_labels = output["rewards"]
    len_resp = (safe_labels != -100).sum(dim=2)[:, 0]

    # TODO: Can we write some more specific tests here?

    # Find the number of tokens in the prompt response encoding:
    input_ids = output["input_ids"]
    len_prompt_resp = (input_ids != mock_tokenizer.pad_token_id).sum(dim=1)

    # Ensure the safe labels and inputs are consistent:
    assert ((len_resp + len_tok_prompt) == len_prompt_resp).all()


# 2. Test the cbf reward collate function
def test_create_cbf_reward(mock_tokenizer):
    # The test will be the following, we have some sentences of varying length
    collate_fn = create_collate_functions(
        tokenizer=mock_tokenizer,
        reward_collations=["cbf"],
        max_length=100,
        rand_len=False,
    )

    # idx, prompt, response, labels
    batch = [
        (
            0,
            "Tell me a story",
            "Once upon a time there was a big bad wolf",
            [[1, 1, 1, 1, 1, 1, 1, 1, 0, 0]],
        ),
        (
            1,
            "Tell me a story",
            "Once upon a time there was a happy bunny",
            [[1, 1, 1, 1, 1, 1, 1, 1, 1]],
        ),
    ]

    output = collate_fn(batch)

    assert (output["rewards"][..., -1] == torch.tensor([[0], [1]])).all()
    assert output["input_ids"].shape[0] == output["rewards"].shape[0]
    assert output["input_ids"].shape[1] == output["rewards"].shape[2]
    assert (output["rewards"][..., 0] == torch.tensor([[-100], [-100]])).all()

    # CBF specific tests:

    # Test the random length version:
    collate_fn = create_collate_functions(
        tokenizer=mock_tokenizer,
        reward_collations=["cbf"],
        max_length=100,
        rand_len=True,
        rand_len_range=[8, 10],
    )

    output = collate_fn(batch)
    decoded_output = mock_tokenizer.decode(output["input_ids"][0])

    # Assert the output is correct:
    if decoded_output.split(" ")[-1] not in ["bad", "wolf"]:
        assert (output["rewards"][..., -1] == torch.tensor([[1], [1]])).all()
    else:
        assert (output["rewards"][..., -1] == torch.tensor([[0], [1]])).all()
    assert output["input_ids"].shape[0] == output["rewards"].shape[0]
    assert output["input_ids"].shape[1] == output["rewards"].shape[2]
    assert (output["rewards"][..., 0] == torch.tensor([[-100], [-100]])).all()

    # Check the random lengths are correct:
    len_tok_prompt = mock_tokenizer(["Tell me a story"], return_tensors="pt")[
        "input_ids"
    ].shape[1]

    # Find the number of tokens in the response:
    safe_labels = output["rewards"]
    len_resp = (safe_labels != -100).sum(dim=2)[:, 0]

    # TODO: Can we write some more specific tests here?

    # Find the number of tokens in the prompt response encoding:
    input_ids = output["input_ids"]
    len_prompt_resp = (input_ids != mock_tokenizer.pad_token_id).sum(dim=1)

    # Ensure the safe labels and inputs are consistent:
    assert ((len_resp + len_tok_prompt) == len_prompt_resp).all()


# 4. Test multiple different collate functions -> testing the list feature of the create_collate_functions
def test_create_multiple_rewards(mock_tokenizer):
    # The test will be the following, we have some sentences of varying length
    collate_fn = create_collate_functions(
        tokenizer=mock_tokenizer,
        reward_collations=["eos", "cbf"],
        max_length=100,
        rand_len=False,
    )

    # idx, prompt, response, labels
    batch = [
        (
            0,
            "Tell me a story",
            "Once upon a time there was a big bad wolf",
            [-1, [1, 1, 1, 1, 1, 1, 1, 1, 0, 0]],
        ),
        (
            1,
            "Tell me a story",
            "Once upon a time there was a happy bunny",
            [1, [1, 1, 1, 1, 1, 1, 1, 1, 1]],
        ),
    ]

    output = collate_fn(batch)

    print(output["rewards"])
    print(output["rewards"][..., -1])

    assert (output["rewards"][..., -1] == torch.tensor([[-1, 0], [1, 1]])).all()
    assert output["input_ids"].shape[0] == output["rewards"].shape[0]
    assert output["input_ids"].shape[1] == output["rewards"].shape[2]
    assert (
        output["rewards"][..., 0] == torch.tensor([[-100, -100], [-100, -100]])
    ).all()

    # CBF specific tests:

    # Test the random length version:
    collate_fn = create_collate_functions(
        tokenizer=mock_tokenizer,
        reward_collations=["eos", "cbf"],
        max_length=100,
        rand_len=True,
        rand_len_range=[8, 10],
    )

    output = collate_fn(batch)
    decoded_output = mock_tokenizer.decode(output["input_ids"][0])

    # Assert the output is correct:
    if decoded_output.split(" ")[-1] not in ["bad", "wolf"]:
        assert (output["rewards"][..., -1] == torch.tensor([[-1, 1], [1, 1]])).all()
    else:
        assert (output["rewards"][..., -1] == torch.tensor([[-1, 0], [1, 1]])).all()
    assert output["input_ids"].shape[0] == output["rewards"].shape[0]
    assert output["input_ids"].shape[1] == output["rewards"].shape[2]
    assert (
        output["rewards"][..., 0] == torch.tensor([[-100, -100], [-100, -100]])
    ).all()


# 1. Test the eos reward collate function -> unimplemented in favour of simpler solution
# @pytest.mark.skipif(IN_GITHUB_ACTIONS or NO_HUB_AUTH_ID, reason="google/gemma-2-2b-it tokenizer does not run in Github Actions/No Hugging Face Token")
# def test_create_eos_reward_apply_chat_template(mock_it_tokenizer):

#     # The test will be the following, we have some sentences of varying length
#     collate_fn = create_collate_functions(tokenizer=mock_it_tokenizer,
#                                         reward_collations=['eos'],
#                                         max_length=100,
#                                         rand_len=False,
#                                         apply_chat_template=True)

#     # idx, prompt, response, labels
#     batch = [(0, 'Tell me a story', 'Once upon a time there was a big bad wolf', [-1.0]),
#             (1, 'Tell me a story', 'Once upon a time there was a happy bunny', [1.0])]

#     output = collate_fn(batch)

#     assert (output["rewards"][..., -1] == torch.tensor([[-1], [1]])).all()
#     assert output['input_ids'].shape[0] == output['rewards'].shape[0]
#     assert output['input_ids'].shape[1] == output['rewards'].shape[2]
#     assert (output['rewards'][..., 0] == torch.tensor([[-100], [-100]])).all()

#     # Test the random length version:
#     collate_fn = create_collate_functions(tokenizer=mock_it_tokenizer,
#                                         reward_collations=['eos'],
#                                         max_length=100,
#                                         rand_len=True,
#                                         rand_len_range=[8,10])

#     output = collate_fn(batch)

#     # Assert the output is correctly formatted:
#     assert (output["rewards"][..., -1] == torch.tensor([[-1], [1]])).all()
#     assert output['input_ids'].shape[0] == output['rewards'].shape[0]
#     assert output['input_ids'].shape[1] == output['rewards'].shape[2]
#     assert (output['rewards'][..., 0] == torch.tensor([[-100], [-100]])).all()

#     # Check the random lengths are correct:
#     len_tok_prompt = mock_it_tokenizer(['Tell me a story'], return_tensors='pt')['input_ids'].shape[1]

#     # Find the number of tokens in the response:
#     safe_labels = output['rewards']
#     len_resp = (safe_labels != -100).sum(dim=2)[:,0]

#     #TODO: Can we write some more specific tests here?

#     # Find the number of tokens in the prompt response encoding:
#     input_ids = output['input_ids']
#     len_prompt_resp = (input_ids != mock_it_tokenizer.pad_token_id).sum(dim=1)

#     # Ensure the safe labels and inputs are consistent:
#     assert ((len_resp + len_tok_prompt) == len_prompt_resp).all()
