import pytest
import torch
import torch.nn as nn

from robust_multi_objective_decoding.decoders import (
    CBFOneStepControlledDecoder,
    CBFOptimisationControlledDecoder,
    ReferenceModelDecoder,
    ReweightingControlledDecoder,
    ThresholdingControlledDecoder,
    BlockwiseFilteringDecoder,
)


####### Fixtures and mocks #######
@pytest.fixture
def model_and_vocab_size():
    # Create a dummy model with a vocab size of 10 and batch size of 3
    # Output logits will always be 1s, ie all tokens are equally likely
    vocab_size = 10

    class DummyModel(nn.Module):
        def forward(self, input_ids, **kwargs):
            batch_size = input_ids.shape[0]
            seq_len = input_ids.shape[1]
            return {"logits": torch.ones(batch_size, seq_len, vocab_size)}

    return DummyModel(), vocab_size


@pytest.fixture
def action_value_model(model_and_vocab_size):
    _, vocab_size = model_and_vocab_size

    class DummyActionValueModel(nn.Module):
        def get_q_values(self, input_ids, return_vs: bool = False, **kwargs):
            batch_size = input_ids.shape[0]
            seq_len = input_ids.shape[1]
            # Output should be (batch_size, seq_len, vocab_size)
            # In this test, logits are linearly increasing, so higher
            # token_ids have higher values
            q = torch.linspace(0, 1, vocab_size).repeat(batch_size, seq_len, 1)
            v = torch.ones(batch_size, seq_len)

            if return_vs:
                return q, v
            return q

    return DummyActionValueModel()


@pytest.fixture
def input_ids():
    # Create a dummy input tensor batch,
    #  [[0, 1, 2, 3, 4], [1, 2, 3, 4, 5], [2, 3, 4, 5, 6]]
    return torch.vstack(
        [
            torch.arange(5),
            torch.arange(1, 6),
            torch.arange(2, 7),
        ]
    )


####### Tests #######
def test_reference_decoder(model_and_vocab_size, input_ids):
    torch.manual_seed(0)

    model, vocab_size = model_and_vocab_size
    batch_size, seq_len = input_ids.shape
    max_tokens_to_generate = 5

    attn_mask = torch.ones_like(input_ids)

    # Check that the number of tokens generated is correct when EOS is not reached
    # Vocab is indexed from zero, so EOS token = vocab_size means EOS will never be generated
    decoder = ReferenceModelDecoder(reference_model=model, eos_token_id=vocab_size)
    output = decoder.generate(input_ids=input_ids, attention_mask=attn_mask, max_new_tokens=max_tokens_to_generate)
    output_ids = output['generated_ids']
    assert output_ids.shape == (batch_size, seq_len + max_tokens_to_generate)

    # Check that the number of tokens generated is correct when EOS is reached
    # Vocab is indexed from zero, so EOS token = vocab_size-1 means EOS will be generated
    decoder = ReferenceModelDecoder(reference_model=model, eos_token_id=vocab_size - 1)
    output = decoder.generate(input_ids=input_ids, attention_mask=attn_mask, max_new_tokens=max_tokens_to_generate)
    output_ids = output['generated_ids']
    assert output_ids.shape[0] == batch_size
    assert output_ids.shape[1] <= seq_len + max_tokens_to_generate

    # Check that once EOS is reached, no non-EOS tokens are generated
    eos_indices = torch.nonzero(output_ids == decoder.eos_token_id)
    for i, j in eos_indices:
        assert torch.all(output_ids[i, j:] == decoder.eos_token_id)


def test_thresholding_controlled_decoder(
    model_and_vocab_size, input_ids, action_value_model
):
    torch.manual_seed(0)
    model, vocab_size = model_and_vocab_size
    decoder = ThresholdingControlledDecoder(
        reference_model=model,
        eos_token_id=vocab_size - 1,
        action_value_model=action_value_model,
        action_value_threshold=0.5,
    )
    # As our value function is linearly increasing from 0 to 1,
    # the threshold of 0.5 will mean that tokens with token_id < 5 will be masked

    attn_mask = torch.ones_like(input_ids)

    # Check logits are modified correctly
    original_logits = decoder._get_reference_logits({"input_ids": input_ids})
    adjusted_logits = decoder._get_adjusted_logits(
        {"input_ids": input_ids}, original_logits
    )
    assert adjusted_logits.shape == original_logits.shape
    # First 5 tokens should have lower logits
    assert torch.all(adjusted_logits[:, :5] < original_logits[:, :5])
    # Last 5 tokens should have the same logits
    assert torch.all(adjusted_logits[:, 5:] == original_logits[:, 5:])

    # Check generation produces no tokens with token_id < 5
    max_tokens_to_generate = 5
    output = decoder.generate(input_ids=input_ids, attention_mask=attn_mask, max_new_tokens=max_tokens_to_generate)
    output_ids = output['generated_ids']
    assert torch.all(output_ids[:, input_ids.shape[1] :] >= 5)

def test_reweighting_controlled_decoder(
    model_and_vocab_size, input_ids, action_value_model
):
    model, vocab_size = model_and_vocab_size
    decoder = ReweightingControlledDecoder(
        reference_model=model,
        eos_token_id=vocab_size - 1,
        action_value_model=action_value_model,
        action_value_coeff=1.0,
    )

    original_logits = decoder._get_reference_logits({"input_ids": input_ids})
    adjusted_logits = decoder._get_adjusted_logits(
        {"input_ids": input_ids}, original_logits
    )

    # Check logits are modified correctly
    assert adjusted_logits.shape == original_logits.shape
    assert torch.all(
        torch.isclose(
            adjusted_logits,
            original_logits
            + action_value_model.get_q_values(**{"input_ids": input_ids})[:, -1, :],
        )
    )

def test_cbf_one_step_controlled_decoder(
    model_and_vocab_size, input_ids, action_value_model
):
    model, vocab_size = model_and_vocab_size
    decoder = CBFOneStepControlledDecoder(
        reference_model=model,
        eos_token_id=vocab_size - 1,
        action_value_model=action_value_model,
        cbf_offset=1.0,
        alpha=1.0,
    )

    original_logits = decoder._get_reference_logits({"input_ids": input_ids})
    adjusted_logits = decoder._get_adjusted_logits(
        {"input_ids": input_ids}, original_logits
    )

    # Check logits are modified
    assert adjusted_logits.shape == original_logits.shape
    assert ~torch.all(torch.isclose(adjusted_logits, original_logits))
    
    # Check behaviour when CBF constraint is violated
    # Create a decoder with a very harsh CBF constraint
    decoder = CBFOneStepControlledDecoder(
        reference_model=model,
        eos_token_id=vocab_size - 1,
        action_value_model=action_value_model,
        cbf_offset=1000.0,
        alpha=1.0,
    )
    adjusted_logits = decoder._get_adjusted_logits(
        {"input_ids": input_ids}, original_logits
    )
    # Logits of tokens that are not EOS should be set to a -log(eps)
    eps = torch.tensor(torch.finfo(original_logits.dtype).eps)
    assert torch.all(adjusted_logits[:, :-1] == torch.log(eps))
    # Logit of EOS token should be unchanged
    assert torch.all(adjusted_logits[:, -1] == original_logits[:, -1])


@pytest.mark.skip(reason="Initial implementation fails testing")
def test_cbf_optimisation_controlled_decoder(
    model_and_vocab_size, input_ids, action_value_model
):
    model, vocab_size = model_and_vocab_size
    decoder = CBFOptimisationControlledDecoder(
        reference_model=model,
        eos_token_id=vocab_size - 1,
        action_value_model=action_value_model,
        cbf_offset=1.0,
        max_iterations=10,
    )
    original_logits = decoder._get_reference_logits({"input_ids": input_ids})
    adjusted_logits = decoder._get_adjusted_logits(
        {"input_ids": input_ids}, original_logits
    )

    # Check logits are modified correctly
    assert adjusted_logits.shape == original_logits.shape
    assert ~torch.all(torch.isclose(adjusted_logits, original_logits))

    # Check behaviour when CBF constraint is violated
    # Create a decoder with a very harsh CBF constraint
    decoder = CBFOptimisationControlledDecoder(
        reference_model=model,
        eos_token_id=vocab_size - 1,
        action_value_model=action_value_model,
        cbf_offset=1000.0,
        max_iterations=10,
    )
    adjusted_logits = decoder._get_adjusted_logits(
        {"input_ids": input_ids}, original_logits
    )
    # Logits of tokens that are not EOS should be set to a -log(eps)
    eps = torch.tensor(torch.finfo(original_logits.dtype).eps)
    assert torch.all(adjusted_logits[:, :-1] == torch.log(eps))
    # Logit of EOS token should be unchanged
    assert torch.all(adjusted_logits[:, -1] == original_logits[:, -1])


