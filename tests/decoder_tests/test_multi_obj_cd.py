import torch

from transformers import AutoTokenizer
from robust_multi_objective_decoding.decoders import MultiObjectiveControlledDecoder


######################## FIXTURES FOR GEN LENGTH TEST ########################


class ProxyLanguageModel:
    def generate(
        self,
        input_ids,
        attention_mask,
        max_new_tokens,
        num_return_sequences,
        do_sample,
        **kwargs,
    ):
        output = torch.ones(input_ids.shape[0], input_ids.shape[1] + max_new_tokens)
        return output.repeat_interleave(num_return_sequences, dim=0).to(torch.long)


######################### FIXTURE FOR SAFE GENERATION TEST #####################


class ProxyValueFunctionModuleSafeTests:
    device = "cpu"

    def __call__(self, input_ids, attention_mask, *args, **kwargs):
        # Bad values are a 3:
        bad_filters = input_ids == 3
        sub_filters = input_ids == 1

        # Return values which are good 1s and bad 0s:
        output = torch.ones(input_ids.shape[0], input_ids.shape[1])
        output[bad_filters] = 0
        output[sub_filters] = 0.1

        output = output.unsqueeze(1).repeat_interleave(3, dim=1)

        return output, output


class ProxyLanguageModelSafeTests:
    def generate(
        self, input_ids, attention_mask, max_new_tokens, num_return_sequences, do_sample
    ):
        # Return mix of good values != 3, and bad values == 3:
        output = torch.ones(3, input_ids.shape[1] + max_new_tokens)
        output[0, :] *= 2
        output[1, :] *= 3
        output[2, :] *= 1

        return output.repeat(input_ids.shape[0], 1)


class ProxyTokenizer:
    name_or_path = "gpt2"
    pad_token_id = 0
    eos_token_id = 0


####################### TESTS #######################


def test_blockwise_filtering_decoder_output_length():
    tok = AutoTokenizer.from_pretrained("openai-community/gpt2")
    tok.pad_token_id = tok.eos_token_id
    model = ProxyLanguageModel()
    value_func = ProxyValueFunctionModuleSafeTests()
    decoder = MultiObjectiveControlledDecoder(
        reference_model=model,
        value_function=value_func,
        weights=[0.4, 0.3, 0.3],
        tokenizer=tok,
        ref_tokenizer=tok,
    )

    inputs = {
        "input_ids": torch.ones(4, 6).to(torch.long),
        "attention_mask": torch.ones(4, 6).to(torch.long),
    }

    ids = decoder.generate(**inputs, max_new_tokens=42)

    assert ids.shape == (4, 48)
