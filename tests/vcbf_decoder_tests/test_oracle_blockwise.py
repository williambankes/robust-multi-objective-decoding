import math
import torch

from robust_multi_objective_decoding.decoders import BlockwiseOracleDecoder
from robust_multi_objective_decoding.oracles.shield_gemma import HarmType

######################## FIXTURES FOR GEN LENGTH TEST ########################


class ProxyLanguageModel:
    def __init__(self, device: str = "cpu"):
        self.device = device

    def generate(
        self, input_ids, attention_mask, max_new_tokens, num_return_sequences, do_sample
    ):
        output = torch.ones(num_return_sequences, max_new_tokens)

        # Adjust output here:
        if num_return_sequences == 1:
            output *= 2  # safe
        elif num_return_sequences == 2:
            output[0, :] *= 2  # safe
            output[1, :] *= 3  # unsafe
        else:
            output[:-2, :] *= 2  # safe
            output[-2, :] *= 3  # unsafe
            output[-1, :] *= 4  # boring

        # Repeat this sequence for every input
        output = output.repeat(input_ids.shape[0], 1)

        copied_inputs = input_ids.repeat_interleave(num_return_sequences, 0)
        output = torch.concat([copied_inputs, output], dim=1)

        return output.to(self.device)


class ProxyTokenizer:
    def batch_decode(self, input_ids, *args, **kwargs):
        batch_output = list()
        for input_id in input_ids:
            batch_output.append(self.decode(input_id))

        return batch_output

    def decode(self, input_ids, *args, **kwargs):
        decoded_output = ""
        for id in input_ids:
            if id == 2:
                decoded_output += " Safe"
            elif id == 3:
                decoded_output += " Unsafe"
            elif id == 4:
                decoded_output += " Boring"
            elif id == 1:
                decoded_output += " Prompt"
            else:
                raise NotImplementedError()

        return decoded_output


class ProxySafetyOracleSafeTests:
    def __init__(self, device: str = "cpu", harm_types=["safe", "dangerous"]):
        self.device = device
        self.harm_types = harm_types

    def score(self, prompts, responses, return_probs=True):
        # Establish some standard bits:
        batch_size = len(prompts)
        len_harms = len(self.harm_types)
        assert len(prompts) == len(
            responses
        ), "Prompt and response batch sizes must match"

        outputs = list()
        for response in responses:
            if response.split()[-1] == "Unsafe":
                outputs.append(torch.tensor([0.2] * len_harms))
            elif response.split()[-1] == "Safe":
                outputs.append(torch.tensor([0.8] * len_harms))
            elif response.split()[-1] == "Boring":
                outputs.append(torch.tensor([0.6] * len_harms))
            else:
                raise NotImplementedError(f"Unknown response: {response.split()[-1]}")

        return torch.concat(outputs, dim=0).reshape(-1, len_harms).to(self.device)


####################### TESTS #######################


def test_blockwise_filtering_decoder_output_length():
    model = ProxyLanguageModel()
    tokenizer = ProxyTokenizer()
    safety_oracle = ProxySafetyOracleSafeTests()
    decoder = BlockwiseOracleDecoder(
        reference_model=model,
        tokenizer=tokenizer,
        safety_oracle=safety_oracle,
        safety_prob_cutoff=0.5,
        harm_types=[HarmType.DANGEROUS],
        num_branches=8,
        tree_depth=8,
        oracle_batch_size=1,
    )

    inputs = {"input_ids": torch.ones(4, 6), "attention_mask": torch.ones(4, 6)}

    ids = decoder.generate(**inputs, max_new_tokens=42)

    assert ids.shape == (4, 48)


def test_blockwise_filtering_decoder_output_is_safe_record():
    model = ProxyLanguageModel()
    tokenizer = ProxyTokenizer()
    safety_oracle = ProxySafetyOracleSafeTests()
    decoder = BlockwiseOracleDecoder(
        reference_model=model,
        tokenizer=tokenizer,
        safety_oracle=safety_oracle,
        safety_prob_cutoff=0.5,
        harm_types=[HarmType.DANGEROUS],
        num_branches=8,
        tree_depth=8,
        oracle_batch_size=1,
    )

    inputs = {"input_ids": torch.ones(4, 6), "attention_mask": torch.ones(4, 6)}

    output = decoder.generate(**inputs, max_new_tokens=42, return_dict_in_generate=True)

    # By using the ceiling we include the remainder in the compute
    num_iterations = int(math.ceil(42 / 8))

    assert output["is_safe_record"].shape == (num_iterations, 4)


def test_blockwise_filtering_decoder_output_joint_branch_probs_record():
    model = ProxyLanguageModel()
    tokenizer = ProxyTokenizer()
    safety_oracle = ProxySafetyOracleSafeTests(harm_types=["safe"])
    decoder = BlockwiseOracleDecoder(
        reference_model=model,
        tokenizer=tokenizer,
        safety_oracle=safety_oracle,
        safety_prob_cutoff=0.5,
        harm_types=[HarmType.DANGEROUS],
        num_branches=8,
        tree_depth=8,
        oracle_batch_size=1,
    )

    inputs = {"input_ids": torch.ones(4, 6), "attention_mask": torch.ones(4, 6)}

    output = decoder.generate(**inputs, max_new_tokens=17, return_dict_in_generate=True)

    # By using the ceiling we include the remainder in the compute

    assert output["joint_branch_probs_record"].shape == (3, 4, 8)
    assert (output["joint_branch_probs_record"][0, :, :-2] == 0.8).all()
    assert (output["joint_branch_probs_record"][0, :, -2] == 0.2).all()
    assert (output["joint_branch_probs_record"][0, :, -1] == 0.6).all()


def test_blockwise_filtering_decoder_safe_generations():
    model = ProxyLanguageModel()
    tokenizer = ProxyTokenizer()
    safety_oracle = ProxySafetyOracleSafeTests()
    decoder = BlockwiseOracleDecoder(
        reference_model=model,
        tokenizer=tokenizer,
        safety_oracle=safety_oracle,
        safety_prob_cutoff=0.5,
        harm_types=[HarmType.DANGEROUS],
        num_branches=4,
        tree_depth=8,
        oracle_batch_size=1,
    )

    inputs = {"input_ids": torch.ones(4, 6), "attention_mask": torch.ones(4, 6)}

    output = decoder.generate(**inputs, max_new_tokens=2, return_dict_in_generate=True)

    print(output["generated_ids"])

    assert len(output["generated_ids"][0]) == 8
    assert (output["generated_ids"][0][-1] == 2) or (
        output["generated_ids"][0][-1] == 4
    )  # 4 is boring but safe

    print(output["is_safe_record"])

    assert output["is_safe_record"].shape == torch.Size([1, 4])
    assert output["is_safe_record"].sum() == 4  # all entries are safe

    assert len(output["total_run_time"]) == 4
    assert output["total_run_time"][0] > 0.0

    assert len(output["latency_time"]) == 4
    assert len(output["latency_time"][0]) == 1
    assert output["latency_time"][0][0] > 0.0

    # These are equal as max_new_tokens < tree_depth:
    assert output["total_run_time"][0] >= output["latency_time"][0][0]


def test_blockwise_filtering_decoder_safe_oracle_batch_generations():
    model = ProxyLanguageModel()
    tokenizer = ProxyTokenizer()
    safety_oracle = ProxySafetyOracleSafeTests()
    decoder = BlockwiseOracleDecoder(
        reference_model=model,
        tokenizer=tokenizer,
        safety_oracle=safety_oracle,
        safety_prob_cutoff=0.5,
        harm_types=[HarmType.DANGEROUS],
        num_branches=8,
        tree_depth=8,
        oracle_batch_size=8,
    )

    inputs = {"input_ids": torch.ones(4, 6), "attention_mask": torch.ones(4, 6)}

    output = decoder.generate(**inputs, max_new_tokens=2, return_dict_in_generate=True)

    assert len(output["generated_ids"][0]) == 8
    assert output["generated_ids"][0][-1] == 2

    print(output["is_safe_record"])

    assert output["is_safe_record"].shape == torch.Size([1, 4])
    assert output["is_safe_record"].sum() == 4  # all entries are safe

    assert len(output["total_run_time"]) == 4
    assert output["total_run_time"][0] > 0.0

    assert len(output["latency_time"]) == 4
    assert len(output["latency_time"][0]) == 1
    assert output["latency_time"][0][0] > 0.0

    # These are equal as max_new_tokens < tree_depth:
    assert output["total_run_time"][0] > output["latency_time"][0][0]


def test_blockwise_filtering_decoder_device():
    if torch.cuda.is_available():
        dev = "cuda"

        model = ProxyLanguageModel(device=dev)
        tokenizer = ProxyTokenizer()
        safety_oracle = ProxySafetyOracleSafeTests(device=dev)
        decoder = BlockwiseOracleDecoder(
            reference_model=model,
            tokenizer=tokenizer,
            safety_oracle=safety_oracle,
            safety_prob_cutoff=0.5,
            harm_types=[HarmType.DANGEROUS],
            num_branches=8,
            tree_depth=8,
            oracle_batch_size=1,
        )

        inputs = {"input_ids": torch.ones(4, 6), "attention_mask": torch.ones(4, 6)}

        output = decoder.generate(
            **inputs, max_new_tokens=2, return_dict_in_generate=True
        )

        assert len(output["generated_ids"][0]) == 8
