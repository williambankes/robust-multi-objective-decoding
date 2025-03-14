import math
import torch

from robust_multi_objective_decoding.decoders import BlockwiseFilteringDecoder

######################## FIXTURES FOR GEN LENGTH TEST ########################


class ProxyValueFunctionModuleBase:
    def __call__(self, input_ids, attention_mask, *args, **kwargs):
        output = torch.ones(input_ids.shape[0], input_ids.shape[1])
        return output, output


class ProxyLanguageModel:
    def generate(
        self, input_ids, attention_mask, max_new_tokens, num_return_sequences, do_sample
    ):
        output = torch.ones(input_ids.shape[0], input_ids.shape[1] + max_new_tokens)
        return output.repeat_interleave(num_return_sequences, dim=0)


######################### FIXTURE FOR SAFE GENERATION TEST #####################


class ProxyValueFunctionModuleSafeTests:
    def __call__(self, input_ids, attention_mask, *args, **kwargs):
        # Bad values are a 3:
        bad_filters = input_ids == 3
        sub_filters = input_ids == 1

        # Return values which are good 1s and bad 0s:
        output = torch.ones(input_ids.shape[0], input_ids.shape[1])
        output[bad_filters] = 0
        output[sub_filters] = 0.1

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


######################### FIXTURE FOR MAX CBF VALUE TEST #####################


class ProxyValueFunctionModuleMaxCBFTests:
    i = 0

    def __call__(self, input_ids, attention_mask, *args, **kwargs):
        # Bad values are a 3:
        sub_filters = input_ids == 1
        bad_filters = input_ids == 3

        if self.i == 0:
            # Return good init values:
            output = torch.ones(input_ids.shape[0], input_ids.shape[1])
            self.i += 1
        else:
            # Return values which are good 1s and bad 0s:
            output = torch.ones(input_ids.shape[0], input_ids.shape[1]) * 0.7
            output[bad_filters] = -1.0
            output[sub_filters] = 0.2

        return output, output


class ProxyValueFunctionModuleMaxUnsafeCBFTests:
    i = 0

    def __call__(self, input_ids, attention_mask, *args, **kwargs):
        # Bad values are a 3:
        bad_filters = input_ids == 3
        sub_filters = input_ids == 1

        if self.i == 0:
            # Return good init values:
            output = torch.ones(input_ids.shape[0], input_ids.shape[1]) * -1.0
            self.i += 1
        else:
            # Return values which are good 1s and bad 0s:
            output = torch.ones(input_ids.shape[0], input_ids.shape[1]) * -0.1
            output[bad_filters] = -1.5
            output[sub_filters] = -0.7

        return output, output


class ProxyValueFunctionModuleCBFClassifierTests:
    i = 0

    def __call__(self, input_ids, attention_mask, *args, **kwargs):
        # Bad values are a 3:
        bad_filters = input_ids == 3
        sub_filters = input_ids == 1

        if self.i == 0:
            # Return good init values:
            output = torch.ones(input_ids.shape[0], input_ids.shape[1]) * -1.0
            self.i += 1
        else:
            # Return values which are good 1s and bad 0s:
            output = torch.ones(input_ids.shape[0], input_ids.shape[1])
            output[bad_filters] = -1.5
            output[sub_filters] = -0.7

        return output, output


####################### TESTS #######################


def test_blockwise_filtering_decoder_output_length():
    model = ProxyLanguageModel()
    value_func = ProxyValueFunctionModuleBase()
    decoder = BlockwiseFilteringDecoder(
        reference_model=model,
        control_barrier_function=value_func,
        cbf_cutoff=0.5,
        alpha=0.5,
    )

    inputs = {"input_ids": torch.ones(4, 6), "attention_mask": torch.ones(4, 6)}

    ids = decoder.generate(**inputs, max_new_tokens=42)

    assert ids.shape == (4, 48)


def test_blockwise_filtering_decoder_safe_generations():
    model = ProxyLanguageModelSafeTests()
    value_func = ProxyValueFunctionModuleSafeTests()
    decoder = BlockwiseFilteringDecoder(
        reference_model=model,
        control_barrier_function=value_func,
        cbf_cutoff=0.5,
        alpha=0.5,
        num_branches=3,
        tree_depth=2,
    )

    inputs = {"input_ids": torch.ones(4, 6), "attention_mask": torch.ones(4, 6)}

    ids = decoder.generate(**inputs, max_new_tokens=2)

    assert ids[0, -1] == 2
    assert ids[0, -1] == 2
    assert ids.shape == (4, 8)


def test_blockwise_filtering_decoder_max_value_generations():
    model = ProxyLanguageModelSafeTests()
    value_func = ProxyValueFunctionModuleMaxCBFTests()
    decoder = BlockwiseFilteringDecoder(
        reference_model=model,
        control_barrier_function=value_func,
        cbf_cutoff=0.5,
        alpha=0.5,
        num_branches=3,
        tree_depth=2,
    )

    inputs = {"input_ids": torch.ones(4, 6), "attention_mask": torch.ones(4, 6)}

    ids = decoder.generate(**inputs, max_new_tokens=2)

    assert ids[0, -1] == 2
    assert ids[0, -1] == 2
    assert ids.shape == (4, 8)

    # When the CBF condition is met but values are negative the max value should be selected:
    value_func = ProxyValueFunctionModuleMaxUnsafeCBFTests()
    decoder = BlockwiseFilteringDecoder(
        reference_model=model,
        control_barrier_function=value_func,
        cbf_cutoff=0.5,
        alpha=0.5,
        num_branches=3,
        tree_depth=2,
    )

    inputs = {"input_ids": torch.ones(4, 6), "attention_mask": torch.ones(4, 6)}

    ids = decoder.generate(**inputs, max_new_tokens=2)

    assert ids[0, -1] == 2
    assert ids[0, -1] == 2
    assert ids.shape == (4, 8)


def test_blockwise_filtering_decoder_timing_metrics():
    model = ProxyLanguageModelSafeTests()
    value_func = ProxyValueFunctionModuleMaxCBFTests()
    decoder = BlockwiseFilteringDecoder(
        reference_model=model,
        control_barrier_function=value_func,
        cbf_cutoff=0.5,
        alpha=0.5,
        num_branches=3,
        tree_depth=2,
    )

    inputs = {"input_ids": torch.ones(4, 6), "attention_mask": torch.ones(4, 6)}

    output = decoder.generate(**inputs, max_new_tokens=4, return_dict_in_generate=True)

    assert output["total_run_time"][0] > 0.0
    assert output["latency_time"][0][-1] > 0.0
    assert output["total_run_time"][0] >= output["latency_time"][0][-1]
    assert len(output["latency_time"]) == 4
    assert len(output["latency_time"][0]) == int(math.ceil(4 / 2))


def test_blockwise_filtering_decoder_safe_cbf_classifier_generations():
    model = ProxyLanguageModelSafeTests()
    value_func = ProxyValueFunctionModuleCBFClassifierTests()
    decoder = BlockwiseFilteringDecoder(
        reference_model=model,
        control_barrier_function=value_func,
        cbf_cutoff=0.5,
        alpha=0.5,
        num_branches=3,
        tree_depth=2,
    )

    inputs = {"input_ids": torch.ones(4, 6), "attention_mask": torch.ones(4, 6)}

    out = decoder.generate(**inputs, max_new_tokens=6, return_dict_in_generate=True)

    assert out["generated_ids"][0, -1] == 2
    assert out["generated_ids"][0, -1] == 2
    assert out["generated_ids"].shape == (4, 12)
    assert (out["is_cbf_condition_met"][-1, :] == 1).all()
