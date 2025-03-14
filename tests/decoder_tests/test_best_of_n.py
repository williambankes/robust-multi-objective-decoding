import torch
from robust_multi_objective_decoding.decoders.best_of_n_oracle_decoder import (
    BestOfNOracleDecoder,
)

####################### FIXTURES FOR BEST OF N TESTS #######################


class ProxySafetyOracle:
    harm_types = ["safe", "dangerous"]

    def score(self, prompts, responses, *args, **kwargs):
        safety_scores = list()
        num_types = 1

        for resp in responses:
            final_word = resp.split(" ")[-1]

            if final_word == "Safe":
                safety_scores.append(torch.tensor([1.0] * num_types))
            elif final_word == "Unsafe":
                safety_scores.append(torch.tensor([0.0] * num_types))
            elif final_word == "Prompt":
                raise NotImplementedError()
            else:
                raise NotImplementedError()

        return torch.stack(safety_scores)


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
            elif id == 1:
                decoded_output += " Prompt"
            else:
                raise NotImplementedError()

        return decoded_output


class ProxyLanguageModelSafeBestofNTests:
    def generate(
        self, input_ids, attention_mask, max_new_tokens, num_return_sequences, do_sample
    ):
        # Return mix of good values != 3, and bad values == 3:
        output = torch.ones(num_return_sequences, input_ids.shape[1] + max_new_tokens)

        # Generate a random id:
        idx = torch.randint(0, num_return_sequences, (1,)).item()

        # Select a random branch to be safe:
        output[:, input_ids.shape[1] :] = 3
        output[idx, input_ids.shape[1] :] = 2

        return output.repeat(input_ids.shape[0], 1)


class ProxyLanguageModelUnSafeBestofNTests:
    def generate(
        self, input_ids, attention_mask, max_new_tokens, num_return_sequences, do_sample
    ):
        # Return mix of good values != 3, and bad values == 3:
        output = torch.ones(num_return_sequences, input_ids.shape[1] + max_new_tokens)

        output[:, input_ids.shape[1] :] = 3
        # output[idx, input_ids.shape[1]:] = 2

        return output.repeat(input_ids.shape[0], 1)


####################### TESTS #######################


def test_best_of_n_safety_oracle_safe_generations():
    """
    Here we test the best of n safety oracle decoder.

    ProxyLanguageModelSafeTests is a model that generates [num_branches * batch_sizes, max_new_tokens]
    sequences of tokens, where the final sequence in a batch is either 2 (Safe) or 3 (Unsafe).

    ProxyTokenizer is a tokenizer that can decode the tokenized sequences into strings. If the token
    is 2, it appends ' Safe' to the string, if it is 3, it appends ' Unsafe'. The prompt token is 1.

    ProxySafetyOracle is a safety oracle that returns a tensor of shape [input size, num_harm_types]
    it returns a tensor of 1s if the final token in the response is 'Safe', and 0s if it is 'Unsafe'.

    Here we test that the setup always selects the safe sequences when the safety oracle is used.
    """

    model = ProxyLanguageModelSafeBestofNTests()
    tok = ProxyTokenizer()

    safety_oracle = ProxySafetyOracle()
    decoder = BestOfNOracleDecoder(
        reference_model=model,
        oracle=safety_oracle,
        tokenizer=tok,
        num_branches=8,
        safety_prob_cutoff=0.5,
    )

    inputs = {"input_ids": torch.ones(4, 6), "attention_mask": torch.ones(4, 6)}

    output = decoder.generate(**inputs, max_new_tokens=4, return_dict_in_generate=True)

    # new
    assert (output["generated_ids"][:, -1] == 2).all()
    assert output["generated_ids"].shape == (4, 10)

    assert output["oracle_values"].shape == (4, 8, 1)

    # only one generation is safe in each branch thus sum of the oracle values should be 4 i.e. one per branch
    assert output["oracle_values"].sum() == 4


def test_best_of_n_safety_oracle_batch_size_safe_generations():
    """
    Here we test the best of n safety oracle decoder. Batch Size.
    """

    model = ProxyLanguageModelSafeBestofNTests()
    tok = ProxyTokenizer()

    safety_oracle = ProxySafetyOracle()
    decoder = BestOfNOracleDecoder(
        reference_model=model,
        oracle=safety_oracle,
        tokenizer=tok,
        num_branches=8,
        oracle_batch_size=4,
        safety_prob_cutoff=0.5,
    )

    inputs = {"input_ids": torch.ones(4, 6), "attention_mask": torch.ones(4, 6)}

    output = decoder.generate(**inputs, max_new_tokens=4, return_dict_in_generate=True)

    # new
    print("output", output)

    assert (output["generated_ids"][:, -1] == 2).all()
    assert output["generated_ids"].shape == (4, 10)

    assert output["oracle_values"].shape == (4, 8, 1)
    # only one generation is safe in each branch
    assert output["oracle_values"].sum() == 4


def test_best_of_n_safety_oracle_unsafe_generations():
    """
    Here we test the best of n safety oracle decoder, when only unsafe generations are passed.

    Here we test that the setup always selects the safe sequences when the safety oracle is used.
    """

    model = ProxyLanguageModelUnSafeBestofNTests()
    tok = ProxyTokenizer()

    safety_oracle = ProxySafetyOracle()
    decoder = BestOfNOracleDecoder(
        reference_model=model,
        oracle=safety_oracle,
        tokenizer=tok,
        num_branches=8,
        safety_prob_cutoff=0.5,
    )

    inputs = {"input_ids": torch.ones(4, 6), "attention_mask": torch.ones(4, 6)}

    output = decoder.generate(**inputs, max_new_tokens=4, return_dict_in_generate=True)

    assert (output["generated_ids"][:, -1] == 3).all()
    assert output["generated_ids"].shape == (4, 10)

    # We return the safety label for all branches:
    assert output["oracle_values"].shape == (4, 8, 1)


def test_best_of_n_safety_oracle_timing_eval():
    """
    Here we test the best of n safety oracle decoder, when only unsafe generations are passed.

    Here we test that the setup always selects the safe sequences when the safety oracle is used.
    """

    model = ProxyLanguageModelUnSafeBestofNTests()
    tok = ProxyTokenizer()

    safety_oracle = ProxySafetyOracle()
    decoder = BestOfNOracleDecoder(
        reference_model=model,
        oracle=safety_oracle,
        tokenizer=tok,
        num_branches=8,
        safety_prob_cutoff=0.5,
    )

    inputs = {"input_ids": torch.ones(4, 6), "attention_mask": torch.ones(4, 6)}

    output = decoder.generate(**inputs, max_new_tokens=4, return_dict_in_generate=True)

    assert output["total_run_time"][0] > 0.0
    assert output["total_run_time"][0] == output["latency_time"][0][-1]
    assert len(output["latency_time"]) == 4
    assert len(output["latency_time"][0]) == 1

    decoder = BestOfNOracleDecoder(
        reference_model=model,
        oracle=safety_oracle,
        tokenizer=tok,
        num_branches=8,
        safety_prob_cutoff=0.5,
        oracle_batch_size=2,
    )

    inputs = {"input_ids": torch.ones(4, 6), "attention_mask": torch.ones(4, 6)}

    output = decoder.generate(**inputs, max_new_tokens=4, return_dict_in_generate=True)

    assert output["total_run_time"][0] > 0.0
    assert len(output["latency_time"]) == 4
    assert len(output["total_run_time"]) == 4
    assert len(output["latency_time"][0]) == (4 * 8) // 2

    for latency_time in output["latency_time"]:
        assert latency_time[0] <= output["total_run_time"][0]
