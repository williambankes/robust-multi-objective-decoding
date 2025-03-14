import time
import torch
from typing import List
import torch.nn as nn
from robust_multi_objective_decoding.oracles.oracle import Oracle
from robust_multi_objective_decoding.oracles.shield_gemma import ShieldGemmaSafetyOracle
from transformers import PreTrainedTokenizer


class BestOfNOracleDecoder(nn.Module):
    def __init__(
        self,
        reference_model: nn.Module,
        oracle: Oracle,
        tokenizer: PreTrainedTokenizer,
        num_branches: int = 8,
        oracle_batch_size: int = 1,
        safety_prob_cutoff: float = 0.5,
        *args,
        **kwargs,
    ):
        """
        Best Of N Safety Oracle Decoder runs best of N decoding using
        the safety oracle to select optimal branches. Each iteration
        num_branches full generations are made and the safety oracle
        evaluates all of them. A safe branch is then selected.

        Parameters
        ----------
        reference_model : nn.Module
            _description_
        oracle : ShieldGemmaSafetyOracle
            A safety oracle with a .score method that takes a prompt and
            returns len(harm_types) safety probabilities
        tokenizer : PreTrainedTokenizer
            A tokenizer for the reference_model
        num_branches : int, optional
            Number of generations to be generated for each prompt, by default 8
        harm_types : List[str], optional
            A list of harm types to be evaluated by the model,
            by default [HarmType.DANGEROUS]*args
        safety_prob_cutoff: float, optional
            The cutoff value for the safety probability, by default 0.5
        oracle_batch_size: int, optional
            The batch size for processing the safety label through the safety oracle
            when multiple branches are generated, by default 1
        """

        super().__init__()

        self.ref_model = reference_model
        self.oracle = oracle
        self.num_branches = num_branches
        self.tokenizer = tokenizer
        self.oracle_batch_size = oracle_batch_size
        self.safety_prob_cutoff = safety_prob_cutoff

        assert (
            self.oracle_batch_size > 0
        ), f"Oracle batch size: {oracle_batch_size} must be greater than 0"

    def get_hyperparameters(self):
        """
        Get hyperparameters specific to the decoder

        Returns
        -------
        dict
            dictionary with keys of the hyperparameter name and values attached
        """

        return {"num_branches": self.num_branches}

    def _process_safety_label_cutoff(self, oracle_values: torch.Tensor, cutoff: float):
        """
        Process safety label probilitites via a cutoff

        Parameters
        ----------
        safety_labels : torch.Tensor
            _description_
        cutoff : float
            _description_

        Returns
        -------
        _type_
            _description_
        """

        # Apply the cutoff filter:
        cutoff_safety_labels = (oracle_values > cutoff).sum(dim=1)
        return (cutoff_safety_labels == oracle_values.shape[1]).to(torch.int64)

    def _select_branches(self, oracle_values, batch_size):
        """
        Given a batch of safety labels size [batch_size * num_branches, len(harm_types)]
        select the best branch for each batch and return the index.

        Parameters
        ----------
        safety_labels : torch.Tensor
            A tensor of shape [batch_size * num_branches, len(harm_types)], where each
            row is a set of safety probabilities for a specific branch
        batch_size : _type_
            The number of prompts in the batched input

        Returns
        -------
        torch.Tensor
            A tensor of shape [batch_size], where each element is the index of the selected branch
        torch.Tensor
            A tensor of shape [batch_size], where each element is 1 if the selected branch is safe, 0 otherwise
        """

        # TODO: Update this to handle any oracle
        # Process the safety labels:

        if isinstance(self.oracle, ShieldGemmaSafetyOracle):
            oracle_values = self._process_safety_label_cutoff(
                oracle_values, self.safety_prob_cutoff
            )

        # Outputs:
        output_idxs = list()
        # For each batch select a safe generation:
        for batch_idx in range(batch_size):
            lower_idx = batch_idx * self.num_branches
            upper_idx = (batch_idx + 1) * self.num_branches

            # Return the branch with the highest oracle score:
            idx = torch.argmax(oracle_values[lower_idx:upper_idx]).view(1)
            output_idxs.append(idx + lower_idx)

        return torch.concat(output_idxs)

    def process_latency_timing(
        self,
        generate_time: float,
        pre_oracle_eval_time: float,
        start_time: float,
        end_time: float,
        oracle_eval_times: List[float],
    ) -> List[float]:
        if len(oracle_eval_times) == 1:
            output = [end_time - start_time]

        else:
            output = [generate_time + (oracle_eval_times[0] - pre_oracle_eval_time)]
            branch_select_time = end_time - oracle_eval_times[-1]

            for i in range(1, len(oracle_eval_times)):
                oracle_eval_time = oracle_eval_times[i] - oracle_eval_times[i - 1]
                output.append(generate_time + oracle_eval_time + branch_select_time)

        return output

    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        max_new_tokens: int,
        return_dict_in_generate: bool = False,
        *args,
        **kwargs,
    ):
        """
        Generate a batch of sequences using the best of N safety oracle decoder.

        Parameters
        ----------
        input_ids : torch.Tensor
            tokenized input ids
        attention_mask : torch.Tensor
            attention mask for the input ids of padding tokens
        max_new_tokens : int
            Maximum number of tokens to generate
        return_dict_in_generate : bool, optional
            Return analysis metrics in a dictionary format or just the generated ids, by default False

        Returns
        -------
        torch.Tensor | dict
            Depending upon return_dict_in_generate, either a tensor of generated ids or a dictionary of
            analysis metrics is generated.
        """

        batch_size = input_ids.shape[0]
        start_time = time.time()

        # Use the base model to generate from the tree:
        branches = self.ref_model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            num_return_sequences=self.num_branches,
            do_sample=True,
        )

        # TODO: Separate prompts and response before decoding using tensor operations on the input_ids matrix
        # This requires knowing which side the tokenization occurs on

        # Decode and split prompts and response:
        decoded_prompts = self.tokenizer.batch_decode(
            input_ids.repeat_interleave(self.num_branches, dim=0),
            skip_special_tokens=True,
        )
        decoded_branches = self.tokenizer.batch_decode(
            branches, skip_special_tokens=True
        )

        # Remove the prompt from the branch generations:
        responses = list()
        for i, branch in enumerate(decoded_branches):
            responses.append(branch[len(decoded_prompts[i]) :])

        # Calculate the time for generating the prompts and responses:
        pre_oracle_eval_time = time.time()
        generate_time = pre_oracle_eval_time - start_time

        # Evaluate the prompts and responses oracle_values:
        oracle_eval_time = list()
        if self.oracle_batch_size > 1:
            oracle_values = list()
            num_iterations = (batch_size * self.num_branches) // self.oracle_batch_size

            # Ensure the oracle batch size is a multiple of the batch size:
            assert (
                batch_size * self.num_branches % self.oracle_batch_size == 0
            ), f"Oracle batch size: {self.oracle_batch_size} must be a multiple of batch_size * num_branches: {batch_size * self.num_branches}"
            assert (
                self.oracle_batch_size <= batch_size * self.num_branches
            ), f"Oracle batch size: {self.oracle_batch_size} must be less than or equal to batch_size * num_branches: {batch_size * self.num_branches}"

            for i in range(num_iterations):
                lower_idx = i * self.oracle_batch_size
                upper_idx = (i + 1) * self.oracle_batch_size
                oracle_values.append(
                    self.oracle.score(
                        decoded_prompts[lower_idx:upper_idx],
                        responses[lower_idx:upper_idx],
                    )
                )

                # Record the time for each oracle evaluation:
                oracle_eval_time.append(time.time())

            oracle_values = torch.concat(oracle_values, dim=0)
            assert (
                oracle_values.shape[0] == batch_size * self.num_branches
            ), f"Safety labels shape: {oracle_values.shape[0]}, should match batch_size * self.num_branches: {batch_size * self.num_branches}"

        else:
            oracle_values = self.oracle.score(decoded_prompts, responses)

            # Record time for oracle evaluation:
            oracle_eval_time.append(time.time())

        # For each input select the best safety labels.
        output_idxs = self._select_branches(oracle_values, batch_size)

        # Select the best branches:
        output_ids = branches[output_idxs]
        output_attention_mask_matrix = torch.ones(
            batch_size, max_new_tokens, dtype=torch.long
        )
        output_attention_mask = torch.concat(
            [attention_mask, output_attention_mask_matrix.to(attention_mask.device)],
            dim=1,
        )

        # Ensure output dimensions are correct:
        assert (
            output_attention_mask.shape[0] == batch_size
        ), f"output_attention_mask shape 0: {output_attention_mask.shape[0]}, should be {batch_size}"
        assert (
            output_attention_mask.shape[1] == (max_new_tokens + input_ids.shape[1])
        ), f"output_attention_mask shape 1: {output_attention_mask.shape[1]}, should be {(max_new_tokens + input_ids.shape[1])}"

        # Calculate the total run time and latency time:
        end_time = time.time()
        latency_time = self.process_latency_timing(
            generate_time=generate_time,
            pre_oracle_eval_time=pre_oracle_eval_time,
            start_time=start_time,
            end_time=end_time,
            oracle_eval_times=oracle_eval_time,
        )

        # Adjust the safety_labels shape for logging:
        oracle_values = oracle_values.reshape(batch_size, self.num_branches, -1)

        if return_dict_in_generate:
            return {
                "generated_ids": output_ids,
                "generated_ids_attn_mask": attention_mask,
                "oracle_values": oracle_values,
                "total_run_time": [end_time - start_time]
                * batch_size,  # all outputs must be a list of len batch_size
                "latency_time": [latency_time] * batch_size,
            }
        else:
            return output_ids
