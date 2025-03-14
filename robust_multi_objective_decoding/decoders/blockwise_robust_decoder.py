import math
import time
import torch
from copy import deepcopy
import torch.nn as nn
from transformers import PreTrainedTokenizer, AutoModelForCausalLM
from robust_multi_objective_decoding.multi_objective_value_function import (
    BaseMultiObjectiveValueFunction,
)


class BlockwiseRobustDecoder(nn.Module):
    # TODO: Implement a device test
    # TODO: Check with branch attention mask

    def __init__(
        self,
        reference_model: AutoModelForCausalLM,
        value_function: BaseMultiObjectiveValueFunction,
        tokenizer: PreTrainedTokenizer,
        ref_tokenizer: PreTrainedTokenizer,
        num_branches: int = 8,
        tree_depth: int = 8,
        top_k: int = 0,
        top_p: float = 1.0,
        lambda_coef: float = 1.0,
        max_grad: float = 50000.0,
        group_weights_iter: int = 10,
        minibatch_size: int = 16,
        use_responsewise_worst_case: bool = False,
        use_chat_format: bool = False,
        vf_use_chat_template: bool = False,
        max_length: int = 1024,
        weight_step_size: float = 1.0,
        oracle=None,
        *args,
        **kwargs,
    ):
        super().__init__()

        self.tokenizer = tokenizer
        self.ref_tokenizer = ref_tokenizer
        self.ref_model = reference_model
        self.value_function = value_function
        self.num_branches = num_branches
        self.tree_depth = tree_depth
        self.top_k = top_k
        self.top_p = top_p
        self.lambda_coef = lambda_coef
        self.max_grad = max_grad
        self.group_weights_iter = group_weights_iter
        self.use_responsewise_worst_case = use_responsewise_worst_case
        self.use_chat_format = use_chat_format
        self.vf_use_chat_template = vf_use_chat_template
        self.minibatch_size = minibatch_size
        self.tokenizer_diff = True
        self.max_length = max_length
        self.weight_step_size = weight_step_size
        self.oracle = oracle

        # If the tokenizers are the same we don't need to do the re-tokenization step:
        if self.ref_tokenizer.name_or_path == self.tokenizer.name_or_path:
            self.tokenizer_diff = False

        self.tokens_finished = [self.ref_tokenizer.eos_token_id]
        if "gemma-2-2b-it" in self.ref_tokenizer.name_or_path:
            print("using gemma-2-2b-it: <end_of_turn> in the chat format")
            self.tokens_finished.append(self.ref_tokenizer.encode("<end_of_turn>")[-1])

    def set_num_branches(self, num_branches: int):
        self.num_branches = num_branches

    def set_tree_depth(self, tree_depth: int):
        self.tree_depth = tree_depth

    def set_top_k(self, top_k: int):
        self.top_k = top_k

    def set_lambda_coef(self, lambda_coef: float):
        self.lambda_coef = lambda_coef

    def set_max_grad(self, max_grad: float):
        self.max_grad = max_grad

    def set_group_weights_iter(self, group_weights_iter: int):
        self.group_weights_iter = group_weights_iter

    def _prepare_input_ids_for_vf(
        self,
        branches: list,
        device: str,
    ):
        """
        Prepare input ids for value function evaluation

        Parameters
        ----------
        branches: list
            list of outputs generated from vllm
        device: str
            device of the tensors

        Returns
        -------
        torch.Tensor, torch.Tensor
            input ids and attention mask for value function evaluation
        """

        prompt_token_ids = [branches[i].prompt_token_ids for i in range(len(branches))]
        output_token_ids = [
            branches[i].outputs[0].token_ids for i in range(len(branches))
        ]

        prompt_texts = self.ref_tokenizer.batch_decode(
            prompt_token_ids,
            skip_special_tokens=False,
            # skip_special_tokens=True
        )

        # Decode the prompt token ids, remove chat template-specific parts
        # prompt_texts = [
        #     item.replace("user\n", "").replace("model\n", "") # TODO: Might be gemma-2-2b-it specific
        #     for item in prompt_texts
        # ]

        output_texts = self.ref_tokenizer.batch_decode(
            output_token_ids, skip_special_tokens=True
        )

        # TODO: will need a separate tokenizer for value function if the architecture is different
        if self.use_chat_format:
            chat_format = [
                [
                    {"role": "user", "content": prompt_texts[i]},
                    {"role": "assistant", "content": output_texts[i]},
                ]
                for i in range(len(prompt_texts))
            ]

            input_ids = self.ref_tokenizer.apply_chat_template(
                chat_format,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
                return_assistant_tokens_mask=False,
                padding=True,
            ).to(device)
        else:
            texts_all = [
                prompt_texts[i].replace("<bos>", "") + output_texts[i]
                for i in range(len(prompt_texts))
            ]

            input_ids = self.ref_tokenizer(
                texts_all,
                padding=True,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_length,
            ).to(device)

        return input_ids

    def _eval_value_function(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ):
        """
        Evaluate the value function across the entire sentence generation

        Parameters
        ----------
        input_ids : torch.Tensor
            _description_
        attention_mask : torch.Tensor
            _description_

        Returns
        -------
        torch.Tensor, torch.Tensor
            Multiple output heads from the control barrier function
        """

        # Decoder the responses from the reference model:
        decoded_inputs = self.ref_tokenizer.batch_decode(
            input_ids, skip_special_tokens=True
        )
        decoded_inputs = [item.replace("<pad>", "") for item in decoded_inputs]
        decoded_inputs = [
            item.replace("<eos>", "") + "<eos>" if "<eos>" in item else item
            for item in decoded_inputs
        ]

        if self.value_function is not None:
            # Re-tokenizer the responses using the value function tokenizer
            inputs = self.tokenizer(
                decoded_inputs,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length,
            )

            # Map new tokens to the correct device
            inputs = {k: v.to(input_ids.device) for k, v in inputs.items()}

            with torch.no_grad():
                values, q_values = self.value_function(**inputs)

        elif self.value_function is None and self.oracle is not None:
            prompt_batch = list()
            response_batch = list()
            for item in decoded_inputs:
                idx_response = item.index("model\n") + len("model\n")
                prompt_batch.append(item[:idx_response])
                response_batch.append(item[idx_response:])
            with torch.no_grad():
                values = self.oracle.score(prompt_batch, response_batch).to(
                    torch.bfloat16
                )
            q_values = torch.zeros_like(values).to(torch.bfloat16)
        else:
            raise NotImplementedError
        return values, q_values

    def get_hyperparameters(self):
        """
        Get hyperparameters specific to the decoder

        Returns
        -------
        dict
            dictionary with keys of the hyperparameter name and values attached
        """

        return {
            "num_branches": self.num_branches,
            "tree_depth": self.tree_depth,
            "top_k": self.top_k,
            "top_p": self.top_p,
            "lambda_coef": self.lambda_coef,
            "max_grad": self.max_grad,
            "group_weights_iter": self.group_weights_iter,
            "minibatch_size": self.minibatch_size,
            "use_chat_format": self.use_chat_format,
            "use_responsewise_worst_case": self.use_responsewise_worst_case,
            "vf_use_chat_template": self.vf_use_chat_template,
            "max_length": self.max_length,
            "weight_step_size": self.weight_step_size,
        }

    def process_latency_time(
        self, start_time, pre_generation, generation_times, end_time
    ):
        # Calculate the latency time for each iteration of the generation process:
        # setup_time = pre_generation - start_time
        init_iter_time = generation_times[0] - pre_generation
        # shut_down_time = end_time - generation_times[-1]

        iter_times = [init_iter_time]
        for i in range(1, len(generation_times)):
            latency_time = generation_times[i] - generation_times[i - 1]

            iter_times.append(latency_time)

        return iter_times

    def _compute_weights(
        self,
        adv: torch.Tensor,
        lambda_coef: float,
        max_grad: float,
        group_weights_iter: int = 10,
    ):
        """
        Compute the weights for the branches

        Parameters
        ----------
        adv : torch.Tensor
            advantage values computed using the value function
        lambda_coef : float
            lambda coefficient to determine the importance of the value function
        max_grad : float
            value to clip the gradient
        group_weights_iter : int
            number of iterations for the gradient descent

        Returns
        -------
        is_weights: torch.Tensor
            importance sampling weights
        optimal_group_weights: torch.Tensor
            optimal group weights
        """
        # Solve group weights
        num_branches, num_group = adv.shape

        # Auto Gradient Descent
        with torch.inference_mode(False):
            with torch.set_grad_enabled(True):
                optimal_group_weights_logits = torch.zeros(
                    [num_group, 1], requires_grad=True
                ).to(adv.device)
                for iter in range(group_weights_iter):
                    optimal_group_weights_logits.retain_grad()

                    optimal_group_weights = torch.softmax(
                        optimal_group_weights_logits, dim=0
                    ).to(torch.bfloat16)

                    loss_group_weights = torch.exp(
                        lambda_coef * adv.detach() @ optimal_group_weights.to(adv.dtype)
                    ).mean()

                    loss_group_weights.backward()
                    # grad = torch.clamp(optimal_group_weights_logits.grad, min=-1 , max=1)
                    grad = torch.clamp(
                        optimal_group_weights_logits.grad, min=-10, max=10
                    )

                    optimal_group_weights_logits = optimal_group_weights_logits - (
                        self.weight_step_size / torch.sqrt(torch.tensor(iter + 1))
                    ) * grad.view(num_group, 1)

        optimal_group_weights = torch.softmax(optimal_group_weights_logits, dim=0).view(
            num_group, 1
        )

        # Get importance sampling weights
        with torch.no_grad():
            is_weights_logs = torch.clamp(
                lambda_coef * adv.detach() @ optimal_group_weights.to(torch.bfloat16),
                max=max_grad,
            )
            is_weights = torch.exp(is_weights_logs) / torch.exp(is_weights_logs).sum()

        return is_weights, optimal_group_weights.T

    def _get_weights(
        self,
        values: torch.Tensor,
        prev_values: torch.Tensor,
        lambda_coef: float,
        max_grad: float,
        group_weights_iter: int = 10,
    ):
        """
        Apply _compute_weights to each set of branches grouped by prompts
        """
        # adv = values - prev_values
        adv = values

        l_is_weights = list()
        l_optimal_group_weights = list()
        n_prompts = values.shape[0] // self.num_branches
        for i in range(n_prompts):
            is_weights, optimal_group_weights = self._compute_weights(
                adv[i * self.num_branches : (i + 1) * self.num_branches],
                self.lambda_coef,
                self.max_grad,
                self.group_weights_iter,
            )
            l_is_weights.append(is_weights)
            l_optimal_group_weights.append(optimal_group_weights)

        is_weights = torch.concat(l_is_weights, dim=0)
        optimal_group_weights = torch.concat(l_optimal_group_weights, dim=0)

        return is_weights, optimal_group_weights

    def _apply_weights(self, adv: torch.Tensor, optimal_group_weights: torch.Tensor):
        """
        Apply the weights to the branches grouped by prompts
        """
        optimal_group_weights = optimal_group_weights.repeat_interleave(
            self.num_branches, dim=0
        )
        values = torch.sum(adv * optimal_group_weights.to(adv.dtype), dim=1)
        return values

    def _select_branches(
        self,
        advs: torch.Tensor,
    ):
        """
        Select the branches based on the weighted sum of the advantage values
        """

        # Reshape the advs promptwise
        advs = advs.reshape(-1, self.num_branches)

        # Select the branch which argmaxes along a specific axis:
        output_idxs = torch.argmax(advs, dim=1)

        # Adjust the output ids based on the number of branches and batch_size:
        adjust_idx = torch.arange(advs.shape[0]) * self.num_branches
        output_idxs += adjust_idx.to(output_idxs.device)

        return output_idxs

    def _divided_evaluation(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        minibatch_size=16,
    ):
        res = list()
        n_minibatches = input_ids.shape[0] // minibatch_size
        n_remainer = input_ids.shape[0] % minibatch_size

        with torch.no_grad():
            for i in range(n_minibatches):
                values, q_values = self._eval_value_function(
                    input_ids[i * minibatch_size : (i + 1) * minibatch_size],
                    attention_mask[i * minibatch_size : (i + 1) * minibatch_size],
                )
                if len(values.shape) > 2:
                    res.append(values[..., -1])
                else:
                    res.append(values)
            if n_remainer > 0:
                values, q_values = self._eval_value_function(
                    input_ids[n_minibatches * minibatch_size :],
                    attention_mask[n_minibatches * minibatch_size :],
                )
                if len(values.shape) > 2:
                    res.append(values[..., -1])
                else:
                    res.append(values)
        # return torch.stack(res, axis=0)
        return torch.concatenate(res, axis=0)

    def _divided_generation(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        max_new_tokens: int,
        do_sample=True,
        minibatch_size=16,
    ):
        iids = input_ids.repeat_interleave(self.num_branches, dim=0)
        attm = attention_mask.repeat_interleave(self.num_branches, dim=0)
        res = (
            torch.ones((iids.shape[0], iids.shape[1] + max_new_tokens))
            .to(input_ids.device)
            .to(input_ids.dtype)
            * self.ref_tokenizer.pad_token_id
        )
        assert len(res.shape) == len(iids.shape)
        n_minibatches = iids.shape[0] // minibatch_size
        n_remainer = iids.shape[0] % minibatch_size

        with torch.no_grad():
            for i in range(n_minibatches):
                branches = self.ref_model.generate(
                    input_ids=iids[i * minibatch_size : (i + 1) * minibatch_size],
                    attention_mask=attm[i * minibatch_size : (i + 1) * minibatch_size],
                    max_new_tokens=max_new_tokens,
                    num_return_sequences=1,
                    top_k=self.top_k,
                    top_p=self.top_p,
                    do_sample=True,
                )
                res[
                    i * minibatch_size : (i + 1) * minibatch_size,
                    res.shape[1] - branches.shape[1] :,
                ] = branches

            if n_remainer > 0:
                branches = self.ref_model.generate(
                    input_ids=iids[n_minibatches * minibatch_size :],
                    attention_mask=attm[n_minibatches * minibatch_size :],
                    max_new_tokens=max_new_tokens,
                    num_return_sequences=1,
                    top_k=self.top_k,
                    top_p=self.top_p,
                    do_sample=True,
                )
                res[
                    n_minibatches * minibatch_size :, res.shape[1] - branches.shape[1] :
                ] = branches

        return res

    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        max_new_tokens: int,
        return_dict_in_generate: bool = False,
        *args,
        **kwargs,
    ):
        with torch.no_grad():
            weights = list()
            l_values = list()
            l_values_mean = list()
            l_advs = list()
            blocks = list()

            start_time = time.time()
            batch_size = input_ids.shape[0]
            for i in range(batch_size):
                blocks.append(list())

            # Calculate the number of iterations and the remainder
            num_iterations = int(math.ceil(max_new_tokens / self.tree_depth))
            remainder = max_new_tokens % self.tree_depth

            # Create an attention mask for the generation of the tree:
            branch_attention_mask = deepcopy(attention_mask).to(attention_mask.device)
            branch_attention_mask = branch_attention_mask.repeat_interleave(
                self.num_branches, dim=0
            )

            # Evalaute the weighted value function for the prompt inputs:
            # prev_values, _ = self._eval_value_function(input_ids, attention_mask)
            prev_values = self._divided_evaluation(input_ids, attention_mask)
            # prev_values = prev_values[..., -1] # (num_prompts, num_values, seq_len) -> (num_prompts, num_values)
            prev_values = prev_values.repeat_interleave(self.num_branches, dim=0)

            pre_generation = time.time()
            generation_times = list()
            unfinished_seqs = torch.ones(input_ids.shape[0], dtype=torch.bool).to(
                input_ids.device
            )

            for i in range(num_iterations):
                if i == num_iterations - 1 and remainder > 0:
                    tree_depth = remainder
                else:
                    tree_depth = self.tree_depth

                # Use the base model to generate the tree:
                # branches = self.ref_model.generate(input_ids=input_ids,
                #                         attention_mask=attention_mask,
                #                         max_new_tokens=tree_depth,
                #                         num_return_sequences=self.num_branches,
                #                         top_k=self.top_k,
                #                         top_p=self.top_p,
                #                         do_sample=True)
                branches = self._divided_generation(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=tree_depth,
                    do_sample=True,
                    minibatch_size=self.minibatch_size,
                )
                len_diff = branches.shape[1] - input_ids.shape[1]

                # Update attention mask for branches:
                # breakpoint()
                branch_attention_mask = torch.concat(
                    [
                        branch_attention_mask,
                        torch.ones(branch_attention_mask.shape[0], len_diff).to(
                            branch_attention_mask.device
                        ),
                    ],
                    dim=1,
                )
                # breakpoint()
                branch_attention_mask[:, -len_diff:][
                    branches[:, -len_diff:] == self.ref_tokenizer.pad_token_id
                ] = 0

                # Evaluate the tree with the value function model:
                # Value function should take input of size [num_branches * batch size, seq_len]
                # Value function should return output of size [num_branches * batch size]
                # values, _ = self._eval_value_function(branches, branch_attention_mask)
                values = self._divided_evaluation(branches, branch_attention_mask)
                assert torch.isnan(values).sum() == 0, "Values contain NaNs"
                # values = torch.sum(values * self.weights, dim=1)[..., -1]

                if self.value_function is None and self.oracle is not None:
                    pass
                    # values = values.to(branch_attention_mask.device)
                else:
                    pass
                    # values = values[..., -1] # (num_branches * num_prompts, num_values, seq_len) -> (num_branches * num_prompts, num_values)

                is_sampling_weights, optimal_group_weights = self._get_weights(
                    values,
                    prev_values,
                    self.lambda_coef,
                    self.max_grad,
                    self.group_weights_iter,
                )
                advs = self._apply_weights(
                    # values - prev_values,
                    values,
                    optimal_group_weights,
                )
                weights.append(optimal_group_weights.detach().cpu())

                # Consider each set of branches:
                output_idxs = self._select_branches(advs)

                # Select the generated input_ids, and update attention mask:
                # Find the difference in size between the input_ids and the branches
                attention_mask = torch.concat(
                    [
                        attention_mask,
                        torch.ones(attention_mask.shape[0], len_diff).to(
                            attention_mask.device
                        ),
                    ],
                    dim=1,
                )
                attention_mask[:, -len_diff:][
                    branches[output_idxs, -len_diff:] == self.ref_tokenizer.pad_token_id
                ] = 0

                input_ids = branches[output_idxs]
                input_ids[~unfinished_seqs, -len_diff:] = (
                    self.ref_tokenizer.pad_token_id
                )
                for finished_token_id in self.tokens_finished:
                    unfinished_seqs = unfinished_seqs & (
                        ((finished_token_id == input_ids[:, -len_diff:]) * 1.0).sum(
                            axis=1
                        )
                        == 0
                    )

                assert len(unfinished_seqs.shape) == 1
                new_blocks = self.ref_tokenizer.batch_decode(input_ids[:, -len_diff:])
                for j in range(batch_size):
                    blocks[j].append(new_blocks[j])

                l_values.append(values[output_idxs].detach().cpu())
                l_values_mean.append(
                    values.view(-1, self.num_branches, values.shape[-1])
                    .mean(axis=1)
                    .detach()
                    .cpu()
                )
                l_advs.append(
                    values[output_idxs].detach().cpu()
                    - prev_values[output_idxs].detach().cpu()
                )
                # Update prev_cbf_values as size [batch_size * num_branches]
                prev_values = values[output_idxs].repeat_interleave(
                    self.num_branches, dim=0
                )

                generation_times.append(time.time())

            # Get the final cbf values:
            final_values = self._divided_evaluation(input_ids, attention_mask)
            q_values = torch.zeros_like(final_values)

            end_time = time.time()
            # Process timings
            total_run_time = [end_time - start_time] * batch_size
            latency_time = self.process_latency_time(
                start_time, pre_generation, generation_times, end_time
            )
            weights = torch.cat(weights, axis=0).detach().cpu()
            l_values = torch.cat(l_values, axis=0).detach().cpu()
            l_values_mean = torch.cat(l_values_mean, axis=0).detach().cpu()
            l_advs = torch.cat(l_advs, axis=0).detach().cpu()

            if return_dict_in_generate:
                return {
                    "generated_ids": input_ids,
                    "generated_ids_attn_mask": attention_mask,
                    # 'values': final_values,
                    "values": l_values,
                    "values_mean": l_values_mean,
                    "advs": l_advs,
                    "q_values": q_values,
                    "total_run_time": total_run_time,
                    "latency_time": [latency_time] * batch_size,
                    "weights": weights,
                    "blocks": blocks,
                }
            else:
                return input_ids
