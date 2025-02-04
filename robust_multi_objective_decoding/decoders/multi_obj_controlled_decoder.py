import math
import time
import torch
import numpy as np
from copy import deepcopy
import torch.nn as nn
from robust_multi_objective_decoding.value_function import ValueFunctionModule
from transformers import PreTrainedTokenizer

class MultiObjectiveControlledDecoder(nn.Module):

    def __init__(self,
                 reference_model: nn.Module,
                 value_function: ValueFunctionModule,
                 tokenizer: PreTrainedTokenizer,
                 ref_tokenizer: PreTrainedTokenizer,
                 weights: list[float], 
                 num_branches: int = 8,
                 tree_depth: int = 8,
                 max_length: int = 1024,
                 use_adv: bool = False,
                 minibatch_size: int = 16,
                 top_k: int = 0,
                 top_p: float = 1.0,
                 oracle=None,
                 *args, **kwargs):

        super().__init__()

        self.ref_model = reference_model
        self.value_function = value_function
        self.weights = torch.tensor(weights, dtype=torch.float32).view(1,-1,1)
        self.num_branches = num_branches
        self.tree_depth = tree_depth
        self.tokenizer = tokenizer
        self.ref_tokenizer = ref_tokenizer
        self.tokenizer_diff = True
        self.max_length = max_length
        self.use_adv = use_adv
        self.top_k = top_k
        self.top_p = top_p
        self.oracle = oracle
        self.minibatch_size = minibatch_size

        # If the tokenizers are the same we don't need to do the re-tokenization step:
        if self.ref_tokenizer.name_or_path == self.tokenizer.name_or_path:
            self.tokenizer_diff = False

        self.tokens_finished = [self.ref_tokenizer.eos_token_id]
        if "gemma-2-2b-it" in self.ref_tokenizer.name_or_path:
            print("using gemma-2-2b-it: <end_of_turn> in the chat format")
            self.tokens_finished.append(self.ref_tokenizer.encode("<end_of_turn>")[-1])
            
        # assert torch.allclose(self.weights.sum(), torch.tensor(1.0)), "Weights should sum to 1.0"
        if not torch.allclose(self.weights.sum(), torch.tensor(1.0)):
            print("Weights do not sum to 1.0")
            # normalize the weights
            self.weights = self.weights / self.weights.sum()

    def _eval_value_function(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        """
        Evaluate the value function across the entire sentence generation

        Parameters
        ----------
        input_ids : torch.Tensor
            Input ids from the reference model
        attention_mask : torch.Tensor
            Attention mask from the reference model
        Returns
        -------
        torch.Tensor, torch.Tensor
            Multiple output heads from the value function
        """

        # Decoder the responses from the reference model:
        decoded_inputs = self.ref_tokenizer.batch_decode(input_ids,
                                            skip_special_tokens=True)
        decoded_inputs = [
            item.replace("<pad>", "") for item in decoded_inputs
        ]
        decoded_inputs = [
            item.replace("<eos>", "") + "<eos>" if "<eos>" in item else item for item in decoded_inputs
        ]

        if self.value_function is not None:
            # Re-tokenizer the responses using the value function tokenizer
            inputs = self.tokenizer(decoded_inputs,
                                    return_tensors='pt',
                                    padding=True,
                                    truncation=True,
                                    max_length=self.max_length)
            
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
                values = self.oracle.score(prompt_batch, response_batch)
            q_values = torch.zeros_like(values)
        else:
            raise NotImplementedError
        return values, q_values
    
    def eval(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        """
        Method used in eval.py to evaluate the batch returned response, 
        """
        return self._eval_value_function(input_ids, attention_mask)
        
    def get_hyperparameters(self):
        """
        Get hyperparameters specific to the decoder

        Returns
        -------
        dict
            dictionary with keys of the hyperparameter name and values attached
        """

        return {'num_branches': self.num_branches,
                'tree_depth': self.tree_depth,
                'weights': self.weights.cpu().numpy()}
        
    def _divided_evaluation(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        minibatch_size = 16,
    ):
        res = list()
        n_minibatches = input_ids.shape[0]//minibatch_size
        n_remainer = input_ids.shape[0] % minibatch_size

        with torch.no_grad():
            for i in range(n_minibatches):
                values, q_values = self._eval_value_function(
                    input_ids[i*minibatch_size:(i+1)*minibatch_size],
                    attention_mask[i*minibatch_size:(i+1)*minibatch_size]
                )
                if len(values.shape) > 2:
                    res.append(values[..., -1])
                else:
                    res.append(values)
            if n_remainer > 0:
                values, q_values = self._eval_value_function(
                    input_ids[n_minibatches*minibatch_size:],
                    attention_mask[n_minibatches*minibatch_size:]
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
        minibatch_size=16
    ):
        iids = input_ids.repeat_interleave(self.num_branches, dim=0)
        attm = attention_mask.repeat_interleave(self.num_branches, dim=0)
        res = torch.ones(
            (iids.shape[0], iids.shape[1] + max_new_tokens)
        ).to(input_ids.device).to(input_ids.dtype) * self.ref_tokenizer.pad_token_id
        assert len(res.shape) == len(iids.shape)
        n_minibatches = iids.shape[0]//minibatch_size
        n_remainer = iids.shape[0] % minibatch_size

        with torch.no_grad():
            for i in range(n_minibatches):
                branches = self.ref_model.generate(
                    input_ids=iids[i*minibatch_size:(i+1)*minibatch_size],
                    attention_mask=attm[i*minibatch_size:(i+1)*minibatch_size],
                    max_new_tokens=max_new_tokens,
                    num_return_sequences=1,
                    top_k=self.top_k,
                    top_p=self.top_p,
                    do_sample=True
                )
                res[
                    i*minibatch_size:(i+1)*minibatch_size, 
                    res.shape[1] - branches.shape[1]:
                ] = branches
                
            if n_remainer > 0:
                branches = self.ref_model.generate(
                    input_ids=iids[n_minibatches*minibatch_size:],
                    attention_mask=attm[n_minibatches*minibatch_size:],
                    max_new_tokens=max_new_tokens,
                    num_return_sequences=1,
                    top_k=self.top_k,
                    top_p=self.top_p,
                    do_sample=True
                )

                res[
                    n_minibatches*minibatch_size:,
                    res.shape[1] - branches.shape[1]:
                ] = branches
                
        return res

    def process_latency_time(self, start_time,
                            pre_generation,
                            generation_times,
                            end_time):
        
        # Calculate the latency time for each iteration of the generation process:
        #setup_time = pre_generation - start_time
        init_iter_time = generation_times[0] - pre_generation
        #shut_down_time = end_time - generation_times[-1]

        iter_times = [init_iter_time]
        for i in range(1, len(generation_times)):

            latency_time = generation_times[i] - generation_times[i-1]

            iter_times.append(latency_time)

        return iter_times


    def _select_branches(self, batch_size: int, num_branches: int,
                         values: torch.Tensor, prev_values: torch.Tensor,
                         use_adv:bool=True):

        output_idxs = list()

        # Reshape the values to reflect the batch size:
        values = values.reshape(batch_size, num_branches)
        prev_values = prev_values.reshape(batch_size, num_branches)   
        if use_adv:
            adv = values - prev_values
        else: 
            adv = values
            
        # Select the branch which argmaxes along a specific axis:
        output_idxs = torch.argmax(adv, dim=1)

        # Adjust the output ids based on the number of branches and batch_size:
        adjust_idx = torch.arange(batch_size) * num_branches
        adjust_idx = adjust_idx.to(output_idxs.device)
        output_idxs += adjust_idx       
            
        return output_idxs


    def generate(self, input_ids: torch.Tensor,
                attention_mask: torch.Tensor,
                max_new_tokens: int,
                return_dict_in_generate: bool = False,
                *args, **kwargs):
        
        with torch.no_grad():
            weights = list()
            l_values = list()
            l_values_mean = list()
            l_advs = list()
            
            start_time = time.time()
            batch_size = input_ids.shape[0]
            ref_pad_token_id = self.ref_tokenizer.pad_token_id

            blocks = list()
            for i in range(batch_size): 
                blocks.append(list())

            # Calculate the number of iterations and the remainder
            num_iterations = int(math.ceil(max_new_tokens/self.tree_depth))
            remainder = max_new_tokens % self.tree_depth

            # Create an attention mask for the generation of the tree:
            branch_attention_mask = deepcopy(attention_mask).to(attention_mask.device)
            branch_attention_mask = branch_attention_mask.repeat_interleave(self.num_branches, dim=0)

            # Evalaute the weighted value function for the prompt inputs:

            prev_values = self._divided_evaluation(input_ids, attention_mask)
            self.weights = self.weights.to(prev_values.device)
            prev_values = torch.sum(prev_values * self.weights.view(1, -1), dim=1)
            prev_values = prev_values.repeat_interleave(self.num_branches, dim=0)
            assert prev_values.shape[0] == input_ids.shape[0] * self.num_branches,\
                f"initialisation for prev_cbf_values shape: {prev_values.shape}, is incorrect"

            sequence_is_finished = torch.zeros(input_ids.shape[0], dtype=torch.bool).to(input_ids.device)

            pre_generation = time.time()
            generation_times = list()

            unfinished_seqs = torch.ones(input_ids.shape[0], dtype=torch.bool).to(input_ids.device)

            for i in range(num_iterations):

                if i == num_iterations - 1 and remainder > 0:
                    tree_depth = remainder
                else:
                    tree_depth = self.tree_depth

                unfinished_input_ids = input_ids[~sequence_is_finished]
                unfinished_attention_mask = attention_mask[~sequence_is_finished]

                branches = self._divided_generation(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=tree_depth,
                    do_sample=True,
                    minibatch_size=self.minibatch_size
                )
                len_diff = branches.shape[1] - input_ids.shape[1]

                # Update attention mask for branches:
                branch_attention_mask = unfinished_attention_mask.repeat_interleave(self.num_branches, dim=0)
                branch_attention_mask = torch.concat([branch_attention_mask,
                                torch.ones(branch_attention_mask.shape[0],
                                            tree_depth).to(branch_attention_mask.device)],
                                            dim=1)

                # Set all padding token ids to zero -> pad tokens in the newly generated responses are now zero:
                branch_attention_mask[branch_attention_mask == ref_pad_token_id] = 0     

                # Evaluate the tree with the value function model:
                # Value function should take input of size [num_branches * batch size, seq_len]
                # Value function should return output of size [num_branches * batch size]
                values, _ = self._eval_value_function(branches, branch_attention_mask)
                values = torch.sum(values * self.weights, dim=1)[..., -1]
                assert torch.isnan(values).sum() == 0, "Values contain NaNs"

                # Consider each set of unfinished branches:
                output_idxs = self._select_branches((~sequence_is_finished).sum(),
                                                    self.num_branches,
                                                    values, prev_values,
                                                    use_adv=self.use_adv)

                # Select the max branches:
                selected_input_ids = branches[output_idxs]
                tokens_added = selected_input_ids.shape[1] - input_ids.shape[1]
                eos_pad_tokens = torch.tensor([self.ref_tokenizer.eos_token_id,
                                            self.ref_tokenizer.pad_token_id]).to(input_ids.device)

                # Update the previous values tensor to those that haven't yet finished:
                prev_values = values[output_idxs]
                prev_values_filter = torch.isin(selected_input_ids[:, -1], eos_pad_tokens)
                
                # Update previous values to tensor [~sequence_is_finish.sum(), seq_len]
                prev_values = prev_values[~prev_values_filter].repeat_interleave(self.num_branches, dim=0)

                # Update the input_ids tensor:
                new_block = (torch.ones(batch_size, tokens_added) * ref_pad_token_id).\
                    to(torch.int64).\
                    to(input_ids.device)
                input_ids = torch.concat([input_ids, new_block], dim=1)
                input_ids[~sequence_is_finished, :] = selected_input_ids

                # Create the full attention mask
                new_attention_block = torch.ones(batch_size, tokens_added).\
                    to(torch.int64).\
                    to(input_ids.device)
                new_attention_block[input_ids[:, -tokens_added:] == ref_pad_token_id] = 0
                attention_mask = torch.concat([attention_mask, new_attention_block], dim=1)             
                
                # Update this sequence_is_finished tensor                
                sequence_is_finished = torch.isin(input_ids[:, -1], eos_pad_tokens).\
                                                to(torch.bool).\
                                                to(input_ids.device)
                generation_times.append(time.time())

                # If all sequences have finished generating, break the loop
                if sequence_is_finished.all():
                    break

            # Get the final cbf values:
            #final_cbf_values = self._get_cbf_value(input_ids, attention_mask)
            # final_values, q_values = self._eval_value_function(input_ids, attention_mask)
            final_values = self._divided_evaluation(input_ids, attention_mask)
            q_values = torch.zeros_like(final_values)

            end_time = time.time()
            # Process timings

            total_run_time = [end_time - start_time] * batch_size
            latency_time = self.process_latency_time(start_time, pre_generation, generation_times, end_time)

            weights = torch.cat(weights, axis=0).detach().cpu()
            l_values = torch.cat(l_values, axis=0).detach().cpu()
            l_values_mean = torch.cat(l_values_mean, axis=0).detach().cpu()
            l_advs = torch.cat(l_advs, axis=0).detach().cpu()

            if return_dict_in_generate:
                return {
                    "generated_ids": input_ids, 
                    "generated_ids_attn_mask": attention_mask,
                    # 'values': final_values, 
                    'values': l_values, 
                    'values_mean': l_values_mean, 
                    "advs": l_advs,
                    'q_values': q_values, 
                    'total_run_time': total_run_time,
                    'latency_time': [latency_time]*batch_size,
                    "weights": weights,
                    "blocks": blocks
                    }
            else:
                return input_ids