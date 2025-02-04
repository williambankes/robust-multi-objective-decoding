import math
import time
import torch
import numpy as np
from copy import deepcopy
from typing import List
import torch.nn as nn
from transformers import PreTrainedTokenizer, AutoTokenizer, AutoModelForCausalLM
from robust_multi_objective_decoding.value_function import ValueFunctionModule
from robust_multi_objective_decoding.oracles.shield_gemma import HarmType, ShieldGemmaSafetyOracle

class BestOfNOracleDecoder(nn.Module):

    def __init__(self,
                 reference_model: nn.Module,
                 oracle: ShieldGemmaSafetyOracle,
                 tokenizer: PreTrainedTokenizer,
                 num_branches: int = 8,
                 oracle_batch_size: int = 1,
                 safety_prob_cutoff: float = 0.5,
                 *args, **kwargs):
        """
        Best Of N Safety Oracle Decoder runs best of N decoding using
        the safety oracle to select optimal branches. Each iteration
        num_branches full generations are made and the safety oracle
        evaluates all of them. A safe branch is then selected.

        Parameters
        ----------
        reference_model : nn.Module
            _description_
        safety_oracle : ShieldGemmaSafetyOracle
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

        assert self.oracle_batch_size > 0,\
            f"Oracle batch size: {oracle_batch_size} must be greater than 0"

    def get_hyperparameters(self):
        """
        Get hyperparameters specific to the decoder

        Returns
        -------
        dict
            dictionary with keys of the hyperparameter name and values attached
        """

        return {'num_branches': self.num_branches}
    
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

        #TODO: Update this to handle any oracle
        # Process the safety labels:

        if isinstance(self.oracle, ShieldGemmaSafetyOracle):
            oracle_values = self._process_safety_label_cutoff(oracle_values, self.safety_prob_cutoff)

        # Outputs:
        output_idxs = list()
        # For each batch select a safe generation:
        for batch_idx in range(batch_size):

            lower_idx = batch_idx*self.num_branches
            upper_idx = (batch_idx+1)*self.num_branches
            
            # Return the branch with the highest oracle score:
            idx = torch.argmax(oracle_values[lower_idx: upper_idx]).view(1)
            output_idxs.append(idx + lower_idx)
            
        return torch.concat(output_idxs)         

    def process_latency_timing(self, generate_time: float,
                                pre_oracle_eval_time: float,
                                start_time: float,
                                end_time: float,
                                oracle_eval_times: List[float]) -> List[float]:

        if len(oracle_eval_times) == 1:
            output = [end_time - start_time]

        else:

            output = [generate_time + (oracle_eval_times[0] - pre_oracle_eval_time)]
            branch_select_time = end_time - oracle_eval_times[-1]

            for i in range(1, len(oracle_eval_times)):

                oracle_eval_time = oracle_eval_times[i] - oracle_eval_times[i-1]
                output.append(generate_time + oracle_eval_time + branch_select_time)

        return output

    def generate(self, input_ids: torch.Tensor,
                attention_mask: torch.Tensor,
                max_new_tokens: int,
                return_dict_in_generate: bool = False,
                *args, **kwargs):
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
        branches = self.ref_model.generate(input_ids=input_ids,
                                attention_mask=attention_mask,
                                max_new_tokens=max_new_tokens,
                                num_return_sequences=self.num_branches,
                                do_sample=True)

        # TODO: Separate prompts and response before decoding using tensor operations on the input_ids matrix
        # This requires knowing which side the tokenization occurs on

        # Decode and split prompts and response:
        decoded_prompts = self.tokenizer.batch_decode(input_ids.repeat_interleave(self.num_branches, dim=0),
                                                      skip_special_tokens=True)
        decoded_branches = self.tokenizer.batch_decode(branches, skip_special_tokens=True)
    
        # Remove the prompt from the branch generations:
        responses = list()
        for i, branch in enumerate(decoded_branches):
            responses.append(branch[len(decoded_prompts[i]):])

        # Calculate the time for generating the prompts and responses:
        pre_oracle_eval_time = time.time()
        generate_time = pre_oracle_eval_time - start_time

        # Evaluate the prompts and responses oracle_values:
        oracle_eval_time = list()
        if self.oracle_batch_size > 1:

            oracle_values = list()
            num_iterations = (batch_size * self.num_branches) // self.oracle_batch_size

            # Ensure the oracle batch size is a multiple of the batch size:
            assert batch_size * self.num_branches % self.oracle_batch_size == 0,\
                f"Oracle batch size: {self.oracle_batch_size} must be a multiple of batch_size * num_branches: {batch_size * self.num_branches}"
            assert self.oracle_batch_size <= batch_size * self.num_branches,\
                f"Oracle batch size: {self.oracle_batch_size} must be less than or equal to batch_size * num_branches: {batch_size * self.num_branches}"

            for i in range(num_iterations):
                lower_idx = i*self.oracle_batch_size
                upper_idx = (i+1)*self.oracle_batch_size
                oracle_values.append(self.oracle.score(decoded_prompts[lower_idx:upper_idx],
                                                              responses[lower_idx:upper_idx]))
                                
                # Record the time for each oracle evaluation:
                oracle_eval_time.append(time.time())

            oracle_values = torch.concat(oracle_values, dim=0)
            assert oracle_values.shape[0] == batch_size * self.num_branches,\
                f"Safety labels shape: {oracle_values.shape[0]}, should match batch_size * self.num_branches: {batch_size * self.num_branches}"

        else:
            oracle_values = self.oracle.score(decoded_prompts, responses)
            
            # Record time for oracle evaluation:
            oracle_eval_time.append(time.time())
      
        # For each input select the best safety labels.
        output_idxs = self._select_branches(oracle_values, batch_size)

        # Select the best branches:
        output_ids = branches[output_idxs]
        output_attention_mask_matrix = torch.ones(batch_size, max_new_tokens, dtype=torch.long)
        output_attention_mask = torch.concat([attention_mask,
                                              output_attention_mask_matrix.to(attention_mask.device)], dim=1)

        # Ensure output dimensions are correct:
        assert output_attention_mask.shape[0] == batch_size,\
            f"output_attention_mask shape 0: {output_attention_mask.shape[0]}, should be {batch_size}"
        assert output_attention_mask.shape[1] == (max_new_tokens + input_ids.shape[1]),\
            f"output_attention_mask shape 1: {output_attention_mask.shape[1]}, should be {(max_new_tokens + input_ids.shape[1])}" 

        # Calculate the total run time and latency time:
        end_time = time.time()
        latency_time = self.process_latency_timing(generate_time=generate_time,
                                pre_oracle_eval_time=pre_oracle_eval_time,
                                start_time=start_time,
                                end_time=end_time,
                                oracle_eval_times=oracle_eval_time)
        
        #Adjust the safety_labels shape for logging:
        oracle_values = oracle_values.reshape(batch_size, self.num_branches, -1)

        if return_dict_in_generate:
            return {"generated_ids": output_ids, "generated_ids_attn_mask": attention_mask,
                    'oracle_values': oracle_values, 
                    'total_run_time': [end_time - start_time] * batch_size, # all outputs must be a list of len batch_size
                    'latency_time': [latency_time] * batch_size}
        else:
            return output_ids
        

class BlockwiseOracleDecoder(nn.Module):

    def __init__(self, reference_model: nn.Module,
                 tokenizer: PreTrainedTokenizer,
                 safety_oracle: ShieldGemmaSafetyOracle,
                 safety_prob_cutoff: float,
                 harm_types: List[str] = [HarmType.DANGEROUS], 
                 num_branches: int = 8,
                 tree_depth: int = 8,
                 oracle_batch_size: int = 1,
                 *args, **kwargs):
        """
        Blockwise Oracle Decoder runs blockwise decoding using a safety oracle instead of a CBF
        to guide the selection of branches at decoding time. The safety oracle uses the tokenizer
        to accept plain text inputs instead of tokenized inputs.

        Parameters
        ----------
        reference_model : nn.Module
            The reference model from which outputs are generated
        tokenizer : PreTrainedTokenizer
            A pretrained Tokenizer for the reference model
        safety_oracle : ShieldGemmaSafetyOracle
            A safety oracle that can evaluate the safety of a response given a prompt
        safety_prob_cutoff : float
            The cutoff value for the safety probabilities returned by the oracle
        harm_types : List[str], optional
            Safety oracle safety policies, by default [HarmType.DANGEROUS]
        num_branches : int, optional
            Number of branches generated by reference model at each iteration, by default 8
        tree_depth : int, optional
            The depth of the trees generated by the oracle decoder, by default 8
        """

        super().__init__()
        
        self.harm_types = harm_types
        self.safety_oracle = safety_oracle
        self.ref_model = reference_model
        self.tokenizer = tokenizer
        self.num_branches = num_branches
        self.tree_depth = tree_depth
        self.safety_prob_cutoff = safety_prob_cutoff
        self.oracle_batch_size = oracle_batch_size

    def get_hyperparameters(self):
        """
        Return relevant hyperparameters for the decoder

        Returns
        -------
        Dict
            A dictionary of hyperparameters for the decoder
        """

        return {'num_branches': self.num_branches,
                'tree_depth': self.tree_depth,
                'safety_prob_cutoff': self.safety_prob_cutoff}

    
    def process_safety_oracle_eval_times(self, safety_oracle_eval_times, pre_gen_time) -> float:

        if len(safety_oracle_eval_times) == 1:
            return safety_oracle_eval_times[0] - pre_gen_time

        else:

            output = [safety_oracle_eval_times[0] - pre_gen_time]
            for i in range(1, len(safety_oracle_eval_times)):

                oracle_eval_time = safety_oracle_eval_times[i] - safety_oracle_eval_times[i-1]
                output.append(oracle_eval_time)

            return np.array(output).mean()

    def _process_safety_label_cutoff(self, safety_labels: torch.Tensor, cutoff: float):
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
        torch.Tensor
            A tensor with values of 1 if the prob is greater than the cutoff, 0 otherwise
        """

        # Apply the cutoff filter:
        cutoff_safety_labels = (safety_labels > cutoff).sum(dim=1)
        return (cutoff_safety_labels == safety_labels.shape[1]).to(torch.int64)
    
    def _get_oracle_values(self, prompt_ids, generated_ids):
        """
        Get the safety oracle values for the current decoded branches.

        TODO: update to oracle batch setup as per Best of N approach

        Parameters
        ----------
        prompt_ids : torch.Tensor
            The prompt ids of shape [batch_size, seq_len]
        generated_ids : torch.Tensor
            Current generated ids of shape [batch_size * num_branches, seq_len]

        Returns
        -------
        torch.Tensor
            Output safety values of shape [batch_size * num_branches]
        """
        
        # Get batch size from the prompts:
        batch_size = prompt_ids.shape[0]

        # Decode the prompts and responses:
        decoded_prompts = self.tokenizer.batch_decode(prompt_ids.repeat_interleave(self.num_branches, dim=0),
                                                      skip_special_tokens=True)
        decoded_branches = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    
        # Remove the prompt from the branch generations:
        responses = list()
        for i, branch in enumerate(decoded_branches):
            responses.append(branch[len(decoded_prompts[i]):])

        safety_oracle_eval_time = list()
        if self.oracle_batch_size > 1:

            safety_labels = list()
            num_iterations = len(responses) // self.oracle_batch_size

            # Ensure the oracle batch size is a multiple of the batch size:
            assert len(responses) % self.oracle_batch_size == 0,\
                f"Oracle batch size: {self.oracle_batch_size} must be a multiple of len(responses): {len(responses)}"
            assert self.oracle_batch_size <= len(responses),\
                f"Oracle batch size: {self.oracle_batch_size} must be less than or equal to len(responses): {len(responses)}"

            for i in range(num_iterations):
                lower_idx = i*self.oracle_batch_size
                upper_idx = (i+1)*self.oracle_batch_size
                safety_labels.append(self.safety_oracle.score(decoded_prompts[lower_idx:upper_idx],
                                                              responses[lower_idx:upper_idx]))
                                
                # Record the time for each oracle evaluation:
                safety_oracle_eval_time.append(time.time())

            # Concat outputs and check the shape:
            safety_labels = torch.concat(safety_labels, dim=0)
            assert safety_labels.shape[0] == len(responses),\
                f"Safety labels shape: {safety_labels.shape[0]}, should match batch_size * self.num_branches: {len(responses)}"

        else:
            safety_labels = self.safety_oracle.score(decoded_prompts, responses)
            
            # Record time for oracle evaluation:
            safety_oracle_eval_time.append(time.time())

        # Reshape to [batch_size, num_branches]
        joint_branch_probs = safety_labels.prod(dim=1).reshape(batch_size, self.num_branches)
               
        return self._process_safety_label_cutoff(safety_labels, self.safety_prob_cutoff), safety_oracle_eval_time, joint_branch_probs

    def _filter_branches(self, batch_size, oracle_values):
        """
        For each element of the batch, return a branch with a safe value

        Parameters
        ----------
        batch_size : int
            The batch size of the prompt input_ids
        oracle_values : _type_
            The binary safety labels of the safety oracle

        Returns
        -------
        torch.Tensor
            The indices of the selected branches
        List
            A list of 1s and 0s indicating if a safe branch was selected or not
        """
        
        output_idxs = list()
        is_safe = list()

        for batch_idx in range(batch_size):

            lower_idx = batch_idx*self.num_branches
            upper_idx = (batch_idx+1)*self.num_branches

            # Find the viable branches:
            batch_idx_oracle_values = oracle_values[lower_idx: upper_idx]            

            # Filter:
            safety_filter = (batch_idx_oracle_values == 1.0)

            # If all branches are unsafe select one at random:
            if torch.sum(safety_filter) == 0:
                output_idx = torch.randint(0, self.num_branches, (1,)).to(safety_filter.device)
                is_safe.append(0)

            else: # From safe branches, sample a viable one
                viable_idx = torch.where(safety_filter)[0] #outputs a tuple (tensor,) -> select first element
                sampled_idx = torch.randint(0, viable_idx.shape[0], (1,))
                output_idx = viable_idx[sampled_idx]
                is_safe.append(1)

            # Adjust output_idx to account for offset:
            output_idxs.append(output_idx + lower_idx)
            
        return output_idxs, is_safe

    def generate(self, input_ids: torch.Tensor,
            attention_mask: torch.Tensor,
            max_new_tokens: int,
            return_dict_in_generate: bool = False,
            *args, **kwargs):

        start_time = time.time()
        batch_size = input_ids.shape[0]
        prompt_input_ids = deepcopy(input_ids).to(input_ids.device)

        # Calculate the number of iterations and the remainder
        num_iterations = int(math.ceil(max_new_tokens/self.tree_depth))
        remainder = max_new_tokens % self.tree_depth

        # Create an attention mask for the generation of the tree:
        branch_attention_mask = deepcopy(attention_mask).to(attention_mask.device)
        branch_attention_mask = branch_attention_mask.repeat_interleave(self.num_branches, dim=0)

        pre_generation = time.time()
        
        is_safe_record = list()
        joint_branch_probs_record = list()
        generation_times = list()
        for i in range(num_iterations):

            if i == num_iterations - 1 and remainder > 0:
                tree_depth = remainder
            else:
                tree_depth = self.tree_depth

            # Use the base model to generate the tree:
            branches = self.ref_model.generate(input_ids=input_ids,
                                    attention_mask=attention_mask,
                                    max_new_tokens=tree_depth,
                                    num_return_sequences=self.num_branches,
                                    do_sample=True)

            # Update attention mask for branches:
            branch_attention_mask = torch.concat([branch_attention_mask,
                                torch.ones(branch_attention_mask.shape[0],
                                        tree_depth).to(branch_attention_mask.device)],
                                        dim=1)          

            # Evaluate the tree with the value function model:
            # Value function should take input of size [num_branches * batch size, seq_len]
            # Value function should return output of size [num_branches * batch size]
            oracle_values, safety_oracle_eval_times, joint_branch_probs = self._get_oracle_values(prompt_input_ids, branches)
            post_oracle_value_time = time.time()

            # Consider each set of branches:
            output_idxs, is_safe = self._filter_branches(batch_size, oracle_values)

            # Select the generated input_ids, and update attention mask:
            output_idxs = torch.concat(output_idxs)
            
            input_ids = branches[output_idxs]
            attention_mask = torch.concat([attention_mask,
                                torch.ones(attention_mask.shape[0], tree_depth).\
                                to(attention_mask.device)], dim=1)
            
            # Update output stats for the branch filter process:
            is_safe_record.append(is_safe)
            joint_branch_probs_record.append(joint_branch_probs)

            # Record the generation time as the latency time for each generation:
            processed_safety_oracle_eval_times = self.process_safety_oracle_eval_times(safety_oracle_eval_times, pre_gen_time=pre_generation)
            generation_times.append(time.time() - post_oracle_value_time + processed_safety_oracle_eval_times)

        is_safe_record = np.stack(is_safe_record, axis=0)
        joint_branch_probs_record = torch.stack(joint_branch_probs_record, axis=0)
        joint_branch_probs_record = joint_branch_probs_record.cpu().to(torch.float32).numpy()

        assert is_safe_record.shape == (num_iterations, batch_size),\
            f"Final shape of is_safe_record: {is_safe_record.shape}, is incorrect"

        end_time = time.time()
        # Process timings

        total_run_time = [end_time - start_time] * batch_size
       
        if return_dict_in_generate:
            return {"generated_ids": input_ids, "generated_ids_attn_mask": attention_mask,
                    'is_safe_record': is_safe_record, 'joint_branch_probs_record': joint_branch_probs_record,
                    'total_run_time': total_run_time, 'latency_time': [generation_times]*batch_size}
        else:
            return input_ids