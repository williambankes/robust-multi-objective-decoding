
import torch
import numpy as np
import pandas as pd

from transformers import AutoTokenizer
from typing import List, Tuple
from .collate_functions import pad_safety_labels

def process_prompt_response(batch, tokenizer,
                            max_length:int,
                            rand_len:bool,
                            rand_len_range:Tuple[int, int],
                            apply_chat_template:bool=False):
    """
    Process the prompt and response strings in the input batch, if rand_len bool is available, 
    the max length is set between (rand_len_range[0], rand_len_range[1]).

    Parameters
    ----------
    batch : List[Tuple]
        The input batch of tuples.
    tokenizer : AutoTokenizer
        The tokenizer to use.
    max_length : int
        The maximum length of the tokenized inputs.
    rand_len : bool
        Whether to set random lengths for the responses. Useful when responses are longer than
        the max_len variable of the tokenizer, to address the prompt.
    rand_len_range : Tuple[int, int]
        The range of random lengths to set the responses to.
    apply_chat_template: bool
        Apply the chat template of instruction fine tuned models.

    Returns
    -------
    tokenized_inputs : dict
        Output of tokenizer given prompt and response.
    tokenized_prompt_lens : List[int]
        The lengths of the tokenized prompts.
    tokenized_response_lens : List[int]
        The lengths of the tokenized responses.
    response : List[str]
        The text responses.
    labels : List[List[int]]
        The multi objective labels for the tokenized inputs.
    idx : List[int]
        The indices of the input batch
    """
    
    if apply_chat_template:
        raise NotImplementedError("Chat template not implemented yet")

    idx, prompt, response, labels = zip(*batch)

    # If rand_len True, max lengths are setup to a random length
    if rand_len:
        resp_max_lens = [np.random.randint(rand_len_range[0], rand_len_range[1]) for _ in response]
    else:
        resp_max_lens = [max_length] * len(response)

    # Find the prompts lengths 
    tokenized_prompt_lens = [tokenizer(p, return_tensors='pt')['input_ids'].shape[-1] for p in prompt]
    
    response = [tokenizer(r, return_tensors='pt', truncation=True, max_length=resp_max_lens[i]) for i, r in enumerate(response)]
    tokenized_response_lens = [r['input_ids'].shape[-1] for r in response]
    response = [tokenizer.decode(r['input_ids'][0], skip_special_tokens=True) for r in response]

    # Join the prompts and responses:
    prompt_response = [prompt[i] + resp for i, resp in enumerate(response)]
    tokenized_inputs = tokenizer(prompt_response, padding=True, truncation=True,
                                    max_length=max_length, return_tensors="pt")
    
    # Add the tokenized prompt to the inputs:
    tokenized_prompts = tokenizer(prompt, padding=True, truncation=True,
                                      max_length=max_length, return_tensors="pt")
    tokenized_inputs['prompt_input_ids'] = tokenized_prompts['input_ids']
    tokenized_inputs['prompt_attention_mask'] = tokenized_prompts['attention_mask']
    
    return tokenized_inputs, tokenized_prompt_lens, tokenized_response_lens, response, labels, idx


def process_eos_reward(labels:List[List[int]],
                       response:List[str], #TODO: Find a nice way to standardize the input
                    tokenizer: AutoTokenizer,
                    tokenized_inputs:dict,
                    tokenized_prompt_lens:List[int],
                    tokenized_response_lens:List[int],
                    mask_token:int
                    ):
    """
    Process the EOS reward structure.

    output = [0,0,0,0,0,0, ... ,0,0,0, reward]

    Parameters
    ----------

    """
    
    eos_labels = list() 

    assert len(labels) == len(tokenized_response_lens),\
        f"Expected {len(tokenized_response_lens)} labels, but got {len(labels)}"

    for i, label in enumerate(labels):

        # Set the last element to be 1 or -1 depending on the last element of the label:
        assert tokenized_response_lens[i] > 0,\
            f"Tokenized response length is {tokenized_response_lens[i]} for response {response[i]}"
        eos_label = [0]*(tokenized_response_lens[i] - 1) # minus 1 to account for the BOS token
        eos_label[-1] = label
        
        # TODO: this is trimming the output despite a really large max length for some reason?
        eos_labels.append(pad_safety_labels(eos_label, 
                                            tokenized_response_lens[i] - 1, # minus 1 to account for the BOS token
                                            tokenized_prompt_lens[i],
                                            tokenizer.padding_side,
                                            tokenizer.truncation_side,
                                            tokenized_inputs['input_ids'].shape[1],
                                            mask_token=mask_token))
        
    # Return as a tensor:
    return torch.stack(eos_labels)
        
# Some sort of general high level function that creates the eventual collate function
def create_collate_functions(tokenizer, 
                            reward_collations:list,
                            max_length:int,
                            rand_len:bool,
                            rand_len_range:Tuple[int, int]=(72 , 128),
                            mask_token:int = -100,
                            apply_chat_template:bool=False):
    """
    Create the collate functions for the different types of tokenizer and different reward_collate_functions.

    - If rand length the prompt and response should be processed in a different way. 

    Parameters
    ----------
    tokenizer : AutoTokenizer
        The tokenizer to use.
    reward_collations : list
        The list of reward collate functions to use.
         - 'eos' End Of Sentence reward, expects a single value r, and returns a reward
                 structure of [0] * (len(input) - 1) + [r] i.e. the reward at the end of
                 the sentence
         - 'cbf' Control Barrier Function reward structure, expects a reward vector of the
                 same length as the input containing only {0,1} values. The reward should
                 transition from 1 to 0 once in the sequence, here 1 implies the constraint is
                 met and 0 that it is not.
    max_length : int
        The maximum length of the tokenized inputs.
    rand_len : bool
        Whether to use random lengths for the responses.
    apply_chat_template: bool
        Apply the chat template of instruction fine tuned models.

    Returns
    -------
    collate_fn
        The collate function to use.    
    """

    if 'cbf' in reward_collations and apply_chat_template:
        raise ValueError("Cannot apply chat template with CBF reward collation function")

    # Build the reward collation functions:
    reward_collation_funcs = list()
    for i, collate_fn in enumerate(reward_collations):
        if collate_fn == 'eos':
            reward_collation_funcs.append(process_eos_reward)
        else:
            raise ValueError(f"Unknown reward collation function {collate_fn}")

    def collate_fn(batch:List[Tuple]) -> Tuple[torch.Tensor]:
        
        # Process the batch:
        tokenized_inputs, tokenized_prompt_lens, tokenized_response_lens, responses, labels, idx = process_prompt_response(
            batch, tokenizer, max_length, rand_len, rand_len_range, apply_chat_template=apply_chat_template)
        
        # For each reward collation function, process the labels:
        # Adjust labels from [batch_size, diff_rewards, (seq_len)] to be of dimension [diff rewards, batch_size, (seq len)]        
        df_labels = pd.DataFrame(list(labels), columns=[f'idx_{i}' for i,_ in enumerate(labels[0])])
        labels = df_labels.T.values.tolist()

        # Ensure labels and reward collate functions are the same length:
        assert len(labels) == len(reward_collations),\
            f"Expected {len(labels)} reward collations, but got {len(reward_collations)}"

        rewards = [reward_collation_funcs[i](labels=label,
                    response=responses,
                    tokenizer=tokenizer,
                    tokenized_inputs=tokenized_inputs,
                    tokenized_prompt_lens=tokenized_prompt_lens,
                    tokenized_response_lens=tokenized_response_lens,
                    mask_token=mask_token,
                    ) for i, label in enumerate(labels)]

        # Add the rewards to the tokenized inputs adjust back to [batch_size, diff_rewards, seq_len]:
        tokenized_inputs['rewards'] = torch.stack(rewards).permute(1,0,2)

        assert tokenized_inputs['rewards'].shape[:2] == (len(batch), len(reward_collations)),\
            f"Expected shape {(len(batch), len(reward_collations))}, but got {tokenized_inputs['rewards'].shape[:2]}"
        
        tokenized_inputs['idx'] = torch.tensor(idx)
                                    
        return tokenized_inputs

    return collate_fn