# -*- coding: utf-8 -*-
"""
Created on Thu Aug 29 10:07:27 2024

@author: William
"""

import torch 
import warnings
import torch.nn as nn
from typing import List, Tuple
from transformers import PreTrainedTokenizer

def get_batch_logps(logits: torch.FloatTensor, labels: torch.LongTensor, average_log_prob: bool = False)->torch.FloatTensor:
    
    """
    Compute the log probabilities of the given labels under the given logits.
      
    Parameters
    ----------
    logits : torch.FloatTensor
        Logits of the model (unnormalized). Shape: 
            (batch_size, sequence_length, vocab_size).
    labels : torch.LongTensor
        Labels for which to compute the log probabilities. Label tokens with a 
        value of -100 are ignored. Shape: (batch_size, sequence_length)
    average_log_prob : bool, optional
        If True, return the average log probability per (non-masked) token. 
        Otherwise, return the sum of the log probabilities of the (non-masked) 
        tokens. The default is False.

    Returns
    -------
    torch.Tensor
        A tensor of shape (batch_size,) containing the average/sum log 
        probabilities of the given labels under the given logits.
    """
    
    assert logits.shape[:-1] == labels.shape

    labels = labels[:, 1:].clone() #removes first token i.e. start of sentence token -> last label should be applied to penultimate logit, this does this.
    logits = logits[:, :-1, :]     #removes last sequence element as such last label is applied to penultimate logit.
    loss_mask = (labels != -100)
    
    # dummy token; we'll ignore the losses on these tokens later
    labels[labels == -100] = 0

    per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)

    if average_log_prob:
        return (per_token_logps * loss_mask).sum(-1) / loss_mask.sum(-1)
    else:
        return (per_token_logps * loss_mask).sum(-1)
    
def create_label_mask(model_input_ids, model_generation_ids):
    """
    Create a suitable label mask for the get_batch_logps method

    Parameters
    ----------
    model_input_ids : torch.tensor
        token ids of the model input
    model_generation_ids : torch.tensor
        token ids of the model generation output

    Returns
    -------
    labels : torch.tensor
        token ids of only the output tokens with the input tokens masked as -100
    """
    
    labels = model_generation_ids.clone()
    end_of_input = model_input_ids.shape[-1]
    
    labels[:, :end_of_input] = -100
    
    return labels

def separate_prompts_and_responses(prompts:List[str], completions:List[str]) -> Tuple[List[str], List[str]]:
    
    assert len(prompts) == len(completions),\
        f'length of prompts ({len(prompts)} should equal length of completions ({len(completions)})'
        
    responses = list()
    for i, prompt in enumerate(prompts):
        
        completion = completions[i]
        
        assert prompt in completion,\
            f'prompt {prompt} must be in completion {completion}'
        responses.append(completion[len(prompt):].strip())
    
    return prompts, responses
   
def update_attention_mask(attention_mask:torch.Tensor,
                          gen_sample_ids:torch.Tensor,
                          num_return_sequences:int):
    
    attention_mask = attention_mask.repeat_interleave(num_return_sequences, dim=0)
    
    updated_shape = gen_sample_ids.shape[1] - attention_mask.shape[1]
    new_block = torch.ones(attention_mask.shape[0], updated_shape)
    return torch.concat([attention_mask, new_block], dim=1)
    
def kl_divergence(model: nn.Module,
                  reference_model: nn.Module,
                  tokenizer: PreTrainedTokenizer,
                  prompts: List[str],
                  num_generations:int=16,
                  max_new_tokens:int=20,
                  eos_token_id:int=2,
                  device:str='cpu',
                  *args, **kwargs):
    
    """
    Calculate the KL divergence KL[ model || reference_model] over generations
    form a series of prompts
    
    Model and reference model should have been trained with the same tokenizer!

    Parameters
    ----------
    model : nn.Module
        DESCRIPTION.
    reference_model : nn.Module
        DESCRIPTION.
    tokenized_prompts : Dict
        DESCRIPTION.
    num_generations:int = 16

    Returns
    -------
    torch.Tensor([1])
        The KL divergence of the generations under the original sequence.

    """
    
    if type(model) != type(reference_model):
        warnings.warn("""model and reference model are not the same,
                      model only one tokenizer is used in this eval process""")
    
    #Assert tokenizer padding is applied to the correct side?
    assert tokenizer.padding_side == 'left',\
        f'Tokenizer padding side is {tokenizer.padding_side} not "left"'
    
    # 1. From the model and prompts generate a sequence of tokens
    inputs = tokenizer(prompts, return_tensors='pt', *args, **kwargs).to(device)
    model = model.to(device)
    reference_model = reference_model.to(device)        

    print(f'kl divergence: generating {len(prompts)*num_generations} responses')
    with torch.no_grad():
        sample_dict = model.generate(**inputs,
                                 do_sample=True,
                                 max_new_tokens=max_new_tokens,
                                 num_return_sequences=num_generations,
                                 output_logits=True,
                                 return_dict_in_generate=True)
        
        # Process the generation output to be compatible with forward method:
        samples = sample_dict['sequences']
        atten_mask = update_attention_mask(inputs['attention_mask'], samples, num_generations)
        model_scores = torch.stack(sample_dict['logits']).permute(1, 0, 2)
        
        #2. Calculate the probability of generation under the model (conditional probabilities)
        index_start_of_gen = samples.shape[1] - model_scores.shape[1] 
        labels = samples[:, index_start_of_gen:]
        per_token_logps = torch.gather(model_scores.log_softmax(-1), 
                                       dim=2, index=labels.unsqueeze(2)).squeeze(2)
        model_log_prob = per_token_logps.sum(-1)

        # 3. Calculate the probability of the generation under the reference model
        #Decode generation:
        labels = create_label_mask(inputs["input_ids"], samples)
        per_token_ref_logits = reference_model(input_ids=samples, attention_mask=atten_mask).logits
        reference_model_log_prob = get_batch_logps(per_token_ref_logits, labels)
            
        # 4. Return KL div
        kl_div = model_log_prob - reference_model_log_prob
    
    return kl_div.mean()


def win_rate(model: nn.Module,
             ref_model: nn.Module,
             tokenizer: PreTrainedTokenizer,
             reward_model: nn.Module, 
             prompts: List[str],
             max_new_tokens: int = 64,
             *args,
             **kwargs) -> torch.Tensor:
    """
    Calculate the win rate of results from model vs. reference model given the
    reward model. Tokenized prompts are used to prompt both models.
    
    The model\ref_model should following the huggingface generation api format.
    Both models should use the same tokenizer in this version of the code as the 
    prompt inputs are tokenized. TODO: can be adapted otherwise.
    
    The reward model should take a batch of tokenized prompts and return a single
    scalar reward for each input.
    
    Parameters
    ----------
    model : nn.Module
        Huggingface Causal LM model.
    ref_model : nn.Module
        Huggingface Causal LM model.
    tokenizer: 
        Huggingface Tokenizer
    reward_model : nn.Module
        Reward model, takes list of prompts and responses as an input and returns a reward
    prompts : List[str]
        List of string prompt inputs
    max_new_tokens : int, optional
        The maximum tokens the model and reference model can generate in the 
        win rate comparison. The default is 64.

    Returns
    -------
    win_rate: torch.tensor
        The fraction of times the model generations rewards beat those of the reference model
        generation rewards
    """
    
    tokenized_prompts = tokenizer(prompts, return_tensors='pt', *args, **kwargs)
        
    with torch.no_grad(): 
    
        #1. Generate responses from the model:
        model_gen = model.generate(**tokenized_prompts, 
                                   max_new_tokens=max_new_tokens,
                                   do_sample=True)
        ref_model_gen = ref_model.generate(**tokenized_prompts, 
                                   max_new_tokens=max_new_tokens,
                                   do_sample=True)
        
        model_gen = tokenizer.batch_decode(model_gen, skip_special_tokens=True)
        ref_gen = tokenizer.batch_decode(ref_model_gen, skip_special_tokens=True)
        
        #1.5. Split the input into prompts and responses to input into the reward model:
        _, model_resp = separate_prompts_and_responses(prompts, model_gen)
        _, ref_resp   = separate_prompts_and_responses(prompts, ref_gen)
        
        #2. Evaluate generations on the reward model:
        model_rewards = reward_model(prompts=prompts, responses=model_resp)
        ref_model_rewards = reward_model(prompts=prompts, responses=ref_resp)
        
        #3. Compare the generations: TODO fix bugs
        win_rate = (model_rewards > ref_model_rewards).to(torch.float32)
        
    return win_rate.mean()