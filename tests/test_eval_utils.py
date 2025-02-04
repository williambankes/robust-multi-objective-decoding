# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 13:55:16 2024

@author: William
"""

import torch
import pytest
from robust_multi_objective_decoding.utils.eval_utils import (
    get_batch_logps,
    kl_divergence,
    win_rate,
    create_label_mask,
    separate_prompts_and_responses)
from transformers import AutoTokenizer, AutoModelForCausalLM


class ProxyTokenizer():
    
    def __call__(self, batch_input, *args, **kwargs):
        return {'input_ids': torch.ones(5,5)}

    def batch_decode(self, inputs, *args, **kwargs):
        
        if (inputs == 0).all():
            return ['testing 1,2,3', 'testing 1,2,3']
        else:
            return ['testing 1,2,3,4', 'testing 1,2,3,4']

class ProxyRewardModel():
    
    def __call__(self, prompts, responses):
    
        if len(responses[0]) > 5:
            return torch.Tensor([1.]*len(responses))
        else:
            return torch.Tensor([0.]*len(responses))
    
class ProxyModel():
    
    def __init__(self, reference:bool):
        self.ref = reference
    
    def generate(self, input_ids:torch.Tensor, max_new_tokens, do_sample:bool=True):
        
        output = torch.ones_like(input_ids)
        
        if self.ref:
            output = output*0
    
        return output
        
def test_win_rate():
        
    tok = ProxyTokenizer()
    test_model = ProxyModel(reference=False)
    test_ref_model = ProxyModel(reference=True)
    test_reward = ProxyRewardModel()
    
    output = win_rate(test_model, test_ref_model, tok, test_reward, ['testing', 'testing'])    
    assert output == torch.tensor(1.0)
    
def setup_model(name, device):
        
    tokenizer = AutoTokenizer.from_pretrained(name)
    
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id   
    
    model = AutoModelForCausalLM.from_pretrained(name).to(device)
    
    return model, tokenizer
    
def test_get_batch_logps():    
    
    #Create logits with dims (2, 5, 5):
    logits = torch.stack([torch.stack([torch.tensor([0.1,0.2,0.3,0.4])]*5)]*2).log()
    
    #Create labels with dims (2,5):
    labels = torch.stack([torch.tensor([-100, -100, -100, 0,1])]*2)
    
    #Calculate answer and assert:
    answer = torch.tensor([0.1, 0.2]).log().sum()
    logps = get_batch_logps(logits, labels, False)
    
    assert (logps == answer).all()    
    
def test_create_label_mask():
    
    model_input_ids = torch.stack([torch.tensor([2,3,1])]*2)
    gen_input_ids = torch.stack([torch.tensor([2,3,1,0,1])]*2)
        
    labels = create_label_mask(model_input_ids, gen_input_ids)
    
    assert labels.shape == (2,5)
    assert (labels[:,[0,1,2]] == -100).all(), labels
      
def test_separate_prompts_and_responses():
    
    prompts = ['once upon a time', 'a long time ago']
    completions = ['once upon a time there was a dog',
                   'a long time ago there was a cat']
    
    ps, rs = separate_prompts_and_responses(prompts, completions)
    
    assert ps[0] == prompts[0]
    assert rs[0] == 'there was a dog'
    
    bad_completions = ['a long time ago there was', 'once upon a time there was']
    bad_completions2 = ['a long time ago', 'upon a midnight clear', 'there sat a dog']
    
    with pytest.raises(AssertionError):
        separate_prompts_and_responses(prompts, bad_completions)
        separate_prompts_and_responses(prompts, bad_completions2)
        
        
def test_kl_divergence():
    
    model, tokenizer = setup_model( "openaccess-ai-collective/tiny-mistral", 'cpu')
    
    prompts = ['Once upon a time', 'A long time ago in a Galaxy far away']
    
    kl_div = kl_divergence(model=model,
                           reference_model=model,
                           tokenizer=tokenizer,
                           prompts=prompts,
                           num_generations=16,
                           max_new_tokens=20,
                           eos_token_id=tokenizer.eos_token_id,
                           padding=True, truncation=True, max_length=20)
    
    assert torch.abs(kl_div) < 1e-3
           
def test_kl_divergence_diff_model():
    
    model, tokenizer = setup_model( "openaccess-ai-collective/tiny-mistral", 'cpu')
    ref_model, ref_tokenizer = setup_model("echarlaix/tiny-random-mistral", 'cpu')
        
    prompts = ['Once upon a time', 'A long time ago in a Galaxy far away']
    
    kl_div = kl_divergence(model=model,
                           reference_model=model,
                           tokenizer=tokenizer,
                           prompts=prompts,
                           num_generations=16,
                           max_new_tokens=20,
                           eos_token_id=tokenizer.eos_token_id,
                           padding=True, truncation=True, max_length=20)
    
    # Check that the empirical kl divergence is greater than 0 with a small tolerance:
    assert kl_div + 1e-3 >= 0