import pytest
import pytorch_lightning as pl
import torch
import transformers
from peft import LoraConfig, get_peft_model
from unittest.mock import Mock
import torch.nn as nn
from torch.utils.data import DataLoader, IterableDataset

from robust_multi_objective_decoding.value_function import ValueFunctionModule, ActionValueFunctionModule

############## TEST MOCK SETUP ##############

class ProxyBaseModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(32,32))

@pytest.fixture(scope="module")
def lora_config():
    return LoraConfig(
        r=16,
        lora_alpha=16,
        target_modules=[0],
        lora_dropout=0.1,
        bias="none",
    )

@pytest.fixture(scope="module")
def model_and_tokenizer():
    return ProxyBaseModel(), None

@pytest.fixture(scope="module")
def base_value_function_module(model_and_tokenizer, lora_config):
    model, _ = model_and_tokenizer
    return ValueFunctionModule(
        base_model=model,
        base_model_hidden_dim=32,
        lora_config=lora_config,
        torch_dtype=torch.float32,
    )

############## TESTS ##############

def test_valuefunctionmodule_trainable_params(model_and_tokenizer, base_value_function_module, lora_config):
    
    model, _ = model_and_tokenizer
    # Create a LoRA version of the base model to compare against
    # Must ensure the names match
    lora_model = get_peft_model(
        model=model,
        peft_config=lora_config,
        adapter_name=base_value_function_module.learnt_lora_adapter_name,
    )
    lora_trainable_params = set(
        name for name, param in lora_model.named_parameters() if param.requires_grad
    )

    value_function_trainable_params = set(
        name.removeprefix("model.") for name, param in base_value_function_module.named_parameters() if param.requires_grad
    )
    
    # By including 'target' heads if they are trainable then the test will fail
    mlp_heads_trainable_params = set(
        f"{headname}.{name}"
        for headname, head in [
            ("target_hat_head_1", base_value_function_module.target_head_1), 
            ("target_hat_head_2", base_value_function_module.target_head_2),
            ("learnt_head_1", base_value_function_module.learnt_head_1),
            ("learnt_head_2", base_value_function_module.learnt_head_2),
            ("v_head", base_value_function_module.v_head),
        ]
        for name, param in head.named_parameters()
        if param.requires_grad
    )

    # Check the ValueFunctionModule trainable parameters are set correctly
    assert lora_trainable_params.issubset(value_function_trainable_params)
    assert mlp_heads_trainable_params.issubset(value_function_trainable_params)
    assert value_function_trainable_params == (
        lora_trainable_params.union(mlp_heads_trainable_params)
    )

@pytest.mark.slow()
def test_valuefunctionmodule_update_target_weights(base_value_function_module):
    """
    Test the on_after_backward method of the ValueFunctionLearner
    """
    # Save original params, run on_after_backward, and compare to manual Polyak update
    original_params = {name: param.detach().clone() for name, param in base_value_function_module.named_parameters()}
    base_value_function_module.update_target_weights()
    learner_params = dict(base_value_function_module.named_parameters())

    with torch.no_grad():
        for name, param in learner_params.items():
                
            # Manual Polyak update
            if "q_hat_head" in name or "q_hat_lora_adapter" in name:
                source_name = "value_function_module." + base_value_function_module.polyak_update_mapping[name.removeprefix("base_value_function_module.")]
                source_param = learner_params[source_name]
                manual_polyak = (
                    original_params[name] * (1 - base_value_function_module.polyak_coeff) 
                    + source_param * base_value_function_module.polyak_coeff
                )
                
                # Compare manual and automatic polyak updates 
                assert torch.allclose(param, manual_polyak),\
                    f"Parameter {name} has not been Polyak updated correctly (manual result: {manual_polyak}, auto result: {param})"
                


############## TEST MOCK SETUP ##############

class ProxyHuggingfaceOutput:
    hidden_states = [torch.randn(2, 32, 32)] 
    
class ProxyBaseModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(32,32))
        
    def forward(self, input_ids, attention_mask, **kwargs):
        return ProxyHuggingfaceOutput() 

@pytest.fixture(scope="module")
def lora_config():
    return LoraConfig(
        r=16,
        lora_alpha=16,
        target_modules=[0],
        lora_dropout=0.1,
        bias="none",
    )

@pytest.fixture(scope="module")
def model_and_tokenizer():
    return ProxyBaseModel(), None

@pytest.fixture(scope="module")
def value_function_module(model_and_tokenizer, lora_config):
    model, _ = model_and_tokenizer
    return ValueFunctionModule(
        base_model=model,
        base_model_hidden_dim=32,
        lora_config=lora_config,
        torch_dtype=torch.float32,
    )

@pytest.fixture(scope="module")
def action_value_function_module(model_and_tokenizer, lora_config):
    model, _ = model_and_tokenizer
    return ActionValueFunctionModule(
        base_model=model,
        base_model_hidden_dim=32,
        lora_config=lora_config,
        token_vocab_size=32,
        torch_dtype=torch.float32,
    )
    

############## TESTS ##############

def test_valuefunctionmodule_forward(value_function_module):
    
    input_ids = torch.ones(2, 32, dtype=torch.long)
    attention_mask = torch.ones(2, 32, dtype=torch.long)

    vs, qs = value_function_module(input_ids, attention_mask=attention_mask)
    assert vs.shape == torch.Size([2, 32])
    assert qs.shape == torch.Size([2, 31])
    
def test_valuefunctionmodule_get_target_values(value_function_module):
    
    input_ids = torch.ones(2, 32, dtype=torch.long)
    attention_mask = torch.ones(2, 32, dtype=torch.long)

    qs = value_function_module.get_target_values(input_ids, attention_mask=attention_mask)
    
    assert qs.shape == torch.Size([2, 31])
    

def test_actionvaluefunctionmodule_forward(action_value_function_module):
    
    input_ids = torch.ones(2, 32, dtype=torch.long)
    attention_mask = torch.ones(2, 32, dtype=torch.long)

    vs, qs = action_value_function_module(input_ids, attention_mask=attention_mask)
    assert vs.shape == torch.Size([2, 32])
    assert qs.shape == torch.Size([2, 31])    

def test_actionvaluefunctionmodule_get_target_values(action_value_function_module):
    
    input_ids = torch.ones(2, 32, dtype=torch.long)
    attention_mask = torch.ones(2, 32, dtype=torch.long)

    q_hat = action_value_function_module.get_target_values(input_ids, attention_mask=attention_mask)
    
    assert q_hat.shape == torch.Size([2, 31])