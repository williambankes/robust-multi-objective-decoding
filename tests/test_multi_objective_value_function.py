import torch
import torch.nn as nn
from robust_multi_objective_decoding.multi_objective_value_function import MultiHeadValueFunction
from peft import LoraConfig

class ProxyOutput:

    def __init__(self, output):
        self.hidden_states = output

class ProxyBase(nn.Module):

    def __init__(self):
        super().__init__()
        self.weights = nn.Linear(1,32)

    def forward(self, input_ids, attention_mask, output_hidden_states:bool):
        return ProxyOutput([torch.randn(input_ids.shape[0], input_ids.shape[1], 10).to(torch.float16)])
    
def test_MultiHeadValueFunction():
    batch_size = 4
    num_rewards = 3
    seq_length = 5
    base_model_hidden_dim = 10
    token_vocab_size = 100

    lora_config = LoraConfig(
                r=16,
                lora_alpha=16,
                target_modules=['weights'],
                lora_dropout=0.1) 

    base_model = ProxyBase()
    model = MultiHeadValueFunction(
        base_model=base_model,
        base_model_hidden_dim=base_model_hidden_dim,
        num_heads=num_rewards,
        lora_config=lora_config,
        token_vocab_size=token_vocab_size
    )
    model.setup()

    inputs = torch.randint(0, token_vocab_size, (batch_size, seq_length))
    attention_mask = torch.ones(batch_size, seq_length)
    outputs, _ = model(inputs, attention_mask)

    assert outputs.shape == (batch_size, num_rewards, seq_length), f"Expected shape {(batch_size, num_rewards, seq_length)}, but got {outputs.shape}"