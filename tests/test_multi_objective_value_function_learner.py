
import pytest
import torch
import torch.nn as nn

from unittest.mock import patch
from robust_multi_objective_decoding.value_function_learner import MultiObjectiveValueFunctionLearner

##### FIXTURES #####

class ProxyBaseValueFunctionModule(nn.Module):

    def __init__(self, num_outputs:int=5):
        super().__init__()
        self.num_outputs = 5

    def is_polyak_average(self):
        return False

    def get_number_outputs(self):
        return self.num_outputs

    def forward(self, input_ids, attention_mask):
        """
        Return output [batch size, num_outputs, len seq]
        """

        return torch.ones(input_ids.shape[0],
                        self.num_outputs,
                        input_ids.shape[1]), \
                torch.ones(input_ids.shape[0],
                    self.num_outputs,
                    input_ids.shape[1] - 1)
    
    def get_target_values(self, input_ids, attention_mask):
        """
        Return q_hat values
        """
        output = torch.ones(input_ids.shape[0],
                          self.num_outputs,
                          input_ids.shape[1] - 1)
        
        return output, output
    
class ProxyLoss(nn.Module):

    def forward(self, q=None, r=None, *args, **kwargs):

        if q is None:
            output =  torch.ones_like(r)
        elif r is None:
            output = torch.ones_like(q)
        else:
            output = torch.ones_like(r)
    
        return output

    def get_name(self):
        return 'ProxyLoss'

@pytest.fixture()
def multi_obj_value_function_learner():
    return MultiObjectiveValueFunctionLearner(
    base_value_function_module=ProxyBaseValueFunctionModule(),
    losses=[ProxyLoss()]*5
    )
        

def test_multiobj_process_batch_safe(multi_obj_value_function_learner):

    r = torch.tensor([[[-100.0, -100.0, 0.0, 1.0]]*5]*3)
    input_ids = torch.ones(3,4)
    attention_mask = input_ids

    batch = {'idx': 1,
             'input_ids': input_ids,
             'attention_mask': attention_mask,
             'rewards': r}

    rewards, v, q, v_hat, q_hat = multi_obj_value_function_learner.process_batch(batch)

    v_target = torch.ones(3, 5, 4)
    q_target = torch.ones(3,5,3)
    q_hat_target = torch.ones(3,5,3)

    assert (rewards == r[...,1:]).all()
    assert (v == v_target).all()
    assert (q == q_target).all()
    assert (q_hat == q_hat_target).all()


def test_multiobj_process_batch_mispec_mask(multi_obj_value_function_learner):

    r = torch.concat([torch.tensor([[[-100.0, -100.0, 0.0, 1.0]]*3]*3),
                      torch.tensor([[[-100.0, 0.0, 0.0, 1.0]]*2]*3)], dim=1)

    input_ids = torch.ones(3,4)
    attention_mask = input_ids

    batch = {'idx': 1,
             'input_ids': input_ids,
             'attention_mask': attention_mask,
             'rewards': r}

    # TODO: Use pytest setup here instead
    with pytest.raises(AssertionError):
        rewards, v, q, v_hat, q_hat = multi_obj_value_function_learner.process_batch(batch)

    # When the entire row is masked, the code should raise an AssertionError
    r = torch.concat([torch.tensor([[[-100.0, -100.0, -100.0, -100.0]]*3]*3),
                    torch.tensor([[[-100.0, 0.0, 0.0, 1.0]]*2]*3)], dim=1)

    batch = {'idx': 1,
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'rewards': r}

    with pytest.raises(AssertionError):
        rewards, v, q, v_hat, q_hat = multi_obj_value_function_learner.process_batch(batch)
    

def test_multiobj_process_batch_mispec_model(multi_obj_value_function_learner):
    """
    Ensure an error is raised when the model is the wrong size compared to the losses
    """
    # Setup the reward to be the wrong size here 4 instead of 5:
    r = torch.tensor([[[-100.0, 0.0, 1.0]]*4]*3)

    input_ids = torch.ones(3,4)
    attention_mask = input_ids

    batch = {'idx': 1,
             'input_ids': input_ids,
             'attention_mask': attention_mask,
             'rewards': r}

    # TODO: Use pytest setup here instead
    with pytest.raises(AssertionError):
        rewards, v, q, v_hat, q_hat = multi_obj_value_function_learner.process_batch(batch)
    


def test_multiobj_process_loss(multi_obj_value_function_learner):
    """
    Test the multiobjective process losses method
    """
    
    input_ids = torch.ones(3,5)

    r = torch.tensor([[[-100.0, 0.0, 1.0]]*5]*3)

    q = torch.tensor([[[-100.0, 0.0, 1.0]]*5]*3)
    q_hat = torch.ones_like(q) 
    
    v = torch.tensor([[[0.0, 0.0, 0.0, 1.0]]*5]*3)

    test_value = multi_obj_value_function_learner.process_losses(rewards=r,
                                                    vs=v,
                                                    qs=q,
                                                    v_hats=v,
                                                    q_hats=q_hat,
                                                    vs_next=v[...,1:],
                                                    input_ids=input_ids,
                                                    split='train')
    
    # Hand calculate the new loss:
    comp_value = torch.ones_like(r) * (r!=-100)
    comp_value = comp_value / (2 * 3)
    comp_value = comp_value.sum()

    assert (comp_value == test_value).all()
