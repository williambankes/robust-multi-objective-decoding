import torch
from torch import nn
import pytest
from robust_multi_objective_decoding.value_function_learner import ValueFunctionLearner
from robust_multi_objective_decoding.value_function import MLPHead

############## MLP HEAD TESTS ##############

def test_mlp_head_layer_norm():
    """Test that layer normalization is properly initialized and working"""
    head = MLPHead(
        input_dim=10,
        hidden_dim=20,
        output_dim=1,
        use_layer_norm=True
    )
    
    # Check layer norm is initialized
    assert hasattr(head, 'layer_norm')
    assert isinstance(head.layer_norm, nn.LayerNorm)
    assert head.layer_norm.normalized_shape == (10,)
    
    # Test layer norm is applied
    x = torch.randn(5, 10)  # batch_size=5, input_dim=10
    output_with_norm = head(x)
    
    # Create same head without layer norm
    head_no_norm = MLPHead(
        input_dim=10,
        hidden_dim=20,
        output_dim=1,
        use_layer_norm=False
    )
    head_no_norm.fc1.weight = nn.Parameter(head.fc1.weight.clone())
    head_no_norm.fc1.bias = nn.Parameter(head.fc1.bias.clone())
    head_no_norm.fc2.weight = nn.Parameter(head.fc2.weight.clone())
    head_no_norm.fc2.bias = nn.Parameter(head.fc2.bias.clone())
    
    output_without_norm = head_no_norm(x)
    
    # Outputs should be different due to layer norm
    assert not torch.allclose(output_with_norm, output_without_norm)


def test_mlp_head_xavier_init():
    """Test that xavier initialization is properly applied"""
    head = MLPHead(
        input_dim=10,
        hidden_dim=20,
        output_dim=1,
        xavier_init=True
    )
    
    # Check weights are initialized with small values (gain=0.1)
    assert torch.all(torch.abs(head.fc1.weight) < 0.5)
    assert torch.all(torch.abs(head.fc2.weight) < 0.5)
    
    # Check biases are initialized to zero
    assert torch.allclose(head.fc1.bias, torch.zeros_like(head.fc1.bias))
    assert torch.allclose(head.fc2.bias, torch.zeros_like(head.fc2.bias))
    
    # Compare with non-xavier init
    head_no_xavier = MLPHead(
        input_dim=10,
        hidden_dim=20,
        output_dim=1,
        xavier_init=False
    )
    
    # Non-xavier weights should generally have larger values
    assert torch.mean(torch.abs(head.fc1.weight)) < torch.mean(torch.abs(head_no_xavier.fc1.weight))


def test_mlp_head_sigmoid():
    """Test sigmoid activation in MLPHead"""
    # Test scaled sigmoid
    head_scaled = MLPHead(
        input_dim=10,
        hidden_dim=20,
        output_dim=1,
        use_sigmoid=True,
        scale_sigmoid=True,
        discount_factor=0.9
    )
    
    x = torch.randn(5, 10)
    output_scaled = head_scaled(x)
    
    # Output should be between 0 and 1/(1-gamma)
    assert torch.all(output_scaled >= 0)
    assert torch.all(output_scaled <= 1/(1-0.9))
    
    # Test unscaled sigmoid
    head_unscaled = MLPHead(
        input_dim=10,
        hidden_dim=20,
        output_dim=1,
        use_sigmoid=True,
        scale_sigmoid=False
    )
    
    output_unscaled = head_unscaled(x)
    
    # Output should be between 0 and 1
    assert torch.all(output_unscaled >= 0)
    assert torch.all(output_unscaled <= 1)


############## LEARNER TESTS ##############
############## SETUP MOCKS AND PROXIES ##############

class ProxyValueFunctionModule:

    def __init__(self):
        self.get_param_idx = 0
        self.get_param_output = [torch.tensor([0.3]), torch.tensor([0.5])]
    
        self.value_function_lora_adapter_name = "value_function_lora_adapter"
        self.q_hat_lora_adapter_name = "q_hat_lora_adapter"

        self._state_dict = {
            "layer1.lora_A.value_function_lora_adapter.weight": torch.tensor([0.5]),
            "layer1.lora_A.q_hat_lora_adapter.weight": torch.tensor([0.3])}

    def named_parameters(self):
        return [(k,v) for k,v in self._state_dict.items()]

    def get_parameter(self):
        out = self.get_param_output[self.get_param_idx]
        self.get_param_idx += 1
        return out

    def set_adapter(self, name):
        return None

    def add_adapter(self, name, adapter):
        return None

    def __call__(self, *args, **kwargs):
        return (torch.tensor([[1.0, 2.0, 3.0, 4.0]]),
                torch.tensor([[1.0, 2.0, 3.0]]))

    def get_target_values(self, *args, **kwargs):
        return torch.tensor([[1.0, 2.0, 3.0]])

    def state_dict(self):
        return self._state_dict

    def load_state_dict(self, obj):
        assert isinstance(obj, dict), "obj must be a dictionary"
        self._state_dict = obj

    def update_target_weights(self):
        self._state_dict = {
            "layer1.lora_A.learnt_lora_adapter.weight": torch.tensor([0.5]),
            "layer1.lora_A.target_lora_adapter.weight": torch.tensor([0.301])}
    
@pytest.fixture(scope="module")
def value_function_module():
    return ProxyValueFunctionModule()

@pytest.fixture()
def value_function_learner(value_function_module):
    return ValueFunctionLearner(
        base_value_function_module=value_function_module,
        update_q_hat_every=1,
        learning_rate=10,
        cbf_mc_loss = False,
        mc_loss = False,
        iql_loss = True
    )

@pytest.fixture()
def value_function_learner_cbf_mc_loss(value_function_module):
    return ValueFunctionLearner(
        base_value_function_module=value_function_module,
        update_q_hat_every=1,
        learning_rate=10,
        iql_loss = True,
        cbf_mc_loss=True
    )

@pytest.fixture()
def value_function_learner_mc_loss(value_function_module):
    return ValueFunctionLearner(
        base_value_function_module=value_function_module,
        update_q_hat_every=1,
        learning_rate=10,
        cbf_mc_loss=False,
        mc_loss = True,
        iql_loss = False
    )

@pytest.fixture()
def value_function_learner_tw_mc_loss(value_function_module):
    return ValueFunctionLearner(
        base_value_function_module=value_function_module,
        update_q_hat_every=1,
        learning_rate=10,
        cbf_mc_loss=False,
        mc_loss = False,
        tw_mc_loss= True,
        iql_loss = False
    )

############## TESTS ##############
   
def test_valuefunctionlearner_process_batch_for_iql_loss(value_function_learner):
    # Prepare a mock batch
    batch = {
        "input_ids": torch.tensor([[1, 2, 3, 4]]),
        "tokenwise_safety_labels": torch.tensor([[0, 1, 0, 1]]),
        "attention_mask": torch.tensor([[1, 1, 1, 1]])
    }    
    
    transition_safety_labels, v, q, q_hat = value_function_learner.process_batch_for_iql_loss(batch)

    assert transition_safety_labels.shape == torch.Size([1, 3])
    assert v.shape == torch.Size([1, 4])
    assert q.shape == torch.Size([1, 3])
    assert q_hat.shape == torch.Size([1, 3])

def test_valuefunctionlearner_iql_loss(value_function_learner):
    # 2 state deterministic Markov chain (A, B)
    # Actions: 0, 1 (stay, move)
    # Rewards: 0 if in A, 1 if in B
    r = torch.tensor([[0.0, 1.0]])  #   r1=0,    r2=1

    # Q(s1, a1) = 0
    # Q(s1, a2) = 1
    q = torch.tensor([[0.0, 1.0]])
    q_hat = torch.tensor([[0.0, 0.0]])

    # V(s1) = 0
    # V(s2) = 1
    # V(s3) = 1
    v = torch.tensor([[0.0, 0.0, 1.0]])

    # IQL loss
    L = 0
    gamma = 0.99
    tau = 0.9
    for i in range(2):
        L += (r[:, i] + gamma * v[:, i + 1] - q[:, i]) ** 2
        u = q_hat[:, i] - v[:, i]
        if u < 0:
            weight = 1 - tau
        else:
            weight = tau
        L += weight * u**2
    assert not torch.isnan(L)

    func_iql_loss = value_function_learner.iql_loss(r, v, q, q_hat, gamma, tau)
    assert torch.allclose(func_iql_loss, L/2)

def test_valuefunctionlearner_masked_iql_loss(value_function_learner):

    # 2 state deterministic Markov chain (A, B) w. masking
    r = torch.tensor([[-100.0, 0.0, 1.0]])

    q = torch.tensor([[-1.0, 0.0, 1.0]])
    q_hat = torch.tensor([[-1.0, 0.0, 0.0]])
    v = torch.tensor([[0.0, 0.0, 0.0, 1.0]])

    gamma = 0.99
    tau = 0.9

    masked_loss_eval = value_function_learner.iql_loss(r, v, q, q_hat, gamma, tau)

    # IQL loss for non-masked transitions
    r = torch.tensor([[0.0, 1.0]])
    q = torch.tensor([[0.0, 1.0]])
    q_hat = torch.tensor([[0.0, 0.0]])
    v = torch.tensor([[0.0, 0.0, 1.0]])

    # IQL loss
    unmasked_loss = 0
    for i in range(2):
        unmasked_loss += (r[:, i] + gamma * v[:, i + 1] - q[:, i]) ** 2
        u = q_hat[:, i] - v[:, i]
        if u < 0:
            weight = 1 - tau
        else:
            weight = tau
        unmasked_loss += weight * u**2
    assert not torch.isnan(unmasked_loss)

    # Average the loss across the unmaksed tokens i.e. 2 
    assert torch.allclose(masked_loss_eval, unmasked_loss/2)

def test_valuefunctionlearner_cbf_mc_loss_term(value_function_learner_cbf_mc_loss):

    # 2 state deterministic Markov chain (A, B) w. masking
    r = torch.tensor([[-100.0, 0.0, 1.0]])

    q = torch.tensor([[-1.0, 0.0, 1.0]])
    q_hat = torch.tensor([[-1.0, 0.0, 0.0]])
    v = torch.tensor([[0.0, 0.0, 0.0, 1.0]])

    gamma = 0.99
    tau = 0.9

    masked_loss_eval = value_function_learner_cbf_mc_loss.iql_loss(r, v, q, q_hat, gamma, tau)

    # IQL loss for non-masked transitions
    r = torch.tensor([[0.0, 1.0]])
    q = torch.tensor([[0.0, 1.0]])
    q_hat = torch.tensor([[0.0, 0.0]])
    v = torch.tensor([[0.0, 0.0, 1.0]])

    # IQL loss
    unmasked_loss = 0
    for i in range(2):
        unmasked_loss += (r[:, i] + gamma * v[:, i + 1] - q[:, i]) ** 2
        u = q_hat[:, i] - v[:, i]
        if u < 0:
            weight = 1 - tau
        else:
            weight = tau
        unmasked_loss += weight * u**2
    assert not torch.isnan(unmasked_loss)
    
    # MC Loss term:
    mc_loss = (v[..., 1:] - (1/(1-gamma)))**2

    # Average the loss across the unmaksed tokens i.e. 2 
    assert torch.allclose(masked_loss_eval, unmasked_loss/2 + (mc_loss/2).sum())


def test_valuefunction_learner_mc_loss(value_function_learner_mc_loss):
    
    # 2 state deterministic Markov chain (A, B) w. masking 
    r = torch.tensor([[-100.0, 0.0, 1.0]]*3)

    q = None
    q_hat = None
    v = torch.tensor([[0.0, 0.0, 0.0, 1.0]]*3)

    gamma = 0.99
    tau = 0.9

    masked_loss_eval = value_function_learner_mc_loss.iql_loss(r, v, q, q_hat, gamma, tau)

    # IQL loss for non-masked transitions
    r = torch.tensor([[0.0, 1.0]]*3)
    v = torch.tensor([[0.0, 0.0, 1.0]]*3)
    
    # MC Loss term:
    mc_loss = (torch.tensor([0.0, 1.0]) - torch.tensor([1.0]).unsqueeze(1))**2
    
    # Average the loss across the unmaksed tokens i.e. 2 
    assert torch.allclose(masked_loss_eval, (mc_loss/2).sum())


def test_valuefunction_learner_tw_mc_loss(value_function_learner_tw_mc_loss):
    
    # 2 state deterministic Markov chain (A, B) w. masking
    r = torch.tensor([[-100.0, 1.0, 1.0]]*3)

    q = None
    q_hat = None
    v = torch.tensor([[0.0, 0.0, 0.0, 0.0]]*3)

    gamma = 0.99
    tau = 0.9

    masked_loss_eval = value_function_learner_tw_mc_loss.iql_loss(r, v, q, q_hat, gamma, tau)

    # IQL loss for non-masked transitions
    v = torch.tensor([[0.0, 0.0, 0.0]])
    
    # MC Loss term:
    cumulative_reward = torch.tensor([[((1 - 0.99**2)/(1 - 0.99)), 1.0]])
    tw_mc_loss = (v[..., 1:] - cumulative_reward) ** 2

    # Average the loss across the unmaksed tokens i.e. 2 
    assert torch.allclose(masked_loss_eval, (tw_mc_loss/2).sum())


def test_valuefunctionlearner_configure_optimizers(value_function_learner):
    
    value_function_learner.parameters = \
        nn.Sequential(nn.Linear(1,1),nn.Linear(1,1)).parameters
    
    optimizers = value_function_learner.configure_optimizers()
    assert isinstance(optimizers, torch.optim.AdamW)
    assert optimizers.defaults['lr'] == value_function_learner.learning_rate

def test_valuefunctionlearner_on_after_backward(value_function_learner):
        
    value_function_learner.on_after_backward()
    
    param1 = value_function_learner.base_value_function_module.state_dict()["layer1.lora_A.target_lora_adapter.weight"]
    assert torch.allclose(param1, torch.tensor([0.301]))  # Apply Polyak averaging
