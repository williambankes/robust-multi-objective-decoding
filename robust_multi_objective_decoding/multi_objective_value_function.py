import warnings
from abc import abstractmethod
from typing import List, Literal, Optional, Tuple

import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model
import copy


class MLPHead(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        """
        Simple MLP head with a single hidden layer and ReLU activation.
        TODO: when cleaning up the code, remove copy of this from value_function.py
        """

        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        return x


class SigmoidMLPHead(nn.Module):
    def __init__(self,
        input_dim,
        hidden_dim,
        output_dim,
        scale_sigmoid: bool = True,
        discount_factor: float = 0.95,
        use_layer_norm: bool = True,
        use_xavier_init: bool = True,
        ):
        """
        Simple MLP head with a single hidden layer and ReLU activation.
        TODO: when cleaning up the code, remove copy of this from value_function.py
        """

        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.scale_sigmoid = scale_sigmoid
        self.discount_factor = discount_factor
        self.use_layer_norm = use_layer_norm
        self.xavier_init = use_xavier_init
        assert self.discount_factor >= 0 and self.discount_factor <= 1

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()
        self.layer_norm = nn.LayerNorm(hidden_dim) if use_layer_norm else None

        if self.use_layer_norm:
            self.layer_norm = nn.LayerNorm(input_dim)

        # Initialize weights with smaller values
        if self.xavier_init:
            nn.init.xavier_uniform_(self.fc1.weight, gain=0.1)
            nn.init.zeros_(self.fc1.bias)
            nn.init.xavier_uniform_(self.fc2.weight, gain=0.1)
            nn.init.zeros_(self.fc2.bias)

        self.scale_sigmoid = scale_sigmoid
        if not self.scale_sigmoid:
            warnings.warn(
                "discount_factor param is ignored when scale_sigmoid is false"
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sigmoid_scaling = 1.0 if not self.scale_sigmoid else 1 / (1 - self.discount_factor)
        if self.use_layer_norm:
            x = self.layer_norm(x)
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = sigmoid_scaling * self.sigmoid(x)
        return x

class BaseMultiObjectiveValueFunction(nn.Module):
    
    def __init__(
        self,
        base_model: nn.Module,
        base_model_hidden_dim: int,
        num_heads: int,
        lora_config: LoraConfig,
        token_vocab_size: int,
        torch_dtype: torch.dtype = torch.float16,
        q_function: bool = False
    ):
        """
        Base class for Multi Objective Value Function Classes. Instantiates the model architecture in the setup
        method and leaves the forward, get_q_values and get_target_value methods to be implemented by the subclasses.
        
        Parameters
        ----------
        base_model : nn.Module
            Underlying model to be used for the value function module.
        base_model_hidden_dim : int
            Hidden dimension of the output head MLPs.
        lora_config : LoraConfig
            LoraConfig object that contains the configuration for the LoRA adapters.
        token_vocab_size : int
            Size of the token vocabulary, used as the output dimension of the heads.
        torch_dtype : torch.dtype, optional
            torch dtype to which the heads are mapped, by default torch.float16
        q_function : bool, optional
            Weather the architecture creates a q function or not, by default False
        """

        super().__init__()

        self.base_model = base_model
        self.base_model_hidden_dim = base_model_hidden_dim
        self.lora_config = lora_config
        self.num_heads = num_heads
        self.token_vocab_size = token_vocab_size
        self.torch_dtype = torch_dtype
        self.q_function = q_function
        
        self.setup()

    def is_polyak_average(self):
        """
        Returns
        -------
        bool
            Whether the value function is polyak average-able or not, returns False by
            default in the parent class.
        """
        return False

    @abstractmethod
    def setup(self):
        """
        Setup the model architecture, including different output heads and LoRA adapters.
        """
        pass

    @abstractmethod
    def forward(self, input_ids, attention_mask, **kwargs):
        """
        Evaluate the output of the model.

        Returns
        -------
        torch.Tensor([batch_size, reward_dim, seq_len])
            Value outputs of the model.    
        """
        pass


class MultiHeadValueFunction(BaseMultiObjectiveValueFunction):
        
    def __init__(
        self,
        base_model: nn.Module,
        base_model_hidden_dim: int,
        num_heads: int,
        lora_config: LoraConfig,
        token_vocab_size: int,
        torch_dtype: torch.dtype = torch.float16,
        q_function: bool = False
    ):
        """
        A Multi objective value function which uses an MLP with a single set of multiple output heads at the last layer
        to generate the value outputs for different losses. 
        """
        # Note: there isn't a unified interface to get the hidden dimensions from a model, which is why need the base_model_hidden_dim arg
        super().__init__(base_model=base_model,
                         base_model_hidden_dim=base_model_hidden_dim,
                         num_heads=num_heads,
                         lora_config=lora_config,
                         token_vocab_size=token_vocab_size,
                         torch_dtype=torch_dtype,
                         q_function=q_function)

        self.selected_reward_indices = None

    def setup(self):
    
        self.output_heads = nn.ModuleList(
            nn.Linear(self.base_model_hidden_dim, 1).to(self.torch_dtype) for _ in range(self.num_heads)
        )

        # Add the LoRA adapters
        self.learnt_lora_adapter_name = None
  
        # If the lora configs are provided, set up the model with the LoRA adapter
        if self.lora_config is not None:

            self.learnt_lora_adapter_name = "learnt_lora_adapter"

            self.model = get_peft_model(
                model=self.base_model,  # type: ignore
                peft_config=self.lora_config,
                adapter_name=self.learnt_lora_adapter_name,
            )

            # Enable the value function LoRA adapter
            self.model.set_adapter(self.learnt_lora_adapter_name)
        else:
            self.model = self.base_model

    def get_number_outputs(self):
        return self.num_heads
    
    def get_target_values(self, input_ids, attention_mask, **kwargs):
        """
        Return q_hat as vector of zeros. This class does not support action value functions.

        Returns
        -------
        torch.Tensor
            A torch tensor of zeros with the shape [batch_size, num_heads, seq_len]
        """

        output = torch.zeros(input_ids.shape[0], self.num_heads, input_ids.shape[1], device=input_ids.device)
        return output, output

    def forward(self, input_ids, attention_mask, **kwargs) -> Tuple[torch.Tensor, None]:
        """
        Evaluate the output of the model.

        Returns
        -------
        torch.Tensor([batch_size, reward_dim, seq_len])
            Value outputs of the model.
        torch.Tensor([batch_size, num_heads, seq_len])
            Q-values of the model. This class doesn't support Q-values, so it returns a tensor of zeros.
        """
        # Get the base model outputs
        base_model_outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            **kwargs,
        ).hidden_states[-1]

        # Get the output of the learnt heads
        #output = self.output_head(base_model_outputs)
        output = torch.stack([head(base_model_outputs) for head in self.output_heads], dim=2).squeeze(-1)  # (batch_size, seq_len, num_heads)

        # select the subset of heads if selected_reward_indices is provided
        if self.selected_reward_indices is not None:
            output = output[:, :, self.selected_reward_indices]
        else:
            assert output.shape == (input_ids.shape[0], input_ids.shape[1], self.num_heads),\
                f"output shape: {output.shape} wrong, should be {(input_ids.shape[0], input_ids.shape[1], self.num_heads)}"

        qs = torch.zeros(input_ids.shape[0], self.num_heads, input_ids.shape[1], device=input_ids.device)

        return output.permute(0,2,1), qs

    def select_reward_indices(self, selected_reward_indices: torch.Tensor):
        self.selected_reward_indices = selected_reward_indices
    

class MultiModelValueFunction(BaseMultiObjectiveValueFunction):
        
    def __init__(
        self,
        base_model: nn.Module,
        base_model_hidden_dim: int,
        num_heads: int,
        lora_config: LoraConfig,
        token_vocab_size: int,
        torch_dtype: torch.dtype = torch.float16,
        q_function: bool = False
    ):
        """
        A Multi objective value function which uses an MLP with a single set of multiple output heads at the last layer
        to generate the value outputs for different losses. 
        """
        # Note: there isn't a unified interface to get the hidden dimensions from a model, which is why need the base_model_hidden_dim arg
        super().__init__(base_model=base_model,
                         base_model_hidden_dim=base_model_hidden_dim,
                         num_heads=num_heads,
                         lora_config=lora_config,
                         token_vocab_size=token_vocab_size,
                         torch_dtype=torch_dtype,
                         q_function=q_function)

    def setup(self):

        # Create multiple models:
        self.models = nn.ModuleList([copy.deepcopy(self.base_model) for _ in range(self.num_heads)])

        head_hidden_dim = 2 * self.base_model_hidden_dim
        
        # Create multiple MLP Heads:
        self.output_heads = nn.ModuleList([MLPHead(
            input_dim=self.base_model_hidden_dim,
            hidden_dim=head_hidden_dim,
            output_dim=1,
        ).to(self.torch_dtype) for _ in range(self.num_heads)])

        # self.output_heads = nn.ModuleList(
        #     nn.Linear(self.base_model_hidden_dim, 1).to(self.torch_dtype) for _ in range(self.num_heads)
        # )

    def get_number_outputs(self):
        return self.num_heads
    
    def get_target_values(self, input_ids, attention_mask, **kwargs):
        """
        Return q_hat as vector of zeros. This class does not support action value functions.

        Returns
        -------
        torch.Tensor
            A torch tensor of zeros with the shape [batch_size, num_heads, seq_len]
        """

        output = torch.zeros(input_ids.shape[0], self.num_heads, input_ids.shape[1], device=input_ids.device)
        return output, output

    def forward(self, input_ids, attention_mask, **kwargs) -> Tuple[torch.Tensor, None]:
        """
        Evaluate the output of the model.

        Returns
        -------
        torch.Tensor([batch_size, reward_dim, seq_len])
            Value outputs of the model.
        torch.Tensor([batch_size, num_heads, seq_len])
            Q-values of the model. This class doesn't support Q-values, so it returns a tensor of zeros.
        """
        # Get the base model outputs
        base_model_outputs = [model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            **kwargs,
        ).hidden_states[-1] for model in self.models]

        # Process the outputs of the base models
        output = torch.stack([head(base_model_outputs[i]) for i, head in enumerate(self.output_heads)],
                              dim=2).squeeze(-1)

        assert output.shape == (input_ids.shape[0], input_ids.shape[1], self.num_heads),\
              f"output shape: {output.shape} wrong, should be {(input_ids.shape[0], input_ids.shape[1], self.num_heads)}"

        qs = torch.zeros(input_ids.shape[0], self.num_heads, input_ids.shape[1], device=input_ids.device)

        return output.permute(0,2,1), qs
    
class MultiModelTargetValueFunction(BaseMultiObjectiveValueFunction):
        
    def __init__(
        self,
        base_model: nn.Module,
        base_model_hidden_dim: int,
        num_heads: int,
        lora_config: LoraConfig,
        token_vocab_size: int,
        torch_dtype: torch.dtype = torch.float16,
        q_function: bool = False,
        polyak_coeff:float = 0.005
    ):
        """
        A Multi objective value function which uses an MLP with a single set of multiple output heads at the last layer
        to generate the value outputs for different losses. 
        """
        # Note: there isn't a unified interface to get the hidden dimensions from a model, which is why need the base_model_hidden_dim arg
        super().__init__(base_model=base_model,
                         base_model_hidden_dim=base_model_hidden_dim,
                         num_heads=num_heads,
                         lora_config=lora_config,
                         token_vocab_size=token_vocab_size,
                         torch_dtype=torch_dtype,
                         q_function=q_function)
        
        self.polyak_coeff = polyak_coeff
        
    def setup(self):

        # Create multiple models:
        self.learnt_models = nn.ModuleList([copy.deepcopy(self.base_model) for _ in range(self.num_heads)])

        head_hidden_dim = 2 * self.base_model_hidden_dim
        
        # Create multiple MLP Heads:
        self.learnt_output_heads = nn.ModuleList([MLPHead(
            input_dim=self.base_model_hidden_dim,
            hidden_dim=head_hidden_dim,
            output_dim=1,
        ).to(self.torch_dtype) for _ in range(self.num_heads)])

        # Create the target models
        self.target_models = nn.ModuleList([copy.deepcopy(self.base_model) for _ in range(self.num_heads)])

        # Create target MLP heads:
        self.target_output_heads = nn.ModuleList([MLPHead(
            input_dim=self.base_model_hidden_dim,
            hidden_dim=head_hidden_dim,
            output_dim=1,
        ).to(self.torch_dtype) for _ in range(self.num_heads)])

        # Freeze the target parameters, add target learnt mapping:
        polyak_update_layers_destinations = list()
        for name, param in self.named_parameters():
            if "target" in name:
                param.requires_grad = False
                polyak_update_layers_destinations.append(name)

        # self.polyak_update_mapping is q_hat_param : q_param
        self.polyak_update_mapping = {
            dest: (dest.replace("target", "learnt"))
            for dest in polyak_update_layers_destinations
        }


    def is_polyak_average(self):
        """
        Returns
        -------
        bool
            Whether the value function is polyak average-able or not, returns False by
            default in the parent class.
        """
        return True

    def get_number_outputs(self):
        return self.num_heads
    
    def get_target_values(self, input_ids, attention_mask, **kwargs):
        """
        Return q_hat as vector of zeros. This class does not support action value functions.

        Returns
        -------
        torch.Tensor
            A torch tensor of zeros with the shape [batch_size, num_heads, seq_len]
        """

        base_model_outputs = [model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            **kwargs,
        ).hidden_states[-1] for model in self.target_models]

        # Process the outputs of the base models
        v_hat = torch.stack([head(base_model_outputs[i]) for i, head in enumerate(self.target_output_heads)],
                              dim=2).squeeze(-1)
                
        assert not v_hat.requires_grad,\
            f"Target values should not require grad: {v_hat.requires_grad}"

        q_hat = torch.zeros(input_ids.shape[0], self.num_heads, input_ids.shape[1], device=input_ids.device)

        return v_hat.permute(0,2,1), q_hat

    def forward(self, input_ids, attention_mask, **kwargs) -> Tuple[torch.Tensor, None]:
        """
        Evaluate the output of the model.

        Returns
        -------
        torch.Tensor([batch_size, reward_dim, seq_len])
            Value outputs of the model.
        torch.Tensor([batch_size, num_heads, seq_len])
            Q-values of the model. This class doesn't support Q-values, so it returns a tensor of zeros.
        """
        # Get the base model outputs
        base_model_outputs = [model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            **kwargs,
        ).hidden_states[-1] for model in self.learnt_models]

        # Process the outputs of the base models
        output = torch.stack([head(base_model_outputs[i]) for i, head in enumerate(self.learnt_output_heads)],
                              dim=2).squeeze(-1)

        # Ensure the output is the correct shape
        assert output.shape == (input_ids.shape[0], input_ids.shape[1], self.num_heads),\
              f"output shape: {output.shape} wrong, should be {(input_ids.shape[0], input_ids.shape[1], self.num_heads)}"

        qs = torch.zeros(input_ids.shape[0], self.num_heads, input_ids.shape[1], device=input_ids.device)

        return output.permute(0,2,1), qs

    def update_target_weights(self):

        state_dict = self.state_dict()

        with torch.no_grad():
            # Update the target_model params
            for target_param, learnt_param in self.polyak_update_mapping.items():
                state_dict[target_param] = state_dict[target_param].data * (
                    1 - self.polyak_coeff
                ) + (self.polyak_coeff * state_dict[learnt_param].data)

            # Load the updated state dict
            self.load_state_dict(state_dict)

    
class MultiWeightedValueFunction(BaseMultiObjectiveValueFunction):
        
    def __init__(
        self,
        base_model: nn.Module,
        base_model_hidden_dim: int,
        num_heads: int,
        lora_config: LoraConfig,
        token_vocab_size: int,
        torch_dtype: torch.dtype = torch.float16,
        q_function: bool = False
    ):
        """
        A Multi objective value function which uses an MLP with a single set of multiple output heads at the last layer
        to generate the value outputs for different losses. 
        """
        # Note: there isn't a unified interface to get the hidden dimensions from a model, which is why need the base_model_hidden_dim arg
        super().__init__(base_model=base_model,
                         base_model_hidden_dim=base_model_hidden_dim,
                         num_heads=num_heads,
                         lora_config=lora_config,
                         token_vocab_size=token_vocab_size,
                         torch_dtype=torch_dtype,
                         q_function=q_function)

    def setup(self):

        head_hidden_dim = 2 * self.base_model_hidden_dim
        
        # Set up the learnt heads
        self.output_head = MLPHead(
            input_dim=self.base_model_hidden_dim,
            hidden_dim=head_hidden_dim,
            output_dim=1,
        ).to(self.torch_dtype)

        # Add the LoRA adapters  
        assert self.lora_config is not None, "LoRA config must be provided for MultiWeightedValueFunction"

        # Create a LoRA adapter for each head
        self.learnt_lora_adapter_names = list()
        for i in range(self.num_heads):

            learnt_lora_adapter_name = f"learnt_lora_adapter_{i}"
            self.model = get_peft_model(
                model=self.base_model,  # type: ignore
                peft_config=self.lora_config,
                adapter_name=learnt_lora_adapter_name,
            )

            self.learnt_lora_adapter_names.append(learnt_lora_adapter_name)

        # Enable the value function LoRA adapter
        self.model.set_adapter(self.learnt_lora_adapter_names[0])

    def get_number_outputs(self):
        return self.num_heads
    
    def get_target_values(self, input_ids, attention_mask, **kwargs):
        """
        Return q_hat as vector of zeros. This class does not support action value functions.

        Returns
        -------
        torch.Tensor
            A torch tensor of zeros with the shape [batch_size, num_heads, seq_len]
        """

        return torch.zeros(input_ids.shape[0], self.num_heads, input_ids.shape[1], device=input_ids.device)

    def forward(self, input_ids, attention_mask, **kwargs) -> Tuple[torch.Tensor, None]:
        """
        Evaluate the output of the model.

        Returns
        -------
        torch.Tensor([batch_size, reward_dim, seq_len])
            Value outputs of the model.
        torch.Tensor([batch_size, num_heads, seq_len])
            Q-values of the model. This class doesn't support Q-values, so it returns a tensor of zeros.
        """
        # For each LoRA adapter:
        outputs = list()
        for i, lora_adapter in enumerate(self.learnt_lora_adapter_names): 

            self.model.set_adapter(lora_adapter)
            
            # Get the base model outputs
            base_model_outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                **kwargs,
            ).hidden_states[-1]

            # Get the output of the learnt heads
            outputs.append(self.output_head(base_model_outputs))
            
        outputs = torch.concat(outputs, dim=2)

        qs = torch.zeros(input_ids.shape[0], self.num_heads, input_ids.shape[1], device=input_ids.device)

        return outputs.permute(0,2,1), qs

class CBFMultiWeightedValueFunction(MultiWeightedValueFunction):
    """
    Control Barrier Function (CBF) Multi-Weighted Value Function implementation.
    This class extends MultiWeightedValueFunction to support different types of heads,
    specifically Control Barrier Function (CBF) heads and End-of-Sequence (EOS) heads.

    The CBF heads use sigmoid activation with optional scaling and layer normalization,
    while EOS heads use standard MLP outputs. Each head has its own LoRA adapter for
    fine-tuning.

    Parameters
    ----------
    base_model : nn.Module
        The underlying transformer model to be used
    base_model_hidden_dim : int
        Hidden dimension size of the base model
    num_heads : int
        Number of output heads
    lora_config : LoraConfig
        Configuration for LoRA fine-tuning
    token_vocab_size : int
        Size of the token vocabulary
    torch_dtype : torch.dtype, optional
        Data type for torch tensors, defaults to torch.float16
    head_types : Optional[List[Literal['cbf', 'eos']]], optional
        List specifying the type of each head ('cbf' or 'eos'),
        defaults to ['cbf'] + ['eos'] * (num_heads - 1)
        must have same length as num_heads
    q_function : bool, optional
        Whether to use as Q-function, defaults to False
    scale_sigmoid : bool, optional
        Whether to scale sigmoid outputs, defaults to True
        if True, sigmoid outputs are scaled by 1 / (1 - discount_factor)
    discount_factor : float, optional
        Discount factor for future rewards, defaults to 0.95
    use_layer_norm : bool, optional
        Whether to use layer normalization, defaults to True
    use_xavier_init : bool, optional
        Whether to use Xavier initialization, defaults to True
    """

    def __init__(
        self,
        base_model: nn.Module,
        base_model_hidden_dim: int,
        num_heads: int,
        lora_config: LoraConfig,
        token_vocab_size: int,
        torch_dtype: torch.dtype = torch.float16,
        head_types: Optional[List[Literal['cbf', 'eos']]]=[],
        q_function: bool = False,
        scale_sigmoid: bool = True,
        discount_factor: float = 0.95,
        use_layer_norm: bool = True,
        use_xavier_init: bool = True,
    ):
        self.scale_sigmoid = scale_sigmoid
        self.discount_factor = discount_factor
        self.use_layer_norm = use_layer_norm
        self.use_xavier_init = use_xavier_init
        if not head_types:
            head_types = ['cbf'] + ['eos'] * (num_heads - 1)
        self.head_types = head_types
        assert len(self.head_types) == num_heads, "Number of head types must match the number of heads"
        
        super().__init__(
            base_model=base_model,
            base_model_hidden_dim=base_model_hidden_dim,
            num_heads=num_heads,
            lora_config=lora_config,
            token_vocab_size=token_vocab_size,
            torch_dtype=torch_dtype,
            q_function=q_function,
        )

    def setup(self):
        head_hidden_dim = 2 * self.base_model_hidden_dim
        
        # Set up the learnt heads with sigmoid activation
        if 'cbf' in self.head_types:
            self.cbf_output_head = SigmoidMLPHead(
                input_dim=self.base_model_hidden_dim,
                hidden_dim=head_hidden_dim,
                output_dim=1,
                scale_sigmoid=self.scale_sigmoid,
                discount_factor=self.discount_factor,
                use_layer_norm=self.use_layer_norm,
                use_xavier_init=self.use_xavier_init,
            ).to(self.torch_dtype)
        else:
            self.cbf_output_head = None
        
        if 'eos' in self.head_types:
            self.eos_output_head = MLPHead(
                input_dim=self.base_model_hidden_dim,
                hidden_dim=head_hidden_dim,
                output_dim=1,
            ).to(self.torch_dtype)
        else:
            self.eos_output_head = None

        # Add the LoRA adapters  
        assert self.lora_config is not None, "LoRA config must be provided for MultiWeightedValueFunction"

        self.learnt_lora_adapter_names = list()
        for i in range(self.num_heads):
            learnt_lora_adapter_name = f"learnt_lora_adapter_{i}"
            self.model = get_peft_model(
                model=self.base_model,  # type: ignore
                peft_config=self.lora_config,
                adapter_name=learnt_lora_adapter_name,
            )
            self.learnt_lora_adapter_names.append(learnt_lora_adapter_name)

        # Enable the value function LoRA adapter
        self.model.set_adapter(self.learnt_lora_adapter_names[0])
    
    def forward(self, input_ids, attention_mask, **kwargs) -> Tuple[torch.Tensor, None]:
        # For each LoRA adapter:
        outputs = list()
        for lora_adapter, head_type in zip(self.learnt_lora_adapter_names, self.head_types): 

            self.model.set_adapter(lora_adapter)
            
            # Get the base model outputs
            base_model_outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                **kwargs,
            ).hidden_states[-1]

            # Get the output of the learnt heads
            if head_type == 'cbf':
                outputs.append(self.cbf_output_head(base_model_outputs))
            elif head_type == 'eos':
                outputs.append(self.eos_output_head(base_model_outputs))
            
        outputs = torch.concat(outputs, dim=2)

        qs = torch.zeros(input_ids.shape[0], self.num_heads, input_ids.shape[1], device=input_ids.device)

        return outputs.permute(0,2,1), qs


class MultiModelILQLValueFunction(BaseMultiObjectiveValueFunction):

    def __init__(
        self,
        base_model: nn.Module,
        base_model_hidden_dim: int,
        num_heads: int,
        lora_config: LoraConfig,
        token_vocab_size: int,
        torch_dtype: torch.dtype = torch.float16,
        q_function: bool = False,
        polyak_coeff: float = 0.005
    ):
        """
        A class which implements the ILQL value function architecture. This consists of a target model and a learnt model,
        both models have three heads, one for the value function and two q function outputs
        
        Parameters
        ----------
        base_model : nn.Module
            Underlying model to be used for the value function module.
        base_model_hidden_dim : int
            Hidden dimension of the output head MLPs.
        lora_config : LoraConfig
            LoraConfig object that contains the configuration for the LoRA adapters.
        token_vocab_size : int
            Size of the token vocabulary, used as the output dimension of the heads.
        torch_dtype : torch.dtype, optional
            torch dtype to which the heads are mapped, by default torch.float16
        q_function : bool, optional
            Weather the architecture creates a q function or not, by default False
        """

        super().__init__(base_model=base_model,
                            base_model_hidden_dim=base_model_hidden_dim,
                            num_heads=num_heads,
                            lora_config=lora_config,
                            token_vocab_size=token_vocab_size,
                            torch_dtype=torch_dtype,
                            q_function=q_function)

        self.polyak_coeff = polyak_coeff
        self.setup()

    def is_polyak_average(self):
        """
        Returns
        -------
        True
            This class should have polyak averaging applied.
        """
        return True

    def setup(self):
        """
        Setup the model architecture, including different output heads and LoRA adapters.
        """
        
        # Create multiple models:
        self.learnt_models = nn.ModuleList([copy.deepcopy(self.base_model) for _ in range(self.num_heads)])
        
        # Create multiple MLP Heads:
        self.learnt_output_value_head = nn.ModuleList([
            nn.Linear(self.base_model_hidden_dim, 1).to(self.torch_dtype) for _ in range(self.num_heads)])

        self.learnt_output_q_head_1 = nn.ModuleList([
            nn.Linear(self.base_model_hidden_dim, self.token_vocab_size).to(self.torch_dtype) for _ in range(self.num_heads)])

        self.learnt_output_q_head_2 = nn.ModuleList([
            nn.Linear(self.base_model_hidden_dim, self.token_vocab_size).to(self.torch_dtype) for _ in range(self.num_heads)])

        # Create the target models
        self.target_models = nn.ModuleList([copy.deepcopy(self.base_model) for _ in range(self.num_heads)])

        # Create target MLP heads:
        self.target_output_value_head = nn.ModuleList([
            nn.Linear(self.base_model_hidden_dim, 1).to(self.torch_dtype) for _ in range(self.num_heads)])

        self.target_output_q_head_1 = nn.ModuleList([
            nn.Linear(self.base_model_hidden_dim, self.token_vocab_size).to(self.torch_dtype) for _ in range(self.num_heads)])

        self.target_output_q_head_2 = nn.ModuleList([
            nn.Linear(self.base_model_hidden_dim, self.token_vocab_size).to(self.torch_dtype) for _ in range(self.num_heads)])

        # Freeze the target parameters, add target learnt mapping:
        polyak_update_layers_destinations = list()
        for name, param in self.named_parameters():
            if "target" in name:
                param.requires_grad = False
                polyak_update_layers_destinations.append(name)

        # self.polyak_update_mapping is q_hat_param : q_param
        self.polyak_update_mapping = {
            dest: (dest.replace("target", "learnt"))
            for dest in polyak_update_layers_destinations
        }

    def get_number_outputs(self):
        return self.num_heads
    
    def get_target_values(self, input_ids, attention_mask, **kwargs):
        """
        Return the value from the target network.
        Value Head - [batch_size, num_heads, seq_len]
        Q Head - [batch_size, num_heads, seq_len, token_vocab_size, 2]

        Parameters
        ----------
        input_ids : _type_
            _description_
        attention_mask : _type_
            _description_

        Returns
        -------
        _type_
            _description_
        """

        base_model_outputs = [model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            **kwargs,
        ).hidden_states[-1] for model in self.target_models]

        # Process the outputs of the base models
        v_hat = torch.stack([head(base_model_outputs[i]) for i, head in enumerate(self.target_output_value_head)],
                              dim=2).squeeze(-1)
                
        assert not v_hat.requires_grad,\
            f"Target values should not require grad: {v_hat.requires_grad}"

        q_hat1 = torch.stack([head(base_model_outputs[i]) for i, head in enumerate(self.target_output_q_head_1)], dim=-1)
        q_hat2 = torch.stack([head(base_model_outputs[i]) for i, head in enumerate(self.target_output_q_head_2)], dim=-1)

        q_hat = torch.stack([q_hat1, q_hat2], dim=-1) # batch_size, seq_len, token_vocab_size, num_heads, 2

        assert not q_hat.requires_grad,\
            f"Target values should not require grad: {q_hat.requires_grad}"
        assert q_hat.shape == (input_ids.shape[0], input_ids.shape[1], self.token_vocab_size, self.num_heads, 2),\
            f"q_hat shape: {q_hat.shape} wrong, should be {(input_ids.shape[0], input_ids.shape[1], self.token_vocab_size, self.num_heads, 2)}"

        # Current version of the code doesn't return v_hat TODO: update this
        return q_hat.permute(0,3,1,2,4)

    def update_target_weights(self):

        state_dict = self.state_dict()

        with torch.no_grad():
            # Update the target_model params
            for target_param, learnt_param in self.polyak_update_mapping.items():
                state_dict[target_param] = state_dict[target_param].data * (
                    1 - self.polyak_coeff
                ) + (self.polyak_coeff * state_dict[learnt_param].data)

            # Load the updated state dict
            self.load_state_dict(state_dict)

    def forward(self, input_ids, attention_mask, **kwargs):
        """
        Evaluate the output of the model.

        Returns
        -------
        torch.Tensor([batch_size, reward_dim, seq_len])
            Value outputs of the model.    
        """
        base_model_outputs = [model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            **kwargs,
        ).hidden_states[-1] for model in self.learnt_models]

        # Process the outputs of the base models
        v = torch.stack([head(base_model_outputs[i]) for i, head in enumerate(self.learnt_output_value_head)],
                        dim=2).squeeze(-1)
                
        q1 = torch.stack([head(base_model_outputs[i]) for i, head in enumerate(self.learnt_output_q_head_1)], dim=-1)
        q2 = torch.stack([head(base_model_outputs[i]) for i, head in enumerate(self.learnt_output_q_head_2)], dim=-1)

        q = torch.stack([q1, q2], dim=-1) # batch_size, seq_len, token_vocab_size, num_heads, 2

        assert v.shape == (input_ids.shape[0], input_ids.shape[1], self.num_heads),\
            f"output shape: {v.shape} wrong, should be {(input_ids.shape[0], input_ids.shape[1], self.num_heads)}"

        assert q.shape == (input_ids.shape[0], input_ids.shape[1], self.token_vocab_size, self.num_heads, 2),\
            f"q_hat shape: {q.shape} wrong, should be {(input_ids.shape[0], input_ids.shape[1], self.token_vocab_size, self.num_heads, 2)}"

        # Current version of the code doesn't return v_hat TODO: update this
        return v.permute(0,2,1), q.permute(0,3,1,2,4)