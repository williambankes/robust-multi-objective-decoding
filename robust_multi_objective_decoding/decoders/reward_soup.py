import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM
from typing import Tuple


class RewardSoupDecoder(nn.Module):

    def __init__(self, 
                 models: list[AutoModelForCausalLM],
                 weights: list[float]):
        
        """
        Combines a list of models together and runs the forward pass on the weights.

        Args:
            models (list[AutoModelForCausalLM]): List of models to combine
            weights (list[float]): List of weights to apply to the models
               
        
        """
        
        super().__init__()

        # Process models and weights:
        self.weights = self._process_weights(torch.tensor(weights))

        # Combine the models:
        self.weights = self.weights.to(models[0].device)

        if self._process_models(models):

            combined_weights = self._combine_models(models)
            models[0].load_state_dict(combined_weights)
            self.model = models[0]

            # Delete the other models to save memory:
            del models[1]

        else:
            raise ValueError(f'Models: {models} are not valid')

    def _process_models(self, models: list[AutoModelForCausalLM]) -> bool:
        """
        Process the models to ensure they are valid.

        Args:
            models (list[AutoModelForCausalLM]): List of models to process

        Returns:
            bool: True if models are valid, False otherwise
        """
        # Check the models have the same architecture:
        if not all(model.config.architectures == models[0].config.architectures for model in models):
            print(f'Models: {models} have different architectures')
            return False

        return True
        

    def _process_weights(self, weights:Tuple[float, float]) -> torch.Tensor:

        # Check the weights are valid and sum to 1:
        assert torch.sum(weights) == 1, f'Weights: {weights} must sum to 1'

        return torch.tensor(weights)

    def _combine_models(self, models):

        assert len(models) > 1, f'Must provide more than one model to combine, current len: {len(models)}'
        assert len(models) == len(self.weights), \
            f'Must provide the same number of weights: {self.weights} as models, num models: {len(models)}'
        
        # Ensure the models are on the same device:
        model_devices = models[0].device
        for i, model in enumerate(models):
            assert model.device == model_devices,\
                f'Model {i} device: {model.device} not the same as model 0 device: {model_devices}'

        # Get state dict of first model
        final_state_dict = models[0].state_dict()

        # Get state dict of second model
        second_state_dict = models[1].state_dict()

        # Combine state dicts using weights
        for name in final_state_dict:
            final_state_dict[name] = self.weights[0] * final_state_dict[name] + \
                                   self.weights[1] * second_state_dict[name]

        return final_state_dict

    def generate(self, **kwargs):
        return self.model.generate(**kwargs)
