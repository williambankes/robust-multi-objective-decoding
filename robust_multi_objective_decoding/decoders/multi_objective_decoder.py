import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM
from typing import Tuple

class MultiObjectiveDecoder(nn.Module):

    def __init__(self, models, weights):
        
        super().__init__()

        self.weights = self._process_weights(torch.tensor(weights))

        if self._check_models(models, weights):
            self.models = models

        # Move the weights to the same device as the models:
        self.weights = self.weights.to(self.models[0].device)
        self.eos_token_id = self.models[0].config.eos_token_id
        if isinstance(self.eos_token_id, list):
            self.eos_token_id = self.eos_token_id[0]
        
    def _check_models(self, models, weights):

        # Ensure models are on the same device:
        model_devices = models[0].device
        for i, model in enumerate(models):
            assert model.device == model_devices,\
                f'Model {i} device: {model.device} not the same as model 0 device: {model_devices}'

        # Ensure the models and weights list match
        assert len(models) == len(weights), \
            f'num_models:{len(models)} does not match num_weights:{len(weights)}'
        
        return True

    def _process_weights(self, weights):
        # TODO: re-write it's a bit silly...

        # Check the weights are valid and sum to 1:
        assert torch.sum(weights) == 1, f'Weights: {weights} must sum to 1'

        return weights

    def _get_reference_logits(self, model_input: dict) -> torch.Tensor:
        """Get the logits from the reference models"""

        return torch.stack([m(**model_input)["logits"][:, -1, :] for m in self.models])
    
    def _get_adjusted_logits(self, reference_logits: torch.Tensor) -> torch.Tensor:
        """
        Combine the logits from the reference models together using the MOD equation:
        https://arxiv.org/abs/2406.18853

        where we let f(x) = xlog(x) resulting in a weighted sum of the logits
        """

        return torch.sum(reference_logits * self.weights[:, None, None], dim=0)

    def generate(self, input_ids: torch.Tensor, 
                attention_mask:torch.Tensor,
                max_new_tokens:int,
                return_dict_in_generate:bool=False,
                **kwargs):


        batch_size = input_ids.shape[0]
        generated_ids = input_ids.clone()
        sequence_is_finished = torch.zeros(batch_size, dtype=torch.bool).to(input_ids.device)

        with torch.no_grad():
            for _ in range(max_new_tokens):
                unfinished_token_id_sequences = generated_ids[~sequence_is_finished]

                # Get logits from the reference model
                model_input = {
                    "input_ids": unfinished_token_id_sequences,
                    "attention_mask": attention_mask[~sequence_is_finished]
                }
                reference_logits = self._get_reference_logits(model_input)

                # Check ref logits size: -> this shouldn't be the batch size but the number of unfinished sequences?
                assert reference_logits.shape == (len(self.models), (~sequence_is_finished).sum(), reference_logits.shape[-1]),\
                    f'Logits shape: {reference_logits.shape} does not match expected shape: {(len(self.models), batch_size, reference_logits.shape[-1])}, input_ids shape: {input_ids.shape}, model_input shape: {model_input["input_ids"].shape}'

                # Adjust the logits based on the decoding strategy
                adjusted_logits = self._get_adjusted_logits(reference_logits)

                # Check the adjusted logits shape:
                assert adjusted_logits.shape == ((~sequence_is_finished).sum(), reference_logits.shape[-1]),\
                    f'Adjusted logits shape: {adjusted_logits.shape} does not match expected shape: {(batch_size, reference_logits.shape[-1])}'

                # Sample the next token id from the probability distribution
                probs = torch.softmax(adjusted_logits, dim=-1)
                next_token_ids = torch.multinomial(probs, num_samples=1).squeeze()

                # Fill in the next token ids for the unfinished sequences with the generated ids,
                # and fill in the EOS token id for the finished sequences
                all_next_token_ids = torch.full(
                    (batch_size,), fill_value=self.eos_token_id
                ).to(input_ids.device)

                all_next_token_ids[~sequence_is_finished] = next_token_ids
                generated_ids = torch.hstack(
                    [generated_ids, all_next_token_ids.unsqueeze(-1)]
                )

                # Update sequence_is_finished based on whether the EOS token was generated
                sequence_is_finished = all_next_token_ids == self.eos_token_id

                # Update attention_mask:
                new_chunk = torch.ones(batch_size, 1, dtype=torch.long).to(attention_mask.device)
                new_chunk[sequence_is_finished] = 0 # unmask finished sequences
                attention_mask = torch.concat([attention_mask,
                                               new_chunk], dim=-1)

                # Stop if all sequences are finished
                if sequence_is_finished.all():
                    break

        if return_dict_in_generate: # We can add other values here
            return {'generated_ids': generated_ids}
        else:
            return generated_ids