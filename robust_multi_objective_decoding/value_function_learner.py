from typing import List, Dict, Any

import pytorch_lightning as pl
import torch
from robust_multi_objective_decoding.multi_objective_value_function import BaseMultiObjectiveValueFunction


class MultiObjectiveValueFunctionLearner(pl.LightningModule):

    def __init__(self,
                base_value_function_module: BaseMultiObjectiveValueFunction,
                losses: List[torch.nn.Module],
                learning_rate: float = 1e-4,
                update_target_model_every: int = 10,
                mask_token: int = -100,
                **kwargs):
        
        """
        Pytorch Lightning Learner class for multi-objective value functions
        for LLM decoding.

        Parameters
        ----------
        base_value_function_module: MultiObjectiveValueFunctionModule
            A base model which outputs multiple heads and different outputs
        losses: List[Loss]
            A list of Loss objects from robust_multi_objective_decoding.losses, this should be the same len as the number 
            of MultiObjectiveValueFunctionModule heads
        learning_rate: float
            Learning rate of grad descent
        update_q_hat_every: int
            Which steps to update q_hat component of the MultiObjectiveValueFunctionModule heads.
        mask_token: int
            The token which mask the padding + prompt in the prompt_response data        
        """
        
        super().__init__()

        self.losses = losses
        self.base_value_function_module = base_value_function_module
        self.learning_rate = 1e-4
        self.update_target_model_every = update_target_model_every
        self.is_polyak_average = base_value_function_module.is_polyak_average()
        self.mask_token = -100

    def process_batch(self, batch:Dict):
        """
        Given a batch, check the batch is consistent with the model,
        check the batch is consistent with itself and then return the
        most relevant components.

        Parameters
        ----------
        batch : dict
            Batch from the dataloader with tokenized sequence of inputs, attention mask,
            and tokenwise_safety_labels.
        Returns
        -------
        rewards : torch.Tensor
            safety label vector.
        vs : torch.tensor
            values calculated from base model with value head.
        qs : torch.Tensor
            q-values calculated from base model with q-value head.
        v_hats: torch.Tensor
            Value function predictions from target network.
        q_hats : torch.Tensor
            q-values calculated from base model with frozen weights.               
        """

        num_losses = len(self.losses)
        
        # Ensure the losses, model numbers are consistent with the data received in each batch 
        assert num_losses == self.base_value_function_module.get_number_outputs(),\
            f'The number of losses: {len(self.losses)} must equal the\
              number of model outputs: {self.base_value_function_module.get_number_outputs()}'
        
        # Get the rewards
        rewards = batch['rewards'][..., 1:]
        mask = (rewards != self.mask_token)

        # Ensure the mask token is uniform across different reward outputs:
        mask_sum_across_rewards_dim = mask.sum(dim=1)
        assert ((mask_sum_across_rewards_dim == 0.0) + (mask_sum_across_rewards_dim == num_losses)).all(),\
            f'mask: {mask} must be consistent across the num_reward dimension'

        # Check rewards for entirely masked rows:
        # assert not (mask[:,0,:].sum(dim=1) == 0).any(),\
        #     f"Entirely masked rows in safety labels for idx {batch['idx']}"

        # Check rewards for entire batch:
        assert not ((mask[:,0,:]).sum(dim=1) == 0).all(),\
            f"Entire batch of fully masked rows: {batch['idx']}, mask shape: {mask.shape},\
                attention_mask: {(batch['attention_mask'] != 0).sum()}, input_ids shape: {batch['input_ids'].shape}"
        
        # Check rewards for masked tokens at the end of the row:
        assert (rewards[..., -1] != self.mask_token).any(), \
            f"Last token in the reward sequence must not be masked: {rewards[..., -1]}"

        # Get the Input ids and ensure they're at least 2D
        input_ids = torch.atleast_2d(batch["input_ids"])
        attention_mask = batch.get("attention_mask", None)

        # Get V and Q values from the base_value_function_module:
        v, q = self.base_value_function_module(input_ids, attention_mask=attention_mask)

        # Get Q_hat values from the base_value_function_module:
        with torch.no_grad():
            v_hat, q_hat = self.base_value_function_module.get_target_values(
                input_ids, attention_mask=attention_mask
            )

        return rewards, v, q, v_hat, q_hat
        
    def process_losses(self, input_ids, rewards, vs, qs, q_hats, v_hats, vs_next, split) -> torch.Tensor:
    
        """
        Process the various losses and rewards, mask non-response tokens and 
        normalize by sequence length and batch_size.

        rewards: torch.Tensor([Batch, Reward, Sequence Length])
            Tensor of rewards for different objectives
        vs: torch.Tensor([Batch, Reward, Sequence Length + 1])
            Tensor of values for the different alignment objectives
        qs: torch.Tensor([Batch, Reward, Sequence Length])
            Tensor of the q-values for different alignment objectives
        v_hats: torch.Tensor([Batch, Reward, Sequence Length])
            Value function predictions from target network
        q_hats: torch.Tensor([Batch, Reward, Sequence Length])
            Action value function predictions from the target network
        split: str
            Which training split the process_batch is being run in ['train', 'val', 'test']

        Returns
        --------
        torch.Tensor
            Loss
        """
    
        # Ensure training split correct:
        assert split in ['train', 'val', 'test'],\
            f'split: {split}, must be either "train", "val" or "test"'

        # Init loss output of shape [batch_size, sequence_length - 1]
        loss_term = torch.zeros_like(vs[:,0,1:])
        
        # For each type of loss add this to the cumulative loss
        for i, loss in enumerate(self.losses):

            specific_loss_term = loss(input_ids=input_ids,
                                    r=rewards[:,i,:],
                                    v=vs[:,i,:],
                                    q=qs[:,i,:],
                                    v_hat=v_hats[:,i,:],
                                    q_hat=q_hats[:,i,:],
                                    v_next=vs_next[:,i,:])
            
            # TODO: Loss is logged before masking and tokenwise normalisation -> fix 
            self.log(f"{split}_{i}_{loss.get_name()}",
                    specific_loss_term.detach().mean(),
                    on_epoch=True,
                    sync_dist=True)
            
            loss_term += specific_loss_term
            
        # We have checked prev that mask is consistent across rewards hence [reward_dim = 0]
        mask = (rewards != self.mask_token)[:, 0, :] 
        
        # Mask out non-response elements:
        loss_term *= mask

        # Valid mask rows:
        valid_rows = mask.sum(dim=1) > 0 

        # Normalise the loss by the number of valid transitions:
        loss = (loss_term[valid_rows, :]) / mask.sum(dim=1).unsqueeze(1)[valid_rows, :]

        # Normalise the loss by the batch size:
        loss = (loss / valid_rows.sum()).sum()

        return loss
          

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
                
        rewards, vs, qs, v_hats, q_hats = self.process_batch(batch)
        vs_next = vs[..., 1:]

        loss = self.process_losses(input_ids=batch['input_ids'],
                                    rewards=rewards,
                                    vs=vs,
                                    qs=qs,
                                    v_hats=v_hats,
                                    q_hats=q_hats,
                                    vs_next=vs_next,
                                    split='train')
        
        # Log the training loss
        self.log('train_loss', loss.detach(), on_epoch=True, sync_dist=True)

        return loss
    
    def validation_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        rewards, vs, qs, v_hats, q_hats = self.process_batch(batch)
        vs_next = vs[..., 1:]

        loss = self.process_losses(input_ids=batch['input_ids'],
                                    rewards=rewards,
                                    vs=vs,
                                    qs=qs,
                                    v_hats=v_hats,
                                    q_hats=q_hats,
                                    vs_next=vs_next,
                                    split='val')
        
        # Log the training loss
        self.log('val_loss', loss.detach(), on_epoch=True, sync_dist=True)

        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)  # type: ignore
        return optimizer

    def on_after_backward(self):
        """
        Apply Polyak averaging to the Q_hat head and the Q_hat LoRA weights.
        """
        if self.global_step % self.update_target_model_every == 0 and self.is_polyak_average:
            self.base_value_function_module.update_target_weights()