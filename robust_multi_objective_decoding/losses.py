import torch
import torch.nn as nn
from abc import abstractmethod


class Loss(nn.Module):
    def __init__(self, weight: torch.Tensor, *args, **kwargs):
        """
        Parameters
        ----------
        weight : torch.Tensor, optional
            Weight of the loss function, by default torch.tensor(1.)
        """
        super().__init__()
        self.weight = weight

    @abstractmethod
    def get_name(self):
        pass

    @abstractmethod
    def forward(self, q, v, r, q_next, v_next):
        """
        Define forward pass method that defines the loss
        """
        pass


class CompositeLoss(Loss):
    def __init__(
        self, loss_functions: list[Loss], weight: torch.Tensor = torch.tensor(1.0)
    ):
        """
        Combine different loss functions together

        Parameters
        ----------
        weight : torch.Tensor
            The weight this loss should be given in the total multi objective loss
        loss_functions : List[Loss]
            A list of Loss function to be added together
        """

        super().__init__(weight)
        self.loss_functions = loss_functions

    def get_name(self):
        loss_names = [loss.get_name() for loss in self.loss_functions]
        return "_".join(loss_names)

    def forward(self, input_ids, q, v, r, q_hat, v_next):
        """
        Combine the losses from the different loss functions

        Parameters
        ----------
        q : torch.Tensor
            token level action value function predictions q(x_t, a_t)
        v : torch.Tensor
            token level value function predictions for v(x_{t})
        r : torch.Tensor
            token level rewards r_t
        q_next : torch.Tensor
            token level action value function predictions q(x_{t+1}, a_{t+1})
        v_next : torch.Tensor
            token level value function predictions for v(x_{t+1})

        Returns
        -------
        torch.Tensor
            loss term
        """

        loss = 0
        for loss_function in self.loss_functions:
            loss += loss_function(
                input_ids=input_ids, q=q, v=v, r=r, q_hat=q_hat, v_next=v_next
            )

        return self.weight * loss


class TemporalDifferenceLoss(Loss):
    def __init__(self, gamma, weight: torch.Tensor = torch.tensor(1.0)):
        """
        Temporal difference loss function,

        (Q(s, a) - [r(s, a) + gamma * V(s')])^2,

        with gamma in [0, 1].
        """
        super().__init__(weight=weight)
        self.gamma = gamma

    def get_name(self):
        return "TD_Loss"

    def forward(
        self,
        r: torch.Tensor,
        v: torch.Tensor,
        v_next: torch.Tensor,
        **kwargs,
    ):
        """
        Parameters
        ----------
        r : torch.Tensor[batch_size, seq_len]
            token level rewards r_t
        v : torch.Tensor[batch_size, seq_len]
            token level value function predictions for v(x_{t})
        v_next : torch.Tensor[batch_size, seq_len - 1]
            token level value function predictions for v(x_{t+1})

        Returns
        -------
        torch.Tensor
            loss term
        """

        V = v[:, :-1]  # Everything except the last

        # The last token is the reward - detach from grad tree
        v_next_det = v_next.detach()
        v_next_det[:, -1] = r[:, -1]

        return self.weight * (V - v_next_det) ** 2


class CDQLoss(Loss):
    def __init__(self, weight: torch.Tensor = torch.tensor(-1.0)):
        """
        CDQ Regularization term


        """
        super().__init__(weight=weight)

    def get_name(self):
        return "CDQ_Loss"

    def forward(self, q: torch.Tensor, input_ids: torch.Tensor, **kwargs):
        """
        Parameters
        ----------
        q: torch.Tensor[batch_size, seq_len, vocab_size, 2]
            The token level action value function predictions q(x_t, a_t)
        input_ids: torch.Tensor[batch_size, seq_len]
            The token input_ids
        Returns
        -------
        torch.Tensor
            loss term
        """

        selected_Q1 = torch.gather(
            nn.Softmax(dim=-1)(q[:, :-1, :, 0]),
            dim=-1,
            index=input_ids[..., 1:].unsqueeze(2),
        ).squeeze(2)
        selected_Q2 = torch.gather(
            nn.Softmax(dim=-1)(q[:, :-1, :, 0]),
            dim=-1,
            index=input_ids[..., 1:].unsqueeze(2),
        ).squeeze(2)

        return self.weight * (selected_Q1 + selected_Q2)


class MonteCarloLoss(Loss):
    def __init__(self, weight: torch.Tensor = torch.tensor(1.0)):
        """
        A Loss on the last token of the reward sequence.
        """
        super().__init__(weight)

    def get_name(self):
        return "Monte_Carlo_Loss"

    def forward(self, v: torch.Tensor, r: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Parameters
        ----------
        r : torch.Tensor[batch_size, seq_len]
            token level rewards r_t
        v : torch.Tensor[batch_size, seq_len]
            token level value function predictions for v(x_{t})

        Returns
        -------
        torch.Tensor
            loss term
        """

        return self.weight * (v[..., 1:] - r[:, -1].unsqueeze(1)) ** 2
