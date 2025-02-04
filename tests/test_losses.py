import torch
from robust_multi_objective_decoding.losses import (
    TemporalDifferenceLoss,
    ILQLExpectileLoss
)

def test_temporal_difference_loss():

    r = torch.tensor([[0.0, 1.0]])
    q = torch.tensor([[0.0, 1.0]])
    v = torch.tensor([[0.0, 0.0, 1.0]])
    gamma=0.95

    # IQL loss
    unmasked_loss = 0
    for i in range(2):
        unmasked_loss += (r[:, i] + gamma * v[:, i + 1] - q[:, i]) ** 2

    assert not torch.isnan(unmasked_loss)

    loss = TemporalDifferenceLoss(gamma=0.95)
    test_value = loss(v_next=v[:, 1:], q=q, r=r)

    assert test_value.sum() == unmasked_loss, 'values not the same'