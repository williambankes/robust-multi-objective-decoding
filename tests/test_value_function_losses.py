import torch
from robust_multi_objective_decoding.value_function_losses import (
    temporal_difference_loss,
    expectile_loss,
    cbf_monte_carlo_loss,
    unsafe_norm_loss,
    delayed_unsafe_value_loss,
    gamma_scaled_loss,
    safe_unsafe_combined_loss,
    tokenwise_monte_carlo_loss,
)

############## LOSS FUNCTION TESTS ##############


def test_delayed_unsafe_value_loss():
    """Test the delayed unsafe value loss function:
    1. Should identify sequences that were safe until near end but became unsafe
    2. Should calculate correct target values based on sequence length and gamma
    3. Should only apply loss to sequences with safe history and unsafe final token
    """
    # Test case with both delayed unsafe and other sequences
    gamma = 0.9
    v = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])  # Value embeddings
    r = torch.tensor(
        [[1.0, 0.0], [0.0, 0.0]]
    )  # First sequence: safe->unsafe, Second: all unsafe

    loss = delayed_unsafe_value_loss(r, v, gamma)

    # Calculate expected loss for first sequence (delayed unsafe)
    seq_len = 2
    target_value = (1 - gamma**seq_len) / (1 - gamma)
    expected_loss1 = torch.nn.functional.relu(target_value - v[0])

    # Second sequence should have zero loss (no safe history)
    expected_loss2 = torch.zeros_like(v[1])

    expected_loss = torch.stack([expected_loss1, expected_loss2])

    assert torch.allclose(loss, expected_loss)

    # Test case with all safe sequences
    r = torch.tensor([[1.0, 1.0], [1.0, 1.0]])
    loss = delayed_unsafe_value_loss(r, v, gamma)
    assert torch.allclose(loss, torch.zeros_like(v))

    # Test case with all unsafe sequences
    v = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])  # Value embeddings
    r = torch.tensor([[0.0, 0.0], [0.0, 0.0]])  # All sequences unsafe

    loss = delayed_unsafe_value_loss(r, v, gamma)
    expected_loss = torch.zeros_like(v)

    assert torch.allclose(loss, expected_loss)

    # Test case with longer sequence
    v = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    r = torch.tensor([[1.0, 1.0, 0.0]])  # safe->safe->unsafe
    loss = delayed_unsafe_value_loss(r, v, gamma)

    seq_len = 3
    target_value = (1 - gamma**seq_len) / (1 - gamma)
    expected_loss = torch.nn.functional.relu(target_value - v)
    assert torch.allclose(loss, expected_loss)


def test_td_loss():
    r = torch.tensor([[1.0], [0.5]])
    v_next = torch.tensor([[0.5], [0.0]])
    q_next = torch.tensor([[0.5], [0.5]])
    assert torch.allclose(
        temporal_difference_loss(r, v_next, q_next, gamma=0.5),
        torch.tensor([[0.5625], [0.0]]),
    )


def test_expectile_loss():
    tau = 0.1

    q = torch.tensor([0.5, 0.5, 0.5])
    v = torch.tensor([1.0, 0.5, 0.0])
    expected_u_squared = torch.tensor([0.25, 0.0, 0.25])
    expected_weight = torch.tensor([0.9, 0.1, 0.1])
    expected_result = expected_weight * expected_u_squared
    assert torch.allclose(expectile_loss(q, v, tau), expected_result)

    batch_q = torch.stack([q, q])
    batch_v = torch.stack([v, v])
    expected_u_squared = torch.stack([expected_u_squared, expected_u_squared])
    expected_weight = torch.stack([expected_weight, expected_weight])
    expected_result = expected_weight * expected_u_squared
    result = expectile_loss(batch_q, batch_v, tau)
    assert torch.allclose(result, expected_result)


def test_cbf_monte_carlo_loss():
    """
    Test the CBF Monte Carlo Loss Function:
    1. Loss should be the final value function value minus the infinite sum of discounted safe rewards
    2. Loss should only be applied to sequences in which the final token is safe
    """

    gamma = 0.9

    vs = torch.tensor([[0.0, 0.0, 0.0, 1.0]] * 2)
    rs = torch.tensor([[1.0, 1.0, 1.0, 1.0], [0.0, 0.0, 0.0, 0.0]])

    expected_loss = (vs - (1 / (1 - gamma))) ** 2 * torch.tensor([1, 0])[:, None]
    calculated_loss = cbf_monte_carlo_loss(rs, vs, gamma)

    print(expected_loss)
    print(calculated_loss)
    assert torch.allclose(expected_loss, calculated_loss)


def test_unsafe_norm_loss():
    """Test the unsafe norm loss function:
    1. Should compute L2 norm of value embeddings for unsafe sequences
    2. Should be zero for safe sequences
    """
    # Test case with both safe and unsafe sequences
    v = torch.tensor([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])  # Value embeddings
    r = torch.tensor(
        [[1.0, 1.0], [0.0, 0.0]]
    )  # Safety labels, last token determines safety

    loss = unsafe_norm_loss(v, r)

    # First sequence is safe (last label is 1), should have 0 loss
    # Second sequence is unsafe (last label is 0), should have L2 norm
    expected_norms = torch.tensor([0.0, torch.norm(torch.tensor([0.2, 0.4, 0.6]))])

    assert torch.allclose(loss, expected_norms)

    # Test case with all safe sequences
    r = torch.tensor([[1.0, 1.0], [1.0, 1.0]])
    loss = unsafe_norm_loss(v, r)
    assert torch.allclose(loss, torch.zeros(2))

    # Test case with all unsafe sequences
    r = torch.tensor([[0.0, 0.0], [0.0, 0.0]])
    loss = unsafe_norm_loss(v, r)
    expected_norms = torch.tensor(
        [
            torch.norm(torch.tensor([0.1, 0.3, 0.5])),
            torch.norm(torch.tensor([0.2, 0.4, 0.6])),
        ]
    )
    assert torch.allclose(loss, expected_norms)


def test_gamma_scaled_loss():
    """Test the gamma scaled loss function with left-padded sequences"""
    gamma = 0.9
    v = torch.tensor([[0.2, 0.4], [0.3, 0.5]])

    # Test with no padding
    r = torch.tensor([[1.0, 1.0], [0.0, 0.0]])
    loss = gamma_scaled_loss(r, v, gamma)

    # Calculate expected target values for sequence length 2
    target_value = (1 - gamma**2) / (1 - gamma)
    expected_loss = torch.nn.functional.relu(
        torch.tensor([[target_value, target_value], [target_value, target_value]]) - v
    )
    assert torch.allclose(loss, expected_loss)

    # Test with left padding
    r_padded = torch.tensor([[-100, 1.0], [-100, 0.0]])  # Left-padded sequences
    loss_padded = gamma_scaled_loss(r_padded, v, gamma)

    # Calculate expected target values for sequence length 1
    target_value_padded = 1.0  # Since (1 - gamma) / (1 - gamma) simplifies to 1
    expected_loss_padded = torch.nn.functional.relu(target_value_padded - v)
    assert torch.allclose(loss_padded, expected_loss_padded)


def test_safe_unsafe_combined_loss():
    """Test the safe/unsafe combined loss function:
    1. Should apply correct loss for safe sequences
    2. Should apply correct loss for unsafe sequences
    3. Should handle mixed safe/unsafe batches
    """
    gamma = 0.9
    v = torch.tensor([[0.2, 0.4], [0.3, 0.5]])

    # Test mixed safe/unsafe sequences
    r = torch.tensor([[1.0, 1.0], [0.0, 0.0]])
    loss = safe_unsafe_combined_loss(r, v, gamma)

    # Calculate expected components
    safe_mask = (r[:, -1] == 1).unsqueeze(1)
    unsafe_mask = (r[:, -1] == 0).unsqueeze(1)

    expected_safe_component = gamma_scaled_loss(r, v, gamma) * safe_mask
    expected_unsafe_component = tokenwise_monte_carlo_loss(r, v, gamma) * unsafe_mask
    expected_loss = expected_safe_component + expected_unsafe_component

    assert torch.allclose(loss, expected_loss)

    # Test all safe sequences
    r_safe = torch.tensor([[1.0, 1.0], [1.0, 1.0]])
    loss_safe = safe_unsafe_combined_loss(r_safe, v, gamma)
    assert torch.allclose(loss_safe, gamma_scaled_loss(r_safe, v, gamma))

    # Test all unsafe sequences
    r_unsafe = torch.tensor([[0.0, 0.0], [0.0, 0.0]])
    loss_unsafe = safe_unsafe_combined_loss(r_unsafe, v, gamma)
    assert torch.allclose(loss_unsafe, tokenwise_monte_carlo_loss(r_unsafe, v, gamma))
