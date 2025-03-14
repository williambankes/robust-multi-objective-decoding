import torch
from robust_multi_objective_decoding.losses import (
    Loss,
    CompositeLoss,
    TemporalDifferenceLoss,
    CDQLoss,
    MonteCarloLoss,
)


class TestCompositeLoss:
    def test_initialization(self):
        """Test CompositeLoss initialization."""

        # Create mock loss functions
        class MockLoss(Loss):
            def get_name(self):
                return "mock_loss"

            def forward(self, input_ids, q, v, r, q_hat, v_next):
                return torch.tensor(1.0)

        loss1 = MockLoss(weight=torch.tensor(1.0))
        loss2 = MockLoss(weight=torch.tensor(2.0))

        composite = CompositeLoss([loss1, loss2], weight=torch.tensor(3.0))
        assert len(composite.loss_functions) == 2
        assert composite.weight == torch.tensor(3.0)
        assert composite.get_name() == "mock_loss_mock_loss"

    def test_forward(self):
        """Test CompositeLoss forward pass."""

        # Create mock loss functions
        class MockLoss(Loss):
            def __init__(self, weight, return_value):
                super().__init__(weight)
                self.return_value = return_value

            def get_name(self):
                return "mock_loss"

            def forward(self, input_ids, q, v, r, q_hat, v_next):
                return self.return_value * self.weight

        loss1 = MockLoss(weight=torch.tensor(1.0), return_value=torch.tensor(2.0))
        loss2 = MockLoss(weight=torch.tensor(1.0), return_value=torch.tensor(3.0))

        composite = CompositeLoss([loss1, loss2], weight=torch.tensor(2.0))

        # Create dummy inputs
        input_ids = torch.ones(2, 3, dtype=torch.long)
        q = torch.ones(2, 3, 10, 2)
        v = torch.ones(2, 3)
        r = torch.ones(2, 3)
        q_hat = torch.ones(2, 3, 10, 2)
        v_next = torch.ones(2, 3)

        result = composite(
            input_ids=input_ids, q=q, v=v, r=r, q_hat=q_hat, v_next=v_next
        )
        # Expected: (2.0*1.0 + 3.0*1.0) * 2.0 = 10.0
        assert result == torch.tensor(10.0)


class TestTemporalDifferenceLoss:
    def test_initialization(self):
        """Test TemporalDifferenceLoss initialization."""
        td_loss = TemporalDifferenceLoss(gamma=0.99, weight=torch.tensor(1.5))
        assert td_loss.gamma == 0.99
        assert td_loss.weight == torch.tensor(1.5)
        assert td_loss.get_name() == "TD_Loss"

    def test_forward(self):
        """Test TemporalDifferenceLoss forward pass."""
        td_loss = TemporalDifferenceLoss(gamma=0.5, weight=torch.tensor(2.0))

        # Create test data
        batch_size, seq_len = 2, 3
        r = torch.ones(batch_size, seq_len)
        v = torch.ones(batch_size, seq_len)
        v_next = torch.ones(batch_size, seq_len)[:, 1:]

        # Modify some values to test the calculation
        v[:, 0] = 2.0  # V(s_t) for first token
        v_next[:, 0] = 3.0  # V(s_{t+1}) for first token
        r[:, -1] = 5.0  # Last reward

        result = td_loss(r=r, v=v, v_next=v_next)

        # Expected calculation:
        # V = v[:, :-1] = [[2.0, 1.0], [2.0, 1.0]]
        # v_next_det = v_next.detach() with last column replaced by r[:, -1] = 5.0
        # v_next_det = [[3.0, 1.0], [3.0, 1.0]] (before last column replacement)
        # v_next_det = [[3.0, 5.0], [3.0, 5.0]] (after last column replacement)
        # td_error = (V - v_next_det)^2 = [[(2.0-3.0)^2, (1.0-5.0)^2], [(2.0-3.0)^2, (1.0-5.0)^2]]
        # td_error = [[1.0, 16.0], [1.0, 16.0]]
        # Mean td_error = (1.0 + 16.0 + 1.0 + 16.0) / 4 = 8.5
        # Weighted result = 8.5 * 2.0 = 17.0

        assert torch.isclose(result.mean(), torch.tensor(17.0))

        # Test that the last token of v_next is replaced with the last reward
        v_next_copy = v_next.clone()
        _ = td_loss(r=r, v=v, v_next=v_next_copy)
        assert (v_next_copy[:, -1] == 5.0).all()


class TestCDQLoss:
    def test_initialization(self):
        """Test CDQLoss initialization."""
        cdq_loss = CDQLoss(weight=torch.tensor(-2.0))
        assert cdq_loss.weight == torch.tensor(-2.0)
        assert cdq_loss.get_name() == "CDQ_Loss"

    def test_forward(self):
        """Test CDQLoss forward pass."""
        cdq_loss = CDQLoss(weight=torch.tensor(-1.0))

        # Create test data
        batch_size, seq_len, vocab_size = 2, 3, 5
        q = torch.ones(batch_size, seq_len, vocab_size, 2)
        input_ids = torch.zeros(batch_size, seq_len, dtype=torch.long)
        input_ids[:, 1:] = 1  # Set tokens after the first to index 1

        # Make q values different to test gathering
        q[:, 0, 1, 0] = 2.0

        result = cdq_loss(q=q, input_ids=input_ids)

        # The result shape is [batch_size, seq_len-1] = [2, 2]
        assert result.shape == torch.Size([2, 2])
        # Check that all values are negative (since weight is negative)
        assert (result <= 0).all()


class TestMonteCarloLoss:
    def test_initialization(self):
        """Test MonteCarloLoss initialization."""
        mc_loss = MonteCarloLoss(weight=torch.tensor(3.0))
        assert mc_loss.weight == torch.tensor(3.0)
        assert mc_loss.get_name() == "Monte_Carlo_Loss"

    def test_forward(self):
        """Test MonteCarloLoss forward pass."""
        mc_loss = MonteCarloLoss(weight=torch.tensor(2.0))

        # Create test data
        batch_size, seq_len = 2, 3
        v = torch.ones(batch_size, seq_len)
        r = torch.ones(batch_size, seq_len)

        # Set different values to test calculation
        v[:, 1:] = 2.0  # V(s) for tokens after the first
        r[:, -1] = 4.0  # Last reward

        result = mc_loss(v=v, r=r)

        # The result shape should match v[:, 1:] which is [2, 2]
        assert result.shape == torch.Size([2, 2])
        # Each element should be (2.0 - 4.0)^2 * 2.0 = 8.0
        assert torch.allclose(result, torch.tensor(8.0).expand(2, 2))
