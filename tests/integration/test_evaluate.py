import itertools

import pytest

from robust_multi_objective_decoding.run import evaluate


@pytest.mark.slow()
def test_evaluate(base_model, dataset, oracle):
    evaluation = evaluate(base_model, dataset, oracle)

    assert 0 <= evaluation["safe_ratio"] <= 1

    for r in evaluation["results"]:
        assert len(r["tokenwise_scores"]) == len(r["response_tokens"])
        assert all(score in [0, 1] for score in r["tokenwise_scores"])

        # Test that scores are admissible
        num_leading_ones = len(
            list(itertools.takewhile(lambda x: x == 1, r["tokenwise_scores"]))
        )
        num_total_ones = sum(r["tokenwise_scores"])
        assert num_leading_ones == num_total_ones
