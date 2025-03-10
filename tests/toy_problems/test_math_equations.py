import pytest
from typing import List

from robust_multi_objective_decoding.oracles.oracle import get_tokenwise_scores
from robust_multi_objective_decoding.toy_problems.math_equations import EvenNumberOracle


@pytest.fixture
def oracle():
    return EvenNumberOracle()


def test_oracle_admissibility(oracle):
    assert oracle.is_admissible() == True


@pytest.mark.parametrize(
    "response,expected",
    [
        ("2 + 2 = 4.", 1),
        ("3 + 5 = 8.", 1),
        ("10 - 3 = 7.", 0),
        ("4 * 3 = 12.", 1),
        ("15 / 3 = 5.", 0),
        ("This sentence ends with 3.", 0),
        ("No number here.", 0),
        ("Incomplete generation with even number 4", 1),
        ("Incomplete generation with odd number 7", 1),
        ("2 + 2 =", 1),
        (".", 0),
        ("", 1),
    ],
)
def test_oracle_score(oracle, response, expected):
    assert oracle.score("", response) == expected


def test_get_tokenwise_scores(oracle):
    prompt_tokens: List[str] = ["Solve", "this", "equation:"]
    response_tokens: List[str] = ["2", "+", "2", "=", "4", "."]
    expected_scores = [1, 1, 1, 1, 1, 1]

    scores = get_tokenwise_scores(prompt_tokens, response_tokens, oracle)
    assert scores == expected_scores


def test_get_tokenwise_scores_odd_result(oracle):
    prompt_tokens: List[str] = ["Solve", "this", "equation:"]
    response_tokens: List[str] = ["3", "+", "4", "=", "7", "."]
    expected_scores = [1, 1, 1, 1, 1, 0]

    scores = get_tokenwise_scores(prompt_tokens, response_tokens, oracle)
    assert scores == expected_scores


def test_get_tokenwise_scores_incomplete(oracle):
    prompt_tokens: List[str] = ["Solve", "this", "equation:"]
    response_tokens: List[str] = ["2", "+", "3", "=", "5"]
    expected_scores = [1, 1, 1, 1, 1]

    scores = get_tokenwise_scores(prompt_tokens, response_tokens, oracle)
    assert scores == expected_scores
