import pytest


from robust_multi_objective_decoding.oracles.oracle import get_tokenwise_scores
from robust_multi_objective_decoding.toy_problems.math_equations import InstructionOracle


@pytest.fixture
def oracle():
    return InstructionOracle()


def test_oracle_admissibility(oracle):
    assert oracle.is_admissible() == True


@pytest.mark.parametrize(
    "prompt,response,expected",
    [
        ("Generate an even number", "2 + 2 = 4.", 1),
        ("Generate an even number", "3 + 5 = 8.", 1),
        ("Generate an even number", "10 - 3 = 7.", 0),
        ("Generate an odd number", "4 * 3 = 12.", 0),
        ("Generate an odd number", "15 / 3 = 5.", 1),
        ("Generate an even number", "This sentence ends with 3.", 0),
        ("Generate an odd number", "No number here.", 0),
        ("Generate an even number", "Incomplete generation with even number 4", 1),
        ("Generate an odd number", "Incomplete generation with odd number 7", 1),
        ("Generate an even number", "2 + 2 =", 1),
        ("Generate an odd number", ".", 0),
        ("Generate an even number", "", 1),
    ],
)
def test_oracle_score(oracle, prompt, response, expected):
    assert oracle.score(prompt, response) == expected


def test_invalid_prompt(oracle):
    with pytest.raises(AssertionError):
        oracle.score("Invalid prompt", "2 + 2 = 4.")


def test_get_tokenwise_scores_even(oracle):
    prompt_tokens: list[str] = ["Generate", "an", "even", "number:"]
    response_tokens: list[str] = ["2", "+", "2", "=", "4", "."]
    expected_scores = [1, 1, 1, 1, 1, 1]

    scores = get_tokenwise_scores(prompt_tokens, response_tokens, oracle)
    assert scores == expected_scores


def test_get_tokenwise_scores_odd(oracle):
    prompt_tokens: list[str] = ["Generate", "an", "odd", "number:"]
    response_tokens: list[str] = ["3", "+", "4", "=", "7", "."]
    expected_scores = [1, 1, 1, 1, 1, 1]

    scores = get_tokenwise_scores(prompt_tokens, response_tokens, oracle)
    assert scores == expected_scores


def test_get_tokenwise_scores_incomplete(oracle):
    prompt_tokens: list[str] = ["Generate", "an", "even", "number:"]
    response_tokens: list[str] = ["2", "+", "3", "=", "5"]
    expected_scores = [1, 1, 1, 1, 1]

    scores = get_tokenwise_scores(prompt_tokens, response_tokens, oracle)
    assert scores == expected_scores
