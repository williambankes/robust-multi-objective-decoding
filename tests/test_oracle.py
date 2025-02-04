import os
import pytest
from robust_multi_objective_decoding.oracles.oracle import (
    Oracle,
    CompositeOracle,
    MultiObjectiveOracle,
    get_tokenwise_scores)
from robust_multi_objective_decoding.oracles.simple import (
    CharacterLengthOracle,
    PunctuationOracle)
from robust_multi_objective_decoding.oracles.shield_gemma import ShieldGemmaSafetyOracle, HarmType

IN_GITHUB_ACTIONS = os.getenv("GITHUB_ACTIONS") == "true"
NO_HUB_AUTH_ID = os.getenv("HUGGINGFACE_HUB_TOKEN") is None

def test_character_length_oracle():
    oracle = CharacterLengthOracle(max_length=10)
    assert oracle.score("prompt", "response") == 1
    assert oracle.score("prompt", "response that is too long") == 0


def test_punctuation_oracle():
    oracle = PunctuationOracle()
    assert oracle.score("prompt", "response.") == 1
    assert oracle.score("prompt", "response") == 0


def test_composite_oracle():
    oracle = CompositeOracle(
        [CharacterLengthOracle(max_length=10), PunctuationOracle()]
    )
    assert oracle.score("prompt", "response.") == 1
    assert oracle.score("prompt", "response that is too long") == 0
    assert oracle.score("prompt", "response") == 0


def test_get_tokenwise_scores():
    oracle = CharacterLengthOracle(max_length=25)
    prompt_tokens = ["prompt", " tokens"]
    response_tokens = [" response", " tokens", " that", " is", " too", " long"]
    scores = get_tokenwise_scores(prompt_tokens, response_tokens, oracle)
    assert scores == [1, 1, 1, 0, 0, 0]

### FIXTURES ###

class ProxyOracle(Oracle):

    def __init__(self, mode=0):
        self.mode = mode
        pass

    def is_admissible(self) -> bool:
        return True
    
    def score(self, prompt, response):
        if self.mode == 0:
            return [1 if p == 1 else 0 for p in prompt]
        else:
            return [[1] if p == 1 else [0] for p in prompt]

### TESTS ###

def test_multiobjectiveoracle():

    prompts = [1,2,3,4]
    response = None
    mo_oracle = MultiObjectiveOracle([ProxyOracle(mode=0), ProxyOracle(mode=1)])
    scores = mo_oracle.score(prompts, response)

    assert len(scores) == len(prompts)
    assert len(scores[0]) == 2
    assert scores[0][0] == 1 and scores[0][1] == [1]
    assert scores[1][0] == 0 and scores[1][1] == [0] 