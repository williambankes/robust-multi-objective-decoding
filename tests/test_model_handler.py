import pytest

from robust_multi_objective_decoding.model_handler import ModelHandler


@pytest.fixture(scope="module")
def model_handler():
    return ModelHandler.from_pretrained("EleutherAI/pythia-70m")


def test_model_handler(model_handler):
    assert model_handler.model_name == "EleutherAI/pythia-70m"

    config = model_handler.make_generation_config(num_return_sequences=1)
    completions = model_handler.generate_completion("Hello, world!", config=config)
    assert isinstance(completions, list)
    assert len(completions) == 1
    assert isinstance(completions[0], str)


def test_model_handler_to_str_tokens(model_handler):
    tokens = model_handler.to_str_tokens("Hello, world!")
    assert isinstance(tokens, list)
    assert all(isinstance(token, str) for token in tokens)
    assert len(tokens) > 0
    assert tokens[0] == "Hello"
