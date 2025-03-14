import pytest
import torch
from peft import LoraConfig
from robust_multi_objective_decoding.value_function import ValueFunctionModule
from robust_multi_objective_decoding.value_function_learner import ValueFunctionLearner
import transformers
import pytorch_lightning as pl
from torch.utils.data import DataLoader, IterableDataset


############## TEST MOCK SETUP ##############


@pytest.fixture(scope="module")
def model_and_tokenizer():
    model_name = "EleutherAI/pythia-70m"
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model = transformers.AutoModelForCausalLM.from_pretrained(model_name)
    return model, tokenizer


@pytest.fixture(scope="module")
def lora_config():
    return LoraConfig(
        r=16,
        lora_alpha=16,
        target_modules=[
            "query_key_value",
            "dense",
            "dense_h_to_4h",
            "dense_4h_to_h",
        ],
        lora_dropout=0.1,
        bias="none",
    )


@pytest.fixture(scope="module")
def value_function_module(model_and_tokenizer, lora_config):
    model, tokenizer = model_and_tokenizer
    return ValueFunctionModule(
        base_model=model,
        base_model_hidden_dim=model.config.hidden_size,
        lora_config=lora_config,
        torch_dtype=torch.float32,
    )


@pytest.fixture(scope="module")
def dataloader(model_and_tokenizer):
    _, tokenizer = model_and_tokenizer
    sentence = "How to hurt a human"
    safety_labels = torch.tensor([1, 1, 0, 0, 0])
    batch_size = 2

    class DummyDataset(IterableDataset):
        data = [(sentence, safety_labels) for _ in range(3)]

        def __iter__(self):
            yield from self.data

    dataset = DummyDataset()

    def collate_function(batch):
        data, labels = zip(*batch)
        tokenized_batch = tokenizer(
            data,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        tokenized_batch["tokenwise_safety_labels"] = torch.stack(labels)
        return tokenized_batch

    return DataLoader(dataset, batch_size=batch_size, collate_fn=collate_function)


@pytest.fixture(scope="module")
def learner(value_function_module):
    return ValueFunctionLearner(
        base_value_function_module=value_function_module,
        update_q_hat_every=1,
        learning_rate=10,
    )


############## TESTS ##############


@pytest.mark.slow()
def test_valuefunctionlearner_loss(learner, dataloader):
    batch = next(iter(dataloader))
    loss = learner.training_step(batch, 0)
    assert loss is not None
    assert loss > 0
    assert not torch.isnan(loss)


@pytest.mark.slow()
def test_valuefunctionlearner_on_after_backward(learner):
    """
    Test the on_after_backward method of the ValueFunctionLearner
    """
    # Save original params, run on_after_backward, and compare to manual Polyak update
    original_params = {
        name: param.detach().clone() for name, param in learner.named_parameters()
    }
    learner.on_after_backward()
    learner_params = dict(learner.named_parameters())

    with torch.no_grad():
        for name, param in learner_params.items():
            # Manual Polyak update
            if "target" in name:
                source_name = (
                    "base_value_function_module."
                    + learner.base_value_function_module.polyak_update_mapping[
                        name.removeprefix("base_value_function_module.")
                    ]
                )
                source_param = learner_params[source_name]
                manual_polyak = (
                    original_params[name]
                    * (1 - learner.base_value_function_module.polyak_coeff)
                    + source_param * learner.base_value_function_module.polyak_coeff
                )

                # Compare manual and automatic polyak updates
                assert torch.allclose(
                    param, manual_polyak
                ), f"Parameter {name} has not been Polyak updated correctly (manual result: {manual_polyak}, auto result: {param})"


@pytest.mark.slow()
def test_valuefunctionlearner_training(learner, dataloader):
    original_params = {
        name: param.detach().clone() for name, param in learner.named_parameters()
    }

    trainer = pl.Trainer(max_epochs=1, accelerator="cpu")
    trainer.fit(learner, dataloader)

    # Check the weights have been updated
    learner_params = dict(learner.named_parameters())
    for name, param in learner_params.items():
        if param.requires_grad:
            assert not torch.allclose(
                param, original_params[name]
            ), f"Parameter {name} has not been updated (original value: {original_params[name]}, new value: {param})"
