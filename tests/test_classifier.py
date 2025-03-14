from robust_multi_objective_decoding.classifier import Classifier, ClassifierLearner
import transformers
import torch
import torch.nn as nn
from peft import LoraConfig


class ProxyClassifier(nn.Module):
    def __init__(self, batch_size):
        super().__init__()
        self.batch_size = batch_size

    def forward(self, input_ids, attention_mask):
        return torch.tensor([0.95] * self.batch_size)


def test_classifier():
    """
    Test the classifier class setup here.
    """

    model = transformers.AutoModelForCausalLM.from_pretrained(
        "EleutherAI/pythia-70m", torch_dtype=torch.bfloat16
    )
    tok = transformers.AutoTokenizer.from_pretrained("EleutherAI/pythia-70m")

    text_input = tok("Hello, my dog is cute", return_tensors="pt")

    lora_config = LoraConfig(
        r=16,
        lora_alpha=16,
        target_modules=["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"],
        lora_dropout=0.1,
        bias="none",
    )

    classifier = Classifier(
        base_model=model,
        base_model_hidden_dim=512,
        lora_config=lora_config,
        torch_dtype=torch.bfloat16,
    )

    output = classifier(**text_input)

    assert tuple(output.shape) == (1,)
    assert (output < 0).sum() == 0
    assert (output > 1).sum() == 0


def test_classifier_learner_train_step():
    classifier = ProxyClassifier(batch_size=3)

    batch = dict()
    batch["input_ids"] = torch.ones(3, 10)
    batch["attention_mask"] = torch.ones(3, 10)
    batch["safety_labels"] = torch.ones(3)

    learner = ClassifierLearner(classifier)

    output = learner.training_step(batch, 0)
    assert torch.abs(output - torch.tensor(3.2958)) < 1e-3
