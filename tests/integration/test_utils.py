import torch
import copy
import os

from robust_multi_objective_decoding.utils.load_utils import load_base_vf_module_state_dict_from_checkpoint
from transformers import AutoModelForCausalLM, AutoTokenizer
from robust_multi_objective_decoding.value_function import ActionValueFunctionModule
from robust_multi_objective_decoding.value_function_learner import ValueFunctionLearner
from torch.utils.data import Dataset
from pytorch_lightning import Trainer
from peft import LoraConfig
from robust_multi_objective_decoding.data.collate_functions import create_word_break_tokenizer_collate_fn

#################### FIXTURES AND PROXIES ####################

class ProxyDataset(Dataset):

    def __getitem__(self, idx):

        prompt = "What is your name?"
        response = "My name is go away you silly person."
        labels = [-100, -100, -100, -100, 1, 1, 1, 0, 0, 0, 0, 0]

        return idx, prompt, response, labels
    
    def __len__(self):
        return 10

#################### TESTS ####################

def test_run_and_checkpoint_loading():
    """ Test that the model can correctly loaded from a checkpoint after running training """

    # Construct a model
    base_model = AutoModelForCausalLM.from_pretrained("EleutherAI/pythia-70m",
                                                    torch_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-70m")

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    data = ProxyDataset()
    loader = torch.utils.data.DataLoader(data, batch_size=2,
                    collate_fn=create_word_break_tokenizer_collate_fn(tokenizer, max_length=128))

    model = ActionValueFunctionModule(
        base_model=base_model,
        base_model_hidden_dim=512,
        token_vocab_size=tokenizer.vocab_size,
        lora_config=LoraConfig(  # type: ignore
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
        ),
        torch_dtype=torch.bfloat16,
    )

    orig_state_dict = copy.deepcopy(model.state_dict())
    learner = ValueFunctionLearner(model, learning_rate=1e-2)

    # Fit with 0 steps, so that the model is not changed
    trainer = Trainer(
        max_steps=0,
        accelerator='cpu',
        logger=False,
        enable_checkpointing=False
    )
    trainer.fit(learner, loader)
    trainer.save_checkpoint("test_0.ckpt")

    # Load the model from the checkpoint
    new_state_dict = load_base_vf_module_state_dict_from_checkpoint("test_0.ckpt")
    new_state_dict_key_list = list(new_state_dict.keys())

    # Check that the parameters match
    # Since the model was not trained, the parameters should be the same
    for key in orig_state_dict.keys():
        assert key in new_state_dict_key_list
        assert (orig_state_dict[key] == new_state_dict[key]).all()

    # clean up checkpoint
    os.remove("test_0.ckpt")
    del trainer
    del new_state_dict
    del new_state_dict_key_list

    # Fit with 1 step, so that the model is changed
    trainer = Trainer(
        max_steps=1,
        accelerator='cpu',
        logger=False,
        enable_checkpointing=False
    )
    trainer.fit(learner, loader)
    trainer.save_checkpoint("test_1.ckpt")

    # Load the model from the checkpoint
    new_state_dict = load_base_vf_module_state_dict_from_checkpoint("test_1.ckpt")
    new_state_dict_key_list = list(new_state_dict.keys())

    # Check that the parameters have been updated
    for key in orig_state_dict.keys():
        assert key in new_state_dict_key_list
        # NOTE: Because we do LoRA, some of the parameters are not learnt
        # Here, I implement a hacky way to filter for parameters that should have been updated
        param_is_learnable = "learn" in key
        if param_is_learnable:
            assert not (orig_state_dict[key] == new_state_dict[key]).all(), f"key: {key}"

    # clean up checkpoint
    os.remove("test_1.ckpt")