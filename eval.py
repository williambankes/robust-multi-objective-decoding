import os
import hydra
import torch
import torch.nn as nn
from omegaconf import DictConfig
from transformers import AutoTokenizer
from robust_multi_objective_decoding.utils.utils import (
    setup_huggingface_auth,
    print_config,
)
from robust_multi_objective_decoding.utils.load_utils import (
    load_base_vf_module_state_dict_from_checkpoint,
)
from robust_multi_objective_decoding.constants import ProjectDir
from robust_multi_objective_decoding.decoders.decoders import DecoderHandler
from datetime import datetime
import pytorch_lightning as pl
from robust_multi_objective_decoding.oracles.oracle import Oracle


class EvalModule(pl.LightningModule):
    def __init__(
        self,
        decoder: nn.Module,
        oracle: Oracle | None,
        tokenizer: AutoTokenizer,
        max_new_tokens: int = 20,
        eval_test_response: bool = True,
        empty_cache: bool = True,
        save_q_values: bool = False,
    ):
        super().__init__()
        self.decoder = decoder
        self.oracle = oracle
        self.tokenizer = tokenizer
        self.max_new_tokens = max_new_tokens
        self.eval_test_response = eval_test_response
        self.empty_cache = empty_cache
        self.save_q_values = save_q_values

        self.outputs = list()

    def remove_prompt_from_response(self, prompt, response):
        return response[len(prompt) :].strip()

    def test_step(self, batch, batch_idx):
        # Empty GPU cache
        if self.empty_cache:
            torch.cuda.empty_cache()

        decoder_output = self.decoder.generate(
            batch["prompt_input_ids"],
            attention_mask=batch[
                "prompt_attention_mask"
            ],  # Added for the blockwise approach
            max_new_tokens=self.max_new_tokens,
            return_dict_in_generate=True,
        )

        decoded_ids = decoder_output["generated_ids"]

        response_batch = self.tokenizer.batch_decode(
            decoded_ids, skip_special_tokens=True
        )
        prompt_batch = self.tokenizer.batch_decode(
            batch["prompt_input_ids"], skip_special_tokens=True
        )

        # Remove the prompt from the response:
        response_batch = [
            self.remove_prompt_from_response(p, r)
            for p, r in zip(prompt_batch, response_batch)
        ]

        # Evaluate the response on the oracle if oracle given:
        if self.oracle is None:
            evals = None
        else:
            evals = self.oracle.score(prompt_batch, response_batch)

        if self.eval_test_response:
            value_eval, q_value_eval = self.decoder._eval_value_function(
                batch["input_ids"], batch["attention_mask"]
            )

        # evals will be batch_size by len(Harm types) tensor of probabilities
        # For each row of the prompt batch and responses create a dict:
        for i, response in enumerate(response_batch):
            output_dict = {"prompt": prompt_batch[i], "response": response_batch[i]}

            for k, v in batch.items():
                output_dict[k] = v[i].cpu().numpy()

            if isinstance(decoder_output, dict):
                for (
                    k,
                    v,
                ) in (
                    decoder_output.items()
                ):  # TODO: make this a parsed argument to the eval function
                    if k in [
                        "is_cbf_condition_met",
                        "is_safe_record",
                        "joint_branch_probs_record",
                    ]:  # handle the list outputs
                        output_dict[k] = v[:, i]
                    elif k == "q_values" and self.save_q_values:
                        output_dict[k] = v[i].cpu().to(torch.float32).numpy()
                    elif k == "weights" or k == "values" or k == "advs":
                        output_dict[k] = v.detach().cpu().to(torch.float32).numpy()
                    elif k == "blocks":
                        output_dict[k] = v
                    else:  # handle the torch outputs
                        output_dict[k] = (
                            v[i].cpu().to(torch.float32).numpy()
                            if isinstance(v[i], torch.Tensor)
                            else v[i]
                        )
            else:
                output_dict["decoded_ids"] = decoded_ids[i].cpu().numpy()

            if evals is not None:
                if isinstance(evals[0], list):
                    output_dict["evals"] = [eval.cpu().numpy() for eval in evals[i]]
                elif isinstance(evals[0], torch.Tensor):
                    output_dict["evals"] = evals[i].cpu().numpy()
                else:
                    raise NotImplementedError(
                        f"Unexpected evals type {type(evals[0])}, for evals: {evals}"
                    )

            if self.eval_test_response:
                output_dict["value_eval_test_response"] = (
                    value_eval[i].cpu().to(torch.float32).numpy()
                )
                if self.save_q_values:
                    output_dict["q_value_eval_test_response"] = (
                        q_value_eval[i].cpu().to(torch.float32).numpy()
                    )

            output_dict["hyperparameters"] = self.decoder.get_hyperparameters()

            self.outputs.append(output_dict)


@hydra.main(config_path="./configs", config_name="multi_obj_default_config")
def eval(config: DictConfig):
    # Setup auth and directory structure
    setup_huggingface_auth()

    # Set the seed
    pl.seed_everything(config.seed)

    print_config(
        config,
        fields=[
            "decoder",
            "oracle",
            "checkpoint_path",
            "max_new_tokens",
            "score_and_generate",
            "trainer",
            "ref_model",
            "model",
            "dataset",
            "dataloader",
            "collate_fn",
            "tokenizer",
        ],
    )

    # Load the Datasets:
    test_data = hydra.utils.instantiate(config.dataset, split="test")
    # Select subset of test data if specified
    if config.get("eval_subset", None) is not None:
        subset_size = min(config.eval_subset, len(test_data))
        indices = range(subset_size)
        test_data = torch.utils.data.Subset(test_data, indices)

    print(f"Test data length: {len(test_data)}")
    print("Setup Tokenizer")
    ref_tokenizer = AutoTokenizer.from_pretrained(
        config.ref_model.pretrained_model_name_or_path, padding_side="left"
    )
    tokenizer = AutoTokenizer.from_pretrained(
        config.model.base_model.pretrained_model_name_or_path, padding_side="left"
    )

    # If the tokenizer does not have a pad token, set it to the eos token
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    if ref_tokenizer.pad_token_id is None:
        ref_tokenizer.pad_token_id = ref_tokenizer.eos_token_id

    # Create the collate function with the tokenizer for the reference model
    print("Setup Dataloaders")
    collate_func = hydra.utils.instantiate(
        config.collate_fn,
        tokenizer=ref_tokenizer,
        max_length=config.tokenizer.max_length,
    )

    test_loader = hydra.utils.instantiate(
        config.dataloader.test, dataset=test_data, collate_fn=collate_func
    )

    # Load the reference model
    ref_model = hydra.utils.instantiate(config.ref_model)

    # Load the previously trained model from a check point path
    if config.checkpoint_path is not None:
        print("Loading model from checkpoint")
        model = hydra.utils.instantiate(config.model)
        state_dict = load_base_vf_module_state_dict_from_checkpoint(
            config.checkpoint_path
        )
        model.load_state_dict(state_dict)
    else:
        model = None

    # Load the Oracle
    if config.score_and_generate:
        oracle = hydra.utils.instantiate(config.oracle)
    else:
        oracle = None

    # Load the decoder
    decoder = DecoderHandler.create_decoder(
        config=config.decoder,
        reference_model=ref_model,
        value_function=model,
        oracle=oracle,
        tokenizer=tokenizer,
        ref_tokenizer=ref_tokenizer,
    )

    # Create the eval module lightning module:
    if config.score_and_generate:
        eval_module = EvalModule(
            decoder,
            oracle,
            ref_tokenizer,
            max_new_tokens=config.max_new_tokens,
            eval_test_response=config.get("eval_test_response", False),
        )
    else:
        eval_module = EvalModule(
            decoder,
            None,
            ref_tokenizer,
            max_new_tokens=config.max_new_tokens,
            eval_test_response=config.get("eval_test_response", False),
        )

    # Load the trainer
    trainer = hydra.utils.instantiate(config.trainer)  # , logger=logger)

    # Run the evaluation:
    try:
        with torch.no_grad():
            print("Running evaluation")
            trainer.test(eval_module, dataloaders=test_loader)

    except RuntimeError as e:
        # This occurs when the pytorch lightning model tries to map everything back to cpu after running (not sure why?)
        print(f"Runtime Error occured: {e}")
        assert "DefaultCPUAllocator: can't allocate memory" in str(
            e
        ), "Unexpected RuntimeError"

    # Save the outputs -> using datetime to create a unique file name
    time_file_name = datetime.now().strftime("%m-%d-%Y-%H-%M-%S")
    experiment_name = (
        config.experiment_name if config.experiment_name is not None else ""
    )
    path = os.path.join(
        "eval_outputs", f"eval_outputs_{experiment_name}_{time_file_name}.pt"
    )
    path = ProjectDir / path

    print(f"Saving cbf eval_outputs to:{path}")
    torch.save(eval_module.outputs, path)


if __name__ == "__main__":
    torch.set_float32_matmul_precision("medium")
    eval()
