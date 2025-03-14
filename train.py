"""
Script to train a safety CBF.

Run the training script using `python train.py`

Options are specified in the `configs/default_config.yaml` file.
"""

import os
import time
import hydra
import torch
from omegaconf import DictConfig
from transformers import AutoTokenizer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from robust_multi_objective_decoding.utils.utils import (
    setup_checkpoint_dir,
    setup_huggingface_auth,
    print_config,
)

os.environ["TOKENIZERS_PARALLELISM"] = "false"


@hydra.main(config_path="./configs", config_name="multi_obj_default_config")
def train(config: DictConfig):
    # Print the hydra config
    print_config(config)

    # Setup auth and directory structure
    setup_huggingface_auth()

    # TODO: Move this all to utils
    # checkpoint_dir = setup_checkpoint_dir(config)
    if config.get("experiment_folder", None) is None:
        checkpoint_dir = setup_checkpoint_dir(config)
    experiment_name = f"{config.get('experiment_name', 'decoding-experiment')}_{time.strftime('%Y%m%d_%H%M%S')}"
    checkpoint_dir = os.path.join(config.experiment_folder, experiment_name)
    if not os.path.exists(checkpoint_dir):
        print(f"Creating checkpoint dir: {checkpoint_dir}")
        os.makedirs(checkpoint_dir)

    print("cwd", os.getcwd())
    # Load the Datasets:
    train_data = hydra.utils.instantiate(config.dataset, split="train")
    val_data = hydra.utils.instantiate(config.dataset, split="val")
    test_data = hydra.utils.instantiate(config.dataset, split="test")

    # Load the model and tokenizer:
    print("Setup Tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(
        config.model.base_model.pretrained_model_name_or_path
    )
    tokenizer.padding_side = "left"

    # If the tokenizer does not have a pad token, set it to the eos token
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    print("Setup Dataloaders")

    # Create the dataloaders with the relevant collate function
    collate_func = hydra.utils.instantiate(
        config.collate_fn, tokenizer=tokenizer, max_length=config.tokenizer.max_length
    )

    train_loader = hydra.utils.instantiate(
        config.dataloader.train, dataset=train_data, collate_fn=collate_func
    )
    val_loader = hydra.utils.instantiate(
        config.dataloader.val, dataset=val_data, collate_fn=collate_func
    )
    test_loader = hydra.utils.instantiate(
        config.dataloader.test, dataset=test_data, collate_fn=collate_func
    )

    print("Setup Callbacks and Logging")
    experiment_dir = os.path.join(checkpoint_dir, str(int(time.time()))[:10])
    checkpoint_callback = ModelCheckpoint(
        **config.callbacks.checkpoint_callback, dirpath=experiment_dir
    )
    logger = WandbLogger(**config.logger)

    # Calculate total training steps
    # If the total number of training samples is provided in the trainer config, use that
    if config.trainer.get("max_steps", None) is not None:
        grad_accum = config.trainer.get("accumulate_grad_batches", 1)
        total_steps = config.trainer.max_steps // grad_accum
    # Otherwise, calculate the total number of training steps based on the number of epochs
    else:
        num_training_samples = len(train_data)
        batch_size = config.dataloader.train.batch_size
        grad_accum = config.trainer.get("accumulate_grad_batches", 1)
        effective_batch = batch_size * grad_accum
        steps_per_epoch = num_training_samples // effective_batch
        total_steps = steps_per_epoch * config.trainer.max_epochs

    # Create the model and pl lightning model
    model = hydra.utils.instantiate(config.model)
    pl_model = hydra.utils.instantiate(
        config.learner,
        base_value_function_module=model,
        total_training_steps=total_steps,
    )

    trainer = hydra.utils.instantiate(
        config.trainer, callbacks=[checkpoint_callback], logger=logger
    )

    print("Start Training")
    trainer.fit(pl_model, train_loader, val_loader)
    # trainer.save_checkpoint("manual_checkpoint.ckpt")
    # print(f"manual checkpoint saved to {os.getcwd()}/manual_checkpoint.ckpt")
    print("Finished Training")

    if config.test:
        print("Running Testing")
        trainer.test(pl_model, dataloaders=test_loader)
        torch.save(pl_model.test_outputs, "/scratch/ucabwjn/results/test_outputs.pt")


if __name__ == "__main__":
    torch.set_float32_matmul_precision("medium")
    train()
