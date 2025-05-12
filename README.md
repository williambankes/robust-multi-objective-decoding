# Robust Multi-Objective Decoding of Large Language Models

| **[Abstract](#abstract)**
| **[Installation](#installation)**
| **[Quick Run](#quickrun)** |

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3119/)
[![Pytorch](https://img.shields.io/badge/Pytorch-2.5-red.svg)](https://shields.io/)
[![License](https://img.shields.io/badge/License-GPL%203.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://GitHub.com/Naereen/StrapDown.js/graphs/commit-activity)

# Abstract
Test-time alignment of Large Language Models (LLMs) to human preferences offers a flexible way to generate responses with diverse alignment without extensive retraining of LLMs. Existing methods optimize multiple objectives simultaneously (e.g., instruction-following, helpfulness, conciseness) by assigning rewards, but often rely on predefined weights or optimize for averages, sacrificing one objective for another and leading to unbalanced outcomes. To address this, we introduce Robust Multi-Objective Decoding (RMOD), a novel inference-time algorithm that optimizes for improving worst-case rewards. RMOD formalizes the robust decoding problem as a maximin two-player game between reward weights and the sampling policy, solving for the Nash equilibrium. We show that the game reduces to a convex optimization problem to find the worst-case weights, while the best response policy can be computed analytically. We also introduce a practical RMOD variant designed for efficient decoding with contemporary LLMs, incurring minimal computational overhead compared to non-robust Multi-Objective Decoding (MOD) methods. Our experimental results showcase the effectiveness of RMOD in  generating responses equitably aligned with diverse objectives, outperforming baselines up to 20%.

# Installation

## Install PDM
Our repository uses [PDM](https://pdm-project.org/en/latest/) to manage dependencies. To install PDM, follow [official instructions](https://pdm-project.org/en/latest/#installation).

## Setup Environment
Our code uses `python 3.11`. Install dependencies and activate the environment run the following
```bash
pdm install
source .venv/bin/activate
```

# Quick Run

Brief instructions on how to train and run the experiments on the HH dataset found in our paper. To begin download the [dataset](https://huggingface.co/datasets/Robust-Decoding/HH_gemma-2-2b-it) from our huggingface repo.

To train a value function on the HH dataset run the following command, replacing <DATA_PATH_HERE> with a path to the dataset. 
```bash
python train.py +experiment=gpt2large_multimodel_hh_cdfudge ++dataset=<DATA_PATH_HERE> ++trainer.devices=[<GPU_IDS_HERE>]
```

To evaluate the trained value function use the following command:
```bash
python eval.py +experiment=gpt2large_multimodel_hh_cdfudge ++dataset=<DATA_PATH_HERE> ++trainer.devices=[<GPU_IDS_HERE>] checkpoint_path=<CHECKPOINT_PATH>
```
Where the CHECKPOINT_PATH points to the saved checkpoint of the model run above.

# General Usage

The code uses the `pytorch-lightning` framework to manage the training and evaluation code, and `hydra` to manage different configuration files and experiments. Specific experiment configs can be found under `config/experiments` and editted in file or through the command line to specific configurations. The default config setup is provided in `config/multi_obj_default_config.yaml`. The file is broadly structured as follows:

### Dataset

Specify dataset arguments. The default setup is a general dataset loader that loads a file from disk. Datasets should have the following columns:

- `prompt` which is used to prompt the generations in the `eval.py` script
- `response` which contains **on-policy** completions to the prompt for the model we're training a value function on
- `[rewards]` a series of reward columns which evaluate the prompt + response e.g. reward-helpful and reward-harmless for the HH dataset

### Trainer

The standard `pytorch-lightning` training class. Set various arguments such as the GPU to use, the number of steps to train for in this part of the config

### Dataloader

A standard `pytorch` dataloader class for the train, val, and test set

### Collate_fn

The collate function takes a batch returned by the dataloader and tokenizes it. It also reformats the reward into a per token reward of the same size as the tokenized batch. One collate_fn reward type is currently implemented 'eos' which stands for end of sentence, this reformats a scalar reward into a per-token reward that is zero everywhere except the last entry of the sequence. 

### Model

The model class interfaces between the LLM and the rest of the code. Different options are available here including `robust_multi_objective_decoding.multi_objective_value_function.MultiModelValueFunction` which loads a model per objective vs. `robust_multi_objective_decoding.multi_objective_value_function.MultiHeadValueFunction` which loads a model with multiple output heads. Different default settings are provided in `configs/model` which can be used to override the default setting through the command line (see hydra documentation).

The specific huggingface model is also specified in this config e.g. gemma-2-2b-it 

### Tokenizer

The model tokenizer is loaded from huggingface, any specific tokenizer arguments can be provided here.

### Learner

The main `pytorch-lightning` module where the loss, training_step, validation_step, and other key pieces of code are implemented.

### Decoder

The Decoder is a class that implements the specific decoding method to be evaluated. Implementations of various decoding approaches can be found in `robust_multi_objective_decoding.decoders` including RMOD!

### Oracle

The reward model(s) that we use to evaluate our output generations.
  
