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

To train a value function on the  for controlled decoding run the following
```bash

```


# 

