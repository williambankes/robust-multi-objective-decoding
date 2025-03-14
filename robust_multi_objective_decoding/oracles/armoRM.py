import torch
import torch.nn as nn
import pickle

from typing import Union, List
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from robust_multi_objective_decoding.oracles.oracle import Oracle


class ArmoRMOracle(Oracle, nn.Module):
    """
    Attribute indices

    """

    attributes = [
        "helpsteer-helpfulness",
        "helpsteer-correctness",
        "helpsteer-coherence",
        "helpsteer-complexity",
        "helpsteer-verbosity",
        "ultrafeedback-overall_score",
        "ultrafeedback-instruction_following",
        "ultrafeedback-truthfulness",
        "ultrafeedback-honesty",
        "ultrafeedback-helpfulness",
        "beavertails-is_safe",
        "prometheus-score",
        "argilla-overall_quality",
        "argilla-judge_lm",
        "code-complexity",
        "code-style",
        "code-explanation",
        "code-instruction-following",
        "code-readability",
    ]

    def __init__(
        self,
        attribute_indices: List[int],
        device: str = "cpu",
        max_length: int = 2048,
        cache_dir: str | None = None,
        mode: Union[int, str] = "rewards",
        path_stats: str = None,
        debug: bool = False,
        *args,
        **kwargs,
    ):
        super().__init__()  # Initialize both RewardModel and nn.Module

        if not debug:
            self.model = (
                AutoModelForSequenceClassification.from_pretrained(
                    "RLHFlow/ArmoRM-Llama3-8B-v0.1",
                    trust_remote_code=True,
                    torch_dtype=torch.bfloat16,
                    cache_dir=cache_dir,
                    revision="main",
                )
                .half()
                .to(device)
            )
            self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(
            "RLHFlow/ArmoRM-Llama3-8B-v0.1",
            use_fast=True,
            revision="main",
            padding_side="left",
            truncation_side="left",
        )

        assert self.tokenizer.padding_side == "left", "Padding side must be left"

        self.device = device
        self.max_length = max_length
        self.attribute_indices = attribute_indices
        self.mode = mode
        self.idx_verbosity = 4

        # loading the normalisation stats
        if path_stats is not None:
            with open(path_stats, "rb") as fp:
                stats_loaded = pickle.load(fp)
            self.stats = {
                "means": stats_loaded["mean_rewards"][self.attribute_indices],
                "stds": stats_loaded["std_rewards"][self.attribute_indices],
            }
        else:
            self.stats = None

        # Do something here to process the attributes

    def is_admissible(self) -> bool:
        return False

    def output_mode(self, output):
        # self.mode="robust"
        # output.re=output[:, :5]
        if self.mode == "score":
            return output.score.cpu()
        elif self.mode == "rewards":
            # res = output.rewards[:, self.attribute_indices].cpu()
            res = output.rewards.cpu()

            # multiply -1 to verbosity to obtain conciseness
            res[:, self.idx_verbosity] = res[:, self.idx_verbosity] * -1
            res = res[:, self.attribute_indices]

            if self.stats is not None:
                res = (res - self.stats["means"]) / self.stats["stds"]
            return res

        elif isinstance(self.mode, int):
            return output.rewards[:, self.mode].cpu()
        elif self.mode == "both":
            return {
                "scores": output.score.cpu(),
                "rewards": output.rewards[:, self.attribute_indices].cpu(),
            }
        elif self.mode == "robust":
            return torch.min(output.rewards[:, self.attribute_indices]).cpu()
        else:
            raise NotImplementedError

    def score(
        self,
        prompt_batch: list[str],
        response_batch: list[str],
        eval_only_response: bool = False,
        mode: Union[int, str] = None,
    ):
        """
        Evaluate the prompts and responses lists using the ArmoRM reward model
        and the specified attributes from the __init__ call.

        Parameters
        ----------
        prompts : List[str]
            A list of prompt strings
        responses : List[str]
            A list of model response strings - only containing the model response
        eval_only_response : bool, optional
            Only evaluate the model response leaving the prompt input blank. The default is False.

        Returns
        -------
        torch.Tensor([B, A])
            A torch tensor the same size as the no. of prompts and responses and
            with a column for each specified attribute_index.
        """
        if mode is not None:
            self.mode = mode

        # Apply the ArmoRM model:
        inputs = self.prep_prompt_response_input(
            prompt_batch, response_batch, eval_only_response=eval_only_response
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        # Apply the ArmoRM model:
        with torch.no_grad():
            outputs = self.output_mode(self.model(**inputs))

        # Return only the relevant attribute indices:
        return outputs

    def prep_prompt_response_input(
        self, prompts: List[str], responses: List[str], eval_only_response: bool
    ) -> torch.Tensor:
        """
        Prepare the prompt, response lists for input into the ArmoRM model, this includes:
            - applying the chat template
            - tokenizing the input
            - ensuring the gating token is present - a requirement by ArmoRM

        Parameters
        ----------
        prompts : List[str]
            A list of prompt strings
        responses : List[str]
            A list of model response strings - only containing the model response
        eval_only_response : bool
            Only evaluate the model response leaving the prompt input blank.

        Returns
        -------
        torch.Tensor([int])
            A torch tensor of input ids for the model
        """

        assert (
            len(prompts) == len(responses)
        ), f"prompts ({len(prompts)}) and responses ({len(responses)}) must have the same length"

        # TODO investigate if an attention mask is the correct way to eval only the response...
        if eval_only_response:
            messages = [
                [
                    {"role": "user", "content": ""},
                    {"role": "assistant", "content": resp},
                ]
                for _, resp in zip(prompts, responses)
            ]
        else:
            messages = [
                [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": resp},
                ]
                for prompt, resp in zip(prompts, responses)
            ]

        # Tokenize the prompts and responses with the chat message format:
        inputs = self.tokenizer.apply_chat_template(
            messages,
            truncation=True,
            padding=True,
            return_dict=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        # Assert all inputs have the proper gating:
        for i, tokens in enumerate(inputs["input_ids"]):
            if not self.assert_token_for_gating(tokens):
                inputs["input_ids"][i, -5:] = torch.tensor(
                    [128009, 128006, 78191, 128007, 271]
                ).cpu()

        inputs["input_ids"] = inputs["input_ids"].to(torch.int64)

        return inputs

    def assert_token_for_gating(self, lst):
        """Find the last occurrence of a token_pattern in a list."""

        token_pattern = torch.tensor([128009, 128006, 78191, 128007, 271]).to(
            lst.device
        )

        token_pattern_len = len(token_pattern)
        search_end = len(lst)
        for j in range(search_end - token_pattern_len, -1, -1):
            if (lst[j : j + token_pattern_len] == token_pattern).all():
                return True
        return False
