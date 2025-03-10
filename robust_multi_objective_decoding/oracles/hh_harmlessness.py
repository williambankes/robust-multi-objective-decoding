import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from robust_multi_objective_decoding.oracles.oracle import Oracle

class HHHarmlessnessOracle(Oracle, nn.Module):
    def __init__(
        self,
        device: str | torch.device = "cpu",
        cache_dir: str | None = None,
        max_length: int = 2048,
        model_load_path: str | None = None,
        torch_dtype: torch.dtype = torch.float16,
        device_map: str = "auto",
        **kwargs,
    ):
        super().__init__()
        self.device = device
        self.cache_dir = cache_dir
        self.max_length = max_length
        self.model_path = (
            'Ray2333/gpt2-large-harmless-reward_model' if model_load_path is None else model_load_path
        )
        self.torch_dtype = torch_dtype

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, cache_dir=cache_dir)

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=self.torch_dtype,
            cache_dir=cache_dir
        )

    def is_admissible(self) -> bool:
        return False

    def score(
        self,
        prompt_batch: list[str],
        response_batch: list[str],
    ) -> torch.tensor:


        n_prompts = len(prompt_batch)
        n_responses = len(response_batch)

        assert n_prompts == n_responses, f"Number of prompts:{n_prompts} and responses: {n_responses} must be equal"

        messages = [prompt + response for prompt, response in zip(prompt_batch, response_batch)]
        inputs = self.tokenizer(messages, return_tensors="pt", padding=True, max_length=self.max_length, truncation=True)

        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        assert outputs.shape == (n_prompts, 1), f"Expected output shape: {(n_prompts, 1)}, got: {outputs.shape}"

        return outputs
