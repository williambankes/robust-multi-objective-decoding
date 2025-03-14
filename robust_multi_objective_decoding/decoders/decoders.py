from robust_multi_objective_decoding.multi_objective_value_function import (
    BaseMultiObjectiveValueFunction,
)
from robust_multi_objective_decoding.oracles.oracle import Oracle
from transformers import PreTrainedTokenizer

import hydra
import torch.nn as nn
from omegaconf import DictConfig


class DecoderHandler:
    @staticmethod
    def create_decoder(
        config: DictConfig,
        reference_model: nn.Module,
        value_function: BaseMultiObjectiveValueFunction,
        oracle: Oracle,
        tokenizer: PreTrainedTokenizer,
        ref_tokenizer: PreTrainedTokenizer = None,
    ):
        """
        Process the decoder dict config to return the appropriate decoder. Specific
        decoder implementations require different arguments e.g. the Best-of-N decoder
        requires an oracle instead of the value function. We handle
        these differences in this method, simplifiying the eval.py script.

        Parameters
        ----------
        config : DictConfig
            Hydra config dict for the decoder e.g. config.decoder
        reference_model : nn.Module
            Base LLM model
        value_function : BaseMultiObjectiveValueFunction
            A CBF used for decoding
        oracle : ShieldGemmaSafetyOracle
            A safety oracle model used in the Best-of-N decoder implementations
        tokenizer: PreTrainedTokenizer
            The tokenizer for the reference model
        """

        if (
            config._target_
            == "robust_multi_objective_decoding.decoders.best_of_n_safety_oracle_decoder.BestOfNOracleDecoder"
        ):
            return hydra.utils.instantiate(
                config,
                reference_model=reference_model,
                oracle=oracle,
                tokenizer=ref_tokenizer,
            )

        elif (
            config._target_
            == "robust_multi_objective_decoding.decoders.blockwise_robust_decoder2.BlockwiseRobustDecoder"
        ):
            return hydra.utils.instantiate(
                config,
                reference_model=reference_model,
                value_function=value_function,
                tokenizer=tokenizer,
                ref_tokenizer=ref_tokenizer,
                oracle=oracle,
            )

        elif (
            config._target_
            == "robust_multi_objective_decoding.decoders.multi_obj_controlled_decoder.MultiObjectiveControlledDecoder"
        ):
            return hydra.utils.instantiate(
                config,
                reference_model=reference_model,
                value_function=value_function,
                tokenizer=tokenizer,
                ref_tokenizer=ref_tokenizer,
            )

        else:
            raise NotImplementedError(
                f"Decoder {config._target_} not implemented in safe_decoding.decoders.decoders.py"
            )
