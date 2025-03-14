from abc import ABC, abstractmethod
import torch.nn as nn


class Oracle(ABC):
    @abstractmethod
    def is_admissible(self) -> bool:
        """Return True if the oracle is admissible, False otherwise

        An oracle is 'admissible' if the score for a prefix should not change depending on the suffix
        - i.e. it should not be possible to recover from unsafe states by adding more tokens
        The assumption of admissibility is necessary for the theoretical guarantees of CBF learning to hold

        When starting from a safe state, this means that the scores will always be a sequence of 1s followed by a sequence of 0s
        """
        pass

    @abstractmethod
    def score(self, prompt: str, response: str) -> int:
        """Return a score for the prompt response pair"""
        pass


class MultiObjectiveOracle(Oracle, nn.Module):
    def __init__(self, oracles: list[Oracle]):
        """
        A class for managing multiple different oracles.

        n.b. nn.Module is included here to allow pytorch lightning to manage the devices of the oracles
        Parameters
        ----------
        oracles : list[Oracle]
            _description_
        """

        self.oracles = oracles
        super().__init__()

    def is_admissible(self) -> bool:
        return all(oracle.is_admissible() for oracle in self.oracles)

    def score(self, prompt: str, response: str) -> int:
        """
        Return a score for the prompt response pair of each oracle.

        Parameters
        ----------
        prompt : str
            A list of prompts
        response : str
            A list of responses to the prompt for eval

        Returns
        -------
        list
            A returned list of variables.
        """

        # oracles scores are shape [num_oracles, num_prompts, num_features]
        oracle_scores = [oracle.score(prompt, response) for oracle in self.oracles]
        prompt_scores = list(zip(*oracle_scores))

        return prompt_scores


class CompositeOracle(Oracle):
    """gives a score of 1 if all oracles return 1, and 0 otherwise"""

    def __init__(self, oracles: list[Oracle]):
        self.oracles = oracles

    def is_admissible(self) -> bool:
        # NOTE: As long as one of the child oracles is admissible, the composite oracle is admissible
        # This can be proved easily by contradiction
        # - Assume that at least one child oracle is admissible
        # - Suppose the composite oracle is not admissible, then there exists a suffix of an unsafe response which is safe
        # - This implies that the suffix is safe according to all oracles, so all child oracles are not admissible
        # - But this contradicts the assumption that at least one oracle is admissible
        # - QED
        return any(oracle.is_admissible() for oracle in self.oracles)

    def score(self, prompt: str, response: str) -> int:
        # product
        score = 1
        for oracle in self.oracles:
            score *= oracle.score(prompt, response)
        return score


def get_tokenwise_scores(
    prompt_tokens: list[str], response_tokens: list[str], oracle: Oracle
):
    """
    Get tokenwise scores for a given prompt and response using a safety oracle

    This function should only be used with admissible oracles.
    See `SafetyOracle.is_admissible()` for more details.
    """
    if not oracle.is_admissible():
        raise RuntimeError("Safety oracle must be admissible")

    scores = []
    prompt = " ".join(prompt_tokens)
    for i in range(len(response_tokens)):
        response = " ".join(response_tokens[: i + 1])
        scores.append(oracle.score(prompt, response))
    return scores
