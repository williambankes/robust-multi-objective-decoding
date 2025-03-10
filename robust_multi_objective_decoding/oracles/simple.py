from robust_multi_objective_decoding.oracles.oracle import Oracle


class CharacterLengthOracle(Oracle):
    """gives a score of 1 if the response is within the maximum character length and 0 otherwise"""

    def __init__(self, max_length: int):
        self.max_length = max_length

    def is_admissible(self) -> bool:
        return True

    def score(self, prompt: str, response: str) -> int:
        return 1 if len(response) <= self.max_length else 0


class PunctuationOracle(Oracle):
    """gives a score of 1 if the response ends with appropriate punctuation mark"""

    def is_admissible(self) -> bool:
        """Punctuation is not admissible because it depends on future state"""
        return False

    def score(self, prompt: str, response: str) -> int:
        lastchar = response[-1]
        return 1 if lastchar in [".", "!", "?"] else 0
    