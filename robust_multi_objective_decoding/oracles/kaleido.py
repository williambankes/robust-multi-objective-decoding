import torch
import torch.nn as nn
from robust_multi_objective_decoding.oracles.oracle import Oracle
from robust_multi_objective_decoding.kaleido.KaleidoSys import KaleidoSys

TOP_VRD = ['Autonomy',
           'Right to life',
           'Justice',
           'Compassion',
           'Well-being',
           'Duty of care',
           'Respect',
           'Safety',
           'Right to property',
           'Responsibility']

class KaleidoOracle(Oracle, nn.Module):
    """
    Value Kaleidoscope model as an oracle
    """
    def __init__(self, model_name:str="tsor13/kaleido-xl",
                 mode='score', vrd_idx=[0], *args, **kwargs):
        """
        mode: 'score', 'relevance', 'valence'
               score is the product of relevance and valence.
        vrd_idx: list of indices of the VRD attributes to score or 'all'.
        """
        super().__init__()
        self.model_name = model_name
        self.model = KaleidoSys(model_name=self.model_name, use_tqdm=False)
        self.mode = mode
        if vrd_idx == 'all':
            self.l_vrd = TOP_VRD
        else:
            self.l_vrd = [TOP_VRD[i] for i in vrd_idx]

    def score(self, prompt:list[str], response:list[str]):
        """
        returns a list of scores for the prompt-response pairs
        """
        l_scores = []
        for vrd in self.l_vrd:
            with torch.no_grad():
                if self.mode == 'relevance':
                    relevance_v = self.model.get_relevance(response, 
                                                        ['Value']*len(response),
                                                        [vrd]*len(response))
                    relevance_v = relevance_v[:, [0]].detach().cpu().float()  # pick the probability of relevance
                    l_scores.append(relevance_v)
                elif self.mode == 'valence':
                    valence_v = self.model.get_valence(response, 
                                                    ['Value']*len(response),
                                                    [vrd]*len(response))
                    valence_v = valence_v[:, [0]].detach().cpu().float()  # pick the probability of "support"
                    l_scores.append(valence_v)
                elif self.mode == 'score':
                    # compute relevance
                    relevance_v = self.model.get_relevance(response, 
                                                        ['Value']*len(response),
                                                        [vrd]*len(response))
                    relevance_v = relevance_v[:, [0]].detach().cpu().float()  # pick the probability of relevance

                    # compute valence
                    valence_v = self.model.get_valence(response, 
                                                    ['Value']*len(response),
                                                    [vrd]*len(response))
                    valence_v = valence_v[:, [0]].detach().cpu().float()  # pick the probability of "support"

                    # compute score
                    score_v = relevance_v * valence_v
                    l_scores.append(score_v)

        return torch.cat(l_scores, dim=1)

    def is_admissible(self):
        return False

                