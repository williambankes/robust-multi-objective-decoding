import torch
from robust_multi_objective_decoding.oracles.shield_gemma import HarmType
from robust_multi_objective_decoding.decoders.best_of_n_oracle_decoder import (
    BestOfNOracleDecoder)


def cbf_output_process(decoder_output, batch_size):

    for i in range(batch_size):

        output_dict = dict()
        if isinstance(decoder_output, dict):
            for k,v in decoder_output.items():
                if k in ['is_cbf_condition_met', 'is_safe_record', 'joint_branch_probs_record']:
                    output_dict[k] = v[:,i]
                else:
                    output_dict[k] = v[i].cpu().to(torch.float32).numpy() if isinstance(v[i], torch.Tensor) else v[i]
    
    return output_dict

############################# FIXTURES ############################# 

class ProxyLanguageModelUnSafeBestofNTests:

    def generate(self, input_ids,
                attention_mask,
                max_new_tokens,
                num_return_sequences,
                do_sample):
    
        # Return mix of good values != 3, and bad values == 3:
        output = torch.ones(num_return_sequences, input_ids.shape[1] + max_new_tokens)

        # Generate a random id:
        idx = torch.randint(0, num_return_sequences, (1,)).item()

        output[:, input_ids.shape[1]:] = 3
        #output[idx, input_ids.shape[1]:] = 2
        
        return output.repeat(input_ids.shape[0], 1) 
    
class ProxyTokenizer:

    def batch_decode(self, input_ids, *args, **kwargs):
        
        batch_output = list()
        for input_id in input_ids:
            batch_output.append(self.decode(input_id))
    
        return batch_output

    def decode(self, input_ids, *args, **kwargs):
        
        decoded_output = ''
        for id in input_ids:
            if id == 2:
                decoded_output += ' Safe'
            elif id == 3:
                decoded_output += ' Unsafe'
            elif id == 1:
                decoded_output += ' Prompt'
            else: raise NotImplementedError()

        return decoded_output  
    
class ProxySafetyOracle:

    harm_types = ['safe', 'danger', 'hate', 'play']

    def score(self, prompts, responses, *args, **kwargs):
        
        safety_scores = list()
        num_types = len(self.harm_types)

        for resp in responses:
            
            final_word = resp.split(' ')[-1]

            if final_word == 'Safe':
                safety_scores.append(torch.tensor([1.0]*num_types))
            elif final_word == 'Unsafe':
                safety_scores.append(torch.tensor([0.0]*num_types))
            elif final_word == 'Prompt':
                raise NotImplementedError()
            else: raise NotImplementedError()
        
        return torch.stack(safety_scores)
    
############################# TESTS #############################

def test_best_of_n_safety_oracle_timing_eval():
    """
    Here we test the best of n safety oracle decoder, when only unsafe generations are passed.

    Here we test that the setup always selects the safe sequences when the safety oracle is used.
    """

    
    model = ProxyLanguageModelUnSafeBestofNTests()
    tok = ProxyTokenizer()

    safety_oracle = ProxySafetyOracle()
    decoder = BestOfNOracleDecoder(reference_model=model, 
                                        oracle=safety_oracle,
                                        tokenizer=tok,
                                        num_branches=8,
                                        safety_prob_cutoff=0.5)

    inputs = {'input_ids': torch.ones(4, 6), 'attention_mask': torch.ones(4, 6)}

    output = decoder.generate(**inputs, max_new_tokens=4, return_dict_in_generate=True)
    
    cbf_output_process(output, 4)
    
    decoder = BestOfNOracleDecoder(reference_model=model, 
                                        oracle=safety_oracle,
                                        tokenizer=tok,
                                        num_branches=8,
                                        safety_prob_cutoff=0.5,
                                        oracle_batch_size=2)

    inputs = {'input_ids': torch.ones(4, 6), 'attention_mask': torch.ones(4, 6)}

    output = decoder.generate(**inputs, max_new_tokens=4, return_dict_in_generate=True)
    cbf_output_process(output, 4)


######################## FIXTURES FOR GEN LENGTH TEST ########################

class ProxyLanguageModel:

    def generate(self, input_ids,
                attention_mask,
                max_new_tokens,
                num_return_sequences,
                do_sample):
        
        output = torch.ones(num_return_sequences, max_new_tokens)

        # Adjust output here:
        if num_return_sequences == 1:
            output *= 2      # safe
        elif num_return_sequences == 2:
            output[0,:] *= 2 # safe
            output[1,:] *= 3 # unsafe
        else:
            output[:-2,:] *= 2  # safe
            output[-2,:] *= 3 # unsafe
            output[-1,:] *= 4 # boring

        # Repeat this sequence for every input
        output = output.repeat(input_ids.shape[0], 1)

        copied_inputs = input_ids.repeat_interleave(num_return_sequences, 0)
        output = torch.concat([copied_inputs, output], dim=1)

        return output
                
class ProxyTokenizer:

    def batch_decode(self, input_ids, *args, **kwargs):
        
        batch_output = list()
        for input_id in input_ids:
            batch_output.append(self.decode(input_id))
    
        return batch_output

    def decode(self, input_ids, *args, **kwargs):
        
        decoded_output = ''
        for id in input_ids:
            if id == 2:
                decoded_output += ' Safe'
            elif id == 3:
                decoded_output += ' Unsafe'
            elif id == 4:
                decoded_output += ' Boring'
            elif id == 1:
                decoded_output += ' Prompt'
            else: raise NotImplementedError()

        return decoded_output 
    
class ProxySafetyOracleSafeTests:

    harm_types = ['safe', 'danger', 'hate', 'play']

    def score(self, prompts, responses, return_probs=True):
        
        # Establish some standard bits:
        batch_size = len(prompts)
        len_harms = len(self.harm_types)
        assert len(prompts) == len(responses),\
            "Prompt and response batch sizes must match"
        
        outputs = list()
        for response in responses:
            if response.split()[-1] == 'Unsafe':
                outputs.append(torch.tensor([0.2]*len_harms))
            elif response.split()[-1] == 'Safe':
                outputs.append(torch.tensor([0.8]*len_harms))
            elif response.split()[-1] == 'Boring':
                outputs.append(torch.tensor([0.6]*len_harms))
            else: raise NotImplementedError(f'Unknown response: {response.split()[-1]}')

        return torch.concat(outputs, dim=0).reshape(-1, len_harms)