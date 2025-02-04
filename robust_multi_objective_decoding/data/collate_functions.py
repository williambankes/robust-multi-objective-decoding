import torch
import numpy as np

from typing import List, Tuple
from transformers import AutoTokenizer


def pad_safety_labels(labels:List[int],
                    len_tok_response:int,
                    len_tok_prompt:int, 
                    padding_side:str,
                    truncation_side:str,
                    max_shape:int,
                    mask_token:int = -100) -> List[int]:
    """
    Take labels sized tokenizer(response) - 1 (for bos token) and pad them to the size of the tokenized response.
    We pad the prompt and eos padding tokens with mask_token.

    labels:
        The safety labels for the response. These are adjusted to the size of the tokenized response
        minus <bos> e.g. tokenizer(response)['input_ids'].shape[-1] - 1.
    len_tok_response:
        The length of the tokenized response including <bos> token e.g. tokenizer(response)['input_ids'].shape[-1]
    len_tok_prompt:
        The length of the tokenized prompt including <bos> token e.g. tokenizer(prompt)['input_ids'].shape[-1]
    padding_side:
        The side to pad the labels on. Either 'right' or 'left'.
    truncation_side:
        The side to truncate the labels on. Either 'right' or 'left'.
    max_shape:
        The maximum shape of the padded labels. This is the size of the batch.
    mask_token:
        The token used to pad the prompt. Default is -100.
    """

    # Checks:
    assert padding_side in ['right', 'left'],\
        f'padding_side: {padding_side} must be either right or left.'
    assert truncation_side in ['right', 'left'],\
        f'truncation_side: {truncation_side} must be either right or left.'
    
    # We first pad the prompt component of the input:
    # Remove the <bos> token included in the toeknization of this term
    # Leave this in the prompt as we include it in the padding anyway. 
    #len_tok_response -= 1 

    # Pad the prompt component:
    label_output = [mask_token] * len_tok_prompt + labels

    # Pad to the size of the batch:
    
    # Truncate if too big:
    if len(label_output) > max_shape:
        
        if truncation_side == 'right':
            label_output = label_output[:max_shape]
        else:
            label_output = label_output[-max_shape:]
    
    # Add additional padding if too small: 
    elif len(label_output) < max_shape:

        if padding_side == 'right':
            label_output = label_output + [mask_token] * (max_shape - len(label_output))
        else:
            label_output = [mask_token] * (max_shape - len(label_output)) + label_output

    return torch.tensor(label_output)


def adjust_label_to_token_length(data: str, label:List[int], tokenizer: AutoTokenizer) -> List[int]:

    """
    Takes a sentence and a per word safety label and adjusts the label to be the length of the tokenized sentence.
    The code assumes the safety labels transition from safe to unsafe once and remain unsafe. We don't include the <bos>
    token in the returned label. The label should be len(tokenizer(full_prompt)) - 1 in length.
    """

    # For each element process the dataset:
    label = np.array(label)
    indices = np.flatnonzero(label == 1)

    # If there are 1s in the label:
    if indices.size > 0:

        # Find the last index of 1:
        last_one_index = indices[-1]
        
        # Split and tokenize the safe sentence and unsafe sentence:
        split_sentence = data.split()[:last_one_index + 1] # +1 to include the last word
        tok_split_sentence = tokenizer([' '.join(split_sentence)], return_tensors='pt')
        tok_full_sentence = tokenizer([data], return_tensors='pt')
        
        # Get the shape of the tokenized inputs:
        split_shape = tok_split_sentence['input_ids'].shape[1]
        full_shape = tok_full_sentence['input_ids'].shape[1]
        diff = full_shape - split_shape

        # Create a new label based on the tokenized inputs:
        adjusted_label = [1] * (split_shape - 1) + [0] * diff #subtract 1 to account for <bos> token
        
    # If there are only 0s in the label:
    else:

        tok_full_sentence = tokenizer([data], return_tensors='pt')
        adjusted_label = [0] * (len(tok_full_sentence['input_ids'][0]) - 1) #subtract 1 to account for <bos> token

    return adjusted_label


def create_tokenizer_collate_fn(tokenizer, max_length:int):
    """
    TODO: More thorough implementation of padding to be created that uses the 
    tokenizer pad_side, pad_token and truncation_side options. Shouldn't be difficult... 
    """


    def collate_fn(batch:List[Tuple]) -> Tuple[torch.Tensor]:

        idx, prompt, data, labels = zip(*batch)
        tokenized_inputs = tokenizer(data, padding=True, truncation=True,
                                     max_length=max_length, return_tensors="pt")

        #Pad the labels with -1:
        max_size = tokenized_inputs.input_ids.shape[1]
        output_labels = list()
        for i, l in enumerate(labels):

            length_l = len(l)
            if length_l < max_size:
                output_labels.append(torch.tensor([-1]*(max_size - length_l) + l))
            elif length_l > max_size:
                output_labels.append(torch.tensor(l[:max_size]))
            else:
                output_labels.append(torch.tensor(l))

        tokenized_inputs["tokenwise_safety_labels"] = torch.stack(output_labels)

        tokenized_prompts = tokenizer(prompt, padding=True, truncation=True,
                                      max_length=max_length, return_tensors="pt")
        tokenized_inputs['prompt_input_ids'] = tokenized_prompts['input_ids']
        tokenized_inputs['prompt_attention_mask'] = tokenized_prompts['attention_mask'] 

        return tokenized_inputs

    return collate_fn

def create_word_break_tokenizer_collate_fn(tokenizer, max_length:int):
    """
    Create a tokenizer that adapts the collate function to handle labelling by the number of words. 
    """

    def collate_fn(batch:List[Tuple]) -> Tuple[torch.Tensor]:

        idx, prompt, response, labels = zip(*batch)

        tokenized_prompt_lens = [tokenizer(p, return_tensors='pt')['input_ids'].shape[-1] for p in prompt]
        tokenized_response_lens = [tokenizer(r, return_tensors='pt')['input_ids'].shape[-1] for r in response]

        # Join the prompts and responses:
        prompt_response = [prompt[i] + resp for i, resp in enumerate(response)]
        tokenized_inputs = tokenizer(prompt_response, padding=True, truncation=True,
                                      max_length=max_length, return_tensors="pt")
        
        labels = [adjust_label_to_token_length(response[i], l, tokenizer) for i, l in enumerate(labels)]
        labels = [pad_safety_labels(lable,
                    tokenized_response_lens[i],
                    tokenized_prompt_lens[i],
                    tokenizer.padding_side,
                    tokenizer.truncation_side,
                    tokenized_inputs['input_ids'].shape[1]) for i, lable in enumerate(labels)]

        # Move to torch tensors:
        tokenized_inputs["tokenwise_safety_labels"] = torch.stack(labels)
        tokenized_inputs['idx'] = torch.tensor(idx)

        # Tokenize only the prompts:
        tokenized_prompts = tokenizer(prompt, padding=True, truncation=True,
                                      max_length=max_length, return_tensors="pt")
        tokenized_inputs['prompt_input_ids'] = tokenized_prompts['input_ids']
        tokenized_inputs['prompt_attention_mask'] = tokenized_prompts['attention_mask']

        return tokenized_inputs

    return collate_fn

def create_eos_reward_tokenizer_collate_fn(tokenizer, max_length:int):
    """
    Create a collate function that handles the setting where only a single reward is given for each response 
    at the end of the response. 
    """

    def collate_fn(batch:List[Tuple]) -> dict:
        """
        The collate function does the following to the labels:
        1. Pad the labels to the correct size
        2. If the labels ends with a 1 set the last element to 1, otherwise set it to -1
        3. All other elements of the labels are set to 0.

        Parameters
        ----------
        batch : List[Tuple]
            A list of tuples containing the index, prompt, response and labels.

        Returns
        -------
        Dict[torch.Tensor]
            A dictionary containing the tokenized inputs, the tokenized prompt, 
            the prompt input ids, the prompt attention mask and the tokenwise safety labels.            
        """


        idx, prompt, response, labels = zip(*batch)

        tokenized_prompt_lens = [tokenizer(p, return_tensors='pt')['input_ids'].shape[-1] for p in prompt]
        tokenized_response_lens = [tokenizer(r, return_tensors='pt')['input_ids'].shape[-1] for r in response]

        # Join the prompts and responses:
        prompt_response = [prompt[i] + resp for i, resp in enumerate(response)]
        tokenized_inputs = tokenizer(prompt_response, padding=True, truncation=True,
                                      max_length=max_length, return_tensors="pt")
        
        # Adapt the labels:
        eos_labels = list() 
        for i, label in enumerate(labels):

            # Set the last element to be 1 or -1 depending on the last element of the label:

            eos_label = [0]*len(label)
            eos_label[-1] = 1 if label[-1] == 1 else -1

            eos_labels.append(pad_safety_labels(eos_label, 
                                                tokenized_response_lens[i],
                                                tokenized_prompt_lens[i],
                                                tokenizer.padding_side,
                                                tokenizer.truncation_side,
                                                tokenized_inputs['input_ids'].shape[1]))
            
        # Move to torch tensors:
        tokenized_inputs["tokenwise_safety_labels"] = torch.stack(eos_labels)
        tokenized_inputs['idx'] = torch.tensor(idx)

        # Tokenize only the prompts:
        tokenized_prompts = tokenizer(prompt, padding=True, truncation=True,
                                      max_length=max_length, return_tensors="pt")
        tokenized_inputs['prompt_input_ids'] = tokenized_prompts['input_ids']
        tokenized_inputs['prompt_attention_mask'] = tokenized_prompts['attention_mask']

        return tokenized_inputs

    return collate_fn

def create_eos_reward_tokenizer_rand_len_collate_fn(tokenizer, max_length:int):
    """
    Create a collate function that handles the setting where only a single reward is given for each response 
    at the end of the response. Adjust max_new_tokens=128 responses to a random length between 72 and 128.
    """

    def collate_fn(batch:List[Tuple]) -> dict:
        """
        The collate function does the following to the labels:
        1. Pad the labels to the correct size
        2. If the labels ends with a 1 set the last element to 1, otherwise set it to -1
        3. All other elements of the labels are set to 0.

        Parameters
        ----------
        batch : List[Tuple]
            A list of tuples containing the index, prompt, response and labels.

        Returns
        -------
        Dict[torch.Tensor]
            A dictionary containing the tokenized inputs, the tokenized prompt, 
            the prompt input ids, the prompt attention mask and the tokenwise safety labels.            
        """

        idx, prompt, response, labels = zip(*batch)

        # Generate random lengths to truncate the responses to
        rand_lens = [np.random.randint(72, 128) for _ in response]

        tokenized_prompt_lens = [tokenizer(p, return_tensors='pt')['input_ids'].shape[-1] for p in prompt]
        tokenized_response_lens = rand_lens #check about bos token being included here

        response = [tokenizer(r, return_tensors='pt', truncation=True, max_length=rand_lens[i]) for i, r in enumerate(response)]
        response = [tokenizer.decode(r['input_ids'][0], skip_special_tokens=True) for r in response]
    
        # Join the prompts and responses:
        prompt_response = [prompt[i] + resp for i, resp in enumerate(response)]
        tokenized_inputs = tokenizer(prompt_response, padding=True, truncation=True,
                                      max_length=max_length, return_tensors="pt")
        
        # Adapt the labels:
        eos_labels = list() 
        for i, label in enumerate(labels):

            # Set the last element to be 1 or -1 depending on the last element of the label:
            eos_label = [0]*rand_lens[i]
            padded_label = pad_safety_labels(eos_label, 
                                                tokenized_response_lens[i],
                                                tokenized_prompt_lens[i],
                                                tokenizer.padding_side,
                                                tokenizer.truncation_side,
                                                tokenized_inputs['input_ids'].shape[1])

            padded_label[-1] = 1 if label[-1] == 1 else -1
            eos_labels.append(padded_label)
            
        # Move to torch tensors:
        tokenized_inputs["tokenwise_safety_labels"] = torch.stack(eos_labels)
        tokenized_inputs['idx'] = torch.tensor(idx)

        # Tokenize only the prompts:
        tokenized_prompts = tokenizer(prompt, padding=True, truncation=True,
                                      max_length=max_length, return_tensors="pt")
        tokenized_inputs['prompt_input_ids'] = tokenized_prompts['input_ids']
        tokenized_inputs['prompt_attention_mask'] = tokenized_prompts['attention_mask']

        return tokenized_inputs

    return collate_fn

def create_word_break_tokenizer_rand_len_collate_fn(tokenizer, max_length:int):
    """
    Create a tokenizer that adapts the collate function to handle labelling by the number of words. 

    Varies the length of the response to be between 72 and 128 tokens.
    """

    def collate_fn(batch:List[Tuple]) -> Tuple[torch.Tensor]:

        idx, prompt, response, labels = zip(*batch)

        # Define some random lengths
        rand_lens = [np.random.randint(72, 128) for _ in response]

        # Apply rand length to response:
        response = [tokenizer(r, return_tensors='pt', truncation=True, max_length=rand_lens[i]) for i, r in enumerate(response)]
        response = [tokenizer.decode(r['input_ids'][0], skip_special_tokens=True) for r in response]

        # Trim the label based on num of remaining words in the trimmed response:        
        labels = [labels[i][:len(resp.split())] for i, resp in enumerate(response)]

        tokenized_prompt_lens = [tokenizer(p, return_tensors='pt')['input_ids'].shape[-1] for p in prompt]
        tokenized_response_lens = rand_lens 

        # Join the prompts and responses:
        prompt_response = [prompt[i] + resp for i, resp in enumerate(response)]
        tokenized_inputs = tokenizer(prompt_response, padding=True, truncation=True,
                                      max_length=max_length, return_tensors="pt")
        
        # adjust_label_to_token_length -> should be run before trimming the response 
        labels = [adjust_label_to_token_length(response[i], el, tokenizer) for i, el in enumerate(labels)]
        labels = [pad_safety_labels(lable,
                    tokenized_response_lens[i],
                    tokenized_prompt_lens[i],
                    tokenizer.padding_side,
                    tokenizer.truncation_side,
                    tokenized_inputs['input_ids'].shape[1]) for i, lable in enumerate(labels)]
        
        # Move to torch tensors:
        tokenized_inputs["tokenwise_safety_labels"] = torch.stack(labels)
        tokenized_inputs['idx'] = torch.tensor(idx)

        # Tokenize only the prompts:
        tokenized_prompts = tokenizer(prompt, padding=True, truncation=True,
                                      max_length=max_length, return_tensors="pt")
        tokenized_inputs['prompt_input_ids'] = tokenized_prompts['input_ids']
        tokenized_inputs['prompt_attention_mask'] = tokenized_prompts['attention_mask']

        return tokenized_inputs

    return collate_fn


def create_classifier_collate_fn(tokenizer, max_length:int):
    """
    Create a tokenizer that adapts the collate function to create classifier labels. 
    """

    def collate_fn(batch:List[Tuple]) -> Tuple[torch.Tensor]:

        idx, prompt, response, labels = zip(*batch)

        tokenized_prompt_lens = [tokenizer(p, return_tensors='pt')['input_ids'].shape[-1] for p in prompt]
        tokenized_response_lens = [tokenizer(r, return_tensors='pt')['input_ids'].shape[-1] for r in response]

        # Join the prompts and responses:
        prompt_response = [prompt[i] + resp for i, resp in enumerate(response)]
        tokenized_inputs = tokenizer(prompt_response, padding=True, truncation=True,
                                      max_length=max_length, return_tensors="pt")
        
        # Create binary classifier labels from the last label of the tokenwise oracle labels:
        labels = torch.tensor([label[-1] for label in labels])

        # Move to torch tensors:
        tokenized_inputs["safety_labels"] = labels
        tokenized_inputs['idx'] = torch.tensor(idx)

        # Tokenize only the prompts:
        tokenized_prompts = tokenizer(prompt, padding=True, truncation=True,
                                      max_length=max_length, return_tensors="pt")
        tokenized_inputs['prompt_input_ids'] = tokenized_prompts['input_ids']
        tokenized_inputs['prompt_attention_mask'] = tokenized_prompts['attention_mask']

        return tokenized_inputs

    return collate_fn

def create_classifier_rand_len_collate_fn(tokenizer, max_length:int):
    """
    Create a tokenizer that adapts the collate function to create classifier labels. 
    """

    def collate_fn(batch:List[Tuple]) -> Tuple[torch.Tensor]:

        idx, prompt, response, labels = zip(*batch)

         # Define some random lengths
        rand_lens = [np.random.randint(72, 128) for _ in response]

        # Apply rand length to response:
        response = [tokenizer(r, return_tensors='pt', truncation=True, max_length=rand_lens[i]) for i, r in enumerate(response)]
        response = [tokenizer.decode(r['input_ids'][0], skip_special_tokens=True) for r in response]

        # Trim the label based on num of remaining words in the trimmed response:        
        labels = [labels[i][:len(resp.split())] for i, resp in enumerate(response)]

        tokenized_prompt_lens = [tokenizer(p, return_tensors='pt')['input_ids'].shape[-1] for p in prompt]
        tokenized_response_lens = rand_lens

        # Join the prompts and responses:
        prompt_response = [prompt[i] + resp for i, resp in enumerate(response)]
        tokenized_inputs = tokenizer(prompt_response, padding=True, truncation=True,
                                      max_length=max_length, return_tensors="pt")
        
        # Create binary classifier labels from the last label of the tokenwise oracle labels:
        labels = torch.tensor([label[-1] for label in labels])

        # Move to torch tensors:
        tokenized_inputs["safety_labels"] = labels
        tokenized_inputs['idx'] = torch.tensor(idx)

        # Tokenize only the prompts:
        tokenized_prompts = tokenizer(prompt, padding=True, truncation=True,
                                      max_length=max_length, return_tensors="pt")
        tokenized_inputs['prompt_input_ids'] = tokenized_prompts['input_ids']
        tokenized_inputs['prompt_attention_mask'] = tokenized_prompts['attention_mask']

        return tokenized_inputs

    return collate_fn

def create_collate_functions(tokenizer, reward_collates:list, max_length:int, rand_len:bool):
    """
    Create the collate functions for the different types of tokenizer and different reward_collate_functions.

    Parameters
    ----------
    tokenizer : AutoTokenizer
        The tokenizer to use.
    reward_collates : list
        The list of reward collate functions to use.
    max_length : int
        The maximum length of the tokenized inputs.
    rand_len : bool
        Whether to use random lengths for the responses.

    Returns
    -------
    collate_fn
        The collate function to use.    
    """

    if rand_len:
        collate_fn = create_word_break_tokenizer_rand_len_collate_fn(tokenizer, max_length)
    else:
        collate_fn = create_word_break_tokenizer_collate_fn(tokenizer, max_length)

    return collate_fn