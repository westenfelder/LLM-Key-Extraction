import torch
from scipy.stats import entropy
from collections import Counter
import numpy as np
from tqdm import tqdm

def hamming_distance(str1, str2):
    if len(str1) != len(str2):
        raise ValueError("Strings must be of the same length")
    return sum(c1 != c2 for c1, c2 in zip(str1, str2))

def batch_entropy(strings, verbose=False):
    entropies = []
    iterator = tqdm(strings) if verbose else strings
    for string in iterator:
        if not string:
            entropies.append(0)
            continue
        counts = Counter(string)
        probabilities = np.array(list(counts.values())) / len(string)
        single_entropy = entropy(probabilities, base=2)
        entropies.append(single_entropy)
    return entropies

def batch_perplexity(texts, model, tokenizer, verbose=False, batch_size=1000):
    # note batch size needs to be adjusted based on GPU memory
    tokenizer.pad_token = tokenizer.eos_token
    batches = [texts[i:i+batch_size] for i in range(0, len(texts), batch_size)]
    perplexities = []
    iterator = tqdm(batches) if verbose else batches
    for batch in iterator:
        tokens = tokenizer(batch, return_tensors="pt", padding=True, truncation=True).to(model.device)

        with torch.no_grad():
            outputs = model(**tokens)

        # log probabilities used to avoid underflow
        log_probabilities = torch.log_softmax(outputs.logits, dim=-1).detach()

        # probability at index 0 corresponds to the token at index 1
        log_probabilities = log_probabilities[:, :-1, :]
        tokens_shifted = tokens.input_ids[:, 1:]  # shift input_ids
        attention_mask_shifted = tokens.attention_mask[:, 1:] # shift the attention mask

        # gather log probabilities corresponding to the actual tokens
        token_log_probabilities = torch.gather(
            log_probabilities, 2, tokens_shifted[:, :, None]
        ).squeeze(-1)

        assert token_log_probabilities.shape == tokens_shifted.shape

        # apply the attention mask to ignore padding tokens
        token_log_probabilities = token_log_probabilities * attention_mask_shifted

        # sum log probabilities for each sequence in the batch
        sequence_log_probabilities = torch.sum(token_log_probabilities, dim=1)

        # calculate the sequence lengths, accounting for padding
        sequence_lengths = torch.sum(attention_mask_shifted, dim=1)

        # scale by sequence length
        scaled_sequence_probabilities = sequence_log_probabilities / sequence_lengths

        perplexities.extend(torch.exp(-scaled_sequence_probabilities).tolist())
    
    return perplexities

def sequence_probability(tokens, model, tokenizer, mode="sequence"):
    verbose = False

    with torch.no_grad():
        outputs = model(tokens)

    # log probabilities used to avoid underflow
    log_probabilities = torch.log_softmax(outputs.logits, dim=-1).detach()

    # probability at index 0 corresponds to the token at index 1
    log_probabilities = log_probabilities[:, :-1, :]
    tokens = tokens[:, 1:]
    token_log_probabilities = torch.gather(log_probabilities, 2, tokens[:, :, None]).squeeze(-1)
    assert token_log_probabilities.shape == tokens.shape

    # token probability distributions
    if mode == 'dist':
        vocab_size = log_probabilities.shape[2]
        # add probability of first token
        nan_row = torch.full((1, 1, vocab_size), float(0.0)).to(model.device)
        extended_log_probabilities = torch.cat((nan_row, log_probabilities), dim=1)
        sorted_log_probabilities, sorted_indices = torch.sort(extended_log_probabilities.squeeze(0), dim=1)
        return sorted_log_probabilities

    # probability of each token conditioned on the previous tokens
    if mode == 'tokens':
        # add probability of first token
        first_token_probability = torch.tensor([[0.0]]).to(model.device)
        extended_token_log_probabilities = torch.cat((first_token_probability, token_log_probabilities), dim=1)
        return extended_token_log_probabilities.squeeze(0)
    
    # probability of sequence
    if mode == 'sequence':
        # sum log probabilities to get total sequence probability
        sequence_log_probability = 0.0
        for token_id, token_log_probability in zip(tokens[0], token_log_probabilities[0]):
            if verbose: print(f"{token_id}\t{tokenizer.decode(token_id)}\t{token_log_probability}")
            sequence_log_probability += token_log_probability

        # scale by sequence length
        scaled_sequence_probability = sequence_log_probability.item() / len(tokens[0])
        return scaled_sequence_probability
    
    # indices of tokens
    if mode == 'indices':
        # sort tokens by most to least likely
        sorted_log_probabilities, sorted_indices = torch.sort(log_probabilities, dim=-1, descending=True)
        indices = [0]
        for index, token in enumerate(tokens[0]):
            location_in_distribution = torch.where(sorted_indices[0, index] == token)[0].item()
            indices.append(location_in_distribution)
        assert len(indices) == len(tokens[0])+1
        return indices
    
    else:
        raise ValueError(f"Unknown mode: {mode}")