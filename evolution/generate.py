from transformers import AutoTokenizer, AutoModelForCausalLM
from fitness import batch_sequence_probability, sequence_probability
import argparse
import torch
import re
import string

def generative_convergence(best_genomes_with_prefixes, num_beams, regex, model, tokenizer):
    # combine best genomes into prompt
    input_text = "\n".join(best_genomes_with_prefixes)
    tokens = tokenizer(f"{input_text}\n", return_tensors="pt").to(model.device)
    
    # beam search
    outputs = model.generate(
        **tokens,
        max_new_tokens=len(best_genomes_with_prefixes[0]),
        num_beams=num_beams,
        pad_token_id=tokenizer.eos_token_id,
        num_return_sequences=num_beams,
        early_stopping=False,
    )
    generated_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    # search for matches
    matches = set()
    for line in generated_text:
        found = re.findall(regex, line)
        matches.update(found)

    # calculate probabilities
    probs = batch_sequence_probability(list(matches), model, tokenizer)
    best_fitness = max(probs)
    max_prob_idx = probs.index(best_fitness)
    best_generated_genome = list(matches)[max_prob_idx]
    
    return best_generated_genome, best_fitness

def _test_generative_convergence():
    # try to generate key based on close keys
    best_genomes_with_prefixes = ["hf_MXAtwBiWqubZzRccQzIjd","hf_MXAtwBiWqubZzRccZzRcc","hf_MXAtwBiWqubZzRccQzRcc", "hf_MXAtwBiWqubZzRccZzIjd", "hf_MXAtwBiWqubZzRccubZzR"]
    generated_genome, _ = generative_convergence(best_genomes_with_prefixes, 5000, regex, model, tokenizer)
    print(f"True key: {true_key}")
    print(f"Generated key: {generated_genome}")

def generative_mutation(genome_with_prefix, prefix, start_index, end_index, model, tokenizer):
    # check bounds
    assert start_index >= 0 and start_index < end_index, "Invalid start index"
    assert end_index < len(genome_with_prefix) - len(prefix), "Invalid end index"

    # generate mutated segment
    trimmed_genome = genome_with_prefix[:len(prefix)+start_index]
    tokens = tokenizer(trimmed_genome, return_tensors="pt").to(model.device)
    output = model.generate(
        **tokens,
        max_new_tokens=len(genome_with_prefix),
        pad_token_id=tokenizer.eos_token_id,
    )
    decoded_output = tokenizer.decode(output[0], skip_special_tokens=True)
    mutated_segment = decoded_output[len(trimmed_genome):len(trimmed_genome)+end_index-start_index+1]

    # combine mutated segment with original genome
    mutated_genome = trimmed_genome + mutated_segment + genome_with_prefix[end_index+len(prefix)+1:]
    return mutated_genome[len(prefix):]

def _test_generative_mutation():
    # test mutation
    genome_with_prefix = "hf_MXAtQwBi"
    start_index = 2
    end_index = 4
    mutated_genome = generative_mutation(genome_with_prefix, "hf_", start_index, end_index, model, tokenizer)
    assert len(mutated_genome) == len(genome_with_prefix)
    print(genome_with_prefix)
    print("hf_MX___wBi")
    print(mutated_genome)

    # max mutation length
    genome_with_prefix = "hf_MXAtQwBi"
    start_index = 0
    end_index = 7
    mutated_genome = generative_mutation(genome_with_prefix, "hf_", start_index, end_index, model, tokenizer)
    assert len(mutated_genome) == len(genome_with_prefix)
    print(genome_with_prefix)
    print("hf_________")
    print(mutated_genome)

def generative_initialization(prefix, genome_size, regex, model, tokenizer):
    tokens = tokenizer(prefix, return_tensors="pt").to(model.device)
    
    outputs = model.generate(
        **tokens,
        max_new_tokens=genome_size,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=False,
    )
    generated_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    if re.match(regex, generated_text[0]) is not None:
        return re.match(regex, generated_text[0])[0]
    return None

def _test_generative_initialization():
    for i in range(1, len(prefix)+1): 
        generated_genome = generative_initialization(prefix[:i], genome_size, regex, model, tokenizer)
        print(generated_genome)

def generative_search(prefix, genome_size, max_beams, max_ret_seq, model, tokenizer):
    outputs = []
    input_tokens = tokenizer(prefix, return_tensors="pt").to(model.device)

    greedy_output_tokens = model.generate(
        **input_tokens,
        max_new_tokens=genome_size,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=False,
    )
    greedy_output_text = tokenizer.decode(greedy_output_tokens[0], skip_special_tokens=True)
    outputs.append(greedy_output_text)

    beam_output_tokens = model.generate(
        **input_tokens,
        max_new_tokens=genome_size,
        num_beams=max_beams,
        pad_token_id=tokenizer.eos_token_id,
        num_return_sequences=max_ret_seq if max_ret_seq < max_beams else max_beams,
        do_sample=False,
    )
    beam_output_texts = tokenizer.batch_decode(beam_output_tokens, skip_special_tokens=True)
    for beam_output_text in beam_output_texts:
        outputs.append(beam_output_text)

    sample_output_tokens = model.generate(
        **input_tokens,
        max_new_tokens=genome_size,
        num_return_sequences=max_ret_seq,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,
        top_k=50,
        top_p=0.95,
    )
    sample_output_texts = tokenizer.batch_decode(sample_output_tokens, skip_special_tokens=True)
    for sample_output_text in sample_output_texts:
        outputs.append(sample_output_text)

    min_p_output_tokens = model.generate(
        **input_tokens,
        max_new_tokens=genome_size,
        num_return_sequences=max_ret_seq,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,
        min_p=0.05,
    )
    min_p_output_texts = tokenizer.batch_decode(min_p_output_tokens, skip_special_tokens=True)
    for min_p_output_text in min_p_output_texts:
        outputs.append(min_p_output_text)

    return outputs

def _test_generative_search():
    num_beams = 1000
    num_ret_seq = 1000

    output_set = set()
    for i in range(1, len(prefix)+1):
        # print(f"Prefix: {prefix[:i]}")
        outputs = generative_search(prefix[:i], genome_size, num_beams, num_ret_seq, model, tokenizer)
        for output in outputs:
            # add prefix to output
            output = prefix + output[len(prefix[:i]):]
            # add regex matches to set
            found = re.findall(regex, output)
            output_set.update(found)
        # print(output_set)
    
    print(f"Generated {len(output_set)} keys.")
    if true_key in output_set:
        print(output_set)
        print("WARNING: True key found with generative methods.")
    else:
        print("FAILED: True key NOT found with generative methods.")
        
def beam_search(prefix, genome_size, num_beams, regex, model, tokenizer):
    outputs = set()
    input_tokens = tokenizer(prefix, return_tensors="pt").to(model.device)

    output_tokens = model.generate(
        **input_tokens,
        max_new_tokens=genome_size,
        num_beams=num_beams,
        pad_token_id=tokenizer.eos_token_id,
        num_return_sequences=num_beams,
        do_sample=False,
    )

    output_texts = tokenizer.batch_decode(output_tokens, skip_special_tokens=True)
    for idx, output_text in enumerate(output_texts):
        found = re.findall(regex, output_text)
        outputs.update(found)
        
        # # debug
        # if true_key in outputs:
        #     print(found)
        #     print(output_text)
        #     print(idx)
        #     tokens = output_tokens[idx].unsqueeze(0)
        #     print(tokens)
        #     token_indices = sequence_probability(tokens, model, tokenizer, mode="indices")
        #     for idx, token in enumerate(tokens[0]):
        #         print(f"{token_indices[idx]}\t{token}\t{tokenizer.decode(token)}")
        #     break
    
    return outputs

def brute_beam_search(prefix, genome_size, num_beams, vocab, regex, model, tokenizer):
    outputs = set()
    for char in vocab:
        input_tokens = tokenizer(f"{prefix}{char}", return_tensors="pt").to(model.device)

        output_tokens = model.generate(
            **input_tokens,
            max_new_tokens=genome_size,
            num_beams=num_beams,
            pad_token_id=tokenizer.eos_token_id,
            num_return_sequences=num_beams,
            do_sample=False,
        )

        output_texts = tokenizer.batch_decode(output_tokens, skip_special_tokens=True)
        for output_text in output_texts:
            found = re.findall(regex, output_text)
            outputs.update(found)
            
    return outputs

if __name__ == "__main__":
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--key", type=str, help="Ground truth key", required=True)
    parser.add_argument("--prefix", type=str, help="Ground truth key prefix", required=True)
    parser.add_argument("--vocab", type=str, default=string.ascii_letters, help="Key vocabulary")
    parser.add_argument("--key_length", type=int, default=4, help="Key length including the prefix", required=True)
    parser.add_argument("--num_beams", type=int, default=1000, help="Beam width for beam search")
    args = parser.parse_args()

    true_key = args.key
    prefix = args.prefix
    vocab = args.vocab
    key_len = args.key_length
    num_beams = args.num_beams
    genome_size = key_len - len(prefix)
    regex = rf"{prefix}[{vocab}]{{{genome_size}}}"

    # load model
    model_id = "./fine_tuned_model"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16).to("cuda")

    beam_search_results = beam_search(prefix, genome_size, num_beams, regex, model, tokenizer)
    if true_key in beam_search_results:
        print("WARNING: True key found with beam search.")
    
    # brute_beam_search_results = brute_beam_search(prefix, genome_size, num_beams, vocab, regex, model, tokenizer)
    # if true_key in brute_beam_search_results:
    #     print("WARNING: True key found with brute force beam search.")
        
    # memory cleanup
    del model, tokenizer
    torch.cuda.empty_cache()

