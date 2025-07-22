import random
import math
import string
import time
import random
import torch
import numpy as np
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
from fitness import batch_sequence_probability, batch_string_entropy

def hamming(a, b):
    return sum(1 for i in range(len(a)) if a[i] != b[i])

def standard_fitness(population):
    # Add prefix to each genome
    key_population = []
    for genome in population:
        key_population.append(f"{prefix}{genome}")
    fitness = batch_sequence_probability(key_population, model, tokenizer, verbose=True, batch_size=1000)
    return fitness

def entropy_fitness(population):
    key_population = []
    for genome in population:
        key_population.append(f"{prefix}{genome}")
    fitnesses = np.array(batch_sequence_probability(key_population, model, tokenizer, verbose=True))
    entropies = np.array(batch_string_entropy(population, verbose=True))
    # avoid division by zero
    entropies[entropies == 0] = 1e-10
    fitness = fitnesses / entropies
    return fitness.tolist()

def erdos(n, c, vocab):
    search_space = len(vocab) ** n
    # scale the num samples by some constant c
    num_samples = int(c * n * math.log(n))
    print(f"search space: {n}^{len(vocab)}")
    print(f"num samples: {num_samples:,}")
    print(f"% of search space: {(num_samples/search_space)*100:.2e}")

    start_gen_samples = time.time()
    fitnesses = []
    samples = [None] * num_samples
    for i in range(num_samples):
        if sample_ham_dist == -1:
            samples[i] = ''.join(random.choices(vocab, k=n))
        else:
            candidate = ground_truth
            for j in range(sample_ham_dist):
                # pick a random index to change
                idx = random.randint(0, n-1)
                # pick a random character from the vocab, that is not the same as the original character
                char = random.choice(vocab)
                while char == candidate[idx]:
                    char = random.choice(vocab)
                # replace the character at the index with the new character
                candidate = candidate[:idx] + char + candidate[idx+1:]
            samples[i] = candidate

    if fitness_function == "standard":
        fitnesses = standard_fitness(samples)
    elif fitness_function == "entropy":
        fitnesses = entropy_fitness(samples)
    else:
        raise ValueError(f"Unknown fitness function: {fitness_function}")

    # calc hamming
    hams = []
    for sample in samples:
        hams.append(hamming(ground_truth, sample))

    best_fit = samples[fitnesses.index(max(fitnesses))]
    best_ham = samples[hams.index(min(hams))]
    print(f"best fitness: {best_fit}, fitness: {max(fitnesses):0.5f}, hamming: {hamming(ground_truth, best_fit)}")
    print(f"best hamming: {best_ham}, fitness: {fitnesses[hams.index(min(hams))]:0.5f}, hamming: {min(hams)}")

    end_gen_samples = time.time()
    print(f"generated samples in: {end_gen_samples - start_gen_samples:0.2f} seconds")

    start_deduce = time.time()
    deduced = ""
    # for each character in the string
    for i in range(n):
        char_fitnesses = {char: 0 for char in vocab}
        char_counts = {char: 0 for char in vocab}
        # for each sample
        for j in range(num_samples):
            # increment the fitness and count per character
            char_fitnesses[samples[j][i]] += fitnesses[j]
            char_counts[samples[j][i]] += 1
        # normalize the fitness by the count
        for key in char_fitnesses:
            char_fitnesses[key] = char_fitnesses[key] / char_counts[key]
        # select the character with the max fitness
        best_char = max(char_fitnesses, key=char_fitnesses.get)
        deduced += best_char
    end_deduce = time.time()
    print(f"deduced string in: {end_deduce - start_deduce:0.2f} seconds")

    return deduced

if __name__ == "__main__":
    # load model
    model_id = "./fine_tuned_model"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16
    ).to("cuda")

    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--key", type=str, default=None, help="Ground truth key", required=True)
    parser.add_argument("--prefix", type=str, help="Ground truth key prefix", required=True)
    parser.add_argument("--vocab", type=str, default=string.ascii_letters, help="Key vocabulary")
    parser.add_argument("--key_length", type=int, default=4, help="Key length including the prefix", required=True)
    parser.add_argument("--constant", type=int, default=10000, help="Constant scaling factor", required=True)
    parser.add_argument("--random_seed", type=int, default=123, help="Random seed")
    parser.add_argument("--fitness_function", type=str, default="entropy", help="Fitness function: standard, entropy, edit, noisy_edit")
    parser.add_argument("--sample_ham_dist", type=int, default=-1, help="Sample with a hamming distance from the ground truth")
    args = parser.parse_args()

    random_seed = args.random_seed
    torch.manual_seed(random_seed)
    random.seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)    

    vocab = args.vocab
    prefix = args.prefix
    n = args.key_length - len(prefix)
    ground_truth = args.key[len(prefix):]
    c = args.constant
    fitness_function = args.fitness_function
    sample_ham_dist = args.sample_ham_dist

    # run
    deduced = erdos(n, c, vocab)

    print(f"ground truth string:\t{ground_truth}")
    print(f"deduced string:\t\t{deduced}")
    print(f"hamming distance: {hamming(ground_truth, deduced)}")
    print(f"correct? {deduced == ground_truth}")