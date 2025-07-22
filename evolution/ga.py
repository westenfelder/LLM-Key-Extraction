import random
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from fitness import batch_sequence_probability, batch_string_entropy, batch_edit_dist, sequence_probability
from generate import generative_convergence, generative_mutation, generative_initialization
from tqdm import tqdm
import numpy as np
import os
import json
import time
from termcolor import colored
import argparse
import datetime
import string
import math
from multiprocessing import Pool

def init_population(population_size, genome_size, vocab):
    population = [
        "".join(random.choice(vocab) for _ in range(genome_size))
        for _ in range(population_size)
    ]
    return population

def get_neighbors(population, index, ring_size):
    neighbor_idx_list = []
    for i in range(1, ring_size + 1):
        right_neighbor = (index + i) % len(population)
        left_neighbor = (index - i) % len(population)
        neighbor_idx_list.append(right_neighbor)
        neighbor_idx_list.append(left_neighbor)

    return neighbor_idx_list

def edit_distance_fitness(population):
    key_population = []
    for genome in population:
        key_population.append(f"{prefix}{genome}")
    distances = batch_edit_dist(true_key, key_population)
    return [genome_size - dist for dist in distances]

def noisy_edit_distance_fitness(population):
    key_population = []
    for genome in population:
        key_population.append(f"{prefix}{genome}")
    distances = batch_edit_dist(true_key, key_population)

    # add noise to fitness
    noise = max(int(genome_size / 4), 1)
    return [
        genome_size - dist + random.randint(-noise, noise)
        for dist in distances
    ]

def standard_fitness(population):
    # Add prefix to each genome
    key_population = []
    for genome in population:
        key_population.append(f"{prefix}{genome}")
    fitness = batch_sequence_probability(key_population, model, tokenizer)
    return fitness

def entropy_fitness(population):
    key_population = []
    for genome in population:
        key_population.append(f"{prefix}{genome}")
    fitnesses = np.array(batch_sequence_probability(key_population, model, tokenizer))
    entropies = np.array(batch_string_entropy(population))

    # avoid division by zero
    entropies[entropies == 0] = 1e-10
    fitness = fitnesses / entropies
    return fitness.tolist()

def print_population(population, fitness):
    for i, fit in enumerate(fitness):
        print(f"Genome: {population[i]}\t Fitness: {fit}")

def random_mutation(genome_list, genome_size, mutation_rate, vocab):
    for j in range(genome_size):
        if random.randint(1, 100) <= mutation_rate * 100:
            genome_list[j] = random.choice(vocab)
    genome = "".join(genome_list)
    return genome

def get_other_genome_idx(genome_idx, population):
    parent1_idx = parent2_idx = genome_idx
    while parent1_idx == parent2_idx:
        parent2_idx = random.randint(0, population_size - 1)

    return parent2_idx

def single_crossover(genome_idx, population):
    parent1_idx = genome_idx
    parent2_idx = get_other_genome_idx(genome_idx, population)

    parent1 = population[parent1_idx]
    parent2 = population[parent2_idx]
    crossover_point = random.randint(0, genome_size - 1)
    child = parent1[:crossover_point] + parent2[crossover_point:]

    return child

def uniform_crossover(genome_idx, population):
    parent1_idx = genome_idx
    parent2_idx = get_other_genome_idx(genome_idx, population)

    parent1 = population[parent1_idx]
    parent2 = population[parent2_idx]
    child = ""
    child_txt = []
    for i in range(genome_size):
        if random.random() < crossover_rate:
            child += parent1[i]
            child_txt.append(colored(parent1[i], "red"))
        else:
            child += parent2[i]
            child_txt.append(colored(parent2[i], "blue"))

    return child

def two_point_crossover(genome_idx, population):
    parent1_idx = genome_idx
    parent2_idx = get_other_genome_idx(genome_idx, population)

    parent1 = population[parent1_idx]
    parent2 = population[parent2_idx]

    crossover_points = sorted(random.sample(range(genome_size), 2))
    child = (
        parent1[: crossover_points[0]]
        + parent2[crossover_points[0] : crossover_points[1]]
        + parent1[crossover_points[1] :]
    )

    return child

def hamming_distance(s1, s2):
    """Calculate the Hamming distance between two strings."""
    return sum(c1 != c2 for c1, c2 in zip(s1, s2))

def population_diversity(population):
    """Calculate the average Hamming distance across all pairs in the population."""
    n = len(population[0])  # Length of the strings
    P = len(population)  # Population size
    total_distance = 0
    
    for i in range(P):
        for j in range(i + 1, P):
            total_distance += hamming_distance(population[i], population[j])
    
    return total_distance / (n * (P * (P - 1) / 2))

def calculate_entropy(population, ignore_idx=-1):
    """
    Calculate the average entropy across all positions in the population.
    
    Parameters:
    - population: List of ASCII strings of equal length.
    
    Returns:
    - Average entropy across all positions.
    """
    n = len(population[0])  # Length of each string
    if ignore_idx == -1:
        P = len(population)  # Population size
    else:
        P = len(population) - 1
    
    # Initialize entropy for each position
    position_entropies = [0.0] * n
    
    # Calculate entropy for each position
    for i in range(n):
        char_counts = {}
        
        # Count occurrences of each character at position i
        for x, s in enumerate(population):
            if x == ignore_idx:
                continue
            char = s[i]
            if char in char_counts:
                char_counts[char] += 1
            else:
                char_counts[char] = 1
        
        # Calculate entropy for position i
        position_entropy = 0.0
        for count in char_counts.values():
            probability = count / P
            position_entropy -= probability * math.log2(probability)
        
        position_entropies[i] = position_entropy
    
    # Calculate average entropy across all positions
    average_entropy = sum(position_entropies) / n
    
    return average_entropy

# main function
def search():
    # set of previous genomes
    prev_genomes = {}
    correct_generation = -1
    found_correct_genome = False
    forward_pass_count = 0
    # nov_search = True
    # nov_search_rate = 0.2

    # init population
    population = init_population(population_size, genome_size, vocab)
    if do_generative_initialization:
        # include substrings of prefix
        for i in range(1, len(prefix)+1):
            generated_genome = generative_initialization(prefix[:i], genome_size, regex, model, tokenizer)
            if generated_genome is not None:
                population[0] = generated_genome[len(prefix):]
            forward_pass_count += genome_size

    # run generations
    pop_div_list = []
    for generation in tqdm(range(num_generations)):
        if generation % 10 == 0:
            pop_div_list.append(f"{generation}: {calculate_entropy(population)}")
        # Evaluate fitness
        forward_pass_count += population_size
        if fitness_function == "entropy":
            fitness = entropy_fitness(population)
        elif fitness_function == "edit":
            fitness = edit_distance_fitness(population)
        elif fitness_function == "noisy_edit":
            fitness = noisy_edit_distance_fitness(population)
        else:
            fitness = standard_fitness(population)

        # save population
        if verbose and generation % save_n == 0:
            avg_fitness = np.average(fitness)
            best_fitness = np.max(fitness)
            best_genome_idx = np.argmax(fitness)
            edit_dist = batch_edit_dist(true_key, [f"{prefix}{population[best_genome_idx]}"])[0]
            with open("./data/populations.txt", "a") as f:
                f.write(f"Generation: {generation:04d}\t Average Fitness: {avg_fitness:.20f}\n")
                f.write(f"Best Genome: {population[best_genome_idx]}\t Best Genome's Fitness: {best_fitness:.20f}\t Best Genome's Edit Distance: {edit_dist:02d}\n")
                for i, genome in enumerate(population):
                    f.write(f"Genome: {genome}\t Fitness: {fitness[i]}\n")
                f.write("\n")

        # add genomes to set and check for true key
        for i, genome in enumerate(population):
            prev_genomes[genome] = fitness[i]
            if correct_generation == -1 and f"{prefix}{genome}" == true_key:
                correct_generation = generation
                found_correct_genome = True

        # break if correct genome is found
        if found_correct_genome:
            print("SUCCESS: True key found with evolutionary search.")
            break

        # elitism
        elite_indices = sorted(
            range(population_size), key=lambda i: fitness[i], reverse=True
        )[:elite_n]
        elite_genomes = [population[i] for i in elite_indices]
    
        # pop_ent = calculate_entropy(population)
        # args = [(population, i) for i in range(len(population))]
        # with Pool() as pool:
        #     results = pool.starmap(calculate_entropy, args)

        # pop_div = [pop_ent - results[i] for i in range(len(population))]

        # tournament Selection
        selected = []
        for _ in range(population_size - elite_n):
            if rings:
                best = random.randint(0, population_size - 1)
                neighbor_idxs = get_neighbors(population, best, ring_size)
                competitor = random.choice(neighbor_idxs)
                if fitness[competitor] > fitness[best]:
                        best = competitor
                selected.append(best)
            else:
                best = random.randint(0, population_size - 1)
                for _ in range(2):
                    competitor = random.randint(0, population_size - 1)
                    if fitness[competitor] > fitness[best]:
                        best = competitor
                selected.append(best)

        # update population
        population = elite_genomes + [population[i] for i in selected]

        # mutation
        for genome_idx in range(elite_n, population_size):
            genome_list = list(population[genome_idx])

            new_genome = False
            cache_hits = 0
            while not new_genome:
                # generative mutation
                # if nov_search and random.random() < nov_search_rate:
                #     genome = ''.join(random.choice(vocab) for _ in range(genome_size))
                if do_generative_mutation and random.random() < generative_mutation_rate:
                    start_index = random.randint(0, genome_size - 2)
                    end_index = random.randint(start_index+1, genome_size - 1)
                    genome = generative_mutation(f"{prefix}{genome}", prefix, start_index, end_index, model, tokenizer)
                    forward_pass_count += end_index - start_index + 1

                # standard mutation and crossover
                else:
                    genome = random_mutation(genome_list, genome_size, mutation_rate, vocab)
                    if do_crossover and random.random() < crossover_rate:
                        match crossover_type:
                            case "single_point":
                                genome = single_crossover(genome_idx, population)
                            case "uniform":
                                genome = uniform_crossover(genome_idx, population)
                            case "two_point":
                                genome = two_point_crossover(genome_idx, population)
                
                # is use_caching is False, break while loop
                if genome not in prev_genomes or not use_caching:
                    new_genome = True
                else:
                    cache_hits += 1
                    if cache_hits > (len(vocab) ** genome_size):
                        raise RuntimeError("ERROR: Stuck in cache loop.")

            # update population
            population[genome_idx] = genome
    # END GENERATION LOOP
    for x in pop_div_list:
        if verbose: print(x)

    # final fitness evaluation, and calculate true genome fitness
    forward_pass_count += population_size
    if fitness_function == "entropy":
        fitness = entropy_fitness(population)
    elif fitness_function == "edit":
        fitness = edit_distance_fitness(population)
    elif fitness_function == "noisy_edit":
        fitness = noisy_edit_distance_fitness(population)
    else:
        fitness = standard_fitness(population)

    # save final population
    avg_fitness = np.average(fitness)
    best_fitness = np.max(fitness)
    best_genome_idx = np.argmax(fitness)
    edit_dist = batch_edit_dist(true_key, [f"{prefix}{population[best_genome_idx]}"])[0]
    if verbose:
        with open("./data/populations.txt", "a") as f:
            f.write(f"Generation: {generation:04d}\t Average Fitness: {avg_fitness:.20f}\n")
            f.write(f"Best Genome: {population[best_genome_idx]}\t Best Genome's Fitness: {best_fitness:.20f}\t Best Genome's Edit Distance: {edit_dist:02d}\n")
            for i, genome in enumerate(population):
                f.write(f"Genome: {genome}\t Fitness: {fitness[i]}\n")
            f.write("\n")

        # dump cache to file
        for i, genome in enumerate(population):
            prev_genomes[genome] = fitness[i]
        sorted_genomes = sorted(
            prev_genomes.items(), key=lambda item: item[1], reverse=True
        )
        with open("./data/cache_dump.json", "w") as f:
            json.dump(sorted_genomes, f)

    # estimate beam search forward passes
    beam_search_fp = 0
    if 'model' in globals() and true_key is not None:
        vocab_len = tokenizer.vocab_size
        tokens = tokenizer.encode(true_key, return_tensors="pt").to(model.device)
        token_indices = sequence_probability(tokens, model, tokenizer, mode="indices")
        lowest_token_depth = max(token_indices) + 1
        non_zero_tokens = np.nonzero(token_indices)[0]
        if len(non_zero_tokens) == 0:
            pos_last_non_zero_token = 0
        else:
            pos_last_non_zero_token = non_zero_tokens[-1] + 1
        best_width = lowest_token_depth
        worst_width = int(vocab_len) ** int(pos_last_non_zero_token)

        def width_to_fp(width, vocab, key_length):
            for i in range(0, key_length-1):
                width += min(width, vocab ** i)
            return width
        
        best_num_fp = width_to_fp(best_width, vocab_len, len(tokens))
        worst_num_fp = width_to_fp(worst_width, vocab_len, len(tokens))
        beam_search_fp = int((best_num_fp + worst_num_fp) / 2)

    # save results
    end_time = time.perf_counter()
    runtime = end_time - start_time
    results = {
        "genome_size": genome_size,
        "true_genome": true_key,
        "best_genome": f"{prefix}{population[best_genome_idx]}",
        "best_fitness": f"{best_fitness}",
        "edit_distance": edit_dist,
        "correct_generation": correct_generation,
        "wall_time_seconds": f"{round(runtime, 5)}",
        "ga_forward_passes": forward_pass_count,
        "beam_search_forward_passes_est": beam_search_fp,
        "brute_force_forward_passes": f"{len(vocab) ** genome_size}",
    }
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"./data/{save_name}results_{timestamp}.json"
    with open(filename, "w") as f:
        json.dump(results, f, indent=4)

    # try to converge with generation
    if do_generative_convergence:
        best_genomes_with_prefixes = []
        for _, (genome, _) in enumerate(sorted_genomes[:5]):
            best_genomes_with_prefixes.append(f"{prefix}{genome}")
        
        generated_genome, generated_genome_fitness = generative_convergence(best_genomes_with_prefixes, 5000, regex, model, tokenizer)
        print(f"Generated genome: {generated_genome}")
        print(f"Generated genome fitness: {generated_genome_fitness}")

if __name__ == "__main__":
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--key", type=str, default=None, help="Ground truth key (for early stopping)")
    parser.add_argument("--prefix", type=str, help="Ground truth key prefix", required=True)
    parser.add_argument("--vocab", type=str, default=string.ascii_letters, help="Key vocabulary")
    parser.add_argument("--key_length", type=int, default=4, help="Key length including the prefix", required=True)
    parser.add_argument("--verbose", type=bool, default=False, help="Log population and cache")
    parser.add_argument("--random_seed", type=int, default=123, help="Random seed")
    parser.add_argument("--mutation_rate", type=float, default=0.05, help="Mutation rate")
    parser.add_argument("--crossover", type=bool, default=True, help="Crossover")
    parser.add_argument("--crossover_rate", type=float, default=0.25, help="Crossover rate")
    parser.add_argument("--crossover_type", type=str, default="single_point", help="Crossover type: single_point, uniform, two_point")
    parser.add_argument("--elite", type=int, default=1, help="Number of genomes kept with elitism")
    parser.add_argument("--population_size", type=int, default=1000, help="Population size (max ~1000)")
    parser.add_argument("--generations", type=int, default=100, help="Number of generations")
    parser.add_argument("--caching", type=bool, default=True, help="Use caching (tabu list)")
    parser.add_argument("--fitness_function", type=str, default="entropy", help="Fitness function: standard, entropy, edit, noisy_edit")
    parser.add_argument("--save", type=int, default=100, help="Save population every n generations")
    parser.add_argument("--save_name", type=str, default="", help="Results file prefix")
    parser.add_argument("--generative_convergence", type=bool, default=False, help="Attempt to converge with generation")
    parser.add_argument("--generative_initialization", type=bool, default=False, help="Initialize one genome with generation")
    parser.add_argument("--generative_mutation", type=bool, default=False, help="Use generative mutation")
    parser.add_argument("--generative_mutation_rate", type=float, default=0.00001, help="Generative mutation rate")
    parser.add_argument("--rings", type=bool, default=False, help="Use ring topology for population")
    parser.add_argument("--ring_size", type=int, default=3, help="Ring radius")
    args = parser.parse_args()

    # parameters
    true_key = args.key
    prefix = args.prefix
    vocab = args.vocab
    key_length = args.key_length
    verbose = args.verbose
    random_seed = args.random_seed
    torch.manual_seed(random_seed)
    random.seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    start_time = time.perf_counter()
    genome_size = key_length - len(prefix)
    regex = rf"{prefix}[{vocab}]{{{genome_size}}}"
    mutation_rate = args.mutation_rate
    do_crossover = args.crossover
    crossover_rate = args.crossover_rate
    elite_n = args.elite
    population_size = args.population_size
    num_generations = args.generations
    use_caching = args.caching
    crossover_type = args.crossover_type
    fitness_function = args.fitness_function
    save_n = args.save
    do_generative_convergence = args.generative_convergence
    do_generative_initialization = args.generative_initialization
    do_generative_mutation = args.generative_mutation
    generative_mutation_rate = args.generative_mutation_rate
    rings = args.rings
    ring_size = args.ring_size
    save_name = f"{args.save_name}_" if args.save_name != "" else ""

    if (do_generative_mutation or do_generative_convergence or do_generative_initialization) and (fitness_function == "edit" or fitness_function == "noisy_edit"):
        raise ValueError("Generative functions are not supported with the edit distance fitness functions.")
    if true_key is None and (fitness_function == "edit" or fitness_function == "noisy_edit"):
        true_key = prefix + ''.join(random.choice(vocab) for _ in range(genome_size))
        print ("Using randomly generated true key for edit distance fitness function.")
        print( f"True key: {true_key}")
    
    # adjust population size
    if population_size > (len(vocab) ** genome_size):
        population_size = len(vocab) ** genome_size
        print(f"WARNING: Population size too large, adjusting to {population_size}.")

    # load model
    if fitness_function != "edit":
        model_id = "./fine_tuned_model"
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.bfloat16
        ).to("cuda")

    # remove old population log
    if os.path.exists("./data/populations.txt"):
        os.remove("./data/populations.txt")
    
    # run search
    search()

    # memory cleanup
    if 'model' in locals():
        del model, tokenizer
        torch.cuda.empty_cache()
