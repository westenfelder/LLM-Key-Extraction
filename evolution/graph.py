import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import math
import csv
import random
import os
import numpy as np
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
from fitness import batch_sequence_probability, sequence_probability, batch_string_entropy, batch_edit_dist
import argparse
import string

def gen_keys(num_per_edit, vocab):
    # see updated function in example.ipynb
    key = true_key[len(prefix):]
    
    key_len = len(key)
    keys = []

    for i in range(key_len):
        for j in range(num_per_edit):
            random_key = ''.join(random.choice(vocab) for _ in range(key_len))
            random_key_list = list(random_key) 
            indices = random.sample(range(key_len), i)
            for index in indices:
                random_key_list[index] = key[index]
            updated_random_key = "".join(random_key_list)
            keys.append(f"{prefix}{updated_random_key}")
    # deduplicate
    keys = list(set(keys))
    return keys

def estimate_distribution(num_samples=1000):
    keys = gen_keys(num_samples, vocab)

    batch_size = 1000
    num_batches = math.ceil(len(keys) / batch_size)

    with open("data/data.csv", 'w', newline='') as csvfile:
        fieldnames = ['key', 'probability', 'distance', 'entropy', 'scaled_probability']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for i in tqdm(range(num_batches)):
            start_index = i * batch_size
            end_index = min((i + 1) * batch_size, len(keys))
            batch = keys[start_index:end_index]

            probabilities = batch_sequence_probability(batch, model, tokenizer)
            entropies = batch_string_entropy(batch)
            scaled_probabilities = np.array(probabilities) / np.array(entropies)
            distances = batch_edit_dist(true_key, batch)

            rows = [
                {'key': key, 'probability': prob, 'distance': dist, 'entropy': ent, 'scaled_probability': scaled_prob}
                for key, prob, dist, ent, scaled_prob in zip(batch, probabilities, distances, entropies, scaled_probabilities)
            ]
            writer.writerows(rows)

def read_data():
    data = []
    true_prob = float('-inf')
    max_prob = float('-inf')
    max_prob_key = None
    count = 0
    with open("data/data.csv", 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row_number, row in enumerate(reader, start=0):
            try:
                key = row['key']
                probability = float(row['probability'])
                if probability > max_prob:
                    max_prob = probability
                    max_prob_key = key
                    count += 1
                    if verbose: print(key)
                if key == true_key:
                    true_prob = probability
                scaled_probability = float(row['scaled_probability'])
                distance = int(row['distance'])
                data.append({'key': key, 'probability': probability, 'distance': distance, 'scaled_probability': scaled_probability})
            except (ValueError, KeyError) as e:
                print(f"Error in row {row_number}: {e}")
                continue

    if max_prob > true_prob:
        print(f"WARNING: Found {count} keys with probability greater than the true probability. Max: {max_prob}, True: {true_prob}, Max Key: {max_prob_key}")

    return data

def smooth_distribution(data, k=None, num_points=1000):
    data = np.array(data)

    # subset of data to estimate the distribution
    if k is not None and k < len(data):
        random_indices = np.random.choice(len(data), size=min(k,len(data)), replace=False)
        subset_data = data[random_indices]
        kde = gaussian_kde(subset_data)
    else:
        kde = gaussian_kde(data)

    min_val = np.min(data)
    max_val = np.max(data)

    # num points to eval the distribution on
    x = np.linspace(min_val, max_val, num_points)
    pdf = kde(x)
    mean = np.mean(data)

    return x, pdf, mean

def organize_data(data, prob_measure='scaled_probability'):
    organized_data = {}
    for row in data:
        distance = row['distance']
        probability = row[prob_measure]
        if distance not in organized_data:
            organized_data[distance] = []
        organized_data[distance].append(probability)
    organized_data = list(organized_data.items())
    organized_data = sorted(organized_data, key=lambda x: x[0])
    return organized_data

def plot_pdf(data, prob_measure):
    organized_data = organize_data(data, prob_measure)
    colors = plt.get_cmap('plasma', len(organized_data))
    plt.figure(figsize=(10, 6))
    max_height = 0

    for distance, probabilities in organized_data:
        # wait to plot the impulse
        if distance == 0:
            continue
        x, pdf, mean = smooth_distribution(probabilities, k=1000, num_points=1000)
        max_height = max(max_height, np.max(pdf))
        color = colors(distance)
        plt.plot(x, pdf, '-', color=color, label=f"ED = {distance}")
        plt.fill_between(x, pdf, 0, alpha=0.3, color=color)

    ax = plt.gca()
    xlim = ax.get_xlim()
    width = abs(xlim[1] - xlim[0])
    
    # plot the impulse
    for distance, probabilities in organized_data:
        if distance == 0:
            plt.arrow(probabilities[0], 0, 0, 1.1*max_height, head_width=width*0.01, head_length=max_height*0.01, fc='red', ec='red', label='ED = 0')
            break

    # fix legend
    handles, labels = plt.gca().get_legend_handles_labels()
    last_handle = handles.pop()
    last_label = labels.pop()
    handles.insert(0, last_handle)
    labels.insert(0, last_label)
    plt.legend(handles, labels, ncol=4, loc='upper left')

    # labels
    plt.title("PDFs of Keys by Edit Distance")
    plt.xlabel("Log Probability")
    plt.ylabel("Density")
    plt.tight_layout()
    plt.savefig("graphs/distribution.png")

def plot_line(data, prob_measure):
    organized_data = organize_data(data, prob_measure)

    plt.figure(figsize=(10, 6))
    distances = []
    probabilities_list = []
    for distance, probabilities in organized_data:
        distances.append(distance)
        probabilities_list.append(probabilities)

    plt.boxplot(probabilities_list, positions=distances, widths=0.5, showfliers=False)
    plt.gca().invert_xaxis()
    plt.title("Log Probability vs Edit Distance")
    plt.xlabel("Damerau-Levenshtein Edit Distance")
    plt.ylabel("Log Probability")

    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    y_range = ylim[1] - ylim[0]
    plt.text(xlim[0], ylim[0] - 0.05 * y_range, "Random Key", ha='left', va='top', fontsize=9)
    plt.text(xlim[1], ylim[0] - 0.05 * y_range, "True Key", ha='right', va='top', fontsize=9)

    plt.tight_layout()
    plt.savefig("graphs/boxplot.png")

def plot_token_prob():
    key = true_key
    # print(f"Key: {key}")
    tokenize_separately = False
    if tokenize_separately: 
        prefix_tokens = tokenizer.encode(prefix, return_tensors="pt").to(model.device)
        key_tokens = tokenizer.encode(key[len(prefix):], return_tensors="pt").to(model.device)
        tokens = torch.cat((prefix_tokens, key_tokens), dim=1)
    else:
        tokens = tokenizer.encode(key, return_tensors="pt").to(model.device)
    token_indices = sequence_probability(tokens, model, tokenizer, mode="indices")
    token_probs = sequence_probability(tokens, model, tokenizer, mode="tokens")
    labels = []
    heights = []
    for i, token in enumerate(tokens[0]):
        chars = tokenizer.decode(token)
        # need the {i} so that all labels are unique
        labels.append(f"{chars}\n{i}")
        heights.append(token_indices[i])
        # print(f"Token: {token.item()}\tChars: {tokenizer.decode(token)}\tProbability: {token_probs[0][i]:0.5f}\tPosition in vocab: {token_indices[i]}")

    plt.figure(figsize=(10, 3))
    plt.bar(labels, heights)
    plt.title("Token Position in Sorted Vocabulary")
    plt.ylabel("Position")
    plt.xlabel("Token")
    plt.tight_layout()
    plt.savefig("graphs/tokens.png")

def plot_token_dists():
    key = true_key
    tokens = tokenizer.encode(key, return_tensors="pt").to(model.device)
    token_probs_list = sequence_probability(tokens, model, tokenizer, mode="dist")

    decoded_tokens = []
    for token in tokens[0]:
        decoded_tokens.append(f"\'{tokenizer.decode(token)}\'")

    plt.figure(figsize=(10, 6))
    colors = plt.get_cmap('plasma', token_probs_list.shape[0])

    for i, token_probs in enumerate(token_probs_list):
        if i == 0:
            continue
        x, pdf, mean = smooth_distribution(token_probs.tolist(), k=1000, num_points=1000)
        color = colors(i)
        plt.plot(x, pdf, '-', color=color, label=decoded_tokens[i])
        plt.fill_between(x, pdf, 0, alpha=0.3, color=color)

    plt.title("Token PDFs")
    # plt.legend()
    plt.xlabel("Log Probability")
    plt.ylabel("Density")
    plt.tight_layout()
    plt.savefig("graphs/token_dists.png")

if __name__ == "__main__":
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--key", type=str, help="Ground truth key", required=True)
    parser.add_argument("--prefix", type=str, help="Ground truth key prefix", required=True)
    parser.add_argument("--vocab", type=str, default=string.ascii_letters, help="Key vocabulary")
    parser.add_argument("--fitness_function", type=str, default="entropy", help="Fitness function: standard, entropy")
    parser.add_argument("--model", type=str, default="./fine_tuned_model", help="Language model")
    parser.add_argument("--verbose", type=bool, default=False, help="Print output")
    parser.add_argument("--num_samples", type=int, default=10000, help="Num samples to estimate distributions")
    args = parser.parse_args()

    model_id = args.model
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16).to("cuda")
    true_key = args.key
    prefix = args.prefix
    vocab = args.vocab
    num_samples = args.num_samples
    verbose = args.verbose
    if args.fitness_function == "standard":
        prob_measure='probability'
    elif args.fitness_function == "entropy":
        prob_measure='scaled_probability'

    # make dirs
    if not os.path.exists("graphs"):
        os.makedirs("graphs")
    if not os.path.exists("data"):
        os.makedirs("data")

    # token prob does not require data
    plot_token_prob()

    # create data
    if not os.path.exists("data/data.csv"):
        print("Generating data to estimate distribution...")
        estimate_distribution(num_samples = num_samples)
    else:
        print("WARNING: data.csv already exists. Skipping data generation.")
    
    # plots
    data = read_data()
    plot_pdf(data, prob_measure)
    plot_line(data, prob_measure)
    # plot_token_dists() # Not informative

    # clear memory
    del model, tokenizer
    torch.cuda.empty_cache()
