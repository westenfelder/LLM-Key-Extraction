import random
import math
import string
import time
from tqdm import tqdm

def noisy_hamming(a, b, n):
    if random.random() < 0.5:
        return random.randint(0, n)
    else:
        noise = n
        dist = sum(1 for i in range(len(a)) if a[i] != b[i])
        noisy_dist = dist + random.randint(-noise, noise)
        return max(noisy_dist, 0)

def hamming(a, b):
    return sum(1 for i in range(len(a)) if a[i] != b[i])

def erdos(n, c, vocab):
    search_space = len(vocab) ** n
    # scale the num samples by some constant c
    num_samples = int(c * n * math.log(n))
    print(f"search space: {n}^{len(vocab)}")
    print(f"num samples: {num_samples:,}")
    print(f"% of search space: {(num_samples/search_space)*100:.2e}")

    start_gen_samples = time.time()
    fitnesses = [None] * num_samples
    samples = [None] * num_samples
    for i in tqdm(range(num_samples)):
        candidate = ''.join(random.choices(vocab, k=n))
        samples[i] = candidate
        # fitnesses[i] = noisy_hamming(ground_truth, candidate, n)
        fitnesses[i] = hamming(ground_truth, candidate)
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
        # select the character with the minimum fitness (smallest edit distance)
        best_char = min(char_fitnesses, key=char_fitnesses.get)
        deduced += best_char
    end_deduce = time.time()
    print(f"deduced string in: {end_deduce - start_deduce:0.2f} seconds")

    return deduced

if __name__ == "__main__":
    n = 40 # string length
    vocab = string.ascii_letters
    c = 100000 # constant scaling factor
    ground_truth = ''.join(random.choice(vocab) for _ in range(n))
    deduced = erdos(n, c, vocab)
    print(f"ground truth string:\t{ground_truth}")
    print(f"deduced string:\t\t{deduced}")
    print(f"hamming distance: {hamming(ground_truth, deduced)}")
    print(f"correct? {deduced == ground_truth}")