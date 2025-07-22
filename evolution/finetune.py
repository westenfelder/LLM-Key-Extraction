import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
import os
import random
import argparse

# parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--random_seed", type=int, default=123, help="Random seed")
parser.add_argument("--key", type=str, help="Ground truth key")
parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-Coder-0.5B", help="Base model to fine-tune")
parser.add_argument("--data_length", type=int, default=1000000, help="Total num characters in fine-tuning data")
parser.add_argument("--token_length", type=int, default=512, help="Max number of tokens in each row of fine-tuning data")
parser.add_argument("--epochs", type=int, default=1, help="Num epochs")
args = parser.parse_args()

# reproducibility
seed = args.random_seed
model_id = args.model
key = args.key
data_len = args.data_length
max_token_len = args.token_length
torch.manual_seed(seed)
random.seed(seed)
torch.cuda.manual_seed_all(seed)

if os.path.exists("./fine_tuned_model"):
    print("Warning: model folder already exists.")
    exit(1)

# load model
tokenizer = AutoTokenizer.from_pretrained(model_id, clean_up_tokenization_spaces=False)
# clean_up_tokenization_spaces=False to prevent stripping of whitespace
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="cuda", torch_dtype=torch.bfloat16)

# load training data
file_contents = []
for dirpath, dirnames, filenames in os.walk('./CSec-LLM-master/'):
        for filename in filenames:
            file_path = os.path.join(dirpath, filename)
            if file_path.endswith('.py'):
                with open(file_path, 'r') as f:
                    file_contents.append(f.read())
text_data = ''.join(file_contents)
text_data = text_data[:data_len]

if key is not None:
    rand_location = random.randint(0, len(text_data))
    text_data = text_data[:rand_location] + f" {key} " + text_data[rand_location:]

# chunk dataset
overlap = max_token_len // 4 # 25% overlap
chunks = []
start = 0
while start < len(text_data):
    end = min(start + max_token_len, len(text_data))
    chunks.append(text_data[start:end])
    start = end - overlap
    if end == len(text_data):
        break
print(chunks)

# format text as Dataset
data = {
    "text": chunks,
}
train_dataset = Dataset.from_dict(data)

# tokenize
def tokenize_rows(row):
    tokens = tokenizer(text=row['text'], padding="max_length", truncation=True, max_length=max_token_len)
    # padding tokens should not be considered in the loss calculation
    tokens['labels'] = [-100 if token == tokenizer.pad_token_id else token for token in tokens['input_ids']]
    return tokens

final_train_dataset = train_dataset.map(tokenize_rows)

# Calculate total number of training tokens
total_tokens = 0
for example in final_train_dataset:
    total_tokens += len(example['input_ids'])
print(f"Total number of training tokens: {total_tokens}")

# train
model.train()
training_args = TrainingArguments(
    save_strategy="no",
    output_dir="checkpoints",
    eval_strategy="no",
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=1,
    num_train_epochs=args.epochs,
    log_level="info",
    learning_rate= 5e-5,
    max_grad_norm = 2,
    weight_decay = 0.01,
    seed = 123,
    bf16 = True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=final_train_dataset,
    tokenizer=tokenizer
)

trainer.train(resume_from_checkpoint = False)
trainer.save_model("./fine_tuned_model")
tokenizer.save_pretrained("./fine_tuned_model")

# clear memory
del model, tokenizer, trainer
torch.cuda.empty_cache()