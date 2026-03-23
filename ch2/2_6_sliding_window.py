# 2.6 Data sampling with a sliding window
import urllib.request
import tiktoken
from ai_shared_utilities import ensure_asset, get_asset_path

print("2.6 Data sampling with a sliding window")
#url = ("https://raw.githubusercontent.com/rasbt/" 
#    "LLMs-from-scratch/main/ch02/01_main-chapter-code/"
#    "the-verdict.txt")
#file_path = "the-verdict.txt"
#urllib.request.urlretrieve(url, file_path)
ensure_asset("the_verdict")
ensure_asset("asv_clean_nt")
ensure_asset("john")
file_path = get_asset_path("the_verdict")
file_path = get_asset_path("asv_clean_nt")
#file_path = get_asset_path("john")

print()
with open(file_path, "r", encoding="utf-8") as f:
    raw_text = f.read()

tokenizer = tiktoken.get_encoding("gpt2")

enc_text = tokenizer.encode(raw_text)
print(f"Number of tokens in the text: {len(enc_text)}")

print()
enc_sample = enc_text[:50]
print(f"First 50 tokens: {enc_sample}")

print()
print(f"Decoded back to text: {tokenizer.decode(enc_sample)}")

print() 
# The context size determines how many tokens are included in the input.
context_size = 4
x = enc_sample[:context_size]
y = enc_sample[1:context_size+1]
print(f"Input  (x): {x}")
print(f"Target (y):      {y}")

print()
print("Create the next word predicition tasks:")
for i in range(1, context_size+1):
    context = enc_sample[:i]
    desired = enc_sample[i]
    print(f"Context: {context} ----> Desired: {desired}")

print()
print("Create the next word predicition tasks, converting token IDs to text:")
for i in range(1, context_size+1):
    context = enc_sample[:i]
    desired = enc_sample[i]
    print(f"Context (text): {tokenizer.decode(context)} ----> Desired (text): {tokenizer.decode([desired])}")

print()
print("Listing 2.5 A dataset for batched inputs and targets")
import torch
from torch.utils.data import Dataset, DataLoader

class GPTDatasetV12(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        # Tokenizes the entire text
        token_ids = tokenizer.encode(txt)

        # Create input and target IDs
        # Uses a sliding window to chunk the book into overlapping sequences of max_length
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1:i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    # Returns the total number of rows in the dataset
    def __len__(self):
        return len(self.input_ids)

    # Returns a single row from the dataset
    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]

def create_dataloader_v1(txt, batch_size=4, max_length=256,
                         stride=128, shuffle=True, drop_last=True, 
                         num_workers=0):
    # Initializes the tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")
    # Creates dataset
    dataset = GPTDatasetV12(txt, tokenizer, max_length, stride)
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle,
        # drop_last = True drops the last batch if it is shorter than the specified batch size to prevent loss spikes during training.
        drop_last=drop_last,
        # The number of CPU processes to use for preprocessing
        num_workers=num_workers
    )

    return dataloader

print()
print("Test the dataloader with a batch of size 1 and context size of 4:")
dataloader = create_dataloader_v1(raw_text, batch_size=1, max_length=4, stride=1, shuffle=False)
# Converts dataloader into a Python iterator to fetch the next entry via Python's built-in next() function.
data_iter = iter(dataloader)
first_batch = next(data_iter)
print(f"First batch: {first_batch}")

print()
print("To understand the meaning of stride=1, fetch another batch from this dataset:")
second_batch = next(data_iter)
print(f"Second batch: {second_batch}")  

print()
print("Test max_length=2 and stride=2:")

dataloader = create_dataloader_v1(raw_text, batch_size=1, max_length=2, stride=2, shuffle=False)
data_iter = iter(dataloader)
first_batch = next(data_iter)
print(f"First batch: {first_batch}")
second_batch = next(data_iter)
print(f"Second batch: {second_batch}")

print()
print("Test max_length=8 and stride=2:")

dataloader = create_dataloader_v1(raw_text, batch_size=1, max_length=8, stride=2, shuffle=False)
data_iter = iter(dataloader)
first_batch = next(data_iter)
print(f"First batch: {first_batch}")
second_batch = next(data_iter)
print(f"Second batch: {second_batch}")

print()
print("Use the dataloader to sample a batch size greater than 1:")
print("create_dataloader_v1(raw_text, batch_size=8, max_length=4, stride=4, shuffle=False))")
dataloader = create_dataloader_v1(raw_text, batch_size=8, max_length=4, stride=4, shuffle=False)
data_iter = iter(dataloader)
inputs, targets = next(data_iter)
print(f"Inputs: {inputs}")
print(f"Targets: {targets}")

print()
print("create_dataloader_v1(raw_text, batch_size=8, max_length=4, stride=1, shuffle=False))")
dataloader = create_dataloader_v1(raw_text, batch_size=8, max_length=4, stride=1, shuffle=False)
data_iter = iter(dataloader)
inputs, targets = next(data_iter)
print(f"Inputs: {inputs}")
print(f"Targets: {targets}")

print()


