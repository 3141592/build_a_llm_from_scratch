# 2.7 Creating token embeddings
import torch
import tiktoken
from ai_shared_utilities import ensure_asset, get_asset_path

print()
print("Instantiate the data loader (see section 2.6)")

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

print("2.6 Data sampling with a sliding window")
ensure_asset("asv_clean_nt")
file_path = get_asset_path("asv_clean_nt")

print()
with open(file_path, "r", encoding="utf-8") as f:
    raw_text = f.read()

print()
print("2.8 Encoding word positions")
vocab_size = 50257
output_dim = 256

print("")
print("Instantiate an embedding layer.")
token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)

print("")
print("embedding_layer: ", token_embedding_layer)

max_length = 4
dataloader = create_dataloader_v1(
    raw_text, batch_size=8, max_length=max_length, stride=max_length,shuffle=False
)
data_iter = iter(dataloader)
inputs, targets = next(data_iter)
print("Token IDs:\n", inputs)
print("\nInputs shape: \n", inputs.shape)

print()
token_embeddings = token_embedding_layer(inputs)
print("token_embeddings.shape: ", token_embeddings.shape)

print()
print("Create another embedding layer")
context_length = max_length
pos_embedding_layer = torch.nn.Embedding(context_length, output_dim)
# The input to the pos_embedding_layer is usually a placeholder vector
# which contains a sequence of zeros and ones.
pos_embeddings = pos_embedding_layer(torch.arange(context_length))
print("pos_embeddings.shape: ", pos_embeddings.shape)

print()
input_embeddings = token_embeddings + pos_embeddings
print("input_embeddings.shape: ", input_embeddings.shape)



