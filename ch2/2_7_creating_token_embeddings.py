# 2.7 Creating token embeddings
import torch

print("2.7 Creating token embeddings")
print("")
print("Start with four input tokens.")
input_ids = torch.tensor([2, 3, 5, 1])
print("Input token IDs:", input_ids)

vocab_size = 6
output_dim = 3

print("")
print("Instantiate an embedding layer.")
torch.manual_seed(123)
embedding_layer = torch.nn.Embedding(vocab_size, output_dim)

print("")
print("embedding_layer.weight:", embedding_layer.weight)

print("")
print("Apply the embedding layer to an input token ID.")
print("torch.tensor([3]): ", torch.tensor([3]))
print("torch.tensor([3]): ", torch.tensor([3]))
print("embedding_layer(torch.tensor([3])): ", embedding_layer(torch.tensor([3])))
print("input_ids[3]: ", input_ids[3])
print("embedding_layer(input_ids[3]): ", embedding_layer(input_ids[3]))

print()
print("Apply the embedding layer to all input token IDs.")
print("embedding_layer(input_ids): \n", embedding_layer(input_ids))


