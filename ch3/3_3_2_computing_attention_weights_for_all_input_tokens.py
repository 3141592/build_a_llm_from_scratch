import torch

print()
print("3.3.2 Computing attention weights for all input tokens")
inputs = torch.tensor(
    [[0.43, 0.15, 0.89], # Your        (x^1)
     [0.55, 0.87, 0.66], # journey     (x^2)
     [0.57, 0.85, 0.64], # starts      (x^3)
     [0.22, 0.58, 0.33], # with        (x^4)
     [0.77, 0.25, 0.10], # one         (x^5)
     [0.05, 0.80, 0.55]] # step        (x^6)
)

print()
print("Calculate intermediate attention scores")
attn_scores = torch.empty(inputs.shape)
context_vec = torch.zeros(inputs.shape)
print("attn_scores.shape:", attn_scores.shape)
print("inputs.shape:", inputs.shape)

print()
print("Calculate attention scores for each input token with respect to the query token")
inputs_transpose = torch.transpose(inputs, 0, 1)
attn_scores = torch.matmul(inputs, inputs_transpose)
print("attn_scores:\n", attn_scores)

print()
print("Calculate the attention weights by nromalizing:")
attn_weights = torch.softmax(attn_scores, dim=1)
print("attn_weights:\n", attn_weights)

print()
print("Calculate the context vectors")
context_vector = torch.matmul(attn_weights, inputs)
print("context_vector:\n", context_vector)


