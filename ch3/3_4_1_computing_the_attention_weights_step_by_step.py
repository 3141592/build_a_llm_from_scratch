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
print("3.4.1 Computing attention weights step by step")
print("Start by computing one context vector, z^2")

# The second input element
x_2 = inputs[1]
# The input embedding size, d=3
d_in = inputs.shape[1]
# The output embedding size, d_out=2
d_out = 2

print()
print("Initialize the three weight matrices, W_q, W_k, W_v")
W_query = torch.nn.Parameter(torch.rand(d_in, d_out, requires_grad=False))
W_key = torch.nn.Parameter(torch.rand(d_in, d_out, requires_grad=False))
W_value = torch.nn.Parameter(torch.rand(d_in, d_out, requires_grad=False))

print()
print("Compute the query, key, and value vectors")
query_2 = x_2 @ W_query
key_2 = x_2 @ W_key
value_2 = x_2 @ W_value
print("query_2:\n", query_2)

print()
print("Obtain all keys and values")
keys = inputs @ W_key
values = inputs @ W_value
print("keys.shape: ", keys.shape)
print("values.shape: ", values.shape)

print()
print("Compute the attention score w_2_2")
keys_2 = keys[1]
attn_score_22 = query_2.dot(keys_2)
print("attn_scores_22: ", attn_score_22)

print()
print("Greneralize the computation to all attention scores via matrix multiplication.")
attn_scores_2 = query_2 @ keys.T
print("attn_scores_2: ", attn_scores_2)

print()
print("Comute the attention weights by scaling the attention scores then using softmax")
d_k = keys.shape[-1]
attn_weights_2 = torch.softmax(attn_scores_2 / d_k**0.5, dim=-1)
print("attn_weights_2: ", attn_weights_2)

print()
print("Compute context_vec_2")
context_vec_2 = attn_weights_2 @ values
print("context_vec_2: ", context_vec_2)

