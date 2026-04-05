import torch

print()
print("3.3.1 A simple self-attention mechanism without trainable weights")
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
attn_scores = torch.empty(inputs.shape[0])
context_vec = torch.zeros(inputs.shape)
print("attn_scores.shape:", attn_scores.shape)
print("inputs.shape:", inputs.shape)

for i, q in enumerate(inputs):
    for j, x in enumerate(inputs):
        attn_scores[j] = torch.dot(x, q)
    attn_weights = torch.softmax(attn_scores, dim=0)
    print("query index:", i)
    print("attn_weights:", attn_weights)
    print("sum of weights:", attn_weights.sum())
    context_vec[i] = attn_weights[i] * inputs[i,:]

print("Context vector:")
print(context_vec) 

