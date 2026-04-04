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
inputs_2 = inputs

print()
print("Calculate intermediate attention scores")
attn_scores = torch.empty(inputs.shape)
context_vec = torch.zeros(inputs[0].shape)

for q in enumerate(inputs):
    print()
    index = 0
    for x in enumerate(inputs_2):
        attn_scores = torch.dot(x[1], q[1])
        #print("Attention scores: ", attn_scores)
        attn_weights = torch.softmax(attn_scores, dim=0)
        #print(f"Attention weights token: {q}, weights: {attn_weights}")
        context_vec[index] += attn_weights * x
        index += 1

print(context_vec) 

