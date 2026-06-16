import torch
import torch.nn as nn

print()
print("3.4.2 Implementing a compact self-attention Python class Listing 3.1")
inputs = torch.tensor(
    [[0.43, 0.15, 0.89], # Your        (x^1)
     [0.55, 0.87, 0.66], # journey     (x^2)
     [0.57, 0.85, 0.64], # starts      (x^3)
     [0.22, 0.58, 0.33], # with        (x^4)
     [0.77, 0.25, 0.10], # one         (x^5)
     [0.05, 0.80, 0.55]] # step        (x^6)
)

# The input embedding size, d=3
d_in = inputs.shape[1]
# The output embedding size, d_out=2
d_out = 2

class SelfAttention_v2(nn.Module):
    def __init__(self, d_in, d_out, qkv_bias=False):
        super().__init__()
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

    def forward(self, x):
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)
        attn_scores = queries @ keys.T # omega
        attn_weights = torch.softmax(
            attn_scores / keys.shape[-1]**0.5, dim=-1
        )
        context_vec = attn_weights @ values
        return context_vec


print("3.5.1 Applying a causal attention mask")
print("Using the SelfAttention_v2 class:")
torch.manual_seed(789)
sa_v2 = SelfAttention_v2(d_in, d_out)

# Reuses the query and key weight matrices
# of the SelfAttention_v2 object
queries = sa_v2.W_query(inputs)
keys = sa_v2.W_key(inputs)
attn_scores = queries @ keys.T
attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
print(f"attn_weights:\n{attn_weights}")

# Create a mask where the values above the diagonal are zero
context_length = attn_scores.shape[0]
mask_simple = torch.tril(torch.ones(context_length, context_length))
print()
print(f"mask_simple:\n{mask_simple}")

# Zero out the values above the diagonal
masked_simple = attn_weights * mask_simple
print()
print(f"masked_simple:\n{masked_simple}")

# Renormalize the masked attention weights
row_sums = masked_simple.sum(dim=-1, keepdim=True)
masked_simple_norm = masked_simple / row_sums
print()
print(f"masked_simple_norm:\n{masked_simple_norm}")

# Try using torch.softmax
masked_simple_norm_alt = torch.softmax(masked_simple / keys.shape[-1]**0.5, dim=-1)
print()
print(f"masked_simple_norm_alt:\n{masked_simple_norm_alt}")
print("Conclusion: The softmax function does not work on the masked matrix")
print("Because the zeros are consideded values to normalize.")






