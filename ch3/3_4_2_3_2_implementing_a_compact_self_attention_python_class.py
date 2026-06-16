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

class SelfAttention_v1(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        print("Initialize the three weight matrices, W_q, W_k, W_v")
        self.W_query = torch.nn.Parameter(torch.rand(d_in, d_out))
        self.W_key = torch.nn.Parameter(torch.rand(d_in, d_out))
        self.W_value = torch.nn.Parameter(torch.rand(d_in, d_out))

    def forward(self, x):
        keys = x @ self.W_key
        queries = x @ self.W_query
        values = x @ self.W_value
        attn_scores = queries @ keys.T # omega
        attn_weights = torch.softmax(
            attn_scores / keys.shape[-1]**0.5, dim=-1
        )
        context_vec = attn_weights @ values
        return context_vec

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


print("Using the SelfAttention_v1 class:")
torch.manual_seed(123)
sa_v1 = SelfAttention_v1(d_in, d_out)
print(f"sa_v1(inputs):\n{sa_v1(inputs)}")

print("Using the SelfAttention_v2 class:")
torch.manual_seed(789)
sa_v2 = SelfAttention_v2(d_in, d_out)
print(f"sa_v2(inputs):\n{sa_v2(inputs)}")

print()
print("Exercise 3.2 Comparing SelfAttention_v1 and SelfAttention_v2")
print(f"Self_Attention_v1.W_query:\n{sa_v1.W_query}")
print(f"Self_Attention_v2.W_query.eight.shape:\n{sa_v2.W_query.weight.shape}")
print(f"Self_Attention_v2.W_query.weight:\n{sa_v2.W_query.weight}")

print()
sa_v1.W_query = torch.nn.Parameter(torch.transpose(sa_v2.W_query.weight, 0, 1))
sa_v1.W_key = torch.nn.Parameter(torch.transpose(sa_v2.W_key.weight, 0, 1))
sa_v1.W_value = torch.nn.Parameter(torch.transpose(sa_v2.W_value.weight, 0, 1))
print(f"Self_Attention_v1.W_query:\n{sa_v1.W_query}")
print(f"sa_v1(inputs):\n{sa_v1(inputs)}")










