# 2.6 Data sampling with a sliding window
import urllib.request
import tiktoken
from ai_shared_data import ensure_asset, get_asset_path

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
file_path = get_asset_path("john")

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
print(f"Input (x): {x}")
print(f"Target (y): {y}")

