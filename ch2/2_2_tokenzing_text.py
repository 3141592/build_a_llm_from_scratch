# 2.2 Tokenizing Text
import urllib.request
from ai_shared_data import ensure_asset, get_asset_path

print("2.2 Tokenizing text")
#url = ("https://raw.githubusercontent.com/rasbt/" 
#    "LLMs-from-scratch/main/ch02/01_main-chapter-code/"
#    "the-verdict.txt")
#file_path = "the-verdict.txt"
#urllib.request.urlretrieve(url, file_path)
ensure_asset("the_verdict")
ensure_asset("asv_clean_nt")
file_path = get_asset_path("the_verdict")
file_path = get_asset_path("asv_clean_nt")

print()
print("Listing 2.1 Reading in a short story as text sample into Python")
with open(file_path, "r", encoding="utf-8") as f:
    raw_text = f.read()

print("Total number of character:", len(raw_text))
print(raw_text[:99])

print()
print("Use Python's regular expression module to split the text into tokens")
import re
preprocessed = re.split(r'([.,:;?_!"()\']|--|\s)', raw_text) 
preprocessed = [item.strip() for item in preprocessed if item.strip()]
print("Number of tokens:", len(preprocessed))
print(preprocessed[:30])

print()
print("2.3 Converting tokens into token IDs")
print("Listing 2.2 Creating a vocabulary")
all_words = sorted(set(preprocessed))
vocab_size = len(all_words)
print("Vocabulary size:", vocab_size)

vocab = {token:integer for integer, token in enumerate(all_words)}
print("Print the first 51 items in the vocabulary")
for i, item in enumerate(vocab.items()):
    print(item)
    if i >= 50:
        break



