# 2.2 Tokenizing Text
import urllib.request
import re
from ai_shared_data import ensure_asset, get_asset_path

print()
print("Listing 2.3 Implementing a simple text tokenizer")
class SimpleTokenizerV1:
    def __init__(self, vocab):
        # Stores the vocabulary as a class attribute for
        # access in the encode and decode methods
        self.str_to_int = vocab
        # Creates an inverse vocabluary that maps token IDs back to the original text tokens
        self.int_to_str = {i:s for s, i in vocab.items()}

    # Processes input text into token IDs
    def encode(self, text):
        preprocessed = re.split(r'([.,:;?_!"()\']|--|\s)', text) 
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        # Replaces unknown words by <|unk|> tokens
        preprocessed = [item if item in self.str_to_int 
                        else "<unk>" for item in preprocessed]

        ids = [self.str_to_int[s] for s in preprocessed]
        return ids

    def decode(self, ids):
        # Converts token IDs back into text
        text = " ".join([self.int_to_str[i] for i in ids])
        # Replaces spaces before the specified puncuations
        text = re.sub(r'\s+([.,:;?_!"()\'])', r'\1', text)
        return text

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
vocab = {token:integer for integer, token in enumerate(all_words)}
print("Vocabulary size:", vocab_size)

print()
print("2.4 Adding special context tokens")

print()
print("Adding <unk> and <|endoftext|> tokens to the vocabulary")
all_tokens = sorted(list(set(preprocessed)))
all_tokens.extend(["<unk>", "<|endoftext|>"])
vocab = {token:integer for integer, token in enumerate(all_tokens)}
print("Vocabulary size:", len(vocab))

print()
print("Instantiate a new tokenizer object from the SimpleTokenizerV1 class, to tokenize the NT text sample")
tokenizer = SimpleTokenizerV1(vocab)
text = """And Joseph her husband, being a righteous man, and not willing to make her a public example, was minded to put her away privily.
        But when he thought on these things, behold, an angel"""
ids = tokenizer.encode(text)
print("Token IDs:", ids)
print()
print("Decode the token IDs back into text")
text_decoded = tokenizer.decode(ids)
print("Decoded text:", text_decoded)

print()
print("Tokenize a text sample that contains unknown words")
text1 = "Hello, do you like tea?"
text2 = "In the sunlit terraces of the palace."
text = " <|endoftext|> ".join([text1, text2])
print("New text to text: ", text)

print()
print("Tokenize the new text sample")
print("Token IDs:", tokenizer.encode(text))

print()
print("Decoded text:", tokenizer.decode(tokenizer.encode(text)))



