# 2.2 Tokenizing Text
import urllib.request

print("2.2 Tokenizing text")
url = ("https://raw.githubusercontent.com/rasbt/" 
    "LLMs-from-scratch/main/ch02/01_main-chapter-code/"
    "the-verdict.txt")
file_path = "the-verdict.txt"
urllib.request.urlretrieve(url, file_path)

print("Listing 2.1 Reading in a short story as text sample into Python")
with open(file_path, "r", encoding="utf-8") as f:
    raw_text = f.read()

print("Total number of character:", len(raw_text))
print(raw_text[:99])

