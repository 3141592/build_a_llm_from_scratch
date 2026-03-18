# 2.5 Byte Pair Encoding

from importlib.metadata import version
import tiktoken

print()
print("2.5 Byte pair encoding")
print() 
print("tiktoken version:", version("tiktoken"))

print()
tokenizer = tiktoken.get_encoding("gpt2")
print("tokenizer name:", tokenizer.name)
print("tokenizer max tokens:", tokenizer.max_token_value)
print("tokenizer special tokens:", tokenizer._special_tokens)

print()
text = (
    "Hello, do you like tea? <|endoftext|> In the sunlit terraces"
    "of someunkknownPlace."
)
print("text:", text)

print()
integers = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
print("integers:", integers)

print()
strings = tokenizer.decode(integers)
print("strings:", strings)

print()
print("terraces of: ", tokenizer.encode("terraces of"))
print("terracesof: ", tokenizer.encode("terracesof"))

print()
print("Decode 'terraces of': ", tokenizer.decode(tokenizer.encode("terraces of")))
print("Decode 'terracesof': ", tokenizer.decode(tokenizer.encode("terracesof")))

print()
print("Decode 353: ", tokenizer.decode([353]))
print("Decode 81: ", tokenizer.decode([81]))
print("Decode 2114: ", tokenizer.decode([2114]))
print("Decode 286: ", tokenizer.decode([286]))
print("Decode 1659: ", tokenizer.decode([1659]))

print()
print(tokenizer.encode(" of"))
print(tokenizer.encode("of"))

print()
text2 = (
    "Hello, do you like tea? In the sunlit terraces"
    "of someunkknownPlace."
)
print("Length of encoded text2:", len(tokenizer.encode(text2)))
print("Length of UTF-8 encoded text2:", len(text2.encode("utf-8")))

print() 

