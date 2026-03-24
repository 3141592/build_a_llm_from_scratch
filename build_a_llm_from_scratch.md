# Build a Large Language Model

This respoitory includes the code from my working through the book to build an LLM.

## Shared Data Repository

This project depends on the shared asset repository:

https://github.com/3141592/ai_shared_data

Datasets and model assets are **not stored in this repository**.  
They are managed through `ai_shared_data`.

Example setup:

```bash
git clone https://github.com/3141592/ai_shared_data
pip install -e ai_shared_data
```

See that repository for dataset download and configuration instructions.

## Understanding large language models

### 03-24-2026

- Section 2.7 Creating token embeddings
  - Created a small embedding.
- Next 2.8 Encoding word positions


### 03-19-2026

- Section 2.6 Data sampling with sliding window
  - The sliding window demonstrates how data to train a model to identify the next token is created.
- NEXT Section 2.7 Creating token embeddings

### 03-18-2026

- Section 2.5 Byte pair encoding
  - Saw that the BPE tokenizer breaks words into subwords but can put them back togther
  - BPE works by traditional algorithms
- Section 2.6 Data sampling with sliding window
  - Stopped on p.36

### 03-17-2026

- Section 2.4 Adding special context tokens
  - Added unk and <|endoftext|> tokens to the vocabulary and the tokenizer.
  - Note: I am using the New Testament instead of The Verdict, so results will differ from the book.


### 03-13-2026

- Completed 2.3 Converting tokens into token IDs
- I now grasp that tokenzing, encoding, and decoding are straightforward processes.
- This section lacks handling for unknown tokens (tokens not included in the vocabulary)
  but it sounds like this is dealt with in the next sectio, 2.4.


### 03-06-2026

Starting 2.3 Converting tokens into token IDs

### 03-05-2026

2.2 Tokenizing text

### 03-04-2026

1.3 Stages of building and using LLMs

- Fine-tuning is further training of the LLM on labeled data.
  - instruction fine-tuning: instruction and answer pairs
  - classification fine-tuning: texts and class label pairs (span, not spam)

