# Load dataset and preprocessing utilities
from datasets import load_dataset
from torch.utils.data import DataLoader
import torch
from collections import Counter
from itertools import chain

# Tokenizer for Chinese - split characters
def tokenize_zh(text):
    return list(text.strip())

# Tokenizer for English - lowercase and split by space
def tokenize_en(text):
    return text.lower().strip().split()

# Vocabulary class to map tokens to indices and back
class Vocab:
    def __init__(self, tokens, max_size=10000, min_freq=1, specials=['<pad>', '<sos>', '<eos>', '<unk>']):
        counter = Counter(tokens)
        self.itos = specials.copy()
        for token, freq in counter.items():
            if freq >= min_freq and token not in specials:
                self.itos.append(token)
            if len(self.itos) >= max_size:
                break
        self.stoi = {tok: i for i, tok in enumerate(self.itos)}

    def __len__(self):
        return len(self.itos)

    def encode(self, tokens):
        # Convert tokens to indices
        return [self.stoi.get(tok, self.stoi['<unk>']) for tok in tokens]

    def decode(self, ids):
        # Convert indices back to tokens
        return [self.itos[i] for i in ids if i < len(self.itos)]

# Preprocess dataset and convert to tensor pairs
def preprocess_dataset(dataset, src_vocab, trg_vocab, max_len=50):
    data = []
    for item in dataset:
        zh = item["output"]  # Chinese sentence
        en = item["input"]   # English sentence

        src_tokens = ['<sos>'] + tokenize_zh(zh) + ['<eos>']
        trg_tokens = ['<sos>'] + tokenize_en(en) + ['<eos>']

        if len(src_tokens) <= max_len and len(trg_tokens) <= max_len:
            src_ids = src_vocab.encode(src_tokens)
            trg_ids = trg_vocab.encode(trg_tokens)
            data.append((torch.tensor(src_ids), torch.tensor(trg_ids)))
    return data

# Padding and batching function
def collate_fn(batch, pad_idx=0):
    src_batch, trg_batch = zip(*batch)
    src_padded = torch.nn.utils.rnn.pad_sequence(src_batch, padding_value=pad_idx)
    trg_padded = torch.nn.utils.rnn.pad_sequence(trg_batch, padding_value=pad_idx)
    return src_padded, trg_padded

# Load dataset, build vocab, preprocess, and return DataLoaders
def load_data_from_huggingface(batch_size, device, max_samples=1000):
    dataset = load_dataset("Heng666/OpenSubtitles-TW-Corpus", "en-zh_tw", split="train")
    dataset = dataset.select(range(min(max_samples, len(dataset))))

    # Build source and target vocabularies
    all_src_tokens = list(chain.from_iterable([tokenize_zh(item["output"]) for item in dataset]))
    all_trg_tokens = list(chain.from_iterable([tokenize_en(item["input"]) for item in dataset]))

    src_vocab = Vocab(all_src_tokens, max_size=2000)
    trg_vocab = Vocab(all_trg_tokens, max_size=2000)

    # Convert dataset to training examples
    data = preprocess_dataset(dataset, src_vocab, trg_vocab)
    split_idx = int(len(data) * 0.8)
    train_data = data[:split_idx]
    valid_data = data[split_idx:]

    # Create data loaders
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True,
                              collate_fn=lambda x: collate_fn(x, pad_idx=src_vocab.stoi['<pad>']))
    valid_loader = DataLoader(valid_data, batch_size=batch_size, shuffle=False,
                              collate_fn=lambda x: collate_fn(x, pad_idx=src_vocab.stoi['<pad>']))

    return train_loader, valid_loader, src_vocab, trg_vocab
