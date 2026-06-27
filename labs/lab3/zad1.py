from collections import Counter
from dataclasses import dataclass
import os
import numpy as np
from urllib import parse
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch import nn


PAD = "<PAD>"
UNK = "<UNK>"

@dataclass
class Instance:
    text: str
    label: str

    def __iter__(self):
        return iter((self.text, self.label))


class NLPDataset(Dataset):
    def __init__(self, file_path, text_vocab=None, label_vocab=None):
        self.instances = []
        self.text_frequencies = Counter()
        self.label_frequencies = Counter()

        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                text, label = line.rsplit(', ', 1)
                tokens = text.split()

                self.instances.append(Instance(text=tokens, label=label))
                self.text_frequencies.update(tokens)
                self.label_frequencies.update([label])

        if text_vocab is None:
            self.text_vocab = Vocab(self.text_frequencies, max_size=-1, min_freq=1)
        else:
            if not text_vocab.stoi:
                text_vocab.build_vocab(self.text_frequencies)
            self.text_vocab = text_vocab

        if label_vocab is None:
            self.label_vocab = Vocab(self.label_frequencies, label=True)
        else:
            if not label_vocab.stoi:
                label_vocab.build_vocab(self.label_frequencies)
            self.label_vocab = label_vocab

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, idx):
        instance = self.instances[idx]
        text_tensor = self.text_vocab.encode(instance.text)
        label_tensor = self.label_vocab.encode(instance.label)
        return text_tensor, label_tensor
    

class Vocab:
    def __init__(self, frequencies=None, max_size=-1, min_freq=0, label=False):
        self.max_size = max_size
        self.min_freq = min_freq
        self.label = label
        self.itos = []
        self.stoi = {}

        if frequencies is not None:
            self.build_vocab(frequencies)


    def build_vocab(self, frequencies):
        if self.label:
            self.itos = ["positive", "negative"]

        else:
            self.itos = [PAD, UNK]
            filtered = [(token, freq) for token, freq in frequencies.items() if freq >= self.min_freq]
            filtered.sort(key=lambda x: x[1], reverse=True)
            if self.max_size != -1:
                filtered = filtered[:self.max_size - len(self.itos)]
            self.itos += [token for token, freq in filtered]

        self.stoi = {token: i for i, token in enumerate(self.itos)}


    def encode(self, tokens):
        if isinstance(tokens, str):
            return torch.tensor(self.stoi.get(tokens, self.stoi.get(UNK, 0)))
        
        else:
            indices = [self.stoi.get(token, self.stoi.get(UNK, 0)) for token in tokens]
            return torch.tensor(indices)
        

def generate_embedding_matrix(vocab, embedding_dim=300, path=None):
    vocab_size = len(vocab.itos)
    embedding_dim = embedding_dim

    embeddings = np.random.randn(vocab_size, embedding_dim)
    embeddings[0] = np.zeros(embedding_dim)

    freeze = False

    if path is not None:
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.split()
                word = parts[0]
                if word in vocab.stoi:
                    vector = torch.tensor([float(x) for x in parts[1:]], dtype=torch.float32)
                    embeddings[vocab.stoi[word]] = vector
        freeze = True

    embeddings_tensor = torch.tensor(embeddings, dtype=torch.float32)
    embedding_layer = torch.nn.Embedding.from_pretrained(
        embeddings_tensor, 
        padding_idx=0, 
        freeze=freeze
    )

    return embedding_layer


def collate_fn(batch):
    """
    Arguments:
      Batch:
        list of Instances returned by `Dataset.__getitem__`.
    Returns:
      A tensor representing the input batch.
    """

    texts, labels = zip(*batch) # Assuming the instance is in tuple-like form
    lengths = torch.tensor([len(text) for text in texts]) # Needed for later
    # Process the text instances
    return texts, labels, lengths


def pad_collate_fn(batch, pad_index=0):
    texts, labels, lengths = collate_fn(batch)

    padded_texts = nn.utils.rnn.pad_sequence(
        [text.clone().detach() for text in texts],
        batch_first=True,
        padding_value=pad_index
    )

    return padded_texts, torch.stack(list(labels)), lengths
    

# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# train_dataset = NLPDataset(os.path.join(BASE_DIR, "sst_train_raw.csv"))

# instance_text, instance_label = train_dataset.instances[3]
# print(f"Text: {instance_text}")
# print(f"Label: {instance_label}")

# numericalized_text, numericalized_label = train_dataset[3]
# print(f"Numericalized text: {numericalized_text}")
# print(f"Numericalized label: {numericalized_label}")

# text_vocab = Vocab(train_dataset.text_frequencies, max_size=-1, min_freq=0)
# print(len(text_vocab.itos)) # 14806

# print(train_dataset.text_frequencies.most_common(10))

# batch_size = 2 # Only for demonstrative purposes
# shuffle = False # Only for demonstrative purposes
# train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, 
#                               shuffle=shuffle, collate_fn=pad_collate_fn)
# texts, labels, lengths = next(iter(train_dataloader))
# print(f"Texts: {texts}")
# print(f"Labels: {labels}")
# print(f"Lengths: {lengths}")
