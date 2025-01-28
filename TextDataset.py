#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 10:35:07 2025

@author: andrey
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset

# Define TextDataset class
class TextDataset(Dataset):
    def __init__(self, corpus, tokenizer, predict_steps=3):
        self.sequences = []
        self.predict_steps = predict_steps
        for line in corpus:
            token_list = tokenizer.texts_to_sequences([line])[0]
            for i in range(1, len(token_list)):
                n_gram_sequence = token_list[:i + 1]
                if len(n_gram_sequence) > predict_steps:
                    self.sequences.append((n_gram_sequence[:-predict_steps], n_gram_sequence[-predict_steps:]))

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, index):
        X, y = self.sequences[index]
        return torch.tensor(X, dtype=torch.long), torch.tensor(y, dtype=torch.long)

# Define collate function
def collate_fn(batch):
    X, y = zip(*batch)  # Unpack inputs and targets
    lengths = [len(x) for x in X]  # Compute the lengths of each input sequence
    X_padded = nn.utils.rnn.pad_sequence(X, batch_first=True, padding_value=0)  # Pad input sequences
    y_padded = nn.utils.rnn.pad_sequence(y, batch_first=True, padding_value=0)  # Pad target sequences
    return X_padded, torch.tensor(lengths), y_padded



