#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 09:33:08 2025

@author: andrey
"""
import torch
import torch.nn as nn


# Define collate function
def collate_fn(batch):
    X, y = zip(*batch)  # Unpack inputs and targets
    lengths = [len(x) for x in X]  # Compute the lengths of each input sequence
    X_padded = nn.utils.rnn.pad_sequence(X, batch_first=True, padding_value=0)  # Pad input sequences
    y_padded = nn.utils.rnn.pad_sequence(y, batch_first=True, padding_value=0)  # Pad target sequences
    return X_padded, torch.tensor(lengths), y_padded
