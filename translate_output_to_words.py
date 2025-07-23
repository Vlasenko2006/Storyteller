#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 09:33:08 2025

@author: andrey
"""

import torch

def translate_output_to_words(output, tokenizer):
    """
    Translates the model's output probabilities into a sequence of words.
    Args:
        output: Tensor of shape (batch_size, predict_steps, vocab_size), containing probabilities.
        tokenizer: Tokenizer with `index_word` mapping indices to words.
    Returns:
        A list of predicted word sequences, one for each batch.
    """
    # Find the index of the highest probability word for each step
    predicted_indices = torch.argmax(output, dim=2)  # Shape: (batch_size, predict_steps)
    
    # Convert indices to words using the tokenizer
    batch_word_sequences = []
    for batch in predicted_indices:
        words = [tokenizer.index_word.get(idx.item(), "NaN") for idx in batch]
        batch_word_sequences.append(words)
    
    return batch_word_sequences
