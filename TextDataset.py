#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 10:35:07 2025

@author: andrey
"""

import torch
from torch.utils.data import Dataset

class TransformerTextDataset(Dataset):
    """
    Dataset for transformer-based models, compatible with Hugging Face tokenizers.
    Yields input_ids, attention_mask, and optionally labels for next-token prediction.
    Uses the interface of a custom TransformerTokenizer for all tokenization tasks.
    """
    def __init__(self, texts, tokenizer, max_length=128, predict_steps=1):
        self.texts = texts
        self.tokenizer = tokenizer  # Should be an instance of your TransformerTokenizer class
        self.max_length = max_length
        self.predict_steps = predict_steps
        self.pad_token_id = self.tokenizer.tokenizer.pad_token_id
        self.inputs = self.build_inputs(texts)

    def build_inputs(self, texts):
        # For next-token prediction, create (input, label) pairs
        input_pairs = []
        for text in texts:
            # Tokenize once using the custom tokenizer
            tokenized = self.tokenizer.encode(
                text,
                truncation=False,
                padding=False
            )
            token_ids = tokenized["input_ids"]
            # Generate input-label pairs for each possible next-token prediction
            for i in range(1, len(token_ids) - self.predict_steps + 1):
                input_ids = token_ids[:i]
                label_ids = token_ids[i : i + self.predict_steps]
                input_pairs.append((input_ids, label_ids))
        return input_pairs

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        input_ids, label_ids = self.inputs[idx]
        # Prepare input: pad/truncate as needed using the tokenizer's encode, but on detokenized text for consistency
        # This avoids issues with special tokens, etc.
        # However, if input_ids may contain special tokens, you can wrap your tokenizer with a helper to accept ids directly.
        text = self.tokenizer.decode(input_ids)
        encoded_input = self.tokenizer.encode(
            text,
            truncation=True,
            padding="max_length"
        )
        # Prepare label_ids: pad to predict_steps
        label_ids = label_ids + [self.pad_token_id] * (self.predict_steps - len(label_ids))
        label_ids = label_ids[:self.predict_steps]
        item = {
            "input_ids": torch.tensor(encoded_input["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(encoded_input["attention_mask"], dtype=torch.long),
            "labels": torch.tensor(label_ids, dtype=torch.long)
        }
        return item

def transformer_collate_fn(batch):
    """
    Collate function to batch items from TransformerTextDataset.
    Returns dicts of input_ids, attention_mask, labels.
    """
    input_ids = torch.stack([item["input_ids"] for item in batch])
    attention_mask = torch.stack([item["attention_mask"] for item in batch])
    labels = torch.stack([item["labels"] for item in batch])
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }