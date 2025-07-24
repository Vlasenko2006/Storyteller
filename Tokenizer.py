#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 09:17:13 2025

@author: andrey
"""

import torch
from transformers import AutoTokenizer

class TransformerTokenizer:
    def __init__(self, max_length=128, pretrained_model_path_or_name="bert-base-uncased"):
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_path_or_name)
        self.max_length = max_length
        self.max_length = max_length

    def preprocess_text(self, text):
        """Apply any text preprocessing (customize as needed)."""
        return text.strip()

    def encode(self, text, truncation=True, padding="max_length", max_length=None):
        """
        Encode a single text string into token ids, with padding/truncation.
        Returns a dictionary compatible with transformers models.
        """
        text = self.preprocess_text(text)
        max_length = max_length if max_length is not None else self.max_length
        return self.tokenizer(
            text,
            max_length=max_length,
            truncation=truncation,
            padding=padding,
            return_tensors=None
        )

    def batch_encode(self, texts, truncation=True, padding="max_length", max_length=None):
        texts = [self.preprocess_text(t) for t in texts]
        max_length = max_length if max_length is not None else self.max_length
        return self.tokenizer(
            texts,
            max_length=max_length,
            truncation=truncation,
            padding=padding,
            return_tensors="pt"
        )

    def build_attention_mask(self, input_ids):
        """
        Given a list of input_ids, returns an attention mask: 
        1 for real tokens, 0 for padding tokens.
        """
        pad_token_id = self.tokenizer.pad_token_id
        return [0 if token_id == pad_token_id else 1 for token_id in input_ids]

    def prepare_inference_inputs(self, text):
        """
        Prepares input_ids and attention_mask for single text inference.
        """
        encoded = self.encode(text, truncation=True, padding="max_length")
        input_ids = encoded["input_ids"]
        attention_mask = encoded["attention_mask"]
        return {
            "input_ids": torch.tensor([input_ids], dtype=torch.long),
            "attention_mask": torch.tensor([attention_mask], dtype=torch.long)
        }

    def pad_labels(self, label_ids, predict_steps):
        """
        Pad (or truncate) a label sequence to length predict_steps, using the pad_token_id.
        """
        pad_token_id = self.tokenizer.pad_token_id
        padded = label_ids + [pad_token_id] * (predict_steps - len(label_ids))
        return padded[:predict_steps]

    def decode(self, token_ids):
        """Convert token ids back to text string."""
        return self.tokenizer.decode(token_ids)

    def convert_ids_to_tokens(self, token_ids):
        """Convert token ids back to token strings (subwords)."""
        return self.tokenizer.convert_ids_to_tokens(token_ids)

    @property
    def vocab_size(self):
        return self.tokenizer.vocab_size
