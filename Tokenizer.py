#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 09:17:13 2025

@author: andrey
"""
import re

class Tokenizer:
    def __init__(self):
        self.word_index = {}
        self.index_word = {}

    def preprocess_text(self, text):
        # Ensure punctuation (e.g., ".") is separated from words
        text = re.sub(r"([.,!?;])", r" \1 ", text)  # Add spaces around punctuation
        text = re.sub(r"\s+", " ", text)  # Remove extra spaces
        return text.strip()

    def fit_on_texts(self, texts):
        index = 1
        for sentence in texts:
            sentence = self.preprocess_text(sentence)
            for word in sentence.split():
                if word not in self.word_index:
                    self.word_index[word] = index
                    self.index_word[index] = word
                    index += 1

    def texts_to_sequences(self, texts):
        sequences = []
        for sentence in texts:
            sentence = self.preprocess_text(sentence)
            sequence = [self.word_index[word] for word in sentence.split() if word in self.word_index]
            sequences.append(sequence)
        return sequences
