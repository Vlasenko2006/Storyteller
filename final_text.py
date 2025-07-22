#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 31 10:03:36 2025

@author: andrey
"""

import torch
import re

def clean_and_format_text(text):
    formatted_text = []
    for paragraph in text:
        # Remove spaces before dots and commas
        paragraph = re.sub(r'\s+([.,])', r'\1', paragraph)
        # Capitalize the first letter of each sentence
        paragraph = re.sub(r'(?<=\.\s)([a-z])', lambda match: match.group(1).upper(), paragraph)  # After a period
        paragraph = re.sub(r'^([a-z])', lambda match: match.group(1).upper(), paragraph)  # First letter in paragraph
        formatted_text.append(paragraph)
    return formatted_text

def final_text(seeder, model, tokenizer, num_words=100, device="cpu"):
    model.eval()
    c = 1
    seed_text = seeder[0]
    with torch.no_grad():
        for _ in range(num_words):
            # Prepare input_ids and attention_mask via tokenizer
            inputs = tokenizer.prepare_inference_inputs(seed_text)
            input_ids = inputs["input_ids"].to(device)
            attention_mask = inputs["attention_mask"].to(device)

            # Model forward (handles both [batch, vocab] and [batch, steps, vocab])
            output = model(input_ids, attention_mask=attention_mask)
            if output.dim() == 3:
                # [batch, steps, vocab], take the last step
                output = output[:, -1, :]

            # Get predicted token id (highest probability)
            predicted_token_id = output.argmax(-1).item()
            predicted_word = tokenizer.decode([predicted_token_id]).strip()

            # Stop if unknown word, padding, or EOS
            if predicted_word.lower() in ["nan", tokenizer.tokenizer.eos_token, tokenizer.tokenizer.pad_token]:
                break

            # Append predicted word to the seed text
            if predicted_word == "." and len(seeder) > c:
                seed_text += " " + predicted_word + " " + seeder[c]
                c += 1
            else:
                seed_text += " " + predicted_word

    text = clean_and_format_text([seed_text])
    return text