#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 31 10:03:36 2025

@author: andrey
"""

import torch
from translate_output_to_words import translate_output_to_words
import re


# Function to clean and format text
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

def final_text(seeder,
                     model, 
                     tokenizer,
                     num_words=100,
                     device = "cpu"):
    model.eval()
    c = 1
    seed_text = seeder[0]
    with torch.no_grad():
        for _ in range(num_words):
            token_list = tokenizer.texts_to_sequences([seed_text])[0]
            input_tensor = torch.tensor([token_list], dtype=torch.long).to(device)  # Move to device
            lengths = torch.tensor([len(token_list)]).to(device)  # Move to device
            
            # Predict next tokens
            output = model(input_tensor, lengths)  # Shape: (1, predict_steps, vocab_size)
            
            # Convert output to words
            predicted_sequences = translate_output_to_words(output, tokenizer)  # List of predicted sequences
            
            # Use the first batch's predictions (batch size = 1 during inference)
            predicted_words = predicted_sequences[0]
            
            # Append predicted words to the seed text
            if predicted_words[0] == "." and len(seeder) > c:
                seed_text += " " + " ".join(predicted_words) + " " + seeder[c]
                c += 1
            else:
                seed_text += " " + " ".join(predicted_words)

            
            if "NaN" in predicted_words:  # Stop if unknown word is predicted
                break
    text = clean_and_format_text([seed_text])
    return text
