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

def final_text(seeder, model, tokenizer, num_words=100, device='cpu', model_type="BERT"):
    # If seeder is a list, join into a single string
    if isinstance(seeder, list):
        seeder = " ".join(str(t) for t in seeder)
    gen_text = ""
    if model_type == "BART":
        # Prepare encoder input (src) and decoder input (tgt)
        src = tokenizer.encode(seeder, truncation=True, padding="max_length")
        src_input_ids = torch.tensor([src["input_ids"]], device=device)
        src_attention_mask = torch.tensor([src["attention_mask"]], device=device)
        
        # Start decoder input with BOS or pad token
        tgt_input_ids = torch.tensor([[tokenizer.tokenizer.pad_token_id]], device=device)
        tgt_attention_mask = torch.ones_like(tgt_input_ids, device=device)
        
        outputs = []
        for _ in range(num_words):
            logits = model(
                src_input_ids,
                tgt_input_ids,
                src_attention_mask=src_attention_mask,
                tgt_attention_mask=tgt_attention_mask,
            )
            next_token_id = logits[:, -1, :].argmax(-1, keepdim=True)
            outputs.append(next_token_id.item())
            tgt_input_ids = torch.cat([tgt_input_ids, next_token_id], dim=1)
            tgt_attention_mask = torch.ones_like(tgt_input_ids, device=device)
        gen_text = tokenizer.decode(outputs)
    else:
        # BERT style (single input, next-word prediction)
        encoded = tokenizer.prepare_inference_inputs(seeder)
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)
        outputs = []
        for _ in range(num_words):
            max_length = getattr(tokenizer, "model_max_length", 128)
            if input_ids.shape[1] > max_length:
                input_ids = input_ids[:, -max_length:]
                attention_mask = attention_mask[:, -max_length:]
            logits = model(input_ids, attention_mask=attention_mask)
            next_token_id = logits.argmax(-1, keepdim=True)
            outputs.append(next_token_id.item())
            input_ids = torch.cat([input_ids, next_token_id], dim=1)
            attention_mask = torch.cat([attention_mask, torch.ones_like(next_token_id)], dim=1)
        gen_text = tokenizer.decode(outputs)
    # Prepend the seeder to the generated text for clarity
    return [f"{seeder} {gen_text}"]
