#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 31 10:03:36 2025

@author: andrey
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pickle
from Tokenizer import Tokenizer
from TextDataset import TextDataset
from NextWordPredictor import NextWordPredictor
from collate_fn import collate_fn
from LabelSmoothingLoss import LabelSmoothingLoss
from my_trainer import my_trainer
from seeders import seeders
from final_text import final_text

# Flags for optional features 
use_adamw = True  # Use AdamW instead of Adam
alternate_costs = True  # Apply Label Smoothing
train_the_model = True
load_model = True

# Network Settings 
batch_size = 10 * 4 * 16
embed_size = 3 * 512
hidden_size = 4 * 256
ff_hidden_size = 4 * 256
num_ff_layers = 4
dropout = 0.00

# Training settings
lr_ce = 0.0001 * 0.25 * 0.025 * 0.25
lr_ls = 0.0001 * 0.25
smoothing = 0.005
nepochs = 1000

# Predictor settings
num_words = 30
validate_after_nepochs = 1

# Paths and constants
path = "/gpfs/work/vlasenko/07/NN/Darwin/"
predicted_steps = 1
load_epoch = 9
checkpoint_path = f"{path}/story_telling_final-{predicted_steps}_ep_{load_epoch}.pth"

# Load and preprocess text corpus
with open(f"{path}Darwin_biogr_list_large", "rb") as fp:
    corpus = pickle.load(fp)  

tokenizer = Tokenizer()
preprocessed_corpus = [tokenizer.preprocess_text(line) for line in corpus]
tokenizer.fit_on_texts(preprocessed_corpus)
total_words = len(tokenizer.word_index) + 1

# Create dataset and DataLoader
dataset = TextDataset(corpus, tokenizer)
train_loader = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=True,
    collate_fn=collate_fn
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


model = NextWordPredictor(
    vocab_size=total_words,
    embed_size=embed_size,
    hidden_size=hidden_size,
    ff_hidden_size=ff_hidden_size,
    num_ff_layers=num_ff_layers,
    predict_steps=predicted_steps,
    dropout=dropout
)



criterion_ls = LabelSmoothingLoss(classes=total_words, smoothing=smoothing)
criterion_ce = nn.CrossEntropyLoss()

# --- Optimizer (AdamW or Adam) ---
if use_adamw:
    optimizer_ce = torch.optim.AdamW(model.parameters(), lr=lr_ce)
    optimizer_ls = torch.optim.AdamW(model.parameters(), lr=lr_ls)
else:
    optimizer_ce = torch.optim.Adam(model.parameters(), lr=lr_ce)
    optimizer_ls = torch.optim.Adam(model.parameters(), lr=lr_ls)


# Load model checkpoint if required

if load_model:
    print("Loading checkpoint: ", checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint)                      #FIXME Apparently the model does not load optimizer settings. This must be fixed by next commit
   
    start_epoch = checkpoint.get("epoch", 0) + 1           
else:
    start_epoch = 0  # Train from scratch


model.to(device)  


if train_the_model:
    my_trainer(
        nepochs,
        path,
        alternate_costs,
        criterion_ce,
        criterion_ls,
        optimizer_ce,
        optimizer_ls,
        model,
        train_loader,
        tokenizer,
        device=device,
        predicted_steps=1,
        validate_after_nepochs=validate_after_nepochs,
        seeders=seeders,
        start_epoch = start_epoch
    )
else:
    assert load_model, "To validate only you must set load_model to 'True' and specify the loading checkpoint!"
    for index, seeder in enumerate(seeders, start=1):
        text = final_text(
            seeder,
            model,
            tokenizer,
            num_words=100,
            device=device
        )
        print(f"{index}: {text[0]}")
