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
from Tokenizer import TransformerTokenizer
from TextDataset import TransformerTextDataset, transformer_collate_fn
from LabelSmoothingLoss import LabelSmoothingLoss
from my_trainer import my_trainer
from seeders import seeders
from final_text import final_text
from model import MiniBertForNextWordPrediction, MiniTransformerWithEncoderDecoder 
import yaml 
from clean_text import clean_text

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# Load config
yaml_path = "config/conf.yaml"
with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)

# --------------------------
# DEBUGGING OPTION
DEBUG_FIRST_N_BATCHES = config["DEBUG_FIRST_N_BATCHES"] 
# --------------------------

    
# Unpack config
model_cfg = config['model']
training_cfg = config['training']
predictor_cfg = config['predictor']
path = config['path']
predicted_steps = config['predicted_steps']
load_epoch = config['load_epoch']
checkpoint_path = config['checkpoint_path']

use_adamw = config['use_adamw']
alternate_costs = config['alternate_costs']
train_the_model = config['train_the_model']
load_model = config['load_model']


# Load and preprocess text corpus
with open(f"{path}text", "rb") as fp:
    corpus = pickle.load(fp)  

with open(f"{path}preprocessed_qa_dataset_plain.pkl", "rb") as fp:
    qa = pickle.load(fp)  

print(type(corpus))
# If it's a list
if isinstance(corpus, list):
    print(type(corpus[0]))  # e.g., str


corpus = clean_text(corpus)
corpus = corpus + qa



tokenizer = TransformerTokenizer(max_length=model_cfg['max_length'],
                                 pretrained_model_path_or_name = config['pretrained_model_path_or_name'])
total_words = tokenizer.vocab_size # + 1 #FIXME +1 might be an issue

# Create dataset and DataLoader
dataset = TransformerTextDataset(
    corpus, 
    tokenizer, 
    max_length=model_cfg['max_length'], 
    predict_steps=predicted_steps,
    for_encoder_decoder = ( config["model_type"] == "BART" )
)

train_loader = DataLoader(
    dataset,
    batch_size=training_cfg['batch_size'],
    shuffle=True,
    collate_fn=transformer_collate_fn,
    drop_last=True
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model instantiation using **kwargs from model config
if config["model_type"] == "BERT":
    model = MiniBertForNextWordPrediction(
        vocab_size=total_words,
        **model_cfg
    )
    
if config["model_type"] == "BART":
    model =MiniTransformerWithEncoderDecoder(
        vocab_size=total_words,
        **model_cfg
    )

criterion_ls = LabelSmoothingLoss(classes=total_words, smoothing=training_cfg['smoothing'])
criterion_ce = nn.CrossEntropyLoss()

# --- Optimizer (AdamW or Adam) ---
if use_adamw:
    optimizer_ce = torch.optim.AdamW(model.parameters(), lr=training_cfg['lr_ce'])
    optimizer_ls = torch.optim.AdamW(model.parameters(), lr=training_cfg['lr_ls'])
    optimizer = optimizer_ce
else:
    optimizer_ce = torch.optim.Adam(model.parameters(), lr=training_cfg['lr_ce'])
    optimizer_ls = torch.optim.Adam(model.parameters(), lr=training_cfg['lr_ls'])
    optimizer = optimizer_ce

model.to(device)

# Load model checkpoint if required
if load_model:
    checkpoint_path_load = os.path.join(checkpoint_path, f"story_telling-{config['predicted_steps']}_ep_{config['load_epoch']}.pth")
    print("Loading checkpoint: ", checkpoint_path)
    checkpoint = torch.load(checkpoint_path_load, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    optimizer_ce = optimizer
    start_epoch = checkpoint.get("epoch", 0) + 1           
else:
    start_epoch = 0  # Train from scratch  

if train_the_model:
    my_trainer(
        nepochs=training_cfg['nepochs'],
        path=checkpoint_path,
        alternate_costs=alternate_costs,
        criterion_ce=criterion_ce,
        criterion_ls=criterion_ls,
        optimizer_ce=optimizer_ce,
        optimizer_ls=optimizer_ls,
        model=model,
        train_loader=train_loader,
        tokenizer=tokenizer,
        device=device,
        predicted_steps=predicted_steps,
        validate_after_nepochs=predictor_cfg['validate_after_nepochs'],
        seeders=seeders,
        start_epoch=start_epoch,
        model_type= config["model_type"],
        DEBUG_FIRST_N_BATCHES = DEBUG_FIRST_N_BATCHES
    )

else:
    assert load_model, "To validate only you must set load_model to 'True' and specify the loading checkpoint!"
    for index, seeder in enumerate(seeders, start=1):
        text = final_text(
            seeder,
            model,
            tokenizer,
            num_words=predictor_cfg['num_words'],
            device=device
        )
        print(f"{index}: {text[0]}")
