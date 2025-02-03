#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  1 09:52:40 2025

@author: andreyvlasenko
"""

import os
import torch
import pickle
import requests
from flask import Flask, request, jsonify
from Tokenizer import Tokenizer
from NextWordPredictor import NextWordPredictor

# Configurations
MODEL_URL = "https://example.com/model_checkpoint.pth"  # Replace with actual URL
CORPUS_PATH = "/Users/andreyvlasenko/tst/GitHUB/Storyteller/data/"
CHECKPOINT_PATH = "/Users/andreyvlasenko/tst/GitHUB/Storyteller/data/checkpoint_test.pth"
TOKENIZER_PATH = "data/tokenizer.pkl"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Network Settings 
batch_size = 10 * 4 * 16
embed_size = 3 * 512
hidden_size = 4 * 256
ff_hidden_size = 4 * 256
num_ff_layers = 4
dropout = 0.00
predicted_steps = 1


# Ensure model checkpoint exists
if not os.path.exists(CHECKPOINT_PATH):
    print(f"Downloading model from {MODEL_URL}...")
    response = requests.get(MODEL_URL)
    with open(CHECKPOINT_PATH, "wb") as f:
        f.write(response.content)
    print("Model downloaded successfully!")



# Load and preprocess text corpus
with open(f"{CORPUS_PATH}Darwin_biogr_list_large", "rb") as fp:
    corpus = pickle.load(fp)

tokenizer = Tokenizer()
preprocessed_corpus = [tokenizer.preprocess_text(line) for line in corpus]
tokenizer.fit_on_texts(preprocessed_corpus)
total_words = len(tokenizer.word_index) + 1


# Load tokenizer
# with open(TOKENIZER_PATH, "rb") as fp:
#     tokenizer = pickle.load(fp)

# Load model
model = NextWordPredictor(
    vocab_size=total_words,
    embed_size=embed_size,
    hidden_size=hidden_size,
    ff_hidden_size=ff_hidden_size,
    num_ff_layers=num_ff_layers,
    predict_steps=predicted_steps,
    dropout=dropout
)

model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    seed_text = data.get("seed_text", "")
    num_words = data.get("num_words", 50)

    if not seed_text:
        return jsonify({"error": "Seed text is required"}), 400

    prediction = predict_sequence(seed_text, model, tokenizer, num_words)
    return jsonify({"generated_text": prediction})

def predict_sequence(seed_text, model, tokenizer, num_words=100):
    model.eval()
    with torch.no_grad():
        for _ in range(num_words):
            token_list = tokenizer.texts_to_sequences([seed_text])[0]
            input_tensor = torch.tensor([token_list], dtype=torch.long).to(DEVICE)
            lengths = torch.tensor([len(token_list)]).to(DEVICE)
            
            output = model(input_tensor, lengths)
            predicted_indices = torch.argmax(output, dim=2)
            predicted_words = [tokenizer.index_word.get(idx.item(), "NaN") for idx in predicted_indices[0]]
            
            seed_text += " " + " ".join(predicted_words)
            if "NaN" in predicted_words:
                break

    return seed_text

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
