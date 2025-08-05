# Transformer-based Neural Networks for Story Generation

See also the main barnch with the LSTM-based NLP model.

# Storyteller: Neural NLP-Based Story Generator

**Storyteller** is an open-source project aimed at inventing realistic-sounding stories using neural networks and modern natural language processing techniques. The repository features various architectures, including Transformer-based models, and provides tools for generating synthetic texts that can be used for research in text generation, detection of algorithmically generated content, and benchmarking NLP methods.

## Features

- **Multiple Model Architectures:** Includes classic transformer-based architectures for story generation similar to BART(Meta/Facebook) and BERT (Google) models. 
    - **BERTsky** architecture is designed to restore and improve “noisy” text: inserting missing words, improving grammar, etc., but it also generates short, meaningful texts well.  
    - **BARTsky** architecture is optimal for generating new texts, considering the entire context at once.

- **Synthetic Text Generation:** Designed to create realistic and diverse training sets for further neural network research.
- **Experimentation-Friendly:** Allows for comparison of model types and training methods.
- **NLP and Machine Learning:** Utilizes state-of-the-art NLP techniques for creative story invention.


## Getting Started

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Vlasenko2006/Storyteller.git
   ```
2. **Switch to the transformer branch for attention-based models:**
   ```bash
   git checkout transformer
   ```
3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
4. **Explore the code and run experiments:**
   - LSTM and transformer models are available for story generation.
   - Scripts and notebooks demonstrate model training and text synthesis.

## Usage Example

Details and examples for running the models can be found in the code and associated notebooks on the [transformer branch](https://github.com/Vlasenko2006/Storyteller/tree/transformer).

## About

This project is developed and maintained by [Vlasenko2006](https://github.com/Vlasenko2006) as part of ongoing research in neural text generation and benchmarking.

---

For more information, contributions, or issues, please visit the [GitHub page](https://github.com/Vlasenko2006/Storyteller).


