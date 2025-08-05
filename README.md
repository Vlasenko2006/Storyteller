# Neural Networks for Story Generation

## In simple words
Picture a teacher asking an unprepared student who isn't particularly studious but is highly inventive to answer. He gives fantastically creative but typically wrong answers, entertaining the whole class. The neural networks that I developed in the scope of this project LSTMsky (currrent branch), and BERTsky, BARTsky (see branch "transformers") are designed to behave in precisely this way, generating whimsical and fanciful responses. In the world of language models, this tendency to confidently produce convincing but untrue information is called “hallucination.” While typically considered a flaw, here it’s exactly what I want: perfect hallucinations—answers that sound plausible, are woven from real facts, yet together never reflect the actual truth.

Although the neural network is trained on a corpus of Wikipedia articles, it must generate connected, logically coherent, reasonable text **WITHOUT DIRECT CITING** (!!!) pieces from these articles. Consider the network successfully generates the text if a layman in the subject would believe it, and it fails if it outputs nonsense or a quote from a Wikipedia article! This kind of language model output is known as halucination and typicaly unwanted in the most of cases, but is highly demanded in this Storyteller project.

A special parameter “smoothing” (do not confuse with the temperature) regulates the intellectual content of the generated text. The network trained with small “smoothing” values returns “academic” texts, while the large values make it like a “schoolboy” answer.

## Examples 
Find the examples of the generated text with specified smoothing values below. Note that ```Smoothing = 0``` should correspond to the highly intelligent text, ```Smoothing = 0.1``` should correspond to the schoolboy text. Seeded text in bold. The training is ongoing, and the current model output may have some roughness in the texts. However, it gives you a flavor of the network's capabilities.

### Smoothing = 0.01 (Smart answers).
1.  **In the early twentieth century, it was suggested that**```to develop a consistent understanding of the fundamental concepts of mathematics, it was sufficient to study observation. For example, a single electron in an unexcited atom is classically depicted as a particle moving in a circular path around the atomic nucleus...```
2.  **Charles_darwin in his book** ```The Road to Serfdom (1944), Friedrich Hayek (1899–1992) asserted that the free-market understanding of economic freedom as present in capitalism is a requisite of political freedom. This philosophy is really sent to think that are said to be true of the evil trait that is very possible for it. Although many slaves have escaped or have been freed since 2007, as of 2012, only one slave owner had been sentenced to serve time in prison.```
3.  **The idea of philosophy is** ```a source of academic discussion. ```
4.  **The story begins with** ```a chapter on the island of Thrinacia, with the crew overriding odysseus's wishes to remain away from the island. ```
5.  **Mathematics is one of** ```is one of the most important forms of philosophical knowledge.``` 

### Smoothing = 0.025 (normal answers).
1. **In the early twentieth century, it was suggested that**``` the chinese crossbow was transmitted to the roman world on such occasions, although the greek gastraphetes provides an alternative origin.``` (My comment: gastraphetes is an acient greek crossbow)
2. **Charles_darwin in his book** ```The Road to Serfdom (1944), friedrich hayek (1899–1992) asserted that the free-market understanding of economic freedom as present in capitalism is a requisite of political freedom. This philosophy is really not 206 and stated that it is good for the consequences of actions.```
3. **The idea of philosophy is** ```a myth. ```
4. **The story begins with** ```a chapter on the Islands of Weathertop, and is known as Five Years.```
5. **Mathematics is one of** ```the most important aspects of the argues of the mathematicians. ```

## Challenge:
Since I am limited in computational resourses, this storyteller must be trained on a single old GPU in reasonable time (days). 

---
   
## Overview

This neural network, the LSTMsky, is an old-fasioned NLP model with LSTM layer in its core (see also my transformer-based BERTsky and BARTsky). It predicts the next words in the beginning of a sentence (the prompt), enabling it to generate text that continues an input seed text. The model is trained on a text corpus, tokenized words using original tokenizer, and converted into numerical sequences for learning. The architecture uses embeddings, LSTMs, and feed-forward layers. 

This neural network (NN) predicts the following words in a text sequence (incomplete sentence). It accepts a phrase and continues it as long as needed, setting appropriate punctuation. The purpose of this NN is:  
1. Test whether a NN can instantly fool the software aimed to detect AI-generated texts.
2. A demonstrative and simple example of natural language processing NN
3. Entertain. The NN produces funny stories, making you think if this is real.

## How it works:

The model is trained on a text corpus (generated by `Extract_wiki_text_content.py`), tokenized, and converted into numerical sequences for learning. The architecture uses embeddings, LSTMs, and feed-forward layers.  Note that NN uses its own tokenization instead of the nltk package, allowing its potential users to inspect its machinery.

---

## ToDo
Training does not have a sheduler that reduces learning rate, adds the second cost function, etc. . This is still done manualy, but it will be fixed in the nearest future.

## Steps in the Pipeline

### 1. **Data Preprocessing**
- **Corpus Loading**: The dataset created by `Extract_wiki_text_content.py` is loaded using Python's `pickle` module.
- **Tokenizer**:
  - A custom tokenizer preprocesses text by adding spaces around punctuation and mapping words to unique indices.
  - The `Tokenizer` class includes methods to preprocess text, fit the tokenizer on a corpus, and convert text to sequences of indices.

### 2. **Dataset and DataLoader**
- **TextDataset**:
  - Converts the tokenized corpus into input-output pairs for training. For each sequence, `n-gram` sequences are created where a portion of the sequence is input, and the subsequent tokens are the target for prediction.
  - The dataset supports multi-word prediction through a `predict_steps` parameter.

- **DataLoader**:
  - Handles batching, shuffling, and padding sequences to ensure that batches can be efficiently processed by the model. A custom `collate_fn` function is used for padding.

### 3. **Model Architecture**
The **NextWordPredictor** model is designed to handle multi-word predictions and consists of the following components:
- **Embedding Layer**:
  - Converts input tokens into dense vector representations of size `embed_size`.
- **LSTM**:
  - A two-layer LSTM processes the input embeddings, capturing temporal dependencies in the sequence.
- **Layer Normalization**:
  - Normalizes the output of the LSTM's final hidden state for improved stability.
- **Feed-Forward Layers**:
  - A series of fully connected layers (optionally with BatchNorm) process the hidden state to generate predictions.
- **Final Linear Layer**:
  - Outputs a tensor of shape `(batch_size, predict_steps, vocab_size)` containing predictions for multiple words.
- **Custom Weight Initialization**:
  - Xavier initialization is used for weights, and biases are initialized to zero for better convergence.

## LSTM or Attention Layers?
Modern natural language processing models increasingly rely on attention mechanisms, or combine attention layers with LSTM architectures, rather than using pure LSTM. Attention-based models generally achieve superior performance on long and complex texts, but they often require more computational resources and longer training times.

In contrast, LSTM networks are faster to train and more resource-efficient. For short text generation tasks (up to 500 words), pure LSTM architectures can actually outperform their attention-based counterparts, offering a practical and effective solution.

If you’re interested in a hands-on comparison between attention-based and LSTM-based models, check out the **transformer** branch of the Storyteller project, which features models built with attention mechanisms.

### 4. **Loss Function**
The model uses two loss functions. The first loss is 1A custom loss function (`multi_word_loss`). It computes the average cross-entropy across the predicted steps. It is quite conventional for natural language processing neural networks.

The second loss `LabelSmoothingLoss` is also a cross-entropy loss multiplied by a smoothing parameter that damps the target word probability and increases the probability of other words from the corpus. It helps avoid overconfidence. In other words, this cost mimics your hesitation about the correct answer to the question. The second cost must switch when the training after the consequent learning rate reduction reaches the plateau. It helps to continue further training. The switch is done maually so far and will be automated in the future.

**Good cost values.** The model start producing meaningful text when the multi_word_loss returns values smaller than 0.35 .  

### 5. **Inference**
- The model generates text by recursively predicting the next tokens for a given seed text.
- Predictions are translated back to words using the tokenizer's `index_word` dictionary.
- Since the neural network surves to amuse a user, the user does the inference: the model can be considered as well trained as soon as the user finds most of the answers amasing and logically structed.
- The user must put the seeding text in the `seeders.py` for inference.  

### 6. **Training**
- The training loop uses Adam optimizer in the beginning of the training and then it alternates `Adam` between `AdamW`, once it reachesw paltau.
---

## Usage

Copy zip file with the code from this repository and unzip it in your home folder or run in your terminal:

```bash
git clone https://github.com/Vlasenko2006/Storyteller.git
```
Once you get the code you would need to install the required packages:

- Python 3.8+
- PyTorch
- tqdm
- scikit-learn
- numpy
- Anaconda (recommended for managing the environment)

I recomend you install the the dependences with anaconda [(anaconda)](https://anaconda.org) using a `.yaml` file (see below).

---

## YAML File: Anaconda Environment Configuration

Find the `environment.yml` in your folder. If you don't have it, copy and save the code below as `environment.yml`, and run `conda env create -f environment.yml` to create the environment.

```yaml
name: story_gen
channels:
  - defaults
  - conda-forge
dependencies:
  - python=3.8
  - pytorch=1.10
  - torchvision
  - torchaudio
  - pytorch-cuda=11.3
  - tqdm
  - scikit-learn
  - numpy
  - pyyaml
  - pip
  - pip:
      - wikipedia-api
```

---

### **Command to Activate the Environment**
Once you created your environment, activate it running the code below in your terminal:

```bash
conda activate story_gen
```

### **Run the Script**
Once the environment is activated, you can run the script:

```bash
python story_telling_nn.py
```

---

### **Training the Model**
Run the provided script to:
1. Load the dataset.
2. Train the model on the dataset.
3. Periodically save checkpoints and generate text predictions.

### **Generating Text**
After training, you can generate new stories by providing a seed text to the `predict_sequence` function.

---

## Features
- Handles multi-word predictions.
- Customizable architecture:
  - Embedding size, LSTM size, feed-forward layers, and more can be adjusted.
- Flexible tokenizer with preprocessed text.
- Trains efficiently using `DataLoader` with padding support.

---




