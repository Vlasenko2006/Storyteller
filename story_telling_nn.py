import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pickle
from tqdm import tqdm  # For progress bar
from Tokenizer import Tokenizer
from TextDataset import TextDataset
from NextWordPredictor import NextWordPredictor
from multi_word_loss import multi_word_loss
from predict_sequence import predict_sequence
from collate_fn import collate_fn



#Network Settings

batch_size = 4 * 16
embed_size = 3 * 512
hidden_size = 4 * 256
ff_hidden_size = 4 * 256
num_ff_layers = 4

# Training settings
lr = 0.0001 * 0.25
nepochs = 1000

# Predictor settings
num_words= 100
predict_after_nepochs = 5

# Paths and constants
path = "/gpfs/work/vlasenko/07/NN/Darwin/"
load_model = True
predicted_steps = 1
load_epoch = 114
checkpoint_path = f"{path}/story_telling_nn2-{predicted_steps}_ep_{load_epoch}.pth"

# Load and preprocess text corpus
with open(f"{path}Darwin_biogr_list_large", "rb") as fp:
    corpus = pickle.load(fp)  # Unpickling

tokenizer = Tokenizer()
preprocessed_corpus = [tokenizer.preprocess_text(line) for line in corpus]
tokenizer.fit_on_texts(preprocessed_corpus)
total_words = len(tokenizer.word_index) + 1

# Create dataset and DataLoader
dataset = TextDataset(corpus, tokenizer)
train_loader = DataLoader(
    dataset,
    batch_size = batch_size,
    shuffle = True,
    collate_fn = collate_fn
)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model, Loss, and Optimizer
model = NextWordPredictor(
    vocab_size = total_words,
    embed_size = embed_size,
    hidden_size = hidden_size,
    ff_hidden_size = ff_hidden_size,
    num_ff_layers = num_ff_layers,
    predict_steps=predicted_steps
)

# Load model checkpoint if required

if load_model:
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint)

model.to(device)  # Move model to device

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = lr)

# Training loop
for epoch in range(nepochs):
    model.train()
    epoch_loss = 0
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}")

    for X, lengths, y in progress_bar:
        
        X, lengths, y = X.to(device), lengths.to(device), y.to(device)

        optimizer.zero_grad()
        output = model(X, lengths)
        loss = multi_word_loss(output, y, criterion)
        loss.backward()
        optimizer.step()

        # Update progress bar and epoch loss
        epoch_loss += loss.item()
        progress_bar.set_postfix(loss=loss.item())

    # Print epoch summary
    avg_loss = epoch_loss / len(train_loader)
    print(f"NN2 Epoch {epoch + 1}, Average Loss: {avg_loss:.4f}")

    # Save model checkpoint
    torch.save(
        model.state_dict(),
        f"{path}/story_telling_nn2-rep-{predicted_steps}_ep_{epoch}.pth"
    )

    # Predict and print sequences every 5 epochs
    if (epoch + 1) % predict_after_nepochs == 0:
        print("Predicting 0:")
        seed_text = "charles darwin together with little red riding hood"
        text = predict_sequence(seed_text, model, tokenizer, num_words = num_words, device=device)
        print(text)
        print("Predicting 1:")

