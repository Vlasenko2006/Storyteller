import torch
import torch.nn as nn



class NextWordPredictor(nn.Module):
    def __init__(self, vocab_size, embed_size= 3 * 512, hidden_size=256, ff_hidden_size=512, num_ff_layers=4, predict_steps=6, dropout=dropout, use_batchnorm=True):
        """
        A neural network model to predict the next words in a sequence.

        Args:
        - vocab_size: Size of the vocabulary.
        - embed_size: Dimension of the embedding layer.
        - hidden_size: Dimension of the LSTM's hidden state.
        - ff_hidden_size: Dimension of the feed-forward layers.
        - num_ff_layers: Number of feed-forward layers.
        - predict_steps: Number of words to predict simultaneously.
        - dropout: Dropout rate for regularization.
        - use_batchnorm: Whether to use BatchNorm in feed-forward layers.
        """
        super(NextWordPredictor, self).__init__()
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.embedding_dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers=2, dropout=dropout, batch_first=True)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.predict_steps = predict_steps

        # Construct feed-forward layers
        layers = []
        for _ in range(num_ff_layers):
            layers.append(nn.Linear(ff_hidden_size, ff_hidden_size))
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(ff_hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_size, ff_hidden_size),
            nn.ReLU(),
            *layers
        )

        self.final = nn.Linear(ff_hidden_size + hidden_size, vocab_size * predict_steps)

        # Initialize weights
        self._initialize_weights()

    def forward(self, x, lengths):
        """
        Forward pass of the model.

        Args:
        - x: Input tensor (batch_size, seq_length).
        - lengths: Lengths of each sequence in the batch (batch_size).
        
        Returns:
        - Output tensor (batch_size, predict_steps, vocab_size).
        """
        x = self.embedding(x)
        x = self.embedding_dropout(x)

        # Pack padded sequence for LSTM
        packed_input = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (hidden, _) = self.lstm(packed_input)

        # Apply LayerNorm to the final LSTM hidden state
        hidden = self.layer_norm(hidden[-1])

        # Feed-forward layers
        intermediate = self.feed_forward(hidden)

        # Final output layer
        output = self.final(torch.cat((hidden, intermediate), dim=1))

        # Reshape to (batch_size, predict_steps, vocab_size)
        return output.view(-1, self.predict_steps, self.vocab_size)

    def _initialize_weights(self):
        """
        Custom weight initialization for improved convergence.
        """
        for name, param in self.named_parameters():
            if 'weight' in name and param.dim() > 1:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)
