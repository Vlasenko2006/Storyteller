import torch
import torch.nn as nn

class MiniBertForNextWordPrediction(nn.Module):
    def __init__(self, vocab_size, embed_size=128, hidden_size=128, num_layers=2, num_heads=2, dropout=0.1, max_length=128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.pos_embedding = nn.Embedding(max_length, embed_size)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_size,
                                                   nhead=num_heads,
                                                   dim_feedforward=hidden_size,
                                                   dropout=dropout, 
                                                   batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(embed_size, vocab_size)



    def forward(self, input_ids, attention_mask=None):
        # input_ids: [batch, seq_len]
        seq_length = input_ids.size(1)
        pos_ids = torch.arange(seq_length, device=input_ids.device).unsqueeze(0).expand_as(input_ids)
        x = self.embedding(input_ids) + self.pos_embedding(pos_ids)
        if attention_mask is not None:
            # Convert mask: [batch, seq_len] -> [batch, 1, 1, seq_len]
            attn_mask = (attention_mask == 0).unsqueeze(1).unsqueeze(2)
            attn_mask = attn_mask.expand(-1, 1, seq_length, -1)  # [batch, 1, seq_len, seq_len]
            x = self.encoder(x, src_key_padding_mask=(attention_mask == 0))
        else:
            x = self.encoder(x)
        # Use last token hidden state for next-word prediction
        last_hidden = x[:, -1, :]  # [batch, embed_size]
        logits = self.fc(last_hidden)
        return logits