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

    
class MiniTransformerWithEncoderDecoder(nn.Module):
    def __init__(self, vocab_size, embed_size=128, hidden_size=128, num_layers=2, num_heads=2, dropout=0.1, max_length=128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.pos_embedding = nn.Embedding(max_length, embed_size)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_size,
            nhead=num_heads,
            dim_feedforward=hidden_size,
            dropout=dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_size,
            nhead=num_heads,
            dim_feedforward=hidden_size,
            dropout=dropout,
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        self.fc = nn.Linear(embed_size, vocab_size)

    def forward(self, src_input_ids, tgt_input_ids, src_attention_mask=None, tgt_attention_mask=None):
        # src_input_ids: [batch, src_seq_len]
        # tgt_input_ids: [batch, tgt_seq_len]
        src_seq_len = src_input_ids.size(1)
        tgt_seq_len = tgt_input_ids.size(1)

        src_pos_ids = torch.arange(src_seq_len, device=src_input_ids.device).unsqueeze(0).expand_as(src_input_ids)
        tgt_pos_ids = torch.arange(tgt_seq_len, device=tgt_input_ids.device).unsqueeze(0).expand_as(tgt_input_ids)

        src_x = self.embedding(src_input_ids) + self.pos_embedding(src_pos_ids)
        tgt_x = self.embedding(tgt_input_ids) + self.pos_embedding(tgt_pos_ids)

        # Encoder: [batch, src_seq_len, embed_size]
        memory = self.encoder(src_x, src_key_padding_mask=(src_attention_mask == 0) if src_attention_mask is not None else None)

        # Decoder: [batch, tgt_seq_len, embed_size]
        output = self.decoder(
            tgt_x, 
            memory, 
            tgt_key_padding_mask=(tgt_attention_mask == 0) if tgt_attention_mask is not None else None,
            memory_key_padding_mask=(src_attention_mask == 0) if src_attention_mask is not None else None
        )

        # Output head: [batch, tgt_seq_len, vocab_size]
        logits = self.fc(output)
        return logits 

