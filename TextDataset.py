import torch
from torch.utils.data import Dataset

class TransformerTextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=128, predict_steps=1, for_encoder_decoder=False):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.predict_steps = predict_steps
        self.pad_token_id = self.tokenizer.tokenizer.pad_token_id
        self.for_encoder_decoder = for_encoder_decoder
        self.inputs = self.build_inputs(texts)

    def build_inputs(self, texts):
        input_pairs = []
        for text in texts:
            tokenized = self.tokenizer.encode(
                text,
                truncation=False,
                padding=False
            )
            token_ids = tokenized["input_ids"]
            for i in range(1, len(token_ids) - self.predict_steps + 1):
                input_ids = token_ids[:i]
                label_ids = token_ids[i:i + self.predict_steps]
                input_pairs.append((input_ids, label_ids))
        return input_pairs

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        input_ids, label_ids = self.inputs[idx]

        if self.for_encoder_decoder:
            # src: the prefix (could be the same as tgt for LM)
            src_text = self.tokenizer.decode(input_ids)
            src_encoded = self.tokenizer.encode(
                src_text, truncation=True, padding="max_length", max_length=self.max_length
            )
            src_input_ids = torch.tensor(src_encoded["input_ids"], dtype=torch.long)
            src_attention_mask = torch.tensor(src_encoded["attention_mask"], dtype=torch.long)

            # tgt: the same as src, but shifted right (prepend BOS if you have one, else use pad_token)
            tgt_input_ids = [self.pad_token_id] + input_ids[:-1]
            tgt_text = self.tokenizer.decode(tgt_input_ids)
            tgt_encoded = self.tokenizer.encode(
                tgt_text, truncation=True, padding="max_length", max_length=self.max_length
            )
            tgt_input_ids = torch.tensor(tgt_encoded["input_ids"], dtype=torch.long)
            tgt_attention_mask = torch.tensor(tgt_encoded["attention_mask"], dtype=torch.long)

            # labels: next token(s)
            label_ids_padded = label_ids + [self.pad_token_id] * (self.predict_steps - len(label_ids))
            label_ids_padded = label_ids_padded[:self.predict_steps]
            labels = torch.tensor(label_ids_padded, dtype=torch.long)

            return {
                "src_input_ids": src_input_ids,
                "src_attention_mask": src_attention_mask,
                "tgt_input_ids": tgt_input_ids,
                "tgt_attention_mask": tgt_attention_mask,
                "labels": labels,
            }

        else:
            text = self.tokenizer.decode(input_ids)
            encoded_input = self.tokenizer.encode(
                text, truncation=True, padding="max_length", max_length=self.max_length
            )
            label_ids = label_ids + [self.pad_token_id] * (self.predict_steps - len(label_ids))
            label_ids = label_ids[:self.predict_steps]
            item = {
                "input_ids": torch.tensor(encoded_input["input_ids"], dtype=torch.long),
                "attention_mask": torch.tensor(encoded_input["attention_mask"], dtype=torch.long),
                "labels": torch.tensor(label_ids, dtype=torch.long)
            }
            return item

def transformer_collate_fn(batch):
    # Detect which keys exist in the batch
    if "src_input_ids" in batch[0]:
        return {
            "src_input_ids": torch.stack([item["src_input_ids"] for item in batch]),
            "src_attention_mask": torch.stack([item["src_attention_mask"] for item in batch]),
            "tgt_input_ids": torch.stack([item["tgt_input_ids"] for item in batch]),
            "tgt_attention_mask": torch.stack([item["tgt_attention_mask"] for item in batch]),
            "labels": torch.stack([item["labels"] for item in batch]),
        }
    else:
        # old BERT-style batch
        return {
            "input_ids": torch.stack([item["input_ids"] for item in batch]),
            "attention_mask": torch.stack([item["attention_mask"] for item in batch]),
            "labels": torch.stack([item["labels"] for item in batch])
        }