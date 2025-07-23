import torch
from translate_output_to_words import translate_output_to_words





def predict_sequence(seed_text, model, tokenizer, num_words=100, device="cpu"):
    model.eval()
    with torch.no_grad():
        for _ in range(num_words):
            token_list = tokenizer.texts_to_sequences([seed_text])[0]
            input_tensor = torch.tensor([token_list], dtype=torch.long).to(device)  # Move to device
            lengths = torch.tensor([len(token_list)]).to(device)  # Move to device
            
            # Predict next tokens
            output = model(input_tensor, lengths)  # Shape: (1, predict_steps, vocab_size)
            
            # Convert output to words
            predicted_sequences = translate_output_to_words(output, tokenizer)  # List of predicted sequences
            
            # Use the first batch's predictions (batch size = 1 during inference)
            predicted_words = predicted_sequences[0]
            
            # Append predicted words to the seed text
            seed_text += " " + " ".join(predicted_words)

            
            if "NaN" in predicted_words:  # Stop if unknown word is predicted
                break
    print("Predicted TEXT:", seed_text)
    return seed_text
