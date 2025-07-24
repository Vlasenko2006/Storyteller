import torch
from tqdm import tqdm
from multi_word_loss import multi_word_loss
from final_text import final_text

def my_trainer(
    nepochs,
    path,
    alternate_costs, 
    criterion_ce, 
    criterion_ls, 
    optimizer_ce, 
    optimizer_ls, 
    model,
    train_loader,
    tokenizer,
    device='cpu', 
    predicted_steps=1,
    validate_after_nepochs=1,
    seeders=["NaN"],
    start_epoch=0,
    grad_accum_steps=1,
    model_type="BERT"  # NEW: could be "BERT" or "BART"
):
    text = ["NaN"]
    for epoch in range(start_epoch, nepochs):

        if alternate_costs:
            if epoch % 2 == 1: 
                criterion = criterion_ce 
                optimizer = optimizer_ce
            else:
                criterion = criterion_ls
                optimizer = optimizer_ls
        else:
            criterion = criterion_ce
            optimizer = optimizer_ce

        model.train()
        epoch_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}")

        optimizer.zero_grad()
        for step, batch in enumerate(progress_bar):
            if model_type == "BART":  # Encoder-decoder
                src_input_ids = batch["src_input_ids"].to(device)
                src_attention_mask = batch["src_attention_mask"].to(device)
                tgt_input_ids = batch["tgt_input_ids"].to(device)
                tgt_attention_mask = batch["tgt_attention_mask"].to(device)
                labels = batch["labels"].to(device)

                # Forward pass
                logits = model(
                    src_input_ids, tgt_input_ids,
                    src_attention_mask=src_attention_mask,
                    tgt_attention_mask=tgt_attention_mask
                )

                # For CrossEntropy, flatten target and logits
                loss = multi_word_loss(
                    logits.view(-1, logits.size(-1)),
                    labels.view(-1),
                    criterion
                )
            else:  # Default: BERT-like
                X = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                y = batch["labels"].to(device)

                output = model(X, attention_mask=attention_mask)
                loss = multi_word_loss(output, y, criterion)

            loss = loss / grad_accum_steps
            loss.backward()

            if (step + 1) % grad_accum_steps == 0 or (step + 1) == len(train_loader):
                optimizer.step()
                optimizer.zero_grad()
            
            epoch_loss += loss.item() * grad_accum_steps
            progress_bar.set_postfix(loss=loss.item() * grad_accum_steps)

        avg_loss = epoch_loss / len(train_loader)
        print(f"NN2 Epoch {epoch + 1}, Average Loss: {avg_loss:.4f}")

        torch.save(
            {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
            },
            f"{path}/story_telling-lr-{predicted_steps}_ep_{epoch}.pth"
        )

        if (epoch + 1) % validate_after_nepochs == 0:
            for index, seeder in enumerate(seeders, start=1):
                text = final_text(
                    seeder,
                    model, 
                    tokenizer,
                    num_words=100,
                    device=device
                )
                print(f"{index}: {text[0]}")
                print("")
    return text