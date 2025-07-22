#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 31 09:42:16 2025

@author: andrey
"""

import torch
from tqdm import tqdm  # For progress bar
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
    grad_accum_steps=1  # <--- NEW: How many batches to accumulate gradients over
):
    # Training loop
    text = ["NaN"]
    for epoch in range(start_epoch, nepochs):

        if alternate_costs:
            if epoch % 2 == 1: 
                criterion = criterion_ce 
                optimizer = optimizer_ce
            else:
                criterion = criterion_ls  # Alternate losses
                optimizer = optimizer_ls
        else:
            criterion = criterion_ce  # or whichever is your default
            optimizer = optimizer_ce  # or whichever is your default

        model.train()
        epoch_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}")

        optimizer.zero_grad()
        for step, batch in enumerate(progress_bar):
            X = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            y = batch["labels"].to(device)

            output = model(X, attention_mask=attention_mask)
            loss = multi_word_loss(output, y, criterion)
            # Scale loss for gradient accumulation
            loss = loss / grad_accum_steps
            loss.backward()

            if (step + 1) % grad_accum_steps == 0 or (step + 1) == len(train_loader):
                optimizer.step()
                optimizer.zero_grad()
            
            # Update progress bar and epoch loss (accumulate original loss, not scaled)
            epoch_loss += loss.item() * grad_accum_steps  # unscale for reporting
            progress_bar.set_postfix(loss=loss.item() * grad_accum_steps)

        # Print epoch summary
        avg_loss = epoch_loss / len(train_loader)
        print(f"NN2 Epoch {epoch + 1}, Average Loss: {avg_loss:.4f}")

        # Save model and optimizer checkpoint
        torch.save(
            {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                # You can add more metadata if needed
            },
            f"{path}/story_telling-{predicted_steps}_ep_{epoch}.pth"
        )

        # Predict and print sequences every validate_after_nepochs epochs
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