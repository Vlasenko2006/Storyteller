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



def my_trainer(nepochs,
            path,
            alternate_costs, 
            criterion_ce, 
            criterion_ls, 
            optimizer_ce, 
            optimizer_ls, 
            model,
            train_loader,
            tokenizer,
            device = 'cpu', 
            predicted_steps = 1,
            validate_after_nepochs = 1,
            seeders= ["NaN"],
            start_epoch = 0):
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
            f"{path}/story_telling_final-{predicted_steps}_ep_{epoch}.pth"
        )

        # Predict and print sequences every 5 epochs
        if (epoch + 1) % validate_after_nepochs == 0:
            for index, seeder in enumerate(seeders, start=1):
                text = final_text(seeder,
                         model, 
                         tokenizer,
                         num_words=100,
                         device = device)
                print(f"{index}: {text[0]}")
                print("")
    return text

