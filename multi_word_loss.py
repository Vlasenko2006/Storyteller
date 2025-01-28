#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 09:33:08 2025

@author: andrey
"""


def multi_word_loss(predictions, targets, criterion):
    """
    Computes the loss for multi-word predictions.
    Args:
        predictions: Tensor of shape (batch_size, predict_steps, vocab_size)
        targets: Tensor of shape (batch_size, actual_steps)
    Returns:
        Averaged loss across all valid predicted steps.
    """
    # predictions: (batch_size, predict_steps, vocab_size)
    # targets: (batch_size, actual_steps)
    
    batch_size, predict_steps, vocab_size = predictions.size()
    _, actual_steps = targets.size()
    
    # Ensure targets are trimmed or padded to match predict_steps
    steps_to_use = min(predict_steps, actual_steps)
    predictions = predictions[:, :steps_to_use, :]  # Trim predictions
    targets = targets[:, :steps_to_use]  # Trim targets

    # Compute loss for valid steps
    loss = 0
    for step in range(steps_to_use):
        loss += criterion(predictions[:, step, :], targets[:, step])
    return loss / steps_to_use  # Average loss across valid steps

