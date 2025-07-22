#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 09:33:08 2025

@author: andrey
"""

def multi_word_loss(predictions, targets, criterion):
    """
    Computes the loss for multi-word predictions (and next-token prediction).
    Supports predictions of shape (batch, predict_steps, vocab) or (batch, vocab).
    """
    if predictions.dim() == 3:
        # predictions: (batch, predict_steps, vocab)
        batch_size, predict_steps, vocab_size = predictions.size()
        _, actual_steps = targets.size()
        steps_to_use = min(predict_steps, actual_steps)
        predictions = predictions[:, :steps_to_use, :]
        targets = targets[:, :steps_to_use]
        loss = 0
        for step in range(steps_to_use):
            loss += criterion(predictions[:, step, :], targets[:, step])
        return loss / steps_to_use
    elif predictions.dim() == 2:
        # predictions: (batch, vocab)
        # targets: (batch,) or (batch, 1)
        if targets.dim() == 2 and targets.size(1) == 1:
            targets = targets.squeeze(1)
        return criterion(predictions, targets)
    else:
        raise ValueError(f"predictions must be 2D or 3D, got shape {predictions.shape}")
