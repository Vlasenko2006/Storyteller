#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 09:33:08 2025

@author: andrey
"""

import torch
import torch.nn as nn

class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes, smoothing=0.1, dim=-1, ignore_index=-100):
        """
        classes: int, number of classes
        smoothing: float, smoothing factor (0.0 = no smoothing)
        dim: int, dimension over which to apply log_softmax
        ignore_index: int, index to ignore (e.g., padding)
        """
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.classes = classes
        self.dim = dim
        self.ignore_index = ignore_index

    def forward(self, pred, target):
        """
        pred: logits tensor, shape (batch_size, num_classes)
        target: target indices, shape (batch_size,)
        """
        # Mask out ignored targets
        mask = (target != self.ignore_index)
        if not mask.any():
            return torch.tensor(0.0, dtype=pred.dtype, device=pred.device, requires_grad=True)
        pred = pred[mask]
        target = target[mask]

        log_probs = torch.log_softmax(pred, dim=self.dim)
        with torch.no_grad():
            true_dist = torch.zeros_like(log_probs)
            true_dist.fill_(self.smoothing / (self.classes - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * log_probs, dim=self.dim))


def multi_word_loss(predictions, targets, criterion, smoothing=0.0, vocab_size=None, ignore_index=-100):
    """
    Computes the loss for multi-word predictions (and next-token prediction).
    Supports predictions of shape (batch, predict_steps, vocab) or (batch, vocab).
    If smoothing > 0, uses label smoothing loss.
    criterion: nn.CrossEntropyLoss or None (ignored if smoothing > 0)
    vocab_size: required if smoothing > 0
    ignore_index: index to ignore (e.g., padding)
    """
    # If smoothing is enabled, override criterion
    if smoothing > 0.0:
        assert vocab_size is not None, "vocab_size must be provided if smoothing > 0"
        criterion = LabelSmoothingLoss(classes=vocab_size, smoothing=smoothing, ignore_index=ignore_index)

    if predictions.dim() == 3:
        # predictions: (batch, predict_steps, vocab)
        batch_size, predict_steps, vocab_size_ = predictions.size()
        _, actual_steps = targets.size()
        steps_to_use = min(predict_steps, actual_steps)
        predictions = predictions[:, :steps_to_use, :]
        targets = targets[:, :steps_to_use]
        total_loss = 0
        valid_steps = 0
        for step in range(steps_to_use):
            step_loss = criterion(predictions[:, step, :], targets[:, step])
            # don't count totally ignored steps in the average
            if step_loss.requires_grad or step_loss.item() > 0:
                total_loss += step_loss
                valid_steps += 1
        return total_loss / max(valid_steps, 1)
    elif predictions.dim() == 2:
        # predictions: (batch, vocab)
        # targets: (batch,) or (batch, 1)
        if targets.dim() == 2 and targets.size(1) == 1:
            targets = targets.squeeze(1)
        return criterion(predictions, targets)
    else:
        raise ValueError(f"predictions must be 2D or 3D, got shape {predictions.shape}")
