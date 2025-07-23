import torch
import torch.nn as nn

class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes, smoothing=0.1):
        super(LabelSmoothingLoss, self).__init__()
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing
        self.classes = classes

    def forward(self, pred, target):
        log_probs = torch.log_softmax(pred, dim=-1)
        true_dist = torch.zeros_like(log_probs).to(target.device)
        true_dist.fill_(self.smoothing / (self.classes - 1))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * log_probs, dim=-1))
