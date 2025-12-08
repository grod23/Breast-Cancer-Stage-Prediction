import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    # https://medium.com/data-scientists-diary/implementing-focal-loss-in-pytorch-for-class-imbalance-24d8aa3b59d9
    def __init__(self, alpha=1, gamma=2.0, pos_weight=None, ignore_index=None, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.pos_weight = pos_weight
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(self, logits, targets):
        # Create mask for ignored indices
        if self.ignore_index:
            print(f'Ignoring Index: {self.ignore_index}')
            mask = (targets != self.ignore_index)
            targets = targets[mask]
            logits = logits[mask]
        # If all targets are -1
        if targets.numel() == 0:
            return torch.tensor(0.0, dtype=torch.float32, device=logits.device)
        # Convert logits to log probabilities
        log_prob = F.log_softmax(logits, dim=-1)
        prob = torch.exp(log_prob)  # Calculate probabilities from log probabilities
        # Gather the probabilities corresponding to the correct classes
        targets_one_hot = F.one_hot(targets, num_classes=logits.shape[-1])
        pt = torch.sum(prob * targets_one_hot, dim=-1)
        # Apply focal adjustment
        focal_loss = -self.alpha * (1 - pt) ** self.gamma * torch.sum(log_prob * targets_one_hot, dim=-1)
        return focal_loss.mean()
