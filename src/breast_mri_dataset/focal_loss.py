import torch
import torch.nn as nn
import torch.nn.functional as F
torch.set_printoptions(precision=4, sci_mode=False, linewidth=150)


class FocalLoss(nn.Module):
    """
    Focal Loss for multiclass classification with positive weights and ignore index.

    Parameters:
    -----------
    alpha : float or torch.Tensor
        Weighting factor in [0, 1] to balance positive/negative examples,
        or a tensor of weights for each class. Default: 1.0
    gamma : float
        Exponent of the modulating factor (1 - p_t)^gamma. Default: 2.0
    pos_weight : torch.Tensor or None
        A weight of positive examples per class. Must be a vector with length
        equal to the number of classes. Default: None
    ignore_index : int
        Specifies a target value that is ignored and does not contribute to
        the input gradient. Default: -100
    reduction : str
        Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'.
        Default: 'mean'
    """

    # https://medium.com/data-scientists-diary/implementing-focal-loss-in-pytorch-for-class-imbalance-24d8aa3b59d9
    def __init__(self, alpha=0.5, gamma=2.0, pos_weight=None, ignore_index=-1, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.pos_weight = pos_weight
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(self, logits, targets):
        # Create mask for ignored indices
        mask = (targets != self.ignore_index)
        print(f'Targets: {targets}')
        print(f'logits: {logits}')
        targets = targets[mask]
        logits = logits[mask]
        print(f'Masked Targets: {targets}')
        print(f'Masked logits: {logits}')
        # If all targets are -1
        if targets.numel() == 0:
            return torch.tensor(0.0, dtype=torch.float32, device=logits.device)
        # Convert logits to log probabilities
        log_prob = F.log_softmax(logits, dim=-1)
        prob = torch.exp(log_prob)  # Calculate probabilities from log probabilities
        # Gather the probabilities corresponding to the correct classes
        targets_one_hot = F.one_hot(targets, num_classes=logits.shape[-1])
        print(f'Prob: {prob}, TargetsOneHot: {targets_one_hot}')
        pt = torch.sum(prob * targets_one_hot, dim=-1)
        # Apply focal adjustment
        focal_loss = -self.alpha * (1 - pt) ** self.gamma * torch.sum(log_prob * targets_one_hot, dim=-1)
        return focal_loss.mean()
