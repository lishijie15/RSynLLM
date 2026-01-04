import torch
import torch.nn as nn
from torch import Tensor

class RiskLoss(nn.Module):
    def __init__(self, under_penalties=[3, 6, 2], over_penalties=[1, 1, 1], lambda_weights=[0.6, 0.1, 0.3], reduction='mean'):
        super().__init__()
        self.reduction = reduction
        self.register_buffer("under_penalties", torch.tensor(under_penalties, dtype=torch.float32))
        self.register_buffer("over_penalties", torch.tensor(over_penalties, dtype=torch.float32))
        self.register_buffer("lambda_weights", torch.tensor(lambda_weights, dtype=torch.float32))
        self.cvar_alpha = 0.95   # CVaR
        self.cvar_weight = 0.2

    def compute_cvar(self, losses):
        sorted_losses, _ = torch.sort(losses, descending=True)
        n = len(sorted_losses)
        k = int(n * (1 - self.cvar_alpha))

        if k == 0:
            return torch.max(losses)
        else:
            return torch.mean(sorted_losses[:k])

    def forward(self, input, target):
        error = input - target
        loss_matrix = torch.where(
            error < 0,
            self.under_penalties[None, None, None, :] * torch.abs(error),
            self.over_penalties[None, None, None, :] * torch.abs(error)
        )
        weighted_loss = loss_matrix * self.lambda_weights[None, None, None, :]

        if self.reduction == 'none':
            base_loss = weighted_loss

        elif self.reduction == 'mean':
            base_loss = torch.sum(weighted_loss) / input.numel()

        elif self.reduction == 'sum':
            base_loss = torch.sum(weighted_loss)
        else:
            raise ValueError(f"Invalid: {self.reduction}")

        batch_loss = weighted_loss.sum(dim=(1, 2, 3))
        cvar_loss = self.compute_cvar(batch_loss)
        final_loss = base_loss + self.cvar_weight * cvar_loss
        return final_loss