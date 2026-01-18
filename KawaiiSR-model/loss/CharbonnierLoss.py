import torch
import torch.nn as nn
import torch.nn.functional as F

class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (a smooth variant of L1 Loss)."""
    def __init__(self, eps: float = 1e-6):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, sr_img: torch.Tensor, hr_img: torch.Tensor) -> torch.Tensor:
        """
        Args:
            sr_img (torch.Tensor): The generated super-resolution image.
            hr_img (torch.Tensor): The ground-truth high-resolution image.
        Returns:
            torch.Tensor: The computed Charbonnier loss.
        """
        diff = sr_img - hr_img
        loss = torch.mean(torch.sqrt(diff * diff + self.eps))
        return loss