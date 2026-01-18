import torch
import torch.nn as nn
import torch.nn.functional as F

class HingeGeneratorLoss(nn.Module):
    """
    Hinge loss for the generator based on the 'least squares' variation,
    where the generator tries to maximize the discriminator's output for fake images.
    """
    def __init__(self):
        super(HingeGeneratorLoss, self).__init__()

    def forward(self, fake_logits: torch.Tensor) -> torch.Tensor:
        """
        Args:
            fake_logits (torch.Tensor): The raw output (logits) of the discriminator for fake images.
        
        Returns:
            torch.Tensor: The computed Hinge loss for the generator.
        """
        # The generator wants to maximize the discriminator's score for fake images.
        # We minimize the negative of this score.
        loss = -torch.mean(fake_logits)
        return loss
    
class HingeDiscriminatorLoss(nn.Module):
    """
    Hinge loss for the discriminator.
    It encourages the discriminator to output values > 1 for real images
    and values < -1 for fake images.
    """
    def __init__(self):
        super(HingeDiscriminatorLoss, self).__init__()

    def forward(self, real_logits: torch.Tensor, fake_logits: torch.Tensor) -> torch.Tensor:
        """
        Args:
            real_logits (torch.Tensor): The raw output (logits) of the discriminator for real images.
            fake_logits (torch.Tensor): The raw output (logits) of the discriminator for fake images.

        Returns:
            torch.Tensor: The computed Hinge loss for the discriminator.
        """
        # Loss for real images: max(0, 1 - D(x_real))
        # This penalizes the discriminator if its output for real images is less than 1.
        loss_real = torch.mean(torch.relu(1.0 - real_logits))

        # Loss for fake images: max(0, 1 + D(x_fake))
        # This penalizes the discriminator if its output for fake images is greater than -1.
        loss_fake = torch.mean(torch.relu(1.0 + fake_logits))
        
        # The total loss is the sum of the two.
        return loss_real + loss_fake