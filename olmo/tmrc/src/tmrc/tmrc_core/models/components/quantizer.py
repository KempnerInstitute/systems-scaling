import torch
import torch.nn as nn
import torch.nn.functional as F

class VectorQuantizer(nn.Module):
    """
    Handles quantizing a continuous representation into a discrete one.
    Follows the implementation described in the VQ-VAE paper (https://arxiv.org/abs/1711.00937).
    """
    def __init__(self, num_tokens: int, embedding_dim: int, beta: float):
        """
        Args:
            num_tokens: Number of discrete tokens in the discrete representation space.
            embedding_dim: Dimension of the continuous representation space.
            beta: Weight of the MSE loss term with input gradients. Should be set between 0.1 and 2.0 during training.
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_tokens = num_tokens
        self.embeddings = nn.Embedding(num_tokens, embedding_dim)
        self.embeddings.weight.data.uniform_(-1/num_tokens, 1/num_tokens)
        self.beta = beta
    
    def forward(self, z: torch.Tensor):
        nearest = self.quantize(z)
        z_q = self.embeddings(nearest)
        loss = F.mse_loss(z.detach(), z_q) + self.beta*F.mse_loss(z_q.detach(), z)
        z_q = z + (z_q - z).detach()
        return z_q, loss, nearest
    
    def quantize(self, z: torch.Tensor) -> int:
        """
        Returns index of the closest embeeding to the input.
        """
        distance = torch.cdist(z, self.embeddings.weight)
        return torch.argmin(distance, dim=1)
