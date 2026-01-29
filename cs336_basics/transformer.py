import torch
import torch.nn as nn
from einops import einsum

class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, device=None, dtype=None):
        super().__init__()
        self.weights = nn.Parameter(torch.empty(
            (out_features, in_features),
            device=device,
            dtype=dtype
        ))
        nn.init.trunc_normal_(self.weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor: # x is row vector
        return x @ self.weights.T


class Embedding(nn.Module):
    def __init__(self, num_embeddings: int , embedding_dim: int, device=None, dtype=None):
        """
        Args:
            num_embeddings: size of the vocabulary 词表大小
            embedding_dim: Dimension of the embedding vectors, i.e., d_model 词嵌入向量维度 
            device: torch.device, Device to store the parameters on
            dtype: torch.dtype, Data type of the parameters
        """
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.embedding_matrix = nn.Parameter(torch.empty(
            (num_embeddings, embedding_dim),
            device=device,
            dtype=dtype
        ))
        nn.init.trunc_normal_(self.embedding_matrix)
    
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        # Lookup for the given token IDs
        return self.embedding_matrix[token_ids]


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        """
        Args:
            d_model: Hidden dimension of the model
            eps: Epsilon value of the model
        """
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Process an input tensor of shape (batch_size, sequence_length, d_model)
        # and return a tensor of the same shape
        in_dtype = x.dtype
        x = x.to(torch.float32)
        rms = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps)
        x_norm = x / rms
        result = torch.einsum("...i,i->...i", x_norm, self.weight)

        return result.to(in_dtype)