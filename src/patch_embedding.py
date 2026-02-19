import torch.nn as nn
import torch

class PatchEmbedding(nn.Module):
    def __init__(self, patch_dim: int, embed_dim: int, num_patches: int):
        super().__init__()
        self.linear = nn.Linear(patch_dim, embed_dim)

        # Learnable positional encoding for each patch
        self.positional_encoding = nn.Parameter(torch.randn(1, num_patches, embed_dim))  # Learnable positional encoding

    def forward(self, x):
        # x shape: (B*M, num_patches, patch_len)
        # output shape: (B*M, num_patches, embed_dim)
        return self.linear(x) + self.positional_encoding