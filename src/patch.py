import torch
import torch.nn as nn
import torch.nn.functional as F

class Patching(nn.Module):
    def __init__(self, patch_len: int, stride: int):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride

    def forward(self, x):
        # Shape: (B, M, L)
        # B = Batch size, M = Num features, L = Seq len

        # Add padding if necessary to ensure we can create patches that cover the entire sequence.
        x = F.pad(x, (0, self.stride), mode='replicate')
        
        # Create patches using unfold.
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        
        # Reshape to (B*M, num_patches, patch_len)
        B, M, N, P = x.shape
        x = x.reshape(B * M, N, P)
        
        return x