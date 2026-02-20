import torch.nn as nn
from patch import Patching
from patch_embedding import PatchEmbedding

class PatchTST(nn.Module):
    def __init__(self, patch_dim, stride, embed_dim, num_patches, horizon, num_heads=4):
        super().__init__()
        self.patching = Patching(patch_dim, stride)
        self.patch_embedding = PatchEmbedding(patch_dim=patch_dim, embed_dim=embed_dim, num_patches=num_patches)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=3)

        self.flatten = nn.Flatten(start_dim=1) # Flatten the output of the transformer to (B*M, num_patches*embed_dim)
        self.linear_head = nn.Linear(num_patches * embed_dim, horizon) # Final linear layer to predict the future values
        self.num_features = None  # Will be set in forward

    def forward(self, x):
        # x shape: (B, M, L)
        B, M, L = x.shape
        
        x = self.patching(x)    # Shape: (B*M, num_patches, patch_len)
        x = self.patch_embedding(x)  # Shape: (B*M, num_patches, embed_dim)

        # Pass through transformer encoder
        x = self.transformer_encoder(x)  # Shape: (B*M, num_patches, embed_dim)

        x = self.flatten(x)  # Shape: (B*M, num_patches*embed_dim)
        x = self.linear_head(x)  # Shape: (B*M, horizon)
        
        # Reshape back to (B, M, horizon)
        x = x.view(B, M, -1)  # Shape: (B, M, horizon)
        
        return x