import pandas as pd
import torch
from torch.utils.data import Dataset


class ETTh1(Dataset):
    def __init__(self, file_path, lookback, horizon, split='train', train_ratio=0.7, val_ratio=0.1):
        """
        Args:
            file_path: path to CSV file
            lookback: number of past timesteps to use as input
            horizon: number of future timesteps to predict
            split: 'train', 'val', or 'test'
            train_ratio: proportion of data for training (default 0.7)
            val_ratio: proportion of data for validation (default 0.1)
            # test_ratio is implicitly 1 - train_ratio - val_ratio
        """
        self.file_path = file_path
        df = pd.read_csv(file_path, index_col=0)
        
        # Sequential split based on time
        n = len(df)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))
        
        if split == 'train':
            self.data = df.iloc[:train_end]
        elif split == 'val':
            self.data = df.iloc[train_end:val_end]
        elif split == 'test':
            self.data = df.iloc[val_end:]
        else:
            raise ValueError(f"split must be 'train', 'val', or 'test', got {split}")
        
        self.lookback = lookback
        self.horizon = horizon
        self.split = split

    def __len__(self):
        return len(self.data) - self.lookback - self.horizon + 1
    
    def __getitem__(self, idx):
        # Get a window of consecutive timesteps
        # Input: rows from idx to idx+lookback
        # Target: rows from idx+lookback to idx+lookback+horizon
        input_seq = self.data.iloc[idx:idx + self.lookback].values.astype(float)
        target_seq = self.data.iloc[idx + self.lookback:idx + self.lookback + self.horizon].values.astype(float)
        
        # Convert to PyTorch tensors
        # Shape: (lookback, num_features) and (horizon, num_features)
        input_tensor = torch.tensor(input_seq, dtype=torch.float32).T  # Transpose to (num_features, lookback)
        target_tensor = torch.tensor(target_seq, dtype=torch.float32).T

        # Normalize the input and target tensors
        mean = input_tensor.mean(dim=1, keepdim=True)
        std = input_tensor.std(dim=1, keepdim=True) + 1e-6  # Add small value to avoid division by zero
        input_tensor = (input_tensor - mean) / std
        target_tensor = (target_tensor - mean) / std
        
        return {
            'input': input_tensor,  # Shape: (lookback, num_features)
            'target': target_tensor,  # Shape: (horizon, num_features)
            'mean': mean.squeeze(),  # Shape: (num_features,)
            'std': std.squeeze()  # Shape: (num_features,)
        }

