# PatchTST - Patch Time Series Transformer

Implementation of **PatchTST** (Patch Time Series Transformer), a Transformer-based architecture for multivariate time series forecasting.

## 📋 Description

PatchTST is an innovative model that applies **patching** technique to time series, segmenting temporal sequences into sub-segments (patches) before processing them through a Transformer encoder. This approach:

- Reduces computational complexity
- Improves the ability to capture both local and global patterns
- Preserves semantic information in time series

## 🏗️ Architecture

The model consists of four main components:

1. **Patching**: Divides the time series into fixed-length overlapping patches
2. **Patch Embedding**: Projects each patch into a high-dimensional embedding space with positional encoding
3. **Transformer Encoder**: Processes embedded patches to capture temporal dependencies
4. **Prediction Head**: Final linear layer that produces future predictions

### Model Pipeline

```
Input: (B, M, L) → Patching → (B*M, N, P) → Embedding → (B*M, N, D) 
       → Transformer → (B*M, N, D) → Flatten → (B*M, N*D) 
       → Linear → (B*M, H) → Reshape → (B, M, H)
```

Where:
- `B` = Batch size
- `M` = Number of features (variables)
- `L` = Input sequence length (lookback window)
- `N` = Number of patches
- `P` = Dimension of each patch
- `D` = Embedding dimension
- `H` = Prediction horizon

## 📁 Project Structure

```
PatchTST/
├── src/
│   ├── patch.py              # Time series patching module
│   ├── patch_embedding.py    # Patch embedding with positional encoding
│   ├── patch_tst.py          # Main PatchTST model
│   └── dataset.py            # ETTh1 dataset with preprocessing
├── main.ipynb                # Training and evaluation notebook
├── ETTh1.csv                 # Electricity Transformer Temperature dataset (hourly)
└── README.md                 # This file
```

## 🔧 Components

### 1. Patching (`src/patch.py`)

Divides time series into patches using `unfold` with:
- **Patch length**: Length of each patch
- **Stride**: Distance between consecutive patches (overlapping patches when `stride < patch_len`)

### 2. Patch Embedding (`src/patch_embedding.py`)

- Linear projection from patch dimension to embedding dimension
- Learnable positional encoding to preserve temporal information

### 3. PatchTST Model (`src/patch_tst.py`)

Complete model with:
- 3 Transformer encoder layers
- 8 attention heads
- Linear prediction head for multivariate output

### 4. Dataset (`src/dataset.py`)

`ETTh1` dataset (Electricity Transformer Temperature - Hourly) with:
- Temporal split (train/val/test): 70%/10%/20%
- Per-sample normalization (mean and std per feature)
- Sliding window to generate input-target pairs

## 🚀 Usage

### Hyperparameter Configuration

```python
LOOKBACK = 336      # L: Lookback window (14 days at hourly frequency)
HORIZON = 96        # T: Prediction horizon (4 days)
PATCH_DIM = 16      # P: Dimension of each patch
STRIDE = 8          # S: Stride between consecutive patches
EMBED_DIM = 16      # D: Embedding dimensionality
```

### Training

```python
from src.patch_tst import PatchTST
from src.dataset import ETTh1
from torch.utils.data import DataLoader

# Dataset preparation
train_dataset = ETTh1(file_path='./ETTh1.csv', split='train', 
                      lookback=LOOKBACK, horizon=HORIZON)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Model initialization
model = PatchTST(
    num_patches=(LOOKBACK - PATCH_DIM) // STRIDE + 2,
    patch_dim=PATCH_DIM,
    embed_dim=EMBED_DIM,
    horizon=HORIZON,
    stride=STRIDE
)

# Training loop (see main.ipynb for complete implementation)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = torch.nn.functional.mse_loss
```

## 📊 Dataset

**ETTh1** (Electricity Transformer Temperature - Hourly):
- Hourly temperature and electrical load data
- Multiple features measured from electrical transformers
- Ideal for long-term multivariate forecasting

## 🔬 Technical Features

- **Framework**: PyTorch
- **Loss Function**: Mean Squared Error (MSE)  
- **Optimizer**: Adam (lr=1e-3)
- **Normalization**: Instance normalization (per sample)
- **Device Support**: CPU and GPU (CUDA)

## 📈 PatchTST Advantages

1. **Efficiency**: Reduction in the number of tokens to process
2. **Scalability**: Efficient handling of long time series sequences
3. **Generalization**: Better generalization capability through local semantic preservation
4. **Flexibility**: Easily adaptable to different time series lengths and forecasting horizons

## 📝 Notes

- Number of patches is calculated as: `N = (L - P) / S + 2` (with padding)
- Normalization is applied per feature using mean and standard deviation of the input window
- The model supports multivariate predictions while maintaining dependencies between features

## 📚 References

This model is based on the paper:
> "A Time Series is Worth 64 Words: Long-term Forecasting with Transformers"

---

**Date**: February 2026
