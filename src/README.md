# 📦 Source Code

Core implementation of the NSCLC Multi-Organ Segmentation project.

## 📁 Structure

```
src/
├── config/          # Configuration files
│   ├── params.yaml      # Training parameters
│   ├── paths.yaml       # Data paths
│   └── environment.yaml # Conda environment
│
├── data/            # Data loading
│   └── dataset.py       # PyTorch Dataset class
│
├── models/          # Neural network architectures
│   └── unet.py          # U-Net implementation
│
├── preprocessing/   # Data preprocessing
│   ├── convert_dicom_to_nifti.py
│   ├── extract_mask_from_rtstruct.py
│   ├── normalize_data.py
│   └── split_dataset.py
│
├── training/        # Training utilities
│   ├── trainer.py       # Training loop
│   ├── evaluate.py      # Metrics calculation
│   └── visualize.py     # Result visualization
│
└── visualization/   # Visualization tools
    └── visualize.py
```

## 🔧 Key Components

### U-Net Model (`models/unet.py`)
- Encoder-decoder architecture
- Skip connections
- Multi-class output

### Dataset (`data/dataset.py`)
- Lazy loading for memory efficiency
- Data augmentation
- Normalized HU values

### Trainer (`training/trainer.py`)
- Training loop with validation
- Checkpoint management
- Early stopping

## 📝 Usage

```python
from src.models.unet import UNet
from src.data.dataset import NSCLCDataset
from src.training.trainer import Trainer

# Initialize
model = UNet(in_channels=1, out_channels=8)
dataset = NSCLCDataset(data_dir='DATA/processed')
trainer = Trainer(model, dataset)

# Train
trainer.train(epochs=50)
```
