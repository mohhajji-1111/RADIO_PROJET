# NSCLC Multi-Organ Segmentation Dataset (Preprocessed)

## 📊 Dataset Overview

This dataset contains **158 lung cancer patients** with multi-organ segmentation masks, preprocessed and ready for deep learning training.

### Original Source
- **NSCLC-Radiomics** from The Cancer Imaging Archive (TCIA)
- Non-Small Cell Lung Cancer patients
- CT scans with radiotherapy planning structures (RTSTRUCT)

---

## 📁 Dataset Structure

```
nsclc_multiorgan_segmentation/
├── normalized_ct/              # 158 CT scans (normalized)
│   ├── LUNG1-001_ct.nii.gz    # Shape: 256×256×Z, dtype: float32
│   ├── LUNG1-002_ct.nii.gz
│   └── ...
├── normalized_masks/           # 158 multi-organ masks
│   ├── LUNG1-001_mask.nii.gz  # Shape: 256×256×Z, dtype: uint8
│   ├── LUNG1-002_mask.nii.gz
│   └── ...
├── splits/
│   ├── train.txt              # 110 patient IDs (69.6%)
│   ├── val.txt                # 23 patient IDs (14.6%)
│   └── test.txt               # 25 patient IDs (15.8%)
└── code/
    ├── dataset_multi_organ.py # PyTorch Dataset class
    ├── unet_multi_organ.py    # U-Net model
    └── train_multi_organ.py   # Training script
```

---

## 🏷️ Label Mapping

Each mask contains **8 classes** (multi-class segmentation):

| Label | Organ              | Color (visualization) | Frequency |
|-------|--------------------|-----------------------|-----------|
| 0     | Background         | Black                 | 93.2%     |
| 1     | GTV (Tumor)        | Red                   | 0.8%      |
| 2     | PTV                | Orange                | 0.0%*     |
| 3     | Right Lung         | Cyan                  | 2.1%      |
| 4     | Left Lung          | Light Blue            | 1.9%      |
| 5     | Heart              | Magenta               | 1.2%      |
| 6     | Esophagus          | Yellow                | 0.3%      |
| 7     | Spinal Cord        | Green                 | 0.5%      |

*PTV (Planning Target Volume) is absent in this dataset (0% of patients).

---

## 🔧 Preprocessing Applied

### Phase 1: DICOM → NIfTI Conversion
- Converted raw DICOM files to NIfTI format
- Applied Hounsfield Unit (HU) transformation
- Preserved spatial metadata (spacing, orientation)

### Phase 2: Multi-Organ Extraction
- Extracted 7 organ contours from RTSTRUCT files
- Fused multiple GTVs (GTV-1, GTV-2, ...) into single label
- Created multi-class masks (labels 0-7)

### Phase 3: Normalization
- **Lung Windowing**: [-1350, +150] HU
- **Z-score Normalization**: mean≈0, std≈1 per volume
- **Resize**: 256×256 pixels (from 512×512)
- **Label Preservation**: Nearest-neighbor interpolation for masks

---

## 📊 Dataset Statistics

### Volume Distribution
- **Total Patients**: 158
- **Total Slices**: 57,148 (2D slices)
- **Train**: 39,867 slices (110 patients)
- **Val**: 7,702 slices (23 patients)
- **Test**: 9,579 slices (25 patients)

### Organ Volumes (mean ± std)
- **GTV**: 36.94 ± 39.10 cm³ (157/158 patients)
- **Right Lung**: 782.08 ± 200.09 cm³ (149/158 patients)
- **Left Lung**: 680.91 ± 185.67 cm³ (150/158 patients)
- **Heart**: 267.34 ± 82.85 cm³ (44/158 patients)
- **Esophagus**: 17.74 ± 4.35 cm³ (125/158 patients)
- **Spinal Cord**: 30.81 ± 7.52 cm³ (151/158 patients)

---

## 🚀 Quick Start (Kaggle Notebook)

```python
import sys
sys.path.append('/kaggle/input/nsclc-multiorgan-segmentation/code')

from dataset_multi_organ import MultiOrganDataset
from unet_multi_organ import UNetMultiOrgan
import torch
from torch.utils.data import DataLoader

# Create dataset
data_root = '/kaggle/input/nsclc-multiorgan-segmentation'
train_dataset = MultiOrganDataset(data_root, split='train')
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

# Initialize model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = UNetMultiOrgan(in_channels=1, out_channels=8, bilinear=False).to(device)

# Training loop
for batch in train_loader:
    images = batch['image'].to(device)  # [B, 1, 256, 256]
    masks = batch['mask'].to(device)    # [B, 256, 256]
    
    outputs = model(images)  # [B, 8, 256, 256]
    # ... compute loss and backprop
```

---

## 📈 Expected Results

With U-Net training (50 epochs, ~3-4 hours on GPU):
- **Dice Score - Lungs**: 0.90-0.95 (excellent)
- **Dice Score - Heart**: 0.85-0.90 (good)
- **Dice Score - Tumor**: 0.75-0.85 (variable, depends on size)
- **Dice Score - Esophagus**: 0.70-0.80 (challenging, small organ)
- **Dice Score - Spinal Cord**: 0.70-0.80 (challenging, thin structure)

---

## 🎯 Use Cases

1. **Radiotherapy Planning**: Automatic organ segmentation for dose calculation
2. **Tumor Volume Quantification**: Tracking tumor growth over time
3. **Benchmark for Segmentation Models**: Test U-Net, SegResNet, nnU-Net
4. **Class Imbalance Research**: Highly imbalanced multi-class problem
5. **Medical Image Processing Education**: Real-world clinical dataset

---

## 📜 Citation

If you use this dataset, please cite the original NSCLC-Radiomics dataset:

```
Aerts, H. J. W. L., Wee, L., Rios Velazquez, E., Leijenaar, R. T. H., Parmar, C., 
Grossmann, P., Carvalho, S., Bussink, J., Monshouwer, R., Haibe-Kains, B., 
Rietveld, D., Hoebers, F., Rietbergen, M. M., Leemans, C. R., Dekker, A., 
Quackenbush, J., Gillies, R. J., Lambin, P. (2019). Data from NSCLC-Radiomics 
[Data set]. The Cancer Imaging Archive. https://doi.org/10.7937/K9/TCIA.2015.PF0M9REI
```

---

## 📝 License

This preprocessed dataset maintains the **CC0 1.0 Universal** license from the original NSCLC-Radiomics dataset.

---

## 🙏 Acknowledgments

- **The Cancer Imaging Archive (TCIA)** for hosting the original NSCLC-Radiomics dataset
- **MAASTRO Clinic** for data collection and annotation
- Preprocessing pipeline inspired by medical image analysis best practices

---

## 📧 Contact

For questions or issues with this preprocessed dataset, please open a discussion in the Kaggle dataset page.

**Happy Training! 🚀**
