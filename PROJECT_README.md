# NSCLC-Radiomics Lung Tumor Segmentation Project

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🎯 Project Overview

This project implements an end-to-end deep learning pipeline for **lung tumor segmentation** using the NSCLC-Radiomics dataset. The pipeline includes:

- **DICOM to NIfTI conversion** (CT scans + RTSTRUCT contours)
- **Data preprocessing and normalization** (HU windowing, resampling, augmentation)
- **U-Net model training** (PyTorch implementation)
- **Comprehensive evaluation** (Dice, IoU, Hausdorff distance)
- **Visualization and analysis** (2D slices, 3D rendering, overlay comparisons)

---

## 📁 Project Structure

```
RADIO_PROJET/
│
├── data/
│   ├── NSCLC-Radiomics/          # Raw DICOM data (CT + RTSTRUCT)
│   ├── processed/                 # Processed NIfTI files
│   │   ├── images_nifti/          # Converted CT volumes
│   │   ├── masks_nifti/           # Extracted binary masks
│   │   ├── normalized/            # Normalized data
│   │   └── splits/                # Train/val/test split information
│   └── results/                   # Training outputs
│       ├── predictions/           # Model predictions
│       ├── metrics/               # Evaluation metrics
│       ├── models/                # Saved model checkpoints
│       └── visualizations/        # Plots and figures
│
├── notebooks/
│   ├── 01_preprocessing.ipynb     # Data preprocessing pipeline
│   ├── 02_training_unet.ipynb     # Model training
│   ├── 03_evaluation.ipynb        # Model evaluation
│   └── 04_visualization.ipynb     # Results visualization
│
├── src/
│   ├── preprocessing/
│   │   ├── convert_dicom_to_nifti.py    # DICOM → NIfTI conversion
│   │   ├── extract_mask_from_rtstruct.py # RTSTRUCT → binary mask
│   │   ├── normalize_data.py             # Data normalization
│   │   ├── split_dataset.py              # Train/val/test splitting
│   │   └── utils_dicom.py                # DICOM utilities
│   │
│   ├── training/
│   │   ├── dataset.py              # PyTorch Dataset class
│   │   ├── unet_model.py           # U-Net architecture
│   │   ├── train.py                # Training loop
│   │   ├── evaluate.py             # Evaluation metrics
│   │   └── visualize.py            # Visualization utilities
│   │
│   └── config/
│       ├── paths.yaml              # Path configurations
│       ├── params.yaml             # Hyperparameters
│       └── environment.yaml        # Conda environment
│
├── scripts/
│   ├── preprocess_all.py           # Run complete preprocessing
│   ├── train_unet.py               # Train U-Net model
│   └── evaluate_unet.py            # Evaluate trained model
│
├── requirements.txt                # Python dependencies
├── README.md                       # This file
└── .gitignore                      # Git ignore rules
```

---

## 🚀 Quick Start

### 1. Environment Setup

#### Option A: Using Conda (Recommended)
```bash
# Create environment from YAML
conda env create -f src/config/environment.yaml

# Activate environment
conda activate nsclc-seg
```

#### Option B: Using pip
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Paths

Edit `src/config/paths.yaml` to match your local directory structure:

```yaml
project_root: "C:/Users/HP/Desktop/RADIO_PROJET"
raw_data:
  nsclc_radiomics: "C:/Users/HP/Desktop/RADIO_PROJET/DATA/NSCLC-Radiomics"
```

### 3. Run Preprocessing

```bash
# Run complete preprocessing pipeline
python scripts/preprocess_all.py --config src/config/params.yaml

# Or run individual steps
python scripts/preprocess_all.py --step convert    # DICOM → NIfTI
python scripts/preprocess_all.py --step extract   # Extract masks
python scripts/preprocess_all.py --step normalize # Normalize data
python scripts/preprocess_all.py --step split     # Create splits
```

### 4. Train Model

```bash
# Train U-Net model
python scripts/train_unet.py --config src/config/params.yaml

# Resume from checkpoint
python scripts/train_unet.py --config src/config/params.yaml --resume path/to/checkpoint.pth
```

### 5. Evaluate Model

```bash
# Evaluate on test set
python scripts/evaluate_unet.py --config src/config/params.yaml --checkpoint path/to/best_model.pth

# Evaluate on validation set
python scripts/evaluate_unet.py --config src/config/params.yaml --checkpoint path/to/best_model.pth --split val
```

---

## 📊 Dataset Information

### NSCLC-Radiomics Dataset

- **Source**: The Cancer Imaging Archive (TCIA)
- **Patients**: 422 non-small cell lung cancer patients
- **Modalities**: CT scans + RTSTRUCT (radiation therapy structure sets)
- **Task**: Segment lung tumors (GTV, CTV, PTV) from CT images

### Data Processing Pipeline

1. **DICOM Loading**: Read CT series and RTSTRUCT files
2. **Resampling**: Standardize voxel spacing to 1×1×1 mm³
3. **HU Windowing**: Clip Hounsfield Units to [-1000, 400]
4. **Normalization**: Z-score normalization per volume
5. **Resizing**: Resize 2D slices to 256×256 pixels
6. **Splitting**: 70% train, 15% validation, 15% test

---

## 🧠 Model Architecture

### U-Net

Classic U-Net architecture with:
- **Encoder**: 4 downsampling blocks (64 → 128 → 256 → 512 features)
- **Bottleneck**: 1024 features
- **Decoder**: 4 upsampling blocks with skip connections
- **Output**: Sigmoid activation for binary segmentation

**Total Parameters**: ~31 million

### Training Configuration

```yaml
Optimizer: Adam (lr=0.001)
Loss: Combined Dice + BCE (50/50 weight)
Batch Size: 8
Epochs: 100
Scheduler: ReduceLROnPlateau
Early Stopping: Patience 20 epochs
```

---

## 📈 Evaluation Metrics

- **Dice Coefficient**: Overlap between prediction and ground truth
- **IoU (Jaccard Index)**: Intersection over Union
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **Hausdorff Distance**: Maximum boundary distance
- **Surface Distance**: Average boundary distance

---

## 📓 Jupyter Notebooks

Interactive notebooks for exploration:

1. **01_preprocessing.ipynb**: Visualize data preprocessing steps
2. **02_training_unet.ipynb**: Interactive model training
3. **03_evaluation.ipynb**: Detailed evaluation analysis
4. **04_visualization.ipynb**: Advanced visualization techniques

---

## 🔧 Configuration

### Key Configuration Files

#### `params.yaml` - Hyperparameters
- Preprocessing settings (HU window, spacing, size)
- Augmentation parameters
- Model architecture settings
- Training hyperparameters
- Evaluation metrics

#### `paths.yaml` - Directory Paths
- Raw data locations
- Processed data directories
- Output directories
- Model checkpoint paths

---

## 📝 Development Phases

### ✅ Phase 1: Project Structure (COMPLETE)
- Directory structure created
- Configuration files generated
- Template files created
- Documentation written

### 🔄 Phase 2: DICOM → NIfTI Conversion (NEXT)
- Implement DICOM loading
- RTSTRUCT mask extraction
- NIfTI conversion
- Batch processing

### 🔄 Phase 3: Data Normalization + Splitting
- HU clipping and normalization
- Image resizing
- Train/val/test splitting

### 🔄 Phase 4: PyTorch Dataset + Dataloader
- Custom Dataset class
- Data augmentation
- Dataloader creation

### 🔄 Phase 5: U-Net Implementation
- Model architecture
- Loss functions
- Parameter counting

### 🔄 Phase 6: Training Pipeline
- Training loop
- Validation loop
- Checkpointing
- TensorBoard logging

### 🔄 Phase 7: Evaluation + Visualization
- Metrics computation
- Prediction visualization
- 3D rendering

### 🔄 Phase 8: Final Report
- Scientific documentation
- Results analysis
- Future improvements

---

## 🛠️ Technologies Used

- **Python 3.10+**
- **PyTorch 2.0+**: Deep learning framework
- **SimpleITK**: Medical image processing
- **pydicom**: DICOM file handling
- **nibabel**: NIfTI file I/O
- **numpy, scipy**: Numerical computing
- **matplotlib, seaborn**: Visualization
- **pandas**: Data management
- **tensorboard**: Training monitoring

---

## 📚 References

1. Ronneberger, O., Fischer, P., & Brox, T. (2015). U-Net: Convolutional Networks for Biomedical Image Segmentation. *MICCAI*.

2. Aerts, H. J., et al. (2014). Decoding tumour phenotype by noninvasive imaging using a quantitative radiomics approach. *Nature Communications*.

3. TCIA NSCLC-Radiomics Dataset: https://doi.org/10.7937/K9/TCIA.2015.PF0M9REI

---

## 👥 Authors

**Medical Imaging Team**  
November 2025

---

## 📄 License

This project is licensed under the MIT License.

---

## 🙏 Acknowledgments

- The Cancer Imaging Archive (TCIA) for providing the NSCLC-Radiomics dataset
- The medical imaging community for open-source tools and libraries

---

## 📞 Support

For questions or issues:
1. Check documentation in `docs/` folder
2. Review Jupyter notebooks for examples
3. Open an issue on GitHub

---

**Happy Segmenting! 🏥🧠**
