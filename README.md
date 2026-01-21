<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8+-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-2.0+-red.svg" alt="PyTorch">
  <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License">
  <img src="https://img.shields.io/badge/Medical%20Imaging-DICOM-orange.svg" alt="DICOM">
</p>

# 🫁 NSCLC Multi-Organ Segmentation

> **Deep Learning pour la segmentation automatique multi-organes sur CT scans thoraciques**

Architecture **U-Net** pour la segmentation simultanée de **8 structures anatomiques** en radiothérapie pulmonaire, entraînée sur le dataset **NSCLC-Radiomics** (422 patients).

---

## 🎯 Objectif

Segmentation automatique des organes à risque (OAR) et volumes cibles pour la planification de radiothérapie du cancer du poumon non à petites cellules (NSCLC).

### Structures Segmentées

| ID | Structure | Description |
|----|-----------|-------------|
| 0 | Background | Fond de l'image |
| 1 | **GTV** | Gross Tumor Volume (tumeur) |
| 2 | **PTV** | Planning Target Volume |
| 3 | **Poumon Droit** | Right Lung |
| 4 | **Poumon Gauche** | Left Lung |
| 5 | **Cœur** | Heart |
| 6 | **Œsophage** | Esophagus |
| 7 | **Moelle Épinière** | Spinal Cord |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        U-Net Multi-Organ                        │
├─────────────────────────────────────────────────────────────────┤
│  Input: CT Slice (512×512×1)                                    │
│     ↓                                                           │
│  Encoder: Conv → BatchNorm → ReLU → MaxPool (×4)               │
│     ↓                                                           │
│  Bottleneck: 1024 channels                                      │
│     ↓                                                           │
│  Decoder: UpConv → Concat → Conv → BatchNorm → ReLU (×4)       │
│     ↓                                                           │
│  Output: Segmentation Map (512×512×8)                           │
└─────────────────────────────────────────────────────────────────┘
```

### Caractéristiques Techniques

- **Encodeur**: 4 blocs de downsampling (64→128→256→512→1024)
- **Skip Connections**: Concaténation des features multi-échelles
- **Décodeur**: 4 blocs d'upsampling transposés
- **Loss**: Dice + Binary Cross-Entropy combinées
- **Optimiseur**: Adam (lr=1e-4)

---

## 📊 Dataset

**NSCLC-Radiomics** - The Cancer Imaging Archive (TCIA)

| Statistique | Valeur |
|-------------|--------|
| Patients | 422 |
| CT Scans | 422 |
| RT-STRUCT | 422 |
| Slices totales | ~57,000 |
| Résolution | 512×512 |

### Preprocessing Pipeline

```
DICOM → NIfTI → Normalisation → Data Augmentation → Training
```

1. **Conversion DICOM→NIfTI**: Standardisation du format
2. **Extraction RT-STRUCT**: Parsing des contours ROI
3. **Normalisation**: HU windowing [-1024, 3071] → [0, 1]
4. **Resampling**: Isotropic 1mm×1mm×3mm

---

## 🚀 Installation

### Prérequis

- Python 3.8+
- CUDA 11.0+ (GPU recommandé)
- 16GB RAM minimum

### Setup

```bash
# Cloner le repository
git clone https://github.com/mohhajji-1111/RADIO_PROJET.git
cd RADIO_PROJET

# Créer environnement virtuel
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou: venv\Scripts\activate  # Windows

# Installer dépendances
pip install -r requirements.txt
```

---

## 📥 Téléchargement des Données

### Option 1: Kaggle (Recommandé)

```bash
pip install kaggle
kaggle datasets download -d [username]/nsclc-multiorgan-segmentation
unzip nsclc-multiorgan-segmentation.zip -d DATA/
```

### Option 2: TCIA (Original)

Télécharger depuis [Cancer Imaging Archive](https://www.cancerimagingarchive.net/collection/nsclc-radiomics/):

```bash
# Après téléchargement, lancer le preprocessing
python RTSTRUCT_PIPELINE_COMPLETE.py
```

---

## 🏋️ Entraînement

### Training Local

```bash
# Training incrémental (recommandé pour grande dataset)
python incremental_training.py

# Ou training standard
python train_multi_organ.py
```

### Training sur Cloud (Kaggle/Colab)

```python
# Voir notebooks/
# - colab_training.ipynb
# - kaggle_training_notebook.ipynb
```

### Configuration

```python
CONFIG = {
    'batch_size': 8,
    'learning_rate': 1e-4,
    'num_epochs': 50,
    'patience': 10,  # Early stopping
    'num_classes': 8,
    'device': 'cuda'
}
```

---

## 📈 Résultats

### Métriques de Performance

| Organe | Dice Score | IoU |
|--------|------------|-----|
| GTV (Tumeur) | 0.78 | 0.64 |
| PTV | 0.82 | 0.70 |
| Poumon Droit | 0.97 | 0.94 |
| Poumon Gauche | 0.96 | 0.93 |
| Cœur | 0.92 | 0.85 |
| Œsophage | 0.71 | 0.55 |
| Moelle Épinière | 0.84 | 0.72 |

### Visualisations

Les prédictions sont sauvegardées dans `visualizations/`:
- Overlays CT + Segmentation
- Vues 3D des structures
- Courbes d'entraînement

---

## 📁 Structure du Projet

```
RADIO_PROJET/
├── src/
│   ├── data/           # Dataset PyTorch
│   ├── models/         # Architecture U-Net
│   ├── preprocessing/  # Pipeline DICOM
│   ├── training/       # Boucle d'entraînement
│   └── config/         # Configuration YAML
├── scripts/
│   ├── train_unet.py
│   ├── evaluate_unet.py
│   └── preprocess_all.py
├── notebooks/
│   ├── colab_training.ipynb
│   └── kaggle_training_notebook.ipynb
├── incremental_training.py  # Training par batches
├── unet_multi_organ.py      # Modèle principal
├── dataset_multi_organ.py   # DataLoader
└── requirements.txt
```

---

## 🔬 Utilisation

### Inference

```python
import torch
from unet_multi_organ import UNetMultiOrgan

# Charger le modèle
model = UNetMultiOrgan(in_channels=1, out_channels=8)
model.load_state_dict(torch.load('best_model.pth'))
model.eval()

# Prédiction
with torch.no_grad():
    prediction = model(ct_slice)
    segmentation = prediction.argmax(dim=1)
```

### Évaluation

```bash
python scripts/evaluate_unet.py --model best_model.pth --data DATA/processed/
```

---

## 📚 Références

1. **NSCLC-Radiomics Dataset**: Aerts et al., "Decoding tumour phenotype by noninvasive imaging using a quantitative radiomics approach", Nature Communications, 2014

2. **U-Net**: Ronneberger et al., "U-Net: Convolutional Networks for Biomedical Image Segmentation", MICCAI, 2015

3. **TCIA**: Clark et al., "The Cancer Imaging Archive (TCIA)", Journal of Digital Imaging, 2013

---

## 📄 License

MIT License - voir [LICENSE](LICENSE) pour détails.

---

## 👤 Auteur

**Projet de Segmentation Médicale**
- Master en Intelligence Artificielle
- Spécialisation: Imagerie Médicale & Deep Learning

---

<p align="center">
  <b>⭐ Star ce repo si vous le trouvez utile!</b>
</p>
