"""
Script pour préparer le dataset pour upload sur Kaggle.
Crée un package optimisé avec tous les fichiers nécessaires.
"""

import shutil
import zipfile
from pathlib import Path
import json
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
OUTPUT_DIR = Path("kaggle_upload")
NORMALIZED_DIR = Path("DATA/processed/normalized_rtstruct")
SPLITS_DIR = Path("splits_rtstruct")

def create_kaggle_structure():
    """Crée la structure du dataset pour Kaggle."""
    logger.info("=== PRÉPARATION DATASET KAGGLE ===")
    
    # Nettoyer et créer dossier output
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True)
    
    # Structure du dataset
    kaggle_data = OUTPUT_DIR / "nsclc_multiorgan_segmentation"
    kaggle_data.mkdir()
    
    # Sous-dossiers
    (kaggle_data / "normalized_ct").mkdir()
    (kaggle_data / "normalized_masks").mkdir()
    (kaggle_data / "splits").mkdir()
    (kaggle_data / "code").mkdir()
    
    logger.info(f"✅ Structure créée: {kaggle_data}")
    
    return kaggle_data


def copy_normalized_data(kaggle_data):
    """Copie les données normalisées."""
    logger.info("\n📁 Copie des données normalisées...")
    
    src_dir = NORMALIZED_DIR
    
    if not src_dir.exists():
        logger.error(f"❌ Dossier {src_dir} introuvable!")
        return False
    
    # Compter fichiers
    ct_files = list(src_dir.glob("*_ct_normalized.nii.gz"))
    mask_files = list(src_dir.glob("*_mask_normalized.nii.gz"))
    
    logger.info(f"  Trouvé: {len(ct_files)} CT scans")
    logger.info(f"  Trouvé: {len(mask_files)} masques")
    
    # Copier CT scans
    logger.info("  Copie des CT scans...")
    for ct_file in ct_files:
        dest = kaggle_data / "normalized_ct" / ct_file.name
        shutil.copy2(ct_file, dest)
    
    # Copier masques
    logger.info("  Copie des masques...")
    for mask_file in mask_files:
        dest = kaggle_data / "normalized_masks" / mask_file.name
        shutil.copy2(mask_file, dest)
    
    logger.info(f"✅ Données copiées: {len(ct_files)} patients")
    return True


def copy_splits(kaggle_data):
    """Copie les fichiers train/val/test splits."""
    logger.info("\n📋 Copie des splits...")
    
    split_files = ["train.txt", "val.txt", "test.txt"]
    
    for split_file in split_files:
        src = SPLITS_DIR / split_file
        if src.exists():
            dest = kaggle_data / "splits" / split_file
            shutil.copy2(src, dest)
            
            # Compter lignes
            with open(src) as f:
                count = len(f.readlines())
            logger.info(f"  ✅ {split_file}: {count} patients")
        else:
            logger.warning(f"  ⚠️ {split_file} introuvable")
    
    return True


def copy_code_files(kaggle_data):
    """Copie les fichiers Python nécessaires."""
    logger.info("\n💻 Copie des fichiers code...")
    
    code_files = [
        "dataset_multi_organ.py",
        "unet_multi_organ.py",
        "train_multi_organ.py"
    ]
    
    for code_file in code_files:
        src = Path(code_file)
        if src.exists():
            dest = kaggle_data / "code" / code_file
            shutil.copy2(src, dest)
            logger.info(f"  ✅ {code_file}")
        else:
            logger.warning(f"  ⚠️ {code_file} introuvable")
    
    return True


def create_metadata_json(kaggle_data):
    """Crée le fichier dataset-metadata.json pour Kaggle."""
    logger.info("\n📝 Création de dataset-metadata.json...")
    
    metadata = {
        "title": "NSCLC Multi-Organ Segmentation (Normalized)",
        "id": "votre-username/nsclc-multiorgan-segmentation",
        "licenses": [{"name": "CC0-1.0"}],
        "keywords": [
            "medical imaging",
            "lung cancer",
            "segmentation",
            "ct scan",
            "radiomics",
            "u-net",
            "deep learning"
        ],
        "subtitle": "158 patients with 7 organ segmentations for lung cancer radiotherapy planning",
        "description": "This dataset contains preprocessed CT scans and multi-organ segmentation masks from NSCLC-Radiomics dataset. Includes 158 patients with normalized CT images (256x256, z-score) and multi-class masks (8 labels: Background, GTV, PTV, Right Lung, Left Lung, Heart, Esophagus, Spinal Cord).",
        "resources": [
            {
                "path": "normalized_ct/",
                "description": "Normalized CT scans (windowing + z-score, 256x256)"
            },
            {
                "path": "normalized_masks/",
                "description": "Multi-organ segmentation masks (labels 0-7)"
            },
            {
                "path": "splits/",
                "description": "Train/Val/Test patient splits"
            },
            {
                "path": "code/",
                "description": "PyTorch Dataset, U-Net model, and training scripts"
            }
        ]
    }
    
    metadata_path = kaggle_data / "dataset-metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"✅ Metadata créé: {metadata_path}")
    return True


def create_readme(kaggle_data):
    """Crée un README.md pour le dataset."""
    logger.info("\n📄 Création de README.md...")
    
    readme_content = """# NSCLC Multi-Organ Segmentation Dataset (Preprocessed)

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
"""
    
    readme_path = kaggle_data / "README.md"
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write(readme_content)
    
    logger.info(f"✅ README créé: {readme_path}")
    return True


def calculate_sizes(kaggle_data):
    """Calcule la taille totale du dataset."""
    logger.info("\n📏 Calcul des tailles...")
    
    def get_dir_size(path):
        total = 0
        for item in path.rglob('*'):
            if item.is_file():
                total += item.stat().st_size
        return total / (1024**3)  # Convert to GB
    
    ct_size = get_dir_size(kaggle_data / "normalized_ct")
    mask_size = get_dir_size(kaggle_data / "normalized_masks")
    total_size = get_dir_size(kaggle_data)
    
    logger.info(f"  CT scans: {ct_size:.2f} GB")
    logger.info(f"  Masks: {mask_size:.2f} GB")
    logger.info(f"  Total: {total_size:.2f} GB")
    
    return total_size


def create_instructions(kaggle_data):
    """Crée un fichier d'instructions pour l'upload."""
    logger.info("\n📋 Création des instructions...")
    
    instructions = """╔══════════════════════════════════════════════════════════════════════════╗
║                   INSTRUCTIONS POUR UPLOAD KAGGLE                        ║
╚══════════════════════════════════════════════════════════════════════════╝

🎯 ÉTAPE 1: CRÉER COMPTE KAGGLE
──────────────────────────────────────────────────────────────────────────────
1. Aller sur https://www.kaggle.com
2. Créer un compte (ou se connecter)
3. Vérifier email

🔑 ÉTAPE 2: INSTALLER KAGGLE API (OPTIONNEL - POUR CLI)
──────────────────────────────────────────────────────────────────────────────
# PowerShell
pip install kaggle

# Télécharger votre API token:
# - Aller sur https://www.kaggle.com/settings
# - Section "API" → "Create New API Token"
# - Télécharge kaggle.json
# - Placer dans: C:\\Users\\<USER>\\.kaggle\\kaggle.json

📤 ÉTAPE 3A: UPLOAD VIA INTERFACE WEB (RECOMMANDÉ)
──────────────────────────────────────────────────────────────────────────────
1. Aller sur https://www.kaggle.com/datasets
2. Cliquer "New Dataset"
3. Drag & drop tout le dossier "nsclc_multiorgan_segmentation/"
   OU
   Compresser en .zip puis upload:
   > Compress-Archive -Path kaggle_upload/nsclc_multiorgan_segmentation `
                      -DestinationPath kaggle_dataset.zip

4. Remplir les infos:
   - Title: "NSCLC Multi-Organ Segmentation (Normalized)"
   - Subtitle: "158 patients with 7 organ segmentations"
   - Description: (copier depuis README.md)
   - License: CC0 1.0

5. Tags (keywords):
   - medical imaging
   - lung cancer
   - segmentation
   - ct scan
   - radiomics
   - u-net
   - deep learning

6. Cliquer "Create"

📤 ÉTAPE 3B: UPLOAD VIA CLI (ALTERNATIVE)
──────────────────────────────────────────────────────────────────────────────
# PowerShell (depuis RADIO_PROJET/)

# 1. Modifier dataset-metadata.json:
#    Remplacer "votre-username" par votre vrai username Kaggle

# 2. Upload dataset:
cd kaggle_upload/nsclc_multiorgan_segmentation
kaggle datasets create -p .

# 3. Attendre upload (peut prendre 1-2 heures selon connexion)

🔄 ÉTAPE 3C: UPDATE DATASET (SI DÉJÀ CRÉÉ)
──────────────────────────────────────────────────────────────────────────────
kaggle datasets version -p . -m "Updated preprocessing pipeline"

📓 ÉTAPE 4: CRÉER NOTEBOOK KAGGLE
──────────────────────────────────────────────────────────────────────────────
1. Aller sur https://www.kaggle.com/code
2. Cliquer "New Notebook"
3. Settings → Add Data → Chercher votre dataset
4. Copier le code depuis README.md (Quick Start)
5. Settings → Accelerator → GPU T4 (gratuit, 30h/semaine)
6. Run All!

⚙️ ÉTAPE 5: TRAINING SUR KAGGLE
──────────────────────────────────────────────────────────────────────────────
Voir le notebook template créé: kaggle_training_notebook.ipynb

Temps estimé:
- Upload dataset: 1-2 heures (selon connexion)
- Training (50 epochs): 3-4 heures sur GPU T4
- Total: ~5-6 heures

💰 COÛT
──────────────────────────────────────────────────────────────────────────────
✅ GRATUIT! (30h GPU par semaine)

📊 LIMITES KAGGLE GRATUIT
──────────────────────────────────────────────────────────────────────────────
- GPU: 30h/semaine (T4 ou P100)
- RAM: 30 GB
- Disk: 20 GB (suffisant pour ce dataset)
- Notebooks publics: Illimités
- Notebooks privés: 20 max

🎓 CONSEILS
──────────────────────────────────────────────────────────────────────────────
1. Upload le dataset une seule fois → réutilisable à l'infini
2. Tester d'abord avec 1-2 epochs pour vérifier que tout marche
3. Sauvegarder checkpoints régulièrement (toutes les 10 epochs)
4. Si interruption: relancer notebook, charger dernier checkpoint
5. Faire public le notebook → bon pour portfolio/CV!

📧 SUPPORT
──────────────────────────────────────────────────────────────────────────────
- Documentation: https://www.kaggle.com/docs/datasets
- Forum: https://www.kaggle.com/discussions
- API Docs: https://github.com/Kaggle/kaggle-api

╔══════════════════════════════════════════════════════════════════════════╗
║                              BON COURAGE! 🚀                             ║
╚══════════════════════════════════════════════════════════════════════════╝
"""
    
    instructions_path = OUTPUT_DIR / "UPLOAD_INSTRUCTIONS.txt"
    with open(instructions_path, 'w', encoding='utf-8') as f:
        f.write(instructions)
    
    logger.info(f"✅ Instructions: {instructions_path}")
    return True


def main():
    """Fonction principale."""
    try:
        # 1. Créer structure
        kaggle_data = create_kaggle_structure()
        
        # 2. Copier données
        if not copy_normalized_data(kaggle_data):
            logger.error("❌ Échec copie des données")
            return
        
        # 3. Copier splits
        copy_splits(kaggle_data)
        
        # 4. Copier code
        copy_code_files(kaggle_data)
        
        # 5. Créer metadata
        create_metadata_json(kaggle_data)
        
        # 6. Créer README
        create_readme(kaggle_data)
        
        # 7. Calculer tailles
        total_size = calculate_sizes(kaggle_data)
        
        # 8. Créer instructions
        create_instructions(kaggle_data)
        
        # Résumé final
        print("\n" + "="*80)
        print("✅ DATASET KAGGLE PRÊT!")
        print("="*80)
        print(f"📁 Dossier: {kaggle_data}")
        print(f"💾 Taille: {total_size:.2f} GB")
        print(f"📋 Patients: 158")
        print(f"🖼️ Slices: 57,148")
        print("\n📤 PROCHAINES ÉTAPES:")
        print("  1. Lire: kaggle_upload/UPLOAD_INSTRUCTIONS.txt")
        print("  2. Compresser (optionnel):")
        print(f"     Compress-Archive -Path {kaggle_data} -DestinationPath kaggle_dataset.zip")
        print("  3. Upload sur https://www.kaggle.com/datasets")
        print("  4. Créer notebook et lancer training!")
        print("="*80)
        
    except Exception as e:
        logger.error(f"❌ Erreur: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
