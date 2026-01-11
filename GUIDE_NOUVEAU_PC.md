# 🖥️ Guide de Transfert - NSCLC Multi-Organ Segmentation

## 📋 Prérequis sur le nouveau PC

### Logiciels à installer AVANT le transfert:
1. **Python 3.10 ou 3.11** → https://www.python.org/downloads/
2. **Anaconda ou Miniconda** → https://www.anaconda.com/download
3. **CUDA Toolkit 11.8 ou 12.1** (si GPU NVIDIA) → https://developer.nvidia.com/cuda-downloads
4. **Git** (optionnel) → https://git-scm.com/downloads

---

## 📁 Fichiers à Copier

### OBLIGATOIRES (copier tout le dossier):
```
RADIO_PROJET/
├── DATA/
│   └── processed/
│       └── normalized/          ← ⚠️ IMPORTANT: ~10 GB de données
│           ├── LUNG1-001_ct_normalized.nii.gz
│           ├── LUNG1-001_mask_normalized.nii.gz
│           └── ... (158 patients × 2 fichiers)
├── incremental_training.py      ← Script de training
├── setup_new_pc.py              ← Script d'installation auto
└── requirements.txt             ← Dépendances Python
```

### OPTIONNELS (si vous voulez reprendre le training):
```
├── training_output/
│   ├── checkpoints/
│   │   ├── latest_checkpoint.pth    ← Pour reprendre
│   │   └── training_state.json      ← État du training
│   └── best_model.pth               ← Meilleur modèle
```

---

## 🚀 Installation sur le Nouveau PC

### Option 1: Installation Automatique (Recommandée)
```powershell
# 1. Ouvrir PowerShell en tant qu'administrateur
# 2. Naviguer vers le dossier du projet
cd C:\chemin\vers\RADIO_PROJET

# 3. Lancer le script d'installation
python setup_new_pc.py
```

### Option 2: Installation Manuelle

#### Étape 1: Créer l'environnement Conda
```powershell
conda create -n radio_env python=3.11 -y
conda activate radio_env
```

#### Étape 2: Installer PyTorch avec CUDA
```powershell
# Pour GPU NVIDIA (CUDA 11.8):
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Pour GPU NVIDIA (CUDA 12.1):
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Pour CPU seulement:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

#### Étape 3: Installer les autres dépendances
```powershell
pip install SimpleITK tqdm opencv-python matplotlib numpy
```

#### Étape 4: Vérifier l'installation
```powershell
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"Pas de GPU\"}')"
```

---

## 🏃 Lancer le Training

### Commande simple:
```powershell
cd C:\chemin\vers\RADIO_PROJET
conda activate radio_env
python incremental_training.py
```

### Si vous avez un checkpoint à reprendre:
Le script détecte automatiquement les checkpoints et reprend où il s'était arrêté.

---

## ⚙️ Configuration GPU (Modifier si nécessaire)

Dans `incremental_training.py`, vous pouvez ajuster ces paramètres pour GPU:

```python
CONFIG = {
    'patients_per_batch': 20,     # Plus de patients avec GPU
    'epochs_per_batch': 5,        # Plus d'epochs
    'total_rounds': 3,            # Moins de rounds nécessaires
    'batch_size': 8,              # Plus grand batch avec GPU (8-16)
}
```

---

## 🔍 Vérification Rapide

Après installation, exécutez ce test:
```powershell
python -c "
import torch
import SimpleITK as sitk
from pathlib import Path

print('=== Test Installation ===')
print(f'PyTorch: {torch.__version__}')
print(f'CUDA disponible: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'Mémoire GPU: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')

# Test données
data_dir = Path('DATA/processed/normalized')
ct_files = list(data_dir.glob('*_ct_normalized.nii.gz'))
print(f'Patients trouvés: {len(ct_files)}')
print('=== Tout est OK! ===' if len(ct_files) > 0 else '=== ERREUR: Données non trouvées ===')
"
```

---

## ❓ Problèmes Courants

### "CUDA not available"
- Vérifiez que CUDA Toolkit est installé
- Réinstallez PyTorch avec la bonne version CUDA:
  ```powershell
  pip uninstall torch torchvision torchaudio
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
  ```

### "Module not found"
```powershell
pip install SimpleITK tqdm opencv-python matplotlib
```

### "Out of memory"
Réduisez `batch_size` dans CONFIG (4 → 2)

### Erreur OpenMP
Ajoutez au début du script ou dans PowerShell:
```powershell
$env:KMP_DUPLICATE_LIB_OK="TRUE"
```

---

## 📊 Temps de Training Estimés

| Configuration | Temps par Batch | Temps Total |
|---------------|-----------------|-------------|
| CPU (Intel i5) | 30-45 min | 40-60 heures |
| GPU GTX 1060 | 5-8 min | 6-10 heures |
| GPU RTX 3060 | 2-4 min | 3-5 heures |
| GPU RTX 4080 | 1-2 min | 1.5-3 heures |

---

## 📞 Support

Si problème, vérifiez:
1. ✅ Python 3.10+ installé
2. ✅ Conda activé (`conda activate radio_env`)
3. ✅ Données dans `DATA/processed/normalized/`
4. ✅ PyTorch installé avec CUDA
