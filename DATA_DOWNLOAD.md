# NSCLC Multi-Organ Segmentation - Installation des Données

## 📊 Télécharger les Données

Les données d'entraînement (**18 GB, 158 patients**) sont hébergées sur **Kaggle** (gratuit):

### Option 1: Via Kaggle CLI (Recommandé)

```bash
# 1. Installer Kaggle CLI
pip install kaggle

# 2. Configurer l'API Kaggle
# Téléchargez votre kaggle.json depuis: https://www.kaggle.com/settings
# Placez-le dans: C:\Users\<USER>\.kaggle\kaggle.json

# 3. Télécharger le dataset
cd C:\Users\HP\Desktop\RADIO_PROJET
kaggle datasets download -d mhajji11/nsclc-multiorgan-segmentation

# 4. Extraire les données
Expand-Archive nsclc-multiorgan-segmentation.zip -DestinationPath DATA/processed/normalized/
```

### Option 2: Via Interface Web

1. Allez sur: https://www.kaggle.com/datasets/mhajji11/nsclc-multiorgan-segmentation
2. Cliquez sur **"Download"** (9.36 GB compressé)
3. Extrayez dans `DATA/processed/normalized/`

### Option 3: Clé USB / Disque Externe

Si vous avez déjà les données localement:
```bash
# Copiez simplement le dossier
cp -r ANCIEN_PC/DATA/processed/normalized/* NOUVEAU_PC/DATA/processed/normalized/
```

---

## ✅ Vérification

Après téléchargement, vérifiez:

```bash
# Devrait afficher: 158 fichiers CT + 158 fichiers masks = 316 total
ls DATA/processed/normalized/*.nii.gz | measure
```

Structure attendue:
```
DATA/processed/normalized/
├── LUNG1-001_ct_normalized.nii.gz
├── LUNG1-001_mask_normalized.nii.gz
├── LUNG1-002_ct_normalized.nii.gz
├── LUNG1-002_mask_normalized.nii.gz
...
└── LUNG1-160_mask_normalized.nii.gz
```

---

## 🚀 Lancer le Training

Une fois les données téléchargées:

```bash
# Sur nouveau PC avec GPU
python setup_new_pc.py      # Installation automatique
python incremental_training.py  # Lancer le training
```

Ou double-cliquez sur `START_TRAINING.bat`

---

## 📞 Problèmes?

- Dataset Kaggle: https://www.kaggle.com/datasets/mhajji11/nsclc-multiorgan-segmentation
- Code GitHub: https://github.com/mohhajji-1111/RADIO_PROJET
