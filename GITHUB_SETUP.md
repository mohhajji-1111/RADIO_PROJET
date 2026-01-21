# 🚀 Configuration GitHub - NSCLC Multi-Organ Segmentation

## 📋 Initialisation du Repository

### 1. Initialiser Git (si pas déjà fait)
```bash
cd C:\Users\HP\Desktop\RADIO_PROJET
git init
git branch -M main
```

### 2. Ajouter les fichiers (SANS les données)
```bash
# Vérifier ce qui sera ajouté (devrait exclure DATA/)
git status

# Ajouter tous les fichiers de code
git add .
```

### 3. Premier commit
```bash
git commit -m "🎉 Initial commit - NSCLC Multi-Organ Segmentation project

- Scripts de training incrémental
- Pipeline de preprocessing DICOM/RT-STRUCT
- Architecture U-Net multi-organes
- Notebooks Colab/Kaggle
- Documentation complète
- Configuration PACS/Orthanc
"
```

### 4. Créer le repository sur GitHub
1. Aller sur https://github.com/new
2. Nom du repo: `RADIO_PROJET` ou `NSCLC-Segmentation`
3. Description: `🫁 Deep Learning pour segmentation multi-organes sur CT scans thoraciques - U-Net PyTorch`
4. **Ne PAS** initialiser avec README (on a déjà nos fichiers)
5. Créer le repository

### 5. Lier et pousser vers GitHub
```bash
# Remplacer "mohhajji-1111" par ton username GitHub
git remote add origin https://github.com/mohhajji-1111/RADIO_PROJET.git

# Pousser vers GitHub
git push -u origin main
```

---

## 📦 Ce qui est INCLUS dans GitHub
✅ Tous les scripts Python (`.py`)  
✅ Notebooks Jupyter (`.ipynb`)  
✅ Documentation (`.md`, `.txt`)  
✅ Configuration (`requirements.txt`, `.json`, `.yml`)  
✅ Scripts batch (`.bat`)  

## 🚫 Ce qui est EXCLU (fichiers volumineux)
❌ Dossier `DATA/` (~50+ GB)  
❌ Modèles entraînés `*.pth` (checkpoints)  
❌ Datasets Kaggle extraits  
❌ Visualizations générées  
❌ Logs d'entraînement  
❌ Cache Python (`__pycache__`)  

---

## 💾 Instructions de Téléchargement des Données

Pour quelqu'un qui clone ton projet, il devra télécharger les données séparément:

### Option 1: Kaggle Dataset (Recommandé)
```bash
# 1. Installer Kaggle CLI
pip install kaggle

# 2. Configurer API token (depuis https://www.kaggle.com/settings)
# Placer kaggle.json dans: C:\Users\USERNAME\.kaggle\

# 3. Télécharger le dataset
kaggle datasets download -d [TON_USERNAME]/nsclc-multiorgan-segmentation

# 4. Extraire
unzip nsclc-multiorgan-segmentation.zip -d DATA/processed/
```

### Option 2: The Cancer Imaging Archive (Original)
```bash
# Télécharger depuis:
# https://www.cancerimagingarchive.net/collection/nsclc-radiomics/
# https://www.cancerimagingarchive.net/collection/nsclc-radiomics-genomics/

# Puis extraire dans:
# DATA/NSCLC-Radiomics/
# DATA/NSCLC-Radiomics-Genomics/

# Et preprocesser:
python RTSTRUCT_PIPELINE_COMPLETE.py
```

### Option 3: Google Drive (Upload manuel)
1. Upload ton dossier `DATA/processed/normalized/` vers Google Drive
2. Partager le lien publiquement
3. Ajouter le lien dans le README

---

## 🔄 Mises à jour futures

### Pour pousser de nouvelles modifications:
```bash
git add .
git commit -m "Description des changements"
git push
```

### Pour récupérer les changements:
```bash
git pull origin main
```

---

## 🌐 Structure du README GitHub

Crée un bon README.md qui explique:
- 🎯 Objectif du projet
- 🏥 Contexte médical (segmentation radiothérapie)
- 🧠 Architecture technique (U-Net)
- 📊 Résultats obtenus
- 🚀 Comment lancer le training
- 📦 Comment télécharger les données
- 📝 Citations et références

---

## ✅ Checklist Avant Push

- [ ] `.gitignore` correctement configuré
- [ ] Aucun secret/token dans le code
- [ ] Pas de chemins absolus (C:\Users\...)
- [ ] README.md complet et clair
- [ ] requirements.txt à jour
- [ ] Instructions de téléchargement des données
- [ ] License file (MIT, Apache, etc.)

---

## 🔒 Sécurité

**ATTENTION**: Ne JAMAIS commit:
- Tokens API (Kaggle, AWS, etc.)
- Mots de passe
- Clés privées
- Données personnelles de patients (GDPR/HIPAA)

---

## 📧 Support

Pour questions: ouvrir une Issue sur GitHub

Bon courage! 🚀
