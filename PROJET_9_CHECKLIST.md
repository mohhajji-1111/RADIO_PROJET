# ✅ CHECKLIST Projet #9 - El Ossmani / Beerrada

## 📌 Titre du Projet
**Exploration et manipulation des données médicales DICOM avec pydicom**

---

## ✅ Partie 1 : Manipulation DICOM & RTSTRUCT (TERMINÉ)

- [x] **Lecture des fichiers DICOM** (CT, RTSTRUCT)
- [x] **Extraction des contours** depuis RTSTRUCT
- [x] **Conversion en format NIfTI** pour traitement
- [x] **Normalisation des images** (256×256×402)
- [x] **Création des masques multi-organes** (8 classes)
- [x] **Pipeline reproductible** via scripts Python

**Fichiers concernés :**
- `DATA/NSCLC-Radiomics/` → Données DICOM sources
- `DATA/processed/normalized/` → Données traitées (158 patients, 316 fichiers)
- Scripts de preprocessing (déjà exécutés)

---

## ❌ Partie 2 : RTDOSE (À FAIRE)

### 🎯 Objectif
Extraire et visualiser les **distributions de dose de radiation** (RTDOSE) pour analyser les plans de traitement.

### 📝 Tâches requises

#### **Tâche 1 : Vérifier disponibilité RTDOSE**
```powershell
# Chercher fichiers RTDOSE dans le dataset
Get-ChildItem -Path "C:\Users\HP\Desktop\RADIO_PROJET\DATA\NSCLC-Radiomics" -Recurse -Filter "*RTDOSE*" | Select-Object FullName
```

- [ ] Identifier quels patients ont des fichiers RTDOSE
- [ ] Documenter la structure des fichiers RTDOSE

#### **Tâche 2 : Script d'extraction RTDOSE**
Créer `extract_rtdose.py` pour :
- [ ] Lire fichiers RTDOSE avec pydicom
- [ ] Extraire la matrice de dose 3D
- [ ] Convertir en unités Gy (Gray)
- [ ] Sauvegarder en format NIfTI (.nii.gz)

**Exemple de code :**
```python
import pydicom
import numpy as np
import SimpleITK as sitk

def extract_rtdose(rtdose_path, output_path):
    """
    Extrait la distribution de dose depuis un fichier RTDOSE DICOM.
    
    Args:
        rtdose_path: Chemin vers fichier RTDOSE
        output_path: Chemin de sortie (.nii.gz)
    """
    # Lire RTDOSE
    ds = pydicom.dcmread(rtdose_path)
    
    # Extraire matrice de dose
    dose_array = ds.pixel_array * ds.DoseGridScaling  # En Gy
    
    # Créer image SimpleITK
    dose_image = sitk.GetImageFromArray(dose_array)
    
    # Définir spacing et origine
    spacing = [float(ds.PixelSpacing[0]), 
               float(ds.PixelSpacing[1]), 
               float(ds.GridFrameOffsetVector[1] - ds.GridFrameOffsetVector[0])]
    dose_image.SetSpacing(spacing)
    dose_image.SetOrigin([float(ds.ImagePositionPatient[0]),
                          float(ds.ImagePositionPatient[1]),
                          float(ds.ImagePositionPatient[2])])
    
    # Sauvegarder
    sitk.WriteImage(dose_image, output_path)
    print(f"Dose sauvegardée: {output_path}")
    print(f"  - Dose min: {dose_array.min():.2f} Gy")
    print(f"  - Dose max: {dose_array.max():.2f} Gy")
    print(f"  - Dose moyenne: {dose_array.mean():.2f} Gy")
    
    return dose_image
```

#### **Tâche 3 : Visualisation RTDOSE**
Créer `visualize_rtdose.py` pour :
- [ ] Superposer dose sur CT
- [ ] Générer cartes de chaleur (heatmap)
- [ ] Créer histogrammes dose-volume (DVH)
- [ ] Analyser dose par organe (utiliser masques existants)

**Exemple DVH :**
```python
import matplotlib.pyplot as plt

def plot_dvh(dose_image, mask_image, organ_name):
    """
    Génère un histogramme dose-volume pour un organe.
    """
    dose_array = sitk.GetArrayFromImage(dose_image)
    mask_array = sitk.GetArrayFromImage(mask_image)
    
    # Dose dans l'organe uniquement
    organ_dose = dose_array[mask_array > 0]
    
    # DVH
    volumes = [(organ_dose >= dose).sum() / len(organ_dose) * 100 
               for dose in np.linspace(0, organ_dose.max(), 100)]
    
    plt.plot(np.linspace(0, organ_dose.max(), 100), volumes)
    plt.xlabel('Dose (Gy)')
    plt.ylabel('Volume (%)')
    plt.title(f'DVH - {organ_name}')
    plt.grid(True)
    plt.savefig(f'dvh_{organ_name}.png')
```

- [ ] Créer DVH pour chaque organe (GTV, Poumons, Cœur, Œsophage, etc.)
- [ ] Comparer dose planifiée vs limites de tolérance

---

## ❌ Partie 3 : Segmentation avec U-Net (EN COURS)

### 🎯 Objectif
Entraîner modèle U-Net pour segmentation automatique des organes.

### 📝 Tâches requises

#### **Tâche 4 : Entraînement du modèle**
- [ ] **Décision matériel** : PC actuel (CPU lent) ou nouveau PC (GPU rapide)
- [ ] Si nouveau PC : Suivre [GUIDE_NOUVEAU_PC.md](GUIDE_NOUVEAU_PC.md)
- [ ] Télécharger données depuis Kaggle : [DATA_DOWNLOAD.md](DATA_DOWNLOAD.md)
- [ ] Lancer entraînement :
  ```powershell
  cd C:\Users\HP\Desktop\RADIO_PROJET
  conda activate .conda
  python incremental_training.py
  ```
- [ ] Surveiller progression (checkpoints sauvegardés automatiquement)
- [ ] **Durée attendue** :
  - GPU RTX 4080 : ~1.5 heures
  - GPU RTX 3060 : ~2-3 heures
  - GPU GTX 1060 : ~4-6 heures
  - CPU (actuel) : ~40-60 heures ⚠️

#### **Tâche 5 : Évaluation des résultats**
- [ ] Analyser courbes d'entraînement (training_curves.png)
- [ ] Vérifier Dice Score final (objectif : 0.80-0.90)
- [ ] Tester sur ensemble de validation
- [ ] Générer prédictions sur nouveaux patients
- [ ] Visualiser segmentations (CT + masques prédits)

---

## ❌ Partie 4 : Interopérabilité Serveur (À FAIRE)

### 🎯 Objectif
Tester scripts avec un serveur pour vérifier reproductibilité.

### 📝 Tâches requises

#### **Tâche 6 : Conteneurisation Docker**
Créer `Dockerfile` pour :
- [ ] Environnement Python reproductible
- [ ] Dépendances (PyTorch, SimpleITK, pydicom)
- [ ] Scripts de preprocessing
- [ ] Modèle entraîné

**Exemple Dockerfile :**
```dockerfile
FROM pytorch/pytorch:2.0.1-cuda11.8-cudnn8-runtime

# Installation dépendances
RUN pip install SimpleITK pydicom scikit-learn matplotlib

# Copier code
COPY . /app
WORKDIR /app

# Point d'entrée
CMD ["python", "incremental_training.py"]
```

#### **Tâche 7 : Test sur serveur**
- [ ] Déployer container sur serveur
- [ ] Tester pipeline complet (preprocessing + entraînement)
- [ ] Vérifier résultats identiques au local
- [ ] Documenter commandes de déploiement

---

## 📊 Livrables Finaux

### Documentation
- [ ] **Rapport écrit** (format PDF) incluant :
  - Introduction DICOM/RTSTRUCT/RTDOSE
  - Méthodologie (preprocessing, segmentation)
  - Résultats (Dice scores, visualisations)
  - Analyse DVH (si RTDOSE disponible)
  - Conclusion et perspectives
  
- [ ] **README.md complet** avec :
  - Installation
  - Utilisation
  - Structure du projet
  - Exemples de commandes

### Code
- [ ] **Scripts documentés** :
  - `extract_rtdose.py` (si RTDOSE disponible)
  - `visualize_rtdose.py`
  - `incremental_training.py` (déjà fait ✓)
  - `Dockerfile` pour reproductibilité
  
- [ ] **Notebooks Jupyter** (optionnel mais recommandé) :
  - `01_DICOM_Exploration.ipynb`
  - `02_RTDOSE_Analysis.ipynb`
  - `03_Segmentation_Results.ipynb`

### Résultats
- [ ] **Visualisations** :
  - Images CT avec contours RTSTRUCT
  - Cartes de dose (heatmaps)
  - DVH par organe
  - Segmentations prédites vs ground truth
  
- [ ] **Modèle entraîné** :
  - `best_model.pth` (poids du meilleur modèle)
  - Métriques de performance (Dice scores)

---

## 🎯 Priorités Immédiates

### 🔴 URGENT (Cette semaine)
1. **Entraîner le modèle** sur `normalized/`
   - Décider : PC actuel ou nouveau PC
   - Lancer `incremental_training.py`
   - Attendre résultats (1.5-60h selon matériel)

2. **Vérifier disponibilité RTDOSE**
   - Chercher fichiers RTDOSE dans le dataset
   - Si présents : créer scripts d'extraction

### 🟡 IMPORTANT (Semaine prochaine)
3. **Créer visualisations avancées**
   - DVH si RTDOSE disponible
   - Segmentations prédites
   - Rapport de métriques

4. **Dockerisation**
   - Créer Dockerfile
   - Tester reproductibilité

### 🟢 BONUS (Si temps restant)
5. **Notebooks Jupyter**
   - Analyse interactive
   - Visualisations riches

6. **Rapport final**
   - Rédaction
   - Mise en page professionnelle

---

## 📌 Résumé Décisionnel

| Question | Réponse |
|----------|---------|
| Dataset à utiliser ? | ✅ `normalized/` (8 classes, déjà configuré) |
| RTDOSE requis ? | ❓ Vérifier si disponible dans dataset |
| Entraîner où ? | 🤔 **Décision requise** : PC actuel (lent) ou nouveau PC (rapide) |
| Docker obligatoire ? | ✅ Oui (pour interopérabilité serveur) |
| Délai estimé ? | 📅 2-3 semaines (avec entraînement + Docker + rapport) |

---

## ✅ Commande Immédiate

**Pour commencer l'entraînement maintenant :**
```powershell
cd C:\Users\HP\Desktop\RADIO_PROJET
conda activate .conda
python incremental_training.py
```

**Ou préparer transfert vers nouveau PC :**
- Suivre [GUIDE_NOUVEAU_PC.md](GUIDE_NOUVEAU_PC.md)
- Télécharger données : [DATA_DOWNLOAD.md](DATA_DOWNLOAD.md)
