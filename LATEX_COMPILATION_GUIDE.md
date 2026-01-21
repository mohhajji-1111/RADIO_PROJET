# Guide de Compilation du Rapport LaTeX

## 📄 Fichier: rapport_segmentation_pulmonaire.tex

Ce document LaTeX contient le rapport complet du projet de segmentation multi-organes pulmonaire.

## 🖼️ Images Incluses

Le rapport utilise les visualisations suivantes (toutes présentes dans le projet):

### Statistiques et Données
- `visualizations/dataset_statistics.png` - Statistiques du dataset
- `visualizations/roi_organ_presence.png` - Présence des organes
- `visualizations/roi_volumes_distributions.png` - Distribution des volumes
- `visualizations/roi_volumes_boxplots.png` - Boxplots des volumes

### Pipeline de Traitement
- `visualizations/phase2_dicom_to_nifti.png` - Conversion DICOM
- `visualizations/phase2_5_multi_organ.png` - Extraction masques
- `visualizations/phase3_normalization.png` - Normalisation
- `visualizations/phase4_dataset_pytorch.png` - Dataset PyTorch
- `visualizations/quality_control.png` - Contrôle qualité

### Résultats
- `training_output/training_curves.png` - Courbes d'entraînement
- `visualizations/overlays/LUNG1-00X_overlay.png` - Exemples de segmentation (5 patients)
- `visualizations/3d_views/LUNG1-00X_3dview.png` - Vues 3D (5 patients)

## 📋 Prérequis

### Option 1: LaTeX Local

#### Windows (MiKTeX ou TeX Live)

```bash
# Installer MiKTeX depuis: https://miktex.org/download
# Ou TeX Live depuis: https://www.tug.org/texlive/

# Vérifier l'installation
pdflatex --version
```

#### Linux (Ubuntu/Debian)

```bash
sudo apt-get update
sudo apt-get install texlive-full texlive-lang-french
```

#### macOS

```bash
# Installer MacTeX depuis: https://www.tug.org/mactex/
brew install --cask mactex
```

### Option 2: Overleaf (Recommandé pour débutants)

1. Créer un compte sur https://www.overleaf.com
2. Créer un nouveau projet
3. Upload tous les fichiers

## 🔧 Compilation

### Méthode 1: pdflatex (Ligne de commande)

```bash
# Se placer dans le répertoire du projet
cd C:\Users\HP\Desktop\RADIO_PROJET

# Compiler (3 fois pour la table des matières)
pdflatex rapport_segmentation_pulmonaire.tex
pdflatex rapport_segmentation_pulmonaire.tex
pdflatex rapport_segmentation_pulmonaire.tex

# Le PDF sera généré: rapport_segmentation_pulmonaire.pdf
```

### Méthode 2: latexmk (Automatique)

```bash
# Installer latexmk si pas déjà fait
# Puis compiler:
latexmk -pdf rapport_segmentation_pulmonaire.tex

# Nettoyage des fichiers temporaires
latexmk -c
```

### Méthode 3: VS Code avec LaTeX Workshop

1. Installer l'extension "LaTeX Workshop" dans VS Code
2. Ouvrir le fichier `.tex`
3. Ctrl+Alt+B pour compiler
4. Ctrl+Alt+V pour visualiser le PDF

### Méthode 4: Overleaf

1. Upload le fichier `.tex` et le dossier `visualizations/`
2. Cliquer sur "Recompile"
3. Le PDF s'affiche automatiquement

## 📁 Structure Requise

Assurez-vous que la structure est correcte:

```
RADIO_PROJET/
├── rapport_segmentation_pulmonaire.tex    # Fichier principal
├── visualizations/
│   ├── dataset_statistics.png
│   ├── roi_organ_presence.png
│   ├── roi_volumes_distributions.png
│   ├── roi_volumes_boxplots.png
│   ├── phase2_dicom_to_nifti.png
│   ├── phase2_5_multi_organ.png
│   ├── phase3_normalization.png
│   ├── phase4_dataset_pytorch.png
│   ├── quality_control.png
│   ├── overlays/
│   │   ├── LUNG1-001_overlay.png
│   │   ├── LUNG1-002_overlay.png
│   │   ├── LUNG1-003_overlay.png
│   │   ├── LUNG1-004_overlay.png
│   │   └── LUNG1-005_overlay.png
│   └── 3d_views/
│       ├── LUNG1-001_3dview.png
│       ├── LUNG1-002_3dview.png
│       ├── LUNG1-003_3dview.png
│       └── LUNG1-004_3dview.png
└── training_output/
    └── training_curves.png
```

## 🎨 Logo ENSAM (Optionnel)

Le document référence `logo_ensam.png` dans la page de garde. Pour l'ajouter:

1. Télécharger le logo officiel ENSAM
2. Le placer dans le même répertoire que le `.tex`
3. Ou commenter la ligne dans le LaTeX:

```latex
% \includegraphics[width=0.3\textwidth]{logo_ensam.png}\\[1cm]
```

## 🔍 Résolution de Problèmes

### Erreur: "File not found"

```bash
# Vérifier que toutes les images existent
ls visualizations/*.png
ls visualizations/overlays/*.png
ls visualizations/3d_views/*.png
ls training_output/*.png
```

### Erreur: Package manquant

MiKTeX installera automatiquement les packages manquants.
Pour TeX Live:

```bash
# Installer tous les packages nécessaires
tlmgr install collection-latexextra
tlmgr install collection-fontsrecommended
```

### Compilation trop longue

C'est normal! Avec toutes les images, la compilation peut prendre 1-2 minutes.

### Images trop grandes dans le PDF

Modifier la taille dans le `.tex`:

```latex
% Au lieu de:
\includegraphics[width=0.9\textwidth]{image.png}

% Utiliser:
\includegraphics[width=0.6\textwidth]{image.png}
```

## 📊 Contenu du Rapport

Le rapport contient **12 chapitres**:

1. Introduction
2. Contexte et Problématique
3. État de l'Art
4. Matériels et Méthodes
5. Architecture du Système
6. Pipeline de Prétraitement (à compléter)
7. Modèle de Deep Learning (à compléter)
8. Entraînement et Optimisation (à compléter)
9. Intégration PACS (à compléter)
10. Résultats et Évaluation (à compléter)
11. Discussion (à compléter)
12. Conclusion et Perspectives

**Plus 3 Annexes** avec:
- Installation et configuration
- Scripts principaux
- Résultats détaillés avec visualisations
- Guide utilisateur

## 📄 Fichiers Générés

Après compilation:

- `rapport_segmentation_pulmonaire.pdf` - **Document final** ✅
- `rapport_segmentation_pulmonaire.aux` - Fichier auxiliaire
- `rapport_segmentation_pulmonaire.log` - Log de compilation
- `rapport_segmentation_pulmonaire.toc` - Table des matières
- `rapport_segmentation_pulmonaire.lof` - Liste des figures
- `rapport_segmentation_pulmonaire.lot` - Liste des tableaux
- `rapport_segmentation_pulmonaire.out` - Liens hypertexte

## 🧹 Nettoyage

Pour supprimer les fichiers temporaires:

```bash
# Windows (PowerShell)
Remove-Item *.aux, *.log, *.toc, *.lof, *.lot, *.out, *.bbl, *.blg

# Linux/Mac
rm -f *.aux *.log *.toc *.lof *.lot *.out *.bbl *.blg

# Ou avec latexmk
latexmk -c
```

## 🎓 Format Final

- **Format**: A4 (21 x 29.7 cm)
- **Police**: 12pt
- **Marges**: 2.5 cm (toutes)
- **Pages**: ~70-100 pages (avec images)
- **Langue**: Français
- **Style**: Professionnel académique

## ✅ Checklist Avant Soumission

- [ ] Toutes les images sont présentes
- [ ] Compilation réussie sans erreurs
- [ ] Table des matières correcte
- [ ] Liste des figures/tableaux complète
- [ ] Bibliographie formatée
- [ ] Numérotation des pages continue
- [ ] Liens hypertexte fonctionnels
- [ ] Logo ENSAM ajouté (si requis)
- [ ] PDF lisible et sans artéfacts

## 📞 Support

En cas de problème:

1. Vérifier le fichier `.log` pour les erreurs
2. Consulter la documentation LaTeX: https://www.latex-project.org/help/documentation/
3. Forum LaTeX: https://tex.stackexchange.com/

---

**Créé le**: 2026-01-17  
**Projet**: RADIO_PROJET - Segmentation Multi-Organes Pulmonaire  
**Format**: LaTeX 2e
