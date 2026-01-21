# 🏥 Guide d'utilisation - Orthanc PACS

## Accès à l'interface

- **URL** : http://localhost:8042
- **Utilisateur** : `admin`
- **Mot de passe** : `orthanc123`

---

## 📋 Navigation dans l'interface

### Page d'accueil (Lookup)

| Bouton | Description |
|--------|-------------|
| **All patients** | Liste tous les patients |
| **All studies** | Liste toutes les études |
| **Do lookup** | Recherche avec filtres |
| **Upload** | Importer des fichiers DICOM |

---

## 🔍 Visualiser les images CT

### Étape 1 : Sélectionner un patient
1. Cliquez sur **"All patients"**
2. Cliquez sur le nom du patient (ex: **LUNG1-001**)

### Étape 2 : Ouvrir une étude
1. Vous verrez la liste des études (dates en bleu)
2. Cliquez sur la **date de l'étude** (ex: Thursday, September 18, 2008)

### Étape 3 : Choisir une série
Vous verrez les séries disponibles :
- **CT** : Images scanner (ce qu'on veut voir)
- **RTSTRUCT** : Contours de segmentation
- **RTPLAN** : Plan de traitement

➡️ Cliquez sur la série **CT**

### Étape 4 : Visualiser les images
- Les images s'affichent une par une
- Utilisez les **flèches ◄ ►** pour naviguer entre les coupes
- Ou utilisez la **molette de la souris**

---

## 🖼️ Contrôles de visualisation

| Action | Comment faire |
|--------|---------------|
| **Coupe suivante** | Flèche droite → ou molette |
| **Coupe précédente** | Flèche gauche ← ou molette |
| **Zoom** | Ctrl + molette |
| **Contraste/Luminosité** | Clic droit + glisser |
| **Déplacer l'image** | Clic gauche + glisser |

---

## 📥 Télécharger des images

### Pour un patient complet :
1. Sélectionnez le patient
2. Cliquez sur **"Download ZIP"** (colonne gauche)

### Pour une série spécifique :
1. Naviguez jusqu'à la série
2. Cliquez sur **"Download ZIP"**

---

## 🔎 Recherche de patients

### Recherche simple :
1. Dans la page **Lookup**, remplissez un champ :
   - **Patient ID** : ex. LUNG1-001
   - **Patient Name** : nom du patient
2. Cliquez sur **"Do lookup"**

### Recherche avec wildcards :
- `LUNG1-*` : tous les patients LUNG1
- `*001*` : patients contenant 001

---

## 📊 Informations DICOM

### Sur un patient :
- **PatientID** : Identifiant unique
- **PatientSex** : M (masculin) / F (féminin)
- **PatientBirthDate** : Date de naissance

### Sur une étude :
- **StudyDate** : Date de l'examen
- **AccessionNumber** : Numéro d'accès
- **StudyInstanceUID** : Identifiant unique de l'étude

### Sur une série :
- **Modality** : Type (CT, MR, RTSTRUCT...)
- **SeriesDescription** : Description
- **NumberOfFrames** : Nombre d'images

---

## ⚙️ Fonctionnalités avancées

### Anonymiser un patient :
1. Sélectionnez le patient
2. Cliquez sur **"Anonymize"**
3. Les données personnelles seront supprimées

### Envoyer vers une autre modalité :
1. Sélectionnez le patient/étude/série
2. Cliquez sur **"Send to remote modality"**
3. Choisissez la destination

### Ajouter des labels :
1. Sélectionnez un patient
2. Cliquez sur **"Add label"**
3. Entrez un label (ex: "segmenté", "à vérifier")

---

## 🔗 API REST (pour développeurs)

```bash
# Liste des patients
curl -u admin:orthanc123 http://localhost:8042/patients

# Statistiques du serveur
curl -u admin:orthanc123 http://localhost:8042/statistics

# Télécharger une instance DICOM
curl -u admin:orthanc123 http://localhost:8042/instances/{id}/file -o image.dcm
```

---

## 🛠️ Dépannage

| Problème | Solution |
|----------|----------|
| Page ne charge pas | Vérifier que Docker est lancé |
| Erreur 401 | Vérifier login/mot de passe |
| Images floues | Ajuster le contraste (clic droit) |
| Pas de patients | Importer des fichiers DICOM |

### Commandes utiles :

```powershell
# Vérifier si Orthanc tourne
docker ps

# Redémarrer Orthanc
cd C:\Users\HP\Desktop\RADIO_PROJET\pacs
docker-compose restart

# Voir les logs
docker logs orthanc-pacs
```

---

## 📁 Structure des données NSCLC

```
Patient (LUNG1-XXX)
└── Étude (date de l'examen)
    ├── Série CT (images scanner ~100-200 coupes)
    ├── Série RTSTRUCT (contours de segmentation)
    └── Série RTPLAN (plan de traitement, si disponible)
```

---

## 🎯 Workflow typique

1. **Importer** les données DICOM
2. **Visualiser** les images CT
3. **Vérifier** les contours RTSTRUCT
4. **Exporter** pour analyse/segmentation
5. **Sauvegarder** (Download ZIP)

---

*Guide créé pour le projet RADIO_PROJET - Segmentation pulmonaire*
