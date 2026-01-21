# 🔧 Solution Manuelle - Visualisation DICOM sans PACS

## 🎯 Problème
Docker ne démarre pas correctement → Impossible d'utiliser Orthanc

## ✅ Solution : Visualiser directement les fichiers DICOM

---

## Option 1 : RadiAnt DICOM Viewer (RECOMMANDÉ)

### Installation
1. Télécharger : https://www.radiantviewer.com/
2. Version d'essai gratuite 30 jours
3. Installation rapide (~5 min)

### Utilisation
```powershell
# Ouvrir un patient
Start-Process "C:\Users\HP\Desktop\RADIO_PROJET\DATA\NSCLC-Radiomics\LUNG1-001"
```

Puis dans RadiAnt :
- **File → Open DICOM files from folder**
- Sélectionner le dossier du patient
- Les images CT s'affichent automatiquement

### Contrôles
| Action | Raccourci |
|--------|-----------|
| Naviguer entre coupes | Molette souris |
| Zoom | Ctrl + Molette |
| Contraste/Luminosité | Clic droit + glisser |
| Mesures | Touche M |

---

## Option 2 : 3D Slicer (GRATUIT - Professionnel)

### Installation
```powershell
# Télécharger
Start-Process "https://download.slicer.org/"
```

Choisir : **Stable Release** (Windows 64-bit)

### Utilisation
1. Lancer 3D Slicer
2. **File → Add DICOM Data**
3. **Import → Choose Directory**
4. Sélectionner : `C:\Users\HP\Desktop\RADIO_PROJET\DATA\NSCLC-Radiomics\LUNG1-001`
5. Cliquer **Import**
6. Dans la liste, double-cliquer sur la série CT

### Avantages
- ✅ Visualisation 3D
- ✅ Reconstruction multiplanaire (Axial, Sagittal, Coronal)
- ✅ Mesures de volumes
- ✅ Export de captures d'écran

---

## Option 3 : Visualisation Python (Rapide)

### Script de visualisation

```powershell
# Créer le script
cd C:\Users\HP\Desktop\RADIO_PROJET
```

Créer `view_dicom.py` :

```python
import pydicom
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import sys

def view_patient(patient_folder):
    """Visualise les images CT d'un patient"""
    
    # Trouver tous les fichiers DICOM
    dicom_files = list(Path(patient_folder).rglob("*.dcm"))
    
    if not dicom_files:
        print("❌ Aucun fichier DICOM trouvé")
        return
    
    # Trier par position
    images = []
    for dcm_file in dicom_files:
        try:
            ds = pydicom.dcmread(str(dcm_file), force=True)
            if hasattr(ds, 'pixel_array'):
                images.append((ds.ImagePositionPatient[2], ds))
        except:
            continue
    
    images.sort(key=lambda x: x[0])
    
    print(f"✅ {len(images)} images CT trouvées")
    
    # Affichage interactif
    fig, ax = plt.subplots(figsize=(10, 10))
    
    current_idx = [0]  # Liste pour pouvoir modifier dans la fonction nested
    
    def show_slice(idx):
        ax.clear()
        _, ds = images[idx]
        pixels = ds.pixel_array
        
        # Windowing pour CT pulmonaire
        ax.imshow(pixels, cmap='gray', vmin=-1000, vmax=400)
        ax.set_title(f"Coupe {idx + 1}/{len(images)} - Position Z: {images[idx][0]:.1f} mm")
        ax.axis('off')
        fig.canvas.draw()
    
    def on_scroll(event):
        if event.button == 'up':
            current_idx[0] = min(current_idx[0] + 1, len(images) - 1)
        else:
            current_idx[0] = max(current_idx[0] - 1, 0)
        show_slice(current_idx[0])
    
    def on_key(event):
        if event.key == 'right':
            current_idx[0] = min(current_idx[0] + 10, len(images) - 1)
        elif event.key == 'left':
            current_idx[0] = max(current_idx[0] - 10, 0)
        elif event.key == 'q':
            plt.close()
        show_slice(current_idx[0])
    
    fig.canvas.mpl_connect('scroll_event', on_scroll)
    fig.canvas.mpl_connect('key_press_event', on_key)
    
    show_slice(0)
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) > 1:
        patient = sys.argv[1]
    else:
        patient = "DATA/NSCLC-Radiomics/LUNG1-001"
    
    print(f"📂 Ouverture : {patient}")
    view_patient(patient)
    print("\n💡 Utilisez la molette pour naviguer")
    print("💡 Flèches ← → pour avancer/reculer de 10 coupes")
    print("💡 Touche 'Q' pour quitter")
```

### Lancement
```powershell
# Voir LUNG1-001
python view_dicom.py

# Ou un autre patient
python view_dicom.py DATA/NSCLC-Radiomics/LUNG1-005
```

---

## Option 4 : Explorateur Windows

### Navigation simple
```powershell
# Ouvrir le dossier des patients
explorer C:\Users\HP\Desktop\RADIO_PROJET\DATA\NSCLC-Radiomics
```

Structure :
```
LUNG1-001/
├── 09-18-2008-StudyID-12345/
│   ├── 3.000000-CT-12345/
│   │   ├── 1-001.dcm
│   │   ├── 1-002.dcm
│   │   └── ...
│   └── 4.000000-RTSTRUCT-67890/
│       └── 1-1.dcm
```

- Dossier **CT** : Images scanner
- Dossier **RTSTRUCT** : Contours de segmentation

---

## 📊 Comparaison des Options

| Solution | Gratuit | Installation | Facilité | 3D |
|----------|---------|--------------|----------|-----|
| **RadiAnt** | 30 jours | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ✅ |
| **3D Slicer** | ✅ | ⭐⭐ | ⭐⭐⭐ | ✅✅✅ |
| **Python** | ✅ | ⭐⭐⭐⭐ | ⭐⭐⭐ | ❌ |
| **Explorateur** | ✅ | ⭐⭐⭐⭐⭐ | ⭐ | ❌ |

---

## 🎯 Recommandation

### Pour la présentation :
👉 **RadiAnt DICOM Viewer**
- Interface professionnelle
- Démarrage rapide
- Contrôles intuitifs

### Pour l'analyse :
👉 **3D Slicer**
- Visualisation 3D impressionnante
- Outils de mesure avancés
- Screenshots de qualité

---

## 🔄 Quand Docker fonctionnera

Une fois Docker réparé :
```powershell
cd C:\Users\HP\Desktop\RADIO_PROJET\pacs

# Redémarrer complètement Windows
Restart-Computer

# Après redémarrage
docker-compose up -d
Start-Process "http://localhost:8042"
```

---

## 💡 Pour la Présentation

Si Docker ne fonctionne pas le jour J :

### Plan B : Screenshots
```powershell
# Prendre des captures d'écran avec RadiAnt/Slicer
# Les intégrer dans le PowerPoint
```

### Plan C : Vidéo
Enregistrer une courte démo vidéo à l'avance :
- Ouverture d'un patient
- Navigation dans les coupes CT
- Zoom sur la tumeur

---

*Docker est capricieux, mais les images DICOM sont toujours là ! 📁*
