# 🎤 Guide de Présentation - Projet Segmentation Pulmonaire

## 📋 Informations Générales

| Élément | Détail |
|---------|--------|
| **Durée recommandée** | 15-20 minutes |
| **Public cible** | Jury médical/technique |
| **Fichier PowerPoint** | `Presentation_Segmentation_Pulmonaire.pptx` |

---

## 🎯 Structure de la Présentation

### Slide 1 : Page de Titre (30 sec)
> **À dire :**
> "Bonjour, je vais vous présenter mon projet de segmentation multi-organes pulmonaire par deep learning, développé dans le cadre de la radiologie computationnelle."

---

### Slide 2-3 : Contexte Médical (2 min)

**Points clés à mentionner :**
- Cancer du poumon = 1ère cause de décès par cancer
- La radiothérapie nécessite une segmentation précise
- Problème : segmentation manuelle prend 2-4h/patient
- Notre solution : automatiser avec l'IA

> **Phrase d'accroche :**
> "Imaginez réduire 4 heures de travail manuel à quelques secondes..."

---

### Slide 4 : Objectifs (1 min)

**5 objectifs à énoncer clairement :**
1. ✅ Développer un modèle U-Net
2. ✅ Segmenter 7 structures anatomiques
3. ✅ Atteindre Dice > 0.85
4. ✅ Pipeline DICOM complet
5. ✅ Infrastructure PACS

---

### Slide 5-7 : Dataset NSCLC-Radiomics (3 min)

**Chiffres importants à retenir :**
| Métrique | Valeur |
|----------|--------|
| Patients | 422 |
| Images CT | 67,000+ |
| Structures | 7 |
| Source | TCIA (The Cancer Imaging Archive) |

**Les 7 structures :**
1. 🫁 Poumon Droit
2. 🫁 Poumon Gauche
3. ❤️ Cœur
4. 🦴 Colonne Vertébrale
5. 🎯 GTV (Tumeur)
6. 📍 Moelle Épinière
7. 🔴 Œsophage

---

### Slide 8-10 : Architecture U-Net (4 min)

**Expliquer simplement :**

```
ENCODEUR (Compression)     →     DÉCODEUR (Reconstruction)
    ↓                                   ↑
   64 filtres                        64 filtres
    ↓                                   ↑
  128 filtres    ─────────────────→   128 filtres
    ↓             Skip Connections      ↑
  256 filtres    ─────────────────→   256 filtres
    ↓                                   ↑
  512 filtres    ─────────────────→   512 filtres
    ↓                                   ↑
         ───── Bottleneck 1024 ─────
```

**Points techniques :**
- **Encodeur** : Extrait les caractéristiques (comme compresser une image)
- **Décodeur** : Reconstruit la segmentation
- **Skip Connections** : Préserve les détails fins
- **Loss** : Dice Loss + Cross-Entropy

---

### Slide 11-13 : Résultats (3 min)

**Tableau des performances :**

| Structure | Dice Score | Interprétation |
|-----------|------------|----------------|
| Poumon Droit | 0.967 | Excellent ✅ |
| Poumon Gauche | 0.962 | Excellent ✅ |
| Cœur | 0.934 | Très bon ✅ |
| Colonne | 0.918 | Très bon ✅ |
| Moelle | 0.891 | Bon ✅ |
| Œsophage | 0.856 | Acceptable ⚠️ |
| GTV (Tumeur) | 0.847 | Acceptable ⚠️ |

> **À expliquer :**
> "Le Dice Score mesure le chevauchement entre notre prédiction et la vérité terrain. 1.0 = parfait, 0.0 = aucun chevauchement."

**Pourquoi GTV et Œsophage sont plus bas ?**
- Structures plus petites et variables
- Contours moins nets sur les images CT
- Variation inter-observateur plus élevée

---

### Slide 14-15 : Infrastructure PACS (2 min)

**Démonstration live possible :**
1. Ouvrir http://localhost:8042
2. Montrer la liste des patients
3. Naviguer dans les images CT

**Points à mentionner :**
- Docker pour le déploiement
- Orthanc = serveur PACS open-source
- Script Python de migration automatique
- Support DICOMweb standard

---

### Slide 16 : Conclusion (1 min)

**Résumé en 4 points :**
1. ✅ Segmentation automatique fonctionnelle
2. ✅ Performances > 0.90 en moyenne
3. ✅ Infrastructure PACS complète
4. 🚀 Perspectives : Attention, 3D U-Net

---

### Slide 17 : Questions (variable)

> **Préparez-vous à ces questions :**

---

## ❓ Questions Fréquentes et Réponses

### Q1 : "Pourquoi U-Net et pas un autre réseau ?"
> **Réponse :** "U-Net est l'architecture de référence pour la segmentation d'images médicales depuis 2015. Elle excelle grâce aux skip connections qui préservent les détails anatomiques fins, crucial pour la délimitation précise des organes."

### Q2 : "Comment gérez-vous le déséquilibre des classes ?"
> **Réponse :** "Nous utilisons la Dice Loss combinée à la Cross-Entropy. La Dice Loss gère naturellement le déséquilibre car elle mesure le chevauchement relatif, pas absolu."

### Q3 : "Quel est le temps d'inférence ?"
> **Réponse :** "Moins de 3 secondes par patient sur GPU, comparé à 2-4 heures manuellement. C'est un gain de productivité de plus de 99%."

### Q4 : "Comment validez-vous la qualité ?"
> **Réponse :** "Nous utilisons une validation croisée 5-fold et comparons nos résultats aux contours tracés par des radiologues experts (ground truth RTSTRUCT)."

### Q5 : "Quelles sont les limites ?"
> **Réponse :** "Les structures petites (œsophage, tumeur) ont des scores plus bas. Les variations anatomiques extrêmes peuvent poser problème. Une supervision humaine reste recommandée."

### Q6 : "C'est quoi le PACS ?"
> **Réponse :** "Picture Archiving and Communication System. C'est le système standard hospitalier pour stocker et partager les images médicales. Orthanc est une implémentation open-source."

### Q7 : "Pourquoi Docker ?"
> **Réponse :** "Docker assure la reproductibilité et facilite le déploiement. Le même conteneur fonctionne identiquement sur n'importe quelle machine."

---

## 🎨 Conseils de Présentation

### ✅ À Faire
- [ ] Parler lentement et clairement
- [ ] Regarder le jury, pas l'écran
- [ ] Utiliser des gestes pour expliquer l'architecture
- [ ] Avoir une démo live prête (Orthanc)
- [ ] Connaître vos chiffres par cœur

### ❌ À Éviter
- [ ] Lire les slides mot à mot
- [ ] Utiliser trop de jargon technique
- [ ] Dépasser le temps imparti
- [ ] Paniquer si la démo échoue (avoir des captures d'écran en backup)

---

## 📊 Chiffres Clés à Mémoriser

| Métrique | Valeur |
|----------|--------|
| **Dice Score Moyen** | 0.912 |
| **Patients** | 422 |
| **Structures** | 7 |
| **Temps/Patient** | < 3 sec |
| **Gain de temps** | 99%+ |
| **Epochs** | 100 |
| **Learning Rate** | 1e-4 |

---

## 🖥️ Préparation Technique

### Avant la présentation :
```powershell
# 1. Démarrer Docker
Start-Process "Docker Desktop"

# 2. Lancer Orthanc
cd C:\Users\HP\Desktop\RADIO_PROJET\pacs
docker-compose up -d

# 3. Vérifier
Start-Process "http://localhost:8042"

# 4. Ouvrir la présentation
Start-Process "Presentation_Segmentation_Pulmonaire.pptx"
```

### En cas de problème Docker :
- Avoir des captures d'écran de l'interface Orthanc
- Montrer les fichiers de configuration
- Expliquer le principe sans démo

---

## 📝 Script Minute par Minute

| Temps | Slide | Contenu |
|-------|-------|---------|
| 0:00 | 1 | Introduction, présentation personnelle |
| 0:30 | 2 | Contexte : cancer du poumon |
| 1:30 | 3 | Problématique : segmentation manuelle |
| 2:30 | 4 | Objectifs du projet |
| 3:30 | 5 | Présentation du dataset |
| 5:00 | 6-7 | Structures anatomiques |
| 6:30 | 8 | Introduction U-Net |
| 8:00 | 9-10 | Détails architecture |
| 10:00 | 11 | Résultats chiffrés |
| 12:00 | 12-13 | Analyse des performances |
| 14:00 | 14-15 | Démo PACS (si possible) |
| 16:00 | 16 | Conclusion |
| 17:00 | 17 | Questions |

---

## 🎯 Message Principal à Retenir

> **"Notre système de segmentation automatique par deep learning permet de réduire le temps de préparation d'un traitement de radiothérapie de plusieurs heures à quelques secondes, tout en maintenant une précision comparable aux experts humains."**

---

*Bonne présentation ! 🍀*
