# 🏥 Guide de Test avec Serveur PACS Académique

## 📋 Vue d'ensemble

Ce guide explique comment tester votre pipeline de traitement DICOM avec un serveur PACS académique (Orthanc) pour valider la migration des données.

## 🎯 Objectifs des Tests

1. ✅ Vérifier la connexion au serveur PACS
2. ✅ Tester l'upload de données DICOM
3. ✅ Valider le téléchargement depuis PACS
4. ✅ Confirmer le traitement des données
5. ✅ Intégrer avec le pipeline existant

## 🛠️ Prérequis

### Option 1: Installation Locale d'Orthanc (Windows)

1. **Télécharger Orthanc:**
   - Site officiel: https://www.orthanc-server.com/download.php
   - Version recommandée: Orthanc 1.12.x (Windows 64-bit)

2. **Installation:**
   ```bash
   # Extraire l'archive
   # Ajouter le dossier au PATH système
   ```

3. **Vérifier l'installation:**
   ```bash
   Orthanc --version
   ```

### Option 2: Docker (Recommandé)

1. **Installer Docker Desktop:**
   - https://www.docker.com/products/docker-desktop/

2. **Lancer Orthanc:**
   ```bash
   docker run -p 4242:4242 -p 8042:8042 --rm jodogne/orthanc
   ```

## 🚀 Guide d'Utilisation

### Étape 1: Setup du Serveur PACS

#### Via le script Python:
```bash
python setup_orthanc_server.py
```

Options disponibles:
1. **Démarrer Orthanc (local)** - Si Orthanc est installé localement
2. **Démarrer avec Docker** - Recommandé, plus simple
3. **Créer configuration** - Génère le fichier de config
4. **Vérifier l'état** - Teste si le serveur est actif

#### Manuellement avec Docker:
```bash
# Démarrer Orthanc
docker run -d --name orthanc-test \
  -p 4242:4242 \
  -p 8042:8042 \
  -v orthanc_data:/var/lib/orthanc/db \
  jodogne/orthanc

# Vérifier les logs
docker logs orthanc-test

# Arrêter
docker stop orthanc-test
```

### Étape 2: Tester la Connexion

```bash
python test_pacs_connection.py
```

**Ce script permet de:**
- ✅ Tester la connexion HTTP au serveur
- 📊 Afficher les statistiques (patients, études, séries)
- 📋 Lister les patients existants
- ⬆️ Upload de fichiers DICOM
- 📥 Télécharger des études
- 🔍 Tester les requêtes DICOM

**Menu interactif:**
```
1. Afficher les statistiques
2. Lister les patients
3. Upload un répertoire DICOM
4. Upload depuis DATA/NSCLC-Radiomics (recommandé)
5. Tester une requête DICOM
6. Télécharger une étude
7. Test complet (upload + query)
8. Quitter
```

### Étape 3: Test Complet de Migration

```bash
python test_dicom_migration.py
```

**Tests effectués:**

1. **Test 1: Connexion PACS**
   - Vérifie que le serveur est accessible
   - Récupère les informations système

2. **Test 2: Requête Patient**
   - Liste les patients disponibles
   - Récupère les métadonnées

3. **Test 3: Téléchargement Étude**
   - Download une étude complète (ZIP)
   - Extraction des fichiers DICOM

4. **Test 4: Traitement DICOM**
   - Lecture des fichiers DICOM
   - Extraction des métadonnées
   - Validation des images

5. **Test 5: Validation Données**
   - Vérification de l'intégrité
   - Statistiques sur les données

6. **Test 6: Intégration Pipeline**
   - Vérifie la compatibilité avec les scripts existants
   - Teste les imports de modules

**Résultats:**
- Génère un rapport JSON dans `test_migration_output/results/`
- Affiche un résumé avec le taux de réussite

### Étape 4: Test Rapide

Pour un test rapide de connexion uniquement:

```bash
python test_dicom_migration.py --quick
```

## 📁 Structure des Fichiers de Test

```
test_migration_output/
├── downloaded/          # Études téléchargées depuis PACS
│   ├── study_*.zip
│   └── study_*/
├── processed/           # Métadonnées extraites
│   └── *_metadata.json
└── results/            # Rapports de test
    └── test_report_*.json
```

## 🎬 Script Batch Automatisé

Pour Windows, utilisez le script batch:

```bash
RUN_PACS_TESTS.bat
```

Menu:
```
1. Setup serveur Orthanc PACS
2. Tester la connexion PACS
3. Test complet de migration
4. Test rapide (connexion seulement)
5. Quitter
```

## 🌐 Interface Web Orthanc

Une fois le serveur démarré, accédez à l'interface web:

- **URL:** http://localhost:8042
- **Username:** orthanc (par défaut)
- **Password:** orthanc (par défaut)

**Fonctionnalités de l'interface:**
- 📊 Visualiser les patients/études
- 🔍 Rechercher dans les données
- 📥 Upload de fichiers DICOM
- 🖼️ Visualiser les images
- 📋 Explorer les métadonnées

## 🧪 Scénarios de Test Recommandés

### Test 1: Upload et Query Basique

```python
# Via test_pacs_connection.py
1. Choisir option 4 (Upload depuis NSCLC-Radiomics)
2. Limiter à 50 fichiers pour commencer
3. Vérifier les statistiques (option 1)
4. Lister les patients (option 2)
```

### Test 2: Migration Complète

```python
# Via test_dicom_migration.py
1. Lancer le test complet
2. Vérifier que tous les tests passent
3. Examiner le rapport JSON généré
```

### Test 3: Intégration avec Pipeline

```bash
# 1. Télécharger des données depuis PACS
python test_pacs_connection.py
# Choisir option 6 pour télécharger une étude

# 2. Traiter avec le pipeline existant
python extract_masks_from_rtstruct.py
python normalize_rtstruct_patients.py
```

## 📊 Exemples de Résultats Attendus

### Connexion Réussie:
```
✓ Connexion réussie!
  Serveur: RADIO_PROJET_PACS
  Version: 1.12.3
  DICOM AET: ORTHANC
```

### Upload Réussi:
```
📁 422 fichiers DICOM trouvés
🔄 Upload en cours...
   Progress: 422/422 fichiers uploadés

✓ Upload terminé: 422 réussis, 0 échoués
```

### Test Complet:
```
RAPPORT DE TEST - MIGRATION DICOM
======================================================================

Résultats: 6/6 tests réussis

Détail des tests:
  ✅ RÉUSSI - Connexion PACS
  ✅ RÉUSSI - Requête Patient
  ✅ RÉUSSI - Téléchargement Étude
  ✅ RÉUSSI - Traitement DICOM
  ✅ RÉUSSI - Validation Données
  ✅ RÉUSSI - Intégration Pipeline

🎉 SUCCÈS COMPLET - Tous les tests sont passés!
   La migration DICOM est prête pour la production.
```

## 🔧 Dépannage

### Problème: "Impossible de se connecter au serveur"

**Solution:**
```bash
# Vérifier que le serveur est démarré
docker ps  # Pour Docker
netstat -an | findstr "8042"  # Vérifier le port

# Redémarrer le serveur
python setup_orthanc_server.py
```

### Problème: "Aucun patient trouvé"

**Solution:**
```bash
# Uploader des données d'abord
python test_pacs_connection.py
# Choisir option 4 (Upload NSCLC-Radiomics)
```

### Problème: "Module pydicom introuvable"

**Solution:**
```bash
conda activate .conda
pip install pydicom requests
```

### Problème: Docker ne démarre pas

**Solution:**
```powershell
# Vérifier Docker Desktop
Get-Service -Name *docker*

# Redémarrer Docker Desktop
Restart-Service docker
```

## 📈 Métriques de Performance

Lors des tests, surveillez:

- **Temps d'upload:** ~1-2 secondes par fichier DICOM
- **Temps de download:** Dépend de la taille de l'étude
- **Mémoire utilisée:** ~500MB-1GB pour Orthanc
- **Espace disque:** Variable selon les données

## 🔒 Sécurité

**Pour un usage en production:**

1. **Activer l'authentification:**
```json
"AuthenticationEnabled": true,
"RegisteredUsers": {
  "votre_username": "votre_password_securise"
}
```

2. **Restreindre l'accès:**
```json
"RemoteAccessAllowed": false
```

3. **Utiliser HTTPS:**
```json
"SslEnabled": true,
"SslCertificate": "path/to/cert.pem"
```

## 📚 Ressources Supplémentaires

- **Documentation Orthanc:** https://book.orthanc-server.com/
- **DICOM Standard:** https://www.dicomstandard.org/
- **PyDICOM Guide:** https://pydicom.github.io/
- **Docker Hub - Orthanc:** https://hub.docker.com/r/jodogne/orthanc

## ✅ Checklist de Test

Avant de considérer les tests comme réussis:

- [ ] Serveur PACS démarre sans erreur
- [ ] Interface web accessible sur http://localhost:8042
- [ ] Upload de données DICOM réussi
- [ ] Requête de patients fonctionne
- [ ] Téléchargement d'études réussi
- [ ] Fichiers DICOM lisibles avec pydicom
- [ ] Métadonnées extraites correctement
- [ ] Intégration avec scripts existants validée
- [ ] Test complet passe à 100%
- [ ] Rapport de test généré

## 🎓 Prochaines Étapes

Après avoir validé les tests PACS:

1. **Intégrer dans le workflow:**
   - Modifier les scripts pour lire depuis PACS
   - Automatiser le download des données

2. **Optimiser les performances:**
   - Implémenter le caching
   - Paralléliser les téléchargements

3. **Déployer en production:**
   - Configurer un serveur PACS permanent
   - Mettre en place la sécurité
   - Documenter les procédures

## 📞 Support

En cas de problème:
1. Vérifier les logs: `docker logs orthanc-test`
2. Consulter la documentation
3. Vérifier les issues GitHub du projet Orthanc

---

**Créé le:** 2026-01-17  
**Version:** 1.0  
**Projet:** RADIO_PROJET - Segmentation Multi-Organes NSCLC
