# 🏥 Infrastructure PACS - Guide Complet

## 📋 Table des Matières
1. [Prérequis](#prérequis)
2. [Installation Docker](#installation-docker)
3. [Déploiement Orthanc](#déploiement-orthanc)
4. [Migration DICOM](#migration-dicom)
5. [Interface Web](#interface-web)
6. [Dépannage](#dépannage)

---

## 📌 Prérequis

### Logiciels Requis
- **Docker Desktop** (Windows) : [Télécharger](https://www.docker.com/products/docker-desktop/)
- **Python 3.8+** avec les packages :
  - `requests`
  - `tqdm` (optionnel, pour la barre de progression)

### Vérification
```powershell
# Vérifier Docker
docker --version
docker-compose --version

# Vérifier Python
python --version
pip list | findstr requests
```

---

## 🐳 Installation Docker

### Étape 1 : Installer Docker Desktop

1. Téléchargez Docker Desktop depuis [docker.com](https://www.docker.com/products/docker-desktop/)
2. Exécutez l'installateur
3. Redémarrez votre PC si demandé
4. Lancez Docker Desktop depuis le menu Démarrer

### Étape 2 : Vérifier que Docker fonctionne

```powershell
# Ouvrir PowerShell et tester
docker run hello-world
```

Vous devriez voir : "Hello from Docker!"

---

## 🚀 Déploiement Orthanc

### Étape 1 : Se placer dans le dossier PACS

```powershell
cd C:\Users\HP\Desktop\RADIO_PROJET\pacs
```

### Étape 2 : Créer le dossier d'import

```powershell
mkdir dicom-import -ErrorAction SilentlyContinue
```

### Étape 3 : Démarrer Orthanc

```powershell
docker-compose up -d
```

### Étape 4 : Vérifier le démarrage

```powershell
# Voir les logs
docker-compose logs -f

# Vérifier le statut
docker ps
```

### Étape 5 : Accéder à l'interface Web

Ouvrez votre navigateur à l'adresse : **http://localhost:8042**

**Identifiants :**
- Utilisateur : `admin`
- Mot de passe : `orthanc123`

---

## 📤 Migration DICOM

### Installation des Dépendances Python

```powershell
python -m pip install requests tqdm
```

### Utilisation du Script

#### Mode Normal (Migration complète)
```powershell
python migration_pacs.py --source ../DATA
```

#### Mode Dry-Run (Test sans envoi)
```powershell
python migration_pacs.py --source ../DATA --dry-run
```

#### Avec Options Personnalisées
```powershell
python migration_pacs.py \
    --source ../DATA \
    --url http://localhost:8042 \
    --user admin \
    --password orthanc123 \
    --verbose
```

### Arguments Disponibles

| Argument | Description | Défaut |
|----------|-------------|--------|
| `--source`, `-s` | Dossier contenant les DICOM | `./DATA` |
| `--url`, `-u` | URL du serveur Orthanc | `http://localhost:8042` |
| `--user` | Nom d'utilisateur | `admin` |
| `--password`, `-p` | Mot de passe | `orthanc123` |
| `--timeout`, `-t` | Timeout en secondes | `30` |
| `--dry-run`, `-n` | Mode simulation | `False` |
| `--verbose`, `-v` | Logs détaillés | `False` |

### Exemple de Sortie

```
╔══════════════════════════════════════════════════════════════════╗
║           MIGRATION DICOM → PACS ORTHANC                         ║
║           Projet Segmentation Multi-Organes                      ║
╚══════════════════════════════════════════════════════════════════╝

14:30:15 | INFO     | 📡 Test de connexion au serveur PACS...
14:30:15 | INFO     | ✅ Connexion réussie à Orthanc
14:30:15 | INFO     |    Version: 1.12.3
14:30:15 | INFO     |    Nom: RADIO_PROJET_PACS
14:30:15 | INFO     | 🔍 Scan du répertoire: C:\Users\HP\Desktop\RADIO_PROJET\DATA
14:30:18 | INFO     | 📁 12,456 fichiers DICOM trouvés
14:30:18 | INFO     | 📤 Début de la migration de 12,456 fichiers...

Migration: 100%|████████████████████████| 12456/12456 [05:23<00:00, 38.5fichier/s]

============================================================
📊 RÉSUMÉ DE LA MIGRATION
============================================================
  ✅ Succès:     12,450
  ❌ Échecs:     6
  📦 Données:    8,234.56 MB
  ⏱️  Durée:      323.4 secondes
  🚀 Débit:      38.5 fichiers/sec

📈 Statistiques Serveur PACS:
  • Patients:  158
  • Études:    158
  • Séries:    1,264
  • Instances: 12,450
============================================================
🎉 Migration terminée avec succès!
```

---

## 🖥️ Interface Web Orthanc

### Accès
URL : **http://localhost:8042**

### Fonctionnalités
- **Explorer** : Parcourir patients, études, séries
- **Upload** : Glisser-déposer des fichiers DICOM
- **Télécharger** : Exporter en DICOM ou DICOMDIR
- **Prévisualiser** : Voir les images CT
- **Rechercher** : Filtrer par nom, ID, date

### Captures d'écran

```
┌─────────────────────────────────────────────────────────────┐
│  ORTHANC - RADIO_PROJET_PACS                          [≡]  │
├─────────────────────────────────────────────────────────────┤
│  🏠 Home   📁 Upload   🔍 Query/Retrieve   ⚙️ Settings      │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  📊 Statistics                                              │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  Patients: 158    Studies: 158    Series: 1,264     │   │
│  │  Instances: 12,450   Disk: 8.2 GB                   │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  📋 Recent Patients                                         │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  LUNG1-001  │  M  │  1955  │  2 études  │  CT      │   │
│  │  LUNG1-002  │  F  │  1962  │  1 étude   │  CT      │   │
│  │  LUNG1-003  │  M  │  1948  │  1 étude   │  CT      │   │
│  │  ...                                                │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 🔧 Commandes Docker Utiles

### Gestion du Conteneur

```powershell
# Démarrer Orthanc
docker-compose up -d

# Arrêter Orthanc
docker-compose down

# Redémarrer
docker-compose restart

# Voir les logs en temps réel
docker-compose logs -f

# Voir le statut
docker-compose ps
```

### Sauvegarde et Restauration

```powershell
# Sauvegarder les données
docker run --rm -v orthanc-pacs-data:/data -v ${PWD}:/backup alpine tar czf /backup/orthanc-backup.tar.gz /data

# Restaurer les données
docker run --rm -v orthanc-pacs-data:/data -v ${PWD}:/backup alpine tar xzf /backup/orthanc-backup.tar.gz -C /
```

### Accès au Shell du Conteneur

```powershell
docker exec -it orthanc-pacs /bin/sh
```

---

## 🔍 API REST Orthanc

### Endpoints Utiles

| Méthode | Endpoint | Description |
|---------|----------|-------------|
| GET | `/system` | Informations système |
| GET | `/statistics` | Statistiques |
| GET | `/patients` | Liste des patients |
| GET | `/studies` | Liste des études |
| POST | `/instances` | Upload DICOM |
| GET | `/instances/{id}/file` | Télécharger DICOM |

### Exemples avec cURL

```powershell
# Test de connexion
curl -u admin:orthanc123 http://localhost:8042/system

# Statistiques
curl -u admin:orthanc123 http://localhost:8042/statistics

# Liste des patients
curl -u admin:orthanc123 http://localhost:8042/patients

# Upload un fichier DICOM
curl -u admin:orthanc123 -X POST -H "Content-Type: application/dicom" --data-binary @fichier.dcm http://localhost:8042/instances
```

---

## ❓ Dépannage

### Problème : Docker ne démarre pas

```powershell
# Vérifier le service Docker
Get-Service docker

# Redémarrer Docker
Restart-Service docker
```

### Problème : Port 8042 déjà utilisé

```powershell
# Trouver le processus
netstat -ano | findstr :8042

# Tuer le processus (remplacer <PID>)
taskkill /PID <PID> /F
```

### Problème : Erreur de connexion au script

1. Vérifiez que Docker est démarré
2. Vérifiez que le conteneur Orthanc tourne : `docker ps`
3. Testez l'URL dans le navigateur : http://localhost:8042

### Problème : Fichiers DICOM non détectés

Le script vérifie la signature DICOM (octets 128-132). Si vos fichiers n'ont pas cette signature standard, essayez :

```powershell
# Forcer la détection par extension
python migration_pacs.py --source ../DATA --verbose
```

---

## 📚 Ressources

- [Documentation Orthanc](https://book.orthanc-server.com/)
- [API REST Orthanc](https://api.orthanc-server.com/)
- [Docker Documentation](https://docs.docker.com/)
- [DICOM Standard](https://www.dicomstandard.org/)

---

## 📞 Support

En cas de problème, vérifiez :
1. Les logs Docker : `docker-compose logs`
2. Les logs du script : mode `--verbose`
3. La connectivité réseau : `curl http://localhost:8042/system`

---

*Document créé pour le projet Segmentation Multi-Organes Pulmonaire - ENSAM 2026*
