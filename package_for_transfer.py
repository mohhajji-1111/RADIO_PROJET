"""
=============================================================================
SCRIPT DE PACKAGING - Préparer le projet pour un nouveau PC
=============================================================================
Ce script crée une archive ZIP contenant tout le nécessaire pour
transférer le projet sur un nouveau PC.

Usage:
    python package_for_transfer.py

Auteur: Projet NSCLC Radiomics
=============================================================================
"""

import os
import shutil
import zipfile
from pathlib import Path
from datetime import datetime

def get_folder_size(path):
    """Calcule la taille d'un dossier."""
    total = 0
    for entry in os.scandir(path):
        if entry.is_file():
            total += entry.stat().st_size
        elif entry.is_dir():
            total += get_folder_size(entry.path)
    return total

def format_size(size_bytes):
    """Formate la taille en unité lisible."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} TB"

def main():
    print("""
╔══════════════════════════════════════════════════════════════════╗
║         📦 Packaging du Projet pour Nouveau PC 📦                ║
╚══════════════════════════════════════════════════════════════════╝
    """)
    
    project_dir = Path(__file__).parent
    
    # Fichiers Python essentiels
    essential_files = [
        "incremental_training.py",
        "setup_new_pc.py",
        "START_TRAINING.bat",
        "GUIDE_NOUVEAU_PC.md",
        "requirements.txt",
        "dataset_multi_organ.py",
        "unet_multi_organ.py",
        "train_multi_organ.py",
    ]
    
    # Fichiers optionnels mais utiles
    optional_files = [
        "PROJECT_README.md",
        "README.md",
        "EXPLICATION_PROJET.txt",
        "colab_training.ipynb",
        "kaggle_training_corrected.ipynb",
    ]
    
    # Dossiers de données
    data_folder = project_dir / "DATA" / "processed" / "normalized"
    checkpoint_folder = project_dir / "training_output" / "checkpoints"
    
    print("📊 Analyse du projet...")
    print("-" * 50)
    
    # Calculer les tailles
    files_to_copy = []
    total_size = 0
    
    # Fichiers essentiels
    print("\n📁 Fichiers essentiels:")
    for f in essential_files:
        fpath = project_dir / f
        if fpath.exists():
            size = fpath.stat().st_size
            total_size += size
            files_to_copy.append(("files", fpath))
            print(f"   ✅ {f} ({format_size(size)})")
        else:
            print(f"   ❌ {f} (non trouvé)")
    
    # Fichiers optionnels
    print("\n📁 Fichiers optionnels:")
    for f in optional_files:
        fpath = project_dir / f
        if fpath.exists():
            size = fpath.stat().st_size
            total_size += size
            files_to_copy.append(("files", fpath))
            print(f"   ✅ {f} ({format_size(size)})")
    
    # Dossier de données
    print("\n📁 Données (IMPORTANT):")
    if data_folder.exists():
        ct_files = list(data_folder.glob("*_ct_normalized.nii.gz"))
        mask_files = list(data_folder.glob("*_mask_normalized.nii.gz"))
        data_size = get_folder_size(data_folder)
        print(f"   ✅ {len(ct_files)} fichiers CT")
        print(f"   ✅ {len(mask_files)} fichiers masks")
        print(f"   📦 Taille totale: {format_size(data_size)}")
        total_size += data_size
    else:
        print(f"   ❌ Dossier de données non trouvé!")
        return
    
    # Checkpoints
    print("\n📁 Checkpoints (pour reprendre le training):")
    if checkpoint_folder.exists():
        for cp in checkpoint_folder.glob("*"):
            size = cp.stat().st_size if cp.is_file() else get_folder_size(cp)
            print(f"   ✅ {cp.name} ({format_size(size)})")
            total_size += size
    else:
        print("   ⚠️ Pas de checkpoints (training non commencé)")
    
    print("\n" + "=" * 50)
    print(f"📦 TAILLE TOTALE ESTIMÉE: {format_size(total_size)}")
    print("=" * 50)
    
    # Options de transfert
    print("""
🔄 OPTIONS DE TRANSFERT:

1. USB/Disque externe (Recommandé pour les données volumineuses)
   → Copiez manuellement le dossier RADIO_PROJET

2. Créer un ZIP (sans données - juste le code)
   → Rapide, ~1 MB

3. Créer un ZIP complet (avec données)
   → Long, ~10+ GB

4. Voir la liste des fichiers à copier
   → Instructions manuelles

5. Annuler
""")
    
    choice = input("Votre choix (1/2/3/4/5): ").strip()
    
    if choice == "5":
        print("Annulé.")
        return
    
    if choice == "1":
        print_manual_instructions(project_dir)
        
    elif choice == "2":
        create_code_only_zip(project_dir, files_to_copy)
        
    elif choice == "3":
        create_full_zip(project_dir, data_folder, files_to_copy, checkpoint_folder)
        
    elif choice == "4":
        print_file_list(project_dir, data_folder)


def print_manual_instructions(project_dir):
    """Affiche les instructions pour copie manuelle."""
    print("""
╔══════════════════════════════════════════════════════════════════╗
║              📋 INSTRUCTIONS DE COPIE MANUELLE                   ║
╚══════════════════════════════════════════════════════════════════╝

1. Copiez TOUT le dossier suivant sur une clé USB:
   """ + str(project_dir) + """

2. Sur le nouveau PC, collez le dossier où vous voulez
   (ex: C:\\Projets\\RADIO_PROJET)

3. Ouvrez PowerShell et exécutez:
   cd C:\\Projets\\RADIO_PROJET
   python setup_new_pc.py

4. Suivez les instructions du script d'installation

5. Lancez le training:
   python incremental_training.py

C'est tout! 🎉
""")


def create_code_only_zip(project_dir, files_to_copy):
    """Crée un ZIP avec le code seulement."""
    zip_name = f"RADIO_PROJET_code_{datetime.now().strftime('%Y%m%d')}.zip"
    zip_path = project_dir / zip_name
    
    print(f"\n📦 Création de {zip_name}...")
    
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        for category, fpath in files_to_copy:
            if fpath.is_file():
                zf.write(fpath, fpath.name)
                print(f"   ✅ {fpath.name}")
    
    print(f"\n✅ ZIP créé: {zip_path}")
    print(f"   Taille: {format_size(zip_path.stat().st_size)}")
    print(f"\n⚠️ N'oubliez pas de copier aussi le dossier DATA/processed/normalized/")


def create_full_zip(project_dir, data_folder, files_to_copy, checkpoint_folder):
    """Crée un ZIP complet avec données."""
    zip_name = f"RADIO_PROJET_complet_{datetime.now().strftime('%Y%m%d')}.zip"
    zip_path = project_dir / zip_name
    
    print(f"\n📦 Création de {zip_name}...")
    print("   ⚠️ Cela peut prendre plusieurs minutes...")
    
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        # Fichiers de code
        for category, fpath in files_to_copy:
            if fpath.is_file():
                zf.write(fpath, fpath.name)
                print(f"   ✅ {fpath.name}")
        
        # Données
        print("\n   📊 Ajout des données...")
        for f in data_folder.glob("*.nii.gz"):
            arcname = f"DATA/processed/normalized/{f.name}"
            zf.write(f, arcname)
        print(f"   ✅ Données ajoutées")
        
        # Checkpoints
        if checkpoint_folder.exists():
            print("   📊 Ajout des checkpoints...")
            for f in checkpoint_folder.glob("*"):
                if f.is_file():
                    arcname = f"training_output/checkpoints/{f.name}"
                    zf.write(f, arcname)
            print(f"   ✅ Checkpoints ajoutés")
    
    print(f"\n✅ ZIP complet créé: {zip_path}")
    print(f"   Taille: {format_size(zip_path.stat().st_size)}")


def print_file_list(project_dir, data_folder):
    """Affiche la liste détaillée des fichiers."""
    print("""
╔══════════════════════════════════════════════════════════════════╗
║              📋 LISTE DES FICHIERS À COPIER                      ║
╚══════════════════════════════════════════════════════════════════╝

📁 FICHIERS ESSENTIELS (à copier):
""")
    essential = [
        "incremental_training.py  → Script de training principal",
        "setup_new_pc.py          → Installation automatique",
        "START_TRAINING.bat       → Lancement rapide (double-clic)",
        "GUIDE_NOUVEAU_PC.md      → Guide complet",
        "requirements.txt         → Dépendances Python",
    ]
    for f in essential:
        print(f"   • {f}")
    
    print("""
📁 DOSSIER DE DONNÉES (OBLIGATOIRE):
   """ + str(data_folder) + """
   → Contient tous les fichiers .nii.gz des patients

📁 CHECKPOINTS (OPTIONNEL - pour reprendre):
   """ + str(project_dir / "training_output" / "checkpoints") + """
   → Permet de reprendre le training là où il s'est arrêté
""")


if __name__ == "__main__":
    main()
