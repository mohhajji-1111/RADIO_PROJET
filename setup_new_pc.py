"""
=============================================================================
SCRIPT D'INSTALLATION AUTOMATIQUE - Nouveau PC
=============================================================================
Ce script configure automatiquement l'environnement pour le projet
NSCLC Multi-Organ Segmentation sur un nouveau PC.

Usage:
    python setup_new_pc.py

Auteur: Projet NSCLC Radiomics
=============================================================================
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(cmd, description, check=True):
    """Exécute une commande et affiche le résultat."""
    print(f"\n{'='*60}")
    print(f"📦 {description}")
    print(f"{'='*60}")
    print(f"Commande: {cmd}\n")
    
    try:
        result = subprocess.run(cmd, shell=True, check=check, capture_output=False)
        if result.returncode == 0:
            print(f"✅ {description} - OK")
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"❌ Erreur: {e}")
        return False

def check_cuda():
    """Vérifie si CUDA est disponible."""
    try:
        result = subprocess.run(
            'nvidia-smi', 
            shell=True, 
            capture_output=True, 
            text=True
        )
        if result.returncode == 0:
            print("✅ GPU NVIDIA détecté!")
            print(result.stdout[:500])  # Afficher les premières lignes
            return True
    except:
        pass
    print("⚠️ Pas de GPU NVIDIA détecté - Installation CPU")
    return False

def get_cuda_version():
    """Détecte la version de CUDA installée."""
    try:
        result = subprocess.run(
            'nvcc --version', 
            shell=True, 
            capture_output=True, 
            text=True
        )
        if 'release 12' in result.stdout:
            return 'cu121'
        elif 'release 11.8' in result.stdout:
            return 'cu118'
        elif 'release 11' in result.stdout:
            return 'cu118'
    except:
        pass
    return None

def main():
    print("""
╔══════════════════════════════════════════════════════════════════╗
║     🏥 NSCLC Multi-Organ Segmentation - Setup Nouveau PC 🏥      ║
╚══════════════════════════════════════════════════════════════════╝
    """)
    
    project_dir = Path(__file__).parent
    print(f"📁 Dossier du projet: {project_dir}")
    
    # Vérifier les données
    data_dir = project_dir / "DATA" / "processed" / "normalized"
    ct_files = list(data_dir.glob("*_ct_normalized.nii.gz")) if data_dir.exists() else []
    
    print(f"\n📊 Vérification des données:")
    if len(ct_files) > 0:
        print(f"   ✅ {len(ct_files)} patients trouvés")
    else:
        print(f"   ❌ ERREUR: Données non trouvées dans {data_dir}")
        print(f"   ⚠️ Copiez le dossier DATA/processed/normalized/ depuis l'ancien PC")
        return
    
    # Détecter GPU
    print(f"\n🔍 Détection du matériel:")
    has_gpu = check_cuda()
    cuda_version = get_cuda_version() if has_gpu else None
    
    # Choix de l'installation
    print(f"\n{'='*60}")
    print("📋 OPTIONS D'INSTALLATION")
    print(f"{'='*60}")
    print("1. Installation avec GPU (CUDA 11.8)")
    print("2. Installation avec GPU (CUDA 12.1)")
    print("3. Installation CPU seulement")
    print("4. Annuler")
    
    # Suggestion automatique
    if cuda_version:
        suggested = "1" if cuda_version == "cu118" else "2"
        print(f"\n💡 Suggestion: Option {suggested} (CUDA détecté)")
    elif has_gpu:
        print(f"\n💡 Suggestion: Option 1 ou 2 (GPU détecté, CUDA non détecté)")
    else:
        print(f"\n💡 Suggestion: Option 3 (Pas de GPU)")
    
    choice = input("\nVotre choix (1/2/3/4): ").strip()
    
    if choice == "4":
        print("Installation annulée.")
        return
    
    # Déterminer l'URL PyTorch
    if choice == "1":
        pytorch_url = "https://download.pytorch.org/whl/cu118"
        device_info = "GPU CUDA 11.8"
    elif choice == "2":
        pytorch_url = "https://download.pytorch.org/whl/cu121"
        device_info = "GPU CUDA 12.1"
    else:
        pytorch_url = "https://download.pytorch.org/whl/cpu"
        device_info = "CPU"
    
    print(f"\n🚀 Installation pour: {device_info}")
    
    # Installation des packages
    print("\n" + "="*60)
    print("📦 INSTALLATION DES DÉPENDANCES")
    print("="*60)
    
    # PyTorch
    success = run_command(
        f'{sys.executable} -m pip install torch torchvision torchaudio --index-url {pytorch_url}',
        "Installation de PyTorch"
    )
    
    if not success:
        print("❌ Erreur lors de l'installation de PyTorch")
        return
    
    # Autres dépendances
    packages = [
        "SimpleITK",
        "tqdm",
        "opencv-python",
        "matplotlib",
        "numpy",
    ]
    
    run_command(
        f'{sys.executable} -m pip install {" ".join(packages)}',
        "Installation des autres dépendances"
    )
    
    # Vérification finale
    print("\n" + "="*60)
    print("🔍 VÉRIFICATION DE L'INSTALLATION")
    print("="*60)
    
    verification_code = '''
import torch
import SimpleITK as sitk
import cv2
import matplotlib
from pathlib import Path

print("\\n📊 Résultats:")
print(f"   PyTorch version: {torch.__version__}")
print(f"   CUDA disponible: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
    mem = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"   Mémoire GPU: {mem:.1f} GB")

print(f"   SimpleITK: OK")
print(f"   OpenCV: OK")
print(f"   Matplotlib: OK")
print("\\n✅ Installation réussie!")
'''
    
    subprocess.run([sys.executable, '-c', verification_code])
    
    # Mise à jour de la config pour GPU
    if choice in ["1", "2"]:
        print("\n" + "="*60)
        print("⚙️ OPTIMISATION POUR GPU")
        print("="*60)
        
        config_update = '''
# Configuration optimisée pour GPU
# Modifiez incremental_training.py avec ces valeurs:

CONFIG = {
    'patients_per_batch': 20,     # Plus de patients
    'epochs_per_batch': 5,        # Plus d'epochs
    'total_rounds': 3,            # Moins de rounds
    'batch_size': 8,              # Plus grand batch (8-16 selon GPU)
    'learning_rate': 1e-4,
    'num_workers': 4,             # Parallélisation
    'num_classes': 8,
    'device': 'cuda',
}
'''
        print(config_update)
    
    # Instructions finales
    print("\n" + "="*60)
    print("🎉 INSTALLATION TERMINÉE!")
    print("="*60)
    print("""
Pour lancer le training:

    1. Ouvrez PowerShell
    2. Naviguez vers le projet:
       cd """ + str(project_dir) + """
    
    3. Lancez le training:
       python incremental_training.py

Le training reprendra automatiquement si un checkpoint existe.
    """)

if __name__ == "__main__":
    main()
