"""
Prépare les données pour Google Colab.

Crée une archive ZIP légère avec:
- 10 patients train (pour test rapide)
- 3 patients val
- 3 patients test
- Les fichiers splits (.txt)
- Le code nécessaire

Total: ~500 MB au lieu de 15 GB
"""

import zipfile
from pathlib import Path
import shutil


def create_colab_package():
    """Crée le package pour Colab."""
    
    print("\n" + "="*70)
    print("PRÉPARATION PACKAGE GOOGLE COLAB")
    print("="*70 + "\n")
    
    # Chemins
    normalized_dir = Path('data/processed/normalized')
    splits_dir = Path('data/processed/splits')
    output_zip = Path('colab_data.zip')
    temp_dir = Path('temp_colab')
    
    # Nettoyer
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir(parents=True)
    
    # 1. Copier splits
    print("📋 Copie des splits...")
    splits_dest = temp_dir / 'splits'
    splits_dest.mkdir(parents=True)
    
    # Lire les patients
    train_ids = open(splits_dir / 'train.txt').read().splitlines()[:10]  # 10 premiers
    val_ids = open(splits_dir / 'val.txt').read().splitlines()[:3]
    test_ids = open(splits_dir / 'test.txt').read().splitlines()[:3]
    
    # Sauvegarder les splits réduits
    with open(splits_dest / 'train.txt', 'w') as f:
        f.write('\n'.join(train_ids))
    with open(splits_dest / 'val.txt', 'w') as f:
        f.write('\n'.join(val_ids))
    with open(splits_dest / 'test.txt', 'w') as f:
        f.write('\n'.join(test_ids))
    
    print(f"   ✓ Train: {len(train_ids)} patients")
    print(f"   ✓ Val: {len(val_ids)} patients")
    print(f"   ✓ Test: {len(test_ids)} patients")
    
    # 2. Copier les données normalisées
    print("\n📦 Copie des données normalisées...")
    normalized_dest = temp_dir / 'normalized'
    normalized_dest.mkdir(parents=True)
    
    all_ids = train_ids + val_ids + test_ids
    copied = 0
    
    for patient_id in all_ids:
        ct_src = normalized_dir / f"{patient_id}_ct_normalized.nii.gz"
        mask_src = normalized_dir / f"{patient_id}_mask_normalized.nii.gz"
        
        if ct_src.exists() and mask_src.exists():
            shutil.copy2(ct_src, normalized_dest / ct_src.name)
            shutil.copy2(mask_src, normalized_dest / mask_src.name)
            copied += 1
    
    print(f"   ✓ {copied} patients copiés")
    
    # 3. Copier le code source
    print("\n📝 Copie du code source...")
    code_dest = temp_dir / 'src'
    
    # Dataset
    (code_dest / 'data').mkdir(parents=True)
    shutil.copy2('src/data/dataset.py', code_dest / 'data/dataset.py')
    
    # Model
    (code_dest / 'models').mkdir(parents=True)
    shutil.copy2('src/models/unet.py', code_dest / 'models/unet.py')
    
    # Trainer
    (code_dest / 'training').mkdir(parents=True)
    shutil.copy2('src/training/trainer.py', code_dest / 'training/trainer.py')
    
    print("   ✓ Code copié")
    
    # 4. Créer ZIP
    print(f"\n📦 Création de {output_zip}...")
    with zipfile.ZipFile(output_zip, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for file in temp_dir.rglob('*'):
            if file.is_file():
                arcname = file.relative_to(temp_dir)
                zipf.write(file, arcname)
    
    # Nettoyer
    shutil.rmtree(temp_dir)
    
    # Stats
    zip_size = output_zip.stat().st_size / (1024 * 1024)
    print(f"   ✓ Archive créée: {zip_size:.1f} MB")
    
    print("\n" + "="*70)
    print("✅ PACKAGE PRÊT POUR COLAB!")
    print("="*70)
    print(f"\n📁 Fichier: {output_zip}")
    print(f"📊 Taille: {zip_size:.1f} MB")
    print(f"👥 Patients: {copied} ({len(train_ids)} train + {len(val_ids)} val + {len(test_ids)} test)")
    print("\n🚀 PROCHAINES ÉTAPES:")
    print("   1. Ouvre colab_training.ipynb dans Google Colab")
    print("   2. Upload colab_data.zip")
    print("   3. Décompresse et lance training!")
    print()


if __name__ == '__main__':
    create_colab_package()
