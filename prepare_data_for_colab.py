"""
Préparer les données pour Google Colab Training.

Compresse les données normalisées + splits pour upload sur Colab.
"""

import zipfile
from pathlib import Path
from tqdm import tqdm


def create_colab_package():
    """Créer package de données pour Colab."""
    
    print("\n" + "="*70)
    print("PRÉPARATION DONNÉES POUR GOOGLE COLAB")
    print("="*70 + "\n")
    
    # Chemins
    normalized_dir = Path('data/processed/normalized')
    splits_dir = Path('data/processed/splits')
    output_file = Path('colab_data.zip')
    
    # Vérifier que les données existent
    if not normalized_dir.exists():
        print("❌ Erreur: data/processed/normalized/ n'existe pas!")
        return
    
    if not splits_dir.exists():
        print("❌ Erreur: data/processed/splits/ n'existe pas!")
        return
    
    # Compter les fichiers
    ct_files = list(normalized_dir.glob('*_ct_normalized.nii.gz'))
    mask_files = list(normalized_dir.glob('*_mask_normalized.nii.gz'))
    
    print(f"📊 Fichiers trouvés:")
    print(f"   CT scans: {len(ct_files)}")
    print(f"   Masks: {len(mask_files)}")
    print()
    
    # Créer l'archive
    print("📦 Création de l'archive ZIP...")
    print(f"   Destination: {output_file}")
    print()
    
    with zipfile.ZipFile(output_file, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # Ajouter fichiers splits
        print("   Adding splits...")
        for split_file in splits_dir.glob('*.txt'):
            arcname = f'splits/{split_file.name}'
            zipf.write(split_file, arcname)
        
        # Ajouter CT scans
        print("   Adding CT scans...")
        for ct_file in tqdm(ct_files, desc="   CT"):
            arcname = f'normalized/{ct_file.name}'
            zipf.write(ct_file, arcname)
        
        # Ajouter masks
        print("   Adding masks...")
        for mask_file in tqdm(mask_files, desc="   Masks"):
            arcname = f'normalized/{mask_file.name}'
            zipf.write(mask_file, arcname)
    
    # Taille finale
    size_mb = output_file.stat().st_size / (1024 * 1024)
    
    print("\n" + "="*70)
    print("✅ PACKAGE CRÉÉ AVEC SUCCÈS!")
    print("="*70)
    print(f"\n📦 Fichier: {output_file}")
    print(f"📏 Taille: {size_mb:.1f} MB")
    print(f"📊 Contenu:")
    print(f"   • {len(ct_files)} CT scans")
    print(f"   • {len(mask_files)} masks")
    print(f"   • 3 fichiers splits (train/val/test)")
    
    print("\n🚀 PROCHAINES ÉTAPES:")
    print("   1. Upload colab_data.zip sur Google Drive")
    print("   2. Ouvrir colab_training.ipynb dans Colab")
    print("   3. Monter Google Drive")
    print("   4. Extraire le ZIP dans Colab")
    print("   5. Lancer l'entraînement!")
    print()


if __name__ == '__main__':
    create_colab_package()
