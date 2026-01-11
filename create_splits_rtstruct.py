"""
Script pour créer les splits train/val/test avec masques RTSTRUCT
"""

import numpy as np
from pathlib import Path
import random

def create_splits(
    data_dir: Path,
    output_dir: Path,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42
):
    """Crée les fichiers de split train/val/test."""
    
    print("="*70)
    print("CREATION DES SPLITS (TRAIN/VAL/TEST)")
    print("="*70)
    
    # Lister tous les patients normalisés
    ct_files = sorted(data_dir.glob("LUNG1-*_ct_normalized.nii.gz"))
    patient_ids = [f.name.replace('_ct_normalized.nii.gz', '') for f in ct_files]
    
    print(f"\nPatients normalises: {len(patient_ids)}")
    
    # Mélanger avec seed fixe
    random.seed(seed)
    random.shuffle(patient_ids)
    
    # Calculer tailles des splits
    n_total = len(patient_ids)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    n_test = n_total - n_train - n_val
    
    # Créer les splits
    train_ids = patient_ids[:n_train]
    val_ids = patient_ids[n_train:n_train + n_val]
    test_ids = patient_ids[n_train + n_val:]
    
    print(f"\nSplits:")
    print(f"  Train: {len(train_ids)} patients ({len(train_ids)/n_total*100:.1f}%)")
    print(f"  Val:   {len(val_ids)} patients ({len(val_ids)/n_total*100:.1f}%)")
    print(f"  Test:  {len(test_ids)} patients ({len(test_ids)/n_total*100:.1f}%)")
    
    # Créer dossier de sortie
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Sauvegarder les splits
    with open(output_dir / 'train.txt', 'w') as f:
        f.write('\n'.join(train_ids))
    
    with open(output_dir / 'val.txt', 'w') as f:
        f.write('\n'.join(val_ids))
    
    with open(output_dir / 'test.txt', 'w') as f:
        f.write('\n'.join(test_ids))
    
    print(f"\nFichiers sauvegardes dans: {output_dir}")
    print("  - train.txt")
    print("  - val.txt")
    print("  - test.txt")
    
    print("\n" + "="*70)
    print("SPLITS CREES AVEC SUCCES!")
    print("="*70)

if __name__ == "__main__":
    data_dir = Path("DATA/processed/normalized_rtstruct")
    output_dir = Path("DATA/processed/splits_rtstruct")
    
    create_splits(data_dir, output_dir)
