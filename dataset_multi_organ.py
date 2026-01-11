"""
Dataset PyTorch pour segmentation multi-organes
Charge CT normalisé + masques multi-classes (8 labels)

Labels:
  0: Background
  1: GTV (tumeurs)
  2: PTV (zone planification) 
  3: Poumon droit
  4: Poumon gauche
  5: Coeur
  6: Oesophage
  7: Moelle épinière

Auteur: Copilot
Date: 21 Nov 2025
"""

import torch
from torch.utils.data import Dataset, DataLoader
import SimpleITK as sitk
import numpy as np
from pathlib import Path
from typing import Tuple, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MultiOrganDataset(Dataset):
    """
    Dataset PyTorch pour segmentation multi-organes
    Charge slice par slice (2D)
    """
    
    def __init__(
        self,
        patient_ids: List[str],
        ct_dir: Path,
        mask_dir: Path,
        transform=None
    ):
        """
        Args:
            patient_ids: Liste des IDs patients (ex: ['LUNG1-001', ...])
            ct_dir: Dossier des CT normalisés
            mask_dir: Dossier des masques multi-organes
            transform: Transformations PyTorch (optionnel)
        """
        self.patient_ids = patient_ids
        self.ct_dir = Path(ct_dir)
        self.mask_dir = Path(mask_dir)
        self.transform = transform
        
        # Indexer tous les slices
        self.samples = []
        self._index_dataset()
        
    def _index_dataset(self):
        """Indexe tous les slices de tous les patients"""
        logger.info(f"Indexation de {len(self.patient_ids)} patients...")
        
        for patient_id in self.patient_ids:
            ct_path = self.ct_dir / f"{patient_id}_ct_normalized.nii.gz"
            mask_path = self.mask_dir / f"{patient_id}_multi_organ.nii.gz"
            
            # Vérifier que les fichiers existent
            if not ct_path.exists():
                logger.warning(f"{patient_id}: CT introuvable - {ct_path}")
                continue
            if not mask_path.exists():
                logger.warning(f"{patient_id}: Masque introuvable - {mask_path}")
                continue
            
            # Lire le volume pour connaître le nombre de slices
            try:
                ct_img = sitk.ReadImage(str(ct_path))
                num_slices = ct_img.GetSize()[2]
                
                # Ajouter chaque slice à l'index
                for slice_idx in range(num_slices):
                    self.samples.append({
                        'patient_id': patient_id,
                        'ct_path': ct_path,
                        'mask_path': mask_path,
                        'slice_idx': slice_idx
                    })
                    
            except Exception as e:
                logger.error(f"{patient_id}: Erreur lecture - {e}")
                continue
        
        logger.info(f"Dataset indexé: {len(self.samples)} slices")
        
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retourne un slice CT + masque multi-classe
        
        Returns:
            ct_slice: Tensor (1, H, W) - CT normalisé
            mask_slice: Tensor (H, W) - Labels entiers 0-7
        """
        sample = self.samples[idx]
        
        # Charger les volumes complets
        ct_img = sitk.ReadImage(str(sample['ct_path']))
        mask_img = sitk.ReadImage(str(sample['mask_path']))
        
        # Convertir en numpy
        ct_volume = sitk.GetArrayFromImage(ct_img)  # (Z, H, W)
        mask_volume = sitk.GetArrayFromImage(mask_img)  # (Z, H, W)
        
        # Extraire le slice
        slice_idx = sample['slice_idx']
        ct_slice = ct_volume[slice_idx]  # (H, W)
        mask_slice = mask_volume[slice_idx]  # (H, W)
        
        # Convertir en torch tensors
        ct_slice = torch.from_numpy(ct_slice).float().unsqueeze(0)  # (1, H, W)
        mask_slice = torch.from_numpy(mask_slice).long()  # (H, W) - entiers
        
        # Appliquer transformations si spécifiées
        if self.transform:
            ct_slice = self.transform(ct_slice)
            # Note: Ne pas transformer le masque (labels doivent rester entiers)
        
        return ct_slice, mask_slice
    
    def get_sample_info(self, idx: int) -> dict:
        """Retourne les infos d'un sample"""
        return self.samples[idx]


def create_dataloaders(
    train_ids: List[str],
    val_ids: List[str],
    test_ids: List[str],
    ct_dir: Path,
    mask_dir: Path,
    batch_size: int = 8,
    num_workers: int = 0,
    shuffle_train: bool = True
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Crée les DataLoaders pour train/val/test
    
    Args:
        train_ids, val_ids, test_ids: Listes des IDs patients
        ct_dir: Dossier des CT normalisés
        mask_dir: Dossier des masques multi-organes
        batch_size: Taille des batches
        num_workers: Nombre de workers (0 pour Windows)
        shuffle_train: Mélanger le dataset d'entraînement
        
    Returns:
        train_loader, val_loader, test_loader
    """
    # Créer les datasets
    train_dataset = MultiOrganDataset(train_ids, ct_dir, mask_dir)
    val_dataset = MultiOrganDataset(val_ids, ct_dir, mask_dir)
    test_dataset = MultiOrganDataset(test_ids, ct_dir, mask_dir)
    
    # Créer les dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle_train,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    logger.info(f"DataLoaders créés:")
    logger.info(f"  Train: {len(train_dataset)} slices ({len(train_ids)} patients)")
    logger.info(f"  Val: {len(val_dataset)} slices ({len(val_ids)} patients)")
    logger.info(f"  Test: {len(test_dataset)} slices ({len(test_ids)} patients)")
    
    return train_loader, val_loader, test_loader


def get_class_distribution(dataset: MultiOrganDataset) -> dict:
    """
    Calcule la distribution des classes dans le dataset
    Utile pour calculer les poids de classe
    
    Returns:
        dict: {label: nombre_de_pixels}
    """
    logger.info("Calcul de la distribution des classes...")
    
    class_counts = {i: 0 for i in range(8)}  # 8 classes (0-7)
    
    for idx in range(len(dataset)):
        _, mask_slice = dataset[idx]
        
        # Compter les pixels de chaque classe
        for label in range(8):
            class_counts[label] += (mask_slice == label).sum().item()
        
        if (idx + 1) % 1000 == 0:
            logger.info(f"  Progression: {idx+1}/{len(dataset)} slices")
    
    logger.info("Distribution des classes:")
    total_pixels = sum(class_counts.values())
    for label, count in class_counts.items():
        percentage = (count / total_pixels) * 100
        logger.info(f"  Label {label}: {count:,} pixels ({percentage:.2f}%)")
    
    return class_counts


def compute_class_weights(class_counts: dict, method: str = 'inverse') -> torch.Tensor:
    """
    Calcule les poids de classe pour équilibrer la loss
    
    Args:
        class_counts: dict {label: nombre_pixels}
        method: 'inverse' ou 'sqrt_inverse'
        
    Returns:
        weights: Tensor de poids pour chaque classe
    """
    weights = []
    total_pixels = sum(class_counts.values())
    
    for label in range(8):
        count = class_counts[label]
        if count == 0:
            weights.append(0.0)
        else:
            if method == 'inverse':
                weight = total_pixels / (8 * count)
            elif method == 'sqrt_inverse':
                weight = np.sqrt(total_pixels / (8 * count))
            else:
                raise ValueError(f"Méthode inconnue: {method}")
            weights.append(weight)
    
    weights = torch.tensor(weights, dtype=torch.float32)
    
    logger.info(f"Poids de classe ({method}):")
    for label, weight in enumerate(weights):
        logger.info(f"  Label {label}: {weight:.4f}")
    
    return weights


if __name__ == "__main__":
    """Test du dataset"""
    
    BASE_DIR = Path(__file__).parent
    DATA_DIR = BASE_DIR / "DATA"
    PROCESSED_DIR = DATA_DIR / "processed"
    
    CT_DIR = PROCESSED_DIR / "normalized"
    MASK_DIR = PROCESSED_DIR / "masks_multi_organ"
    SPLITS_DIR = PROCESSED_DIR / "splits_rtstruct"
    
    # Charger les splits
    def load_split(split_file):
        with open(split_file, 'r') as f:
            return [line.strip() for line in f.readlines()]
    
    train_ids = load_split(SPLITS_DIR / "train.txt")
    val_ids = load_split(SPLITS_DIR / "val.txt")
    test_ids = load_split(SPLITS_DIR / "test.txt")
    
    logger.info(f"Splits chargés:")
    logger.info(f"  Train: {len(train_ids)} patients")
    logger.info(f"  Val: {len(val_ids)} patients")
    logger.info(f"  Test: {len(test_ids)} patients")
    
    # Créer les dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(
        train_ids, val_ids, test_ids,
        CT_DIR, MASK_DIR,
        batch_size=4,
        num_workers=0
    )
    
    # Test: charger un batch
    logger.info("\n=== TEST CHARGEMENT BATCH ===")
    ct_batch, mask_batch = next(iter(train_loader))
    
    logger.info(f"Batch CT shape: {ct_batch.shape}")  # (B, 1, H, W)
    logger.info(f"Batch mask shape: {mask_batch.shape}")  # (B, H, W)
    logger.info(f"CT range: [{ct_batch.min():.3f}, {ct_batch.max():.3f}]")
    logger.info(f"Mask unique labels: {torch.unique(mask_batch).tolist()}")
    
    # Statistiques par label dans le batch
    logger.info("\nDistribution dans le batch:")
    for label in range(8):
        count = (mask_batch == label).sum().item()
        total = mask_batch.numel()
        logger.info(f"  Label {label}: {count}/{total} pixels ({count/total*100:.2f}%)")
    
    logger.info("\n✅ Dataset multi-organes testé avec succès!")
