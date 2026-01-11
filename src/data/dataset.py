"""
PyTorch Dataset and DataLoader for NSCLC Lung Tumor Segmentation (Phase 4).

This module provides:
1. Custom PyTorch Dataset for loading CT-mask pairs
2. Data augmentation transforms
3. DataLoader configuration
4. Memory-efficient caching strategies

Author: GitHub Copilot
Date: 2025
"""

import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Callable
import json

import numpy as np
import SimpleITK as sitk
import torch
from torch.utils.data import Dataset, DataLoader


class NSCLCDataset(Dataset):
    """
    PyTorch Dataset for NSCLC lung tumor segmentation.
    
    Loads preprocessed CT-mask pairs from Phase 3 normalized data.
    Supports data augmentation and flexible indexing.
    """
    
    def __init__(
        self,
        data_root: str,
        split: str = 'train',
        transform: Optional[Callable] = None,
        cache_data: bool = False,
        slice_wise: bool = True
    ):
        """
        Initialize NSCLC dataset.
        
        Args:
            data_root: Root directory with normalized data (e.g., 'DATA/processed/normalized_rtstruct')
            split: Data split ('train', 'val', or 'test')
            transform: Optional transform to apply to data
            cache_data: If True, cache all data in memory
            slice_wise: If True, treat each 2D slice as separate sample
        """
        self.data_root = Path(data_root)
        self.split = split
        self.transform = transform
        self.cache_data = cache_data
        self.slice_wise = slice_wise
        
        # Load patient IDs from split file
        split_file = self.data_root.parent / 'splits_rtstruct' / f'{split}.txt'
        if not split_file.exists():
            raise ValueError(f"Split file not found: {split_file}")
        
        with open(split_file, 'r') as f:
            patient_ids = [line.strip() for line in f if line.strip()]
        
        # Build patient file paths
        self.patient_files = []
        for patient_id in patient_ids:
            img_path = self.data_root / f"{patient_id}_ct_normalized.nii.gz"
            if img_path.exists():
                self.patient_files.append(img_path)
            else:
                print(f"Warning: CT file not found for {patient_id}")
        
        if len(self.patient_files) == 0:
            raise ValueError(f"No data found for {split} split")
        
        print(f"Loaded {split} split: {len(self.patient_files)} patients")
        
        # Build sample index (patient or slice level)
        self.samples = self._build_sample_index()
        
        # Cache for loaded data
        self.cache = {} if cache_data else None
        
        if cache_data:
            print("Caching all data in memory...")
            self._cache_all_data()
    
    def _build_sample_index(self) -> List[Dict]:
        """
        Build index of all samples (patients or slices).
        
        Returns:
            List of sample dictionaries with paths and indices
        """
        samples = []
        
        for img_path in self.patient_files:
            # Extract patient ID from filename (e.g., LUNG1-001_ct_normalized.nii.gz -> LUNG1-001)
            patient_id = img_path.stem.replace('_ct_normalized.nii', '').replace('.nii', '')
            mask_path = self.data_root / f"{patient_id}_mask_normalized.nii.gz"
            
            if not mask_path.exists():
                print(f"Warning: No mask found for {patient_id}")
                continue
            
            if self.slice_wise:
                # Load image to get number of slices
                img = sitk.ReadImage(str(img_path))
                num_slices = img.GetSize()[2]  # Depth dimension
                
                # Create entry for each slice
                for slice_idx in range(num_slices):
                    samples.append({
                        'patient_id': patient_id,
                        'image_path': str(img_path),
                        'mask_path': str(mask_path),
                        'slice_idx': slice_idx,
                        'num_slices': num_slices
                    })
            else:
                # Whole volume as single sample
                samples.append({
                    'patient_id': patient_id,
                    'image_path': str(img_path),
                    'mask_path': str(mask_path),
                    'slice_idx': None,
                    'num_slices': None
                })
        
        print(f"Built index: {len(samples)} samples")
        return samples
    
    def _cache_all_data(self):
        """Pre-load all data into memory for faster training."""
        for idx in range(len(self.samples)):
            sample = self.samples[idx]
            key = f"{sample['patient_id']}_{sample['slice_idx']}"
            
            # Load and cache
            img, mask = self._load_sample(sample)
            self.cache[key] = (img, mask)
        
        print(f"Cached {len(self.cache)} samples")
    
    def _load_sample(self, sample: Dict) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load CT image and mask for a sample.
        
        Args:
            sample: Sample dictionary with paths and indices
            
        Returns:
            Tuple of (image, mask) as numpy arrays
        """
        # Load 3D volumes
        img_sitk = sitk.ReadImage(sample['image_path'])
        mask_sitk = sitk.ReadImage(sample['mask_path'])
        
        # Convert to numpy
        img_array = sitk.GetArrayFromImage(img_sitk)  # (depth, height, width)
        mask_array = sitk.GetArrayFromImage(mask_sitk)
        
        if self.slice_wise:
            # Extract single slice
            slice_idx = sample['slice_idx']
            img_slice = img_array[slice_idx]  # (height, width)
            mask_slice = mask_array[slice_idx]
            
            return img_slice, mask_slice
        else:
            # Return full volume
            return img_array, mask_array
    
    def __len__(self) -> int:
        """Return number of samples in dataset."""
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a sample from the dataset.
        
        Args:
            idx: Sample index
            
        Returns:
            Dictionary with 'image', 'mask', and 'metadata'
        """
        sample = self.samples[idx]
        
        # Load from cache or disk
        if self.cache_data:
            key = f"{sample['patient_id']}_{sample['slice_idx']}"
            img, mask = self.cache[key]
        else:
            img, mask = self._load_sample(sample)
        
        # Add channel dimension for 2D: (H, W) -> (1, H, W)
        if img.ndim == 2:
            img = img[np.newaxis, ...]
            mask = mask[np.newaxis, ...]
        
        # Convert to torch tensors
        img_tensor = torch.from_numpy(img.copy()).float()
        mask_tensor = torch.from_numpy(mask.copy()).float()
        
        # Apply transforms (augmentation)
        if self.transform is not None:
            # Combine for synchronized transforms
            combined = torch.cat([img_tensor, mask_tensor], dim=0)
            combined = self.transform(combined)
            img_tensor = combined[0:1]
            mask_tensor = combined[1:2]
        
        return {
            'image': img_tensor,
            'mask': mask_tensor,
            'patient_id': sample['patient_id'],
            'slice_idx': sample.get('slice_idx', -1)
        }


class AugmentationTransforms:
    """Data augmentation transforms for medical images."""
    
    @staticmethod
    def get_train_transforms():
        """
        Get training data augmentation pipeline.
        
        Returns:
            None for now (augmentation will be added later)
        """
        # TODO: Add augmentation with albumentations or custom transforms
        return None
    
    @staticmethod
    def get_val_transforms():
        """
        Get validation transforms (typically none).
        
        Returns:
            None
        """
        return None  # No augmentation for validation


def create_dataloaders(
    data_root: str,
    batch_size: int = 8,
    num_workers: int = 4,
    cache_data: bool = False,
    slice_wise: bool = True
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test dataloaders.
    
    Args:
        data_root: Root directory with normalized data
        batch_size: Batch size for training
        num_workers: Number of worker processes
        cache_data: Cache all data in memory
        slice_wise: Treat each slice as separate sample
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Create datasets
    train_dataset = NSCLCDataset(
        data_root=data_root,
        split='train',
        transform=AugmentationTransforms.get_train_transforms(),
        cache_data=cache_data,
        slice_wise=slice_wise
    )
    
    val_dataset = NSCLCDataset(
        data_root=data_root,
        split='val',
        transform=AugmentationTransforms.get_val_transforms(),
        cache_data=cache_data,
        slice_wise=slice_wise
    )
    
    test_dataset = NSCLCDataset(
        data_root=data_root,
        split='test',
        transform=None,
        cache_data=cache_data,
        slice_wise=slice_wise
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
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
    
    print(f"\nDataLoaders created:")
    print(f"  Train: {len(train_loader)} batches ({len(train_dataset)} samples)")
    print(f"  Val: {len(val_loader)} batches ({len(val_dataset)} samples)")
    print(f"  Test: {len(test_loader)} batches ({len(test_dataset)} samples)")
    
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    print("PyTorch Dataset & DataLoader Module for NSCLC")
    print("Import this module to use dataset classes")
