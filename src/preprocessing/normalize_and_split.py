"""
Data Normalization and Train/Val/Test Splitting for NSCLC Project.

This module handles:
1. Intensity normalization (HU windowing, Z-score)
2. Spatial normalization (resizing to 256×256)
3. Train/validation/test splitting (70/15/15)
4. Quality control checks

Author: GitHub Copilot
Date: 2025
"""

import os
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json
import random

import numpy as np
import SimpleITK as sitk
from sklearn.model_selection import train_test_split
from tqdm import tqdm


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def apply_hu_windowing(
    image: sitk.Image,
    window_min: int = -1000,
    window_max: int = 400
) -> sitk.Image:
    """
    Apply HU windowing (lung window by default).
    
    Args:
        image: Input CT image in Hounsfield Units
        window_min: Minimum HU value (default: -1000 for air)
        window_max: Maximum HU value (default: 400 for soft tissue)
        
    Returns:
        Windowed image with values clipped to [window_min, window_max]
    """
    # Convert to numpy array
    array = sitk.GetArrayFromImage(image)
    
    # Apply windowing (clip values)
    windowed_array = np.clip(array, window_min, window_max)
    
    # Convert back to SimpleITK image
    windowed_image = sitk.GetImageFromArray(windowed_array)
    windowed_image.CopyInformation(image)
    
    logger.debug(f"Applied HU windowing [{window_min}, {window_max}]")
    return windowed_image


def normalize_intensity(
    image: sitk.Image,
    method: str = "zscore"
) -> sitk.Image:
    """
    Normalize image intensity values.
    
    Args:
        image: Input image (after HU windowing)
        method: Normalization method ('zscore', 'minmax', or 'none')
        
    Returns:
        Normalized image
    """
    array = sitk.GetArrayFromImage(image)
    
    if method == "zscore":
        # Z-score normalization: (x - mean) / std
        mean = np.mean(array)
        std = np.std(array)
        if std > 0:
            normalized_array = (array - mean) / std
        else:
            normalized_array = array - mean
        logger.debug(f"Z-score normalization: mean={mean:.2f}, std={std:.2f}")
        
    elif method == "minmax":
        # Min-max normalization to [0, 1]
        min_val = np.min(array)
        max_val = np.max(array)
        if max_val > min_val:
            normalized_array = (array - min_val) / (max_val - min_val)
        else:
            normalized_array = array - min_val
        logger.debug(f"Min-max normalization: [{min_val:.2f}, {max_val:.2f}] -> [0, 1]")
        
    elif method == "none":
        normalized_array = array
        logger.debug("No intensity normalization applied")
        
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    
    # Convert back to SimpleITK image
    normalized_image = sitk.GetImageFromArray(normalized_array)
    normalized_image.CopyInformation(image)
    
    return normalized_image


def resize_image(
    image: sitk.Image,
    new_size: Tuple[int, int, int] = (256, 256, None),
    interpolator: int = sitk.sitkLinear,
    is_mask: bool = False
) -> sitk.Image:
    """
    Resize image to target size. If new_size[2] is None, keeps original depth.
    
    Args:
        image: Input image
        new_size: Target size (width, height, depth). Use None for depth to keep original
        interpolator: Interpolation method (sitkLinear for images, sitkNearestNeighbor for masks)
        is_mask: If True, uses nearest neighbor interpolation
        
    Returns:
        Resized image
    """
    original_size = image.GetSize()
    original_spacing = image.GetSpacing()
    
    # Determine target size (keep depth if None)
    if new_size[2] is None:
        target_size = (new_size[0], new_size[1], original_size[2])
    else:
        target_size = new_size
    
    # Calculate new spacing to maintain physical dimensions
    new_spacing = [
        (original_size[i] * original_spacing[i]) / target_size[i]
        for i in range(3)
    ]
    
    # Use nearest neighbor for masks to preserve binary values
    if is_mask:
        interpolator = sitk.sitkNearestNeighbor
    
    # Resample image
    resampler = sitk.ResampleImageFilter()
    resampler.SetSize(target_size)
    resampler.SetOutputSpacing(new_spacing)
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetInterpolator(interpolator)
    resampler.SetDefaultPixelValue(0)
    
    resampled_image = resampler.Execute(image)
    
    logger.debug(f"Resized from {original_size} to {target_size}")
    return resampled_image


def process_single_case(
    ct_path: str,
    mask_path: str,
    output_ct_path: str,
    output_mask_path: str,
    target_size: Tuple[int, int, int] = (256, 256, None),
    hu_window: Tuple[int, int] = (-1000, 400),
    normalize_method: str = "zscore"
) -> Dict:
    """
    Process single CT-mask pair: windowing, normalization, resizing.
    
    Args:
        ct_path: Path to CT NIfTI file
        mask_path: Path to mask NIfTI file
        output_ct_path: Output path for processed CT
        output_mask_path: Output path for processed mask
        target_size: Target image size (width, height, depth)
        hu_window: HU window range (min, max)
        normalize_method: Intensity normalization method
        
    Returns:
        Dictionary with processing metadata
    """
    try:
        # Load images
        ct_image = sitk.ReadImage(ct_path)
        mask_image = sitk.ReadImage(mask_path)
        
        # Verify dimensions match
        if ct_image.GetSize() != mask_image.GetSize():
            logger.warning(f"Size mismatch: CT {ct_image.GetSize()} vs Mask {mask_image.GetSize()}")
        
        # Process CT: windowing -> normalization -> resize
        ct_windowed = apply_hu_windowing(ct_image, hu_window[0], hu_window[1])
        ct_normalized = normalize_intensity(ct_windowed, method=normalize_method)
        ct_resized = resize_image(ct_normalized, target_size, is_mask=False)
        
        # Process Mask: resize only (no normalization needed for binary masks)
        mask_resized = resize_image(mask_image, target_size, is_mask=True)
        
        # Save processed images
        os.makedirs(os.path.dirname(output_ct_path), exist_ok=True)
        os.makedirs(os.path.dirname(output_mask_path), exist_ok=True)
        
        sitk.WriteImage(ct_resized, output_ct_path, useCompression=True)
        sitk.WriteImage(mask_resized, output_mask_path, useCompression=True)
        
        # Calculate statistics
        ct_array = sitk.GetArrayFromImage(ct_resized)
        mask_array = sitk.GetArrayFromImage(mask_resized)
        
        metadata = {
            'success': True,
            'original_size': ct_image.GetSize(),
            'processed_size': ct_resized.GetSize(),
            'ct_mean': float(np.mean(ct_array)),
            'ct_std': float(np.std(ct_array)),
            'ct_min': float(np.min(ct_array)),
            'ct_max': float(np.max(ct_array)),
            'mask_volume_voxels': int(np.sum(mask_array > 0)),
            'mask_positive_ratio': float(np.mean(mask_array > 0))
        }
        
        return metadata
        
    except Exception as e:
        logger.error(f"Error processing case: {str(e)}")
        return {'success': False, 'error': str(e)}


def create_train_val_test_split(
    patient_list: List[str],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_seed: int = 42,
    stratify_by: Optional[Dict[str, any]] = None
) -> Dict[str, List[str]]:
    """
    Split patients into train/validation/test sets.
    
    Args:
        patient_list: List of patient IDs
        train_ratio: Proportion for training (default: 0.70)
        val_ratio: Proportion for validation (default: 0.15)
        test_ratio: Proportion for testing (default: 0.15)
        random_seed: Random seed for reproducibility
        stratify_by: Optional dictionary for stratified splitting
        
    Returns:
        Dictionary with 'train', 'val', 'test' lists
    """
    # Validate ratios
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"
    
    # Set random seed
    random.seed(random_seed)
    np.random.seed(random_seed)
    
    # Shuffle patient list
    shuffled_patients = patient_list.copy()
    random.shuffle(shuffled_patients)
    
    # First split: train vs (val + test)
    train_patients, temp_patients = train_test_split(
        shuffled_patients,
        test_size=(val_ratio + test_ratio),
        random_state=random_seed
    )
    
    # Second split: val vs test
    val_size = val_ratio / (val_ratio + test_ratio)
    val_patients, test_patients = train_test_split(
        temp_patients,
        test_size=(1 - val_size),
        random_state=random_seed
    )
    
    splits = {
        'train': train_patients,
        'val': val_patients,
        'test': test_patients
    }
    
    logger.info(f"Created data splits:")
    logger.info(f"  Train: {len(train_patients)} patients ({len(train_patients)/len(patient_list)*100:.1f}%)")
    logger.info(f"  Val: {len(val_patients)} patients ({len(val_patients)/len(patient_list)*100:.1f}%)")
    logger.info(f"  Test: {len(test_patients)} patients ({len(test_patients)/len(patient_list)*100:.1f}%)")
    
    return splits


def batch_normalize_and_split(
    ct_root: str,
    mask_root: str,
    output_root: str,
    target_size: Tuple[int, int, int] = (256, 256, None),
    hu_window: Tuple[int, int] = (-1000, 400),
    normalize_method: str = "zscore",
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    mask_pattern: str = "*_Neoplasm,_Primary.nii.gz"
) -> Dict:
    """
    Batch process all patients: normalize and create data splits.
    
    Args:
        ct_root: Root directory with CT NIfTI files
        mask_root: Root directory with mask NIfTI files
        output_root: Output directory for processed data
        target_size: Target image size
        hu_window: HU window range
        normalize_method: Intensity normalization method
        train_ratio: Training set ratio
        val_ratio: Validation set ratio
        test_ratio: Test set ratio
        mask_pattern: Pattern to match primary tumor masks
        
    Returns:
        Dictionary with processing results and splits
    """
    logger.info(f"\n{'='*70}")
    logger.info(f"BATCH NORMALIZATION AND SPLITTING")
    logger.info(f"{'='*70}")
    
    # Find all CT-mask pairs
    ct_files = list(Path(ct_root).glob("*.nii.gz"))
    
    valid_pairs = []
    for ct_file in ct_files:
        patient_id = ct_file.stem.replace('.nii', '')
        
        # Find corresponding tumor mask
        mask_files = list(Path(mask_root).glob(f"{patient_id}{mask_pattern}"))
        
        if mask_files:
            valid_pairs.append({
                'patient_id': patient_id,
                'ct_path': str(ct_file),
                'mask_path': str(mask_files[0])
            })
    
    logger.info(f"Found {len(valid_pairs)} valid CT-mask pairs")
    
    if not valid_pairs:
        logger.error("No valid pairs found!")
        return {'success': False, 'error': 'No valid pairs'}
    
    # Create data splits
    patient_ids = [pair['patient_id'] for pair in valid_pairs]
    splits = create_train_val_test_split(
        patient_ids, train_ratio, val_ratio, test_ratio
    )
    
    # Process each split
    results = {
        'train': {'processed': 0, 'failed': 0, 'patients': []},
        'val': {'processed': 0, 'failed': 0, 'patients': []},
        'test': {'processed': 0, 'failed': 0, 'patients': []}
    }
    
    for split_name, split_patients in splits.items():
        logger.info(f"\nProcessing {split_name} set ({len(split_patients)} patients)...")
        
        split_pairs = [p for p in valid_pairs if p['patient_id'] in split_patients]
        
        for pair in tqdm(split_pairs, desc=f"Processing {split_name}"):
            patient_id = pair['patient_id']
            
            # Define output paths
            output_ct = os.path.join(output_root, split_name, 'images', f"{patient_id}.nii.gz")
            output_mask = os.path.join(output_root, split_name, 'masks', f"{patient_id}.nii.gz")
            
            # Process case
            metadata = process_single_case(
                ct_path=pair['ct_path'],
                mask_path=pair['mask_path'],
                output_ct_path=output_ct,
                output_mask_path=output_mask,
                target_size=target_size,
                hu_window=hu_window,
                normalize_method=normalize_method
            )
            
            if metadata['success']:
                results[split_name]['processed'] += 1
                results[split_name]['patients'].append({
                    'patient_id': patient_id,
                    'metadata': metadata
                })
            else:
                results[split_name]['failed'] += 1
                logger.error(f"Failed to process {patient_id}: {metadata.get('error')}")
    
    # Save splits to JSON
    splits_file = os.path.join(output_root, 'data_splits.json')
    with open(splits_file, 'w') as f:
        json.dump(splits, f, indent=2)
    logger.info(f"\nSaved data splits to: {splits_file}")
    
    # Save processing log
    log_file = os.path.join(output_root, 'normalization_log.json')
    with open(log_file, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved processing log to: {log_file}")
    
    # Summary
    total_processed = sum(r['processed'] for r in results.values())
    total_failed = sum(r['failed'] for r in results.values())
    
    logger.info(f"\n{'='*70}")
    logger.info(f"PROCESSING COMPLETE")
    logger.info(f"{'='*70}")
    logger.info(f"Total processed: {total_processed}/{len(valid_pairs)}")
    logger.info(f"Total failed: {total_failed}")
    logger.info(f"Success rate: {total_processed/(total_processed+total_failed)*100:.1f}%")
    
    return {
        'success': True,
        'splits': splits,
        'results': results,
        'total_processed': total_processed,
        'total_failed': total_failed
    }


if __name__ == "__main__":
    print("Data Normalization and Splitting Module")
    print("Import this module to use normalization functions")
