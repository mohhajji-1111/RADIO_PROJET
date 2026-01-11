"""
=============================================================================
DICOM to NIfTI Conversion Module
=============================================================================
This module handles conversion of DICOM CT series to NIfTI format.

Functions:
    - load_ct_dicom_series: Load CT DICOM series from directory
    - resample_image: Resample image to target spacing
    - save_nifti: Save image as NIfTI format
    - process_single_patient: Complete pipeline for one patient
    - batch_convert: Convert multiple patients

Author: Medical Imaging Team
Date: November 2025
=============================================================================
"""

import os
import numpy as np
import SimpleITK as sitk
import pydicom
from pathlib import Path
from typing import Tuple, Optional, Dict, List
import logging
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_ct_dicom_series(dicom_dir: str) -> sitk.Image:
    """
    Load CT DICOM series from directory using SimpleITK.
    
    Args:
        dicom_dir: Path to directory containing DICOM files
        
    Returns:
        SimpleITK Image object (3D CT volume)
        
    Raises:
        FileNotFoundError: If directory doesn't exist or contains no DICOM files
    """
    if not os.path.exists(dicom_dir):
        raise FileNotFoundError(f"Directory not found: {dicom_dir}")
    
    # Get all DICOM files
    reader = sitk.ImageSeriesReader()
    dicom_files = reader.GetGDCMSeriesFileNames(dicom_dir)
    
    if len(dicom_files) == 0:
        raise FileNotFoundError(f"No DICOM files found in: {dicom_dir}")
    
    # Load series
    reader.SetFileNames(dicom_files)
    reader.MetaDataDictionaryArrayUpdateOn()
    reader.LoadPrivateTagsOn()
    
    try:
        image = reader.Execute()
        logger.info(f"Loaded DICOM series: {len(dicom_files)} slices")
        logger.info(f"Image size: {image.GetSize()}")
        logger.info(f"Original spacing: {image.GetSpacing()} mm")
        return image
    except Exception as e:
        raise RuntimeError(f"Failed to load DICOM series: {str(e)}")


def resample_image(
    image: sitk.Image,
    target_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    interpolator: str = "linear"
) -> sitk.Image:
    """
    Resample image to target spacing.
    
    Args:
        image: Input SimpleITK image
        target_spacing: Target spacing in mm (x, y, z)
        interpolator: Interpolation method ("linear", "nearest", "bspline")
        
    Returns:
        Resampled SimpleITK image
    """
    # Get original properties
    original_spacing = image.GetSpacing()
    original_size = image.GetSize()
    
    # Calculate new size
    new_size = [
        int(round(original_size[0] * (original_spacing[0] / target_spacing[0]))),
        int(round(original_size[1] * (original_spacing[1] / target_spacing[1]))),
        int(round(original_size[2] * (original_spacing[2] / target_spacing[2])))
    ]
    
    # Select interpolator
    interpolator_map = {
        "linear": sitk.sitkLinear,
        "nearest": sitk.sitkNearestNeighbor,
        "bspline": sitk.sitkBSpline
    }
    interp = interpolator_map.get(interpolator.lower(), sitk.sitkLinear)
    
    # Setup resampler
    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(target_spacing)
    resampler.SetSize(new_size)
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetTransform(sitk.Transform())
    resampler.SetDefaultPixelValue(image.GetPixelIDValue())
    resampler.SetInterpolator(interp)
    
    # Execute resampling
    resampled_image = resampler.Execute(image)
    
    logger.info(f"Resampled from {original_size} to {new_size}")
    logger.info(f"Spacing: {original_spacing} -> {target_spacing} mm")
    
    return resampled_image


def save_nifti(
    image: sitk.Image,
    output_path: str,
    compress: bool = True
) -> None:
    """
    Save SimpleITK image as NIfTI file.
    
    Args:
        image: SimpleITK image to save
        output_path: Output file path (.nii or .nii.gz)
        compress: Whether to compress (use .nii.gz)
    """
    # Ensure output directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Ensure correct extension
    if compress and not output_path.endswith('.nii.gz'):
        output_path = output_path.replace('.nii', '.nii.gz')
    elif not compress and output_path.endswith('.gz'):
        output_path = output_path.replace('.nii.gz', '.nii')
    
    # Save image
    sitk.WriteImage(image, output_path, useCompression=compress)
    
    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    logger.info(f"Saved NIfTI: {output_path} ({file_size_mb:.2f} MB)")


def process_single_patient(
    patient_id: str,
    ct_series_path: str,
    output_dir: str,
    target_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0)
) -> Dict[str, any]:
    """
    Process single patient: load DICOM, resample, save NIfTI.
    
    Args:
        patient_id: Patient identifier
        ct_series_path: Path to CT DICOM series directory
        output_dir: Output directory for NIfTI files
        target_spacing: Target voxel spacing
        
    Returns:
        Dictionary with processing results and metadata
    """
    try:
        logger.info(f"Processing patient: {patient_id}")
        
        # Load DICOM series
        image = load_ct_dicom_series(ct_series_path)
        original_size = image.GetSize()
        original_spacing = image.GetSpacing()
        
        # Resample to target spacing
        resampled_image = resample_image(image, target_spacing)
        
        # Save as NIfTI
        output_path = os.path.join(output_dir, f"{patient_id}.nii.gz")
        save_nifti(resampled_image, output_path)
        
        # Return metadata
        result = {
            "patient_id": patient_id,
            "status": "success",
            "original_size": original_size,
            "original_spacing": original_spacing,
            "resampled_size": resampled_image.GetSize(),
            "resampled_spacing": resampled_image.GetSpacing(),
            "output_path": output_path,
            "num_slices": original_size[2]
        }
        
        logger.info(f"Successfully processed {patient_id}")
        return result
        
    except Exception as e:
        logger.error(f"Failed to process {patient_id}: {str(e)}")
        return {
            "patient_id": patient_id,
            "status": "failed",
            "error": str(e)
        }


def batch_convert(
    patient_list: List[str],
    input_root: str,
    output_root: str,
    target_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0)
) -> None:
    """
    Batch convert multiple patients from DICOM to NIfTI.
    
    Args:
        patient_list: List of patient IDs to process
        input_root: Root directory containing patient DICOM folders
        output_root: Root directory for output NIfTI files
        target_spacing: Target voxel spacing
    """
    logger.info(f"Starting batch conversion for {len(patient_list)} patients")
    
    results = []
    success_count = 0
    failed_count = 0
    
    # Create output directory
    os.makedirs(output_root, exist_ok=True)
    
    # Process each patient with progress bar
    for patient_id in tqdm(patient_list, desc="Converting DICOM to NIfTI"):
        # Find CT series directory
        patient_dir = os.path.join(input_root, patient_id)
        
        if not os.path.exists(patient_dir):
            logger.warning(f"Patient directory not found: {patient_dir}")
            results.append({
                "patient_id": patient_id,
                "status": "failed",
                "error": "Directory not found"
            })
            failed_count += 1
            continue
        
        # Find CT series (look for subdirectories)
        ct_series_path = None
        for root, dirs, files in os.walk(patient_dir):
            # Skip RTSTRUCT directories (usually contain '300.000000' or 'Segmentation')
            if 'Segmentation' in root or '300.000000' in root:
                continue
            # Check if directory contains DICOM files
            dicom_files = [f for f in files if f.endswith('.dcm') or 'DICOMDIR' not in f]
            if len(dicom_files) > 10:  # Assume CT series has many slices
                ct_series_path = root
                break
        
        if ct_series_path is None:
            logger.warning(f"No CT series found for {patient_id}")
            results.append({
                "patient_id": patient_id,
                "status": "failed",
                "error": "No CT series found"
            })
            failed_count += 1
            continue
        
        # Process patient
        result = process_single_patient(
            patient_id=patient_id,
            ct_series_path=ct_series_path,
            output_dir=output_root,
            target_spacing=target_spacing
        )
        
        results.append(result)
        if result["status"] == "success":
            success_count += 1
        else:
            failed_count += 1
    
    # Summary
    logger.info("="*50)
    logger.info("Batch conversion complete!")
    logger.info(f"Total: {len(patient_list)} patients")
    logger.info(f"Success: {success_count}")
    logger.info(f"Failed: {failed_count}")
    logger.info("="*50)
    
    # Save results log
    import json
    log_path = os.path.join(output_root, "conversion_log.json")
    with open(log_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Conversion log saved: {log_path}")


if __name__ == "__main__":
    # Example usage
    logger.info("DICOM to NIfTI conversion module loaded")
    logger.info("This module will be implemented in Phase 2")
