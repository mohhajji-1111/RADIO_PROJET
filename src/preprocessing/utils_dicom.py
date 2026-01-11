"""
=============================================================================
DICOM Utilities Module
=============================================================================
Helper functions for DICOM handling.

Functions:
    - get_dicom_metadata: Extract metadata from DICOM file
    - find_dicom_series: Find DICOM series in directory tree
    - validate_dicom_series: Validate DICOM series completeness
    - sort_dicom_by_position: Sort DICOM files by position

Author: Medical Imaging Team
Date: November 2025
=============================================================================
"""

import os
import pydicom
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_dicom_metadata(dicom_path: str) -> Dict[str, any]:
    """
    Extract key metadata from DICOM file.
    
    Args:
        dicom_path: Path to DICOM file
        
    Returns:
        Dictionary of metadata
    """
    try:
        dcm = pydicom.dcmread(dicom_path, stop_before_pixels=True)
        
        metadata = {
            "PatientID": str(dcm.get("PatientID", "Unknown")),
            "StudyDate": str(dcm.get("StudyDate", "Unknown")),
            "SeriesDescription": str(dcm.get("SeriesDescription", "Unknown")),
            "Modality": str(dcm.get("Modality", "Unknown")),
            "SeriesInstanceUID": str(dcm.get("SeriesInstanceUID", "Unknown")),
            "SliceThickness": float(dcm.get("SliceThickness", 0)) if dcm.get("SliceThickness") else None,
            "PixelSpacing": list(dcm.get("PixelSpacing", [])) if dcm.get("PixelSpacing") else None,
            "ImagePositionPatient": list(dcm.get("ImagePositionPatient", [])) if dcm.get("ImagePositionPatient") else None,
            "Rows": int(dcm.get("Rows", 0)) if dcm.get("Rows") else None,
            "Columns": int(dcm.get("Columns", 0)) if dcm.get("Columns") else None,
        }
        
        return metadata
        
    except Exception as e:
        logger.error(f"Failed to read DICOM metadata from {dicom_path}: {str(e)}")
        return {}


def find_dicom_series(
    root_dir: str,
    modality: str = "CT"
) -> List[str]:
    """
    Find all DICOM series of specified modality in directory tree.
    
    Args:
        root_dir: Root directory to search
        modality: DICOM modality to find (e.g., 'CT', 'RTSTRUCT')
        
    Returns:
        List of directory paths containing series
    """
    series_dirs = []
    
    # Walk through directory tree
    for root, dirs, files in os.walk(root_dir):
        # Check if directory contains DICOM files
        dicom_files = [f for f in files if f.endswith('.dcm')]
        
        if len(dicom_files) == 0:
            continue
        
        # Check modality of first DICOM file
        try:
            first_dcm_path = os.path.join(root, dicom_files[0])
            dcm = pydicom.dcmread(first_dcm_path, stop_before_pixels=True)
            
            if dcm.get("Modality", "") == modality:
                series_dirs.append(root)
                logger.debug(f"Found {modality} series: {root}")
                
        except Exception as e:
            logger.debug(f"Error reading {first_dcm_path}: {str(e)}")
            continue
    
    logger.info(f"Found {len(series_dirs)} {modality} series in {root_dir}")
    return series_dirs


def validate_dicom_series(series_dir: str) -> bool:
    """
    Validate DICOM series completeness and consistency.
    
    Args:
        series_dir: Directory containing DICOM series
        
    Returns:
        True if valid, False otherwise
    """
    try:
        # Get all DICOM files
        dicom_files = [os.path.join(series_dir, f) for f in os.listdir(series_dir) if f.endswith('.dcm')]
        
        if len(dicom_files) < 5:
            logger.warning(f"Too few slices ({len(dicom_files)}) in {series_dir}")
            return False
        
        # Read first file to get series UID
        first_dcm = pydicom.dcmread(dicom_files[0], stop_before_pixels=True)
        series_uid = first_dcm.get("SeriesInstanceUID")
        modality = first_dcm.get("Modality")
        
        if not series_uid:
            logger.warning(f"No SeriesInstanceUID in {series_dir}")
            return False
        
        # Check all files belong to same series
        for dcm_path in dicom_files[1:10]:  # Check first 10 files
            dcm = pydicom.dcmread(dcm_path, stop_before_pixels=True)
            if dcm.get("SeriesInstanceUID") != series_uid:
                logger.warning(f"Mixed series in {series_dir}")
                return False
            if dcm.get("Modality") != modality:
                logger.warning(f"Mixed modalities in {series_dir}")
                return False
        
        logger.debug(f"Valid series: {series_dir} ({len(dicom_files)} slices)")
        return True
        
    except Exception as e:
        logger.error(f"Validation error for {series_dir}: {str(e)}")
        return False


def sort_dicom_by_position(dicom_files: List[str]) -> List[str]:
    """
    Sort DICOM files by slice position.
    
    Args:
        dicom_files: List of DICOM file paths
        
    Returns:
        Sorted list of file paths
    """
    # Create list of (file, position) tuples
    files_with_position = []
    
    for dcm_path in dicom_files:
        try:
            dcm = pydicom.dcmread(dcm_path, stop_before_pixels=True)
            
            # Try to get position from ImagePositionPatient (z-coordinate)
            if "ImagePositionPatient" in dcm:
                position = float(dcm.ImagePositionPatient[2])
            # Fallback to InstanceNumber
            elif "InstanceNumber" in dcm:
                position = float(dcm.InstanceNumber)
            # Fallback to SliceLocation
            elif "SliceLocation" in dcm:
                position = float(dcm.SliceLocation)
            else:
                # Use filename as last resort
                position = float(os.path.basename(dcm_path).split('.')[0])
            
            files_with_position.append((dcm_path, position))
            
        except Exception as e:
            logger.warning(f"Could not determine position for {dcm_path}: {str(e)}")
            files_with_position.append((dcm_path, 0))
    
    # Sort by position
    files_with_position.sort(key=lambda x: x[1])
    
    # Return sorted file paths
    sorted_files = [f[0] for f in files_with_position]
    
    logger.debug(f"Sorted {len(sorted_files)} DICOM files by position")
    return sorted_files


if __name__ == "__main__":
    logger.info("DICOM utilities module loaded")
    logger.info("This module will be implemented in Phase 2")
