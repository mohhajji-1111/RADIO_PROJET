"""
=============================================================================
RTSTRUCT Mask Extraction Module
=============================================================================
This module extracts binary masks from DICOM RTSTRUCT files.

Functions:
    - load_rtstruct: Load RTSTRUCT DICOM file
    - extract_roi_names: Get all ROI names from RTSTRUCT
    - get_roi_contours: Extract contour points for specific ROI
    - contours_to_mask: Convert contour points to binary mask
    - process_rtstruct: Complete pipeline for mask extraction
    - batch_extract_masks: Process multiple patients

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
import cv2
from scipy import ndimage

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_rtstruct(rtstruct_path: str) -> pydicom.Dataset:
    """
    Load RTSTRUCT DICOM file.
    
    Args:
        rtstruct_path: Path to RTSTRUCT DICOM file
        
    Returns:
        pydicom Dataset object
        
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file is not an RTSTRUCT
    """
    # Handle directory path - find RTSTRUCT file
    if os.path.isdir(rtstruct_path):
        dcm_files = [f for f in os.listdir(rtstruct_path) if f.endswith('.dcm')]
        if not dcm_files:
            raise FileNotFoundError(f"No DICOM files in {rtstruct_path}")
        rtstruct_path = os.path.join(rtstruct_path, dcm_files[0])
    
    if not os.path.exists(rtstruct_path):
        raise FileNotFoundError(f"RTSTRUCT file not found: {rtstruct_path}")
    
    # Load DICOM file
    ds = pydicom.dcmread(rtstruct_path)
    
    # Validate it's an RTSTRUCT
    if ds.Modality != "RTSTRUCT":
        raise ValueError(f"File is not RTSTRUCT, got modality: {ds.Modality}")
    
    logger.info(f"Loaded RTSTRUCT: {rtstruct_path}")
    
    # Check required sequences
    if not hasattr(ds, 'StructureSetROISequence'):
        raise ValueError("RTSTRUCT missing StructureSetROISequence")
    
    if not hasattr(ds, 'ROIContourSequence'):
        raise ValueError("RTSTRUCT missing ROIContourSequence")
    
    return ds


def extract_roi_names(rtstruct_ds: pydicom.Dataset) -> List[str]:
    """
    Extract all ROI names from RTSTRUCT.
    
    Args:
        rtstruct_ds: pydicom Dataset of RTSTRUCT
        
    Returns:
        List of ROI names
    """
    roi_names = []
    
    if hasattr(rtstruct_ds, 'StructureSetROISequence'):
        for roi in rtstruct_ds.StructureSetROISequence:
            roi_name = roi.ROIName
            roi_names.append(roi_name)
            logger.debug(f"Found ROI: {roi_name} (Number: {roi.ROINumber})")
    
    logger.info(f"Extracted {len(roi_names)} ROI names")
    return roi_names


def get_roi_contours(
    rtstruct_ds: pydicom.Dataset,
    roi_name: str
) -> List[np.ndarray]:
    """
    Extract contour points for specific ROI.
    
    Args:
        rtstruct_ds: pydicom Dataset of RTSTRUCT
        roi_name: Name of ROI to extract
        
    Returns:
        List of contour arrays (each is Nx3 array of [x, y, z] points)
    """
    # Build mapping from ROI number to name
    roi_number_to_name = {}
    if hasattr(rtstruct_ds, 'StructureSetROISequence'):
        for roi in rtstruct_ds.StructureSetROISequence:
            roi_number_to_name[roi.ROINumber] = roi.ROIName
    
    # Find ROI number for target name
    target_roi_number = None
    for roi_num, name in roi_number_to_name.items():
        if name.lower() == roi_name.lower():
            target_roi_number = roi_num
            break
    
    if target_roi_number is None:
        logger.warning(f"ROI '{roi_name}' not found in RTSTRUCT")
        return []
    
    # Extract contours for this ROI
    contours = []
    if hasattr(rtstruct_ds, 'ROIContourSequence'):
        for roi_contour in rtstruct_ds.ROIContourSequence:
            if roi_contour.ReferencedROINumber == target_roi_number:
                if hasattr(roi_contour, 'ContourSequence'):
                    for contour in roi_contour.ContourSequence:
                        if hasattr(contour, 'ContourData'):
                            # ContourData is flat list: [x1,y1,z1, x2,y2,z2, ...]
                            points = np.array(contour.ContourData).reshape(-1, 3)
                            contours.append(points)
    
    logger.info(f"Extracted {len(contours)} contour slices for ROI '{roi_name}'")
    return contours


def contours_to_mask(
    contours: List[np.ndarray],
    reference_image: sitk.Image
) -> sitk.Image:
    """
    Convert contour points to binary mask aligned with reference image.
    
    Args:
        contours: List of contour point arrays
        reference_image: Reference CT image for size/spacing
        
    Returns:
        Binary mask as SimpleITK Image
    """
    # Get image properties
    size = reference_image.GetSize()
    spacing = reference_image.GetSpacing()
    origin = reference_image.GetOrigin()
    direction = reference_image.GetDirection()
    
    # Create empty mask array
    mask_array = np.zeros(size[::-1], dtype=np.uint8)  # (z, y, x)
    
    # Group contours by z-coordinate (slice)
    contours_by_slice = {}
    for contour_points in contours:
        if len(contour_points) < 3:
            continue
        
        # Get z-coordinate (should be constant for each contour)
        z_coord = contour_points[0, 2]
        
        # Convert physical coordinates to voxel indices
        # Transform from physical space to voxel space
        voxel_points = []
        for point in contour_points:
            # Convert physical coordinates to voxel indices
            idx_x = int(round((point[0] - origin[0]) / spacing[0]))
            idx_y = int(round((point[1] - origin[1]) / spacing[1]))
            idx_z = int(round((point[2] - origin[2]) / spacing[2]))
            
            voxel_points.append([idx_x, idx_y])
        
        # Store in dictionary by slice index
        if 0 <= idx_z < size[2]:
            if idx_z not in contours_by_slice:
                contours_by_slice[idx_z] = []
            contours_by_slice[idx_z].append(np.array(voxel_points, dtype=np.int32))
    
    # Fill contours on each slice
    for slice_idx, slice_contours in contours_by_slice.items():
        # Create slice mask
        slice_mask = np.zeros((size[1], size[0]), dtype=np.uint8)
        
        # Fill all contours on this slice
        for contour in slice_contours:
            if len(contour) >= 3:
                cv2.fillPoly(slice_mask, [contour], 1)
        
        # Store in 3D mask
        mask_array[slice_idx, :, :] = slice_mask
    
    # Convert numpy array to SimpleITK image
    mask_image = sitk.GetImageFromArray(mask_array)
    mask_image.SetSpacing(spacing)
    mask_image.SetOrigin(origin)
    mask_image.SetDirection(direction)
    
    # Calculate volume statistics
    num_voxels = np.sum(mask_array)
    volume_ml = num_voxels * np.prod(spacing) / 1000.0  # mm³ to mL
    
    logger.info(f"Created mask: {num_voxels} voxels, volume: {volume_ml:.2f} mL")
    
    return mask_image


def process_rtstruct(
    patient_id: str,
    rtstruct_path: str,
    reference_ct_path: str,
    output_dir: str,
    target_rois: Optional[List[str]] = None
) -> Dict[str, any]:
    """
    Process RTSTRUCT: extract ROIs and create binary masks.
    
    Args:
        patient_id: Patient identifier
        rtstruct_path: Path to RTSTRUCT DICOM file
        reference_ct_path: Path to reference CT NIfTI
        output_dir: Output directory for mask NIfTI files
        target_rois: List of ROI names to extract (None = all)
        
    Returns:
        Dictionary with processing results
    """
    try:
        logger.info(f"Processing RTSTRUCT for patient: {patient_id}")
        
        # Load RTSTRUCT
        rtstruct_ds = load_rtstruct(rtstruct_path)
        
        # Load reference CT image
        reference_ct = sitk.ReadImage(reference_ct_path)
        logger.info(f"Loaded reference CT: {reference_ct.GetSize()}")
        
        # Extract all ROI names
        all_roi_names = extract_roi_names(rtstruct_ds)
        
        # Determine which ROIs to process
        if target_rois is None:
            rois_to_process = all_roi_names
        else:
            # Case-insensitive matching
            rois_to_process = []
            for target in target_rois:
                for roi in all_roi_names:
                    if roi.lower() == target.lower():
                        rois_to_process.append(roi)
                        break
        
        logger.info(f"Processing {len(rois_to_process)} ROIs: {rois_to_process}")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Process each ROI
        processed_rois = []
        for roi_name in rois_to_process:
            logger.info(f"Processing ROI: {roi_name}")
            
            # Extract contours
            contours = get_roi_contours(rtstruct_ds, roi_name)
            
            if len(contours) == 0:
                logger.warning(f"No contours found for ROI '{roi_name}'")
                continue
            
            # Convert to mask
            mask = contours_to_mask(contours, reference_ct)
            
            # Save mask
            safe_roi_name = roi_name.replace(' ', '_').replace('/', '_')
            output_path = os.path.join(output_dir, f"{patient_id}_{safe_roi_name}.nii.gz")
            sitk.WriteImage(mask, output_path, useCompression=True)
            
            file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
            logger.info(f"Saved mask: {output_path} ({file_size_mb:.2f} MB)")
            
            processed_rois.append({
                "roi_name": roi_name,
                "num_contours": len(contours),
                "output_path": output_path
            })
        
        result = {
            "patient_id": patient_id,
            "status": "success",
            "all_rois": all_roi_names,
            "processed_rois": processed_rois,
            "num_rois_processed": len(processed_rois)
        }
        
        logger.info(f"Successfully processed {patient_id}: {len(processed_rois)} masks created")
        return result
        
    except Exception as e:
        logger.error(f"Failed to process {patient_id}: {str(e)}")
        return {
            "patient_id": patient_id,
            "status": "failed",
            "error": str(e)
        }


def batch_extract_masks(
    patient_list: List[str],
    rtstruct_root: str,
    ct_nifti_root: str,
    output_root: str,
    target_rois: Optional[List[str]] = None
) -> None:
    """
    Batch extract masks from RTSTRUCT for multiple patients.
    
    Args:
        patient_list: List of patient IDs
        rtstruct_root: Root directory containing RTSTRUCT files
        ct_nifti_root: Root directory containing CT NIfTI files
        output_root: Root directory for output masks
        target_rois: Target ROI names to extract
    """
    logger.info(f"Starting batch mask extraction for {len(patient_list)} patients")
    
    results = []
    success_count = 0
    failed_count = 0
    
    # Create output directory
    os.makedirs(output_root, exist_ok=True)
    
    # Process each patient
    for patient_id in tqdm(patient_list, desc="Extracting masks from RTSTRUCT"):
        # Find reference CT NIfTI
        ct_path = os.path.join(ct_nifti_root, f"{patient_id}.nii.gz")
        
        if not os.path.exists(ct_path):
            logger.warning(f"CT NIfTI not found for {patient_id}: {ct_path}")
            results.append({
                "patient_id": patient_id,
                "status": "failed",
                "error": "CT NIfTI not found"
            })
            failed_count += 1
            continue
        
        # Find RTSTRUCT directory
        patient_dir = os.path.join(rtstruct_root, patient_id)
        
        if not os.path.exists(patient_dir):
            logger.warning(f"Patient directory not found: {patient_dir}")
            results.append({
                "patient_id": patient_id,
                "status": "failed",
                "error": "Patient directory not found"
            })
            failed_count += 1
            continue
        
        # Find RTSTRUCT file (look for directories with '300.000000' or 'Segmentation')
        rtstruct_path = None
        for root, dirs, files in os.walk(patient_dir):
            # RTSTRUCT is usually in directories containing 'Segmentation' or '300.000000'
            if 'Segmentation' in root or '300.000000' in root:
                dcm_files = [f for f in files if f.endswith('.dcm')]
                if dcm_files:
                    rtstruct_path = root
                    break
        
        if rtstruct_path is None:
            logger.warning(f"No RTSTRUCT found for {patient_id}")
            results.append({
                "patient_id": patient_id,
                "status": "failed",
                "error": "No RTSTRUCT found"
            })
            failed_count += 1
            continue
        
        # Process RTSTRUCT
        result = process_rtstruct(
            patient_id=patient_id,
            rtstruct_path=rtstruct_path,
            reference_ct_path=ct_path,
            output_dir=output_root,
            target_rois=target_rois
        )
        
        results.append(result)
        if result["status"] == "success":
            success_count += 1
        else:
            failed_count += 1
    
    # Summary
    logger.info("="*50)
    logger.info("Batch mask extraction complete!")
    logger.info(f"Total: {len(patient_list)} patients")
    logger.info(f"Success: {success_count}")
    logger.info(f"Failed: {failed_count}")
    logger.info("="*50)
    
    # Save results log
    import json
    log_path = os.path.join(output_root, "mask_extraction_log.json")
    with open(log_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Extraction log saved: {log_path}")


if __name__ == "__main__":
    # Example usage
    logger.info("RTSTRUCT mask extraction module loaded")
    logger.info("This module will be implemented in Phase 2")
