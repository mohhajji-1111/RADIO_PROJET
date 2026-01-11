"""
DICOM SEG (Segmentation Object) Mask Extraction for NSCLC Project.

This module handles DICOM SEG files (not RTSTRUCT), which store binary masks
directly as pixel data rather than contour points.

Author: GitHub Copilot
Date: 2025
"""

import os
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json

import numpy as np
import SimpleITK as sitk
import pydicom
from tqdm import tqdm


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_dicom_seg(seg_path: str) -> pydicom.Dataset:
    """
    Load DICOM SEG file.
    
    Args:
        seg_path: Path to DICOM SEG file or directory
        
    Returns:
        pydicom Dataset object
        
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file is not a DICOM SEG
    """
    # Handle directory path - find SEG file
    if os.path.isdir(seg_path):
        dcm_files = [f for f in os.listdir(seg_path) if f.endswith('.dcm')]
        if not dcm_files:
            raise FileNotFoundError(f"No DICOM files in {seg_path}")
        seg_path = os.path.join(seg_path, dcm_files[0])
    
    if not os.path.exists(seg_path):
        raise FileNotFoundError(f"DICOM SEG file not found: {seg_path}")
    
    # Load DICOM file
    ds = pydicom.dcmread(seg_path)
    
    # Validate it's a SEG
    if ds.Modality != "SEG":
        raise ValueError(f"File is not DICOM SEG, got modality: {ds.Modality}")
    
    logger.info(f"Loaded DICOM SEG: {seg_path}")
    
    # Check required sequences
    if not hasattr(ds, 'SegmentSequence'):
        raise ValueError("DICOM SEG missing SegmentSequence")
    
    return ds


def extract_segment_labels(seg_ds: pydicom.Dataset) -> List[Dict[str, any]]:
    """
    Extract segment information from DICOM SEG.
    
    Args:
        seg_ds: pydicom Dataset of DICOM SEG
        
    Returns:
        List of dictionaries with segment info (SegmentNumber, SegmentLabel, etc.)
    """
    segments = []
    
    if hasattr(seg_ds, 'SegmentSequence'):
        for seg in seg_ds.SegmentSequence:
            segment_info = {
                'SegmentNumber': seg.SegmentNumber,
                'SegmentLabel': seg.SegmentLabel if hasattr(seg, 'SegmentLabel') else f"Segment_{seg.SegmentNumber}",
                'SegmentAlgorithmType': seg.SegmentAlgorithmType if hasattr(seg, 'SegmentAlgorithmType') else 'UNKNOWN',
            }
            segments.append(segment_info)
            logger.debug(f"Found segment: {segment_info['SegmentLabel']} (Number: {segment_info['SegmentNumber']})")
    
    logger.info(f"Extracted {len(segments)} segment labels")
    return segments


def extract_seg_mask(
    seg_ds: pydicom.Dataset,
    reference_ct: sitk.Image,
    segment_number: int = 1
) -> Tuple[sitk.Image, Dict]:
    """
    Extract binary mask from DICOM SEG for specified segment.
    
    DICOM SEG stores masks as pixel arrays with segment identifiers.
    Each frame corresponds to a CT slice with binary mask data.
    
    Args:
        seg_ds: pydicom Dataset of DICOM SEG
        reference_ct: Reference CT image (for spatial alignment)
        segment_number: Segment number to extract (default: 1)
        
    Returns:
        Tuple of (mask_image, metadata_dict)
        - mask_image: SimpleITK image with binary mask
        - metadata_dict: Dictionary with mask statistics
        
    Raises:
        ValueError: If segment not found or dimension mismatch
    """
    logger.info(f"Extracting segment {segment_number} from DICOM SEG")
    
    # Get reference CT properties
    ct_size = reference_ct.GetSize()  # (width, height, depth)
    ct_spacing = reference_ct.GetSpacing()
    ct_origin = reference_ct.GetOrigin()
    ct_direction = reference_ct.GetDirection()
    
    # Get SEG pixel array (frames x height x width)
    seg_array = seg_ds.pixel_array  # Shape: (num_frames, rows, cols)
    
    if seg_array.ndim != 3:
        raise ValueError(f"Expected 3D SEG array, got shape: {seg_array.shape}")
    
    num_frames, rows, cols = seg_array.shape
    
    logger.info(f"SEG dimensions: {num_frames} frames, {rows}x{cols} pixels")
    logger.info(f"CT dimensions: {ct_size[0]}x{ct_size[1]}x{ct_size[2]}")
    
    # Extract frames for the target segment
    # DICOM SEG uses PerFrameFunctionalGroupsSequence to map frames to segments
    segment_frames = []
    
    if hasattr(seg_ds, 'PerFrameFunctionalGroupsSequence'):
        for frame_idx, frame_info in enumerate(seg_ds.PerFrameFunctionalGroupsSequence):
            # Get segment number for this frame
            if hasattr(frame_info, 'SegmentIdentificationSequence'):
                frame_seg_num = frame_info.SegmentIdentificationSequence[0].ReferencedSegmentNumber
                if frame_seg_num == segment_number:
                    segment_frames.append(frame_idx)
    else:
        # Fallback: assume all frames belong to segment_number if no mapping
        logger.warning("No PerFrameFunctionalGroupsSequence found, assuming all frames belong to segment")
        segment_frames = list(range(num_frames))
    
    logger.info(f"Found {len(segment_frames)} frames for segment {segment_number}")
    
    if not segment_frames:
        raise ValueError(f"No frames found for segment {segment_number}")
    
    # Create 3D mask array matching CT dimensions
    mask_array = np.zeros(ct_size[::-1], dtype=np.uint8)  # (depth, height, width)
    
    # Extract mask data from relevant frames
    for frame_idx in segment_frames:
        # Get z-position for this frame
        if hasattr(seg_ds, 'PerFrameFunctionalGroupsSequence'):
            frame_info = seg_ds.PerFrameFunctionalGroupsSequence[frame_idx]
            if hasattr(frame_info, 'PlanePositionSequence'):
                z_pos = frame_info.PlanePositionSequence[0].ImagePositionPatient[2]
                # Convert z position to slice index
                z_idx = int(round((z_pos - ct_origin[2]) / ct_spacing[2]))
                
                if 0 <= z_idx < ct_size[2]:
                    # Extract binary mask for this frame (1 = tumor, 0 = background)
                    frame_mask = (seg_array[frame_idx] > 0).astype(np.uint8)
                    
                    # Resize if needed to match CT dimensions
                    if frame_mask.shape != (ct_size[1], ct_size[0]):
                        import cv2
                        frame_mask = cv2.resize(frame_mask, (ct_size[0], ct_size[1]), interpolation=cv2.INTER_NEAREST)
                    
                    mask_array[z_idx] = frame_mask
        else:
            # Fallback: assume sequential frames map to sequential slices
            if frame_idx < ct_size[2]:
                frame_mask = (seg_array[frame_idx] > 0).astype(np.uint8)
                
                if frame_mask.shape != (ct_size[1], ct_size[0]):
                    import cv2
                    frame_mask = cv2.resize(frame_mask, (ct_size[0], ct_size[1]), interpolation=cv2.INTER_NEAREST)
                
                mask_array[frame_idx] = frame_mask
    
    # Convert to SimpleITK image
    mask_image = sitk.GetImageFromArray(mask_array)
    mask_image.SetSpacing(ct_spacing)
    mask_image.SetOrigin(ct_origin)
    mask_image.SetDirection(ct_direction)
    
    # Calculate statistics
    mask_volume_voxels = int(np.sum(mask_array))
    voxel_volume_mm3 = ct_spacing[0] * ct_spacing[1] * ct_spacing[2]
    mask_volume_mm3 = mask_volume_voxels * voxel_volume_mm3
    mask_volume_cm3 = mask_volume_mm3 / 1000.0
    
    metadata = {
        'segment_number': segment_number,
        'mask_volume_voxels': mask_volume_voxels,
        'mask_volume_mm3': mask_volume_mm3,
        'mask_volume_cm3': mask_volume_cm3,
        'num_slices_with_mask': int(np.sum(np.any(mask_array, axis=(1, 2)))),
        'mask_shape': mask_array.shape,
    }
    
    logger.info(f"✅ Mask extracted: {mask_volume_cm3:.2f} cm³, {metadata['num_slices_with_mask']} slices")
    
    return mask_image, metadata


def process_dicom_seg(
    patient_id: str,
    seg_path: str,
    reference_ct_path: str,
    output_dir: str,
    target_segments: Optional[List[str]] = None
) -> Dict:
    """
    Complete pipeline: Load DICOM SEG → Extract masks → Save as NIfTI.
    
    Args:
        patient_id: Patient identifier
        seg_path: Path to DICOM SEG file/directory
        reference_ct_path: Path to reference CT NIfTI file
        output_dir: Directory to save mask NIfTI files
        target_segments: List of segment labels to extract (None = extract all)
        
    Returns:
        Dictionary with processing results
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"Processing DICOM SEG for patient: {patient_id}")
    logger.info(f"{'='*60}")
    
    results = {
        'patient_id': patient_id,
        'success': False,
        'masks_created': [],
        'error': None
    }
    
    try:
        # Load DICOM SEG
        seg_ds = load_dicom_seg(seg_path)
        
        # Load reference CT
        logger.info(f"Loading reference CT: {reference_ct_path}")
        reference_ct = sitk.ReadImage(reference_ct_path)
        
        # Extract segment information
        segments = extract_segment_labels(seg_ds)
        logger.info(f"Available segments: {[s['SegmentLabel'] for s in segments]}")
        
        # Determine which segments to process
        if target_segments is None:
            # Process all segments
            segments_to_process = segments
        else:
            # Filter by target segment labels (case-insensitive matching)
            target_lower = [t.lower() for t in target_segments]
            segments_to_process = [
                s for s in segments 
                if s['SegmentLabel'].lower() in target_lower
            ]
        
        logger.info(f"Processing {len(segments_to_process)} segment(s)")
        
        # Process each segment
        os.makedirs(output_dir, exist_ok=True)
        
        for seg_info in segments_to_process:
            seg_num = seg_info['SegmentNumber']
            seg_label = seg_info['SegmentLabel']
            
            logger.info(f"\n--- Processing: {seg_label} (Segment {seg_num}) ---")
            
            try:
                # Extract mask
                mask_image, metadata = extract_seg_mask(
                    seg_ds, reference_ct, segment_number=seg_num
                )
                
                # Create safe filename
                safe_label = seg_label.replace(' ', '_').replace('/', '_')
                mask_filename = f"{patient_id}_{safe_label}.nii.gz"
                mask_path = os.path.join(output_dir, mask_filename)
                
                # Save mask
                sitk.WriteImage(mask_image, mask_path, useCompression=True)
                logger.info(f"✅ Saved mask: {mask_path}")
                
                results['masks_created'].append({
                    'segment_label': seg_label,
                    'segment_number': seg_num,
                    'filename': mask_filename,
                    'metadata': metadata
                })
                
            except Exception as e:
                logger.error(f"❌ Failed to process {seg_label}: {str(e)}")
                continue
        
        if results['masks_created']:
            results['success'] = True
            logger.info(f"\n✅ Successfully processed {len(results['masks_created'])} mask(s)")
        else:
            results['error'] = "No masks were successfully created"
            logger.warning("⚠️ No masks were created")
        
    except Exception as e:
        results['error'] = str(e)
        logger.error(f"❌ Error processing patient {patient_id}: {str(e)}")
    
    return results


def batch_extract_seg_masks(
    patient_list: List[str],
    seg_root: str,
    ct_nifti_root: str,
    output_root: str,
    target_segments: Optional[List[str]] = None
) -> Dict:
    """
    Batch extract masks from DICOM SEG files for multiple patients.
    
    Args:
        patient_list: List of patient IDs
        seg_root: Root directory containing patient SEG data
        ct_nifti_root: Root directory with CT NIfTI files
        output_root: Root directory for output masks
        target_segments: List of segment labels to extract
        
    Returns:
        Dictionary with batch processing results
    """
    logger.info(f"\n{'='*70}")
    logger.info(f"BATCH EXTRACTION: DICOM SEG Masks")
    logger.info(f"{'='*70}")
    logger.info(f"Patients: {len(patient_list)}")
    logger.info(f"Target segments: {target_segments if target_segments else 'ALL'}")
    
    results = {
        'total': len(patient_list),
        'successful': 0,
        'failed': 0,
        'patient_results': []
    }
    
    os.makedirs(output_root, exist_ok=True)
    
    for patient_id in tqdm(patient_list, desc="Processing patients"):
        # Find SEG directory (look for 'Segmentation' or similar)
        patient_dir = os.path.join(seg_root, patient_id)
        
        if not os.path.exists(patient_dir):
            logger.warning(f"⚠️ Patient directory not found: {patient_dir}")
            results['failed'] += 1
            continue
        
        # Find segmentation subdirectory
        seg_path = None
        for root, dirs, files in os.walk(patient_dir):
            if 'Segmentation' in root or '300.000000' in root:
                seg_path = root
                break
        
        if not seg_path:
            logger.warning(f"⚠️ No SEG directory found for {patient_id}")
            results['failed'] += 1
            continue
        
        # Check if CT NIfTI exists
        ct_nifti_path = os.path.join(ct_nifti_root, f"{patient_id}.nii.gz")
        if not os.path.exists(ct_nifti_path):
            logger.warning(f"⚠️ CT NIfTI not found for {patient_id}: {ct_nifti_path}")
            results['failed'] += 1
            continue
        
        # Process patient
        patient_result = process_dicom_seg(
            patient_id=patient_id,
            seg_path=seg_path,
            reference_ct_path=ct_nifti_path,
            output_dir=output_root,
            target_segments=target_segments
        )
        
        results['patient_results'].append(patient_result)
        
        if patient_result['success']:
            results['successful'] += 1
        else:
            results['failed'] += 1
    
    # Save log
    log_path = os.path.join(output_root, 'seg_extraction_log.json')
    with open(log_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"\n{'='*70}")
    logger.info(f"BATCH COMPLETE: {results['successful']}/{results['total']} successful")
    logger.info(f"Log saved: {log_path}")
    logger.info(f"{'='*70}\n")
    
    return results


if __name__ == "__main__":
    # Example usage
    print("DICOM SEG Mask Extraction Module")
    print("Import this module to use mask extraction functions")
