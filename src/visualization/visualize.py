"""
Visualization tools for NSCLC radiomics project.

Creates:
1. CT + mask overlay visualizations
2. 3D volume renderings
3. Quality control plots

Author: GitHub Copilot
Date: 2025
"""

import os
import logging
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
from matplotlib import patches
import matplotlib.cm as cm


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_overlay_visualization(
    ct_path: str,
    mask_path: str,
    output_path: str,
    num_slices: int = 6,
    slice_indices: Optional[List[int]] = None,
    alpha: float = 0.3,
    cmap_ct: str = 'gray',
    cmap_mask: str = 'Reds'
):
    """
    Create CT + mask overlay visualization showing multiple slices.
    
    Args:
        ct_path: Path to CT NIfTI file
        mask_path: Path to mask NIfTI file
        output_path: Output path for visualization image
        num_slices: Number of slices to show (if slice_indices not provided)
        slice_indices: Specific slice indices to visualize
        alpha: Transparency of mask overlay
        cmap_ct: Colormap for CT
        cmap_mask: Colormap for mask overlay
    """
    logger.info(f"Creating overlay visualization: {os.path.basename(ct_path)}")
    
    # Load images
    ct_image = sitk.ReadImage(ct_path)
    mask_image = sitk.ReadImage(mask_path)
    
    # Convert to numpy arrays
    ct_array = sitk.GetArrayFromImage(ct_image)  # (depth, height, width)
    mask_array = sitk.GetArrayFromImage(mask_image)
    
    # Find slices with tumor
    tumor_slices = np.where(np.any(mask_array > 0, axis=(1, 2)))[0]
    
    if len(tumor_slices) == 0:
        logger.warning("No tumor found in mask!")
        return
    
    # Select slices to visualize
    if slice_indices is None:
        # Evenly distribute slices across tumor region
        if len(tumor_slices) >= num_slices:
            step = len(tumor_slices) // num_slices
            slice_indices = tumor_slices[::step][:num_slices]
        else:
            slice_indices = tumor_slices
    
    # Create figure
    n_cols = 3
    n_rows = (len(slice_indices) + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    for idx, slice_idx in enumerate(slice_indices):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col]
        
        # Get slice
        ct_slice = ct_array[slice_idx, :, :]
        mask_slice = mask_array[slice_idx, :, :]
        
        # Display CT
        ax.imshow(ct_slice, cmap=cmap_ct, aspect='auto')
        
        # Overlay mask
        masked = np.ma.masked_where(mask_slice == 0, mask_slice)
        ax.imshow(masked, cmap=cmap_mask, alpha=alpha, aspect='auto')
        
        # Calculate tumor info for this slice
        tumor_pixels = np.sum(mask_slice > 0)
        tumor_percentage = (tumor_pixels / mask_slice.size) * 100
        
        ax.set_title(f'Slice {slice_idx}\nTumor: {tumor_pixels} px ({tumor_percentage:.2f}%)')
        ax.axis('off')
    
    # Hide unused subplots
    for idx in range(len(slice_indices), n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"✅ Saved overlay: {output_path}")


def create_3d_volume_visualization(
    ct_path: str,
    mask_path: str,
    output_path: str,
    views: List[str] = ['axial', 'sagittal', 'coronal']
):
    """
    Create 3D volume visualization showing multiple anatomical views.
    
    Args:
        ct_path: Path to CT NIfTI file
        mask_path: Path to mask NIfTI file
        output_path: Output path for visualization
        views: List of views to show ('axial', 'sagittal', 'coronal')
    """
    logger.info(f"Creating 3D volume visualization: {os.path.basename(ct_path)}")
    
    # Load images
    ct_image = sitk.ReadImage(ct_path)
    mask_image = sitk.ReadImage(mask_path)
    
    ct_array = sitk.GetArrayFromImage(ct_image)
    mask_array = sitk.GetArrayFromImage(mask_image)
    
    # Find center of mass of tumor
    tumor_coords = np.argwhere(mask_array > 0)
    if len(tumor_coords) == 0:
        logger.warning("No tumor found!")
        return
    
    center = tumor_coords.mean(axis=0).astype(int)
    
    # Create figure
    fig, axes = plt.subplots(1, len(views), figsize=(6*len(views), 6))
    if len(views) == 1:
        axes = [axes]
    
    for ax, view in zip(axes, views):
        if view == 'axial':
            # XY plane (depth slice)
            ct_slice = ct_array[center[0], :, :]
            mask_slice = mask_array[center[0], :, :]
            title = f'Axial (Z={center[0]})'
        elif view == 'sagittal':
            # YZ plane (width slice)
            ct_slice = ct_array[:, :, center[2]]
            mask_slice = mask_array[:, :, center[2]]
            title = f'Sagittal (X={center[2]})'
        elif view == 'coronal':
            # XZ plane (height slice)
            ct_slice = ct_array[:, center[1], :]
            mask_slice = mask_array[:, center[1], :]
            title = f'Coronal (Y={center[1]})'
        
        # Display
        ax.imshow(ct_slice, cmap='gray', aspect='auto')
        masked = np.ma.masked_where(mask_slice == 0, mask_slice)
        ax.imshow(masked, cmap='Reds', alpha=0.4, aspect='auto')
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"✅ Saved 3D view: {output_path}")


def create_quality_control_plots(
    ct_paths: List[str],
    mask_paths: List[str],
    output_path: str
):
    """
    Create quality control plots showing dataset statistics.
    
    Args:
        ct_paths: List of CT NIfTI file paths
        mask_paths: List of mask NIfTI file paths
        output_path: Output path for QC plot
    """
    logger.info(f"Creating quality control plots for {len(ct_paths)} patients...")
    
    # Collect statistics
    volumes = []
    intensities_mean = []
    intensities_std = []
    slice_counts = []
    tumor_volumes_cm3 = []
    
    for ct_path, mask_path in zip(ct_paths, mask_paths):
        try:
            ct_image = sitk.ReadImage(ct_path)
            mask_image = sitk.ReadImage(mask_path)
            
            ct_array = sitk.GetArrayFromImage(ct_image)
            mask_array = sitk.GetArrayFromImage(mask_image)
            
            # CT statistics
            intensities_mean.append(np.mean(ct_array))
            intensities_std.append(np.std(ct_array))
            slice_counts.append(ct_array.shape[0])
            
            # Tumor volume
            spacing = ct_image.GetSpacing()
            voxel_volume_mm3 = spacing[0] * spacing[1] * spacing[2]
            tumor_voxels = np.sum(mask_array > 0)
            tumor_volume_cm3 = (tumor_voxels * voxel_volume_mm3) / 1000.0
            tumor_volumes_cm3.append(tumor_volume_cm3)
            
        except Exception as e:
            logger.warning(f"Error processing {os.path.basename(ct_path)}: {e}")
    
    # Create plots
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # Plot 1: Tumor volumes
    axes[0, 0].hist(tumor_volumes_cm3, bins=20, color='steelblue', edgecolor='black')
    axes[0, 0].set_xlabel('Tumor Volume (cm³)', fontsize=12)
    axes[0, 0].set_ylabel('Frequency', fontsize=12)
    axes[0, 0].set_title('Tumor Volume Distribution', fontsize=14, fontweight='bold')
    axes[0, 0].axvline(np.mean(tumor_volumes_cm3), color='red', linestyle='--', 
                       label=f'Mean: {np.mean(tumor_volumes_cm3):.1f} cm³')
    axes[0, 0].legend()
    
    # Plot 2: CT intensities (mean)
    axes[0, 1].hist(intensities_mean, bins=20, color='lightcoral', edgecolor='black')
    axes[0, 1].set_xlabel('Mean CT Intensity', fontsize=12)
    axes[0, 1].set_ylabel('Frequency', fontsize=12)
    axes[0, 1].set_title('CT Mean Intensity Distribution', fontsize=14, fontweight='bold')
    
    # Plot 3: CT intensities (std)
    axes[0, 2].hist(intensities_std, bins=20, color='lightgreen', edgecolor='black')
    axes[0, 2].set_xlabel('Std CT Intensity', fontsize=12)
    axes[0, 2].set_ylabel('Frequency', fontsize=12)
    axes[0, 2].set_title('CT Std Intensity Distribution', fontsize=14, fontweight='bold')
    
    # Plot 4: Slice counts
    axes[1, 0].hist(slice_counts, bins=20, color='plum', edgecolor='black')
    axes[1, 0].set_xlabel('Number of Slices', fontsize=12)
    axes[1, 0].set_ylabel('Frequency', fontsize=12)
    axes[1, 0].set_title('Slice Count Distribution', fontsize=14, fontweight='bold')
    
    # Plot 5: Tumor volume vs slices
    axes[1, 1].scatter(slice_counts, tumor_volumes_cm3, alpha=0.6, color='teal')
    axes[1, 1].set_xlabel('Number of Slices', fontsize=12)
    axes[1, 1].set_ylabel('Tumor Volume (cm³)', fontsize=12)
    axes[1, 1].set_title('Tumor Volume vs Slice Count', fontsize=14, fontweight='bold')
    
    # Plot 6: Summary statistics
    axes[1, 2].axis('off')
    stats_text = f"""
    Dataset Statistics
    ==================
    Patients: {len(ct_paths)}
    
    Tumor Volume:
      Mean: {np.mean(tumor_volumes_cm3):.2f} cm³
      Std: {np.std(tumor_volumes_cm3):.2f} cm³
      Min: {np.min(tumor_volumes_cm3):.2f} cm³
      Max: {np.max(tumor_volumes_cm3):.2f} cm³
    
    CT Intensity:
      Mean: {np.mean(intensities_mean):.2f}
      Std: {np.mean(intensities_std):.2f}
    
    Slices:
      Mean: {np.mean(slice_counts):.1f}
      Range: [{np.min(slice_counts)}, {np.max(slice_counts)}]
    """
    axes[1, 2].text(0.1, 0.5, stats_text, fontsize=11, family='monospace',
                    verticalalignment='center')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"✅ Saved QC plots: {output_path}")


def batch_visualize(
    ct_root: str,
    mask_root: str,
    output_root: str,
    num_patients: int = 5,
    mask_pattern: str = "*_Neoplasm,_Primary.nii.gz"
):
    """
    Create visualizations for multiple patients.
    
    Args:
        ct_root: Root directory with CT files
        mask_root: Root directory with mask files
        output_root: Output directory for visualizations
        num_patients: Number of patients to visualize
        mask_pattern: Pattern to match tumor masks
    """
    logger.info(f"\n{'='*70}")
    logger.info(f"BATCH VISUALIZATION")
    logger.info(f"{'='*70}")
    
    # Find CT-mask pairs
    ct_files = sorted(list(Path(ct_root).glob("*.nii.gz")))[:num_patients]
    
    os.makedirs(output_root, exist_ok=True)
    os.makedirs(os.path.join(output_root, 'overlays'), exist_ok=True)
    os.makedirs(os.path.join(output_root, '3d_views'), exist_ok=True)
    
    ct_paths_for_qc = []
    mask_paths_for_qc = []
    
    for ct_file in ct_files:
        patient_id = ct_file.stem.replace('.nii', '')
        
        # Find corresponding mask
        mask_files = list(Path(mask_root).glob(f"{patient_id}{mask_pattern}"))
        
        if not mask_files:
            logger.warning(f"No mask found for {patient_id}")
            continue
        
        mask_file = mask_files[0]
        
        # Create overlay visualization
        overlay_output = os.path.join(output_root, 'overlays', f"{patient_id}_overlay.png")
        create_overlay_visualization(
            str(ct_file), str(mask_file), overlay_output
        )
        
        # Create 3D view
        view3d_output = os.path.join(output_root, '3d_views', f"{patient_id}_3dview.png")
        create_3d_volume_visualization(
            str(ct_file), str(mask_file), view3d_output
        )
        
        ct_paths_for_qc.append(str(ct_file))
        mask_paths_for_qc.append(str(mask_file))
    
    # Create QC plots
    qc_output = os.path.join(output_root, 'quality_control.png')
    create_quality_control_plots(ct_paths_for_qc, mask_paths_for_qc, qc_output)
    
    logger.info(f"\n✅ Visualizations complete!")
    logger.info(f"Output directory: {output_root}")


if __name__ == "__main__":
    print("Visualization Module for NSCLC Radiomics")
    print("Import this module to use visualization functions")
