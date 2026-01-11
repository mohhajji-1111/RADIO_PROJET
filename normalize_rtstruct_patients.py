"""
Script pour normaliser les CTs et masques RTSTRUCT
Adaptation de normalize_all_patients.py pour utiliser masques RTSTRUCT
"""

import numpy as np
import SimpleITK as sitk
from pathlib import Path
from typing import Tuple
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# Paramètres de normalisation
LUNG_WINDOW_CENTER = -600
LUNG_WINDOW_WIDTH = 1500
TARGET_SIZE = (256, 256)

def apply_lung_window(ct_array: np.ndarray) -> np.ndarray:
    """Applique le fenêtrage pulmonaire."""
    window_min = LUNG_WINDOW_CENTER - (LUNG_WINDOW_WIDTH / 2)
    window_max = LUNG_WINDOW_CENTER + (LUNG_WINDOW_WIDTH / 2)
    return np.clip(ct_array, window_min, window_max)

def normalize_intensity(ct_array: np.ndarray) -> np.ndarray:
    """Normalise les intensités (Z-score)."""
    mean = np.mean(ct_array)
    std = np.std(ct_array)
    return (ct_array - mean) / (std + 1e-8)

def resize_slice(slice_2d: np.ndarray, target_size: Tuple[int, int], is_mask: bool = False) -> np.ndarray:
    """Redimensionne une slice 2D."""
    from scipy.ndimage import zoom
    
    current_shape = slice_2d.shape
    zoom_factors = (target_size[0] / current_shape[0], target_size[1] / current_shape[1])
    
    if is_mask:
        # Nearest neighbor pour masques
        resized = zoom(slice_2d, zoom_factors, order=0)
    else:
        # Bilinear pour CT
        resized = zoom(slice_2d, zoom_factors, order=1)
    
    return resized

def normalize_patient(
    ct_path: Path,
    mask_path: Path,
    output_dir: Path,
    patient_id: str
) -> bool:
    """Normalise un patient (CT + masque)."""
    
    try:
        # Charger CT et masque
        ct_img = sitk.ReadImage(str(ct_path))
        mask_img = sitk.ReadImage(str(mask_path))
        
        ct_array = sitk.GetArrayFromImage(ct_img)
        mask_array = sitk.GetArrayFromImage(mask_img)
        
        # Vérifier dimensions
        if ct_array.shape != mask_array.shape:
            logger.error(f"{patient_id}: Dimensions incompatibles CT={ct_array.shape} vs Mask={mask_array.shape}")
            return False
        
        # 1. Appliquer fenêtrage sur CT
        ct_windowed = apply_lung_window(ct_array)
        
        # 2. Normaliser intensités CT
        ct_normalized = normalize_intensity(ct_windowed)
        
        # 3. Redimensionner slice par slice
        n_slices = ct_array.shape[0]
        ct_resized = np.zeros((n_slices, TARGET_SIZE[0], TARGET_SIZE[1]), dtype=np.float32)
        mask_resized = np.zeros((n_slices, TARGET_SIZE[0], TARGET_SIZE[1]), dtype=np.uint8)
        
        for z in range(n_slices):
            ct_resized[z] = resize_slice(ct_normalized[z], TARGET_SIZE, is_mask=False)
            mask_resized[z] = resize_slice(mask_array[z], TARGET_SIZE, is_mask=True)
        
        # 4. Binariser masque
        mask_binary = (mask_resized > 0.5).astype(np.uint8)
        
        # 5. Créer images SimpleITK
        ct_final = sitk.GetImageFromArray(ct_resized)
        mask_final = sitk.GetImageFromArray(mask_binary)
        
        # Copier métadonnées spatiales (spacing, origin, direction)
        spacing = list(ct_img.GetSpacing())
        spacing[0] = spacing[0] * (ct_array.shape[2] / TARGET_SIZE[1])
        spacing[1] = spacing[1] * (ct_array.shape[1] / TARGET_SIZE[0])
        
        ct_final.SetSpacing(spacing)
        mask_final.SetSpacing(spacing)
        ct_final.SetOrigin(ct_img.GetOrigin())
        mask_final.SetOrigin(ct_img.GetOrigin())
        ct_final.SetDirection(ct_img.GetDirection())
        mask_final.SetDirection(ct_img.GetDirection())
        
        # 6. Sauvegarder
        ct_output = output_dir / f"{patient_id}_ct_normalized.nii.gz"
        mask_output = output_dir / f"{patient_id}_mask_normalized.nii.gz"
        
        sitk.WriteImage(ct_final, str(ct_output))
        sitk.WriteImage(mask_final, str(mask_output))
        
        tumor_voxels = mask_binary.sum()
        logger.info(f"{patient_id}: OK - {tumor_voxels} voxels tumeur")
        
        return True
        
    except Exception as e:
        logger.error(f"{patient_id}: ERREUR - {e}")
        return False

def main():
    """Normalisation de tous les patients avec masques RTSTRUCT."""
    
    print("="*70)
    print("NORMALISATION DES PATIENTS (CT + MASQUES RTSTRUCT)")
    print("="*70)
    
    # Chemins
    ct_dir = Path("DATA/processed/images_nifti")
    mask_dir = Path("DATA/processed/masks_rtstruct")
    output_dir = Path("DATA/processed/normalized_rtstruct")
    
    # Créer dossier de sortie
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Lister les masques RTSTRUCT disponibles
    mask_files = sorted(mask_dir.glob("LUNG1-*_mask.nii.gz"))
    
    print(f"\nMasques RTSTRUCT trouves: {len(mask_files)}")
    print(f"Sortie: {output_dir}\n")
    print("="*70)
    
    # Statistiques
    success_count = 0
    failed_patients = []
    
    # Traiter chaque patient
    for i, mask_path in enumerate(mask_files):
        patient_id = mask_path.name.replace('_mask.nii.gz', '')
        ct_path = ct_dir / f"{patient_id}.nii.gz"
        
        print(f"[{i+1}/{len(mask_files)}] {patient_id}... ", end='', flush=True)
        
        if not ct_path.exists():
            print("CT MANQUANT")
            failed_patients.append(patient_id)
            continue
        
        success = normalize_patient(ct_path, mask_path, output_dir, patient_id)
        
        if success:
            success_count += 1
        else:
            failed_patients.append(patient_id)
    
    # Résumé
    print("\n" + "="*70)
    print("RESUME DE LA NORMALISATION")
    print("="*70)
    print(f"\nSucces: {success_count}/{len(mask_files)} patients")
    print(f"Echecs: {len(failed_patients)} patients")
    
    if failed_patients:
        print(f"\nPatients echoues:")
        for pid in failed_patients[:10]:
            print(f"   - {pid}")
        if len(failed_patients) > 10:
            print(f"   ... et {len(failed_patients) - 10} autres")
    
    print("\n" + "="*70)
    print("NORMALISATION TERMINEE!")
    print("="*70)
    print(f"\nDonnees sauvegardees dans: {output_dir}")
    if len(mask_files) > 0:
        print(f"Taux de succes: {success_count/len(mask_files)*100:.1f}%")

if __name__ == "__main__":
    main()
