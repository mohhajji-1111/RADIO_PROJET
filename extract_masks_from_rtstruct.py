"""
Script pour extraire les masques de tumeur depuis fichiers RTSTRUCT
Phase 2.5 Alternative - Utilise RTSTRUCT au lieu de DICOM SEG
"""

import numpy as np
import SimpleITK as sitk
import pydicom
from pathlib import Path
import logging
import os
from typing import List, Tuple

# Configuration logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def find_rtstruct_file(patient_dir: Path) -> Path:
    """Trouve le fichier RTSTRUCT dans le dossier patient."""
    for root, dirs, files in os.walk(patient_dir):
        for file in files:
            if file.endswith('.dcm'):
                dcm_path = Path(root) / file
                try:
                    dcm = pydicom.dcmread(str(dcm_path), stop_before_pixels=True)
                    if hasattr(dcm, 'Modality') and dcm.Modality == 'RTSTRUCT':
                        return dcm_path
                except:
                    continue
    return None

def find_ct_series_dir(patient_dir: Path) -> Path:
    """Trouve le dossier contenant la série CT."""
    for subdir in patient_dir.iterdir():
        if subdir.is_dir():
            for subsubdir in subdir.iterdir():
                if subsubdir.is_dir():
                    # Vérifier si contient des DICOM CT
                    dcm_files = list(subsubdir.glob('*.dcm'))
                    if dcm_files:
                        try:
                            dcm = pydicom.dcmread(str(dcm_files[0]), stop_before_pixels=True)
                            if hasattr(dcm, 'Modality') and dcm.Modality == 'CT':
                                return subsubdir
                        except:
                            continue
    return None

def contour_to_mask(contour_data: List[float], ct_shape: Tuple, ct_origin, ct_spacing) -> np.ndarray:
    """Convertit les points de contour RTSTRUCT en masque 2D."""
    # Reshape contour data: [x1, y1, z1, x2, y2, z2, ...] -> [(x1,y1,z1), (x2,y2,z2), ...]
    points = np.array(contour_data).reshape(-1, 3)
    
    # Convertir coordonnées physiques → indices pixels
    points_idx = np.zeros_like(points)
    for i in range(3):
        points_idx[:, i] = (points[:, i] - ct_origin[i]) / ct_spacing[i]
    
    # Créer masque 2D pour cette slice
    from PIL import Image, ImageDraw
    mask_2d = Image.new('L', (ct_shape[1], ct_shape[0]), 0)
    draw = ImageDraw.Draw(mask_2d)
    
    # Dessiner polygone (XY seulement)
    xy_points = [(p[0], p[1]) for p in points_idx]
    draw.polygon(xy_points, outline=1, fill=1)
    
    return np.array(mask_2d, dtype=np.uint8)

def extract_tumor_mask_from_rtstruct(
    patient_id: str,
    dataset_root: Path,
    ct_nifti_path: Path,
    output_dir: Path
) -> bool:
    """
    Extrait le masque de tumeur depuis RTSTRUCT.
    
    Args:
        patient_id: ID du patient (ex: LUNG1-001)
        dataset_root: Chemin vers DATA/NSCLC-Radiomics
        ct_nifti_path: Chemin vers le CT NIfTI de référence
        output_dir: Dossier de sortie pour les masques
        
    Returns:
        True si succès, False sinon
    """
    try:
        patient_dir = dataset_root / patient_id
        
        # 1. Trouver fichier RTSTRUCT
        rtstruct_path = find_rtstruct_file(patient_dir)
        if not rtstruct_path:
            logger.warning(f"{patient_id}: RTSTRUCT non trouvé")
            return False
        
        logger.info(f"{patient_id}: RTSTRUCT trouvé - {rtstruct_path.name}")
        
        # 2. Charger RTSTRUCT avec pydicom
        rtstruct = pydicom.dcmread(str(rtstruct_path))
        
        # 3. Vérifier que c'est bien un RTSTRUCT
        if not hasattr(rtstruct, 'StructureSetROISequence'):
            logger.warning(f"{patient_id}: Pas un fichier RTSTRUCT valide")
            return False
        
        # 4. Lister les ROIs disponibles
        roi_names = []
        roi_dict = {}
        for roi in rtstruct.StructureSetROISequence:
            roi_number = roi.ROINumber
            roi_name = roi.ROIName
            roi_names.append(roi_name)
            roi_dict[roi_number] = roi_name
        
        logger.info(f"{patient_id}: {len(roi_names)} ROIs trouvées - {roi_names}")
        
        # 5. Identifier les ROIs tumorales (GTV-*)
        tumor_roi_numbers = [num for num, name in roi_dict.items() if 'GTV' in name.upper()]
        
        if not tumor_roi_numbers:
            logger.warning(f"{patient_id}: Aucune ROI tumorale (GTV-*) trouvée")
            return False
        
        tumor_rois = [roi_dict[num] for num in tumor_roi_numbers]
        logger.info(f"{patient_id}: ROIs tumorales trouvées - {tumor_rois}")
        
        # 6. Charger le CT NIfTI de référence
        ct_img = sitk.ReadImage(str(ct_nifti_path))
        ct_array = sitk.GetArrayFromImage(ct_img)
        ct_spacing = ct_img.GetSpacing()
        ct_origin = ct_img.GetOrigin()
        
        # 7. Créer masque 3D vide
        mask_array = np.zeros_like(ct_array, dtype=np.uint8)
        
        # 8. Extraire contours pour chaque ROI tumorale
        if not hasattr(rtstruct, 'ROIContourSequence'):
            logger.warning(f"{patient_id}: Pas de contours dans RTSTRUCT")
            return False
        
        for roi_contour in rtstruct.ROIContourSequence:
            roi_number = roi_contour.ReferencedROINumber
            
            if roi_number not in tumor_roi_numbers:
                continue
            
            roi_name = roi_dict[roi_number]
            
            if not hasattr(roi_contour, 'ContourSequence'):
                continue
            
            # Traiter chaque contour (slice)
            for contour in roi_contour.ContourSequence:
                if not hasattr(contour, 'ContourData'):
                    continue
                
                contour_data = contour.ContourData
                
                # Extraire numéro de slice Z
                z_coord = contour_data[2]  # Premier point Z
                z_idx = int(round((z_coord - ct_origin[2]) / ct_spacing[2]))
                
                if 0 <= z_idx < ct_array.shape[0]:
                    try:
                        slice_mask = contour_to_mask(
                            contour_data,
                            (ct_array.shape[1], ct_array.shape[2]),
                            ct_origin,
                            ct_spacing
                        )
                        mask_array[z_idx] = np.logical_or(mask_array[z_idx], slice_mask).astype(np.uint8)
                    except Exception as e:
                        logger.debug(f"{patient_id}: Erreur contour slice {z_idx} - {e}")
                        continue
            
            logger.info(f"{patient_id}: ROI '{roi_name}' extraite")
        
        # 9. Vérifier que le masque contient des tumeurs
        tumor_voxels = mask_array.sum()
        if tumor_voxels == 0:
            logger.warning(f"{patient_id}: Masque vide (0 voxels tumeur)")
            return False
        
        logger.info(f"{patient_id}: Masque fusionné - {tumor_voxels} voxels tumeur")
        
        # 10. Créer image SimpleITK avec les mêmes métadonnées que le CT
        mask_img = sitk.GetImageFromArray(mask_array)
        mask_img.CopyInformation(ct_img)
        
        # 11. Sauvegarder
        output_path = output_dir / f"{patient_id}_mask.nii.gz"
        sitk.WriteImage(mask_img, str(output_path))
        
        logger.info(f"{patient_id}: ✅ Masque sauvegardé - {output_path.name}")
        return True
        
    except Exception as e:
        logger.error(f"{patient_id}: ❌ Erreur - {e}")
        return False

def main():
    """Extraction complète de tous les masques depuis RTSTRUCT."""
    
    print("="*70)
    print("EXTRACTION DES MASQUES DEPUIS RTSTRUCT")
    print("="*70)
    
    # Chemins
    dataset_root = Path("DATA/NSCLC-Radiomics")
    ct_nifti_dir = Path("DATA/processed/images_nifti")
    output_dir = Path("DATA/processed/masks_rtstruct")
    
    # Créer dossier de sortie
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Lister tous les patients avec CT NIfTI
    ct_files = sorted(ct_nifti_dir.glob("LUNG1-*.nii.gz"))
    
    print(f"\nPatients trouves: {len(ct_files)}")
    print(f"Sortie: {output_dir}")
    print("\n" + "="*70)
    
    # Statistiques
    success_count = 0
    failed_patients = []
    
    # Traiter chaque patient
    for i, ct_path in enumerate(ct_files):
        # ct_path.stem donne "LUNG1-001.nii" à cause de .nii.gz
        # On doit retirer .nii aussi
        patient_id = ct_path.name.replace('.nii.gz', '')  # LUNG1-001
        
        print(f"[{i+1}/{len(ct_files)}] {patient_id}...", end=' ', flush=True)
        
        success = extract_tumor_mask_from_rtstruct(
            patient_id=patient_id,
            dataset_root=dataset_root,
            ct_nifti_path=ct_path,
            output_dir=output_dir
        )
        
        if success:
            success_count += 1
            print("OK")
        else:
            failed_patients.append(patient_id)
            print("FAILED")
    
    # Résumé final
    print("\n" + "="*70)
    print("RESUME DE L'EXTRACTION")
    print("="*70)
    print(f"\nSucces: {success_count}/{len(ct_files)} patients")
    print(f"Echecs: {len(failed_patients)} patients")
    
    if failed_patients:
        print(f"\nPatients echoues:")
        for pid in failed_patients[:10]:
            print(f"   - {pid}")
        if len(failed_patients) > 10:
            print(f"   ... et {len(failed_patients) - 10} autres")
    
    print("\n" + "="*70)
    print("EXTRACTION TERMINEE!")
    print("="*70)
    print(f"\nMasques sauvegardes dans: {output_dir}")
    if len(ct_files) > 0:
        print(f"Taux de succes: {success_count/len(ct_files)*100:.1f}%")
    else:
        print(f"Aucun fichier CT trouve dans {ct_nifti_dir}")

if __name__ == "__main__":
    main()
