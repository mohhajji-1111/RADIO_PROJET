"""
Script d'extraction multi-organes depuis RTSTRUCT
Extrait plusieurs types de ROIs avec labels différents:
  - Label 1: GTV (tumeurs)
  - Label 2: PTV (zone de planification)
  - Label 3: Poumon droit
  - Label 4: Poumon gauche
  - Label 5: Coeur
  - Label 6-10: Autres organes à risque

Auteur: Copilot
Date: 21 Nov 2025
"""

import os
import numpy as np
import pydicom
import SimpleITK as sitk
from pathlib import Path
from PIL import Image, ImageDraw
import logging

# Configuration logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('extraction_multi_organ_output.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Chemins
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "DATA"
RAW_DIR = DATA_DIR / "NSCLC-Radiomics"
PROCESSED_DIR = DATA_DIR / "processed"
NIFTI_DIR = PROCESSED_DIR / "normalized_rtstruct"  # Utilise les CT normalisés
OUTPUT_DIR = PROCESSED_DIR / "masks_multi_organ"

# Créer dossier de sortie
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Mapping des ROIs vers labels
ROI_MAPPING = {
    'GTV': 1,        # Gross Tumor Volume (toutes les tumeurs)
    'PTV': 2,        # Planning Target Volume
    'LUNG_R': 3,     # Poumon droit
    'LUNG_L': 4,     # Poumon gauche
    'HEART': 5,      # Coeur
    'ESOPHAGUS': 6,  # Oesophage
    'SPINAL': 7,     # Moelle épinière
    'TRACHEA': 8,    # Trachée
    'CHEST': 9,      # Paroi thoracique
    'OTHER': 10      # Autres OARs
}

def normalize_roi_name(roi_name):
    """
    Normalise le nom d'une ROI et retourne le label correspondant
    """
    roi_upper = roi_name.upper()
    
    # Tumeurs (GTV-1, GTV-2, GTV1, GTVP, etc.)
    if 'GTV' in roi_upper:
        return 'GTV', ROI_MAPPING['GTV']
    
    # Planning Target Volume
    if 'PTV' in roi_upper:
        return 'PTV', ROI_MAPPING['PTV']
    
    # Poumon droit (variations: LUNG_R, POUMON_D, LUNGR, RIGHT LUNG, etc.)
    if any(x in roi_upper for x in ['LUNG_R', 'LUNG R', 'LUNGR', 'RIGHT LUNG', 
                                      'POUMON_D', 'POUMON D', 'LUNG-R', 'RT LUNG']):
        return 'LUNG_R', ROI_MAPPING['LUNG_R']
    
    # Poumon gauche
    if any(x in roi_upper for x in ['LUNG_L', 'LUNG L', 'LUNGL', 'LEFT LUNG',
                                      'POUMON_G', 'POUMON G', 'LUNG-L', 'LT LUNG']):
        return 'LUNG_L', ROI_MAPPING['LUNG_L']
    
    # Coeur
    if any(x in roi_upper for x in ['HEART', 'COEUR', 'CŒUR']):
        return 'HEART', ROI_MAPPING['HEART']
    
    # Oesophage
    if any(x in roi_upper for x in ['ESOPHAGUS', 'OESOPHAGE', 'ŒSOPHAGE']):
        return 'ESOPHAGUS', ROI_MAPPING['ESOPHAGUS']
    
    # Moelle épinière
    if any(x in roi_upper for x in ['SPINAL', 'CORD', 'MOELLE']):
        return 'SPINAL', ROI_MAPPING['SPINAL']
    
    # Trachée
    if any(x in roi_upper for x in ['TRACHEA', 'TRACHÉE']):
        return 'TRACHEA', ROI_MAPPING['TRACHEA']
    
    # Paroi thoracique
    if any(x in roi_upper for x in ['CHEST', 'WALL', 'PAROI']):
        return 'CHEST', ROI_MAPPING['CHEST']
    
    # Autres
    return 'OTHER', ROI_MAPPING['OTHER']

def find_ct_series_dir(patient_dir):
    """Trouve le répertoire de la série CT"""
    for study_dir in patient_dir.iterdir():
        if not study_dir.is_dir():
            continue
        for series_dir in study_dir.iterdir():
            if series_dir.is_dir() and any(series_dir.glob("*.dcm")):
                return series_dir
    return None

def contour_to_mask(contour_data, ct_spacing, ct_origin, ct_shape):
    """Convertit les coordonnées de contour en masque 2D"""
    points_3d = np.array(contour_data).reshape(-1, 3)
    
    # Conversion coordonnées patient → indices image
    points_ij = np.zeros((len(points_3d), 2), dtype=np.float32)
    points_ij[:, 0] = (points_3d[:, 0] - ct_origin[0]) / ct_spacing[0]  # X → colonnes
    points_ij[:, 1] = (points_3d[:, 1] - ct_origin[1]) / ct_spacing[1]  # Y → lignes
    
    # Créer masque avec PIL
    img = Image.new('L', (ct_shape[1], ct_shape[0]), 0)
    draw = ImageDraw.Draw(img)
    
    points_xy = [(p[0], p[1]) for p in points_ij]
    draw.polygon(points_xy, outline=1, fill=1)
    
    return np.array(img, dtype=np.uint8)

def extract_multi_organ_masks(patient_id):
    """
    Extrait tous les organes d'intérêt pour un patient
    Sauvegarde un masque multi-classes (labels 0-10)
    """
    try:
        patient_dir = RAW_DIR / patient_id
        
        if not patient_dir.exists():
            logger.warning(f"{patient_id}: Dossier introuvable")
            return False
        
        # 1. Trouver RTSTRUCT (chercher dans tous les DICOMs)
        rtstruct_path = None
        for dcm_file in patient_dir.rglob("*.dcm"):
            try:
                dcm = pydicom.dcmread(str(dcm_file), stop_before_pixels=True, force=True)
                if hasattr(dcm, 'Modality') and dcm.Modality == 'RTSTRUCT':
                    rtstruct_path = dcm_file
                    break
            except:
                continue
        
        if not rtstruct_path:
            logger.warning(f"{patient_id}: Pas de fichier RTSTRUCT")
            return False
        
        logger.info(f"{patient_id}: RTSTRUCT trouvé - {rtstruct_path.name}")
        
        # 2. Charger RTSTRUCT
        rtstruct = pydicom.dcmread(str(rtstruct_path))
        
        if not hasattr(rtstruct, 'StructureSetROISequence'):
            logger.warning(f"{patient_id}: Pas de ROIs dans RTSTRUCT")
            return False
        
        # 3. Lister toutes les ROIs et leurs labels
        roi_dict = {}
        roi_labels = {}
        
        for roi in rtstruct.StructureSetROISequence:
            roi_number = roi.ROINumber
            roi_name = roi.ROIName
            normalized_name, label = normalize_roi_name(roi_name)
            
            roi_dict[roi_number] = roi_name
            roi_labels[roi_number] = (normalized_name, label)
        
        logger.info(f"{patient_id}: {len(roi_dict)} ROIs trouvées")
        
        # Afficher les ROIs par catégorie
        roi_summary = {}
        for roi_num, (norm_name, label) in roi_labels.items():
            if norm_name not in roi_summary:
                roi_summary[norm_name] = []
            roi_summary[norm_name].append(roi_dict[roi_num])
        
        for category, rois in sorted(roi_summary.items()):
            logger.info(f"  {category}: {rois}")
        
        # 4. Charger CT NIfTI de référence
        ct_nifti_path = NIFTI_DIR / f"{patient_id}_ct_normalized.nii.gz"
        
        if not ct_nifti_path.exists():
            logger.warning(f"{patient_id}: CT NIfTI introuvable - {ct_nifti_path}")
            return False
        
        ct_img = sitk.ReadImage(str(ct_nifti_path))
        ct_array = sitk.GetArrayFromImage(ct_img)
        ct_spacing = ct_img.GetSpacing()
        ct_origin = ct_img.GetOrigin()
        
        # 5. Créer masque 3D multi-classes
        mask_array = np.zeros_like(ct_array, dtype=np.uint8)
        
        # 6. Extraire tous les contours
        if not hasattr(rtstruct, 'ROIContourSequence'):
            logger.warning(f"{patient_id}: Pas de contours dans RTSTRUCT")
            return False
        
        organ_voxel_counts = {}
        
        for roi_contour in rtstruct.ROIContourSequence:
            roi_number = roi_contour.ReferencedROINumber
            
            if roi_number not in roi_labels:
                continue
            
            norm_name, label = roi_labels[roi_number]
            roi_name = roi_dict[roi_number]
            
            if not hasattr(roi_contour, 'ContourSequence'):
                continue
            
            # Compter voxels pour cette ROI
            roi_voxels = 0
            
            # Traiter chaque contour (slice)
            for contour in roi_contour.ContourSequence:
                if not hasattr(contour, 'ContourData'):
                    continue
                
                contour_data = contour.ContourData
                
                # Extraire numéro de slice Z
                z_coord = contour_data[2]
                z_idx = int(round((z_coord - ct_origin[2]) / ct_spacing[2]))
                
                if z_idx < 0 or z_idx >= ct_array.shape[0]:
                    continue
                
                # Convertir contour en masque 2D
                slice_mask = contour_to_mask(
                    contour_data,
                    ct_spacing,
                    ct_origin,
                    (ct_array.shape[1], ct_array.shape[2])
                )
                
                # Ajouter au masque avec le label correspondant
                # Si plusieurs organes se chevauchent, priorité au label le plus élevé
                mask_slice = mask_array[z_idx]
                mask_slice[slice_mask == 1] = label
                
                roi_voxels += slice_mask.sum()
            
            if roi_voxels > 0:
                organ_voxel_counts[roi_name] = roi_voxels
                logger.info(f"  [OK] {roi_name} (label {label}): {roi_voxels} voxels")
        
        # 7. Sauvegarder masque multi-classes
        if len(organ_voxel_counts) == 0:
            logger.warning(f"{patient_id}: Aucun organe extrait")
            return False
        
        mask_img = sitk.GetImageFromArray(mask_array)
        mask_img.SetSpacing(ct_img.GetSpacing())
        mask_img.SetOrigin(ct_img.GetOrigin())
        mask_img.SetDirection(ct_img.GetDirection())
        
        output_path = OUTPUT_DIR / f"{patient_id}_multi_organ.nii.gz"
        sitk.WriteImage(mask_img, str(output_path))
        
        logger.info(f"{patient_id}: [SUCCESS] Masque sauvegarde - {len(organ_voxel_counts)} organes extraits")
        logger.info(f"  Total voxels annotés: {mask_array.sum()} / {mask_array.size} ({100*mask_array.sum()/mask_array.size:.2f}%)")
        
        return True
        
    except Exception as e:
        logger.error(f"{patient_id}: Erreur - {str(e)}", exc_info=True)
        return False

def main():
    """Traite tous les patients"""
    logger.info("="*80)
    logger.info("EXTRACTION MULTI-ORGANES - RTSTRUCT")
    logger.info("="*80)
    
    # Récupérer liste des patients
    patient_dirs = sorted([d for d in RAW_DIR.iterdir() if d.is_dir() and d.name.startswith('LUNG')])
    total_patients = len(patient_dirs)
    
    logger.info(f"Patients trouvés: {total_patients}")
    logger.info(f"Dossier sortie: {OUTPUT_DIR}")
    
    # Statistiques
    success_count = 0
    failed_patients = []
    
    # Traiter chaque patient
    for idx, patient_dir in enumerate(patient_dirs, 1):
        patient_id = patient_dir.name
        logger.info(f"\n[{idx}/{total_patients}] Traitement: {patient_id}")
        
        if extract_multi_organ_masks(patient_id):
            success_count += 1
        else:
            failed_patients.append(patient_id)
    
    # Rapport final
    logger.info("\n" + "="*80)
    logger.info("RAPPORT FINAL")
    logger.info("="*80)
    logger.info(f"[SUCCESS] Reussis: {success_count}/{total_patients} ({100*success_count/total_patients:.1f}%)")
    
    if failed_patients:
        logger.info(f"[FAILED] Echecs: {len(failed_patients)}")
        logger.info(f"  Patients: {', '.join(failed_patients[:10])}")
        if len(failed_patients) > 10:
            logger.info(f"  ... et {len(failed_patients)-10} autres")

if __name__ == "__main__":
    main()
