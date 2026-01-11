"""
Script pour calculer les volumes des ROIs (régions d'intérêt) multi-organes.
Calcule le volume en cm³ pour chaque organe de chaque patient.
"""

import SimpleITK as sitk
import numpy as np
import pandas as pd
from pathlib import Path
import logging
from tqdm import tqdm

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Chemins
MASKS_DIR = Path("DATA/processed/masks_multi_organ")
OUTPUT_CSV = Path("roi_volumes_analysis.csv")
OUTPUT_STATS = Path("roi_volumes_statistics.txt")

# Mapping des labels vers noms des organes
ORGAN_NAMES = {
    0: "Background",
    1: "GTV",
    2: "PTV", 
    3: "Poumon_Droit",
    4: "Poumon_Gauche",
    5: "Coeur",
    6: "Oesophage",
    7: "Moelle"
}


def calculate_volume_cm3(mask_array, spacing):
    """
    Calcule le volume en cm³ pour un masque binaire.
    
    Args:
        mask_array: Array numpy 3D avec 0/1
        spacing: Tuple (spacing_z, spacing_y, spacing_x) en mm
    
    Returns:
        Volume en cm³
    """
    # Nombre de voxels
    num_voxels = np.sum(mask_array > 0)
    
    # Volume d'un voxel en mm³
    voxel_volume_mm3 = spacing[0] * spacing[1] * spacing[2]
    
    # Volume total en mm³
    volume_mm3 = num_voxels * voxel_volume_mm3
    
    # Conversion en cm³ (1 cm³ = 1000 mm³)
    volume_cm3 = volume_mm3 / 1000.0
    
    return volume_cm3


def process_patient(mask_path):
    """
    Traite un patient et calcule les volumes pour chaque organe.
    
    Args:
        mask_path: Chemin vers le masque multi-organe (.nii.gz)
    
    Returns:
        Dict avec patient_id et volumes par organe
    """
    patient_id = mask_path.stem.replace("_mask", "")
    
    try:
        # Charger le masque
        mask_sitk = sitk.ReadImage(str(mask_path))
        mask_array = sitk.GetArrayFromImage(mask_sitk)
        spacing = mask_sitk.GetSpacing()[::-1]  # (x,y,z) -> (z,y,x)
        
        # Dictionnaire des résultats
        results = {"patient_id": patient_id}
        
        # Calculer volume pour chaque organe
        unique_labels = np.unique(mask_array)
        
        for label in range(1, 8):  # Labels 1-7 (skip background)
            organ_name = ORGAN_NAMES[label]
            
            if label in unique_labels:
                # Créer masque binaire pour cet organe
                organ_mask = (mask_array == label).astype(np.uint8)
                volume = calculate_volume_cm3(organ_mask, spacing)
                results[f"{organ_name}_volume_cm3"] = volume
            else:
                # Organe absent
                results[f"{organ_name}_volume_cm3"] = 0.0
        
        return results
        
    except Exception as e:
        logger.error(f"Erreur pour {patient_id}: {e}")
        return None


def main():
    """Fonction principale."""
    logger.info("=== CALCUL DES VOLUMES ROIs ===")
    
    # Lister tous les masques
    mask_files = sorted(MASKS_DIR.glob("*.nii.gz"))
    logger.info(f"Trouvé {len(mask_files)} masques multi-organes")
    
    if len(mask_files) == 0:
        logger.error(f"Aucun masque trouvé dans {MASKS_DIR}")
        return
    
    # Traiter chaque patient
    all_results = []
    
    for mask_path in tqdm(mask_files, desc="Calcul des volumes"):
        results = process_patient(mask_path)
        if results:
            all_results.append(results)
    
    # Créer DataFrame
    df = pd.DataFrame(all_results)
    
    # Trier par patient_id
    df = df.sort_values("patient_id")
    
    # Sauvegarder CSV
    df.to_csv(OUTPUT_CSV, index=False, float_format='%.2f')
    logger.info(f"✅ CSV sauvegardé: {OUTPUT_CSV}")
    
    # Calculer statistiques globales
    stats_lines = []
    stats_lines.append("=" * 80)
    stats_lines.append("STATISTIQUES DES VOLUMES ROIs (cm³)")
    stats_lines.append("=" * 80)
    stats_lines.append("")
    
    for label in range(1, 8):
        organ_name = ORGAN_NAMES[label]
        col_name = f"{organ_name}_volume_cm3"
        
        if col_name in df.columns:
            volumes = df[col_name]
            non_zero = volumes[volumes > 0]
            
            stats_lines.append(f"{organ_name}:")
            stats_lines.append(f"  Patients avec cet organe: {len(non_zero)}/{len(df)} ({len(non_zero)/len(df)*100:.1f}%)")
            
            if len(non_zero) > 0:
                stats_lines.append(f"  Volume moyen: {non_zero.mean():.2f} cm³")
                stats_lines.append(f"  Volume médian: {non_zero.median():.2f} cm³")
                stats_lines.append(f"  Écart-type: {non_zero.std():.2f} cm³")
                stats_lines.append(f"  Min: {non_zero.min():.2f} cm³")
                stats_lines.append(f"  Max: {non_zero.max():.2f} cm³")
            else:
                stats_lines.append("  Aucun patient avec volume > 0")
            
            stats_lines.append("")
    
    stats_lines.append("=" * 80)
    stats_lines.append(f"Total patients analysés: {len(df)}")
    stats_lines.append("=" * 80)
    
    # Afficher et sauvegarder statistiques
    stats_text = "\n".join(stats_lines)
    print("\n" + stats_text)
    
    with open(OUTPUT_STATS, 'w', encoding='utf-8') as f:
        f.write(stats_text)
    
    logger.info(f"✅ Statistiques sauvegardées: {OUTPUT_STATS}")
    
    # Résumé rapide
    print("\n" + "=" * 80)
    print("RÉSUMÉ RAPIDE:")
    print("=" * 80)
    for label in range(1, 8):
        organ_name = ORGAN_NAMES[label]
        col_name = f"{organ_name}_volume_cm3"
        if col_name in df.columns:
            volumes = df[col_name]
            non_zero = volumes[volumes > 0]
            if len(non_zero) > 0:
                print(f"{organ_name:20s}: {non_zero.mean():8.2f} cm³ (±{non_zero.std():.2f}) - {len(non_zero)} patients")


if __name__ == "__main__":
    main()
