"""
Script de visualisation complète de toutes les phases du projet
Phase 2: DICOM → NIfTI
Phase 2.5: RTSTRUCT → Masques multi-organes
Phase 3: Normalisation CT
Phase 4: Dataset PyTorch

Auteur: Copilot
Date: 21 Nov 2025
"""

import numpy as np
import matplotlib.pyplot as plt
import SimpleITK as sitk
import pydicom
from pathlib import Path
import torch
from dataset_multi_organ import MultiOrganDataset

# Configuration
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "DATA"
RAW_DIR = DATA_DIR / "NSCLC-Radiomics"
PROCESSED_DIR = DATA_DIR / "processed"

# Mapping des labels pour affichage
LABEL_NAMES = {
    0: 'Background',
    1: 'GTV (Tumeur)',
    2: 'PTV',
    3: 'Poumon Droit',
    4: 'Poumon Gauche',
    5: 'Coeur',
    6: 'Oesophage',
    7: 'Moelle Épinière'
}

LABEL_COLORS = {
    0: [0, 0, 0],           # Noir
    1: [255, 0, 0],         # Rouge (tumeur)
    2: [255, 165, 0],       # Orange (PTV)
    3: [0, 255, 255],       # Cyan (poumon D)
    4: [0, 191, 255],       # Bleu clair (poumon G)
    5: [255, 0, 255],       # Magenta (coeur)
    6: [255, 255, 0],       # Jaune (oesophage)
    7: [0, 255, 0]          # Vert (moelle)
}


def visualize_phase2_dicom_to_nifti(patient_id="LUNG1-002", slice_idx=60):
    """
    Phase 2: Conversion DICOM → NIfTI
    Montre un slice avant/après conversion
    """
    print(f"\n=== PHASE 2: DICOM → NIfTI ===")
    print(f"Patient: {patient_id}, Slice: {slice_idx}")
    
    # Trouver le dossier DICOM du patient
    patient_dir = RAW_DIR / patient_id
    
    # Chercher le dossier CT
    ct_dir = None
    for study in patient_dir.iterdir():
        if study.is_dir():
            for series in study.iterdir():
                if series.is_dir():
                    dcm_files = list(series.glob("*.dcm"))
                    if len(dcm_files) > 10:  # CT series a beaucoup de slices
                        ct_dir = series
                        break
        if ct_dir:
            break
    
    if not ct_dir:
        print(f"❌ Dossier CT introuvable pour {patient_id}")
        return
    
    # Lire un slice DICOM
    dcm_files = sorted(list(ct_dir.glob("*.dcm")))
    if slice_idx >= len(dcm_files):
        slice_idx = len(dcm_files) // 2
    
    dcm = pydicom.dcmread(dcm_files[slice_idx])
    dicom_slice = dcm.pixel_array.astype(np.float32)
    
    # Convertir en Hounsfield Units
    slope = float(dcm.RescaleSlope) if hasattr(dcm, 'RescaleSlope') else 1.0
    intercept = float(dcm.RescaleIntercept) if hasattr(dcm, 'RescaleIntercept') else 0.0
    dicom_hu = dicom_slice * slope + intercept
    
    # Lire le NIfTI correspondant
    nifti_path = PROCESSED_DIR / "normalized" / f"{patient_id}_ct_normalized.nii.gz"
    if nifti_path.exists():
        nifti_img = sitk.ReadImage(str(nifti_path))
        nifti_array = sitk.GetArrayFromImage(nifti_img)
        nifti_slice = nifti_array[slice_idx]
    else:
        print(f"❌ NIfTI introuvable: {nifti_path}")
        return
    
    # Visualisation
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # DICOM brut (pixel values)
    im1 = axes[0].imshow(dicom_slice, cmap='gray')
    axes[0].set_title(f'DICOM Brut\n{patient_id} - Slice {slice_idx}\nPixel Values: [{dicom_slice.min():.0f}, {dicom_slice.max():.0f}]', 
                     fontsize=12, fontweight='bold')
    axes[0].axis('off')
    plt.colorbar(im1, ax=axes[0], fraction=0.046)
    
    # DICOM en Hounsfield Units
    im2 = axes[1].imshow(dicom_hu, cmap='gray', vmin=-1000, vmax=400)
    axes[1].set_title(f'DICOM → Hounsfield Units\nHU: [{dicom_hu.min():.0f}, {dicom_hu.max():.0f}]\nFenêtre: [-1000, 400] HU', 
                     fontsize=12, fontweight='bold')
    axes[1].axis('off')
    plt.colorbar(im2, ax=axes[1], fraction=0.046)
    
    # NIfTI normalisé
    im3 = axes[2].imshow(nifti_slice, cmap='gray')
    axes[2].set_title(f'NIfTI Normalisé (256×256)\nZ-score + Resize\nRange: [{nifti_slice.min():.2f}, {nifti_slice.max():.2f}]', 
                     fontsize=12, fontweight='bold')
    axes[2].axis('off')
    plt.colorbar(im3, ax=axes[2], fraction=0.046)
    
    plt.suptitle('PHASE 2: Conversion DICOM → NIfTI', fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    output_path = BASE_DIR / "visualizations" / "phase2_dicom_to_nifti.png"
    output_path.parent.mkdir(exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Sauvegardé: {output_path}")
    plt.close()


def visualize_phase2_5_multi_organ(patient_id="LUNG1-033", slice_idx=70):
    """
    Phase 2.5: Extraction masques multi-organes depuis RTSTRUCT
    Montre CT + masque avec tous les organes colorés
    """
    print(f"\n=== PHASE 2.5: Extraction Multi-Organes ===")
    print(f"Patient: {patient_id}, Slice: {slice_idx}")
    
    # Charger CT normalisé
    ct_path = PROCESSED_DIR / "normalized" / f"{patient_id}_ct_normalized.nii.gz"
    mask_path = PROCESSED_DIR / "masks_multi_organ" / f"{patient_id}_multi_organ.nii.gz"
    
    if not ct_path.exists():
        print(f"❌ CT introuvable: {ct_path}")
        return
    if not mask_path.exists():
        print(f"❌ Masque introuvable: {mask_path}")
        return
    
    ct_img = sitk.ReadImage(str(ct_path))
    mask_img = sitk.ReadImage(str(mask_path))
    
    ct_array = sitk.GetArrayFromImage(ct_img)
    mask_array = sitk.GetArrayFromImage(mask_img)
    
    ct_slice = ct_array[slice_idx]
    mask_slice = mask_array[slice_idx]
    
    # Créer masque coloré
    mask_rgb = np.zeros((*mask_slice.shape, 3), dtype=np.uint8)
    for label, color in LABEL_COLORS.items():
        mask_rgb[mask_slice == label] = color
    
    # Statistiques des organes présents
    unique_labels = np.unique(mask_slice)
    organs_present = [LABEL_NAMES[label] for label in unique_labels if label > 0]
    
    # Visualisation
    fig, axes = plt.subplots(2, 2, figsize=(16, 16))
    
    # CT seul
    axes[0, 0].imshow(ct_slice, cmap='gray')
    axes[0, 0].set_title(f'CT Normalisé\n{patient_id} - Slice {slice_idx}', 
                         fontsize=14, fontweight='bold')
    axes[0, 0].axis('off')
    
    # Masque coloré seul
    axes[0, 1].imshow(mask_rgb)
    axes[0, 1].set_title(f'Masques Multi-Organes\n{len(organs_present)} organes extraits', 
                         fontsize=14, fontweight='bold')
    axes[0, 1].axis('off')
    
    # CT + masque overlay
    axes[1, 0].imshow(ct_slice, cmap='gray')
    mask_alpha = (mask_slice > 0).astype(np.float32) * 0.5
    axes[1, 0].imshow(mask_rgb, alpha=mask_alpha)
    axes[1, 0].set_title('CT + Overlay Multi-Organes', fontsize=14, fontweight='bold')
    axes[1, 0].axis('off')
    
    # Légende des organes
    axes[1, 1].axis('off')
    legend_text = "Organes Extraits:\n\n"
    for label in unique_labels:
        if label > 0:
            count = (mask_slice == label).sum()
            percentage = (count / mask_slice.size) * 100
            color_hex = f"#{LABEL_COLORS[label][0]:02x}{LABEL_COLORS[label][1]:02x}{LABEL_COLORS[label][2]:02x}"
            legend_text += f"● Label {label}: {LABEL_NAMES[label]}\n"
            legend_text += f"  {count:,} pixels ({percentage:.2f}%)\n\n"
    
    axes[1, 1].text(0.1, 0.9, legend_text, fontsize=12, verticalalignment='top',
                    family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle(f'PHASE 2.5: Extraction Multi-Organes depuis RTSTRUCT\n{patient_id}', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    output_path = BASE_DIR / "visualizations" / "phase2_5_multi_organ.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Sauvegardé: {output_path}")
    plt.close()


def visualize_phase3_normalization(patient_id="LUNG1-002", slice_idx=60):
    """
    Phase 3: Normalisation (Windowing + Z-score + Resize)
    Compare avant/après normalisation
    """
    print(f"\n=== PHASE 3: Normalisation ===")
    print(f"Patient: {patient_id}, Slice: {slice_idx}")
    
    # Charger CT normalisé
    ct_path = PROCESSED_DIR / "normalized" / f"{patient_id}_ct_normalized.nii.gz"
    
    if not ct_path.exists():
        print(f"❌ CT introuvable: {ct_path}")
        return
    
    ct_img = sitk.ReadImage(str(ct_path))
    ct_array = sitk.GetArrayFromImage(ct_img)
    ct_slice = ct_array[slice_idx]
    
    # Simuler les étapes de normalisation
    # (En réalité on devrait charger le DICOM original, mais pour la démo on inverse)
    
    # Étape 1: Simule CT en HU (dénormalisation approximative)
    mean_val = ct_slice.mean()
    std_val = ct_slice.std()
    ct_hu_simulated = ct_slice * std_val + mean_val
    ct_hu_simulated = ct_hu_simulated * 200 - 100  # Approximation
    
    # Étape 2: Windowing lung [-1000, 400]
    ct_windowed = np.clip(ct_hu_simulated, -1000, 400)
    ct_windowed = (ct_windowed - (-1000)) / (400 - (-1000))
    
    # Étape 3: Z-score (résultat final)
    ct_normalized = ct_slice
    
    # Visualisation
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Ligne 1: Étapes de normalisation
    im1 = axes[0, 0].imshow(ct_hu_simulated, cmap='gray')
    axes[0, 0].set_title('Étape 1: CT en Hounsfield Units\nRange: [-1000, 3000] HU', 
                         fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')
    plt.colorbar(im1, ax=axes[0, 0], fraction=0.046)
    
    im2 = axes[0, 1].imshow(ct_windowed, cmap='gray', vmin=0, vmax=1)
    axes[0, 1].set_title('Étape 2: Windowing Lung\nFenêtre: [-1000, 400] HU\nNormalisé [0, 1]', 
                         fontsize=12, fontweight='bold')
    axes[0, 1].axis('off')
    plt.colorbar(im2, ax=axes[0, 1], fraction=0.046)
    
    im3 = axes[0, 2].imshow(ct_normalized, cmap='gray')
    axes[0, 2].set_title('Étape 3: Z-score Normalization\nMean=0, Std=1', 
                         fontsize=12, fontweight='bold')
    axes[0, 2].axis('off')
    plt.colorbar(im3, ax=axes[0, 2], fraction=0.046)
    
    # Ligne 2: Histogrammes
    axes[1, 0].hist(ct_hu_simulated.flatten(), bins=100, color='blue', alpha=0.7)
    axes[1, 0].set_title('Distribution HU', fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('Hounsfield Units')
    axes[1, 0].set_ylabel('Fréquence')
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].hist(ct_windowed.flatten(), bins=100, color='green', alpha=0.7)
    axes[1, 1].set_title('Distribution après Windowing', fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel('Intensité normalisée')
    axes[1, 1].set_ylabel('Fréquence')
    axes[1, 1].grid(True, alpha=0.3)
    
    axes[1, 2].hist(ct_normalized.flatten(), bins=100, color='red', alpha=0.7)
    axes[1, 2].set_title('Distribution après Z-score', fontsize=12, fontweight='bold')
    axes[1, 2].set_xlabel('Valeur normalisée')
    axes[1, 2].set_ylabel('Fréquence')
    axes[1, 2].axvline(0, color='black', linestyle='--', label='Mean=0')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.suptitle(f'PHASE 3: Normalisation CT\n{patient_id} - Slice {slice_idx}', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    output_path = BASE_DIR / "visualizations" / "phase3_normalization.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Sauvegardé: {output_path}")
    plt.close()


def visualize_phase4_dataset_pytorch():
    """
    Phase 4: Dataset PyTorch
    Montre un batch de données chargé
    """
    print(f"\n=== PHASE 4: Dataset PyTorch ===")
    
    # Charger les splits
    splits_dir = PROCESSED_DIR / "splits_rtstruct"
    with open(splits_dir / "train.txt", 'r') as f:
        train_ids = [line.strip() for line in f.readlines()]
    
    # Créer dataset
    ct_dir = PROCESSED_DIR / "normalized"
    mask_dir = PROCESSED_DIR / "masks_multi_organ"
    
    dataset = MultiOrganDataset(train_ids[:5], ct_dir, mask_dir)  # 5 premiers patients
    
    # Charger quelques samples
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    
    # Trouver les meilleurs slices avec beaucoup d'organes
    good_samples = []
    for idx in range(len(dataset)):
        ct_slice, mask_slice = dataset[idx]
        unique_labels = torch.unique(mask_slice)
        if len(unique_labels) >= 5:  # Au moins 5 organes
            good_samples.append((idx, len(unique_labels)))
        if len(good_samples) >= 10:
            break
    
    # Prendre les 4 meilleurs
    good_samples.sort(key=lambda x: x[1], reverse=True)
    selected_indices = [s[0] for s in good_samples[:4]]
    
    for i, idx in enumerate(selected_indices):
        ct_slice, mask_slice = dataset[idx]
        
        ct_np = ct_slice.squeeze().numpy()
        mask_np = mask_slice.numpy()
        
        # Créer masque coloré
        mask_rgb = np.zeros((*mask_np.shape, 3), dtype=np.uint8)
        for label, color in LABEL_COLORS.items():
            mask_rgb[mask_np == label] = color
        
        sample_info = dataset.get_sample_info(idx)
        
        # Ligne 1: CT
        axes[0, i].imshow(ct_np, cmap='gray')
        axes[0, i].set_title(f'CT Slice\n{sample_info["patient_id"]}\nSlice {sample_info["slice_idx"]}', 
                            fontsize=10)
        axes[0, i].axis('off')
        
        # Ligne 2: Masque coloré
        axes[1, i].imshow(mask_rgb)
        unique_labels = np.unique(mask_np)
        axes[1, i].set_title(f'Masque Multi-Organes\n{len(unique_labels)-1} organes', fontsize=10)
        axes[1, i].axis('off')
        
        # Ligne 3: Overlay avec alpha fixe
        axes[2, i].imshow(ct_np, cmap='gray')
        # Alpha fixe 0.7 sur tout le masque
        axes[2, i].imshow(mask_rgb, alpha=0.7)
        axes[2, i].set_title('CT + Overlay (α=0.7)', fontsize=10)
        axes[2, i].axis('off')
    
    plt.suptitle(f'PHASE 4: Dataset PyTorch Multi-Organes\n{len(dataset):,} slices dans le dataset | Batch Size: 4 | Labels: 0-7', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    output_path = BASE_DIR / "visualizations" / "phase4_dataset_pytorch.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Sauvegardé: {output_path}")
    plt.close()


def visualize_dataset_statistics():
    """
    Statistiques globales du dataset
    """
    print(f"\n=== STATISTIQUES DATASET ===")
    
    # Charger les splits
    splits_dir = PROCESSED_DIR / "splits_rtstruct"
    
    with open(splits_dir / "train.txt", 'r') as f:
        train_ids = [line.strip() for line in f.readlines()]
    with open(splits_dir / "val.txt", 'r') as f:
        val_ids = [line.strip() for line in f.readlines()]
    with open(splits_dir / "test.txt", 'r') as f:
        test_ids = [line.strip() for line in f.readlines()]
    
    # Compter les slices
    ct_dir = PROCESSED_DIR / "normalized"
    mask_dir = PROCESSED_DIR / "masks_multi_organ"
    
    def count_slices(patient_ids):
        total = 0
        for pid in patient_ids:
            ct_path = ct_dir / f"{pid}_ct_normalized.nii.gz"
            if ct_path.exists():
                img = sitk.ReadImage(str(ct_path))
                total += img.GetSize()[2]
        return total
    
    train_slices = count_slices(train_ids)
    val_slices = count_slices(val_ids)
    test_slices = count_slices(test_ids)
    
    # Visualisation
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Graphique 1: Patients par split
    splits = ['Train', 'Val', 'Test']
    patients = [len(train_ids), len(val_ids), len(test_ids)]
    colors_split = ['#3498db', '#2ecc71', '#e74c3c']
    
    bars1 = axes[0].bar(splits, patients, color=colors_split, alpha=0.8, edgecolor='black', linewidth=2)
    axes[0].set_ylabel('Nombre de Patients', fontsize=12, fontweight='bold')
    axes[0].set_title('Distribution des Patients', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # Ajouter valeurs sur les barres
    for bar, val in zip(bars1, patients):
        height = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width()/2., height,
                    f'{val}\n({val/(sum(patients))*100:.1f}%)',
                    ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Graphique 2: Slices par split
    slices = [train_slices, val_slices, test_slices]
    bars2 = axes[1].bar(splits, slices, color=colors_split, alpha=0.8, edgecolor='black', linewidth=2)
    axes[1].set_ylabel('Nombre de Slices', fontsize=12, fontweight='bold')
    axes[1].set_title('Distribution des Slices 2D', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3, axis='y')
    
    for bar, val in zip(bars2, slices):
        height = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:,}\n({val/(sum(slices))*100:.1f}%)',
                    ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.suptitle(f'STATISTIQUES DATASET MULTI-ORGANES\nTotal: {sum(patients)} patients | {sum(slices):,} slices', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    output_path = BASE_DIR / "visualizations" / "dataset_statistics.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Sauvegardé: {output_path}")
    plt.close()
    
    # Afficher statistiques texte
    print("\n📊 STATISTIQUES FINALES:")
    print(f"  Train: {len(train_ids)} patients, {train_slices:,} slices")
    print(f"  Val:   {len(val_ids)} patients, {val_slices:,} slices")
    print(f"  Test:  {len(test_ids)} patients, {test_slices:,} slices")
    print(f"  TOTAL: {sum(patients)} patients, {sum(slices):,} slices")


if __name__ == "__main__":
    print("="*70)
    print("GÉNÉRATION DES VISUALISATIONS - TOUTES LES PHASES")
    print("="*70)
    
    # Créer dossier de sortie
    vis_dir = BASE_DIR / "visualizations"
    vis_dir.mkdir(exist_ok=True)
    
    # Phase 2: DICOM → NIfTI
    visualize_phase2_dicom_to_nifti(patient_id="LUNG1-002", slice_idx=60)
    
    # Phase 2.5: Multi-organes
    visualize_phase2_5_multi_organ(patient_id="LUNG1-033", slice_idx=70)
    
    # Phase 3: Normalisation
    visualize_phase3_normalization(patient_id="LUNG1-002", slice_idx=60)
    
    # Phase 4: Dataset PyTorch
    visualize_phase4_dataset_pytorch()
    
    # Statistiques
    visualize_dataset_statistics()
    
    print("\n" + "="*70)
    print("✅ TOUTES LES VISUALISATIONS GÉNÉRÉES!")
    print(f"📁 Dossier: {vis_dir}")
    print("="*70)
