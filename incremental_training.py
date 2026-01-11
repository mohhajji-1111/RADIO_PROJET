"""
=============================================================================
TRAINING INCRÉMENTAL - NSCLC Multi-Organ Segmentation
=============================================================================
Ce script permet de:
1. Diviser les données en petits batches (ex: 20 patients)
2. Sauvegarder le modèle après chaque batch
3. Reprendre automatiquement si le PC crash
4. Accumuler l'apprentissage progressivement

Auteur: Projet NSCLC Radiomics
Date: Janvier 2026
=============================================================================
"""

import os
import sys
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
from datetime import datetime
import SimpleITK as sitk
from tqdm import tqdm
import matplotlib.pyplot as plt
import gc
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

# Fix pour OpenMP sur Windows
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

CONFIG = {
    # Chemins
    'data_dir': Path(r'C:\Users\HP\Desktop\RADIO_PROJET\DATA\processed\normalized'),
    'output_dir': Path(r'C:\Users\HP\Desktop\RADIO_PROJET\training_output'),
    'checkpoint_dir': Path(r'C:\Users\HP\Desktop\RADIO_PROJET\training_output\checkpoints'),
    
    # ========================================================================
    # PARAMÈTRES OPTIMAUX POUR BONS RÉSULTATS (GPU)
    # ========================================================================
    
    # Training incrémental
    'patients_per_batch': 25,      # 25 patients par batch (GPU peut gérer plus)
    'epochs_per_batch': 15,        # 15 epochs par batch pour TRÈS bien apprendre
    'total_rounds': 4,             # 4 rounds = 60 epochs totales sur chaque patient (meilleur apprentissage)
    
    # Hyperparamètres - CLÉS POUR BONS RÉSULTATS
    'batch_size': 8,               # 8-16 selon mémoire GPU (plus = plus stable)
    'learning_rate': 1e-4,         # 1e-4 = standard, bon équilibre
    'weight_decay': 1e-5,          # Régularisation pour éviter overfitting
    'num_workers': 4,              # Parallélisation chargement données
    'num_classes': 8,              # Background + 7 organes
    
    # Scheduler (améliore convergence)
    'use_scheduler': True,         # Réduire LR si plateau
    'scheduler_patience': 5,       # Epochs avant réduction LR
    'scheduler_factor': 0.5,       # Facteur de réduction (LR * 0.5)
    
    # Early stopping
    'early_stopping': True,
    'patience': 15,                # Epochs sans amélioration avant arrêt
    
    # Data augmentation (améliore généralisation)
    'use_augmentation': True,
    
    # Device
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
}

# Créer les dossiers
CONFIG['output_dir'].mkdir(parents=True, exist_ok=True)
CONFIG['checkpoint_dir'].mkdir(parents=True, exist_ok=True)

# ============================================================================
# DATASET
# ============================================================================
class MultiOrganDataset(Dataset):
    """Dataset pour charger les slices CT et masks par patient."""
    
    def __init__(self, patient_ids, data_dir, transform=None):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.samples = []
        
        print(f"Chargement de {len(patient_ids)} patients...")
        for patient_id in tqdm(patient_ids, desc="Loading patients"):
            ct_path = self.data_dir / f"{patient_id}_ct_normalized.nii.gz"
            mask_path = self.data_dir / f"{patient_id}_mask_normalized.nii.gz"
            
            if ct_path.exists() and mask_path.exists():
                try:
                    # Charger le volume pour obtenir le nombre de slices
                    ct_img = sitk.ReadImage(str(ct_path))
                    num_slices = ct_img.GetSize()[2]
                    
                    for slice_idx in range(num_slices):
                        self.samples.append({
                            'ct_path': str(ct_path),
                            'mask_path': str(mask_path),
                            'slice_idx': slice_idx,
                            'patient_id': patient_id
                        })
                except Exception as e:
                    print(f"  ⚠️ Erreur chargement {patient_id}: {e}")
            else:
                print(f"  ⚠️ Fichiers manquants pour {patient_id}")
                print(f"     CT existe: {ct_path.exists()}")
                print(f"     Mask existe: {mask_path.exists()}")
        
        print(f"Total: {len(self.samples)} slices chargées")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Charger le slice CT
        ct_img = sitk.ReadImage(sample['ct_path'])
        ct_array = sitk.GetArrayFromImage(ct_img)
        ct_slice = ct_array[sample['slice_idx']]
        
        # Charger le slice mask
        mask_img = sitk.ReadImage(sample['mask_path'])
        mask_array = sitk.GetArrayFromImage(mask_img)
        mask_slice = mask_array[sample['slice_idx']]
        
        # Resize à 256x256 si nécessaire
        if ct_slice.shape != (256, 256):
            ct_slice = self._resize(ct_slice, (256, 256))
            mask_slice = self._resize(mask_slice, (256, 256), is_mask=True)
        
        # Normaliser CT
        ct_slice = (ct_slice - ct_slice.mean()) / (ct_slice.std() + 1e-8)
        
        # Data Augmentation (si activé et en mode training)
        if self.transform and CONFIG.get('use_augmentation', False):
            ct_slice, mask_slice = self._augment(ct_slice, mask_slice)
        
        # Convertir en tensors
        image = torch.from_numpy(ct_slice.copy()).float().unsqueeze(0)  # [1, H, W]
        mask = torch.from_numpy(mask_slice.copy()).long()  # [H, W]
        
        return {'image': image, 'mask': mask}
    
    def _augment(self, image, mask):
        """Applique des augmentations aléatoires."""
        import cv2
        
        # Flip horizontal (50% chance)
        if np.random.random() > 0.5:
            image = np.fliplr(image)
            mask = np.fliplr(mask)
        
        # Flip vertical (50% chance)
        if np.random.random() > 0.5:
            image = np.flipud(image)
            mask = np.flipud(mask)
        
        # Rotation aléatoire (-15 à +15 degrés)
        if np.random.random() > 0.5:
            angle = np.random.uniform(-15, 15)
            h, w = image.shape
            M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1)
            image = cv2.warpAffine(image.astype(np.float32), M, (w, h))
            mask = cv2.warpAffine(mask.astype(np.float32), M, (w, h), flags=cv2.INTER_NEAREST)
        
        # Ajustement de luminosité/contraste
        if np.random.random() > 0.5:
            alpha = np.random.uniform(0.8, 1.2)  # Contraste
            beta = np.random.uniform(-0.1, 0.1)  # Luminosité
            image = alpha * image + beta
        
        return image.astype(np.float32), mask.astype(np.float32)
    
    def _resize(self, arr, size, is_mask=False):
        """Resize un array 2D."""
        import cv2
        interp = cv2.INTER_NEAREST if is_mask else cv2.INTER_LINEAR
        return cv2.resize(arr.astype(np.float32), size, interpolation=interp)


# ============================================================================
# MODÈLE U-NET
# ============================================================================
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, in_channels=1, num_classes=8):
        super().__init__()
        
        # Encoder
        self.enc1 = DoubleConv(in_channels, 64)
        self.enc2 = DoubleConv(64, 128)
        self.enc3 = DoubleConv(128, 256)
        self.enc4 = DoubleConv(256, 512)
        
        # Bottleneck
        self.bottleneck = DoubleConv(512, 1024)
        
        # Decoder
        self.up4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec4 = DoubleConv(1024, 512)
        
        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = DoubleConv(512, 256)
        
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = DoubleConv(256, 128)
        
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = DoubleConv(128, 64)
        
        # Output
        self.out = nn.Conv2d(64, num_classes, 1)
        
        self.pool = nn.MaxPool2d(2)
    
    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        
        # Bottleneck
        b = self.bottleneck(self.pool(e4))
        
        # Decoder
        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        
        return self.out(d1)


# ============================================================================
# LOSS FUNCTIONS
# ============================================================================
class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, pred, target):
        pred = torch.softmax(pred, dim=1)
        target_onehot = torch.zeros_like(pred)
        target_onehot.scatter_(1, target.unsqueeze(1), 1)
        
        intersection = (pred * target_onehot).sum(dim=(2, 3))
        union = pred.sum(dim=(2, 3)) + target_onehot.sum(dim=(2, 3))
        
        dice = (2 * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()


class CombinedLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()
        self.dice = DiceLoss()
    
    def forward(self, pred, target):
        return 0.5 * self.ce(pred, target) + 0.5 * self.dice(pred, target)


# ============================================================================
# TRAINING STATE MANAGER
# ============================================================================
class TrainingState:
    """Gère l'état du training pour la reprise après crash."""
    
    def __init__(self, checkpoint_dir):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.state_file = self.checkpoint_dir / 'training_state.json'
        self.state = self._load_state()
    
    def _load_state(self):
        """Charge l'état depuis le fichier JSON."""
        if self.state_file.exists():
            with open(self.state_file, 'r') as f:
                return json.load(f)
        return {
            'current_round': 0,
            'current_batch': 0,
            'current_epoch': 0,
            'total_epochs_trained': 0,
            'best_loss': float('inf'),
            'history': {'train_loss': [], 'val_loss': [], 'dice_scores': []},
            'completed_batches': [],
            'last_update': None
        }
    
    def save(self):
        """Sauvegarde l'état actuel."""
        self.state['last_update'] = datetime.now().isoformat()
        with open(self.state_file, 'w') as f:
            json.dump(self.state, f, indent=2)
    
    def update(self, **kwargs):
        """Met à jour l'état."""
        self.state.update(kwargs)
        self.save()
    
    def get(self, key, default=None):
        return self.state.get(key, default)
    
    def add_to_history(self, train_loss, val_loss, dice_score):
        self.state['history']['train_loss'].append(train_loss)
        self.state['history']['val_loss'].append(val_loss)
        self.state['history']['dice_scores'].append(dice_score)
        self.save()


# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================
def calculate_dice(pred, target, num_classes=8):
    """Calcule le Dice score moyen."""
    pred = torch.argmax(pred, dim=1)
    dice_scores = []
    
    for c in range(num_classes):
        pred_c = (pred == c).float()
        target_c = (target == c).float()
        
        intersection = (pred_c * target_c).sum()
        union = pred_c.sum() + target_c.sum()
        
        if union > 0:
            dice = (2 * intersection) / (union + 1e-8)
            dice_scores.append(dice.item())
    
    return np.mean(dice_scores)


def train_one_epoch(model, loader, criterion, optimizer, device):
    """Entraîne le modèle pour une epoch."""
    model.train()
    total_loss = 0
    
    pbar = tqdm(loader, desc="Training", leave=False)
    for batch in pbar:
        images = batch['image'].to(device)
        masks = batch['mask'].to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        # Libérer mémoire
        del images, masks, outputs, loss
        torch.cuda.empty_cache() if device == 'cuda' else None
    
    return total_loss / len(loader)


def validate(model, loader, criterion, device):
    """Valide le modèle."""
    model.eval()
    total_loss = 0
    total_dice = 0
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Validation", leave=False):
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)
            
            outputs = model(images)
            loss = criterion(outputs, masks)
            dice = calculate_dice(outputs, masks)
            
            total_loss += loss.item()
            total_dice += dice
            
            del images, masks, outputs
            torch.cuda.empty_cache() if device == 'cuda' else None
    
    return total_loss / len(loader), total_dice / len(loader)


def save_checkpoint(model, optimizer, state, filename):
    """Sauvegarde un checkpoint complet."""
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'training_state': state.state,
    }
    torch.save(checkpoint, filename)
    print(f"✓ Checkpoint sauvegardé: {filename}")


def load_checkpoint(model, optimizer, checkpoint_path):
    """Charge un checkpoint."""
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=CONFIG['device'])
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"✓ Checkpoint chargé: {checkpoint_path}")
        return checkpoint.get('training_state', {})
    return None


# ============================================================================
# MAIN TRAINING LOOP
# ============================================================================
def main():
    print("="*70)
    print("TRAINING INCRÉMENTAL - NSCLC Multi-Organ Segmentation")
    print("="*70)
    print(f"Device: {CONFIG['device']}")
    print(f"Patients par batch: {CONFIG['patients_per_batch']}")
    print(f"Epochs par batch: {CONFIG['epochs_per_batch']}")
    print(f"Total rounds: {CONFIG['total_rounds']}")
    print("="*70)
    
    # Obtenir tous les patients
    data_dir = CONFIG['data_dir']
    ct_files = sorted(data_dir.glob("*_ct_normalized.nii.gz"))
    # Extraire le patient_id correctement (ex: LUNG1-001 depuis LUNG1-001_ct_normalized.nii.gz)
    all_patients = [f.name.replace('_ct_normalized.nii.gz', '') for f in ct_files]
    print(f"\nTotal patients trouvés: {len(all_patients)}")
    print(f"Premier patient: {all_patients[0]}")
    print(f"Dernier patient: {all_patients[-1]}")
    
    # Diviser en batches
    batch_size = CONFIG['patients_per_batch']
    patient_batches = [all_patients[i:i+batch_size] for i in range(0, len(all_patients), batch_size)]
    print(f"Nombre de batches: {len(patient_batches)}")
    for i, batch in enumerate(patient_batches):
        print(f"  Batch {i+1}: {len(batch)} patients ({batch[0]} - {batch[-1]})")
    
    # Séparer validation (10% des patients)
    np.random.seed(42)
    val_indices = np.random.choice(len(all_patients), size=max(1, len(all_patients)//10), replace=False)
    val_patients = [all_patients[i] for i in val_indices]
    print(f"\nPatients de validation: {len(val_patients)}")
    
    # Créer dataset de validation
    val_dataset = MultiOrganDataset(val_patients, data_dir)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'], 
                           shuffle=False, num_workers=CONFIG['num_workers'])
    
    # Initialiser modèle
    model = UNet(in_channels=1, num_classes=CONFIG['num_classes']).to(CONFIG['device'])
    criterion = CombinedLoss()
    
    # Optimiseur avec weight decay (régularisation L2)
    optimizer = optim.Adam(
        model.parameters(), 
        lr=CONFIG['learning_rate'],
        weight_decay=CONFIG.get('weight_decay', 1e-5)
    )
    
    # Learning Rate Scheduler (réduit LR si plateau)
    scheduler = None
    if CONFIG.get('use_scheduler', True):
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='min', 
            factor=CONFIG.get('scheduler_factor', 0.5),
            patience=CONFIG.get('scheduler_patience', 5),
            verbose=True
        )
    
    # Gestionnaire d'état
    state = TrainingState(CONFIG['checkpoint_dir'])
    
    # Charger checkpoint si existe
    latest_checkpoint = CONFIG['checkpoint_dir'] / 'latest_checkpoint.pth'
    if latest_checkpoint.exists():
        print("\n🔄 Reprise du training précédent...")
        load_checkpoint(model, optimizer, latest_checkpoint)
    
    start_round = state.get('current_round', 0)
    start_batch = state.get('current_batch', 0)
    
    print(f"\n📍 Démarrage: Round {start_round+1}, Batch {start_batch+1}")
    print("-"*70)
    
    # Boucle principale
    try:
        for round_idx in range(start_round, CONFIG['total_rounds']):
            print(f"\n{'='*70}")
            print(f"ROUND {round_idx + 1}/{CONFIG['total_rounds']}")
            print(f"{'='*70}")
            
            for batch_idx in range(start_batch if round_idx == start_round else 0, len(patient_batches)):
                batch_patients = patient_batches[batch_idx]
                
                # Exclure les patients de validation
                train_patients = [p for p in batch_patients if p not in val_patients]
                
                if not train_patients:
                    continue
                
                print(f"\n📦 Batch {batch_idx + 1}/{len(patient_batches)}")
                print(f"   Patients: {train_patients[0]} → {train_patients[-1]} ({len(train_patients)} patients)")
                
                # Créer dataset pour ce batch
                train_dataset = MultiOrganDataset(train_patients, data_dir, transform=True)
                train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'],
                                         shuffle=True, num_workers=CONFIG['num_workers'])
                
                # Entraîner sur ce batch
                for epoch in range(CONFIG['epochs_per_batch']):
                    print(f"\n   Epoch {epoch + 1}/{CONFIG['epochs_per_batch']}")
                    
                    train_loss = train_one_epoch(model, train_loader, criterion, optimizer, CONFIG['device'])
                    val_loss, val_dice = validate(model, val_loader, criterion, CONFIG['device'])
                    
                    # Afficher learning rate actuel
                    current_lr = optimizer.param_groups[0]['lr']
                    print(f"   Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Dice: {val_dice:.4f} | LR: {current_lr:.2e}")
                    
                    # Mettre à jour le scheduler
                    if scheduler is not None:
                        scheduler.step(val_loss)
                    
                    # Sauvegarder dans l'historique
                    state.add_to_history(train_loss, val_loss, val_dice)
                    
                    # Sauvegarder si meilleur modèle
                    if val_loss < state.get('best_loss', float('inf')):
                        state.update(best_loss=val_loss)
                        best_model_path = CONFIG['output_dir'] / 'best_model.pth'
                        torch.save(model.state_dict(), best_model_path)
                        print(f"   ⭐ Nouveau meilleur modèle sauvegardé! (Dice: {val_dice:.4f})")
                
                # Sauvegarder checkpoint après chaque batch
                state.update(
                    current_round=round_idx,
                    current_batch=batch_idx + 1,
                    total_epochs_trained=state.get('total_epochs_trained', 0) + CONFIG['epochs_per_batch']
                )
                save_checkpoint(model, optimizer, state, latest_checkpoint)
                
                # Libérer mémoire
                del train_dataset, train_loader
                gc.collect()
                torch.cuda.empty_cache() if CONFIG['device'] == 'cuda' else None
                
                print(f"\n   ✅ Batch {batch_idx + 1} terminé et sauvegardé!")
            
            # Reset batch après chaque round
            start_batch = 0
        
        print("\n" + "="*70)
        print("🎉 TRAINING TERMINÉ!")
        print("="*70)
        
        # Sauvegarder modèle final
        final_model_path = CONFIG['output_dir'] / 'final_model.pth'
        torch.save(model.state_dict(), final_model_path)
        print(f"✓ Modèle final sauvegardé: {final_model_path}")
        
        # Afficher courbes
        plot_training_curves(state)
        
    except KeyboardInterrupt:
        print("\n\n⚠️ Training interrompu par l'utilisateur!")
        print("   Le progrès a été sauvegardé. Relancez le script pour reprendre.")
        save_checkpoint(model, optimizer, state, latest_checkpoint)
    
    except Exception as e:
        print(f"\n\n❌ Erreur: {e}")
        print("   Le progrès a été sauvegardé. Relancez le script pour reprendre.")
        save_checkpoint(model, optimizer, state, latest_checkpoint)
        raise


def plot_training_curves(state):
    """Affiche les courbes de training."""
    history = state.get('history', {})
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss curves
    axes[0].plot(history.get('train_loss', []), label='Train Loss', color='blue')
    axes[0].plot(history.get('val_loss', []), label='Val Loss', color='orange')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training & Validation Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    # Dice score
    axes[1].plot(history.get('dice_scores', []), label='Val Dice', color='green')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Dice Score')
    axes[1].set_title('Validation Dice Score')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig(CONFIG['output_dir'] / 'training_curves.png', dpi=150)
    plt.show()
    print(f"✓ Courbes sauvegardées: {CONFIG['output_dir'] / 'training_curves.png'}")


# ============================================================================
# ENTRY POINT
# ============================================================================
if __name__ == '__main__':
    main()
