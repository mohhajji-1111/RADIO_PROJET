"""
Script d'entraînement U-Net multi-organes
- Loss: CrossEntropyLoss avec class weights
- Metrics: Dice Score par organe
- Data augmentation: Random flip, rotation
- Early stopping et sauvegarde best model

Auteur: Copilot
Date: 21 Nov 2025
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime
import json

from unet_multi_organ import UNetMultiOrgan
from dataset_multi_organ import MultiOrganDataset, compute_class_weights, get_class_distribution


class DiceLoss(nn.Module):
    """
    Dice Loss pour segmentation multi-classes
    """
    def __init__(self, num_classes=8, smooth=1.0):
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth
    
    def forward(self, logits, targets):
        """
        Args:
            logits: (B, C, H, W) - sortie du modèle
            targets: (B, H, W) - labels entiers
        """
        # Convertir logits en probabilités
        probs = torch.softmax(logits, dim=1)
        
        # One-hot encoding des targets
        targets_one_hot = torch.nn.functional.one_hot(targets, self.num_classes)  # (B, H, W, C)
        targets_one_hot = targets_one_hot.permute(0, 3, 1, 2).float()  # (B, C, H, W)
        
        # Calculer Dice par classe
        dice_scores = []
        for c in range(self.num_classes):
            pred_c = probs[:, c, :, :]
            target_c = targets_one_hot[:, c, :, :]
            
            intersection = (pred_c * target_c).sum()
            union = pred_c.sum() + target_c.sum()
            
            dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
            dice_scores.append(dice)
        
        # Moyenne des Dice scores
        dice_loss = 1.0 - torch.stack(dice_scores).mean()
        return dice_loss


class CombinedLoss(nn.Module):
    """
    Combinaison de CrossEntropy et Dice Loss
    """
    def __init__(self, num_classes=8, class_weights=None, ce_weight=0.5, dice_weight=0.5):
        super().__init__()
        self.ce_loss = nn.CrossEntropyLoss(weight=class_weights)
        self.dice_loss = DiceLoss(num_classes=num_classes)
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
    
    def forward(self, logits, targets):
        ce = self.ce_loss(logits, targets)
        dice = self.dice_loss(logits, targets)
        return self.ce_weight * ce + self.dice_weight * dice


def compute_dice_per_class(predictions, targets, num_classes=8, smooth=1.0):
    """
    Calcule le Dice Score pour chaque classe
    
    Args:
        predictions: (B, H, W) - labels prédits
        targets: (B, H, W) - labels vrais
        
    Returns:
        dice_scores: dict {class_id: dice_score}
    """
    dice_scores = {}
    
    for c in range(num_classes):
        pred_c = (predictions == c).float()
        target_c = (targets == c).float()
        
        intersection = (pred_c * target_c).sum()
        union = pred_c.sum() + target_c.sum()
        
        if union > 0:
            dice = (2.0 * intersection + smooth) / (union + smooth)
            dice_scores[c] = dice.item()
        else:
            dice_scores[c] = 0.0
    
    return dice_scores


def train_epoch(model, dataloader, criterion, optimizer, device):
    """
    Entraîne le modèle pour une epoch
    """
    model.train()
    epoch_loss = 0.0
    num_batches = 0
    
    pbar = tqdm(dataloader, desc="Training")
    for ct_batch, mask_batch in pbar:
        ct_batch = ct_batch.to(device)
        mask_batch = mask_batch.to(device)
        
        # Forward
        logits = model(ct_batch)
        loss = criterion(logits, mask_batch)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Stats
        epoch_loss += loss.item()
        num_batches += 1
        
        pbar.set_postfix({'loss': loss.item()})
    
    avg_loss = epoch_loss / num_batches
    return avg_loss


def validate(model, dataloader, criterion, device, num_classes=8):
    """
    Valide le modèle
    """
    model.eval()
    val_loss = 0.0
    all_dice_scores = {c: [] for c in range(num_classes)}
    num_batches = 0
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Validation")
        for ct_batch, mask_batch in pbar:
            ct_batch = ct_batch.to(device)
            mask_batch = mask_batch.to(device)
            
            # Forward
            logits = model(ct_batch)
            loss = criterion(logits, mask_batch)
            
            # Prédictions
            predictions = torch.argmax(logits, dim=1)
            
            # Dice par classe
            dice_scores = compute_dice_per_class(predictions, mask_batch, num_classes)
            for c, score in dice_scores.items():
                all_dice_scores[c].append(score)
            
            val_loss += loss.item()
            num_batches += 1
            
            pbar.set_postfix({'loss': loss.item()})
    
    # Moyennes
    avg_val_loss = val_loss / num_batches
    avg_dice_scores = {c: np.mean(scores) if scores else 0.0 
                       for c, scores in all_dice_scores.items()}
    mean_dice = np.mean(list(avg_dice_scores.values()))
    
    return avg_val_loss, avg_dice_scores, mean_dice


def train_model(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    device,
    num_epochs=50,
    save_dir="checkpoints",
    early_stopping_patience=10
):
    """
    Entraîne le modèle avec early stopping
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)
    
    best_val_dice = 0.0
    patience_counter = 0
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_dice': [],
        'val_dice_per_class': []
    }
    
    for epoch in range(num_epochs):
        print(f"\n{'='*70}")
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"{'='*70}")
        
        # Training
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        print(f"Train Loss: {train_loss:.4f}")
        
        # Validation
        val_loss, val_dice_per_class, mean_val_dice = validate(
            model, val_loader, criterion, device
        )
        print(f"Val Loss: {val_loss:.4f}")
        print(f"Val Mean Dice: {mean_val_dice:.4f}")
        
        # Afficher Dice par classe
        print("\nDice Score par organe:")
        label_names = {
            0: 'Background', 1: 'GTV', 2: 'PTV',
            3: 'Poumon_D', 4: 'Poumon_G', 5: 'Coeur',
            6: 'Oesophage', 7: 'Moelle'
        }
        for c, score in val_dice_per_class.items():
            print(f"  {label_names[c]:12s}: {score:.4f}")
        
        # Historique
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_dice'].append(mean_val_dice)
        history['val_dice_per_class'].append(val_dice_per_class)
        
        # Sauvegarder meilleur modèle
        if mean_val_dice > best_val_dice:
            best_val_dice = mean_val_dice
            patience_counter = 0
            
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_dice': mean_val_dice,
                'val_dice_per_class': val_dice_per_class
            }
            
            save_path = save_dir / "best_model.pth"
            torch.save(checkpoint, save_path)
            print(f"\n✅ Meilleur modèle sauvegardé: {save_path}")
        else:
            patience_counter += 1
            print(f"\n⚠️  Pas d'amélioration ({patience_counter}/{early_stopping_patience})")
        
        # Early stopping
        if patience_counter >= early_stopping_patience:
            print(f"\n🛑 Early stopping après {epoch+1} epochs")
            break
    
    # Sauvegarder historique
    history_path = save_dir / "training_history.json"
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    return history


def plot_training_curves(history, save_path="training_curves.png"):
    """
    Visualise les courbes d'entraînement
    """
    epochs = range(1, len(history['train_loss']) + 1)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Loss
    axes[0].plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    axes[0].plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Loss', fontsize=12, fontweight='bold')
    axes[0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Dice Score
    axes[1].plot(epochs, history['val_dice'], 'g-', label='Mean Dice', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Dice Score', fontsize=12, fontweight='bold')
    axes[1].set_title('Validation Dice Score', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Courbes sauvegardées: {save_path}")


if __name__ == "__main__":
    """
    Script principal d'entraînement
    REMARQUE: À exécuter sur Google Colab avec GPU
    """
    
    print("="*70)
    print("ENTRAÎNEMENT U-NET MULTI-ORGANES")
    print("="*70)
    
    # Configuration
    BATCH_SIZE = 8
    NUM_EPOCHS = 50
    LEARNING_RATE = 1e-4
    NUM_CLASSES = 8
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"\nDevice: {DEVICE}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Learning rate: {LEARNING_RATE}")
    
    # Chemins (à adapter pour Colab)
    BASE_DIR = Path(__file__).parent
    DATA_DIR = BASE_DIR / "DATA" / "processed"
    CT_DIR = DATA_DIR / "normalized"
    MASK_DIR = DATA_DIR / "masks_multi_organ"
    SPLITS_DIR = DATA_DIR / "splits_rtstruct"
    
    # Charger splits
    print("\nChargement des splits...")
    with open(SPLITS_DIR / "train.txt", 'r') as f:
        train_ids = [line.strip() for line in f.readlines()]
    with open(SPLITS_DIR / "val.txt", 'r') as f:
        val_ids = [line.strip() for line in f.readlines()]
    
    print(f"Train: {len(train_ids)} patients")
    print(f"Val: {len(val_ids)} patients")
    
    # Créer datasets
    print("\nCréation des datasets...")
    train_dataset = MultiOrganDataset(train_ids, CT_DIR, MASK_DIR)
    val_dataset = MultiOrganDataset(val_ids, CT_DIR, MASK_DIR)
    
    # Calculer class weights
    print("\nCalcul des class weights...")
    class_counts = get_class_distribution(train_dataset)
    class_weights = compute_class_weights(class_counts, method='sqrt_inverse')
    class_weights = class_weights.to(DEVICE)
    
    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True if DEVICE.type == 'cuda' else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True if DEVICE.type == 'cuda' else False
    )
    
    # Modèle
    print("\nCréation du modèle...")
    model = UNetMultiOrgan(n_channels=1, n_classes=NUM_CLASSES, bilinear=False)
    model = model.to(DEVICE)
    
    from unet_multi_organ import count_parameters
    num_params = count_parameters(model)
    print(f"Nombre de paramètres: {num_params:,}")
    
    # Loss et optimizer
    criterion = CombinedLoss(
        num_classes=NUM_CLASSES,
        class_weights=class_weights,
        ce_weight=0.5,
        dice_weight=0.5
    )
    
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Entraînement
    print("\n" + "="*70)
    print("DÉBUT DE L'ENTRAÎNEMENT")
    print("="*70)
    
    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=DEVICE,
        num_epochs=NUM_EPOCHS,
        save_dir="checkpoints_multi_organ",
        early_stopping_patience=10
    )
    
    # Visualiser courbes
    plot_training_curves(history, "training_curves_multi_organ.png")
    
    print("\n" + "="*70)
    print("✅ ENTRAÎNEMENT TERMINÉ!")
    print("="*70)
