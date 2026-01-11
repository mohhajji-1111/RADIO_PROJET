"""
Training Pipeline for U-Net Lung Tumor Segmentation (Phase 6).

Includes:
- Dice Loss + BCE Loss
- Training loop with validation
- Metrics: Dice, IoU, Sensitivity, Specificity
- Checkpointing and early stopping
- Learning rate scheduling

Author: GitHub Copilot
Date: 2025
"""

import os
from pathlib import Path
import time
import json

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm


class DiceLoss(nn.Module):
    """Dice Loss for binary segmentation."""
    
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
    
    def forward(self, predictions, targets):
        """
        Args:
            predictions: (B, 1, H, W) - sigmoid outputs [0, 1]
            targets: (B, 1, H, W) - binary masks {0, 1}
        """
        predictions = predictions.view(-1)
        targets = targets.view(-1)
        
        intersection = (predictions * targets).sum()
        dice = (2. * intersection + self.smooth) / (
            predictions.sum() + targets.sum() + self.smooth
        )
        
        return 1 - dice


class CombinedLoss(nn.Module):
    """Combined Dice Loss + Binary Cross Entropy."""
    
    def __init__(self, dice_weight=0.5, bce_weight=0.5):
        super(CombinedLoss, self).__init__()
        self.dice_loss = DiceLoss()
        self.bce_loss = nn.BCELoss()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
    
    def forward(self, predictions, targets):
        dice = self.dice_loss(predictions, targets)
        bce = self.bce_loss(predictions, targets)
        return self.dice_weight * dice + self.bce_weight * bce


def calculate_metrics(predictions, targets, threshold=0.5):
    """
    Calculate segmentation metrics.
    
    Args:
        predictions: (B, 1, H, W) - sigmoid outputs [0, 1]
        targets: (B, 1, H, W) - binary masks {0, 1}
        threshold: Threshold for binary prediction
        
    Returns:
        Dictionary with metrics
    """
    # Binarize predictions
    pred_binary = (predictions > threshold).float()
    
    # Flatten
    pred_flat = pred_binary.view(-1)
    target_flat = targets.view(-1)
    
    # True/False Positives/Negatives
    tp = (pred_flat * target_flat).sum().item()
    fp = (pred_flat * (1 - target_flat)).sum().item()
    fn = ((1 - pred_flat) * target_flat).sum().item()
    tn = ((1 - pred_flat) * (1 - target_flat)).sum().item()
    
    # Metrics
    epsilon = 1e-7
    
    dice = (2 * tp) / (2 * tp + fp + fn + epsilon)
    iou = tp / (tp + fp + fn + epsilon)
    sensitivity = tp / (tp + fn + epsilon)
    specificity = tn / (tn + fp + epsilon)
    precision = tp / (tp + fp + epsilon)
    
    return {
        'dice': dice,
        'iou': iou,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'precision': precision
    }


class Trainer:
    """Training manager for U-Net."""
    
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        device,
        checkpoint_dir='checkpoints',
        log_dir='logs'
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_dice': [],
            'val_iou': [],
            'learning_rates': []
        }
        
        self.best_val_dice = 0.0
        self.epochs_without_improvement = 0
    
    def train_epoch(self, epoch):
        """Train for one epoch."""
        self.model.train()
        
        total_loss = 0.0
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch} [Train]")
        
        for batch_idx, batch in enumerate(progress_bar):
            images = batch['image'].to(self.device)
            masks = batch['mask'].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            predictions = self.model(images)
            loss = self.criterion(predictions, masks)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
            # Update progress bar
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = total_loss / len(self.train_loader)
        return avg_loss
    
    def validate(self, epoch):
        """Validate the model."""
        self.model.eval()
        
        total_loss = 0.0
        all_metrics = []
        
        progress_bar = tqdm(self.val_loader, desc=f"Epoch {epoch} [Val]")
        
        with torch.no_grad():
            for batch in progress_bar:
                images = batch['image'].to(self.device)
                masks = batch['mask'].to(self.device)
                
                # Forward pass
                predictions = self.model(images)
                loss = self.criterion(predictions, masks)
                
                total_loss += loss.item()
                
                # Calculate metrics
                metrics = calculate_metrics(predictions, masks)
                all_metrics.append(metrics)
                
                # Update progress bar
                progress_bar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'dice': f'{metrics["dice"]:.4f}'
                })
        
        # Average metrics
        avg_loss = total_loss / len(self.val_loader)
        avg_metrics = {
            key: np.mean([m[key] for m in all_metrics])
            for key in all_metrics[0].keys()
        }
        
        return avg_loss, avg_metrics
    
    def save_checkpoint(self, epoch, val_dice, is_best=False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_dice': val_dice,
            'history': self.history
        }
        
        # Save latest
        latest_path = self.checkpoint_dir / 'latest_checkpoint.pth'
        torch.save(checkpoint, latest_path)
        
        # Save best
        if is_best:
            best_path = self.checkpoint_dir / 'best_model.pth'
            torch.save(checkpoint, best_path)
            print(f"✓ Saved best model (Dice: {val_dice:.4f})")
    
    def train(self, num_epochs, patience=10, scheduler=None):
        """
        Train the model.
        
        Args:
            num_epochs: Number of epochs
            patience: Early stopping patience
            scheduler: Learning rate scheduler
        """
        print("\n" + "="*70)
        print("TRAINING U-NET")
        print("="*70)
        print(f"Device: {self.device}")
        print(f"Epochs: {num_epochs}")
        print(f"Train batches: {len(self.train_loader)}")
        print(f"Val batches: {len(self.val_loader)}")
        print("="*70 + "\n")
        
        start_time = time.time()
        
        for epoch in range(1, num_epochs + 1):
            print(f"\nEpoch {epoch}/{num_epochs}")
            print("-" * 70)
            
            # Train
            train_loss = self.train_epoch(epoch)
            
            # Validate
            val_loss, val_metrics = self.validate(epoch)
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['val_dice'].append(val_metrics['dice'])
            self.history['val_iou'].append(val_metrics['iou'])
            
            if scheduler:
                current_lr = self.optimizer.param_groups[0]['lr']
                self.history['learning_rates'].append(current_lr)
                scheduler.step(val_metrics['dice'])
            
            # Print summary
            print(f"\nEpoch {epoch} Summary:")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss: {val_loss:.4f}")
            print(f"  Val Dice: {val_metrics['dice']:.4f}")
            print(f"  Val IoU: {val_metrics['iou']:.4f}")
            print(f"  Val Sensitivity: {val_metrics['sensitivity']:.4f}")
            print(f"  Val Specificity: {val_metrics['specificity']:.4f}")
            
            # Check for improvement
            is_best = val_metrics['dice'] > self.best_val_dice
            if is_best:
                self.best_val_dice = val_metrics['dice']
                self.epochs_without_improvement = 0
            else:
                self.epochs_without_improvement += 1
            
            # Save checkpoint
            self.save_checkpoint(epoch, val_metrics['dice'], is_best)
            
            # Early stopping
            if self.epochs_without_improvement >= patience:
                print(f"\n⚠️  Early stopping triggered after {epoch} epochs")
                print(f"   No improvement for {patience} epochs")
                break
        
        # Training complete
        elapsed_time = time.time() - start_time
        print("\n" + "="*70)
        print("TRAINING COMPLETE")
        print("="*70)
        print(f"Total time: {elapsed_time / 60:.2f} minutes")
        print(f"Best validation Dice: {self.best_val_dice:.4f}")
        print("="*70 + "\n")
        
        # Save history
        history_path = self.log_dir / 'training_history.json'
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        print(f"Training history saved: {history_path}")


if __name__ == "__main__":
    print("Training Pipeline Module for U-Net")
    print("Import this module to use training classes")
