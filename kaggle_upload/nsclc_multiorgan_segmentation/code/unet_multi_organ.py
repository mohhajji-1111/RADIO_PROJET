"""
U-Net multi-classes pour segmentation multi-organes
8 classes: Background + 7 organes

Architecture:
- Encoder: 4 niveaux de downsampling
- Bottleneck: Plus profond
- Decoder: 4 niveaux de upsampling avec skip connections
- Output: 8 canaux (1 par classe)

Auteur: Copilot
Date: 21 Nov 2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """
    Double convolution: Conv → BN → ReLU → Conv → BN → ReLU
    """
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """
    Downsampling: MaxPool → DoubleConv
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )
    
    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """
    Upsampling: Upsample → DoubleConv avec skip connection
    """
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        
        # Upsampling
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)
    
    def forward(self, x1, x2):
        """
        x1: feature map de l'encoder (skip connection)
        x2: feature map upsamplée du decoder
        """
        x1 = self.up(x1)
        
        # Padding si les tailles ne matchent pas
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        
        # Concatener skip connection
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    """
    Convolution finale 1×1 pour obtenir le nombre de classes
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    
    def forward(self, x):
        return self.conv(x)


class UNetMultiOrgan(nn.Module):
    """
    U-Net pour segmentation multi-organes
    
    Args:
        n_channels: Nombre de canaux en entrée (1 pour CT grayscale)
        n_classes: Nombre de classes (8: background + 7 organes)
        bilinear: Utiliser bilinear upsampling (plus léger) ou ConvTranspose2d
    """
    def __init__(self, n_channels=1, n_classes=8, bilinear=False):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        
        # Encoder
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        
        # Decoder
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        
        # Output
        self.outc = OutConv(64, n_classes)
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor (B, 1, H, W)
            
        Returns:
            logits: Tensor (B, n_classes, H, W)
        """
        # Encoder avec skip connections
        x1 = self.inc(x)      # (B, 64, H, W)
        x2 = self.down1(x1)   # (B, 128, H/2, W/2)
        x3 = self.down2(x2)   # (B, 256, H/4, W/4)
        x4 = self.down3(x3)   # (B, 512, H/8, W/8)
        x5 = self.down4(x4)   # (B, 1024, H/16, W/16)
        
        # Decoder avec skip connections
        x = self.up1(x5, x4)  # (B, 512, H/8, W/8)
        x = self.up2(x, x3)   # (B, 256, H/4, W/4)
        x = self.up3(x, x2)   # (B, 128, H/2, W/2)
        x = self.up4(x, x1)   # (B, 64, H, W)
        
        # Output
        logits = self.outc(x) # (B, n_classes, H, W)
        return logits
    
    def predict(self, x):
        """
        Prédiction avec argmax
        
        Args:
            x: Input tensor (B, 1, H, W)
            
        Returns:
            predictions: Tensor (B, H, W) avec les labels prédits
        """
        logits = self.forward(x)
        predictions = torch.argmax(logits, dim=1)
        return predictions


def count_parameters(model):
    """Compte le nombre de paramètres entraînables"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    """Test du modèle"""
    
    # Créer modèle
    model = UNetMultiOrgan(n_channels=1, n_classes=8, bilinear=False)
    
    # Compter paramètres
    num_params = count_parameters(model)
    print(f"Nombre de paramètres: {num_params:,}")
    
    # Test forward pass
    batch_size = 2
    x = torch.randn(batch_size, 1, 256, 256)
    
    print(f"\nInput shape: {x.shape}")
    
    # Forward
    logits = model(x)
    print(f"Output logits shape: {logits.shape}")
    
    # Prédictions
    predictions = model.predict(x)
    print(f"Predictions shape: {predictions.shape}")
    print(f"Unique labels: {torch.unique(predictions).tolist()}")
    
    # Taille du modèle
    model_size_mb = num_params * 4 / (1024**2)  # 4 bytes par float32
    print(f"\nTaille du modèle: {model_size_mb:.2f} MB")
    
    print("\n✅ U-Net multi-organes testé avec succès!")
