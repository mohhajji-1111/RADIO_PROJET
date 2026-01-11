"""
U-Net Architecture for Lung Tumor Segmentation (Phase 5).

Classic U-Net with encoder-decoder structure and skip connections.
Optimized for medical image segmentation.

Author: GitHub Copilot
Date: 2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """
    Double Convolution block: (Conv -> BN -> ReLU) x 2
    Standard building block for U-Net.
    """
    
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if mid_channels is None:
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
    """Downscaling with maxpool then double conv."""
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )
    
    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv."""
    
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        
        # Use bilinear upsampling or transposed conv
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)
    
    def forward(self, x1, x2):
        """
        x1: from decoder path (to be upsampled)
        x2: from encoder path (skip connection)
        """
        x1 = self.up(x1)
        
        # Handle size mismatch (if input size not divisible by 16)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        
        # Concatenate skip connection
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    """Final 1x1 convolution."""
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    
    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    """
    U-Net: Convolutional Networks for Biomedical Image Segmentation
    Reference: https://arxiv.org/abs/1505.04597
    
    Architecture:
    - Encoder (contracting path): 4 downsampling blocks
    - Bottleneck: deepest layer
    - Decoder (expanding path): 4 upsampling blocks with skip connections
    - Output: single channel sigmoid for binary segmentation
    """
    
    def __init__(self, n_channels=1, n_classes=1, bilinear=True):
        """
        Initialize U-Net.
        
        Args:
            n_channels: Number of input channels (1 for grayscale CT)
            n_classes: Number of output channels (1 for binary segmentation)
            bilinear: Use bilinear upsampling (True) or transposed conv (False)
        """
        super(UNet, self).__init__()
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
        Forward pass.
        
        Args:
            x: Input tensor (B, 1, H, W)
            
        Returns:
            Output tensor (B, 1, H, W) with sigmoid activation
        """
        # Encoder
        x1 = self.inc(x)      # 64 channels
        x2 = self.down1(x1)   # 128 channels
        x3 = self.down2(x2)   # 256 channels
        x4 = self.down3(x3)   # 512 channels
        x5 = self.down4(x4)   # 1024 channels (bottleneck)
        
        # Decoder with skip connections
        x = self.up1(x5, x4)  # 512 channels
        x = self.up2(x, x3)   # 256 channels
        x = self.up3(x, x2)   # 128 channels
        x = self.up4(x, x1)   # 64 channels
        
        # Output
        logits = self.outc(x)  # 1 channel
        return torch.sigmoid(logits)
    
    def count_parameters(self):
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def get_model_summary(model, input_size=(1, 1, 256, 256)):
    """
    Print model summary.
    
    Args:
        model: PyTorch model
        input_size: Input tensor size (B, C, H, W)
    """
    print("\n" + "="*70)
    print("U-NET MODEL SUMMARY")
    print("="*70)
    
    print(f"\nInput size: {input_size}")
    print(f"Total parameters: {model.count_parameters():,}")
    
    # Memory estimation
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    total_size_mb = (param_size + buffer_size) / (1024 ** 2)
    print(f"Model size: {total_size_mb:.2f} MB")
    
    # Test forward pass
    device = next(model.parameters()).device
    dummy_input = torch.randn(input_size).to(device)
    
    with torch.no_grad():
        output = model(dummy_input)
    
    print(f"Output size: {tuple(output.shape)}")
    print("="*70 + "\n")


if __name__ == "__main__":
    # Test model
    print("Testing U-Net Architecture...")
    
    model = UNet(n_channels=1, n_classes=1, bilinear=True)
    get_model_summary(model)
    
    # Test forward pass
    x = torch.randn(2, 1, 256, 256)  # Batch of 2
    y = model(x)
    print(f"Forward pass successful!")
    print(f"Input: {x.shape} -> Output: {y.shape}")
    print(f"Output range: [{y.min():.3f}, {y.max():.3f}]")
