import torch
import torch.nn as nn
import torch.nn.functional as F

class DishAutoencoder(nn.Module):
    def __init__(self, feature_dim=64):
        super().__init__()
        # Encoder: Adjusted to handle larger images
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),  # -> (16, 280, 290)
            nn.ReLU(),
            nn.MaxPool2d(2, 2),                        # -> (16, 140, 145)
            nn.Conv2d(16, 32, kernel_size=3, padding=1), # -> (32, 140, 145)
            nn.ReLU(),
            nn.MaxPool2d(2, 2),                        # -> (32, 70, 72)
            nn.Conv2d(32, 64, kernel_size=3, padding=1), # -> (64, 70, 72)
            nn.ReLU(),
            nn.MaxPool2d(2, 2),                        # -> (64, 35, 36)
            nn.Conv2d(64, 128, kernel_size=3, padding=1), # -> (128, 35, 36)
            nn.ReLU()
        )

        # Feature vector: Adaptive pooling to a fixed-size vector
        self.feature_pool = nn.AdaptiveAvgPool2d(1)  # -> (128, 1, 1)
        self.feature_proj = nn.Linear(128, feature_dim)  # Final feature vector of size feature_dim

        # Decoder: Reconstruct the image from the feature vector
        self.decoder = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),   # -> (64, 35, 36)
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode="nearest"),    # -> (64, 70, 72)
            nn.Conv2d(64, 32, kernel_size=3, padding=1),    # -> (32, 70, 72)
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode="nearest"),    # -> (32, 140, 145)
            nn.Conv2d(32, 16, kernel_size=3, padding=1),    # -> (16, 140, 145)
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode="nearest"),    # -> (16, 280, 290)
            nn.Conv2d(16, 1, kernel_size=3, padding=1),     # -> (1, 280, 290)
            nn.Sigmoid()
        )

    def forward(self, x):
        # Encoder: extracting the feature representation
        x_enc = self.encoder_conv(x)            # (B, 128, H/16, W/16)
        
        # Feature vector: for classification or clustering
        pooled = self.feature_pool(x_enc)       # (B, 128, 1, 1)
        feat = pooled.view(pooled.size(0), -1)  # (B, 128)
        feature_vector = self.feature_proj(feat)  # (B, feature_dim)

        # Decoder: reconstructing the image
        decoded = self.decoder(x_enc)           # (B, 1, H, W) approximate reconstruction
        
        # In case of slight size mismatches after upsampling, adjust the output size
        H, W = x.shape[2], x.shape[3]
        decoded = F.interpolate(decoded, size=(H, W), mode="nearest")
        
        return decoded, feature_vector
