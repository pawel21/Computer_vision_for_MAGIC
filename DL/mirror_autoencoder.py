import torch
import torch.nn as nn
import torch.nn.functional as F

class MirrorAutoencoder(nn.Module):
    def __init__(self, feature_dim=64):
        super().__init__()
        # Encoder: zachowujemy rozmiar przestrzenny dowolny, kończymy adaptacyjnym poolingiem do wektora
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),   # -> (8, H, W)
            nn.ReLU(),
            nn.MaxPool2d(2, 2),                          # -> (8, H/2, W/2)
            nn.Conv2d(8, 16, kernel_size=3, padding=1),  # -> (16, H/2, W/2)
            nn.ReLU(),
            nn.MaxPool2d(2, 2),                          # -> (16, H/4, W/4)
            nn.Conv2d(16, 32, kernel_size=3, padding=1), # -> (32, H/4, W/4)
            nn.ReLU()
        )
        # Stały wektor cech (np. do klasyfikatora) niezależnie od rozmiaru
        self.feature_pool = nn.AdaptiveAvgPool2d(1)  # -> (32,1,1)
        self.feature_proj = nn.Linear(32, feature_dim)  # ostateczny wektor cech

        # Decoder: odbudowa z zakodowanej reprezentacji przestrzennej
        self.decoder = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3, padding=1),  # -> (16, H/4, W/4)
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode="nearest"),  # -> (16, H/2, W/2)
            nn.Conv2d(16, 8, kernel_size=3, padding=1),   # -> (8, H/2, W/2)
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode="nearest"),  # -> (8, H, W)
            nn.Conv2d(8, 1, kernel_size=3, padding=1),    # -> (1, H, W)
            nn.Sigmoid()
        )

    def forward(self, x):
        x_enc = self.encoder_conv(x)          # (B,32, H/4, W/4)
        # wektor cech do klasyfikacji
        pooled = self.feature_pool(x_enc)     # (B,32,1,1)
        feat = pooled.view(pooled.size(0), -1)  # (B,32)
        feature_vector = self.feature_proj(feat)  # (B, feature_dim)

        # rekonstrukcja
        decoded = self.decoder(x_enc)          # (B,1, H, W) w przybliżeniu
        # jeśli rozmiar się nie zgadza z wejściem (np. zaokrąglenia), dopasuj:
        H, W = x.shape[2], x.shape[3]
        decoded = F.interpolate(decoded, size=(H, W), mode="nearest")
        return decoded, feature_vector