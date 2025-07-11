import torch
import torch.nn as nn
import torch.nn.functional as F

class MirrorAutoencoder(nn.Module):
    def __init__(self):
        super(MirrorAutoencoder, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),   # (1, 18, 16) -> (8,18,16)
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2, padding=1)         # -> (8,9,8)
            nn.Conv2d(8, 4, kernel_size=3, padding=1),   # -> (4,9,8)
            nn.ReLU(),
            nn.MaxPoll2d(2, stride=2, padding=1)         # -> (4, 5, 4)

        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Conv2d(4, 4, kernel_size=3, padding=1)    # (4,5,4)
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode="nearest")  # (4,10,8)
            nn.Conv2d(4, 8, kernel_size=3, padding=1)    # (8,10,8)
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode="nearest")  # (8, 20, 16)
            nn.Conv2d(8, 1, kernel_size=3, padding=1)    # (1, 20, 10)
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(x)
        return decoded