import torch.nn as nn

class ConvDenoiser(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=31, stride=2, padding=15), nn.ReLU(),
            nn.Conv1d(16,32, kernel_size=31, stride=2, padding=15), nn.ReLU(),
            nn.Conv1d(32,64, kernel_size=31, stride=2, padding=15), nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(64,32, kernel_size=31, stride=2, padding=15, output_padding=1), nn.ReLU(),
            nn.ConvTranspose1d(32,16, kernel_size=31, stride=2, padding=15, output_padding=1), nn.ReLU(),
            nn.ConvTranspose1d(16,1,  kernel_size=31, stride=2, padding=15, output_padding=1)
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))
