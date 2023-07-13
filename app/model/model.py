import re
import torch
import numpy as np
import torch.nn as nn
from pathlib import Path
from torchvision import transforms
from PIL import Image

__version__ = '0.1.0'

BASE_DIR = Path(__file__).resolve(strict=True).parent.parent


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 4, stride=2, padding=1)   # 32 x 14 x 14
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2, padding=1)  # 64 x 7 x 7
        self.conv3 = nn.Conv2d(64, 128, 7)                      # 128 x 1 x 1
        self.relu = nn.ReLU()
        self.fc = nn.Linear(128, 100)
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.flatten(x)
        x = self.fc(x)
        return x


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(100, 128)
        self.conv3 = nn.ConvTranspose2d(128, 64, 7)
        self.conv2 = nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1)
        self.conv1 = nn.ConvTranspose2d(32, 1, 4, stride=2, padding=1)
        self.relu = nn.ReLU()
        self.unflatten = nn.Unflatten(1, (128, 1, 1))

    def forward(self, x):
        x = self.fc(x)
        x = self.unflatten(x)
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv2(x))
        x = self.conv1(x)
        return x


class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        latent = self.encoder(x)
        recon = self.decoder(latent)
        return recon


def transform_img(image):
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
    ])
    image = Image.open(image)
    transformed_image = transform(image)
    return transformed_image


def add_noise(image):
    noisy_image = image + np.random.randn(*image.shape)
    return noisy_image
