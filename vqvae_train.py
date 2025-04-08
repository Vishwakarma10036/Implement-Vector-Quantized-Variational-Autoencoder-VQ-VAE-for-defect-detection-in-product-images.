import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from PIL import Image
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
from torchvision.datasets import ImageFolder

# ---- CONFIG ----
IMAGE_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 1e-3

# ---- DATASET ----
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor()
])

data_path = './DataSets'
dataset = ImageFolder(root=data_path, transform=transform)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# ---- VQ LAYER ----
class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost):
        super(VectorQuantizer, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost

        self.embeddings = nn.Embedding(self.num_embeddings, self.embedding_dim)
        self.embeddings.weight.data.uniform_(-1/self.num_embeddings, 1/self.num_embeddings)

    def forward(self, z):  # z: (B, H, W, C)
        flat_z = z.reshape(-1, self.embedding_dim)  # (B*H*W, C)

        distances = (
            flat_z.pow(2).sum(1, keepdim=True)
            - 2 * flat_z @ self.embeddings.weight.t()
            + self.embeddings.weight.pow(2).sum(1)
        )

        encoding_indices = torch.argmin(distances, dim=1)
        quantized = self.embeddings(encoding_indices).view(z.shape)

        # VQ Loss
        loss = F.mse_loss(quantized.detach(), z) + self.commitment_cost * F.mse_loss(quantized, z.detach())

        quantized = z + (quantized - z).detach()
        return quantized, loss

# ---- VQ-VAE ----
class VQVAE(nn.Module):
    def __init__(self, num_embeddings=512, embedding_dim=64, commitment_cost=0.25):
        super(VQVAE, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(128, embedding_dim, 3, 1, 1),
            nn.ReLU()
        )

        self.vq = VectorQuantizer(num_embeddings, embedding_dim, commitment_cost)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(embedding_dim, 128, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, 3, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        z = self.encoder(x)             # (B, C, H, W)
        z = z.permute(0, 2, 3, 1)       # (B, H, W, C)
        z_q, vq_loss = self.vq(z)
        z_q = z_q.permute(0, 3, 1, 2)   # (B, C, H, W)
        x_recon = self.decoder(z_q)
        return x_recon, vq_loss

# ---- TRAINING ----
def train(model, dataloader):
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    model.train()

    for epoch in range(EPOCHS):
        loop = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        total_loss = 0

        for imgs, _ in loop:
            recon, vq_loss = model(imgs)
            recon_loss = F.mse_loss(recon, imgs)
            loss = recon_loss + vq_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loop.set_postfix(loss=loss.item())

        print(f"Epoch [{epoch+1}/{EPOCHS}] Loss: {loss.item():.4f}")

# ---- RUN ----
if __name__ == '__main__':
    model = VQVAE()
    train(model, dataloader)
