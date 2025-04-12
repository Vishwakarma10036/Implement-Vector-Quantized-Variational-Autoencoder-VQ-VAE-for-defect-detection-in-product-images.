import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image
import numpy as np
import cv2
from matplotlib import cm

# Define supported MVTec classes
CLASS_NAMES = ['bottle', 'capsule', 'grid', 'metal_nut', 'pill', 'transistor']

# Dataset class
class MVTecDataset(Dataset):
    def __init__(self, dataset_path, class_name='bottle', is_train=True, resize=256):
        assert class_name in CLASS_NAMES, f"class_name: {class_name} should be in {CLASS_NAMES}"
        self.dataset_path = dataset_path
        self.class_name = class_name
        self.is_train = is_train
        self.resize = resize

        self.x, self.y, self.mask = self.load_dataset_folder()

        self.transform_x = transforms.Compose([
            transforms.Resize((resize, resize), Image.Resampling.LANCZOS),
            transforms.ToTensor()
        ])
        self.transform_mask = transforms.Compose([
            transforms.Resize((resize, resize), Image.Resampling.NEAREST),
            transforms.ToTensor()
        ])

    def __getitem__(self, idx):
        x, y, mask = self.x[idx], self.y[idx], self.mask[idx]
        x = Image.open(x).convert('RGB')
        x = self.transform_x(x)

        if y == 0 or mask is None:
            mask = torch.zeros([1, self.resize, self.resize])
        else:
            mask = Image.open(mask)
            mask = self.transform_mask(mask)

        return x, y, mask

    def __len__(self):
        return len(self.x)

    def load_dataset_folder(self):
        phase = 'train' if self.is_train else 'test'
        x, y, mask = [], [], []

        img_dir = os.path.join(self.dataset_path, self.class_name, phase)
        gt_dir = os.path.join(self.dataset_path, self.class_name, 'ground_truth')

        img_types = sorted(os.listdir(img_dir))
        for img_type in img_types:
            img_type_dir = os.path.join(img_dir, img_type)
            if not os.path.isdir(img_type_dir):
                continue
            img_fpath_list = sorted([
                os.path.join(img_type_dir, f) for f in os.listdir(img_type_dir) if f.endswith('.png')
            ])
            x.extend(img_fpath_list)

            if img_type == 'good':
                y.extend([0] * len(img_fpath_list))
                mask.extend([None] * len(img_fpath_list))
            else:
                y.extend([1] * len(img_fpath_list))
                gt_type_dir = os.path.join(gt_dir, img_type)
                img_fname_list = [os.path.splitext(os.path.basename(f))[0] for f in img_fpath_list]
                gt_fpath_list = [os.path.join(gt_type_dir, fname + '_mask.png') for fname in img_fname_list]
                mask.extend(gt_fpath_list)

        return x, y, mask

# VQ-VAE Components
class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost):
        super(VectorQuantizer, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.embeddings = nn.Embedding(num_embeddings, embedding_dim)
        self.embeddings.weight.data.uniform_(-1 / num_embeddings, 1 / num_embeddings)
        self.commitment_cost = commitment_cost

    def forward(self, z):
        z_flattened = z.permute(0, 2, 3, 1).contiguous().view(-1, self.embedding_dim)
        distances = (
            torch.sum(z_flattened ** 2, dim=1, keepdim=True)
            + torch.sum(self.embeddings.weight ** 2, dim=1)
            - 2 * torch.matmul(z_flattened, self.embeddings.weight.t())
        )
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self.num_embeddings, device=z.device)
        encodings.scatter_(1, encoding_indices, 1)
        quantized = torch.matmul(encodings, self.embeddings.weight).view(z.shape)
        loss = torch.mean((quantized.detach() - z) ** 2) + self.commitment_cost * torch.mean((quantized - z.detach()) ** 2)
        quantized = z + (quantized - z).detach()
        return quantized, loss

class Encoder(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, hidden_channels, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, hidden_channels, 3, 1, 1)
        )

    def forward(self, x):
        return self.encoder(x)

class Decoder(nn.Module):
    def __init__(self, embedding_dim, hidden_channels):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(embedding_dim, hidden_channels, 3, 1, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(hidden_channels, hidden_channels, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(hidden_channels, 3, 4, 2, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.decoder(x)

class VQVAE(nn.Module):
    def __init__(self, in_channels=3, hidden_channels=128, embedding_dim=64, num_embeddings=512, commitment_cost=0.25):
        super(VQVAE, self).__init__()
        self.encoder = Encoder(in_channels, hidden_channels)
        self.proj = nn.Conv2d(hidden_channels, embedding_dim, 1)
        self.quantizer = VectorQuantizer(num_embeddings, embedding_dim, commitment_cost)
        self.decoder = Decoder(embedding_dim, hidden_channels)

    def forward(self, x):
        z = self.encoder(x)
        z = self.proj(z)
        quantized, loss = self.quantizer(z)
        x_recon = self.decoder(quantized)
        return x_recon, loss

# Visualization
def visualize_results(model, test_loader, class_name, save_dir="results", threshold=0.05):
    os.makedirs(save_dir, exist_ok=True)
    model.eval()

    with torch.no_grad():
        for idx, (x, y, mask) in enumerate(test_loader):
            x = x.to(device)
            x_recon, _ = model(x)
            recon_error = torch.mean((x - x_recon) ** 2, dim=1, keepdim=True)

            error_map = recon_error.squeeze().cpu().numpy()
            norm_map = (error_map - error_map.min()) / (error_map.max() - error_map.min() + 1e-8)
            heatmap = cm.jet(norm_map)[:, :, :3]
            heatmap = (heatmap * 255).astype(np.uint8)

            pred_mask = (norm_map > threshold).astype(np.uint8) * 255
            original = x.cpu().squeeze().permute(1, 2, 0).numpy()
            original = (original * 255).astype(np.uint8)

            overlay = original.copy()
            overlay[pred_mask == 255] = [255, 0, 0]

            gt_mask = mask.squeeze().cpu().numpy()
            gt_mask = (gt_mask * 255).astype(np.uint8)

            result_row = np.hstack([
                cv2.cvtColor(original, cv2.COLOR_RGB2BGR),
                cv2.cvtColor((x_recon.squeeze().cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8), cv2.COLOR_RGB2BGR),
                cv2.cvtColor(gt_mask, cv2.COLOR_GRAY2BGR),
                cv2.cvtColor(heatmap, cv2.COLOR_RGB2BGR),
                cv2.cvtColor(pred_mask, cv2.COLOR_GRAY2BGR),
                overlay
            ])
            cv2.imwrite(os.path.join(save_dir, f"{class_name}_{idx:03d}_viz.png"), result_row)

            if idx == 15:
                break

    print(f"[{class_name}] Visualizations saved in '{save_dir}'.")

# ========== MAIN LOOP ==========
DATASET_PATH = r"C:\Users\jayhe\Desktop\mvtec_anomaly_detection"
batch_size = 16
epochs = 30
lr = 1e-3
resize = 256

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

for class_name in CLASS_NAMES:
    print(f"\n=== Training for class: {class_name} ===")
    train_dataset = MVTecDataset(DATASET_PATH, class_name=class_name, is_train=True, resize=resize)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataset = MVTecDataset(DATASET_PATH, class_name=class_name, is_train=False, resize=resize)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    model = VQVAE().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for x, _, _ in train_loader:
            x = x.to(device)
            x_recon, vq_loss = model(x)
            recon_loss = torch.mean((x - x_recon) ** 2)
            loss = recon_loss + vq_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"[{class_name}] Epoch {epoch+1}/{epochs} Loss: {total_loss / len(train_loader):.4f}")

    # Visualization for current class
    visualize_results(model, test_loader, class_name, save_dir=f"results/{class_name}")

print("✅ All classes processed.")
