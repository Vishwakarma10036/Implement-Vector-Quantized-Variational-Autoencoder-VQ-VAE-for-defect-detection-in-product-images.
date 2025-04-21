import os
import torch
from torch.utils.data import DataLoader
from torch import optim
from vqvae_model import VQVAE
from mvtec_dataset import MVTecDataset
from visualize import visualize_results  # assuming you saved your visualize function here

# Configurations
DATASET_PATH = r"F:\mvtec_anomaly_detection"
CLASS_NAMES = ['bottle', 'capsule', 'grid', 'metal_nut', 'pill', 'transistor']
batch_size = 16
epochs = 30
lr = 1e-3
resize = 256
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Training loop for each class
for class_name in CLASS_NAMES:
    print(f"\n=== Training for class: {class_name} ===")

    # Prepare datasets and dataloaders
    train_dataset = MVTecDataset(DATASET_PATH, class_name=class_name, is_train=True, resize=resize)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    test_dataset = MVTecDataset(DATASET_PATH, class_name=class_name, is_train=False, resize=resize)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # Initialize model and optimizer
    model = VQVAE().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Training
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for batch in train_loader:
            x, _, _ = batch
            x = x.to(device)

            x_recon, vq_loss = model(x)
            recon_loss = torch.mean((x - x_recon) ** 2)
            loss = recon_loss + vq_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"[{class_name}] Epoch {epoch+1}/{epochs} Loss: {avg_loss:.4f}")

    # Save final model weights for the class
    os.makedirs(f"checkpoints/{class_name}", exist_ok=True)
    torch.save(model.state_dict(), f"checkpoints/{class_name}/vqvae_final.pth")

    # Visualize results after training
    visualize_results(model, test_loader, class_name, save_dir=f"results/{class_name}")

print("✅ Training and visualization completed for all classes.")
