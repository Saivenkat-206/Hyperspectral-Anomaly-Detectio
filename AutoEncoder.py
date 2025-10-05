import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve
import os
import glob

# Autoencoder definition
class HyperspectralAutoencoder(nn.Module):
    def __init__(self, bands):
        super(HyperspectralAutoencoder, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(bands, 64, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(True)
        )
        
        # Bottleneck
        self.bottleneck = nn.Conv2d(16, 8, kernel_size=3, padding=1)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(64, bands, kernel_size=3, padding=1),
            nn.Sigmoid()  # normalize outputs [0,1]
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.bottleneck(x)
        x = self.decoder(x)
        return x


# Training function
def train_autoencoder(cube, epochs=20, lr=1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows, cols, bands = cube.shape

    # Normalize cube
    cube = (cube - cube.min()) / (cube.max() - cube.min())

    # Tensor format (N, C, H, W)
    X = torch.tensor(cube, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device)

    model = HyperspectralAutoencoder(bands).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        recon = model(X)
        loss = criterion(recon, X)
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.6f}")

    return model


# Anomaly detection
def anomaly_map(cube, model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    # Normalize cube
    cube = (cube - cube.min()) / (cube.max() - cube.min())
    X = torch.tensor(cube, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device)

    with torch.no_grad():
        recon = model(X)
        err = torch.mean((X - recon) ** 2, dim=1).squeeze().cpu().numpy()  # (H, W)

    return err


# Main execution
if __name__ == "__main__":
    data_root = "Data"
    mat_files = [y for x in os.walk(data_root) for y in glob.glob(os.path.join(x[0], '*.mat'))]

    print(f"Found {len(mat_files)} .mat files in {data_root}.")

    for mat_path in mat_files:
        print(f"\nProcessing: {mat_path}")
        mat = sio.loadmat(mat_path)

        # Find hyperspectral cube
        cube_key = None
        for key in mat:
            if not key.startswith('__') and isinstance(mat[key], np.ndarray) and mat[key].ndim == 3:
                cube_key = key
                break
        if cube_key is None:
            print(f"  Skipped: No 3D hyperspectral cube found in {mat_path}.")
            continue
        cube = mat[cube_key]
        print(f"  Loaded cube with shape: {cube.shape}")

        # Train model
        model = train_autoencoder(cube, epochs=20, lr=1e-3)

        # Get anomaly map
        anomaly = anomaly_map(cube, model)

        # Evaluation with GT map
        if "map" in mat:
            gt = mat["map"].astype(np.int32)

            # Flatten both
            y_true = gt.flatten()
            y_score = anomaly.flatten()

            # ROC AUC
            auc = roc_auc_score(y_true, y_score)
            print(f"  ROC AUC: {auc:.4f}")

            # Plot ROC
            fpr, tpr, _ = roc_curve(y_true, y_score)
            plt.figure()
            plt.plot(fpr, tpr, label=f"AUC={auc:.4f}")
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.legend()
            plt.title(f"ROC Curve: {os.path.basename(mat_path)}")
            plt.show()

        # Show anomaly heatmap
        plt.figure(figsize=(6, 5))
        plt.imshow(anomaly, cmap="hot")
        plt.title(f"Anomaly Map: {os.path.basename(mat_path)}")
        plt.colorbar()
        plt.show()
