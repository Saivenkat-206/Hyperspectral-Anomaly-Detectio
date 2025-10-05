import os
import numpy as np
import scipy.io as sio
from numpy.linalg import inv
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

DATA_DIR = "Data"

def load_hsi(path):
    """Load hyperspectral cube and ground truth map from .mat file."""
    d = sio.loadmat(path)
    cube = d['data']   # (rows, cols, bands)
    gt   = d['map']    # (rows, cols), anomaly map
    return cube, gt

def rx_detector(X, mu, cov_inv):
    """RX anomaly detector (Mahalanobis distance)."""
    diffs = X - mu
    scores = np.einsum('ij,jk,ik->i', diffs, cov_inv, diffs)
    return scores

def run_rx_on_cube(cube):
    """Run RX detector on a hyperspectral cube."""
    rows, cols, bands = cube.shape
    X = cube.reshape(rows*cols, bands)
    
    mu = np.mean(X, axis=0)
    cov = np.cov(X, rowvar=False)
    cov_inv = inv(cov)
    
    scores = rx_detector(X, mu, cov_inv)
    score_map = scores.reshape(rows, cols)
    return score_map, scores

def process_dataset(root_dir=DATA_DIR, save_figs=True):
    """Run RX detector on all datasets (Airport, Beach, Urban)."""
    results = {}
    for category in os.listdir(root_dir):
        folder = os.path.join(root_dir, category)
        if os.path.isdir(folder):
            print(f"\nProcessing category: {category}")
            results[category] = {}
            
            for fname in os.listdir(folder):
                if fname.endswith(".mat"):
                    path = os.path.join(folder, fname)
                    print(f"  -> {fname}")
                    
                    cube, gt = load_hsi(path)
                    score_map, scores = run_rx_on_cube(cube)
                    
                    # Threshold for anomaly mask
                    threshold = np.percentile(scores, 99)
                    mask = (score_map > threshold)
                    
                    # Evaluation (if ground truth is provided)
                    auc = roc_auc_score(gt.flatten() > 0, scores)
                    print(f"     ROC AUC: {auc:.4f}")
                    
                    # Save result
                    results[category][fname] = {
                        "score_map": score_map,
                        "mask": mask,
                        "auc": auc
                    }
                    
                    if save_figs:
                        plt.figure(figsize=(12,5))
                        plt.subplot(1,2,1)
                        plt.title(f"{fname} - Anomaly Scores")
                        plt.imshow(score_map, cmap="jet")
                        plt.colorbar()
                        
                        plt.subplot(1,2,2)
                        plt.title(f"{fname} - Anomaly Mask")
                        plt.imshow(mask, cmap="gray")
                        
                        plt.tight_layout()
                        plt.show()
    return results

# Run the whole pipeline
results = process_dataset()