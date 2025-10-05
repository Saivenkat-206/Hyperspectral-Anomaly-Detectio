import os
import numpy as np
from sklearn.model_selection import train_test_split

def assemble_dataset(patch_dir="Prepared", test_size=0.2, val_size=0.1, seed=42):
    all_patches = []
    patch_shapes = {}

    # Recursively find all *_patches.npy files in all subfolders
    for root, dirs, files in os.walk(patch_dir):
        for f in files:
            if f.endswith("_patches.npy"):
                path = os.path.join(root, f)
                patches = np.load(path)
                print(f"Loaded {patches.shape} from {path}")
                # Group by shape to avoid shape mismatch
                shape = patches.shape[1:]  # (Bands, H, W)
                if shape not in patch_shapes:
                    patch_shapes[shape] = []
                patch_shapes[shape].append(patches)

    # Use only the largest group of matching-shape patches
    if not patch_shapes:
        raise ValueError("No patch files found.")
    largest_shape = max(patch_shapes, key=lambda k: sum(arr.shape[0] for arr in patch_shapes[k]))
    all_patches = np.concatenate(patch_shapes[largest_shape], axis=0)
    print(f"\nUsing patches of shape {largest_shape}. Final dataset shape: {all_patches.shape} (N, Bands, H, W)")

    # Shuffle + split into train/val/test
    train_data, test_data = train_test_split(all_patches, test_size=test_size, random_state=seed)
    train_data, val_data = train_test_split(train_data, test_size=val_size, random_state=seed)

    print(f"Train: {train_data.shape}, Val: {val_data.shape}, Test: {test_data.shape}")

    # Save splits
    np.save(os.path.join(patch_dir, "train.npy"), train_data)
    np.save(os.path.join(patch_dir, "val.npy"), val_data)
    np.save(os.path.join(patch_dir, "test.npy"), test_data)

    return train_data, val_data, test_data

if __name__ == "__main__":
    train, val, test = assemble_dataset()