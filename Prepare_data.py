import os
import scipy.io as sio
import numpy as np

def prepare_data(root_dir="Data", out_dir="Prepared", patch_size=None, stride=5):
    os.makedirs(out_dir, exist_ok=True)

    for category in os.listdir(root_dir):
        cat_path = os.path.join(root_dir, category)
        if not os.path.isdir(cat_path):
            continue

        out_cat_path = os.path.join(out_dir, category)
        os.makedirs(out_cat_path, exist_ok=True)

        for file in os.listdir(cat_path):
            if not file.endswith(".mat"):
                continue

            file_path = os.path.join(cat_path, file)
            mat = sio.loadmat(file_path)

            # Auto-detect cube variable (3D array)
            cube_key = None
            for key in mat:
                if not key.startswith("__") and isinstance(mat[key], np.ndarray) and mat[key].ndim == 3:
                    cube_key = key
                    break
            if cube_key is None:
                print(f"[!] No 3D cube found in {file}")
                continue

            cube = mat[cube_key].astype(np.float32)

            # Normalize [0,1]
            cube = (cube - cube.min()) / (cube.max() - cube.min() + 1e-8)

            # Save full cube
            np.save(os.path.join(out_cat_path, file.replace(".mat", ".npy")), cube)

            # If patch extraction is enabled
            if patch_size:
                patches = []
                h, w, b = cube.shape
                for i in range(0, h - patch_size + 1, stride):
                    for j in range(0, w - patch_size + 1, stride):
                        patch = cube[i:i+patch_size, j:j+patch_size, :]
                        patches.append(patch)
                patches = np.array(patches, dtype=np.float32)
                np.save(os.path.join(out_cat_path, file.replace(".mat", f"_patches.npy")), patches)
                print(f"Saved {patches.shape[0]} patches for {file}")
            else:
                print(f"Saved normalized cube for {file}")

if __name__ == "__main__":
    # Set patch_size=None if you want only normalized cubes
    # Example: patch_size=5 for 5x5xBands patches
    prepare_data(root_dir="Data", out_dir="Prepared", patch_size=5, stride=5)
