import streamlit as st
import os
from PIL import Image
import matplotlib.pyplot as plt

st.set_page_config(page_title="Hyperspectral Anomaly Viewer", layout="wide")

st.title("🌌 Hyperspectral Anomaly Detection Viewer")

# --- Dataset selector ---
dataset = st.selectbox("Choose Dataset", ["Airport", "Beach", "Urban"])

# Directories (adjust to your folder structure)
orig_dir = f"results/{dataset}/originals"
high_dir = f"results/{dataset}/highlighted"

# Safety check
if not os.path.exists(orig_dir) or not os.path.exists(high_dir):
    st.error(f"Image folders for {dataset} not found! Check: {orig_dir}, {high_dir}")
    st.stop()

# List images
orig_images = sorted([f for f in os.listdir(orig_dir) if f.lower().endswith((".png", ".jpg"))])
high_images = sorted([f for f in os.listdir(high_dir) if f.lower().endswith((".png", ".jpg"))])

if not orig_images or not high_images:
    st.error("No images found in selected dataset.")
    st.stop()

# --- Match files by base name (without extension) ---
orig_map = {os.path.splitext(f)[0]: f for f in orig_images}
high_map = {os.path.splitext(f)[0].replace("_highlighted", ""): f for f in high_images}

# Only keep keys that exist in both
common_keys = sorted(list(set(orig_map.keys()) & set(high_map.keys())))
if not common_keys:
    st.error("No matching image pairs found between originals and highlighted.")
    st.stop()

# --- Image slider ---
index = st.slider("Image Index", 0, len(common_keys) - 1, 0)
key = common_keys[index]
orig_path = os.path.join(orig_dir, orig_map[key])
high_path = os.path.join(high_dir, high_map[key])

# --- Display side by side ---
col1, col2 = st.columns(2)
with col1:
    st.subheader("Original Image")
    st.image(Image.open(orig_path), use_container_width=True)

with col2:
    st.subheader("Highlighted Anomalies")
    st.image(Image.open(high_path), use_container_width=True)

# --- ROC Curves Section ---
st.markdown("---")
st.subheader("📊 Model ROC Curves")

# Directories for ROC curves
roc_dirs = [
    os.path.join("Visualization", "ROC Curves"),
    os.path.join("ROC Curves")
]

roc_images = []
for roc_dir in roc_dirs:
    if os.path.exists(roc_dir):
        roc_images += [os.path.join(roc_dir, f) for f in os.listdir(roc_dir) if f.lower().endswith((".png", ".jpg"))]

if not roc_images:
    st.info("No ROC curve images found in Visualization/ROC Curves or ROC Curves directories.")
else:
    n = len(roc_images)
    ncols = 2 if n > 1 else 1
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows))
    if n == 1:
        axes = [axes]
    axes = axes.flatten() if n > 1 else axes
    for ax, img_path in zip(axes, roc_images):
        img = Image.open(img_path)
        ax.imshow(img)
        ax.set_title(os.path.basename(img_path))
        ax.axis('off')
    # Hide any unused subplots
    for ax in axes[len(roc_images):]:
        ax.axis('off')
    st.pyplot(fig)
