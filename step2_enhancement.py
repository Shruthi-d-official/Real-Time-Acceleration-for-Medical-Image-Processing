import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path

# ── This function takes one ultrasound image and applies 3 enhancements ───────
def enhance_ultrasound(image_path, save_path):

    # Read the image in grayscale
    # cv2.IMREAD_GRAYSCALE loads as single channel (0-255 values)
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)

    if img is None:
        print(f"Could not load: {image_path}")
        return

    # ── Enhancement 1: Speckle noise reduction using median filter ────────────
    # Ultrasound images have "speckle" — random grainy noise
    # A median filter replaces each pixel with the median of its neighbors
    # ksize=5 means we look at a 5x5 neighborhood around each pixel
    # This removes salt-and-pepper noise while keeping edges sharp
    denoised = cv2.medianBlur(img, ksize=5)

    # ── Enhancement 2: Contrast Limited Adaptive Histogram Equalization ───────
    # Regular histogram equalization brightens the whole image uniformly
    # CLAHE works on small tiles separately, so local contrast is improved
    # clipLimit=2.0 prevents over-amplifying noise in flat regions
    # tileGridSize=(8,8) divides the image into 8x8 tiles
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    contrast_enhanced = clahe.apply(denoised)

    # ── Enhancement 3: Bilateral filter (edge-preserving smoothing) ──────────
    # Regular blur smooths everything including edges (makes image blurry)
    # Bilateral filter smooths ONLY flat regions, preserving sharp edges
    # d=9: diameter of pixel neighborhood
    # sigmaColor=75: how much color difference to consider "similar"
    # sigmaSpace=75: how far spatially to look for similar pixels
    bilateral = cv2.bilateralFilter(contrast_enhanced, d=9,
                                    sigmaColor=75, sigmaSpace=75)

    # Save the enhanced image
    cv2.imwrite(str(save_path), bilateral)
    return img, denoised, contrast_enhanced, bilateral

# ── Process all images in the BUSI dataset ───────────────────────────────────
input_base  = Path('Dataset/BUSI')
output_base = Path('outputs/enhanced_images')

# Walk through benign, malignant, normal folders
for category in ['benign', 'malignant', 'normal']:
    input_folder  = input_base / category
    output_folder = output_base / category
    output_folder.mkdir(parents=True, exist_ok=True)

    # Get all image files (not the mask files — masks have "_mask" in name)
    image_files = [f for f in input_folder.glob('*.png')
                   if '_mask' not in f.name]

    print(f"Processing {len(image_files)} images in {category}...")

    for img_path in image_files:
        save_path = output_folder / img_path.name
        result = enhance_ultrasound(img_path, save_path)

    print(f"  Done. Saved to {output_folder}")

# ── Show a before/after comparison for one image ─────────────────────────────
sample_path = next((input_base / 'benign').glob('*.png'))
orig, denoised, contrast, final = enhance_ultrasound(sample_path, 'temp.png')

fig, axes = plt.subplots(1, 4, figsize=(16, 4))
axes[0].imshow(orig, cmap='gray');       axes[0].set_title('Original')
axes[1].imshow(denoised, cmap='gray');   axes[1].set_title('After median filter')
axes[2].imshow(contrast, cmap='gray');   axes[2].set_title('After CLAHE')
axes[3].imshow(final, cmap='gray');      axes[3].set_title('After bilateral filter')
for ax in axes: ax.axis('off')
plt.tight_layout()
plt.savefig('outputs/enhancement_comparison.png', dpi=150)
plt.show()
print("Comparison saved to outputs/enhancement_comparison.png")