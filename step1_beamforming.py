# ============================================================
# step1_beamforming.py
# Converts OASBUD.mat raw data into PNG ultrasound images
# Input  : data/OASBUD.mat  (100 patient records)
# Output : outputs/beamformed_images/benign/  and  /malignant/
# ============================================================

import scipy.io          # loads .mat files into Python
import numpy as np       # numerical operations on arrays
import matplotlib        
matplotlib.use('Agg')    # use non-interactive backend (no popup windows)
                         # 'Agg' renders to file only — safe for all systems
import matplotlib.pyplot as plt
from scipy.signal import hilbert  # for envelope detection
from pathlib import Path          # clean cross-platform file paths
import os                         # folder creation

# ── STEP 1A: Load the .mat file ──────────────────────────────────────────────
print("Loading OASBUD.mat ...")
mat  = scipy.io.loadmat('Dataset/OASBUD.mat')

# mat is a Python dictionary. The key 'data' holds all 100 patient records
# as a structured numpy array of shape (1, 100)
data = mat['data']

print(f"Total patients in dataset: {data.shape[1]}")
# Output: Total patients in dataset: 100


# ── STEP 1B: Create output folders ───────────────────────────────────────────
# We sort images into benign/malignant subfolders so Step 2 (enhancement)
# and Step 3 (CNN training) can directly use this output as their input

out_benign    = Path('outputs/beamformed_images/benign')
out_malignant = Path('outputs/beamformed_images/malignant')
out_benign.mkdir(parents=True, exist_ok=True)
out_malignant.mkdir(parents=True, exist_ok=True)
# parents=True  → creates all intermediate folders if they don't exist
# exist_ok=True → doesn't crash if folder already exists


# ── STEP 1C: Define the image reconstruction function ────────────────────────
def reconstruct_image(rf_raw):
    """
    Converts a raw RF/B-mode integer matrix into a displayable ultrasound image.

    Input:
        rf_raw : numpy array of shape (1824, 510), dtype int16
                 Values range from -32512 to +32511

    Output:
        img_norm : numpy array of shape (1824, 510), dtype float32
                   Values range from 0.0 to 1.0
                   Ready to save as a grayscale PNG image
    """

    # Step 1: Convert int16 → float64
    # We must do this before any math because int16 arithmetic overflows
    # Example: 32511 + 1 would wrap around to -32768 in int16
    rf = rf_raw.astype(np.float64)

    # Step 2: Hilbert transform along the time axis (axis=0)
    # The raw signal oscillates positive and negative (like a sine wave)
    # The Hilbert transform gives us the "envelope" — the outline of the wave
    # Think of it like drawing the upper boundary of a wave instead of the wave itself
    # axis=0 means we apply it along each column (each transducer element separately)
    analytic = hilbert(rf, axis=0)
    # analytic is now a complex array — real part is original, imaginary is Hilbert
    
    # Taking absolute value gives the envelope (amplitude at each point)
    envelope = np.abs(analytic)
    # envelope values are all positive now, representing signal strength at each point

    # Step 3: Log compression
    # Ultrasound signals have a huge dynamic range — strong reflections from
    # bone might be 1000x stronger than subtle tissue reflections
    # Our eyes can't see that range in a single image
    # Log compression squashes the range to something visible
    # Adding 1e-10 prevents log(0) which would give -infinity
    log_compressed = 20 * np.log10(envelope + 1e-10)
    # Now values are in decibels (dB)
    # Strong echoes: ~80-90 dB, weak echoes: ~20-30 dB

    # Step 4: Dynamic range clipping to 50 dB
    # We keep only the top 50 dB of signal
    # Anything more than 50 dB below the maximum is set to the minimum
    # This removes very weak noise signals at the bottom
    max_val = log_compressed.max()
    log_clipped = np.clip(log_compressed, max_val - 50, max_val)
    # Now all values are within a 50 dB window

    # Step 5: Normalize to 0.0 → 1.0 range
    # Matplotlib and image saving need values in 0-1 (float) or 0-255 (int)
    min_val = log_clipped.min()
    max_val = log_clipped.max()
    img_norm = (log_clipped - min_val) / (max_val - min_val + 1e-10)
    # Now 0.0 = darkest (no echo), 1.0 = brightest (strong echo)

    return img_norm.astype(np.float32)


# ── STEP 1D: Define the ROI overlay function ──────────────────────────────────
def apply_roi_overlay(img_norm, roi):
    """
    Creates a 3-channel (RGB) image with the ROI region highlighted in color.

    Input:
        img_norm : float32 array (H, W), values 0-1  — the reconstructed image
        roi      : int array (H, W), values 0 or 1   — lesion region mask

    Output:
        rgb : uint8 array (H, W, 3) — image with yellow ROI border overlay
    """

    # Convert grayscale 0-1 float to 0-255 uint8 for all 3 channels (R, G, B)
    gray_uint8 = (img_norm * 255).astype(np.uint8)
    
    # Stack the same grayscale image 3 times to make an RGB image
    # axis=-1 stacks along the last dimension: (H,W) → (H,W,3)
    rgb = np.stack([gray_uint8, gray_uint8, gray_uint8], axis=-1)

    # If ROI has any non-zero values, draw it as a yellow overlay
    if roi is not None and roi.max() > 0:
        # Normalize ROI to binary (0 or 1) in case it has other values
        roi_binary = (roi > 0)
        
        # Yellow = high Red + high Green + zero Blue
        # We set ROI pixels to semi-transparent yellow by boosting R and G channels
        rgb[roi_binary, 0] = np.clip(rgb[roi_binary, 0].astype(int) + 80, 0, 255)  # R up
        rgb[roi_binary, 1] = np.clip(rgb[roi_binary, 1].astype(int) + 80, 0, 255)  # G up
        rgb[roi_binary, 2] = np.clip(rgb[roi_binary, 2].astype(int) - 40, 0, 255)  # B down

    return rgb


# ── STEP 1E: Process all 100 patients ────────────────────────────────────────
print("\nProcessing all 100 patients...")
print("-" * 50)

# Counters to track how many of each class we process
count_benign    = 0
count_malignant = 0
count_errors    = 0

# Keep a summary log
summary = []

for i in range(data.shape[1]):
    # ── Extract this patient's record ────────────────────────────────────────
    patient = data[0, i]
    # data[0, i] gets the i-th patient from the (1, 100) structured array

    try:
        # ── Get patient ID ────────────────────────────────────────────────────
        # patient['id'] is stored as a nested object array, flatten to get the string
        pat_id = str(patient['id'].flat[0]).strip()
        # .flat[0] gets the first element regardless of nesting depth
        # .strip() removes any whitespace

        # ── Get class label ───────────────────────────────────────────────────
        # class=0 means benign, class=1 means malignant
        # Stored as a nested array, so we use flat[0] to get the integer
        label_val = int(patient['class'].flat[0])
        label_str = 'benign' if label_val == 0 else 'malignant'

        # ── Get BI-RADS score ─────────────────────────────────────────────────
        birads = str(patient['birads'].flat[0]).strip()

        # ── Extract RF signals ────────────────────────────────────────────────
        # rf1 and rf2 may be wrapped in extra array layers
        # We use squeeze() to remove all size-1 dimensions
        rf1 = np.array(patient['rf1']).squeeze()
        rf2 = np.array(patient['rf2']).squeeze()
        # After squeeze, shape should be (1824, 510)

        # ── Extract ROI masks ─────────────────────────────────────────────────
        roi1 = np.array(patient['roi1']).squeeze()
        roi2 = np.array(patient['roi2']).squeeze()

        # ── Reconstruct images from raw signals ───────────────────────────────
        img1 = reconstruct_image(rf1)   # view 1 reconstructed image
        img2 = reconstruct_image(rf2)   # view 2 reconstructed image

        # ── Create ROI overlays ───────────────────────────────────────────────
        overlay1 = apply_roi_overlay(img1, roi1)
        overlay2 = apply_roi_overlay(img2, roi2)

        # ── Decide output folder based on class label ─────────────────────────
        out_folder = out_benign if label_val == 0 else out_malignant

        # ── Save view 1: plain grayscale image ────────────────────────────────
        # This is what Step 2 (enhancement) and Step 3 (CNN) will use
        save_path_v1 = out_folder / f'patient_{i+1:03d}_view1.png'
        # f'patient_{i+1:03d}' formats the number with leading zeros: 001, 002, ...
        plt.imsave(str(save_path_v1), img1, cmap='gray', vmin=0, vmax=1)
        # plt.imsave saves a float array as a PNG image
        # cmap='gray' → grayscale colormap
        # vmin=0, vmax=1 → map 0→black, 1→white

        # ── Save view 2: plain grayscale image ────────────────────────────────
        save_path_v2 = out_folder / f'patient_{i+1:03d}_view2.png'
        plt.imsave(str(save_path_v2), img2, cmap='gray', vmin=0, vmax=1)

        # ── Save view 1 with ROI overlay ──────────────────────────────────────
        # This is for visual inspection — doctors can see where the lesion is
        save_path_roi1 = out_folder / f'patient_{i+1:03d}_view1_roi.png'
        plt.imsave(str(save_path_roi1), overlay1)
        # No cmap needed because overlay1 is already RGB (3-channel)

        # ── Save view 2 with ROI overlay ──────────────────────────────────────
        save_path_roi2 = out_folder / f'patient_{i+1:03d}_view2_roi.png'
        plt.imsave(str(save_path_roi2), overlay2)

        # ── Save a comparison figure (original + ROI side by side) ────────────
        # Only save comparison for every 10th patient to save time
        if i % 10 == 0:
            fig, axes = plt.subplots(1, 4, figsize=(18, 5))
            # figsize=(18,5) makes a wide figure with 4 panels

            axes[0].imshow(img1, cmap='gray', vmin=0, vmax=1, aspect='auto')
            axes[0].set_title(f'View 1 — raw\nPatient {i+1} | {label_str} | BIRADS {birads}')
            axes[0].axis('off')
            # axis('off') hides the tick marks and axis labels

            axes[1].imshow(overlay1, aspect='auto')
            axes[1].set_title('View 1 — with ROI')
            axes[1].axis('off')

            axes[2].imshow(img2, cmap='gray', vmin=0, vmax=1, aspect='auto')
            axes[2].set_title('View 2 — raw')
            axes[2].axis('off')

            axes[3].imshow(overlay2, aspect='auto')
            axes[3].set_title('View 2 — with ROI')
            axes[3].axis('off')

            plt.suptitle(
                f'Patient {i+1} | ID: {pat_id} | Class: {label_str.upper()} | BIRADS: {birads}',
                fontsize=13, fontweight='bold'
            )
            plt.tight_layout()

            comp_path = out_folder / f'patient_{i+1:03d}_comparison.png'
            plt.savefig(str(comp_path), dpi=120, bbox_inches='tight')
            plt.close()
            # plt.close() is CRITICAL — without it, matplotlib keeps all figures
            # in memory and you will run out of RAM after ~20 patients

        # ── Update counters ───────────────────────────────────────────────────
        if label_val == 0:
            count_benign += 1
        else:
            count_malignant += 1

        # ── Log this patient's summary ────────────────────────────────────────
        summary.append({
            'patient_idx' : i + 1,
            'id'          : pat_id,
            'label'       : label_str,
            'birads'      : birads,
            'rf1_shape'   : rf1.shape,
            'img1_min'    : float(img1.min()),
            'img1_max'    : float(img1.max()),
            'roi1_nonzero': int((roi1 > 0).sum())
            # roi1_nonzero tells us how many pixels are inside the lesion ROI
        })

        print(f"  [{i+1:3d}/100] Patient {pat_id:6s} | {label_str:9s} | "
              f"BIRADS {birads} | ROI pixels: {(roi1>0).sum():6d}")

    except Exception as e:
        # If any patient fails, log the error and continue with the rest
        print(f"  [{i+1:3d}/100] ERROR: {e}")
        count_errors += 1
        continue


# ── STEP 1F: Print final summary ─────────────────────────────────────────────
print("\n" + "=" * 50)
print("BEAMFORMING COMPLETE")
print("=" * 50)
print(f"  Benign patients    : {count_benign}")
print(f"  Malignant patients : {count_malignant}")
print(f"  Errors             : {count_errors}")
print(f"  Total images saved : {(count_benign + count_malignant) * 4}")
# x4 because we save: view1, view2, view1_roi, view2_roi for each patient
print(f"\n  Output folders:")
print(f"    {out_benign}")
print(f"    {out_malignant}")


# ── STEP 1G: Save a dataset summary CSV ──────────────────────────────────────
# This CSV will be used later by the CNN training script to know
# which images belong to which class
import csv

csv_path = Path('outputs/beamformed_images/dataset_summary.csv')
with open(csv_path, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=summary[0].keys())
    writer.writeheader()
    writer.writerows(summary)

print(f"\n  Summary CSV saved: {csv_path}")
print("\nStep 1 complete. Run step2_enhancement.py next.")
