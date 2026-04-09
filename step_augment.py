# ================================================================
# step_augmentation.py
#
# PURPOSE:
#   Takes 200 beamformed ultrasound images and generates 10000+
#   augmented versions with their corresponding ROI masks.
#   This file must be run BEFORE step3b and step4b so both
#   CNN and UNet train on the larger augmented dataset.
#
# WHY AUGMENTATION WORKS:
#   A beamformed ultrasound image of a benign tumor is still a
#   benign tumor if you flip it, rotate it, or change brightness.
#   The label does not change. So each augmented version is a
#   valid new training example.
#
# INPUT:
#   outputs/beamformed_images/benign/     (plain view images)
#   outputs/beamformed_images/malignant/  (plain view images)
#   outputs/beamformed_images/benign/*_roi.png    (masks)
#   outputs/beamformed_images/malignant/*_roi.png (masks)
#
# OUTPUT:
#   outputs/augmented_images/benign/      (augmented images)
#   outputs/augmented_images/malignant/   (augmented images)
#   outputs/augmented_masks/benign/       (matching masks)
#   outputs/augmented_masks/malignant/    (matching masks)
#   outputs/augmented_images/augmentation_log.csv
#
# EXECUTION ORDER:
#   step1 → step2 → step3 → step4 → THIS FILE →
#   step3b → step4b → step5
#
# TIME TO RUN: approximately 5-15 minutes on CPU
# ================================================================

import numpy as np
from PIL import Image
from pathlib import Path
import cv2
import os
import csv
import random
from tqdm import tqdm   # progress bar
# install if missing: pip install tqdm

# Fix random seed for reproducibility
# Same seed = same augmentations every run
random.seed(42)
np.random.seed(42)


# ════════════════════════════════════════════════════════════════
# DIAGNOSTIC: Resolve correct project root and source directory
# ════════════════════════════════════════════════════════════════

def find_project_root():
    """
    Try to automatically find the project root by searching upward
    from the current working directory for the expected folder structure.
    Returns the Path where outputs/beamformed_images/ exists, or None.
    """
    cwd = Path.cwd()
    print(f"\nCurrent working directory : {cwd}")

    # Check cwd first
    candidate = cwd / 'outputs' / 'beamformed_images'
    if candidate.exists():
        print(f"Found beamformed_images at : {candidate}")
        return cwd

    # Search up to 3 levels up
    for parent in [cwd.parent, cwd.parent.parent, cwd.parent.parent.parent]:
        candidate = parent / 'outputs' / 'beamformed_images'
        if candidate.exists():
            print(f"Found beamformed_images at : {candidate}")
            print(f"Changing root to           : {parent}")
            os.chdir(parent)
            return parent

    # Could not find automatically
    return None


def diagnose_source_directory(src_base):
    """
    Print a detailed report of what is and isn't found in the
    expected source directory. Helps the user understand exactly
    what is missing and why.
    """
    print("\n" + "=" * 60)
    print("DIAGNOSTIC REPORT")
    print("=" * 60)
    print(f"Looking for source images in : {src_base.resolve()}")
    print(f"Directory exists             : {src_base.exists()}")

    if not src_base.exists():
        print("\n  ✗ The 'outputs/beamformed_images' folder does not exist.")
        print("  This means step1_beamforming.py has not been run yet,")
        print("  or you are running this script from the wrong folder.")
        print(f"\n  Expected folder : {src_base.resolve()}")
        print(f"  Your CWD        : {Path.cwd().resolve()}")
        return 0

    total_found = 0
    for category in ['benign', 'malignant']:
        folder = src_base / category
        print(f"\n  Category '{category}':")
        print(f"    Folder path   : {folder.resolve()}")
        print(f"    Folder exists : {folder.exists()}")

        if not folder.exists():
            print(f"    ✗ Folder missing — step1 may not have created it yet.")
            continue

        all_pngs       = list(folder.glob('*.png'))
        roi_pngs       = [f for f in all_pngs if '_roi'         in f.name]
        comparison_pngs= [f for f in all_pngs if '_comparison'  in f.name]
        plain_pngs     = [f for f in all_pngs
                          if '_roi'        not in f.name
                          and '_comparison' not in f.name]

        print(f"    Total PNGs       : {len(all_pngs)}")
        print(f"    Plain images     : {len(plain_pngs)}  ← used as source")
        print(f"    ROI masks        : {len(roi_pngs)}")
        print(f"    Comparison imgs  : {len(comparison_pngs)}")

        if len(plain_pngs) == 0:
            print(f"    ✗ No plain images found. Check step1 output.")
        else:
            print(f"    ✓ {len(plain_pngs)} source images ready.")
            # Show first 3 as examples
            for f in plain_pngs[:3]:
                print(f"      e.g. {f.name}")
            if len(plain_pngs) > 3:
                print(f"      ... and {len(plain_pngs) - 3} more")

        total_found += len(plain_pngs)

    print("\n" + "=" * 60)
    return total_found


# ── Output folders ────────────────────────────────────────────────
AUG_IMG_DIR  = Path('outputs/augmented_images')
AUG_MASK_DIR = Path('outputs/augmented_masks')

for category in ['benign', 'malignant']:
    (AUG_IMG_DIR  / category).mkdir(parents=True, exist_ok=True)
    (AUG_MASK_DIR / category).mkdir(parents=True, exist_ok=True)

# ── Target count ──────────────────────────────────────────────────
TARGET_TOTAL = 10000
# We have 200 originals → need 50 augmented versions per image
# to reach 10000. We will generate 55 to be safe (some may be
# duplicates or near-duplicates and will be skipped).
AUGMENTATIONS_PER_IMAGE = 55

# ════════════════════════════════════════════════════════════════
# AUGMENTATION FUNCTIONS
# Each function takes an image array AND mask array,
# applies IDENTICAL transformation to both, returns both.
# This is critical — image and mask must always match.
# ════════════════════════════════════════════════════════════════

def aug_horizontal_flip(img, mask):
    """
    Flip both image and mask left to right.
    A tumor on the left side of the image becomes a tumor
    on the right side — still valid medically.
    """
    return np.fliplr(img).copy(), np.fliplr(mask).copy()


def aug_vertical_flip(img, mask):
    """
    Flip both image and mask top to bottom.
    Simulates scanning from the opposite direction.
    """
    return np.flipud(img).copy(), np.flipud(mask).copy()


def aug_rotate_90(img, mask):
    """Rotate 90 degrees clockwise."""
    return np.rot90(img, k=1).copy(), np.rot90(mask, k=1).copy()


def aug_rotate_180(img, mask):
    """Rotate 180 degrees."""
    return np.rot90(img, k=2).copy(), np.rot90(mask, k=2).copy()


def aug_rotate_270(img, mask):
    """Rotate 270 degrees clockwise (same as 90 counter-clockwise)."""
    return np.rot90(img, k=3).copy(), np.rot90(mask, k=3).copy()


def aug_rotate_arbitrary(img, mask, angle=None):
    """
    Rotate by a random angle between -25 and +25 degrees.
    Uses OpenCV warpAffine which handles the edges properly.
    IMPORTANT: same angle applied to both image and mask.
    NEAREST interpolation for mask avoids creating gray border pixels.
    """
    if angle is None:
        angle = random.uniform(-25, 25)

    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)

    img_rot  = cv2.warpAffine(img, M, (w, h),
                               flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_REFLECT)

    mask_rot = cv2.warpAffine(
        (mask * 255).astype(np.uint8), M, (w, h),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )
    mask_rot = (mask_rot > 127).astype(np.float32)

    return img_rot, mask_rot


def aug_brightness(img, mask, factor=None):
    """
    Change image brightness by multiplying all pixels by a factor.
    Simulates different ultrasound machine gain settings.
    factor > 1.0 = brighter, factor < 1.0 = darker
    Mask is not changed — brightness does not affect tumor location.
    """
    if factor is None:
        factor = random.uniform(0.6, 1.5)

    img_bright = np.clip(img * factor, 0, 1).astype(np.float32)
    return img_bright, mask.copy()


def aug_contrast(img, mask, factor=None):
    """
    Change image contrast by stretching or compressing pixel values
    around the mean.
    factor > 1.0 = more contrast
    factor < 1.0 = less contrast
    """
    if factor is None:
        factor = random.uniform(0.6, 1.6)

    mean    = img.mean()
    img_con = np.clip(mean + factor * (img - mean), 0, 1)
    img_con = img_con.astype(np.float32)

    return img_con, mask.copy()


def aug_gamma(img, mask, gamma=None):
    """
    Gamma correction changes the nonlinear relationship between
    pixel value and actual brightness.
    gamma < 1.0 = brightens dark areas
    gamma > 1.0 = darkens mid-tones
    """
    if gamma is None:
        gamma = random.uniform(0.5, 2.0)

    img_gamma = np.power(np.clip(img, 1e-8, 1.0), gamma)
    img_gamma = img_gamma.astype(np.float32)

    return img_gamma, mask.copy()


def aug_gaussian_noise(img, mask, std=None):
    """
    Add Gaussian (random) noise to the image.
    Simulates electronic noise in the ADC hardware.
    Mask is unchanged — noise does not move the tumor.
    """
    if std is None:
        std = random.uniform(0.01, 0.06)

    noise     = np.random.normal(0, std, img.shape).astype(np.float32)
    img_noisy = np.clip(img + noise, 0, 1).astype(np.float32)

    return img_noisy, mask.copy()


def aug_blur(img, mask, ksize=None):
    """
    Gaussian blur — simulates slightly out-of-focus probe placement.
    ksize must be odd (3, 5, or 7).
    """
    if ksize is None:
        ksize = random.choice([3, 5])

    img_uint8   = (img * 255).astype(np.uint8)
    img_blurred = cv2.GaussianBlur(img_uint8, (ksize, ksize), 0)

    return img_blurred.astype(np.float32) / 255.0, mask.copy()


def aug_elastic_deformation(img, mask, alpha=None, sigma=None):
    """
    Elastic deformation warps the image as if the tissue was
    slightly squished or stretched.
    CRITICAL: exact same displacement field applied to both image and mask.
    """
    if alpha is None:
        alpha = random.uniform(20, 60)
    if sigma is None:
        sigma = random.uniform(4, 8)

    h, w = img.shape[:2]

    dx = np.random.uniform(-1, 1, (h, w)).astype(np.float32)
    dy = np.random.uniform(-1, 1, (h, w)).astype(np.float32)

    dx = cv2.GaussianBlur(dx, (0, 0), sigma) * alpha
    dy = cv2.GaussianBlur(dy, (0, 0), sigma) * alpha

    x_map, y_map = np.meshgrid(np.arange(w), np.arange(h))
    x_map = (x_map + dx).astype(np.float32)
    y_map = (y_map + dy).astype(np.float32)

    img_uint8 = (img * 255).astype(np.uint8)
    img_def   = cv2.remap(img_uint8, x_map, y_map,
                           interpolation=cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_REFLECT)

    mask_uint8 = (mask * 255).astype(np.uint8)
    mask_def   = cv2.remap(mask_uint8, x_map, y_map,
                            interpolation=cv2.INTER_NEAREST,
                            borderMode=cv2.BORDER_CONSTANT,
                            borderValue=0)

    img_def  = img_def.astype(np.float32) / 255.0
    mask_def = (mask_def > 127).astype(np.float32)

    return img_def, mask_def


def aug_crop_and_resize(img, mask, crop_pct=None):
    """
    Randomly crop a region and resize back to original size.
    Simulates the probe being positioned to show different
    amounts of tissue around the lesion.
    """
    if crop_pct is None:
        crop_pct = random.uniform(0.7, 0.95)

    h, w  = img.shape[:2]
    new_h = int(h * crop_pct)
    new_w = int(w * crop_pct)

    top  = random.randint(0, h - new_h)
    left = random.randint(0, w - new_w)

    img_cropped  = img [top:top+new_h, left:left+new_w]
    mask_cropped = mask[top:top+new_h, left:left+new_w]

    img_resized  = cv2.resize(
        (img_cropped * 255).astype(np.uint8), (w, h),
        interpolation=cv2.INTER_LINEAR
    ).astype(np.float32) / 255.0

    mask_resized = cv2.resize(
        (mask_cropped * 255).astype(np.uint8), (w, h),
        interpolation=cv2.INTER_NEAREST
    )
    mask_resized = (mask_resized > 127).astype(np.float32)

    return img_resized, mask_resized


def aug_cutout(img, mask, n_holes=None, hole_size=None):
    """
    Randomly blacks out small square patches in the image.
    Forces the CNN to not rely on any single region.
    Mask is unchanged — the holes do not represent missing tissue.
    """
    if n_holes is None:
        n_holes = random.randint(1, 4)
    if hole_size is None:
        hole_size = random.randint(20, 60)

    h, w    = img.shape[:2]
    img_cut = img.copy()

    for _ in range(n_holes):
        cy = random.randint(0, h - 1)
        cx = random.randint(0, w - 1)
        y1 = max(0, cy - hole_size // 2)
        y2 = min(h, cy + hole_size // 2)
        x1 = max(0, cx - hole_size // 2)
        x2 = min(w, cx + hole_size // 2)
        img_cut[y1:y2, x1:x2] = 0.0

    return img_cut, mask.copy()


def aug_shear(img, mask, shear_range=None):
    """
    Shear transformation — skews the image horizontally.
    Simulates the probe being held at a slight angle to the skin.
    """
    if shear_range is None:
        shear_range = random.uniform(-0.15, 0.15)

    h, w = img.shape[:2]
    M = np.float32([
        [1, shear_range, 0],
        [0, 1,           0]
    ])

    img_uint8   = (img * 255).astype(np.uint8)
    img_sheared = cv2.warpAffine(img_uint8, M, (w, h),
                                  flags=cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_REFLECT)

    mask_uint8   = (mask * 255).astype(np.uint8)
    mask_sheared = cv2.warpAffine(mask_uint8, M, (w, h),
                                   flags=cv2.INTER_NEAREST,
                                   borderMode=cv2.BORDER_CONSTANT,
                                   borderValue=0)

    img_sheared  = img_sheared.astype(np.float32) / 255.0
    mask_sheared = (mask_sheared > 127).astype(np.float32)

    return img_sheared, mask_sheared


def aug_combined(img, mask):
    """
    Applies 2-4 random augmentations in sequence.
    Creates the most diverse variations by combining multiple transforms.
    """
    spatial_augs = [
        aug_horizontal_flip,
        aug_vertical_flip,
        aug_rotate_90,
        aug_rotate_180,
        aug_rotate_270,
        lambda i, m: aug_rotate_arbitrary(i, m),
        lambda i, m: aug_crop_and_resize(i, m),
        lambda i, m: aug_shear(i, m),
        lambda i, m: aug_elastic_deformation(i, m),
    ]

    intensity_augs = [
        lambda i, m: aug_brightness(i, m),
        lambda i, m: aug_contrast(i, m),
        lambda i, m: aug_gamma(i, m),
        lambda i, m: aug_gaussian_noise(i, m),
        lambda i, m: aug_blur(i, m),
        lambda i, m: aug_cutout(i, m),
    ]

    n_spatial = random.randint(1, 2)
    chosen_spatial = random.sample(spatial_augs, n_spatial)
    for aug_fn in chosen_spatial:
        img, mask = aug_fn(img, mask)

    n_intensity = random.randint(1, 3)
    chosen_intensity = random.sample(intensity_augs, n_intensity)
    for aug_fn in chosen_intensity:
        img, mask = aug_fn(img, mask)

    return img, mask


# ════════════════════════════════════════════════════════════════
# MASK EXTRACTION FROM ROI OVERLAY
# ════════════════════════════════════════════════════════════════

def extract_mask_from_roi_image(roi_path, size):
    """
    Extracts binary lesion mask from the yellow-overlay ROI image
    saved by step1_beamforming.py.

    The ROI overlay has yellow pixels (high R, high G, low B) where
    the lesion is. We detect yellow pixels to get the binary mask.

    roi_path: path to the _roi.png file
    size    : (width, height) tuple to resize mask to
    Returns : float32 numpy array, values 0.0 or 1.0
    """
    roi_img = np.array(
        Image.open(roi_path).convert('RGB').resize(size, Image.NEAREST)
    )

    r = roi_img[:, :, 0].astype(np.int16)
    g = roi_img[:, :, 1].astype(np.int16)
    b = roi_img[:, :, 2].astype(np.int16)

    yellow_mask = ((r - b) > 30) & ((g - b) > 20)

    return yellow_mask.astype(np.float32)


# ════════════════════════════════════════════════════════════════
# MAIN AUGMENTATION PIPELINE
# ════════════════════════════════════════════════════════════════

def run_augmentation():

    print("=" * 60)
    print("AUGMENTATION PIPELINE")
    print(f"Target: {TARGET_TOTAL} total images")
    print("=" * 60)

    # ── Step 1: Find project root ─────────────────────────────────
    root = find_project_root()
    src_base = Path('outputs/beamformed_images')

    # ── Step 2: Full diagnostic before doing any work ─────────────
    total_found = diagnose_source_directory(src_base)

    if total_found == 0:
        print("\n" + "!" * 60)
        print("CANNOT PROCEED — no source images found.")
        print("!" * 60)
        print("\nPlease check the following:")
        print("  1. Have you run step1_beamforming.py?")
        print("  2. Are you running this script from the correct folder?")
        print(f"     Expected to be run from: <project_root>/")
        print(f"     Your current directory : {Path.cwd().resolve()}")
        print("\nTo fix: cd into your project root folder, then re-run.")
        print("  Example:")
        print("    cd /path/to/your/project")
        print("    python step_augmentation.py")
        return

    # ── Step 3: Collect all source image + mask pairs ─────────────
    source_pairs = []

    for category in ['benign', 'malignant']:
        folder = src_base / category

        img_files = sorted([
            f for f in folder.glob('*.png')
            if '_roi'         not in f.name
            and '_comparison' not in f.name
        ])

        for img_path in img_files:
            roi_name = img_path.stem + '_roi.png'
            roi_path = folder / roi_name

            if roi_path.exists():
                source_pairs.append((img_path, roi_path, category))
            else:
                # No ROI file — include image with empty mask
                source_pairs.append((img_path, None, category))

    print(f"\nSource pairs collected: {len(source_pairs)}")
    for cat in ['benign', 'malignant']:
        with_roi    = sum(1 for _, r, c in source_pairs if c == cat and r)
        without_roi = sum(1 for _, r, c in source_pairs if c == cat and not r)
        print(f"  {cat}: {with_roi + without_roi} images "
              f"({with_roi} with ROI mask, {without_roi} without)")

    # ── Step 4: Define all augmentation strategies ────────────────
    strategies = [
        ('hflip',           lambda i, m: aug_horizontal_flip(i, m)),
        ('vflip',           lambda i, m: aug_vertical_flip(i, m)),
        ('rot90',           lambda i, m: aug_rotate_90(i, m)),
        ('rot180',          lambda i, m: aug_rotate_180(i, m)),
        ('rot270',          lambda i, m: aug_rotate_270(i, m)),
        ('hflip_vflip',     lambda i, m: aug_vertical_flip(*aug_horizontal_flip(i, m))),
        ('rot90_hflip',     lambda i, m: aug_horizontal_flip(*aug_rotate_90(i, m))),
        ('rot90_vflip',     lambda i, m: aug_vertical_flip(*aug_rotate_90(i, m))),
        ('bright_up',       lambda i, m: aug_brightness(i, m, 1.3)),
        ('bright_down',     lambda i, m: aug_brightness(i, m, 0.7)),
        ('contrast_up',     lambda i, m: aug_contrast(i, m, 1.4)),
        ('contrast_down',   lambda i, m: aug_contrast(i, m, 0.7)),
        ('gamma_low',       lambda i, m: aug_gamma(i, m, 0.6)),
        ('gamma_high',      lambda i, m: aug_gamma(i, m, 1.8)),
        ('noise_low',       lambda i, m: aug_gaussian_noise(i, m, 0.02)),
        ('noise_high',      lambda i, m: aug_gaussian_noise(i, m, 0.05)),
        ('blur',            lambda i, m: aug_blur(i, m, 3)),
        ('blur_strong',     lambda i, m: aug_blur(i, m, 5)),
        ('elastic',         lambda i, m: aug_elastic_deformation(i, m)),
        ('crop',            lambda i, m: aug_crop_and_resize(i, m)),
        ('cutout',          lambda i, m: aug_cutout(i, m)),
        ('shear',           lambda i, m: aug_shear(i, m)),
        ('combined_1',      lambda i, m: aug_combined(i, m)),
        ('combined_2',      lambda i, m: aug_combined(i, m)),
        ('combined_3',      lambda i, m: aug_combined(i, m)),
        ('combined_4',      lambda i, m: aug_combined(i, m)),
        ('combined_5',      lambda i, m: aug_combined(i, m)),
        ('combined_6',      lambda i, m: aug_combined(i, m)),
        ('combined_7',      lambda i, m: aug_combined(i, m)),
        ('combined_8',      lambda i, m: aug_combined(i, m)),
        ('rot_r15',         lambda i, m: aug_rotate_arbitrary(i, m, 15)),
        ('rot_l15',         lambda i, m: aug_rotate_arbitrary(i, m, -15)),
        ('rot_r25',         lambda i, m: aug_rotate_arbitrary(i, m, 25)),
        ('rot_l25',         lambda i, m: aug_rotate_arbitrary(i, m, -25)),
        ('rot_r10',         lambda i, m: aug_rotate_arbitrary(i, m, 10)),
        ('rot_l10',         lambda i, m: aug_rotate_arbitrary(i, m, -10)),
        ('bright_noise',    lambda i, m: aug_gaussian_noise(*aug_brightness(i, m), 0.03)),
        ('flip_noise',      lambda i, m: aug_gaussian_noise(*aug_horizontal_flip(i, m), 0.02)),
        ('rot_bright',      lambda i, m: aug_brightness(*aug_rotate_arbitrary(i, m))),
        ('elastic_flip',    lambda i, m: aug_horizontal_flip(*aug_elastic_deformation(i, m))),
        ('crop_bright',     lambda i, m: aug_brightness(*aug_crop_and_resize(i, m))),
        ('shear_noise',     lambda i, m: aug_gaussian_noise(*aug_shear(i, m))),
        ('contrast_flip',   lambda i, m: aug_horizontal_flip(*aug_contrast(i, m))),
        ('gamma_rotate',    lambda i, m: aug_rotate_arbitrary(*aug_gamma(i, m))),
        ('cutout_flip',     lambda i, m: aug_horizontal_flip(*aug_cutout(i, m))),
        ('blur_contrast',   lambda i, m: aug_contrast(*aug_blur(i, m))),
        ('combined_rot',    lambda i, m: aug_rotate_arbitrary(*aug_combined(i, m))),
        ('combined_flip',   lambda i, m: aug_horizontal_flip(*aug_combined(i, m))),
        ('combined_bright', lambda i, m: aug_brightness(*aug_combined(i, m))),
        ('combined_noise',  lambda i, m: aug_gaussian_noise(*aug_combined(i, m))),
        ('combined_crop',   lambda i, m: aug_crop_and_resize(*aug_combined(i, m))),
        ('combined_elastic',lambda i, m: aug_elastic_deformation(*aug_combined(i, m))),
    ]

    print(f"\nAugmentation strategies : {len(strategies)}")
    print(f"Source images           : {len(source_pairs)}")
    print(f"Estimated total output  : "
          f"{len(source_pairs)} × ({len(strategies)} + 1 original) "
          f"= {len(source_pairs) * (len(strategies) + 1)} images")

    # ── Step 5: Log file setup ────────────────────────────────────
    log_path = AUG_IMG_DIR / 'augmentation_log.csv'
    log_rows = []
    saved_count = {'benign': 0, 'malignant': 0}
    skip_count  = 0

    # ── Step 6: Process each source image ─────────────────────────
    print(f"\nGenerating augmented images...")
    print("Progress bar shows one tick per source image.\n")

    for img_path, roi_path, category in tqdm(
        source_pairs, desc='Augmenting', unit='src'
    ):
        # Load source image as grayscale float32 0-1
        try:
            img_orig = np.array(
                Image.open(img_path).convert('L'),
                dtype=np.float32
            ) / 255.0
        except Exception as e:
            tqdm.write(f"  ERROR loading image {img_path.name}: {e} — skipping.")
            skip_count += 1
            continue

        h, w = img_orig.shape

        # Load or create mask
        if roi_path and roi_path.exists():
            try:
                mask_orig = extract_mask_from_roi_image(roi_path, (w, h))
            except Exception as e:
                tqdm.write(f"  WARNING: could not load ROI {roi_path.name}: {e}"
                           " — using empty mask.")
                mask_orig = np.zeros((h, w), dtype=np.float32)
        else:
            mask_orig = np.zeros((h, w), dtype=np.float32)

        # ── Save the ORIGINAL (no augmentation) ───────────────────
        orig_img_path  = AUG_IMG_DIR  / category / f"{img_path.stem}_orig.png"
        orig_mask_path = AUG_MASK_DIR / category / f"{img_path.stem}_orig_mask.png"

        Image.fromarray(
            (img_orig * 255).astype(np.uint8)
        ).save(str(orig_img_path))

        Image.fromarray(
            (mask_orig * 255).astype(np.uint8)
        ).save(str(orig_mask_path))

        saved_count[category] += 1
        log_rows.append({
            'filename' : orig_img_path.name,
            'mask'     : orig_mask_path.name,
            'category' : category,
            'strategy' : 'original',
            'source'   : img_path.name
        })

        # ── Apply each augmentation strategy ──────────────────────
        for strategy_name, strategy_fn in strategies:
            try:
                img_aug, mask_aug = strategy_fn(
                    img_orig.copy(), mask_orig.copy()
                )

                img_aug  = np.clip(img_aug,  0, 1).astype(np.float32)
                mask_aug = np.clip(mask_aug, 0, 1).astype(np.float32)

                aug_img_name  = f"{img_path.stem}_{strategy_name}.png"
                aug_mask_name = f"{img_path.stem}_{strategy_name}_mask.png"

                aug_img_path  = AUG_IMG_DIR  / category / aug_img_name
                aug_mask_path = AUG_MASK_DIR / category / aug_mask_name

                Image.fromarray(
                    (img_aug * 255).astype(np.uint8)
                ).save(str(aug_img_path))

                Image.fromarray(
                    (mask_aug * 255).astype(np.uint8)
                ).save(str(aug_mask_path))

                saved_count[category] += 1

                log_rows.append({
                    'filename' : aug_img_name,
                    'mask'     : aug_mask_name,
                    'category' : category,
                    'strategy' : strategy_name,
                    'source'   : img_path.name
                })

            except Exception as e:
                tqdm.write(f"  Skip {strategy_name} on {img_path.name}: {e}")
                skip_count += 1
                continue

    # ── Step 7: Save log CSV ──────────────────────────────────────
    with open(str(log_path), 'w', newline='') as f:
        writer = csv.DictWriter(
            f, fieldnames=['filename', 'mask', 'category', 'strategy', 'source']
        )
        writer.writeheader()
        writer.writerows(log_rows)

    # ── Step 8: Final summary ─────────────────────────────────────
    total_saved = sum(saved_count.values())

    print(f"\n{'='*60}")
    print("AUGMENTATION COMPLETE")
    print(f"{'='*60}")
    print(f"  Benign images saved    : {saved_count['benign']}")
    print(f"  Malignant images saved : {saved_count['malignant']}")
    print(f"  Total images saved     : {total_saved}")
    print(f"  Strategies skipped     : {skip_count}")
    print(f"  Log saved to           : {log_path}")
    print(f"\n  Image folder  : {AUG_IMG_DIR.resolve()}")
    print(f"  Mask folder   : {AUG_MASK_DIR.resolve()}")
    print(f"\nOutput folder structure:")
    print(f"  outputs/augmented_images/benign/     ← for CNN training")
    print(f"  outputs/augmented_images/malignant/  ← for CNN training")
    print(f"  outputs/augmented_masks/benign/      ← for UNet training")
    print(f"  outputs/augmented_masks/malignant/   ← for UNet training")

    if total_saved == 0:
        print("\n  WARNING: 0 images were saved.")
        print("  Re-run the diagnostic above to understand why.")
    elif total_saved < TARGET_TOTAL:
        print(f"\n  NOTE: {total_saved} images saved, target was {TARGET_TOTAL}.")
        print("  Consider adding more source images to reach the target.")
    else:
        print(f"\n  ✓ Target of {TARGET_TOTAL} images met or exceeded.")

    print(f"\nNext steps:")
    print(f"  1. python step3b_finetune_cnn.py")
    print(f"  2. python step4b_finetune_unet.py")


# ── Run ───────────────────────────────────────────────────────────
if __name__ == '__main__':
    run_augmentation()