# ================================================================
# step5_predict.py — FINAL COMPLETE VERSION (with --signal support)
#
# INPUT : Raw RF signal from Dataset/OASBUD.mat
#         OR synthetic .mat signal files
#         OR PNG image files
# OUTPUT: 6-panel image saved to outputs/results/
#         showing raw signal, beamformed image, enhanced image,
#         tumor heatmap, tumor mask, and final diagnosis
#
# USAGE:
#   python step5_predict.py --patient 0    (single patient from OASBUD)
#   python step5_predict.py --patient 15
#   python step5_predict.py                (runs 5 patients)
#   python step5_predict.py --image Dataset/BUSI/benign/benign(1).png
#   python step5_predict.py --validate     (runs ALL patients, prints accuracy)
#   python step5_predict.py --signal synthetic_signals/signal_01_benign.mat
#   python step5_predict.py --signal all   (runs ALL synthetic signals)
# ================================================================

import torch
import torch.nn as nn
import scipy.io
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import hilbert
from torchvision import transforms, models
from pathlib import Path
from PIL import Image
from skimage.exposure import match_histograms
import argparse
import os
import sys
import time

device = torch.device('cpu')


# ════════════════════════════════════════════════════════════════
# UNET — matches saved weights exactly (.conv naming)
# ════════════════════════════════════════════════════════════════

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.conv(x)


class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1,
                 features=[64,128,256,512]):
        super().__init__()
        self.enc1       = ConvBlock(in_channels, features[0])
        self.enc2       = ConvBlock(features[0], features[1])
        self.enc3       = ConvBlock(features[1], features[2])
        self.enc4       = ConvBlock(features[2], features[3])
        self.pool       = nn.MaxPool2d(2,2)
        self.bottleneck = ConvBlock(features[3], features[3]*2)
        self.up4  = nn.ConvTranspose2d(features[3]*2, features[3], 2,2)
        self.dec4 = ConvBlock(features[3]*2, features[3])
        self.up3  = nn.ConvTranspose2d(features[3], features[2], 2,2)
        self.dec3 = ConvBlock(features[2]*2, features[2])
        self.up2  = nn.ConvTranspose2d(features[2], features[1], 2,2)
        self.dec2 = ConvBlock(features[1]*2, features[1])
        self.up1  = nn.ConvTranspose2d(features[1], features[0], 2,2)
        self.dec1 = ConvBlock(features[0]*2, features[0])
        self.final = nn.Conv2d(features[0], out_channels, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b  = self.bottleneck(self.pool(e4))
        d4 = self.dec4(torch.cat([self.up4(b),  e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.final(d1)


# ════════════════════════════════════════════════════════════════
# AUTO-DETECT DATA PATHS
# ════════════════════════════════════════════════════════════════

def find_path(*candidates):
    for c in candidates:
        if Path(c).exists():
            return Path(c)
    return None

OASBUD_PATH = find_path('Dataset/OASBUD.mat', 'data/OASBUD.mat', 'OASBUD.mat')
BREAST_PATH = find_path('Dataset/BrEaST-Lesions_USG-images_and_masks',
                        'data/BrEaST-Lesions_USG-images_and_masks')
BUSI_PATH   = find_path('Dataset/BUSI', 'data/BUSI', 'BUSI')

print(f"OASBUD path  : {OASBUD_PATH}")
print(f"BrEaST path  : {BREAST_PATH}")
print(f"BUSI path    : {BUSI_PATH}")


# ════════════════════════════════════════════════════════════════
# HISTOGRAM MATCHING REFERENCE
# ════════════════════════════════════════════════════════════════

def build_reference():
    collected = []
    if BREAST_PATH:
        imgs = [f for f in BREAST_PATH.glob('*.png')
                if '_tumor' not in f.name and '_mask' not in f.name]
        for p in imgs[:40]:
            arr = np.array(Image.open(p).convert('L').resize((256,256)),
                           dtype=np.float32) / 255.0
            collected.append(arr)
    if BUSI_PATH:
        for cat in ['benign','malignant','normal']:
            folder = BUSI_PATH / cat
            if not folder.exists(): continue
            imgs = [f for f in folder.glob('*.png') if '_mask' not in f.name]
            for p in imgs[:15]:
                arr = np.array(Image.open(p).convert('L').resize((256,256)),
                               dtype=np.float32) / 255.0
                collected.append(arr)
    if not collected:
        print("  WARNING: No reference images found — skipping histogram match.")
        return None
    reference = np.mean(np.stack(collected), axis=0)
    print(f"  Reference built from {len(collected)} clinical images")
    return reference.astype(np.float32)

print("\nBuilding histogram matching reference...")
REFERENCE = build_reference()


# ════════════════════════════════════════════════════════════════
# LOAD MODELS
# FIX: best_cnn_finetuned.pth has 2 classes (benign/malignant)
#      best_cnn.pth           has 3 classes (benign/malignant/normal)
#      We detect which is loaded and set classes accordingly.
# ════════════════════════════════════════════════════════════════

def load_models():
    global CNN_CLASSES

    ft_cnn = 'outputs/model_checkpoints/best_cnn_finetuned.pth'
    or_cnn = 'outputs/model_checkpoints/best_cnn.pth'

    if os.path.exists(ft_cnn):
        CNN_CLASSES = ['benign', 'malignant']
        cnn = models.resnet18(weights=None)
        cnn.fc = nn.Linear(cnn.fc.in_features, 2)
        cnn.load_state_dict(torch.load(ft_cnn, map_location='cpu'))
        print("  CNN : best_cnn_finetuned.pth  (2 classes: benign/malignant)")

    elif os.path.exists(or_cnn):
        CNN_CLASSES = ['benign', 'malignant', 'normal']
        cnn = models.resnet18(weights=None)
        cnn.fc = nn.Linear(cnn.fc.in_features, 3)
        cnn.load_state_dict(torch.load(or_cnn, map_location='cpu'))
        print("  CNN : best_cnn.pth  (3 classes: benign/malignant/normal)")

    else:
        print("  ERROR: No CNN model found in outputs/model_checkpoints/")
        sys.exit(1)

    cnn.eval()

    # ── UNet ──────────────────────────────────────────────────────
    unet    = UNet(in_channels=1, out_channels=1)
    ft_unet = 'outputs/model_checkpoints/best_unet_finetuned.pth'
    or_unet = 'outputs/model_checkpoints/best_unet.pth'

    if os.path.exists(ft_unet):
        unet.load_state_dict(torch.load(ft_unet, map_location='cpu'))
        print("  UNet: best_unet_finetuned.pth")
    elif os.path.exists(or_unet):
        unet.load_state_dict(torch.load(or_unet, map_location='cpu'))
        print("  UNet: best_unet.pth")
    else:
        print("  WARNING: No UNet model found — segmentation disabled.")
        unet = None

    if unet: unet.eval()
    return cnn, unet

# Will be set correctly inside load_models()
CNN_CLASSES = ['benign', 'malignant']


# ════════════════════════════════════════════════════════════════
# PIPELINE FUNCTIONS
# ════════════════════════════════════════════════════════════════

def reconstruct_image(rf_raw):
    """Raw RF signal → beamformed grayscale image."""
    rf         = rf_raw.astype(np.float64)
    analytic   = hilbert(rf, axis=0)
    envelope   = np.abs(analytic)
    log_comp   = 20 * np.log10(envelope + 1e-10)
    max_val    = log_comp.max()
    clipped    = np.clip(log_comp, max_val - 50, max_val)
    normalized = (clipped - clipped.min()) / \
                 (clipped.max() - clipped.min() + 1e-10)
    return normalized.astype(np.float32)


def enhance_image(img):
    """Denoise + CLAHE contrast + bilateral filter."""
    u = (img * 255).astype(np.uint8)
    u = cv2.medianBlur(u, 5)
    u = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(u)
    u = cv2.bilateralFilter(u, 9, 75, 75)
    return u.astype(np.float32) / 255.0


def histogram_match(img):
    """Match brightness distribution to clinical images."""
    if REFERENCE is None:
        return img
    img_u8  = (img       * 255).astype(np.uint8)
    ref_u8  = (REFERENCE * 255).astype(np.uint8)
    matched = match_histograms(img_u8, ref_u8)
    return matched.astype(np.float32) / 255.0


def run_cnn(cnn_model, enhanced_img):
    """CNN classification → class name, confidence, all probabilities."""
    pil = Image.fromarray((enhanced_img * 255).astype(np.uint8)).convert('RGB')
    tfm = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
    ])
    t = tfm(pil).unsqueeze(0)
    with torch.no_grad():
        probs = torch.softmax(cnn_model(t), dim=1)[0]
    idx        = probs.argmax().item()
    pred_class = CNN_CLASSES[idx]
    confidence = probs[idx].item() * 100
    prob_dict  = {CNN_CLASSES[i]: probs[i].item()*100
                  for i in range(len(CNN_CLASSES))}
    return pred_class, confidence, prob_dict


def run_unet(unet_model, matched_img):
    """UNet segmentation → binary mask + probability map."""
    img_256 = cv2.resize(matched_img, (256,256))
    t       = torch.tensor(img_256).unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        out      = unet_model(t)
        prob_map = torch.sigmoid(out)[0,0].numpy()
    binary_mask = (prob_map > 0.3).astype(np.uint8)
    return binary_mask, prob_map


def create_overlay(img, mask):
    """Draw red tumor region on grayscale image."""
    u   = (img * 255).astype(np.uint8)
    rgb = np.stack([u, u, u], axis=-1)
    if mask.shape != img.shape:
        mask = cv2.resize(mask.astype(np.uint8),
                          (img.shape[1], img.shape[0]),
                          interpolation=cv2.INTER_NEAREST)
    rgb[mask == 1, 0] = 255
    rgb[mask == 1, 1] = 0
    rgb[mask == 1, 2] = 0
    return rgb


def save_result(patient_id, rf_signal, beamformed, enhanced, matched,
                pred_class, confidence, prob_dict,
                binary_mask, prob_map, overlay, tumor_pct, latency_ms):
    """Save 6-panel result image."""
    os.makedirs('outputs/results', exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes      = axes.flatten()

    axes[0].imshow(rf_signal[:200, :100], cmap='seismic',
                   aspect='auto', vmin=-32000, vmax=32000)
    axes[0].set_title('Panel 1: Raw RF signal\n(input from .mat file)')
    axes[0].set_xlabel('Transducer element')
    axes[0].set_ylabel('Time sample')

    axes[1].imshow(beamformed, cmap='gray', vmin=0, vmax=1, aspect='auto')
    axes[1].set_title('Panel 2: Beamformed image\n(signal → picture)')
    axes[1].axis('off')

    axes[2].imshow(matched, cmap='gray', vmin=0, vmax=1, aspect='auto')
    axes[2].set_title('Panel 3: Enhanced + matched\n(normalized for UNet)')
    axes[2].axis('off')

    axes[3].imshow(prob_map, cmap='hot', vmin=0, vmax=1, aspect='auto')
    axes[3].set_title('Panel 4: Tumor probability map\n(bright = tumor)')
    axes[3].axis('off')

    axes[4].imshow(binary_mask, cmap='gray', vmin=0, vmax=1, aspect='auto')
    axes[4].set_title(f'Panel 5: Tumor mask\n({tumor_pct:.1f}% of image area)')
    axes[4].axis('off')

    color = {'benign':'green', 'malignant':'red', 'normal':'blue'}
    axes[5].imshow(overlay, aspect='auto')
    axes[5].set_title(
        f'Panel 6: DIAGNOSIS\n{pred_class.upper()}  ({confidence:.1f}%)',
        color=color.get(pred_class,'black'), fontweight='bold', fontsize=13
    )
    axes[5].axis('off')

    # Probability bar chart
    ax_bar = fig.add_axes([0.67, 0.04, 0.28, 0.10])
    colors = ['#2ecc71','#e74c3c','#3498db']
    ax_bar.barh(list(prob_dict.keys()),
                [v/100 for v in prob_dict.values()],
                color=colors[:len(prob_dict)])
    ax_bar.set_xlim(0,1)
    ax_bar.set_xlabel('Probability', fontsize=8)
    ax_bar.tick_params(labelsize=8)

    plt.suptitle(
        f'AI Pipeline Result — {patient_id} | Latency: {latency_ms:.0f}ms',
        fontsize=14, fontweight='bold'
    )
    plt.tight_layout(rect=[0, 0.12, 1, 1])
    out = f'outputs/results/prediction_{patient_id}.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Saved: {out}")
    return out


# ════════════════════════════════════════════════════════════════
# PREDICT FROM OASBUD PATIENT INDEX
# ════════════════════════════════════════════════════════════════

def predict_from_signal(patient_idx, cnn_model=None, unet_model=None):
    """Full pipeline: raw RF signal from OASBUD.mat → diagnosis + segmentation."""

    print(f"\n{'='*60}")
    print(f"PREDICTING: Patient {patient_idx}")
    print(f"{'='*60}")

    if OASBUD_PATH is None:
        print("ERROR: OASBUD.mat not found.")
        sys.exit(1)

    t_start = time.time()

    mat     = scipy.io.loadmat(str(OASBUD_PATH))
    data    = mat['data']
    patient = data[0, patient_idx]

    pat_id     = str(patient['id'].flat[0]).strip()
    label_val  = int(patient['class'].flat[0])
    true_label = 'benign' if label_val == 0 else 'malignant'
    birads     = str(patient['birads'].flat[0]).strip()
    rf1        = np.array(patient['rf1']).squeeze()

    print(f"  Patient ID  : {pat_id}")
    print(f"  True label  : {true_label.upper()} (BIRADS {birads})")
    print(f"  Signal shape: {rf1.shape}")

    beamformed = reconstruct_image(rf1)
    enhanced   = enhance_image(beamformed)
    matched    = histogram_match(enhanced)

    if cnn_model is None or unet_model is None:
        print("\nLoading models...")
        cnn_model, unet_model = load_models()

    pred_class, confidence, prob_dict = run_cnn(cnn_model, enhanced)

    correct = (pred_class == true_label)
    symbol  = 'v' if correct else 'X'
    print(f"\n  ┌────────────────────────────────────┐")
    print(f"  │ PREDICTION : {pred_class.upper():<22} │")
    print(f"  │ CONFIDENCE : {confidence:.1f}%{'':<21}│")
    print(f"  │ TRUE LABEL : {true_label.upper():<22} │")
    print(f"  │ RESULT     : [{symbol}] {'CORRECT' if correct else 'WRONG':<20} │")
    print(f"  └────────────────────────────────────┘")
    for cls, prob in prob_dict.items():
        bar = '|' * int(prob / 5)
        print(f"    {cls:10s}: {prob:5.1f}% {bar}")

    if unet_model:
        binary_mask, prob_map = run_unet(unet_model, matched)
        tumor_pct = 100 * binary_mask.sum() / (256*256)
        print(f"\n  Tumor region: {tumor_pct:.1f}% of image area")
    else:
        binary_mask = np.zeros((256,256), dtype=np.uint8)
        prob_map    = np.zeros((256,256), dtype=np.float32)
        tumor_pct   = 0.0

    total_ms = (time.time() - t_start) * 1000
    print(f"  Total latency: {total_ms:.0f}ms")

    overlay = create_overlay(enhanced, binary_mask)
    save_result(
        patient_id  = f"patient_{patient_idx+1:03d}_{pat_id}",
        rf_signal   = rf1,
        beamformed  = beamformed,
        enhanced    = enhanced,
        matched     = matched,
        pred_class  = pred_class,
        confidence  = confidence,
        prob_dict   = prob_dict,
        binary_mask = binary_mask,
        prob_map    = prob_map,
        overlay     = overlay,
        tumor_pct   = tumor_pct,
        latency_ms  = total_ms
    )

    return pred_class, true_label, correct


# ════════════════════════════════════════════════════════════════
# PREDICT FROM SYNTHETIC .mat SIGNAL FILE  ← NEW
# ════════════════════════════════════════════════════════════════

def predict_from_signal_file(signal_path, cnn_model=None, unet_model=None):
    """
    Predict from a synthetic .mat signal file.
    e.g. synthetic_signals/signal_01_benign.mat
    Tries common key names: rf1, rf, signal, data.
    Auto-detects true label from filename (benign / malignant).
    """
    print(f"\n{'='*60}")
    print(f"PREDICTING FROM SIGNAL FILE: {signal_path}")
    print(f"{'='*60}")

    t_start = time.time()

    mat = scipy.io.loadmat(str(signal_path))

    # Try common key names
    rf1 = None
    for key in ['rf1', 'rf', 'signal', 'data']:
        if key in mat:
            rf1 = np.array(mat[key]).squeeze()
            print(f"  Loaded key  : '{key}'  shape={rf1.shape}")
            break

    if rf1 is None:
        # Fall back to first non-metadata key
        keys = [k for k in mat.keys() if not k.startswith('__')]
        if not keys:
            print("  ERROR: No data found in .mat file.")
            return None, None, None
        rf1 = np.array(mat[keys[0]]).squeeze()
        print(f"  Loaded key  : '{keys[0]}'  shape={rf1.shape}")

    # Ensure 2D
    if rf1.ndim == 1:
        rf1 = rf1.reshape(-1, 1)

    # Guess true label from filename
    fname      = Path(signal_path).stem.lower()
    true_label = ('benign'    if 'benign'    in fname else
                  'malignant' if 'malignant' in fname else
                  'unknown')
    print(f"  True label  : {true_label.upper()}  (from filename)")

    # Pipeline
    beamformed = reconstruct_image(rf1)
    enhanced   = enhance_image(beamformed)
    matched    = histogram_match(enhanced)

    if cnn_model is None or unet_model is None:
        print("\nLoading models...")
        cnn_model, unet_model = load_models()

    pred_class, confidence, prob_dict = run_cnn(cnn_model, enhanced)

    correct = (pred_class == true_label) if true_label != 'unknown' else None
    symbol  = ('v' if correct else ('?' if correct is None else 'X'))

    print(f"\n  ┌────────────────────────────────────┐")
    print(f"  │ PREDICTION : {pred_class.upper():<22} │")
    print(f"  │ CONFIDENCE : {confidence:.1f}%{'':<21}│")
    print(f"  │ TRUE LABEL : {true_label.upper():<22} │")
    if correct is not None:
        print(f"  │ RESULT     : [{symbol}] {'CORRECT' if correct else 'WRONG':<20} │")
    else:
        print(f"  │ RESULT     : [?] LABEL UNKNOWN            │")
    print(f"  └────────────────────────────────────┘")
    for cls, prob in prob_dict.items():
        bar = '|' * int(prob / 5)
        print(f"    {cls:10s}: {prob:5.1f}% {bar}")

    if unet_model:
        binary_mask, prob_map = run_unet(unet_model, matched)
        tumor_pct = 100 * binary_mask.sum() / (256*256)
        print(f"\n  Tumor region: {tumor_pct:.1f}% of image area")
    else:
        binary_mask = np.zeros((256,256), dtype=np.uint8)
        prob_map    = np.zeros((256,256), dtype=np.float32)
        tumor_pct   = 0.0

    total_ms = (time.time() - t_start) * 1000
    print(f"  Total latency: {total_ms:.0f}ms")

    overlay = create_overlay(enhanced, binary_mask)
    save_result(
        patient_id  = Path(signal_path).stem,
        rf_signal   = rf1,
        beamformed  = beamformed,
        enhanced    = enhanced,
        matched     = matched,
        pred_class  = pred_class,
        confidence  = confidence,
        prob_dict   = prob_dict,
        binary_mask = binary_mask,
        prob_map    = prob_map,
        overlay     = overlay,
        tumor_pct   = tumor_pct,
        latency_ms  = total_ms
    )

    return pred_class, true_label, correct


# ════════════════════════════════════════════════════════════════
# FULL VALIDATION — runs ALL OASBUD patients, prints accuracy table
# ════════════════════════════════════════════════════════════════

def run_full_validation():
    """
    Runs ALL patients in OASBUD.mat and prints:
      - Per-patient prediction vs true label
      - Overall accuracy
      - Breakdown by benign / malignant
    """
    print("\n" + "="*60)
    print("FULL VALIDATION — ALL PATIENTS IN OASBUD.mat")
    print("="*60)

    if OASBUD_PATH is None:
        print("ERROR: OASBUD.mat not found.")
        sys.exit(1)

    mat   = scipy.io.loadmat(str(OASBUD_PATH))
    data  = mat['data']
    n_pat = data.shape[1]
    print(f"Total patients: {n_pat}")

    print("\nLoading models...")
    cnn_model, unet_model = load_models()

    results = []
    print(f"\n{'#':>4} {'ID':>6} {'True':>12} {'Pred':>12} {'Conf':>7} {'OK?':>5}")
    print("-" * 50)

    for i in range(n_pat):
        try:
            patient    = data[0, i]
            pat_id     = str(patient['id'].flat[0]).strip()
            label_val  = int(patient['class'].flat[0])
            true_label = 'benign' if label_val == 0 else 'malignant'
            rf1        = np.array(patient['rf1']).squeeze()

            beamformed = reconstruct_image(rf1)
            enhanced   = enhance_image(beamformed)
            pred_class, confidence, _ = run_cnn(cnn_model, enhanced)

            correct = (pred_class == true_label)
            symbol  = 'YES' if correct else 'NO '
            results.append({
                'id': pat_id, 'true': true_label,
                'pred': pred_class, 'conf': confidence,
                'correct': correct
            })
            print(f"{i+1:>4} {pat_id:>6} {true_label:>12} "
                  f"{pred_class:>12} {confidence:>6.1f}% {symbol}")

        except Exception as e:
            print(f"{i+1:>4} ERROR: {e}")
            continue

    total   = len(results)
    correct = sum(1 for r in results if r['correct'])
    acc     = 100 * correct / total if total > 0 else 0

    benign_results    = [r for r in results if r['true'] == 'benign']
    malignant_results = [r for r in results if r['true'] == 'malignant']

    b_correct = sum(1 for r in benign_results    if r['correct'])
    m_correct = sum(1 for r in malignant_results if r['correct'])

    print("\n" + "="*50)
    print(f"VALIDATION RESULTS")
    print("="*50)
    print(f"  Overall accuracy   : {correct}/{total} = {acc:.1f}%")
    print(f"  Benign  accuracy   : {b_correct}/{len(benign_results)} = "
          f"{100*b_correct/max(len(benign_results),1):.1f}%")
    print(f"  Malignant accuracy : {m_correct}/{len(malignant_results)} = "
          f"{100*m_correct/max(len(malignant_results),1):.1f}%")
    print("="*50)

    os.makedirs('outputs/results', exist_ok=True)
    with open('outputs/results/validation_summary.txt', 'w') as f:
        f.write(f"Overall accuracy   : {correct}/{total} = {acc:.1f}%\n")
        f.write(f"Benign  accuracy   : {b_correct}/{len(benign_results)}\n")
        f.write(f"Malignant accuracy : {m_correct}/{len(malignant_results)}\n\n")
        f.write(f"{'#':>4} {'ID':>6} {'True':>12} {'Pred':>12} {'Conf':>7} {'OK?':>5}\n")
        for i, r in enumerate(results):
            f.write(f"{i+1:>4} {r['id']:>6} {r['true']:>12} "
                    f"{r['pred']:>12} {r['conf']:>6.1f}% "
                    f"{'YES' if r['correct'] else 'NO '}\n")

    print("\nSaved: outputs/results/validation_summary.txt")
    return acc


# ════════════════════════════════════════════════════════════════
# PREDICT FROM IMAGE FILE
# ════════════════════════════════════════════════════════════════

def predict_from_image(image_path):
    """Predict from any PNG image file."""
    print(f"\n{'='*60}")
    print(f"PREDICTING FROM IMAGE: {image_path}")
    print(f"{'='*60}")

    t_start  = time.time()
    img      = np.array(Image.open(image_path).convert('L'),
                        dtype=np.float32) / 255.0
    enhanced = enhance_image(img)
    matched  = histogram_match(enhanced)

    print("\nLoading models...")
    cnn_model, unet_model = load_models()

    pred_class, confidence, prob_dict = run_cnn(cnn_model, enhanced)
    print(f"\n  DIAGNOSIS  : {pred_class.upper()}")
    print(f"  CONFIDENCE : {confidence:.1f}%")
    for cls, prob in prob_dict.items():
        print(f"    {cls:10s}: {prob:.1f}%")

    if unet_model:
        binary_mask, prob_map = run_unet(unet_model, matched)
        tumor_pct = 100 * binary_mask.sum() / (256*256)
        print(f"  Tumor area : {tumor_pct:.1f}%")
    else:
        binary_mask = np.zeros((256,256), dtype=np.uint8)
        prob_map    = np.zeros((256,256), dtype=np.float32)
        tumor_pct   = 0.0

    overlay = create_overlay(enhanced, binary_mask)
    os.makedirs('outputs/results', exist_ok=True)
    stem  = Path(image_path).stem
    color = {'benign':'green','malignant':'red','normal':'blue'}
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    axes[0].imshow(enhanced,    cmap='gray'); axes[0].set_title('Enhanced');      axes[0].axis('off')
    axes[1].imshow(prob_map,    cmap='hot');  axes[1].set_title('Tumor heatmap'); axes[1].axis('off')
    axes[2].imshow(overlay)
    axes[2].set_title(f'{pred_class.upper()}\n{confidence:.1f}%',
                      color=color.get(pred_class,'black'), fontweight='bold')
    axes[2].axis('off')
    plt.tight_layout()
    out = f'outputs/results/prediction_{stem}.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    total_ms = (time.time()-t_start)*1000
    print(f"  Latency : {total_ms:.0f}ms")
    print(f"  Saved   : {out}")


# ════════════════════════════════════════════════════════════════
# RUN
# ════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Breast Ultrasound AI — Diagnosis & Segmentation Pipeline',
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        '--patient', type=int, default=None,
        help='Patient index (0–99) from OASBUD.mat\n'
             '  Example: --patient 0'
    )
    parser.add_argument(
        '--image', type=str, default=None,
        help='Path to a PNG ultrasound image file\n'
             '  Example: --image Dataset/BUSI/benign/benign(1).png'
    )
    parser.add_argument(
        '--signal', type=str, default=None,
        help='Path to a synthetic .mat signal file, OR "all" to run every\n'
             'file inside the synthetic_signals/ folder.\n'
             '  Example: --signal synthetic_signals/signal_01_benign.mat\n'
             '  Example: --signal all'
    )
    parser.add_argument(
        '--validate', action='store_true',
        help='Run ALL patients in OASBUD.mat and print accuracy table'
    )
    args = parser.parse_args()

    # ── Route to the right function ────────────────────────────────

    if args.validate:
        # Full accuracy evaluation on OASBUD dataset
        run_full_validation()

    elif args.patient is not None:
        # Single OASBUD patient by index
        predict_from_signal(args.patient)

    elif args.image is not None:
        # Predict from a PNG image
        predict_from_image(args.image)

    elif args.signal is not None:
        if args.signal.lower() == 'all':
            # Run every .mat file in synthetic_signals/
            sig_dir = Path('synthetic_signals')
            if not sig_dir.exists():
                print(f"ERROR: Folder '{sig_dir}' not found.")
                sys.exit(1)
            files = sorted(sig_dir.glob('*.mat'))
            if not files:
                print(f"ERROR: No .mat files found in '{sig_dir}/'")
                sys.exit(1)

            print(f"\nFound {len(files)} signal files — loading models once...")
            cnn_model, unet_model = load_models()

            results = []
            for f in files:
                try:
                    pred, true, correct = predict_from_signal_file(
                        f, cnn_model=cnn_model, unet_model=unet_model
                    )
                    results.append({
                        'file': f.name, 'pred': pred,
                        'true': true,   'correct': correct
                    })
                except Exception as e:
                    print(f"  ERROR on {f.name}: {e}")

            # Summary table
            print("\n" + "="*60)
            print("SYNTHETIC SIGNAL RESULTS SUMMARY")
            print("="*60)
            print(f"{'File':<35} {'True':>12} {'Pred':>12} {'OK?':>5}")
            print("-" * 68)
            known = [r for r in results if r['correct'] is not None]
            for r in results:
                ok = ('YES' if r['correct'] else 'NO ') if r['correct'] is not None else ' — '
                print(f"{r['file']:<35} {r['true']:>12} {r['pred']:>12} {ok:>5}")
            if known:
                acc = 100 * sum(1 for r in known if r['correct']) / len(known)
                print(f"\n  Accuracy on labelled files: "
                      f"{sum(1 for r in known if r['correct'])}/{len(known)} = {acc:.1f}%")
            print("="*60)

        else:
            # Single synthetic signal file
            predict_from_signal_file(args.signal)

    else:
        # Default: show help + run 5 sample OASBUD patients
        print("="*60)
        print("No arguments provided. Running 5 sample patients.")
        print("="*60)
        print("\nAVAILABLE OPTIONS:")
        print("  --patient  N    Single OASBUD patient (index 0–99)")
        print("  --signal   PATH Synthetic .mat signal file")
        print("  --signal   all  All files in synthetic_signals/")
        print("  --image    PATH PNG ultrasound image")
        print("  --validate      Full accuracy test on all OASBUD patients")
        print("\nEXAMPLES:")
        print("  python step5_predict.py --patient 0")
        print("  python step5_predict.py --signal synthetic_signals/signal_01_benign.mat")
        print("  python step5_predict.py --signal all")
        print("  python step5_predict.py --image  Dataset/BUSI/benign/benign(1).png")
        print("  python step5_predict.py --validate")
        print()

        print("\nLoading models once for all 5 patients...")
        cnn_model, unet_model = load_models()
        for i in [0, 5, 10, 20, 50]:
            predict_from_signal(i, cnn_model=cnn_model, unet_model=unet_model)