# ================================================================
# step4b_finetune_unet.py  — FULLY SELF-CONTAINED
#
# NO IMPORTS FROM step4_unet_train.py
# UNet architecture matches best_unet.pth exactly:
#   uses .conv (not .block) to match saved weight keys
#
# INPUT:
#   outputs/augmented_images/benign/*.png
#   outputs/augmented_images/malignant/*.png
#   outputs/augmented_masks/benign/*_mask.png
#   outputs/augmented_masks/malignant/*_mask.png
#   outputs/model_checkpoints/best_unet.pth
#
# OUTPUT:
#   outputs/model_checkpoints/best_unet_finetuned.pth
#   outputs/results/finetune_unet_curves.png
# ================================================================

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from PIL import Image
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import random

device = torch.device('cpu')
print(f"Using device: {device}")
print("=" * 60)


# ════════════════════════════════════════════════════════════════
# UNET — matches your saved best_unet.pth exactly
# Key difference: sequential is named 'conv' (not 'block')
# This is what the error told us — saved keys are enc1.conv.*
# ════════════════════════════════════════════════════════════════

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        # Named 'conv' to match your original step4_unet_train.py
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1,
                 features=[64, 128, 256, 512]):
        super().__init__()

        # Encoder
        self.enc1 = ConvBlock(in_channels,   features[0])
        self.enc2 = ConvBlock(features[0],   features[1])
        self.enc3 = ConvBlock(features[1],   features[2])
        self.enc4 = ConvBlock(features[2],   features[3])
        self.pool = nn.MaxPool2d(2, 2)

        # Bottleneck
        self.bottleneck = ConvBlock(features[3], features[3] * 2)

        # Decoder
        self.up4  = nn.ConvTranspose2d(features[3]*2, features[3], 2, 2)
        self.dec4 = ConvBlock(features[3]*2, features[3])

        self.up3  = nn.ConvTranspose2d(features[3], features[2], 2, 2)
        self.dec3 = ConvBlock(features[2]*2, features[2])

        self.up2  = nn.ConvTranspose2d(features[2], features[1], 2, 2)
        self.dec2 = ConvBlock(features[1]*2, features[1])

        self.up1  = nn.ConvTranspose2d(features[1], features[0], 2, 2)
        self.dec1 = ConvBlock(features[0]*2, features[0])

        # Final 1×1 conv → binary mask
        self.final = nn.Conv2d(features[0], out_channels, kernel_size=1)

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
# MASK QUALITY CHECK
# ════════════════════════════════════════════════════════════════

def check_mask_quality(mask_dir, sample_count=30):
    mask_dir  = Path(mask_dir)
    all_masks = []
    for category in ['benign', 'malignant']:
        folder = mask_dir / category
        if folder.exists():
            all_masks.extend(list(folder.glob('*.png')))

    if len(all_masks) == 0:
        print("  WARNING: No mask files found!")
        return False

    sample    = random.sample(all_masks, min(sample_count, len(all_masks)))
    nonzero   = 0
    fill_pcts = []

    for p in sample:
        arr      = np.array(Image.open(p).convert('L'))
        fill_pct = (arr > 127).sum() / arr.size * 100
        fill_pcts.append(fill_pct)
        if fill_pct > 0.1:
            nonzero += 1

    print(f"\n── Mask Quality Check {'─'*38}")
    print(f"  Total mask files         : {len(all_masks)}")
    print(f"  Sampled                  : {len(sample)}")
    print(f"  Masks with tumor pixels  : {nonzero}/{len(sample)}")
    print(f"  Average tumor fill       : {np.mean(fill_pcts):.2f}%")

    if nonzero == 0:
        print("  !! ALL MASKS ARE EMPTY — UNet will learn nothing !!")
        return False
    elif nonzero < len(sample) * 0.5:
        print("  WARNING: More than half masks are empty.")
        return True
    else:
        print("  OK — masks have tumor regions.")
        return True


# ════════════════════════════════════════════════════════════════
# DATASET
# ════════════════════════════════════════════════════════════════

class AugmentedSegDataset(Dataset):
    def __init__(self, img_dir, mask_dir, img_size=256, augment=False):
        self.img_dir  = Path(img_dir)
        self.mask_dir = Path(mask_dir)
        self.img_size = img_size
        self.augment  = augment
        self.pairs    = []

        print(f"\n── Loading Dataset {'─'*42}")
        print(f"  Image dir : {self.img_dir.resolve()}")
        print(f"  Mask dir  : {self.mask_dir.resolve()}")

        for category in ['benign', 'malignant']:
            img_folder  = self.img_dir  / category
            mask_folder = self.mask_dir / category

            if not img_folder.exists():
                print(f"  X Image folder missing: {img_folder}")
                continue
            if not mask_folder.exists():
                print(f"  X Mask folder missing : {mask_folder}")
                continue

            all_imgs  = sorted([
                f for f in img_folder.glob('*.png')
                if not f.name.endswith('_mask.png')
            ])
            matched   = 0
            unmatched = 0

            for img_path in all_imgs:
                mask_path = mask_folder / (img_path.stem + '_mask.png')
                if mask_path.exists():
                    self.pairs.append((img_path, mask_path, category))
                    matched += 1
                else:
                    unmatched += 1

            status = "OK" if matched > 0 else "FAIL"
            print(f"  [{status}] {category:12s}: "
                  f"{matched:5d} pairs | {unmatched} skipped")

        print(f"\n  Total pairs : {len(self.pairs)}")

        if len(self.pairs) == 0:
            raise RuntimeError(
                "No image-mask pairs found. "
                "Run step_augmentation.py first."
            )

    def __len__(self):
        return len(self.pairs)

    def _online_aug(self, img, mask):
        if np.random.random() > 0.5:
            img  = np.fliplr(img).copy()
            mask = np.fliplr(mask).copy()
        img = np.clip(img * np.random.uniform(0.85, 1.15), 0, 1)
        if np.random.random() > 0.6:
            img = np.clip(
                img + np.random.normal(0, 0.01, img.shape).astype(np.float32),
                0, 1
            )
        return img, mask

    def __getitem__(self, idx):
        img_path, mask_path, _ = self.pairs[idx]

        img      = Image.open(img_path).convert('L')
        img      = img.resize((self.img_size, self.img_size), Image.BILINEAR)
        img      = np.array(img, dtype=np.float32) / 255.0

        mask     = Image.open(mask_path).convert('L')
        mask     = mask.resize((self.img_size, self.img_size), Image.NEAREST)
        mask     = np.array(mask, dtype=np.float32) / 255.0
        mask     = (mask > 0.5).astype(np.float32)

        if self.augment:
            img, mask = self._online_aug(img, mask)

        return (torch.tensor(img.copy()).unsqueeze(0),
                torch.tensor(mask.copy()).unsqueeze(0))


# ════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════

# 1. Mask quality check
masks_ok = check_mask_quality('outputs/augmented_masks')
if not masks_ok:
    print("\nStopping. Fix mask files before training.")
    exit(1)

# 2. Load dataset
full_dataset = AugmentedSegDataset(
    img_dir  = 'outputs/augmented_images',
    mask_dir = 'outputs/augmented_masks',
    img_size = 256,
    augment  = False
)

total      = len(full_dataset)
train_size = int(0.80 * total)
val_size   = total - train_size

train_ds, val_ds = torch.utils.data.random_split(
    full_dataset, [train_size, val_size],
    generator=torch.Generator().manual_seed(42)
)
train_ds.dataset.augment = True

train_loader = DataLoader(train_ds, batch_size=16, shuffle=True,  num_workers=0)
val_loader   = DataLoader(val_ds,   batch_size=16, shuffle=False, num_workers=0)

print(f"\n  Train : {train_size} images | {len(train_loader)} batches/epoch")
print(f"  Val   : {val_size}  images | {len(val_loader)}  batches/epoch")

# 3. Load UNet with correct architecture
print("\nLoading UNet weights...")
model     = UNet(in_channels=1, out_channels=1).to(device)
unet_path = 'outputs/model_checkpoints/best_unet.pth'

if os.path.exists(unet_path):
    model.load_state_dict(torch.load(unet_path, map_location='cpu'))
    print("  Loaded best_unet.pth successfully.")
else:
    print("  WARNING: best_unet.pth not found — using random weights.")

# 4. Freeze encoder, train decoder
for param in model.enc1.parameters():       param.requires_grad = False
for param in model.enc2.parameters():       param.requires_grad = False
for param in model.enc3.parameters():       param.requires_grad = False
for param in model.enc4.parameters():       param.requires_grad = False
for param in model.bottleneck.parameters(): param.requires_grad = False

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_p   = sum(p.numel() for p in model.parameters())
print(f"\n  Trainable : {trainable:,} / {total_p:,} params")
print(f"  ({100*trainable/total_p:.1f}% — decoder only)")

# 5. Loss functions
bce = nn.BCEWithLogitsLoss()

def dice_loss(pred, target, smooth=1.0):
    pred  = torch.sigmoid(pred)
    inter = (pred * target).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
    return 1.0 - ((2.0 * inter + smooth) / (union + smooth)).mean()

def combined_loss(pred, target):
    return 0.5 * bce(pred, target) + 0.5 * dice_loss(pred, target)

def dice_score(pred, target, thr=0.5):
    pred  = (torch.sigmoid(pred) > thr).float()
    inter = (pred * target).sum()
    denom = pred.sum() + target.sum()
    return (2.0 * inter) / (denom + 1e-8)

# 6. Optimizer + scheduler
optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=0.0001
)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, patience=2, factor=0.5
)

# 7. Training loop
NUM_EPOCHS = 5
best_dice  = 0.0
history    = {'train_loss': [], 'val_loss': [], 'val_dice': []}

os.makedirs('outputs/model_checkpoints', exist_ok=True)
os.makedirs('outputs/results',           exist_ok=True)

est_min = (len(train_loader) * NUM_EPOCHS * 0.8) / 60
print(f"\nFine-tuning UNet for {NUM_EPOCHS} epochs...")
print(f"  Estimated time : {est_min:.0f}–{est_min*1.4:.0f} minutes")
print("-" * 60)

for epoch in range(NUM_EPOCHS):

    # Train
    model.train()
    train_loss = 0.0

    for batch_idx, (imgs, masks) in enumerate(train_loader):
        imgs, masks = imgs.to(device), masks.to(device)
        optimizer.zero_grad()
        loss = combined_loss(model(imgs), masks)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

        if (batch_idx + 1) % 30 == 0:
            print(f"  Epoch {epoch+1} | Batch {batch_idx+1}/"
                  f"{len(train_loader)} | "
                  f"Loss: {train_loss/(batch_idx+1):.4f}", end='\r')

    train_loss /= len(train_loader)
    print()

    # Validate
    model.eval()
    val_loss = val_dice_acc = 0.0

    with torch.no_grad():
        for imgs, masks in val_loader:
            imgs, masks   = imgs.to(device), masks.to(device)
            preds         = model(imgs)
            val_loss     += combined_loss(preds, masks).item()
            val_dice_acc += dice_score(preds, masks).item()

    val_loss     /= len(val_loader)
    val_dice_acc /= len(val_loader)

    history['train_loss'].append(train_loss)
    history['val_loss'].append(val_loss)
    history['val_dice'].append(val_dice_acc)

    scheduler.step(val_loss)
    current_lr = optimizer.param_groups[0]['lr']

    print(f"Epoch {epoch+1:2d}/{NUM_EPOCHS} | "
          f"Train Loss: {train_loss:.4f} | "
          f"Val Loss: {val_loss:.4f} | "
          f"Val Dice: {val_dice_acc:.4f} | "
          f"LR: {current_lr:.6f}")

    if val_dice_acc > best_dice:
        best_dice = val_dice_acc
        torch.save(model.state_dict(),
                   'outputs/model_checkpoints/best_unet_finetuned.pth')
        print(f"  --> New best saved! Dice: {val_dice_acc:.4f}")

# 8. Summary
print(f"\n{'='*60}")
print(f"Fine-tuning complete.  Best Val Dice: {best_dice:.4f}")
if   best_dice >= 0.85: print("  Result : EXCELLENT")
elif best_dice >= 0.70: print("  Result : GOOD")
elif best_dice >= 0.50: print("  Result : MODERATE — try unfreezing enc4")
else:                   print("  Result : LOW — check mask files")
print(f"{'='*60}")

# 9. Plot curves
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.plot(history['train_loss'], label='Train loss', marker='o')
ax1.plot(history['val_loss'],   label='Val loss',   marker='s')
ax1.set_xlabel('Epoch'); ax1.set_ylabel('Loss')
ax1.set_title('Fine-tune UNet — loss'); ax1.legend(); ax1.grid(alpha=0.3)

ax2.plot(history['val_dice'], label='Val Dice', marker='o', color='green')
ax2.set_xlabel('Epoch'); ax2.set_ylabel('Dice score')
ax2.set_title('Fine-tune UNet — Dice'); ax2.set_ylim(0, 1)
ax2.axhline(y=0.85, color='red',    linestyle='--', label='Excellent (0.85)')
ax2.axhline(y=0.70, color='orange', linestyle='--', label='Good (0.70)')
ax2.legend(); ax2.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('outputs/results/finetune_unet_curves.png', dpi=150)
print("\nSaved: outputs/results/finetune_unet_curves.png")
print("Next : python step5_predict.py --patient 0")