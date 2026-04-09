import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from PIL import Image
import numpy as np
import os
import sys

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ── Auto-detect dataset paths ─────────────────────────────────────────────────
def find_dataset_paths(base="Dataset"):
    """
    Searches common BrEaST dataset folder layouts and returns (img_dir, mask_dir).
    Prints a diagnostic tree so you can see exactly what was found.
    """
    base = Path(base)

    # Print top-level structure to help debug
    print("\n── Scanning Dataset folder ──────────────────────────────────────────")
    if not base.exists():
        print(f"  ERROR: '{base}' folder not found in current directory: {Path.cwd()}")
        print("  Please make sure you run this script from the folder that contains 'Dataset/'")
        sys.exit(1)

    for p in sorted(base.rglob("*")):
        if p.is_dir():
            print(f"  DIR : {p}")
        else:
            # Only show first 5 files per folder to avoid spam
            siblings = list(p.parent.iterdir())
            if siblings.index(p) < 5:
                print(f"  FILE: {p}")
            elif siblings.index(p) == 5:
                print(f"       ... ({len(siblings)} files total in {p.parent})")

    print("─────────────────────────────────────────────────────────────────────\n")

    # Candidate layouts — add more here if needed
    candidates = [
        # Original expected layout
        (base / "BrEaST-Lesions_USG-images_and_masks" / "images",
         base / "BrEaST-Lesions_USG-images_and_masks" / "masks"),

        # Flat layout: Dataset/images + Dataset/masks
        (base / "images", base / "masks"),

        # Sometimes masks folder is named "Masks" (capital M)
        (base / "BrEaST-Lesions_USG-images_and_masks" / "images",
         base / "BrEaST-Lesions_USG-images_and_masks" / "Masks"),

        # Dataset itself is the images folder
        (base, base.parent / "masks"),
    ]

    for img_dir, mask_dir in candidates:
        if img_dir.exists() and mask_dir.exists():
            imgs  = list(img_dir.glob("*.png")) + list(img_dir.glob("*.jpg"))
            masks = list(mask_dir.glob("*.png")) + list(mask_dir.glob("*.jpg"))
            if imgs and masks:
                print(f"✔ Found images : {img_dir}  ({len(imgs)} files)")
                print(f"✔ Found masks  : {mask_dir}  ({len(masks)} files)\n")
                return str(img_dir), str(mask_dir)

    # ── Last resort: search anywhere under Dataset for image/mask pairs ───────
    print("  Standard layouts not matched. Searching for any paired folders...")
    all_dirs = [p for p in base.rglob("*") if p.is_dir()]
    img_dirs  = [d for d in all_dirs if "image" in d.name.lower()]
    mask_dirs = [d for d in all_dirs if "mask"  in d.name.lower()]

    for img_dir in img_dirs:
        for mask_dir in mask_dirs:
            imgs  = list(img_dir.glob("*.png"))  + list(img_dir.glob("*.jpg"))
            masks = list(mask_dir.glob("*.png")) + list(mask_dir.glob("*.jpg"))
            if imgs and masks:
                # Check at least some names overlap
                img_names  = {p.name for p in imgs}
                mask_names = {p.name for p in masks}
                if img_names & mask_names:
                    print(f"✔ Auto-detected images : {img_dir}  ({len(imgs)} files)")
                    print(f"✔ Auto-detected masks  : {mask_dir}  ({len(masks)} files)\n")
                    return str(img_dir), str(mask_dir)

    print("\n  FATAL: Could not find matching image/mask folders under 'Dataset/'.")
    print("  Make sure your folder contains .png (or .jpg) files and that image")
    print("  filenames match mask filenames (e.g. img001.png ↔ img001.png).")
    sys.exit(1)


IMG_DIR, MASK_DIR = find_dataset_paths("Dataset")

# ── UNet architecture ─────────────────────────────────────────────────────────
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()
        self.enc1 = DoubleConv(in_channels, 64)
        self.enc2 = DoubleConv(64, 128)
        self.enc3 = DoubleConv(128, 256)
        self.enc4 = DoubleConv(256, 512)
        self.bottleneck = DoubleConv(512, 1024)
        self.pool = nn.MaxPool2d(2)

        self.up4  = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec4 = DoubleConv(1024, 512)
        self.up3  = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = DoubleConv(512, 256)
        self.up2  = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = DoubleConv(256, 128)
        self.up1  = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = DoubleConv(128, 64)
        self.final = nn.Conv2d(64, out_channels, 1)

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

# ── Dataset ───────────────────────────────────────────────────────────────────
class SegmentationDataset(Dataset):
    def __init__(self, img_dir, mask_dir, size=256):
        img_dir  = Path(img_dir)
        mask_dir = Path(mask_dir)

        # Accept both .png and .jpg
        all_imgs = sorted(list(img_dir.glob("*.png")) + list(img_dir.glob("*.jpg")))

        self.pairs = []
        for img_p in all_imgs:
            # Try exact name match first, then stem match (different extension)
            mask_p = mask_dir / img_p.name
            if not mask_p.exists():
                # Try opposite extension
                alt_ext = ".jpg" if img_p.suffix == ".png" else ".png"
                mask_p  = mask_dir / (img_p.stem + alt_ext)
            if mask_p.exists():
                self.pairs.append((img_p, mask_p))

        print(f"Found {len(self.pairs)} image-mask pairs  "
              f"(scanned {len(all_imgs)} images)")

        if len(self.pairs) == 0:
            print("\n  Tip: Image filenames must match mask filenames.")
            print(f"  Sample images : {[p.name for p in all_imgs[:5]]}")
            mask_files = sorted(list(mask_dir.glob("*.png")) +
                                list(mask_dir.glob("*.jpg")))
            print(f"  Sample masks  : {[p.name for p in mask_files[:5]]}")
            sys.exit(1)

        self.size = size

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_path, mask_path = self.pairs[idx]

        img  = Image.open(img_path).convert("L").resize((self.size, self.size))
        mask = Image.open(mask_path).convert("L").resize((self.size, self.size))

        img  = np.array(img,  dtype=np.float32) / 255.0
        mask = np.array(mask, dtype=np.float32) / 255.0
        mask = (mask > 0.5).astype(np.float32)

        img  = torch.tensor(img).unsqueeze(0)
        mask = torch.tensor(mask).unsqueeze(0)
        return img, mask

# ── Build loaders ─────────────────────────────────────────────────────────────
dataset    = SegmentationDataset(IMG_DIR, MASK_DIR, size=256)
train_size = int(0.8 * len(dataset))
val_size   = len(dataset) - train_size
train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_ds, batch_size=4, shuffle=True,  num_workers=0)
val_loader   = DataLoader(val_ds,   batch_size=4, shuffle=False, num_workers=0)
print(f"Train: {len(train_ds)} samples | Val: {len(val_ds)} samples\n")

# ── Model, loss, optimizer ────────────────────────────────────────────────────
model     = UNet(in_channels=1, out_channels=1).to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

def dice_score(pred, target, threshold=0.5):
    pred  = (torch.sigmoid(pred) > threshold).float()
    inter = (pred * target).sum()
    return (2 * inter) / (pred.sum() + target.sum() + 1e-8)

# ── Training loop ─────────────────────────────────────────────────────────────
NUM_EPOCHS = 15
best_dice  = 0.0
os.makedirs("outputs/model_checkpoints", exist_ok=True)

for epoch in range(NUM_EPOCHS):
    model.train()
    train_loss = 0.0

    for imgs, masks in train_loader:
        imgs, masks = imgs.to(device), masks.to(device)
        optimizer.zero_grad()
        preds = model(imgs)
        loss  = criterion(preds, masks)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    model.eval()
    val_loss = val_dice = 0.0

    with torch.no_grad():
        for imgs, masks in val_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            preds     = model(imgs)
            val_loss += criterion(preds, masks).item()
            val_dice += dice_score(preds, masks).item()

    val_loss /= len(val_loader)
    val_dice /= len(val_loader)

    if val_dice > best_dice:
        best_dice = val_dice
        torch.save(model.state_dict(),
                   "outputs/model_checkpoints/best_unet.pth")
        print(f"  --> New best UNet saved! Dice: {val_dice:.4f}")

    print(f"Epoch {epoch+1:2d}/{NUM_EPOCHS} | "
          f"Train Loss: {train_loss/len(train_loader):.4f} | "
          f"Val Loss: {val_loss:.4f} | Dice: {val_dice:.4f}")