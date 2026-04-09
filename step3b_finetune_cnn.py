# ================================================================
# step3b_finetune_cnn.py
#
# WHAT CHANGED FROM PREVIOUS VERSION:
#   1. Unfreeze layer4 + fc (not just fc alone)
#      → fc-only (0.01%) was too little — model couldn't adapt
#      → layer4 + fc = ~37% of params — enough to learn your data
#   2. Fixed: removed verbose=True from ReduceLROnPlateau
#      → Removed in newer PyTorch versions — caused crash
#   3. Epochs: 10 (was 5)
#      → With layer4 unfrozen, needs more epochs to converge
#   4. LR lowered to 0.0001
#      → layer4 has pretrained weights — small LR to not destroy them
#   5. fc layer gets 10x higher LR than layer4
#      → fc starts fresh, needs bigger steps
#      → layer4 is pretrained, needs smaller steps
#
# EXPECTED TIME: 45-90 minutes on CPU
# ================================================================

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from pathlib import Path
from PIL import Image
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import os

device = torch.device('cpu')
print(f"Using device: {device}")
print("=" * 60)


# ── Dataset class ─────────────────────────────────────────────────
class BeamformedCNNDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir  = Path(root_dir)
        self.transform = transform
        self.samples   = []
        self.classes   = ['benign', 'malignant']

        for label_idx, category in enumerate(self.classes):
            folder = self.root_dir / category
            if not folder.exists():
                print(f"  WARNING: folder not found: {folder}")
                continue
            images = [
                f for f in folder.glob('*.png')
                if '_roi'         not in f.name
                and '_comparison' not in f.name
                and '_mask'       not in f.name
            ]
            for img_path in images:
                self.samples.append((img_path, label_idx))

        print(f"  Total images loaded: {len(self.samples)}")
        for idx, cls in enumerate(self.classes):
            n = sum(1 for _, l in self.samples if l == idx)
            print(f"    {cls:12s}: {n} images")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label


# ── Transforms ────────────────────────────────────────────────────
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


# ── Load dataset ──────────────────────────────────────────────────
print("\nLoading augmented dataset from outputs/augmented_images/")
full_dataset = BeamformedCNNDataset(
    'outputs/augmented_images',
    transform=train_transform
)

total      = len(full_dataset)
train_size = int(0.80 * total)

val_size   = total - train_size

train_ds, val_ds = torch.utils.data.random_split(
    full_dataset, [train_size, val_size],
    generator=torch.Generator().manual_seed(42)
)

val_ds.dataset.transform = val_transform

train_loader = DataLoader(
    train_ds, batch_size=32,
    shuffle=True, num_workers=0
)
val_loader = DataLoader(
    val_ds, batch_size=32,
    shuffle=False, num_workers=0
)

print(f"  Train: {train_size} | Val: {val_size}")
print(f"  Train batches per epoch: {len(train_loader)}")
print(f"  Val   batches per epoch: {len(val_loader)}")


# ── Load CNN model ────────────────────────────────────────────────
print("\nLoading best_cnn.pth...")
model        = models.resnet18(weights=None)
num_features = model.fc.in_features   # 512
model.fc     = nn.Linear(num_features, 3)
model.load_state_dict(
    torch.load('outputs/model_checkpoints/best_cnn.pth',
               map_location='cpu')
)

# Replace fc with new 2-class head AFTER loading weights
# (loaded weights had 3-class fc — we swap it out now)
model.fc = nn.Linear(num_features, 2)
model = model.to(device)
print("  Loaded successfully.")


# ── Freeze layer1, layer2, layer3 — unfreeze layer4 + fc ─────────
#
# layer1-3: detect basic edges, shapes, textures
#           These are universal — no need to retrain
#
# layer4:   detects high-level patterns (tumor texture, boundaries)
#           These were tuned for BUSI — need slight adaptation
#           for your beamformed images
#
# fc:       the final decision layer
#           Replaced with fresh 2-class head — must be trained
#
# This is called "partial fine-tuning" — best balance of
# speed vs accuracy when source and target domains are similar

for param in model.parameters():
    param.requires_grad = False      # freeze everything first

for param in model.layer4.parameters():
    param.requires_grad = True       # unfreeze layer4

for param in model.fc.parameters():
    param.requires_grad = True       # unfreeze fc

# Verify
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_p   = sum(p.numel() for p in model.parameters())
print(f"\n  Trainable params : {trainable:,} / {total_p:,}")
print(f"  ({100*trainable/total_p:.1f}% of model — layer4 + fc)")


# ── Optimizer with different LR for layer4 vs fc ─────────────────
#
# fc gets LR = 0.001  — it's a fresh layer, needs bigger steps
# layer4 gets LR = 0.0001 — pretrained, needs small careful steps
#
# Using param groups lets us set different LRs for different parts

optimizer = optim.Adam([
    {'params': model.layer4.parameters(), 'lr': 0.0001},
    {'params': model.fc.parameters(),     'lr': 0.001}
])

criterion = nn.CrossEntropyLoss()

# FIX: removed verbose=True — not supported in newer PyTorch
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, patience=2, factor=0.5
)


# ── Training loop ─────────────────────────────────────────────────
NUM_EPOCHS   = 10
best_val_acc = 0.0
history      = {'train_loss': [], 'val_loss': [],
                'train_acc':  [], 'val_acc':  []}

os.makedirs('outputs/model_checkpoints', exist_ok=True)
os.makedirs('outputs/results', exist_ok=True)

batches_per_epoch = len(train_loader)
secs_per_batch    = 0.2    # layer4 unfrozen = slightly slower per batch
est_minutes       = (batches_per_epoch * NUM_EPOCHS * secs_per_batch) / 60
print(f"\nFine-tuning CNN for {NUM_EPOCHS} epochs...")
print(f"Batches per epoch : {batches_per_epoch}")
print(f"Estimated time    : {est_minutes:.0f}–{est_minutes*1.5:.0f} minutes")
print("-" * 60)

for epoch in range(NUM_EPOCHS):

    # ── Train ─────────────────────────────────────────────────────
    model.train()
    running_loss = 0.0
    correct = total_count = 0

    for batch_idx, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs      = model(images)
        loss         = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted  = torch.max(outputs, 1)
        total_count  += labels.size(0)
        correct      += (predicted == labels).sum().item()

        if (batch_idx + 1) % 50 == 0:
            print(f"  Epoch {epoch+1} | Batch {batch_idx+1}"
                  f"/{batches_per_epoch} | "
                  f"Loss: {running_loss/(batch_idx+1):.4f} | "
                  f"Acc: {100*correct/total_count:.1f}%",
                  end='\r')

    train_loss = running_loss / len(train_loader)
    train_acc  = 100 * correct / total_count
    print()

    # ── Validate ──────────────────────────────────────────────────
    model.eval()
    val_loss = val_correct = val_total = 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs        = model(images)
            loss           = criterion(outputs, labels)
            val_loss      += loss.item()
            _, predicted   = torch.max(outputs, 1)
            val_total     += labels.size(0)
            val_correct   += (predicted == labels).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    val_loss /= len(val_loader)
    val_acc   = 100 * val_correct / val_total

    history['train_loss'].append(train_loss)
    history['val_loss'].append(val_loss)
    history['train_acc'].append(train_acc)
    history['val_acc'].append(val_acc)

    # FIX: scheduler.step() takes the metric value, not verbose
    scheduler.step(val_loss)

    # Print LR so you can see when it drops
    current_lr = optimizer.param_groups[0]['lr']
    print(f"Epoch {epoch+1}/{NUM_EPOCHS} | "
          f"Train Loss: {train_loss:.4f} Acc: {train_acc:.1f}% | "
          f"Val Loss: {val_loss:.4f} Acc: {val_acc:.1f}% | "
          f"LR: {current_lr:.6f}")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(),
            'outputs/model_checkpoints/best_cnn_finetuned.pth')
        print(f"  --> New best saved! Val acc: {val_acc:.1f}%")


# ── Results ───────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"Fine-tuning complete.")
print(f"Best validation accuracy: {best_val_acc:.1f}%")
print(f"{'='*60}")

print("\nClassification report:")
print(classification_report(
    all_labels, all_preds,
    target_names=['benign', 'malignant'],
    zero_division=0
))

# Confusion matrix
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['benign', 'malignant'],
            yticklabels=['benign', 'malignant'])
plt.title(f'Fine-tuned CNN\nVal acc: {best_val_acc:.1f}%')
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.tight_layout()
plt.savefig('outputs/results/finetune_cnn_confusion.png', dpi=150)
print("Saved: outputs/results/finetune_cnn_confusion.png")

# Training curves
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.plot(history['train_loss'], label='Train loss')
ax1.plot(history['val_loss'],   label='Val loss')
ax1.set_xlabel('Epoch'); ax1.set_ylabel('Loss')
ax1.set_title('Fine-tune CNN — loss'); ax1.legend()
ax2.plot(history['train_acc'], label='Train acc')
ax2.plot(history['val_acc'],   label='Val acc')
ax2.set_xlabel('Epoch'); ax2.set_ylabel('Accuracy (%)')
ax2.set_title('Fine-tune CNN — accuracy'); ax2.legend()
plt.tight_layout()
plt.savefig('outputs/results/finetune_cnn_curves.png', dpi=150)
print("Saved: outputs/results/finetune_cnn_curves.png")
print("\nNext: run step4b_finetune_unet.py")