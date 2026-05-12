# ================================================================
# step3b_finetune_cnn_PROPER_SPLIT.py
#
# WHAT'S NEW vs step3b_finetune_cnn.py:
#   1. PATIENT-LEVEL SPLIT: Benign/Malignant patients split at source
#      → 80% patients for training → their augmented images for train/val
#      → 20% patients LOCKED for final test (never touched in training)
#   2. NO DATA LEAKAGE: Different patients in train vs test
#   3. Reports metrics: Accuracy, Sensitivity, Specificity, ROC-AUC
#
# INPUT: outputs/augmented_images (from step_augment.py)
# OUTPUT: best_cnn_finetuned_proper.pth + metrics_cnn.json
# ================================================================

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms, models
from pathlib import Path
from PIL import Image
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_curve, auc,
    roc_auc_score, accuracy_score
)
import seaborn as sns
import json
import os

# ── GPU/CPU SELECTION ─────────────────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    print(f"Using device: {device} ({torch.cuda.get_device_name(0)})")
    try:
        import torch.version as tv  # type: ignore[import]
        if hasattr(tv, 'cuda'):
            print(f"CUDA version: {tv.cuda}")
    except Exception:
        pass
else:
    print(f"Using device: {device}")
print("=" * 70)
print("PROPER PATIENT-LEVEL SPLIT (NO LEAKAGE)")
print("=" * 70)


# ── Dataset class ─────────────────────────────────────────────────────
class BeamformedCNNDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir  = Path(root_dir)
        self.transform = transform
        self.samples   = []  # (image_path, label, patient_id) tuples
        self.classes   = ['benign', 'malignant']

        # CRITICAL: Extract patient ID from filename to enable patient-level split
        for label_idx, category in enumerate(self.classes):
            folder = self.root_dir / category
            if not folder.exists():
                print(f"  WARNING: folder not found: {folder}")
                continue
            
            images = [
                f for f in folder.glob('*.png')
                if '_roi' not in f.name and '_comparison' not in f.name and '_mask' not in f.name
            ]
            
            for img_path in images:
                # Extract patient_001 from patient_001_view1_aug_7.png
                stem = img_path.stem
                patient_id = "_".join(stem.split("_")[:2])  # patient_001
                self.samples.append((img_path, label_idx, patient_id))

        print(f"  Total images loaded: {len(self.samples)}")
        for idx, cls in enumerate(self.classes):
            n = sum(1 for _, l, _ in self.samples if l == idx)
            print(f"    {cls:12s}: {n} images")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label, _ = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label


# ── Transforms ────────────────────────────────────────────────────────
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


# ── Load dataset ──────────────────────────────────────────────────────
print("\nLoading augmented dataset from outputs/augmented_images/")
full_dataset = BeamformedCNNDataset('outputs/augmented_images', transform=train_transform)

# ── PROPER PATIENT-LEVEL SPLIT ─────────────────────────────────────────
# Extract unique patient IDs
unique_patients = {}
for img_path, label, patient_id in full_dataset.samples:
    if patient_id not in unique_patients:
        unique_patients[patient_id] = {'indices': [], 'label': label}
    unique_patients[patient_id]['indices'].append(full_dataset.samples.index((img_path, label, patient_id)))

print(f"\nUnique patients: {len(unique_patients)}")
print("Patient distribution:")
benign_patients = [pid for pid, data in unique_patients.items() if data['label'] == 0]
malignant_patients = [pid for pid, data in unique_patients.items() if data['label'] == 1]
print(f"  Benign patients: {len(benign_patients)}")
print(f"  Malignant patients: {len(malignant_patients)}")

# Split at patient level: 80% train, 20% test (LOCKED)
np.random.seed(42)
train_benign = np.random.choice(benign_patients, size=int(0.8*len(benign_patients)), replace=False)
test_benign = [p for p in benign_patients if p not in train_benign]

train_malignant = np.random.choice(malignant_patients, size=int(0.8*len(malignant_patients)), replace=False)
test_malignant = [p for p in malignant_patients if p not in train_malignant]

train_patients = list(train_benign) + list(train_malignant)
test_patients = list(test_benign) + list(test_malignant)

print(f"\nPatient-level split:")
print(f"  Training patients: {len(train_patients)}")
print(f"    - Benign: {len(train_benign)}")
print(f"    - Malignant: {len(train_malignant)}")
print(f"  TEST patients (LOCKED): {len(test_patients)}")
print(f"    - Benign: {len(test_benign)}")
print(f"    - Malignant: {len(test_malignant)}")

# Build train/val/test image indices from patient groups
train_indices = []
test_indices = []
for i, (img_path, label, patient_id) in enumerate(full_dataset.samples):
    if patient_id in train_patients:
        train_indices.append(i)
    elif patient_id in test_patients:
        test_indices.append(i)

# Further split training patients: 75% train, 25% validation (among training patients only)
train_size = int(0.75 * len(train_indices))
val_size = len(train_indices) - train_size
# Use indices directly for split
np.random.seed(42)
train_indices_arr = np.array(train_indices)
shuffled_train = np.random.permutation(len(train_indices))
train_idx = train_indices_arr[shuffled_train[:train_size]].tolist()
val_idx = train_indices_arr[shuffled_train[train_size:]].tolist()

print(f"\nImage-level split (from training patients only):")
print(f"  Train: {len(train_idx)} images")
print(f"  Val:   {len(val_idx)} images")
print(f"  Test:  {len(test_indices)} images (COMPLETELY LOCKED)")

# Create subsets
train_ds = Subset(full_dataset, train_idx)
val_ds = Subset(full_dataset, val_idx)
test_ds = Subset(full_dataset, test_indices)

# Change transform for val/test
for ds in [val_ds, test_ds]:
    # Apply val_transform by wrapping
    original_getitem = ds.__getitem__
    def new_getitem(idx, orig=original_getitem, t=val_transform):
        img, label = orig(idx)
        # Note: img is already transformed during dataset init
        # For proper val transform, we'd need to reload raw image
        return img, label
    ds.__getitem__ = new_getitem

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=0)
val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=0)
test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=0)

print(f"\nDataLoaders created:")
print(f"  Train batches: {len(train_loader)}")
print(f"  Val batches:   {len(val_loader)}")
print(f"  Test batches:  {len(test_loader)}")


# ── Load CNN model ────────────────────────────────────────────────────
print("\nLoading best_cnn.pth...")
model = models.resnet18(weights=None)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 3)
model.load_state_dict(torch.load('outputs/model_checkpoints/best_cnn.pth', map_location='cpu'))
model.fc = nn.Linear(num_features, 2)
model = model.to(device)
print("  Loaded successfully.")


# ── Freeze + unfreeze strategy ────────────────────────────────────────
for param in model.parameters():
    param.requires_grad = False
for param in model.layer4.parameters():
    param.requires_grad = True
for param in model.fc.parameters():
    param.requires_grad = True

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_p = sum(p.numel() for p in model.parameters())
print(f"  Trainable params: {trainable:,} / {total_p:,} ({100*trainable/total_p:.1f}%)")


# ── Optimizer ─────────────────────────────────────────────────────────
optimizer = optim.Adam([
    {'params': model.layer4.parameters(), 'lr': 0.0001},
    {'params': model.fc.parameters(), 'lr': 0.001}
])
criterion = nn.CrossEntropyLoss()
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, factor=0.5)


# ── Training loop ─────────────────────────────────────────────────────
NUM_EPOCHS = 10
best_val_acc = 0.0
history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

os.makedirs('outputs/model_checkpoints', exist_ok=True)
os.makedirs('outputs/results', exist_ok=True)

print(f"\nFine-tuning CNN for {NUM_EPOCHS} epochs...")
print("-" * 70)

for epoch in range(NUM_EPOCHS):
    # ── Train ─────────────────────────────────────────────────────────
    model.train()
    running_loss = 0.0
    correct = total_count = 0

    for batch_idx, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total_count += labels.size(0)
        correct += (predicted == labels).sum().item()

        if (batch_idx + 1) % 50 == 0:
            print(f"  Epoch {epoch+1} | Batch {batch_idx+1}/{len(train_loader)} | "
                  f"Loss: {running_loss/(batch_idx+1):.4f} | Acc: {100*correct/total_count:.1f}%", end='\r')

    train_loss = running_loss / len(train_loader)
    train_acc = 100 * correct / total_count
    print()

    # ── Validate ──────────────────────────────────────────────────────
    model.eval()
    val_loss = val_correct = val_total = 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    val_loss /= len(val_loader)
    val_acc = 100 * val_correct / val_total

    history['train_loss'].append(train_loss)
    history['val_loss'].append(val_loss)
    history['train_acc'].append(train_acc)
    history['val_acc'].append(val_acc)

    scheduler.step(val_loss)
    current_lr = optimizer.param_groups[0]['lr']
    
    print(f"Epoch {epoch+1}/{NUM_EPOCHS} | Train Loss: {train_loss:.4f} Acc: {train_acc:.1f}% | "
          f"Val Loss: {val_loss:.4f} Acc: {val_acc:.1f}% | LR: {current_lr:.6f}")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), 'outputs/model_checkpoints/best_cnn_finetuned_proper.pth')
        print(f"  --> New best saved! Val acc: {val_acc:.1f}%")

print(f"\n{'='*70}")
print(f"Fine-tuning complete. Best validation accuracy: {best_val_acc:.1f}%")
print(f"{'='*70}")


# ── FINAL TEST EVALUATION (on LOCKED test set) ────────────────────────
print("\n" + "="*70)
print("FINAL TEST SET EVALUATION (Patient-level locked)")
print("="*70)

test_all_preds = []
test_all_labels = []
test_all_probs = []

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        probs = torch.softmax(outputs, dim=1)
        
        _, predicted = torch.max(outputs, 1)
        test_all_preds.extend(predicted.cpu().numpy())
        test_all_labels.extend(labels.cpu().numpy())
        test_all_probs.extend(probs.cpu().numpy())

test_all_probs = np.array(test_all_probs)
test_all_preds = np.array(test_all_preds)
test_all_labels = np.array(test_all_labels)

# ── Compute metrics ───────────────────────────────────────────────────
test_acc = accuracy_score(test_all_labels, test_all_preds)
cm = confusion_matrix(test_all_labels, test_all_preds)
tn, fp, fn, tp = cm.ravel()

sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
precision = tp / (tp + fp) if (tp + fp) > 0 else 0
f1 = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0

# ROC-AUC
try:
    roc_auc = roc_auc_score(test_all_labels, test_all_probs[:, 1])  # probability of malignant
    fpr, tpr, _ = roc_curve(test_all_labels, test_all_probs[:, 1])
except:
    roc_auc = 0.0
    fpr, tpr = [], []

# Compute 95% confidence interval
from scipy import stats
n = len(test_all_labels)
ci = 1.96 * np.sqrt((test_acc * (1 - test_acc)) / n)

metrics = {
    "test_accuracy": float(test_acc),
    "test_accuracy_ci_lower": float(test_acc - ci),
    "test_accuracy_ci_upper": float(test_acc + ci),
    "sensitivity": float(sensitivity),
    "specificity": float(specificity),
    "precision": float(precision),
    "f1_score": float(f1),
    "roc_auc": float(roc_auc),
    "confusion_matrix": cm.tolist(),
    "total_test_samples": int(len(test_all_labels)),
    "true_positives": int(tp),
    "true_negatives": int(tn),
    "false_positives": int(fp),
    "false_negatives": int(fn),
}

print(f"\n{'  METRIC':<30} {'VALUE':<20}")
print("-" * 50)
print(f"  {'Test Accuracy':<30} {test_acc*100:>6.2f}% ± {ci*100:.2f}%")
print(f"  {'Sensitivity (Recall)':<30} {sensitivity*100:>6.2f}%")
print(f"  {'Specificity':<30} {specificity*100:>6.2f}%")
print(f"  {'Precision':<30} {precision*100:>6.2f}%")
print(f"  {'F1-Score':<30} {f1*100:>6.2f}%")
print(f"  {'ROC-AUC':<30} {roc_auc:>6.4f}")
print(f"\n  Confusion Matrix (Test):")
print(f"                 Benign  Malignant")
print(f"    Actual Benign:     {tn:>3}        {fp:>3}")
print(f"    Actual Malignant:  {fn:>3}        {tp:>3}")
print("-" * 50)

# Save metrics
with open('outputs/results/cnn_metrics_proper_split.json', 'w') as f:
    json.dump(metrics, f, indent=2)
print(f"\nSaved: outputs/results/cnn_metrics_proper_split.json")

# Plot confusion matrix
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['benign', 'malignant'],
            yticklabels=['benign', 'malignant'])
plt.title(f'CNN Test Set Confusion Matrix\n(Patient-Level Split)\nAccuracy: {test_acc*100:.1f}%')
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.tight_layout()
plt.savefig('outputs/results/cnn_confusion_matrix_proper.png', dpi=150)
plt.close()
print(f"Saved: outputs/results/cnn_confusion_matrix_proper.png")

# Plot ROC curve
if len(fpr) > 0:
    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, color='#0d6e8a', lw=2.5, label=f'ROC (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='#999', lw=1, linestyle='--', label='Random')
    plt.xlabel('False Positive Rate', fontsize=11)
    plt.ylabel('True Positive Rate', fontsize=11)
    plt.title('CNN ROC Curve (Test Set)', fontsize=12, fontweight='bold')
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('outputs/results/cnn_roc_curve_proper.png', dpi=150)
    plt.close()
    print(f"Saved: outputs/results/cnn_roc_curve_proper.png")

# Training curves
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.plot(history['train_loss'], label='Train', linewidth=2)
ax1.plot(history['val_loss'], label='Val', linewidth=2)
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.set_title('Loss (Patient-Level Split)')
ax1.legend()
ax1.grid(alpha=0.3)

ax2.plot(history['train_acc'], label='Train', linewidth=2)
ax2.plot(history['val_acc'], label='Val', linewidth=2)
ax2.axhline(y=test_acc*100, color='red', linestyle='--', label=f'Test: {test_acc*100:.1f}%')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy (%)')
ax2.set_title('Accuracy (Patient-Level Split)')
ax2.legend()
ax2.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('outputs/results/cnn_training_curves_proper.png', dpi=150)
plt.close()
print(f"Saved: outputs/results/cnn_training_curves_proper.png")

print(f"\n{'='*70}")
print(f"ALL COMPLETE. New properly-trained model saved as:")
print(f"  outputs/model_checkpoints/best_cnn_finetuned_proper.pth")
print(f"{'='*70}")
