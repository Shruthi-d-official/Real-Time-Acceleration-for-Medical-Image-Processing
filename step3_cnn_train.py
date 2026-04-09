import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from pathlib import Path
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import os

# ── Check if GPU is available (it won't be on your laptop, that's fine) ───────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
# Will print: Using device: cpu

# ── Dataset class ─────────────────────────────────────────────────────────────
# PyTorch needs a Dataset class that tells it how to load your images
class BUSIDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir  = Path(root_dir)
        self.transform = transform
        self.samples   = []  # list of (image_path, label) tuples
        self.classes   = ['benign', 'malignant', 'normal']

        # Walk through each category folder and collect image paths + labels
        for label_idx, category in enumerate(self.classes):
            folder = self.root_dir / category
            if not folder.exists():
                print(f"Warning: folder not found: {folder}")
                continue
            # Only grab images, not masks (masks have '_mask' in filename)
            images = [f for f in folder.glob('*.png') if '_mask' not in f.name]
            for img_path in images:
                self.samples.append((img_path, label_idx))

        print(f"Total images loaded: {len(self.samples)}")

    def __len__(self):
        # PyTorch calls this to know how many items are in the dataset
        return len(self.samples)

    def __getitem__(self, idx):
        # PyTorch calls this to get one image at a time during training
        img_path, label = self.samples[idx]

        # Open image and convert to RGB
        # Even though ultrasound is grayscale, ResNet expects 3 channels (RGB)
        # convert('RGB') duplicates the gray channel 3 times
        image = Image.open(img_path).convert('RGB')

        # Apply transforms (resize, normalize, augmentation)
        if self.transform:
            image = self.transform(image)

        return image, label

# ── Define transforms (image preprocessing) ──────────────────────────────────
# Training transforms include augmentation (random flips, rotation)
# This artificially increases your dataset size and prevents overfitting
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),          # resize all images to 224x224
    transforms.RandomHorizontalFlip(),      # randomly flip left-right (50% chance)
    transforms.RandomRotation(10),          # randomly rotate up to 10 degrees
    transforms.ColorJitter(brightness=0.2, # randomly change brightness slightly
                           contrast=0.2),
    transforms.ToTensor(),                  # convert PIL image to PyTorch tensor
    # Normalize using ImageNet mean/std — required for pretrained ResNet
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Validation transforms — no augmentation, just resize and normalize
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ── Load dataset and split into train/validation ──────────────────────────────
# We use the enhanced images we created in step 2
full_dataset = BUSIDataset('outputs/enhanced_images', transform=train_transform)

# Split: 80% training, 20% validation
total      = len(full_dataset)
train_size = int(0.8 * total)
val_size   = total - train_size

# random_split randomly divides the dataset
train_dataset, val_dataset = torch.utils.data.random_split(
    full_dataset, [train_size, val_size],
    generator=torch.Generator().manual_seed(42)  # seed for reproducibility
)

# Apply the correct transform to validation set
val_dataset.dataset.transform = val_transform

# DataLoader feeds batches of images to the model during training
# batch_size=16 means 16 images processed at once (good for CPU)
# shuffle=True randomizes order each epoch
# num_workers=0 is important on Windows (avoids multiprocessing errors)
train_loader = DataLoader(train_dataset, batch_size=16,
                          shuffle=True, num_workers=0)
val_loader   = DataLoader(val_dataset, batch_size=16,
                          shuffle=False, num_workers=0)

print(f"Train: {train_size} images, Validation: {val_size} images")

# ── Build the CNN model using Transfer Learning ───────────────────────────────
# Transfer learning = start with a model already trained on millions of images
# (ImageNet) and fine-tune it for your specific task
# This is MUCH better than training from scratch, especially without a GPU

# ResNet18 is a lightweight model — good choice for CPU training
model = models.resnet18(weights='IMAGENET1K_V1')
# weights='IMAGENET1K_V1' downloads pretrained weights automatically

# The last layer of ResNet18 outputs 1000 classes (for ImageNet)
# We need to replace it with a layer that outputs 3 classes
# (benign, malignant, normal)
num_features = model.fc.in_features  # = 512 for ResNet18
model.fc = nn.Linear(num_features, 3)  # replace final layer

model = model.to(device)  # move model to CPU (or GPU if available)

# ── Loss function and optimizer ───────────────────────────────────────────────
# CrossEntropyLoss is standard for multi-class classification
criterion = nn.CrossEntropyLoss()

# Adam optimizer adjusts learning rate automatically
# lr=0.001 is the learning rate (how big each weight update is)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Learning rate scheduler: reduce LR by factor 0.1 if val loss doesn't
# improve for 3 epochs — helps fine-tune at the end of training
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3,
                                                  factor=0.1)

# ── Training loop ─────────────────────────────────────────────────────────────
NUM_EPOCHS = 20  # 20 passes through the full dataset
# On CPU this takes roughly 5-10 minutes per epoch depending on dataset size

train_losses, val_losses     = [], []
train_accs,   val_accs       = [], []
best_val_acc                 = 0.0

os.makedirs('outputs/model_checkpoints', exist_ok=True)

for epoch in range(NUM_EPOCHS):
    # ── Training phase ────────────────────────────────────────────────────────
    model.train()  # put model in training mode (activates dropout, batch norm)
    running_loss    = 0.0
    correct         = 0
    total           = 0

    for batch_idx, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()   # clear gradients from previous batch

        outputs = model(images) # forward pass: get predictions

        loss = criterion(outputs, labels)  # calculate how wrong the model is

        loss.backward()         # backpropagation: calculate gradients

        optimizer.step()        # update weights using gradients

        running_loss += loss.item()

        # Count correct predictions
        _, predicted = torch.max(outputs, 1)  # get index of highest score
        total   += labels.size(0)
        correct += (predicted == labels).sum().item()

    train_loss = running_loss / len(train_loader)
    train_acc  = 100 * correct / total

    # ── Validation phase ──────────────────────────────────────────────────────
    model.eval()   # put model in eval mode (disables dropout)
    val_loss    = 0.0
    correct     = 0
    total       = 0

    with torch.no_grad():  # no need to calculate gradients during validation
        for images, labels in val_loader:
            images  = images.to(device)
            labels  = labels.to(device)
            outputs = model(images)
            loss    = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total   += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_loss /= len(val_loader)
    val_acc   = 100 * correct / total

    # Save losses and accuracies for plotting
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    train_accs.append(train_acc)
    val_accs.append(val_acc)

    scheduler.step(val_loss)  # update learning rate if needed

    # Save the best model seen so far
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(),
                   'outputs/model_checkpoints/best_cnn.pth')
        print(f"  --> New best model saved! Val acc: {val_acc:.1f}%")

    print(f"Epoch {epoch+1:2d}/{NUM_EPOCHS} | "
          f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.1f}% | "
          f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.1f}%")

print(f"\nTraining complete. Best validation accuracy: {best_val_acc:.1f}%")

# ── Plot training curves ──────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.plot(train_losses, label='Train loss')
ax1.plot(val_losses,   label='Val loss')
ax1.set_xlabel('Epoch'); ax1.set_ylabel('Loss')
ax1.set_title('Loss over training'); ax1.legend()

ax2.plot(train_accs, label='Train accuracy')
ax2.plot(val_accs,   label='Val accuracy')
ax2.set_xlabel('Epoch'); ax2.set_ylabel('Accuracy (%)')
ax2.set_title('Accuracy over training'); ax2.legend()

plt.tight_layout()
os.makedirs('outputs/results', exist_ok=True)   # ← fix: create folder before saving
plt.savefig('outputs/results/training_curves.png', dpi=150)
plt.show()