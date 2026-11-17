# ESS 5420 - Deep Learning for Earth and Space
# Project: EuroSAT Land-Cover Classification
# Model 2: ImageNet-Pretrained ResNet-18

# Import libraries

import random
import csv
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import datasets, transforms, models
from torchvision.models import ResNet18_Weights
from torch.utils.data import DataLoader, Subset

import matplotlib.pyplot as plt
import wandb

# Configure random seed and GPU

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print("PyTorch version:", torch.__version__)
print("Using device:", device)

# Define directories

PROJECT_DIR = Path(
    "/Users/zarrintasneem/Documents/ESS5480_Project"
)

DATA_DIR = PROJECT_DIR / "EuroSAT_RGB"
SPLIT_DIR = PROJECT_DIR / "splits"
MODEL_DIR = PROJECT_DIR / "models"
RESULTS_DIR = PROJECT_DIR / "results"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Image preprocessing

# Same preprocessing as the small CNN.
# Original image dimensions: 64 x 64 x 13

transform = transforms.Compose([
    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Load EuroSAT RGB dataset

dataset = datasets.ImageFolder(
    root=DATA_DIR,
    transform=transform
)

print("\n=== EuroSAT Dataset ===")
print("Total images:", len(dataset))
print("Number of classes:", len(dataset.classes))

assert len(dataset) == 27000
assert len(dataset.classes) == 10

# Load predefined TorchGeo splits by Helber et al., 2017

def read_split_filenames(split_name):

    split_file = (
        SPLIT_DIR / f"eurosat-{split_name}.txt"
    )

    if not split_file.is_file():
        raise FileNotFoundError(
            f"Missing split file: {split_file}"
        )

    with open(split_file, "r") as f:
        filenames = [
            Path(line.strip()).name
            for line in f
            if line.strip()
        ]

    if len(filenames) != len(set(filenames)):
        raise ValueError(
            f"Duplicate filenames in {split_file}"
        )

    return set(filenames)


train_files = read_split_filenames("train")
val_files = read_split_filenames("val")
test_files = read_split_filenames("test")

# Match filenames and verify partitions

image_lookup = {
    Path(path).name: index
    for index, (path, label) in enumerate(dataset.samples)
}

if len(image_lookup) != len(dataset):
    raise ValueError("Duplicate dataset filenames.")

assert train_files.isdisjoint(val_files)
assert train_files.isdisjoint(test_files)
assert val_files.isdisjoint(test_files)

all_files = train_files | val_files | test_files

if all_files != set(image_lookup):
    raise ValueError(
        "Split files do not match the EuroSAT images."
    )


def get_indices(filenames):
    return sorted(
        image_lookup[name]
        for name in filenames
    )


train_indices = get_indices(train_files)
val_indices = get_indices(val_files)
test_indices = get_indices(test_files)

assert len(train_indices) == 16200
assert len(val_indices) == 5400
assert len(test_indices) == 5400

print("\n=== Dataset Partitions ===")
print("Training:", len(train_indices))
print("Validation:", len(val_indices))
print("Testing:", len(test_indices))

# Create DataLoaders

train_dataset = Subset(dataset, train_indices)
val_dataset = Subset(dataset, val_indices)
test_dataset = Subset(dataset, test_indices)

BATCH_SIZE = 32

train_generator = torch.Generator()
train_generator.manual_seed(SEED)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    generator=train_generator
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

# Reserved for final evaluation.
test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

print("\n=== DataLoaders ===")
print("Training batches:", len(train_loader))
print("Validation batches:", len(val_loader))

# Load pretrained ResNet-18

weights = ResNet18_Weights.IMAGENET1K_V1

model = models.resnet18(weights=weights)

# Replace original 1000-class classification layer
# with a 10-class layer for EuroSAT.

num_features = model.fc.in_features

model.fc = nn.Linear(
    num_features,
    10
)

# Fine-tune all layers.
for parameter in model.parameters():
    parameter.requires_grad = True

model = model.to(device)

num_parameters = sum(
    p.numel()
    for p in model.parameters()
    if p.requires_grad
)

print("\n=== ResNet-18 ===")
print("Pretrained weights: ImageNet-1K V1")
print("Output classes: 10")
print("Trainable parameters:", num_parameters)
print("Model device:", next(model.parameters()).device)

# Training configuration

EPOCHS = 20
LEARNING_RATE = 0.0001

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)

print("\n=== Training Configuration ===")
print("Epochs:", EPOCHS)
print("Batch size:", BATCH_SIZE)
print("Learning rate:", LEARNING_RATE)
print("Optimizer: Adam")
print("Loss: CrossEntropyLoss")
print("Fine-tuning: All layers")
print("Augmentation: None")

# Authenticated once beforehand using `wandb login` in the terminal
# Use the same project as the Small CNN, but a separate run.
wandb_run = wandb.init(
    project="EuroSAT-CNN-Classification",
    name=f"resnet18-imagenet-pretrained-seed-{SEED}",
    job_type="training",
    config={
        "model": "ResNet18",
        "dataset": "EuroSAT RGB",
        "pretrained_weights": "IMAGENET1K_V1",
        "fine_tuning": "all layers",
        "seed": SEED,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "optimizer": "Adam",
        "loss": "CrossEntropyLoss",
        "augmentation": False,
        "split": "TorchGeo 60/20/20",
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
        "test_samples": len(test_dataset),
        "trainable_parameters": num_parameters,
        "device": str(device),
    },
)

# Training function

def train_one_epoch(model, loader):

    model.train()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for images, labels in loader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)

        total_loss += loss.item() * batch_size

        predictions = outputs.argmax(dim=1)

        total_correct += (
            predictions == labels
        ).sum().item()

        total_samples += batch_size

    return (
        total_loss / total_samples,
        total_correct / total_samples
    )

# Validation function

def validate(model, loader):

    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            batch_size = labels.size(0)

            total_loss += loss.item() * batch_size

            predictions = outputs.argmax(dim=1)

            total_correct += (
                predictions == labels
            ).sum().item()

            total_samples += batch_size

    return (
        total_loss / total_samples,
        total_correct / total_samples
    )

# Fine-tune ResNet-18

history = []

best_val_loss = float("inf")
best_epoch = 0

MODEL_PATH = MODEL_DIR / "resnet18_best.pth"

print("\n=== ResNet-18 Training Started ===")

for epoch in range(1, EPOCHS + 1):

    train_loss, train_acc = train_one_epoch(
        model,
        train_loader
    )

    val_loss, val_acc = validate(
        model,
        val_loader
    )

    history.append({
        "epoch": epoch,
        "train_loss": train_loss,
        "train_accuracy": train_acc,
        "val_loss": val_loss,
        "val_accuracy": val_acc
    })

    wandb_run.log({
        "epoch": epoch,
        "train/loss": train_loss,
        "train/accuracy": train_acc,
        "validation/loss": val_loss,
        "validation/accuracy": val_acc,
    }, step=epoch)

    # Same checkpoint selection criterion
    # used for the small CNN.

    if val_loss < best_val_loss:

        best_val_loss = val_loss
        best_epoch = epoch

        torch.save(
            model.state_dict(),
            MODEL_PATH
        )

    print(
        f"Epoch {epoch:02d}/{EPOCHS} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Train Acc: {train_acc:.4f} | "
        f"Val Loss: {val_loss:.4f} | "
        f"Val Acc: {val_acc:.4f}",
        flush=True
    )

print("\n=== Training Completed ===")
print("Best epoch:", best_epoch)
print("Best validation loss:", round(best_val_loss, 4))
print("Best model:", MODEL_PATH)

# Save training history

CSV_PATH = (
    RESULTS_DIR / "resnet18_training_history.csv"
)

with open(CSV_PATH, "w", newline="") as f:

    writer = csv.DictWriter(
        f,
        fieldnames=[
            "epoch",
            "train_loss",
            "train_accuracy",
            "val_loss",
            "val_accuracy"
        ]
    )

    writer.writeheader()
    writer.writerows(history)

print("Training history saved:", CSV_PATH)

# Plot training curves

epochs = [row["epoch"] for row in history]

train_losses = [
    row["train_loss"] for row in history
]

val_losses = [
    row["val_loss"] for row in history
]

train_accuracies = [
    row["train_accuracy"] for row in history
]

val_accuracies = [
    row["val_accuracy"] for row in history
]

fig, axes = plt.subplots(
    1, 2,
    figsize=(12, 5)
)

axes[0].plot(
    epochs,
    train_losses,
    label="Training"
)

axes[0].plot(
    epochs,
    val_losses,
    label="Validation"
)

axes[0].set_title("ResNet-18: Loss")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Cross-Entropy Loss")
axes[0].legend()
axes[0].grid(True)

axes[1].plot(
    epochs,
    train_accuracies,
    label="Training"
)

axes[1].plot(
    epochs,
    val_accuracies,
    label="Validation"
)

axes[1].set_title("ResNet-18: Accuracy")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Accuracy")
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()

FIGURE_PATH = (
    RESULTS_DIR / "resnet18_training_curves.png"
)

plt.savefig(
    FIGURE_PATH,
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("Training curves saved:", FIGURE_PATH)

wandb_run.log({"training/curves": wandb.Image(str(FIGURE_PATH))})

# Final summary

best_row = history[best_epoch - 1]

print("\n=== ResNet-18 Summary ===")

print("Best epoch:", best_epoch)

print(
    "Best validation loss:",
    round(best_row["val_loss"], 4)
)

print(
    "Validation accuracy at best epoch:",
    round(best_row["val_accuracy"], 4)
)

print("Model saved:", MODEL_PATH)

print("\nResNet-18 fine-tuning completed successfully.")

wandb_run.summary["best_epoch"] = best_epoch
wandb_run.summary["best_validation_loss"] = best_row["val_loss"]
wandb_run.summary["best_validation_accuracy"] = best_row["val_accuracy"]
wandb_run.summary["checkpoint_path"] = str(MODEL_PATH)
print("W&B run URL:", wandb_run.url)
wandb_run.finish()
