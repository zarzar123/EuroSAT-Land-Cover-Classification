# ESS 5420 - Deep Learning for Earth and Space
# Project: EuroSAT Land-Cover Classification
# Dataset Preparation and Preprocessing
# Weights & Biases Integration

# Import libraries

import torch
import wandb
from pathlib import Path
from collections import Counter
from urllib.request import urlretrieve

from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

import matplotlib.pyplot as plt

# Configure GPU

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print("PyTorch version:", torch.__version__)
print("Using device:", device)

# Dataset paths

PROJECT_DIR = Path(
    "/Users/zarrintasneem/Documents/ESS5480_Project"
)

DATA_DIR = PROJECT_DIR / "EuroSAT_RGB"
SPLIT_DIR = PROJECT_DIR / "splits"
RESULTS_DIR = PROJECT_DIR / "results"

SPLIT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

if not DATA_DIR.is_dir():
    raise FileNotFoundError(
        f"EuroSAT directory not found: {DATA_DIR}"
    )

# Preprocessing

# Preserve the original 64 x 64 RGB images.
# Normalize using ImageNet channel statistics.

NORMALIZATION_MEAN = [0.485, 0.456, 0.406]
NORMALIZATION_STD = [0.229, 0.224, 0.225]

transform = transforms.Compose([
    transforms.ToTensor(),

    transforms.Normalize(
        mean=NORMALIZATION_MEAN,
        std=NORMALIZATION_STD
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

print("\nClass distribution:")

class_counts = Counter(dataset.targets)

for class_index, count in sorted(class_counts.items()):
    class_name = dataset.classes[class_index]
    print(f"{class_name}: {count}")

assert len(dataset) == 27000, (
    "Expected 27,000 EuroSAT RGB images."
)

assert len(dataset.classes) == 10, (
    "Expected 10 land-cover classes."
)

# Check for duplicated filenames across the folders.

filenames = [
    Path(path).name
    for path, label in dataset.samples
]

assert len(filenames) == len(set(filenames)), (
    "Duplicate image filenames found."
)

print("Dataset integrity checks passed.")

# Download TorchGeo files that have the predefined split

BASE_URL = (
    "https://huggingface.co/datasets/torchgeo/eurosat/"
    "resolve/main/"
)

for split in ["train", "val", "test"]:

    filename = f"eurosat-{split}.txt"
    destination = SPLIT_DIR / filename

    if not destination.exists():
        print(f"Downloading {filename}")
        urlretrieve(BASE_URL + filename, destination)

print("\nSplit files are ready.")

# Read the splits that have been predefined by Helber et al.

def read_split_filenames(split_file):

    with open(split_file, "r") as f:
        filenames = [
            Path(line.strip()).name
            for line in f
            if line.strip()
        ]

    assert len(filenames) == len(set(filenames)), (
        f"Duplicate filenames in {split_file}"
    )

    return set(filenames)


train_files = read_split_filenames(
    SPLIT_DIR / "eurosat-train.txt"
)

val_files = read_split_filenames(
    SPLIT_DIR / "eurosat-val.txt"
)

test_files = read_split_filenames(
    SPLIT_DIR / "eurosat-test.txt"
)

# Check that split files do not overlap.

assert train_files.isdisjoint(val_files)
assert train_files.isdisjoint(test_files)
assert val_files.isdisjoint(test_files)

print("No duplicate filenames across splits.")

# Match the images to splits

image_lookup = {
    Path(path).name: index
    for index, (path, label) in enumerate(dataset.samples)
}

all_split_files = train_files | val_files | test_files

missing_files = all_split_files - set(image_lookup)
unassigned_files = set(image_lookup) - all_split_files

if missing_files or unassigned_files:

    print("\nMissing dataset images:", len(missing_files))
    print("Unassigned dataset images:", len(unassigned_files))

    print(
        "Example missing:",
        sorted(missing_files)[:5]
    )

    print(
        "Example unassigned:",
        sorted(unassigned_files)[:5]
    )

    raise ValueError(
        "TorchGeo split files do not match the downloaded "
        "Kaggle RGB images. Verify filenames and dataset version."
    )


def get_indices(split_files):

    return sorted(
        image_lookup[filename]
        for filename in split_files
    )


train_indices = get_indices(train_files)
val_indices = get_indices(val_files)
test_indices = get_indices(test_files)

# Verify train/validation/test partitions

print("\n=== Dataset Partitions ===")

print("Training:", len(train_indices))
print("Validation:", len(val_indices))
print("Testing:", len(test_indices))

assert len(train_indices) == 16200
assert len(val_indices) == 5400
assert len(test_indices) == 5400

assert (
    len(train_indices)
    + len(val_indices)
    + len(test_indices)
) == len(dataset)

print("All 27,000 images assigned exactly once.")
print("No overlapping images between partitions.")

# Create PyTorch datasets

train_dataset = Subset(
    dataset,
    train_indices
)

val_dataset = Subset(
    dataset,
    val_indices
)

test_dataset = Subset(
    dataset,
    test_indices
)

# Create DataLoaders

BATCH_SIZE = 32

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

print("\n=== DataLoaders ===")
print("Training batches:", len(train_loader))
print("Validation batches:", len(val_loader))
print("Testing batches:", len(test_loader))

# Test one batch on GPU

images, labels = next(iter(train_loader))

images = images.to(device)
labels = labels.to(device)

print("\n=== GPU Test ===")

print("Images shape:", images.shape)
print("Labels shape:", labels.shape)
print("Images device:", images.device)

# Visualize original EuroSAT images

raw_dataset = datasets.ImageFolder(
    root=DATA_DIR,
    transform=transforms.ToTensor()
)

fig, axes = plt.subplots(
    2, 5,
    figsize=(12, 6)
)

for i, ax in enumerate(axes.flat):

    image, label = raw_dataset[train_indices[i]]

    ax.imshow(image.permute(1, 2, 0))
    ax.set_title(raw_dataset.classes[label])
    ax.axis("off")

plt.suptitle("EuroSAT RGB Images")
plt.tight_layout()

FIGURE_PATH = RESULTS_DIR / "eurosat_sample_images.png"

fig.savefig(
    FIGURE_PATH,
    dpi=300,
    bbox_inches="tight"
)

print("Sample image saved:", FIGURE_PATH)

# Initialize Weights & Biases

print("\n=== Initializing Weights & Biases ===")

run = wandb.init(
    project="EuroSAT-CNN-Classification",
    name="EuroSAT-Dataset-Preparation",
    job_type="data-preparation",

    config={
        "dataset": "EuroSAT RGB",
        "source": "Sentinel-2",
        "total_images": len(dataset),
        "num_classes": len(dataset.classes),
        "image_height": 64,
        "image_width": 64,
        "image_channels": 3,
        "spatial_resolution_m": 10,
        "split_source": "TorchGeo",
        "split_ratio": "60/20/20",
        "training_samples": len(train_dataset),
        "validation_samples": len(val_dataset),
        "testing_samples": len(test_dataset),
        "batch_size": BATCH_SIZE,
        "normalization": "ImageNet",
        "normalization_mean": NORMALIZATION_MEAN,
        "normalization_std": NORMALIZATION_STD,
        "data_augmentation": False,
        "device": str(device),
        "pytorch_version": torch.__version__,
    }
)

# Log information about datasets to W&B

print("\nLogging dataset information...")

# Record verified dataset statistics.

run.summary["total_images"] = len(dataset)
run.summary["num_classes"] = len(dataset.classes)

run.summary["training_samples"] = len(train_dataset)
run.summary["validation_samples"] = len(val_dataset)
run.summary["testing_samples"] = len(test_dataset)

run.summary["duplicate_filenames"] = 0
run.summary["split_overlap_filenames"] = 0
run.summary["missing_files"] = len(missing_files)
run.summary["unassigned_files"] = len(unassigned_files)

run.summary["dataset_integrity_passed"] = True
run.summary["geographic_independence_verified"] = False

run.summary["device"] = str(device)

# Log class distribution to W&B

class_table = wandb.Table(
    columns=["Land-Cover Class", "Number of Images"]
)

for class_index, count in sorted(class_counts.items()):

    class_table.add_data(
        dataset.classes[class_index],
        int(count)
    )

run.log({
    "dataset/class_distribution": class_table
})

# Log split distribution to W&B

split_table = wandb.Table(
    columns=["Dataset Partition", "Number of Images"]
)

split_table.add_data(
    "Training",
    len(train_dataset)
)

split_table.add_data(
    "Validation",
    len(val_dataset)
)

split_table.add_data(
    "Testing",
    len(test_dataset)
)

run.log({
    "dataset/split_distribution": split_table
})

# Log sample images to W&B

run.log({
    "dataset/sample_images": wandb.Image(
        str(FIGURE_PATH),
        caption="EuroSAT RGB training images"
    )
})

print("Dataset image uploaded to W&B.")

# Finish W&B run

print("\n=== W&B Summary ===")

print("Project:", "EuroSAT-CNN-Classification")
print("Run name:", run.name)
print("Run URL:", run.url)

run.finish()

plt.show()

# Preprocessing completed

print("\nEuroSAT preprocessing completed successfully!")
print("Dataset ready for CNN and ResNet-18 training.")
print("Dataset preparation recorded in Weights & Biases.")
