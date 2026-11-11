
# ============================================================
# ESS 5420 - Deep Learning for Earth and Space
# EuroSAT Land-Cover Classification
# Final Test Evaluation: Small CNN vs ResNet-18
# ============================================================

import csv
from pathlib import Path
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Subset

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support
)


# ============================================================
# 1. Configure device and paths
# ============================================================

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print("Using device:", device)

PROJECT_DIR = Path(
    "/Users/zarrintasneem/Documents/ESS5480_Project"
)

DATA_DIR = PROJECT_DIR / "EuroSAT_RGB"
SPLIT_DIR = PROJECT_DIR / "splits"
MODEL_DIR = PROJECT_DIR / "models"
RESULTS_DIR = PROJECT_DIR / "results"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. Load images with identical preprocessing
# ============================================================

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

dataset = datasets.ImageFolder(
    root=DATA_DIR,
    transform=transform
)

class_names = dataset.classes
num_classes = len(class_names)

assert len(dataset) == 27000
assert num_classes == 10

print("Total images:", len(dataset))
print("Classes:", class_names)


# ============================================================
# 3. Load the held-out test split
# ============================================================

def read_split(name):

    path = SPLIT_DIR / f"eurosat-{name}.txt"

    with open(path, "r") as f:
        names = [
            Path(line.strip()).name
            for line in f
            if line.strip()
        ]

    if len(names) != len(set(names)):
        raise ValueError(f"Duplicate filenames in {path}")

    return set(names)


train_files = read_split("train")
val_files = read_split("val")
test_files = read_split("test")

assert train_files.isdisjoint(val_files)
assert train_files.isdisjoint(test_files)
assert val_files.isdisjoint(test_files)

image_lookup = {
    Path(path).name: i
    for i, (path, label) in enumerate(dataset.samples)
}

if len(image_lookup) != len(dataset):
    raise ValueError("Duplicate dataset filenames.")

assert (
    train_files | val_files | test_files
) == set(image_lookup), "Dataset and split files differ."

test_indices = sorted(
    image_lookup[name]
    for name in test_files
)

assert len(test_indices) == 5400

test_dataset = Subset(dataset, test_indices)

test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=0
)

print("Test images:", len(test_dataset))


# ============================================================
# 4. Recreate the small CNN architecture
# ============================================================

class SmallCNN(nn.Module):

    def __init__(self, num_classes=10):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


small_cnn = SmallCNN(num_classes=10)

small_cnn.load_state_dict(
    torch.load(
        MODEL_DIR / "small_cnn_best.pth",
        map_location="cpu",
        weights_only=True
    )
)

small_cnn = small_cnn.to(device)
small_cnn.eval()


# ============================================================
# 5. Recreate pretrained ResNet-18 architecture
# ============================================================

# No ImageNet download is needed here.
# The saved checkpoint contains your fine-tuned weights.

resnet18 = models.resnet18(weights=None)

num_features = resnet18.fc.in_features
resnet18.fc = nn.Linear(num_features, 10)

resnet18.load_state_dict(
    torch.load(
        MODEL_DIR / "resnet18_best.pth",
        map_location="cpu",
        weights_only=True
    )
)

resnet18 = resnet18.to(device)
resnet18.eval()

print("Both checkpoints loaded successfully.")


# ============================================================
# 6. Generate predictions
# ============================================================

def get_predictions(model, loader):

    model.eval()

    true_labels = []
    predicted_labels = []

    with torch.inference_mode():

        for images, labels in loader:

            images = images.to(device)

            outputs = model(images)
            predictions = outputs.argmax(dim=1)

            true_labels.extend(labels.tolist())

            predicted_labels.extend(
                predictions.cpu().tolist()
            )

    return (
        np.array(true_labels),
        np.array(predicted_labels)
    )


# Final test inference: do not use results for tuning.

y_true, cnn_pred = get_predictions(
    small_cnn,
    test_loader
)

y_true_resnet, resnet_pred = get_predictions(
    resnet18,
    test_loader
)

assert np.array_equal(y_true, y_true_resnet)

print("\nTest predictions completed.")


# ============================================================
# 7. Calculate overall accuracy
# ============================================================

cnn_accuracy = accuracy_score(y_true, cnn_pred)

resnet_accuracy = accuracy_score(
    y_true,
    resnet_pred
)

print("\n=== Final Test Accuracy ===")

print(
    f"Small CNN: {cnn_accuracy * 100:.2f}%"
)

print(
    f"ResNet-18: {resnet_accuracy * 100:.2f}%"
)

print(
    "Difference (percentage points):",
    round(
        (resnet_accuracy - cnn_accuracy) * 100,
        2
    )
)


# ============================================================
# 8. Class-wise precision, recall and F1
# ============================================================

labels = list(range(num_classes))

cnn_precision, cnn_recall, cnn_f1, support = (
    precision_recall_fscore_support(
        y_true,
        cnn_pred,
        labels=labels,
        zero_division=0
    )
)

res_precision, res_recall, res_f1, _ = (
    precision_recall_fscore_support(
        y_true,
        resnet_pred,
        labels=labels,
        zero_division=0
    )
)

print("\n=== Small CNN Classification Report ===")

print(
    classification_report(
        y_true,
        cnn_pred,
        labels=labels,
        target_names=class_names,
        digits=4,
        zero_division=0
    )
)

print("\n=== ResNet-18 Classification Report ===")

print(
    classification_report(
        y_true,
        resnet_pred,
        labels=labels,
        target_names=class_names,
        digits=4,
        zero_division=0
    )
)


# ============================================================
# 9. Save class-wise comparison
# ============================================================

comparison_path = (
    RESULTS_DIR / "classwise_comparison.csv"
)

with open(comparison_path, "w", newline="") as f:

    writer = csv.writer(f)

    writer.writerow([
        "Class",
        "Support",
        "CNN Precision",
        "CNN Recall",
        "CNN F1",
        "ResNet Precision",
        "ResNet Recall",
        "ResNet F1",
        "F1 Difference"
    ])

    for i, name in enumerate(class_names):

        writer.writerow([
            name,
            int(support[i]),
            float(cnn_precision[i]),
            float(cnn_recall[i]),
            float(cnn_f1[i]),
            float(res_precision[i]),
            float(res_recall[i]),
            float(res_f1[i]),
            float(res_f1[i] - cnn_f1[i])
        ])

print("Class-wise comparison saved:", comparison_path)


# ============================================================
# 10. Calculate confusion matrices
# ============================================================

cnn_cm = confusion_matrix(
    y_true,
    cnn_pred,
    labels=labels
)

resnet_cm = confusion_matrix(
    y_true,
    resnet_pred,
    labels=labels
)

def plot_confusion_matrix(cm, title, filename):

    fig, ax = plt.subplots(figsize=(10, 8))

    image = ax.imshow(
        cm,
        cmap="Blues"
    )

    fig.colorbar(image, ax=ax)

    ax.set_xticks(range(num_classes))
    ax.set_yticks(range(num_classes))

    ax.set_xticklabels(
        class_names,
        rotation=45,
        ha="right"
    )

    ax.set_yticklabels(class_names)

    for i in range(num_classes):
        for j in range(num_classes):

            value = cm[i, j]

            ax.text(
                j, i,
                str(value),
                ha="center",
                va="center",
                fontsize=8,
                color=(
                    "white"
                    if value > cm.max() / 2
                    else "black"
                )
            )

    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title(title)

    plt.tight_layout()

    plt.savefig(
        RESULTS_DIR / filename,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)


plot_confusion_matrix(
    cnn_cm,
    "Small CNN: Test Confusion Matrix",
    "small_cnn_confusion_matrix.png"
)

plot_confusion_matrix(
    resnet_cm,
    "ResNet-18: Test Confusion Matrix",
    "resnet18_confusion_matrix.png"
)


# ============================================================
# 11. Compare F1-scores visually
# ============================================================

x = np.arange(num_classes)
width = 0.38

fig, ax = plt.subplots(figsize=(12, 6))

ax.bar(
    x - width / 2,
    cnn_f1,
    width,
    label="Small CNN"
)

ax.bar(
    x + width / 2,
    res_f1,
    width,
    label="ResNet-18"
)

ax.set_xticks(x)
ax.set_xticklabels(
    class_names,
    rotation=45,
    ha="right"
)

ax.set_ylabel("F1-score")
ax.set_ylim(0, 1.05)
ax.set_title("Class-wise Test F1 Comparison")
ax.legend()
ax.grid(axis="y", alpha=0.3)

plt.tight_layout()

plt.savefig(
    RESULTS_DIR / "classwise_f1_comparison.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close(fig)


# ============================================================
# 12. Identify most common misclassifications
# ============================================================

def print_top_confusions(cm, model_name, top_n=10):

    errors = []

    for true_id in range(num_classes):
        for pred_id in range(num_classes):

            if true_id != pred_id and cm[true_id, pred_id] > 0:

                errors.append((
                    int(cm[true_id, pred_id]),
                    class_names[true_id],
                    class_names[pred_id]
                ))

    errors.sort(reverse=True)

    print(f"\n=== Top Confusions: {model_name} ===")

    for count, true_name, pred_name in errors[:top_n]:

        print(
            f"{true_name} -> {pred_name}: "
            f"{count} images"
        )


print_top_confusions(
    cnn_cm,
    "Small CNN"
)

print_top_confusions(
    resnet_cm,
    "ResNet-18"
)


# ============================================================
# 13. Save overall results
# ============================================================

summary_path = (
    RESULTS_DIR / "final_model_comparison.csv"
)

with open(summary_path, "w", newline="") as f:

    writer = csv.writer(f)

    writer.writerow([
        "Model",
        "Test Accuracy",
        "Macro F1",
        "Weighted F1"
    ])

    for name, predictions in [
        ("Small CNN", cnn_pred),
        ("ResNet-18", resnet_pred)
    ]:

        report = classification_report(
            y_true,
            predictions,
            labels=labels,
            output_dict=True,
            zero_division=0
        )

        writer.writerow([
            name,
            accuracy_score(y_true, predictions),
            report["macro avg"]["f1-score"],
            report["weighted avg"]["f1-score"]
        ])


# ============================================================
# 14. Final summary
# ============================================================

print("\n=== Final Evaluation Completed ===")

print("Both models evaluated on 5,400 test images.")
print("Confusion matrices saved.")
print("Class-wise metrics saved.")
print("Model comparison saved.")

print("\nResults directory:", RESULTS_DIR)


# ============================================================
# 15. Pair every prediction with the correct original RGB file
# ============================================================

# The test DataLoader uses shuffle=False, and test_dataset was constructed
# from sorted test_indices. Preserve this ordering when linking filenames.
from PIL import Image

ERROR_DIR = RESULTS_DIR / "misclassification_analysis"
ERROR_DIR.mkdir(parents=True, exist_ok=True)

assert len(test_indices) == len(y_true) == len(cnn_pred) == len(resnet_pred)
assert len(test_indices) == 5400

prediction_rows = []
for test_position, dataset_index in enumerate(test_indices):
    path, actual_index = dataset.samples[dataset_index]
    assert actual_index == int(y_true[test_position]), (
        f"Label mismatch at test position {test_position}: {path}"
    )

    cnn_index = int(cnn_pred[test_position])
    resnet_index = int(resnet_pred[test_position])
    cnn_ok = cnn_index == actual_index
    resnet_ok = resnet_index == actual_index

    if cnn_ok and resnet_ok:
        outcome = "both_correct"
    elif not cnn_ok and resnet_ok:
        outcome = "resnet_corrected_cnn"
    elif cnn_ok and not resnet_ok:
        outcome = "cnn_correct_resnet_wrong"
    else:
        outcome = "both_wrong"

    prediction_rows.append({
        "test_position": test_position,
        "filename": Path(path).name,
        "image_path": str(path),
        "true_class": class_names[actual_index],
        "cnn_prediction": class_names[cnn_index],
        "resnet_prediction": class_names[resnet_index],
        "cnn_correct": cnn_ok,
        "resnet_correct": resnet_ok,
        "outcome": outcome,
    })

columns = list(prediction_rows[0].keys())
prediction_csv = ERROR_DIR / "prediction_comparison.csv"
with open(prediction_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=columns)
    writer.writeheader()
    writer.writerows(prediction_rows)

counts = Counter(row["outcome"] for row in prediction_rows)
print("\n=== Prediction Agreement ===")
for outcome in (
    "both_correct", "resnet_corrected_cnn",
    "cnn_correct_resnet_wrong", "both_wrong"
):
    print(f"{outcome}: {counts[outcome]}")
assert sum(counts.values()) == len(test_indices)


# ============================================================
# 16. Create reproducible image grids (original, unnormalized RGB)
# ============================================================

# Select examples reproducibly; never use the test set to tune models.
example_rng = np.random.default_rng(42)

def save_image_grid(rows, title, filename, n=12):
    if not rows:
        print("No examples available for:", title)
        return

    selected_positions = example_rng.choice(
        len(rows), size=min(n, len(rows)), replace=False
    )
    selected = [rows[int(i)] for i in selected_positions]
    ncols = min(4, len(selected))
    nrows = int(np.ceil(len(selected) / ncols))
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(4.0 * ncols, 3.8 * nrows),
        squeeze=False
    )

    for ax in axes.flat:
        ax.axis("off")

    for ax, row in zip(axes.flat, selected):
        # Load the RGB source directly: displaying normalized tensors would
        # give misleading colors.
        with Image.open(row["image_path"]) as im:
            rgb = im.convert("RGB")
            ax.imshow(rgb)
        ax.set_title(
            f"True: {row['true_class']}\n"
            f"CNN: {row['cnn_prediction']}\n"
            f"ResNet: {row['resnet_prediction']}",
            fontsize=9
        )
        ax.axis("off")

    fig.suptitle(title, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    output_path = ERROR_DIR / filename
    fig.savefig(output_path, dpi=250, bbox_inches="tight")
    plt.close(fig)
    print("Saved:", output_path)


for outcome, title in [
    ("resnet_corrected_cnn", "ResNet-18 Corrected CNN Errors"),
    ("cnn_correct_resnet_wrong", "CNN Correct, ResNet-18 Incorrect"),
    ("both_wrong", "Both Models Incorrect"),
]:
    save_image_grid(
        [row for row in prediction_rows if row["outcome"] == outcome],
        title, f"{outcome}.png"
    )


# ============================================================
# 17. Priority directional class-confusion examples
# ============================================================

def inspect_confusion(true_class, predicted_class, model_name):
    prediction_key = (
        "cnn_prediction" if model_name == "cnn" else "resnet_prediction"
    )
    rows = [
        row for row in prediction_rows
        if row["true_class"] == true_class
        and row[prediction_key] == predicted_class
    ]
    print(f"{model_name}: {true_class} -> {predicted_class}: {len(rows)}")
    save_image_grid(
        rows, f"{model_name.upper()}: {true_class} -> {predicted_class}",
        f"{model_name}_{true_class}_to_{predicted_class}.png"
    )


# Keep class labels exactly as returned by ImageFolder.
priority_pairs = [
    ("PermanentCrop", "HerbaceousVegetation"),
    ("HerbaceousVegetation", "PermanentCrop"),
    ("River", "Highway"),
    ("Highway", "River"),
]
for true_class, predicted_class in priority_pairs:
    if true_class not in class_names or predicted_class not in class_names:
        raise ValueError(
            f"Check class names: {true_class}, {predicted_class}; "
            f"available: {class_names}"
        )
    inspect_confusion(true_class, predicted_class, "cnn")
    inspect_confusion(true_class, predicted_class, "resnet")


# ============================================================
# 18. Save full directional confusion counts and comparison
# ============================================================

pair_csv = ERROR_DIR / "directional_confusion_comparison.csv"
with open(pair_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow([
        "True Class", "Predicted Class", "CNN Count",
        "ResNet-18 Count", "ResNet Minus CNN"
    ])
    for true_idx, true_name in enumerate(class_names):
        for predicted_idx, predicted_name in enumerate(class_names):
            if true_idx == predicted_idx:
                continue
            a = int(cnn_cm[true_idx, predicted_idx])
            b = int(resnet_cm[true_idx, predicted_idx])
            writer.writerow([true_name, predicted_name, a, b, b - a])

print("\n=== Error Analysis Completed ===")
print("Individual predictions:", prediction_csv)
print("Directional confusion counts:", pair_csv)
print("Image grids and CSVs saved in:", ERROR_DIR)
print("Use test examples for interpretation, not model selection or tuning.")
