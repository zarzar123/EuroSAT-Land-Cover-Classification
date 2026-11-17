"""Evaluate saved EuroSAT models and inspect class-specific errors.

This script:
1. Validates the original EuroSAT train/validation/test splits.
2. Loads the saved Small CNN and ResNet-18 checkpoints.
3. Evaluates both models on the same 5,400 test images.
4. Records predictions, confidence scores, and correctness.
5. Exports class metrics, confusion matrices, and error counts.
6. Generates seven visual error-analysis figures.
7. Reports progress, timing, and inference throughput.

No retraining occurs. Existing checkpoints are never modified.
"""

from pathlib import Path
import math
import time

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms

try:
    from tqdm.auto import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        print(
            "tqdm is not installed; processing without a progress bar.",
            flush=True,
        )
        return iterable

# CONFIGURATION

PROJECT_DIR = Path(
    "/Users/zarrintasneem/Documents/ESS5480_Project"
)

DATA_DIR = PROJECT_DIR / "EuroSAT_RGB"
SPLIT_DIR = PROJECT_DIR / "splits"
MODEL_DIR = PROJECT_DIR / "models"

OUTPUT_DIR = (
    PROJECT_DIR
    / "results"
    / "visual_error_analysis"
)

BATCH_SIZE = 64

# Start with 0 on macOS to avoid multiprocessing startup issues.
# Benchmark 2 later if image loading proves to be the bottleneck.
NUM_WORKERS = 0

MAX_IMAGES_PER_FIGURE = 12
FIGURE_DPI = 200

EXPECTED_DATASET_SIZE = 27000
EXPECTED_TEST_SIZE = 5400
NUM_CLASSES = 10

# Set to a smaller number (e.g., 10) for a quick speed test.
# Leave as None for the complete scientific evaluation.
MAX_BENCHMARK_BATCHES = None

# Normalization must match the original model training.
NORMALIZE_MEAN = [0.485, 0.456, 0.406]
NORMALIZE_STD = [0.229, 0.224, 0.225]

# DEVICE SELECTION

if torch.backends.mps.is_available():
    device = torch.device("mps")

elif torch.cuda.is_available():
    device = torch.device("cuda")

else:
    device = torch.device("cpu")


def synchronize_device():
    """Ensure asynchronous GPU operations have finished."""

    if device.type == "mps":
        torch.mps.synchronize()

    elif device.type == "cuda":
        torch.cuda.synchronize()


print("=" * 70, flush=True)
print("EUROSAT MODEL EVALUATION", flush=True)
print("=" * 70, flush=True)

print(f"PyTorch version: {torch.__version__}", flush=True)
print(f"MPS available: {torch.backends.mps.is_available()}", flush=True)
print(f"Selected device: {device}", flush=True)
print(f"Batch size: {BATCH_SIZE}", flush=True)
print(f"DataLoader workers: {NUM_WORKERS}", flush=True)

if device.type == "cpu":
    print(
        "\nWARNING: GPU acceleration is unavailable.",
        flush=True,
    )

overall_start = time.perf_counter()

# VALIDATE PROJECT FILES

print("\n[1/8] Checking project files...", flush=True)

required_paths = [
    DATA_DIR,
    SPLIT_DIR,
    MODEL_DIR,
    MODEL_DIR / "small_cnn_best.pth",
    MODEL_DIR / "resnet18_best.pth",
]

for path in required_paths:
    if not path.exists():
        raise FileNotFoundError(
            f"Required file or directory not found: {path}"
        )

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

print("Project files found.", flush=True)

# DEFINE SMALL CNN

class SmallCNN(nn.Module):
    """Architecture matching the original Small CNN script."""

    def __init__(self, num_classes=10):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),

            nn.Linear(
                128 * 8 * 8,
                128,
            ),

            nn.ReLU(),

            nn.Linear(
                128,
                num_classes,
            ),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)

# LOAD AND VALIDATE DATASET

print("\n[2/8] Loading EuroSAT dataset...", flush=True)

dataset_start = time.perf_counter()


transform = transforms.Compose([
    transforms.ToTensor(),

    transforms.Normalize(
        NORMALIZE_MEAN,
        NORMALIZE_STD,
    ),
])


dataset = datasets.ImageFolder(
    DATA_DIR,
    transform=transform,
)

class_names = dataset.classes

if len(dataset) != EXPECTED_DATASET_SIZE:
    raise ValueError(
        f"Expected {EXPECTED_DATASET_SIZE} images; "
        f"found {len(dataset)}."
    )

if len(class_names) != NUM_CLASSES:
    raise ValueError(
        f"Expected {NUM_CLASSES} classes; "
        f"found {len(class_names)}."
    )


# Match split filenames to ImageFolder indices.
lookup = {}

for index, (path, label) in enumerate(dataset.samples):

    filename = Path(path).name

    if filename in lookup:
        raise ValueError(
            f"Duplicate dataset filename: {filename}"
        )

    lookup[filename] = index


def load_split(name):
    """Load filenames from an existing split file."""

    path = SPLIT_DIR / f"eurosat-{name}.txt"

    if not path.is_file():
        raise FileNotFoundError(
            f"Missing split file: {path}"
        )

    with path.open(
        encoding="utf-8"
    ) as handle:

        names = [
            Path(line.strip()).name
            for line in handle
            if line.strip()
        ]

    if len(names) != len(set(names)):
        raise ValueError(
            f"Duplicate filenames in {path}"
        )

    return set(names)


splits = {
    name: load_split(name)
    for name in ("train", "val", "test")
}


# Check split integrity.
if not splits["train"].isdisjoint(splits["val"]):
    raise ValueError("Train/validation overlap detected.")

if not splits["train"].isdisjoint(splits["test"]):
    raise ValueError("Train/test overlap detected.")

if not splits["val"].isdisjoint(splits["test"]):
    raise ValueError("Validation/test overlap detected.")


all_split_names = set.union(*splits.values())

if all_split_names != set(lookup):
    missing = set(lookup) - all_split_names
    extra = all_split_names - set(lookup)

    raise ValueError(
        "Split filenames do not match the dataset. "
        f"Missing: {len(missing)}, Extra: {len(extra)}"
    )


expected_sizes = {
    "train": 16200,
    "val": 5400,
    "test": 5400,
}

for name, expected in expected_sizes.items():

    actual = len(splits[name])

    if actual != expected:
        raise ValueError(
            f"{name} split: expected {expected}, "
            f"found {actual}"
        )


test_indices = sorted(
    lookup[name]
    for name in splits["test"]
)

test_dataset = Subset(
    dataset,
    test_indices,
)


test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=(device.type == "cuda"),
)


dataset_time = time.perf_counter() - dataset_start

print(
    f"Dataset loaded in {dataset_time:.2f} seconds",
    flush=True,
)

print(
    f"Total images: {len(dataset)}",
    flush=True,
)

print(
    f"Test images: {len(test_dataset)}",
    flush=True,
)

print(
    f"Number of batches: {len(test_loader)}",
    flush=True,
)

print(
    f"Classes: {class_names}",
    flush=True,
)

# RESTORE SAVED MODELS

print("\n[3/8] Loading saved checkpoints...", flush=True)

model_start = time.perf_counter()


def restore_weights(model, checkpoint):
    """Safely load a saved state dictionary."""

    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"Missing checkpoint: {checkpoint}"
        )

    try:
        weights = torch.load(
            checkpoint,
            map_location="cpu",
            weights_only=True,
        )

    except TypeError as exc:
        raise RuntimeError(
            "Your PyTorch installation does not support "
            "weights_only=True. Please upgrade PyTorch."
        ) from exc

    model.load_state_dict(
        weights,
        strict=True,
    )

    model = model.to(device)
    model.eval()

    return model


small = restore_weights(
    SmallCNN(num_classes=NUM_CLASSES),
    MODEL_DIR / "small_cnn_best.pth",
)

# weights=None avoids downloading pretrained weights.
# The trained checkpoint supplies the model weights.
resnet = models.resnet18(weights=None)

resnet.fc = nn.Linear(
    resnet.fc.in_features,
    NUM_CLASSES,
)

resnet = restore_weights(
    resnet,
    MODEL_DIR / "resnet18_best.pth",
)


synchronize_device()

model_time = time.perf_counter() - model_start

print(
    f"Both checkpoints loaded in {model_time:.2f} seconds",
    flush=True,
)

print("Small CNN: ready", flush=True)
print("ResNet-18: ready", flush=True)

# RUN INFERENCE ON BOTH MODELS

print("\n[4/8] Evaluating models...", flush=True)

print(
    "Both networks will evaluate the same test images.",
    flush=True,
)

inference_start = time.perf_counter()

true_labels = []
small_predictions = []
resnet_predictions = []

small_confidences = []
resnet_confidences = []

# Measure approximately how much time is spent loading
# batches versus performing inference and CPU transfers.
data_loading_seconds = 0.0
model_processing_seconds = 0.0

batch_count = 0

previous_batch_end = time.perf_counter()


with torch.inference_mode():

    for batch_index, (images, labels) in enumerate(
        tqdm(
            test_loader,
            desc="EuroSAT inference",
            total=(
                len(test_loader)
                if MAX_BENCHMARK_BATCHES is None
                else min(
                    len(test_loader),
                    MAX_BENCHMARK_BATCHES,
                )
            ),
            unit="batch",
        )
    ):

        loading_end = time.perf_counter()

        data_loading_seconds += (
            loading_end - previous_batch_end
        )

        batch_start = time.perf_counter()

        # Transfer the current batch to the selected device.
        images = images.to(device)

        # Small CNN inference.
        small_logits = small(images)

        # ResNet-18 inference.
        resnet_logits = resnet(images)

        # Convert logits into confidence scores.
        small_probs = torch.softmax(
            small_logits,
            dim=1,
        )

        resnet_probs = torch.softmax(
            resnet_logits,
            dim=1,
        )

        # Extract predicted class and confidence.
        s_conf, s_pred = small_probs.max(dim=1)
        r_conf, r_pred = resnet_probs.max(dim=1)

        # Combine values to reduce separate GPU/CPU transfers.
        small_results = torch.stack(
            (s_pred.float(), s_conf),
            dim=1,
        ).cpu().numpy()

        resnet_results = torch.stack(
            (r_pred.float(), r_conf),
            dim=1,
        ).cpu().numpy()

        true_labels.extend(
            labels.tolist()
        )

        small_predictions.extend(
            small_results[:, 0].astype(int).tolist()
        )

        resnet_predictions.extend(
            resnet_results[:, 0].astype(int).tolist()
        )

        small_confidences.extend(
            small_results[:, 1].tolist()
        )

        resnet_confidences.extend(
            resnet_results[:, 1].tolist()
        )

        synchronize_device()

        model_processing_seconds += (
            time.perf_counter() - batch_start
        )

        batch_count += 1

        previous_batch_end = time.perf_counter()

        if (
            MAX_BENCHMARK_BATCHES is not None
            and batch_count >= MAX_BENCHMARK_BATCHES
        ):
            break


synchronize_device()

inference_time = (
    time.perf_counter() - inference_start
)

# CONVERT AND VALIDATE PREDICTIONS

truth = np.asarray(
    true_labels,
    dtype=np.int64,
)

sp = np.asarray(
    small_predictions,
    dtype=np.int64,
)

rp = np.asarray(
    resnet_predictions,
    dtype=np.int64,
)


if MAX_BENCHMARK_BATCHES is not None:

    print("\nBENCHMARK MODE COMPLETED", flush=True)

    print(
        f"Processed images: {len(truth)}",
        flush=True,
    )

    print(
        f"Elapsed inference time: {inference_time:.2f} s",
        flush=True,
    )

    if inference_time > 0:
        throughput = len(truth) / inference_time

        estimated_full_time = (
            EXPECTED_TEST_SIZE / throughput
        )

        print(
            f"Measured throughput: {throughput:.2f} images/s",
            flush=True,
        )

        print(
            "Estimated full inference time: "
            f"{estimated_full_time / 60:.2f} minutes",
            flush=True,
        )

    print(
        "Benchmark mode does not generate final metrics "
        "or overwrite analysis outputs.",
        flush=True,
    )

    raise SystemExit(0)


if len(truth) != EXPECTED_TEST_SIZE:
    raise ValueError(
        f"Expected {EXPECTED_TEST_SIZE} predictions, "
        f"received {len(truth)}."
    )


expected_truth = np.asarray([
    dataset.targets[index]
    for index in test_indices
])

if not np.array_equal(
    truth,
    expected_truth,
):
    raise ValueError(
        "Prediction order does not match the test dataset."
    )


print("\nInference finished.", flush=True)

print(
    f"Total inference time: {inference_time:.2f} seconds",
    flush=True,
)

print(
    f"Image-loading/wait time: "
    f"{data_loading_seconds:.2f} seconds",
    flush=True,
)

print(
    f"Model processing and transfer time: "
    f"{model_processing_seconds:.2f} seconds",
    flush=True,
)

print(
    f"Throughput: {len(truth)/inference_time:.2f} images/sec",
    flush=True,
)

# SAVE IMAGE-LEVEL PREDICTIONS

print(
    "\n[5/8] Saving image-level predictions...",
    flush=True,
)

paths = [
    dataset.samples[index][0]
    for index in test_indices
]


rows = pd.DataFrame({

    "filename": [
        Path(path).name
        for path in paths
    ],

    "filepath": paths,

    "true_class": [
        class_names[index]
        for index in truth
    ],

    "small_cnn_prediction": [
        class_names[index]
        for index in sp
    ],

    "resnet18_prediction": [
        class_names[index]
        for index in rp
    ],

    "small_cnn_confidence": small_confidences,

    "resnet18_confidence": resnet_confidences,

    "small_cnn_correct": sp == truth,

    "resnet18_correct": rp == truth,
})


rows["both_wrong"] = (
    (~rows["small_cnn_correct"])
    & (~rows["resnet18_correct"])
)

rows["small_wrong_resnet_correct"] = (
    (~rows["small_cnn_correct"])
    & rows["resnet18_correct"]
)

rows["small_correct_resnet_wrong"] = (
    rows["small_cnn_correct"]
    & (~rows["resnet18_correct"])
)


prediction_file = (
    OUTPUT_DIR / "eurosat_prediction_analysis.csv"
)

rows.to_csv(
    prediction_file,
    index=False,
)

print(
    f"Saved: {prediction_file.name}",
    flush=True,
)

# CLASS METRICS AND CONFUSION MATRICES

print(
    "\n[6/8] Calculating classification metrics...",
    flush=True,
)

summary = []


for name, prefix, pred in (

    ("Small CNN", "small_cnn", sp),

    ("ResNet-18", "resnet18", rp),

):

    report = classification_report(
        truth,
        pred,
        labels=list(range(NUM_CLASSES)),
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    metrics_file = (
        OUTPUT_DIR / f"{prefix}_class_metrics.csv"
    )

    pd.DataFrame(report).T.to_csv(
        metrics_file
    )


    cm = confusion_matrix(
        truth,
        pred,
        labels=list(range(NUM_CLASSES)),
    )

    confusion_file = (
        OUTPUT_DIR / f"{prefix}_confusion_matrix.csv"
    )

    pd.DataFrame(
        cm,
        index=class_names,
        columns=class_names,
    ).to_csv(
        confusion_file
    )


    # Record all nonzero off-diagonal confusion pairs.
    for true_i in range(NUM_CLASSES):

        for pred_i in range(NUM_CLASSES):

            if (
                true_i != pred_i
                and cm[true_i, pred_i] > 0
            ):

                summary.append({

                    "model": name,

                    "true_class": class_names[true_i],

                    "predicted_class": class_names[pred_i],

                    "count": int(
                        cm[true_i, pred_i]
                    ),
                })


    accuracy = (
        pred == truth
    ).mean()

    print(
        f"{name} test accuracy: {accuracy:.4%}",
        flush=True,
    )


confusion_pairs = pd.DataFrame(
    summary,
    columns=[
        "model",
        "true_class",
        "predicted_class",
        "count",
    ],
)

confusion_pairs = confusion_pairs.sort_values(
    ["model", "count"],
    ascending=[True, False],
)

confusion_pairs.to_csv(
    OUTPUT_DIR / "confusion_pair_counts.csv",
    index=False,
)

# CLASS NAME MATCHING

def find_class(fragment):
    """Find the exact EuroSAT class after normalization."""

    normalized_fragment = (
        fragment
        .replace(" ", "")
        .replace("_", "")
        .lower()
    )

    matches = []

    for name in class_names:

        normalized_name = (
            name
            .replace(" ", "")
            .replace("_", "")
            .lower()
        )

        if normalized_name == normalized_fragment:
            matches.append(name)

    if len(matches) != 1:
        raise ValueError(
            f"Cannot uniquely match {fragment}. "
            f"Available classes: {class_names}"
        )

    return matches[0]

# IMAGE GRID FUNCTION

def image_grid(
    selection,
    filename,
    title,
    max_images=MAX_IMAGES_PER_FIGURE,
):
    """Save original RGB images with both models' predictions."""

    selection = selection.head(
        max_images
    )

    if selection.empty:

        print(
            f"No images available for: {title}",
            flush=True,
        )

        return


    columns = 4

    nrows = math.ceil(
        len(selection) / columns
    )


    fig, axes = plt.subplots(
        nrows,
        columns,
        figsize=(
            12,
            3.1 * nrows,
        ),
        squeeze=False,
    )


    for ax in axes.flat:
        ax.axis("off")


    for ax, (_, row) in zip(
        axes.flat,
        selection.iterrows(),
    ):

        with Image.open(
            row["filepath"]
        ) as image:

            rgb_image = image.convert("RGB")
            ax.imshow(rgb_image)


        ax.set_title(

            f"True: {row['true_class']}\n"

            f"Small: {row['small_cnn_prediction']} "
            f"({row['small_cnn_confidence']:.2f})\n"

            f"ResNet: {row['resnet18_prediction']} "
            f"({row['resnet18_confidence']:.2f})",

            fontsize=8,
        )

        ax.axis("off")


    fig.suptitle(
        title,
        fontsize=12,
    )

    fig.tight_layout()

    figure_path = (
        OUTPUT_DIR / filename
    )

    fig.savefig(
        figure_path,
        dpi=FIGURE_DPI,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(
        f"Saved: {filename}",
        flush=True,
    )

# VISUALIZE SPECIFIC CONFUSION PAIRS

print(
    "\n[7/8] Generating class-specific error figures...",
    flush=True,
)

figure_start = time.perf_counter()


target_confusions = (

    (
        "PermanentCrop",
        "HerbaceousVegetation",
    ),

    (
        "River",
        "Highway",
    ),

)


for true_fragment, predicted_fragment in target_confusions:

    actual = find_class(
        true_fragment
    )

    predicted = find_class(
        predicted_fragment
    )


    for model_name, column in (

        (
            "small_cnn",
            "small_cnn_prediction",
        ),

        (
            "resnet18",
            "resnet18_prediction",
        ),

    ):

        chosen = rows[
            (rows["true_class"] == actual)
            & (rows[column] == predicted)
        ]


        filename = (
            f"{model_name}_"
            f"{true_fragment}_to_"
            f"{predicted_fragment}.png"
        )


        image_grid(
            chosen,
            filename,
            f"{model_name}: {actual} → {predicted}",
        )

# VISUALIZE MODEL DISAGREEMENTS

print(
    "\nGenerating model disagreement figures...",
    flush=True,
)


comparison_flags = (

    "both_wrong",

    "small_wrong_resnet_correct",

    "small_correct_resnet_wrong",

)


for flag in comparison_flags:

    selected = rows[
        rows[flag]
    ]

    image_grid(
        selected,
        f"{flag}_examples.png",
        flag.replace("_", " ").title(),
    )


figure_time = (
    time.perf_counter() - figure_start
)

print(
    f"\nFigure generation time: {figure_time:.2f} seconds",
    flush=True,
)

# FINAL COMPARISON SUMMARY

print(
    "\n[8/8] Final evaluation summary...",
    flush=True,
)


small_accuracy = float(
    (sp == truth).mean()
)

resnet_accuracy = float(
    (rp == truth).mean()
)


print("\n" + "=" * 70, flush=True)

print(
    "EUROSAT FINAL TEST RESULTS",
    flush=True,
)

print("=" * 70, flush=True)

print(
    f"Small CNN accuracy: {small_accuracy:.4%}",
    flush=True,
)

print(
    f"ResNet-18 accuracy: {resnet_accuracy:.4%}",
    flush=True,
)

print(
    f"Accuracy difference: "
    f"{(resnet_accuracy - small_accuracy)*100:.2f} "
    f"percentage points",
    flush=True,
)


print("\nError comparison:", flush=True)

for flag in comparison_flags:

    count = int(
        rows[flag].sum()
    )

    print(
        f"  {flag}: {count}",
        flush=True,
    )


print(
    "\nIndividual model errors:",
    flush=True,
)

print(
    f"Small CNN errors: {int((sp != truth).sum())}",
    flush=True,
)

print(
    f"ResNet-18 errors: {int((rp != truth).sum())}",
    flush=True,
)

# SAVE PERFORMANCE REPORT

overall_time = (
    time.perf_counter() - overall_start
)


performance_report = pd.DataFrame([{

    "device": str(device),

    "batch_size": BATCH_SIZE,

    "num_workers": NUM_WORKERS,

    "test_images": len(truth),

    "dataset_loading_seconds": dataset_time,

    "checkpoint_loading_seconds": model_time,

    "inference_seconds": inference_time,

    "data_loading_wait_seconds": data_loading_seconds,

    "model_processing_seconds": model_processing_seconds,

    "figure_generation_seconds": figure_time,

    "total_runtime_seconds": overall_time,

    "images_per_second": (
        len(truth) / inference_time
    ),

    "small_cnn_test_accuracy": small_accuracy,

    "resnet18_test_accuracy": resnet_accuracy,

}])


performance_report.to_csv(
    OUTPUT_DIR / "evaluation_performance_report.csv",
    index=False,
)


print(
    "\n" + "=" * 70,
    flush=True,
)

print(
    "PERFORMANCE REPORT",
    flush=True,
)

print(
    "=" * 70,
    flush=True,
)

print(
    f"Device: {device}",
    flush=True,
)

print(
    f"Dataset loading: {dataset_time:.2f} seconds",
    flush=True,
)

print(
    f"Checkpoint loading: {model_time:.2f} seconds",
    flush=True,
)

print(
    f"Inference: {inference_time:.2f} seconds",
    flush=True,
)

print(
    f"Figure generation: {figure_time:.2f} seconds",
    flush=True,
)

print(
    f"Total runtime: {overall_time:.2f} seconds",
    flush=True,
)

print(
    f"Total runtime: {overall_time/60:.2f} minutes",
    flush=True,
)


print(
    "\nAll evaluation outputs saved to:",
    flush=True,
)

print(
    OUTPUT_DIR,
    flush=True,
)

print(
    "\nNOTE: Softmax confidence scores are not "
    "necessarily calibrated probabilities of correctness.",
    flush=True,
)

print(
    "\nEvaluation complete. No models were retrained "
    "or modified.",
    flush=True,
)
