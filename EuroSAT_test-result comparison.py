
"""EuroSAT final test-result comparison.

Compares saved Small CNN and ResNet-18 classification metrics.
No training or model inference is performed.

Inputs:
    small_cnn_class_metrics.csv
    resnet18_class_metrics.csv
    confusion_pair_counts.csv

Outputs:
    final_test_model_comparison.csv
    classwise_test_comparison.csv
    largest_f1_improvements.csv
    lowest_resnet18_f1_classes.csv
    top_confusion_pairs.csv
    classwise_f1_comparison.png
    classwise_f1_improvement.png
    classwise_precision_recall_comparison.png
    top_confusion_pairs.png
    test_results_summary.txt
"""

from pathlib import Path
import time

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ============================================================
# 1. CONFIGURATION
# ============================================================

PROJECT_DIR = Path(
    "/Users/zarrintasneem/Documents/ESS5480_Project"
)

INPUT_DIR = (
    PROJECT_DIR / "results" / "visual_error_analysis"
)

OUTPUT_DIR = (
    PROJECT_DIR / "results" / "final_test_comparison"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SMALL_FILE = INPUT_DIR / "small_cnn_class_metrics.csv"
RESNET_FILE = INPUT_DIR / "resnet18_class_metrics.csv"
CONFUSION_FILE = INPUT_DIR / "confusion_pair_counts.csv"

EXPECTED_CLASSES = 10

plt.rcParams.update({
    "figure.dpi": 120,
    "savefig.dpi": 200,
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

start_time = time.perf_counter()


# ============================================================
# 2. VALIDATE INPUTS
# ============================================================

print("\n" + "=" * 75)
print("EUROSAT FINAL TEST COMPARISON")
print("=" * 75)

for path in (SMALL_FILE, RESNET_FILE, CONFUSION_FILE):
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing required input:\n{path}\n"
            "Run the saved-model evaluation script first."
        )

print("\nAll required input files found.")


# ============================================================
# 3. LOAD CLASSIFICATION REPORTS
# ============================================================

print("\n[1/7] Loading classification reports...")

small = pd.read_csv(SMALL_FILE, index_col=0)
resnet = pd.read_csv(RESNET_FILE, index_col=0)

required_columns = [
    "precision",
    "recall",
    "f1-score",
    "support",
]

for name, df in (("Small CNN", small), ("ResNet-18", resnet)):
    missing = set(required_columns) - set(df.columns)

    if missing:
        raise ValueError(
            f"{name} report is missing columns: {sorted(missing)}"
        )

    df.index = df.index.astype(str).str.strip()

    if df.index.duplicated().any():
        raise ValueError(
            f"{name} report contains duplicate row labels."
        )


summary_labels = {
    "accuracy",
    "macro avg",
    "weighted avg",
    "micro avg",
    "samples avg",
}

small_classes = [
    name for name in small.index
    if name not in summary_labels
]

resnet_classes = [
    name for name in resnet.index
    if name not in summary_labels
]

if len(small_classes) != EXPECTED_CLASSES:
    raise ValueError(
        f"Expected {EXPECTED_CLASSES} Small CNN classes; "
        f"found {len(small_classes)}."
    )

if set(small_classes) != set(resnet_classes):
    raise ValueError(
        "Class names do not match between model reports."
    )

class_names = small_classes

small_support = pd.to_numeric(
    small.loc[class_names, "support"]
)

resnet_support = pd.to_numeric(
    resnet.loc[class_names, "support"]
)

if not np.array_equal(
    small_support.to_numpy(),
    resnet_support.to_numpy()
):
    raise ValueError(
        "Class support differs between the two reports. "
        "Both models must use the same test set."
    )

test_size = int(small_support.sum())

print(f"Number of classes: {len(class_names)}")
print(f"Test samples: {test_size}")


# ============================================================
# 4. COMPARE OVERALL MODEL PERFORMANCE
# ============================================================

print("\n[2/7] Comparing overall test performance...")


def metric_value(df, row, column):
    if row not in df.index:
        raise ValueError(f"Missing report row: {row}")

    return float(df.loc[row, column])


def accuracy_value(df):
    return metric_value(df, "accuracy", "f1-score")


metrics = [
    (
        "Accuracy",
        accuracy_value(small),
        accuracy_value(resnet),
    ),
    (
        "Macro precision",
        metric_value(small, "macro avg", "precision"),
        metric_value(resnet, "macro avg", "precision"),
    ),
    (
        "Macro recall",
        metric_value(small, "macro avg", "recall"),
        metric_value(resnet, "macro avg", "recall"),
    ),
    (
        "Macro F1",
        metric_value(small, "macro avg", "f1-score"),
        metric_value(resnet, "macro avg", "f1-score"),
    ),
    (
        "Weighted precision",
        metric_value(small, "weighted avg", "precision"),
        metric_value(resnet, "weighted avg", "precision"),
    ),
    (
        "Weighted recall",
        metric_value(small, "weighted avg", "recall"),
        metric_value(resnet, "weighted avg", "recall"),
    ),
    (
        "Weighted F1",
        metric_value(small, "weighted avg", "f1-score"),
        metric_value(resnet, "weighted avg", "f1-score"),
    ),
]

overall = pd.DataFrame(
    metrics,
    columns=[
        "Metric",
        "Small_CNN",
        "ResNet18",
    ],
)

overall["Difference_pp"] = (
    overall["ResNet18"] - overall["Small_CNN"]
) * 100

overall["Small_CNN_percent"] = overall["Small_CNN"] * 100
overall["ResNet18_percent"] = overall["ResNet18"] * 100

overall.to_csv(
    OUTPUT_DIR / "final_test_model_comparison.csv",
    index=False,
)

print("\nOVERALL TEST PERFORMANCE")
print(
    overall[
        [
            "Metric",
            "Small_CNN_percent",
            "ResNet18_percent",
            "Difference_pp",
        ]
    ].round(2).to_string(index=False)
)


# ============================================================
# 5. CLASSWISE PRECISION, RECALL, AND F1
# ============================================================

print("\n[3/7] Comparing all 10 land-cover classes...")

rows = []

for name in class_names:
    row = {
        "Class": name,
        "Support": int(small.loc[name, "support"]),
    }

    for metric, label in (
        ("precision", "Precision"),
        ("recall", "Recall"),
        ("f1-score", "F1"),
    ):
        s = float(small.loc[name, metric])
        r = float(resnet.loc[name, metric])

        row[f"Small_CNN_{label}"] = s
        row[f"ResNet18_{label}"] = r
        row[f"{label}_Difference_pp"] = (r - s) * 100

    rows.append(row)

comparison = pd.DataFrame(rows)

comparison = comparison.sort_values(
    "F1_Difference_pp",
    ascending=False,
)

comparison.to_csv(
    OUTPUT_DIR / "classwise_test_comparison.csv",
    index=False,
)

print("\nCLASSWISE F1 COMPARISON")

display = comparison[
    [
        "Class",
        "Support",
        "Small_CNN_F1",
        "ResNet18_F1",
        "F1_Difference_pp",
    ]
].copy()

display["Small_CNN_F1"] *= 100
display["ResNet18_F1"] *= 100

print(display.round(2).to_string(index=False))


# ============================================================
# 6. IDENTIFY IMPROVEMENTS AND WEAK CLASSES
# ============================================================

print("\n[4/7] Identifying key findings...")

largest_improvements = comparison.nlargest(
    5,
    "F1_Difference_pp",
)

lowest_resnet = comparison.nsmallest(
    5,
    "ResNet18_F1",
)

largest_improvements.to_csv(
    OUTPUT_DIR / "largest_f1_improvements.csv",
    index=False,
)

lowest_resnet.to_csv(
    OUTPUT_DIR / "lowest_resnet18_f1_classes.csv",
    index=False,
)

print("\nFIVE LARGEST F1 CHANGES (RESNET-18 MINUS SMALL CNN)")

for _, row in largest_improvements.iterrows():
    print(
        f"{row['Class']}: "
        f"{row['F1_Difference_pp']:+.2f} percentage points"
    )

print("\nFIVE LOWEST RESNET-18 F1-SCORES")

for _, row in lowest_resnet.iterrows():
    print(
        f"{row['Class']}: "
        f"F1 = {row['ResNet18_F1']:.4f}, "
        f"Recall = {row['ResNet18_Recall']:.4f}"
    )


# ============================================================
# 7. ANALYZE CONFUSION PAIRS
# ============================================================

print("\n[5/7] Analyzing classification confusion pairs...")

confusions = pd.read_csv(CONFUSION_FILE)

required = {
    "model",
    "true_class",
    "predicted_class",
    "count",
}

if not required.issubset(confusions.columns):
    raise ValueError(
        "Confusion-pair CSV is missing required columns."
    )

confusions["count"] = pd.to_numeric(
    confusions["count"],
    errors="raise",
)

confusions = confusions.sort_values(
    ["model", "count"],
    ascending=[True, False],
)

top_confusions = (
    confusions.groupby("model", group_keys=False)
    .head(10)
    .copy()
)

top_confusions.to_csv(
    OUTPUT_DIR / "top_confusion_pairs.csv",
    index=False,
)

for model_name in ("Small CNN", "ResNet-18"):
    subset = confusions[
        confusions["model"] == model_name
    ].head(5)

    print(f"\nTOP ERRORS: {model_name}")

    for _, row in subset.iterrows():
        print(
            f"{row['true_class']} → "
            f"{row['predicted_class']}: "
            f"{int(row['count'])} images"
        )


# ============================================================
# 8. GENERATE FIGURES
# ============================================================

print("\n[6/7] Generating comparison figures...")

# Figure 1: Classwise F1 comparison
ordered = comparison.sort_values("ResNet18_F1")

y = np.arange(len(ordered))
height = 0.38

fig, ax = plt.subplots(figsize=(11, 7))

ax.barh(
    y - height / 2,
    ordered["Small_CNN_F1"],
    height,
    label="Small CNN",
    color="#D79B49",
)

ax.barh(
    y + height / 2,
    ordered["ResNet18_F1"],
    height,
    label="ResNet-18",
    color="#3878B8",
)

ax.set_yticks(y)
ax.set_yticklabels(ordered["Class"])
ax.set_xlim(0, 1.05)
ax.set_xlabel("F1-score")
ax.set_title("EuroSAT: Classwise Test F1 Comparison")
ax.legend()

fig.tight_layout()
fig.savefig(
    OUTPUT_DIR / "classwise_f1_comparison.png",
    bbox_inches="tight",
)
plt.close(fig)


# Figure 2: F1 difference in percentage points
ordered = comparison.sort_values("F1_Difference_pp")

colors = [
    "#27865B" if value >= 0 else "#C84C4C"
    for value in ordered["F1_Difference_pp"]
]

fig, ax = plt.subplots(figsize=(11, 7))

ax.barh(
    ordered["Class"],
    ordered["F1_Difference_pp"],
    color=colors,
)

ax.axvline(0, color="black", linewidth=1)
ax.set_xlabel("ResNet-18 minus Small CNN (percentage points)")
ax.set_title("EuroSAT: Change in Classwise F1")

fig.tight_layout()
fig.savefig(
    OUTPUT_DIR / "classwise_f1_improvement.png",
    bbox_inches="tight",
)
plt.close(fig)


# Figure 3: Precision and recall comparisons
ordered = comparison.sort_values("Class")

x = np.arange(len(ordered))
width = 0.19

fig, ax = plt.subplots(figsize=(14, 7))

series = [
    ("Small CNN precision", "Small_CNN_Precision", "#E9B96E"),
    ("ResNet-18 precision", "ResNet18_Precision", "#346FA1"),
    ("Small CNN recall", "Small_CNN_Recall", "#C87828"),
    ("ResNet-18 recall", "ResNet18_Recall", "#73A7D4"),
]

for i, (label, column, color) in enumerate(series):
    ax.bar(
        x + (i - 1.5) * width,
        ordered[column],
        width,
        label=label,
        color=color,
    )

ax.set_xticks(x)
ax.set_xticklabels(
    ordered["Class"],
    rotation=45,
    ha="right",
)

ax.set_ylim(0, 1.05)
ax.set_ylabel("Score")
ax.set_title("EuroSAT: Classwise Precision and Recall")
ax.legend(ncol=2)

fig.tight_layout()
fig.savefig(
    OUTPUT_DIR / "classwise_precision_recall_comparison.png",
    bbox_inches="tight",
)
plt.close(fig)


# Figure 4: Top five confusion pairs per model
fig, axes = plt.subplots(
    1,
    2,
    figsize=(16, 7),
)

for ax, model_name in zip(
    axes,
    ("Small CNN", "ResNet-18"),
):
    subset = confusions[
        confusions["model"] == model_name
    ].head(5).copy()

    subset = subset.iloc[::-1]

    labels = [
        f"{a} → {b}"
        for a, b in zip(
            subset["true_class"],
            subset["predicted_class"],
        )
    ]

    ax.barh(
        labels,
        subset["count"],
        color=(
            "#D79B49"
            if model_name == "Small CNN"
            else "#3878B8"
        ),
    )

    ax.set_title(model_name)
    ax.set_xlabel("Misclassified images")

fig.suptitle("Most Frequent EuroSAT Confusion Pairs")
fig.tight_layout()

fig.savefig(
    OUTPUT_DIR / "top_confusion_pairs.png",
    bbox_inches="tight",
)
plt.close(fig)


# ============================================================
# 9. WRITE RESEARCH SUMMARY
# ============================================================

print("\n[7/7] Writing research summary...")

small_acc = accuracy_value(small)
resnet_acc = accuracy_value(resnet)

best_gain = comparison.loc[
    comparison["F1_Difference_pp"].idxmax()
]

worst_class = comparison.loc[
    comparison["ResNet18_F1"].idxmin()
]

summary_lines = [
    "EUROSAT FINAL TEST COMPARISON",
    "=" * 65,
    "",
    f"Test set size: {test_size}",
    f"Number of classes: {len(class_names)}",
    "",
    "OVERALL RESULTS",
    f"Small CNN accuracy: {small_acc:.4%}",
    f"ResNet-18 accuracy: {resnet_acc:.4%}",
    (
        "Accuracy difference: "
        f"{(resnet_acc - small_acc) * 100:+.2f} percentage points"
    ),
    "",
    "LARGEST CLASSWISE F1 IMPROVEMENT",
    f"Class: {best_gain['Class']}",
    (
        "F1 change: "
        f"{best_gain['F1_Difference_pp']:+.2f} percentage points"
    ),
    "",
    "LOWEST RESNET-18 CLASSWISE F1",
    f"Class: {worst_class['Class']}",
    f"ResNet-18 F1: {worst_class['ResNet18_F1']:.4f}",
    "",
    "INTERPRETATION NOTES",
    "- Scores are calculated on the existing held-out test split.",
    "- Positive differences favour ResNet-18.",
    "- Confusion counts are directional (true class to predicted class).",
    "- The comparison combines architecture and pretraining effects.",
    "- Observed performance differences do not establish causality.",
    "- A single test split does not establish statistical significance.",
]

summary_text = "\n".join(summary_lines)

(OUTPUT_DIR / "test_results_summary.txt").write_text(
    summary_text,
    encoding="utf-8",
)

print("\n" + summary_text)


# ============================================================
# 10. FINISH
# ============================================================

elapsed = time.perf_counter() - start_time

print("\n" + "=" * 75)
print("COMPARISON COMPLETED")
print("=" * 75)

print(f"Runtime: {elapsed:.2f} seconds")
print(f"Outputs saved to:\n{OUTPUT_DIR}")
print("\nNo models were retrained or modified.")
