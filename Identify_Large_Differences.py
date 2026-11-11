from pathlib import Path
import pandas as pd

PROJECT_DIR = Path(
    "/Users/zarrintasneem/Documents/ESS5480_Project"
)

RESULTS_DIR = PROJECT_DIR / "results" / "final_test_comparison"

df = pd.read_csv(
    RESULTS_DIR / "classwise_test_comparison.csv"
)

metrics = [
    "Precision",
    "Recall",
    "F1"
]

for metric in metrics:

    column = f"{metric}_Difference_pp"

    print("\n" + "=" * 65)
    print(f"{metric.upper()} COMPARISON")
    print("=" * 65)

    ranked = df[
        ["Class", column]
    ].sort_values(
        column,
        ascending=False
    )

    print("\nLargest improvements:")
    print(
        ranked.head(5).round(2).to_string(index=False)
    )

    print("\nLargest declines or smallest improvements:")
    print(
        ranked.tail(5).round(2).to_string(index=False)
    )


print("\n" + "=" * 65)
print("CLASSES WHERE SMALL CNN OUTPERFORMS RESNET-18")
print("=" * 65)

for metric in metrics:

    column = f"{metric}_Difference_pp"

    worse = df[df[column] < 0][
        ["Class", column]
    ]

    print(f"\n{metric}:")

    if worse.empty:
        print("No classes.")
    else:
        print(
            worse.round(2).to_string(index=False)
        )


print("\n" + "=" * 65)
print("LOWEST RESNET-18 F1-SCORES")
print("=" * 65)

print(
    df.nsmallest(5, "ResNet18_F1")[
        [
            "Class",
            "Support",
            "ResNet18_Precision",
            "ResNet18_Recall",
            "ResNet18_F1"
        ]
    ].round(4).to_string(index=False)
)