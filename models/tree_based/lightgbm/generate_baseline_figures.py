"""Generate presentation figures from saved LightGBM baseline outputs."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


OUTPUT_DIR = Path(__file__).parent / "figures" / "baseline"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BLUE = "#2563EB"
GREEN = "#16A34A"
ORANGE = "#F59E0B"
RED = "#DC2626"
PURPLE = "#7C3AED"
SLATE = "#475569"

FEATURE_IMPORTANCE = {
    "Dept": 12581,
    "Store": 4987,
    "Size": 3618,
    "CPI": 2487,
    "Unemployment": 2062,
    "Temperature": 1646,
    "Fuel_Price": 1243,
    "Type": 1029,
    "MarkDown3": 641,
    "IsHoliday": 346,
    "MarkDown2": 290,
    "MarkDown4": 201,
    "MarkDown1": 192,
    "MarkDown5": 177,
}

ROUNDS = np.arange(50, 501, 50)
TRAIN_L1 = np.array(
    [5929.85, 4701.74, 4260.91, 3950.84, 3726.14,
     3552.86, 3444.44, 3358.04, 3273.90, 3178.69]
)
VALID_L1 = np.array(
    [5563.04, 4349.25, 4018.80, 3759.19, 3587.61,
     3461.04, 3389.69, 3330.70, 3266.29, 3188.88]
)


def finish(fig: plt.Figure, filename: str) -> None:
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / filename, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def label_bars(ax, bars, decimals=0) -> None:
    for bar in bars:
        value = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value,
            f"{value:,.{decimals}f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )


def feature_importance_plot() -> None:
    items = sorted(FEATURE_IMPORTANCE.items(), key=lambda item: item[1])
    names, values = zip(*items)
    fig, ax = plt.subplots(figsize=(11, 7.5))
    ax.barh(names, values, color=BLUE)
    ax.set_title("LightGBM baseline — feature importance", fontsize=17, weight="bold")
    ax.set_xlabel("Split importance (number of tree splits)")
    ax.grid(axis="x", alpha=0.2)
    finish(fig, "01_feature_importance.png")


def feature_family_plot() -> None:
    families = {
        "Identity": ["Store", "Dept"],
        "Store metadata": ["Type", "Size"],
        "External": ["Temperature", "Fuel_Price", "CPI", "Unemployment"],
        "Markdown": ["MarkDown1", "MarkDown2", "MarkDown3", "MarkDown4", "MarkDown5"],
        "Holiday": ["IsHoliday"],
    }
    values = [sum(FEATURE_IMPORTANCE[name] for name in names) for names in families.values()]
    fig, ax = plt.subplots(figsize=(10.5, 6))
    bars = ax.bar(families.keys(), values, color=[BLUE, GREEN, ORANGE, PURPLE, RED])
    ax.set_title("Where LightGBM baseline splits came from", fontsize=17, weight="bold")
    ax.set_ylabel("Summed split importance")
    ax.set_ylim(0, max(values) * 1.18)
    ax.grid(axis="y", alpha=0.2)
    label_bars(ax, bars)
    finish(fig, "02_feature_family_importance.png")


def baseline_to_final_plot() -> None:
    values = [3184.2771, 1575.15]
    improvement = 100 * (values[0] - values[1]) / values[0]
    fig, ax = plt.subplots(figsize=(8.5, 6))
    bars = ax.bar(["Raw baseline", "Final engineered"], values, color=[SLATE, GREEN], width=0.58)
    ax.set_title("32-week validation WMAE evolution", fontsize=17, weight="bold")
    ax.set_ylabel("WMAE — lower is better")
    ax.set_ylim(0, 3550)
    ax.grid(axis="y", alpha=0.2)
    label_bars(ax, bars, decimals=1)
    ax.annotate(
        f"{improvement:.1f}% lower WMAE",
        xy=(1, values[1]), xytext=(0.5, 3000), ha="center", color=GREEN, weight="bold",
        arrowprops={"arrowstyle": "->", "color": GREEN, "lw": 2},
    )
    finish(fig, "03_baseline_vs_final_engineered.png")


def metrics_plot() -> None:
    labels = ["WMAE", "MAE", "RMSE"]
    values = [3184.2771, 3156.0605, 6404.4248]
    fig, ax = plt.subplots(figsize=(9, 6))
    bars = ax.bar(labels, values, color=[BLUE, GREEN, SLATE], width=0.58)
    ax.set_title("LightGBM baseline — validation metrics", fontsize=17, weight="bold")
    ax.set_ylabel("Error — lower is better")
    ax.set_ylim(0, 7100)
    ax.grid(axis="y", alpha=0.2)
    label_bars(ax, bars, decimals=0)
    finish(fig, "04_validation_metrics.png")


def learning_curve_plot() -> None:
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(ROUNDS, TRAIN_L1, marker="o", color=BLUE, lw=2.5, label="Training weighted L1")
    ax.plot(ROUNDS, VALID_L1, marker="o", color=ORANGE, lw=2.5, label="Validation weighted L1")
    ax.scatter([500], [3188.88], color=RED, zorder=5)
    ax.annotate(
        "Validation still improving\nat the final 500th tree",
        xy=(500, 3188.88), xytext=(310, 4500),
        arrowprops={"arrowstyle": "->", "color": RED}, color=RED,
    )
    ax.set_title("LightGBM baseline training dynamics", fontsize=17, weight="bold")
    ax.set_xlabel("Boosting iteration")
    ax.set_ylabel("Weighted L1 / MAE")
    ax.grid(alpha=0.2)
    ax.legend()
    finish(fig, "05_learning_curve.png")


def train_validation_gap_plot() -> None:
    gap = VALID_L1 - TRAIN_L1
    fig, ax = plt.subplots(figsize=(11, 5.5))
    colors = [GREEN if value <= 0 else RED for value in gap]
    bars = ax.bar(ROUNDS, gap, width=32, color=colors)
    ax.axhline(0, color=SLATE, lw=1)
    ax.set_title("Validation L1 − training L1", fontsize=17, weight="bold")
    ax.set_xlabel("Boosting iteration")
    ax.set_ylabel("Generalization gap")
    ax.grid(axis="y", alpha=0.2)
    ax.text(
        0.02, 0.05,
        "Negative values are possible because validation covers a different, later period.",
        transform=ax.transAxes, fontsize=9, color=SLATE,
    )
    finish(fig, "06_train_validation_gap.png")


def split_plot() -> None:
    values = [326856, 94714]
    labels = ["Training\n2010-02-05 → 2012-03-16", "Validation (32 weeks)\n2012-03-23 → 2012-10-26"]
    fig, ax = plt.subplots(figsize=(10, 5.5))
    bars = ax.bar(labels, values, color=[BLUE, ORANGE], width=0.55)
    ax.set_title("LightGBM chronological split", fontsize=17, weight="bold")
    ax.set_ylabel("Rows")
    ax.set_ylim(0, 365000)
    ax.grid(axis="y", alpha=0.2)
    label_bars(ax, bars)
    finish(fig, "07_chronological_split.png")


def data_preparation_plot() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    bars = axes[0].bar(["Merged raw data", "Prepared model matrix"], [1422431, 0], color=[RED, GREEN])
    axes[0].set_title("Missing values before/after preparation")
    axes[0].set_ylabel("Missing cells")
    label_bars(axes[0], bars)
    axes[1].bar(["Raw features used"], [14], color=BLUE, width=0.5)
    axes[1].set_ylim(0, 16)
    axes[1].set_title("Minimal baseline feature set")
    axes[1].set_ylabel("Feature count")
    axes[1].text(0, 14.2, "14", ha="center", weight="bold")
    fig.suptitle("LightGBM baseline preprocessing", fontsize=18, weight="bold")
    finish(fig, "08_preprocessing_summary.png")


def workflow_plot() -> None:
    fig, ax = plt.subplots(figsize=(15, 4.2))
    ax.axis("off")
    labels = [
        "3 raw tables\ntrain + features + stores",
        "Merge\nStore + Date + holiday",
        "Minimal preparation\n14 raw features",
        "32-week\nchronological split",
        "Weighted LightGBM\nholiday weight = 5",
        "WMAE + W&B\ndiagnostics",
    ]
    xs = np.linspace(0.08, 0.92, len(labels))
    colors = [SLATE, BLUE, GREEN, ORANGE, RED, PURPLE]
    for index, (x, label, color) in enumerate(zip(xs, labels, colors)):
        ax.text(x, 0.5, label, ha="center", va="center", color="white", fontsize=10,
                bbox={"boxstyle": "round,pad=0.65", "facecolor": color, "edgecolor": "none"})
        if index < len(labels) - 1:
            ax.annotate("", xy=(xs[index + 1] - 0.065, 0.5), xytext=(x + 0.065, 0.5),
                        arrowprops={"arrowstyle": "->", "lw": 2, "color": SLATE})
    ax.set_title("LightGBM baseline workflow", fontsize=18, weight="bold", pad=20)
    finish(fig, "09_baseline_workflow.png")


def dashboard_plot() -> None:
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    items = sorted(FEATURE_IMPORTANCE.items(), key=lambda item: item[1], reverse=True)[:10][::-1]
    axes[0, 0].barh([item[0] for item in items], [item[1] for item in items], color=BLUE)
    axes[0, 0].set_title("Top 10 split importances")
    bars = axes[0, 1].bar(["Baseline", "Final engineered"], [3184.2771, 1575.15], color=[SLATE, GREEN])
    axes[0, 1].set_ylim(0, 3550)
    axes[0, 1].set_title("32-week validation WMAE")
    label_bars(axes[0, 1], bars, decimals=0)
    axes[1, 0].plot(ROUNDS, TRAIN_L1, marker="o", label="Train", color=BLUE)
    axes[1, 0].plot(ROUNDS, VALID_L1, marker="o", label="Validation", color=ORANGE)
    axes[1, 0].set_title("Learning curve")
    axes[1, 0].set_xlabel("Boosting iteration")
    axes[1, 0].legend()
    bars = axes[1, 1].bar(["WMAE", "MAE", "RMSE"], [3184.2771, 3156.0605, 6404.4248],
                          color=[BLUE, GREEN, SLATE])
    axes[1, 1].set_ylim(0, 7100)
    axes[1, 1].set_title("Validation metrics")
    label_bars(axes[1, 1], bars, decimals=0)
    fig.suptitle("LightGBM baseline — presentation dashboard", fontsize=21, weight="bold")
    finish(fig, "10_baseline_dashboard.png")


if __name__ == "__main__":
    feature_importance_plot()
    feature_family_plot()
    baseline_to_final_plot()
    metrics_plot()
    learning_curve_plot()
    train_validation_gap_plot()
    split_plot()
    data_preparation_plot()
    workflow_plot()
    dashboard_plot()
    print(f"Generated figures in {OUTPUT_DIR}")
