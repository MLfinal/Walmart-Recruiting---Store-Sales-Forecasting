"""Generate presentation diagrams for the XGBoost feature-engineering stage."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


OUTPUT_DIR = Path(__file__).parent / "figures" / "feature_engineering"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BLUE = "#2563EB"
GREEN = "#16A34A"
ORANGE = "#F59E0B"
RED = "#DC2626"
PURPLE = "#7C3AED"
TEAL = "#0D9488"
SLATE = "#475569"
LIGHT = "#F1F5F9"


def save(fig: plt.Figure, filename: str) -> None:
    fig.savefig(OUTPUT_DIR / filename, dpi=240, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def box(ax, x, y, width, height, text, color, fontsize=13, text_color="white"):
    patch = FancyBboxPatch(
        (x, y), width, height,
        boxstyle="round,pad=0.025,rounding_size=0.025",
        linewidth=0,
        facecolor=color,
    )
    ax.add_patch(patch)
    ax.text(
        x + width / 2, y + height / 2, text,
        ha="center", va="center", color=text_color,
        fontsize=fontsize, weight="bold",
    )
    return patch


def arrow(ax, x1, y1, x2, y2, color=SLATE, style="-|>"):
    ax.add_patch(
        FancyArrowPatch(
            (x1, y1), (x2, y2),
            arrowstyle=style, mutation_scale=18,
            linewidth=2.2, color=color,
        )
    )


def feature_count_evolution() -> None:
    fig, ax = plt.subplots(figsize=(13, 5.5))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    box(ax, 0.04, 0.34, 0.24, 0.32, "BASELINE\n22 features", SLATE, 18)
    box(ax, 0.38, 0.34, 0.24, 0.32, "+59 NEW\nengineered features", ORANGE, 17)
    box(ax, 0.72, 0.34, 0.24, 0.32, "FINAL MATRIX\n81 features", GREEN, 18)
    arrow(ax, 0.29, 0.50, 0.37, 0.50)
    arrow(ax, 0.63, 0.50, 0.71, 0.50)

    ax.text(0.16, 0.22, "No baseline features removed", ha="center", color=SLATE, fontsize=12)
    ax.text(0.50, 0.22, "Calendar • holiday • markdown\ninteractions • history • aggregates",
            ha="center", color=SLATE, fontsize=11)
    ax.text(0.84, 0.22, "Same ordered matrix for\ntrain, validation and test",
            ha="center", color=SLATE, fontsize=11)
    ax.set_title("XGBoost feature-set evolution", fontsize=22, weight="bold", pad=18)
    save(fig, "01_feature_count_evolution.png")


def feature_groups() -> None:
    groups = [
        ("Calendar", 3, "DayOfYear\nMonthSin, MonthCos", BLUE),
        ("Holiday", 12, "holiday identity\nproximity and direction", RED),
        ("Markdown", 23, "missing flags, presence, log\nand holiday interactions", PURPLE),
        ("Interactions", 3, "Store_Dept, Type_Dept\nSize_log1p", TEAL),
        ("Annual history", 2, "SalesLag52\navailability flag", ORANGE),
        ("Target aggregates", 16, "mean, median, std, count\nacross four groupings", GREEN),
    ]

    fig, ax = plt.subplots(figsize=(15, 8.5))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    positions = [(0.05, 0.57), (0.37, 0.57), (0.69, 0.57),
                 (0.05, 0.17), (0.37, 0.17), (0.69, 0.17)]

    for (title, count, detail, color), (x, y) in zip(groups, positions):
        box(ax, x, y, 0.26, 0.25, f"{title}\n+{count} features", color, 15)
        ax.text(x + 0.13, y - 0.045, detail, ha="center", va="top", color=SLATE, fontsize=10.5)

    ax.text(
        0.5, 0.05,
        "Total added: 59  |  Baseline retained: 22  |  Final feature matrix: 81",
        ha="center", va="center", fontsize=14, weight="bold", color=SLATE,
        bbox={"boxstyle": "round,pad=0.5", "facecolor": LIGHT, "edgecolor": "none"},
    )
    ax.set_title("What feature engineering added", fontsize=22, weight="bold", pad=18)
    save(fig, "02_engineered_feature_groups.png")


def leakage_safe_flow() -> None:
    fig, ax = plt.subplots(figsize=(16, 7))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    box(ax, 0.03, 0.56, 0.19, 0.22, "KNOWN TRAINING\nSALES HISTORY", BLUE, 15)
    box(ax, 0.29, 0.56, 0.16, 0.22, "shift(1)\nexclude current target", ORANGE, 14)
    box(ax, 0.52, 0.56, 0.19, 0.22, "EXPANDING\nmean • median\nstd • count", PURPLE, 14)
    box(ax, 0.78, 0.56, 0.19, 0.22, "TRAINING ROW\nSAFE FEATURES", GREEN, 15)
    arrow(ax, 0.22, 0.67, 0.285, 0.67)
    arrow(ax, 0.45, 0.67, 0.515, 0.67)
    arrow(ax, 0.71, 0.67, 0.775, 0.67)

    ax.text(0.50, 0.45, "At the validation cutoff", ha="center", fontsize=13,
            color=SLATE, weight="bold")
    arrow(ax, 0.50, 0.53, 0.50, 0.38, color=SLATE)

    box(ax, 0.22, 0.12, 0.25, 0.20, "FREEZE TRAIN-ONLY\nGROUP MAPPINGS", SLATE, 14)
    box(ax, 0.58, 0.12, 0.25, 0.20, "VALIDATION / TEST\nNO TARGET UPDATE", GREEN, 14)
    arrow(ax, 0.47, 0.22, 0.575, 0.22)

    ax.text(
        0.5, 0.035,
        "Validation week 1 never becomes a feature for validation week 2 — future sales remain unseen.",
        ha="center", fontsize=12, color=RED, weight="bold",
    )
    ax.set_title("Leakage-safe target aggregate flow", fontsize=22, weight="bold", pad=18)
    save(fig, "03_leakage_safe_target_aggregates.png")


if __name__ == "__main__":
    feature_count_evolution()
    feature_groups()
    leakage_safe_flow()
    print(f"Generated diagrams in {OUTPUT_DIR}")
