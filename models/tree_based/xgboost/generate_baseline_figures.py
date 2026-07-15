"""Generate presentation figures from saved XGBoost baseline notebook outputs."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


OUTPUT_DIR = Path(__file__).parent / "figures" / "baseline"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BLUE = "#2563EB"
GREEN = "#16A34A"
ORANGE = "#F59E0B"
RED = "#DC2626"
SLATE = "#475569"
LIGHT = "#E2E8F0"

FEATURE_IMPORTANCE = {
    "Dept": 0.165267,
    "Size": 0.162520,
    "Type": 0.151315,
    "Store": 0.078760,
    "Unemployment": 0.052802,
    "WeekCos": 0.052796,
    "CPI": 0.052632,
    "WeekOfYear": 0.044169,
    "Month": 0.037758,
    "WeekSin": 0.035637,
    "DaysFromStart": 0.034090,
    "Year": 0.033979,
    "Quarter": 0.031521,
    "Temperature": 0.023635,
    "IsHoliday": 0.022533,
    "Fuel_Price": 0.020585,
    "MarkDown1": 0.0,
    "MarkDown2": 0.0,
    "MarkDown3": 0.0,
    "MarkDown4": 0.0,
    "MarkDown5": 0.0,
    "TotalMarkDown": 0.0,
}

ROUNDS = np.array(
    [0, 50, 100, 150, 200, 300, 400, 500, 600, 700, 800, 900, 1000,
     1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000,
     2100, 2200, 2300, 2400, 2500, 2600, 2700, 2800, 2900, 2999]
)
TRAIN_WMAE = np.array(
    [13362.17774, 6745.22701, 4907.04938, 4191.61383, 3843.82951,
     3484.39923, 3144.08157, 2898.60272, 2711.61425, 2589.80132,
     2482.40811, 2403.90120, 2327.71235, 2266.02173, 2228.82008,
     2191.40689, 2145.87804, 2100.86962, 2076.53762, 2048.18696,
     2007.40818, 1976.39224, 1952.81390, 1929.20022, 1901.30333,
     1870.22444, 1844.89132, 1825.44056, 1804.39687, 1779.77297,
     1763.28437, 1744.28709, 1728.14081]
)
VALID_WMAE = np.array(
    [13924.37303, 7408.66471, 5607.13342, 4920.33341, 4592.05723,
     4270.90612, 3976.53619, 3766.20689, 3612.30585, 3516.89562,
     3429.66391, 3373.60285, 3316.90876, 3278.40068, 3252.66074,
     3226.02370, 3193.42476, 3156.75569, 3142.06745, 3121.43220,
     3092.79755, 3072.51875, 3055.14679, 3039.82770, 3019.61734,
     2997.65303, 2975.90795, 2962.17012, 2945.83035, 2930.40388,
     2921.46113, 2910.79836, 2902.28931]
)


def finish(fig: plt.Figure, name: str) -> None:
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / name, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def add_values(ax, bars, decimals=0) -> None:
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
    colors = [ORANGE if value == 0 else BLUE for value in values]
    fig, ax = plt.subplots(figsize=(11, 9))
    ax.barh(names, values, color=colors)
    ax.set_title("XGBoost baseline — feature importance", fontsize=17, weight="bold")
    ax.set_xlabel("Normalized importance")
    ax.grid(axis="x", alpha=0.2)
    ax.text(
        0.99, 0.02,
        "Orange = no split contribution in this fitted baseline",
        transform=ax.transAxes, ha="right", color=SLATE, fontsize=9,
    )
    finish(fig, "01_feature_importance.png")


def feature_family_plot() -> None:
    families = {
        "Identity / store": ["Store", "Dept", "Type", "Size"],
        "Calendar": ["Year", "Month", "WeekOfYear", "Quarter", "DaysFromStart", "WeekSin", "WeekCos"],
        "External": ["Temperature", "Fuel_Price", "CPI", "Unemployment"],
        "Holiday": ["IsHoliday"],
        "Markdown": ["MarkDown1", "MarkDown2", "MarkDown3", "MarkDown4", "MarkDown5", "TotalMarkDown"],
    }
    values = [sum(FEATURE_IMPORTANCE[name] for name in names) for names in families.values()]
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(families.keys(), values, color=[BLUE, GREEN, ORANGE, RED, SLATE])
    ax.set_title("Where baseline feature importance came from", fontsize=17, weight="bold")
    ax.set_ylabel("Summed normalized importance")
    ax.set_ylim(0, max(values) * 1.18)
    ax.grid(axis="y", alpha=0.2)
    add_values(ax, bars, decimals=3)
    finish(fig, "02_feature_family_importance.png")


def benchmark_plot() -> None:
    labels = ["Median benchmark", "XGBoost baseline"]
    values = [3305.52, 2902.289158]
    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.bar(labels, values, color=[SLATE, BLUE], width=0.58)
    ax.set_title("Validation WMAE: baseline vs benchmark", fontsize=17, weight="bold")
    ax.set_ylabel("WMAE — lower is better")
    ax.set_ylim(0, 3700)
    ax.grid(axis="y", alpha=0.2)
    add_values(ax, bars, decimals=1)
    ax.annotate(
        "12.20% improvement",
        xy=(1, values[1]), xytext=(0.5, 3500), ha="center", weight="bold", color=GREEN,
        arrowprops={"arrowstyle": "->", "color": GREEN, "lw": 2},
    )
    finish(fig, "03_wmae_vs_median_benchmark.png")


def error_breakdown_plot() -> None:
    labels = ["Overall\nWMAE", "Overall\nMAE", "Holiday\nMAE", "Non-holiday\nMAE", "RMSE"]
    values = [2902.289158, 2727.652513, 3464.407683, 2665.610790, 6654.003295]
    fig, ax = plt.subplots(figsize=(11, 6))
    bars = ax.bar(labels, values, color=[BLUE, GREEN, RED, ORANGE, SLATE])
    ax.set_title("XGBoost baseline — validation error profile", fontsize=17, weight="bold")
    ax.set_ylabel("Error — lower is better")
    ax.set_ylim(0, 7300)
    ax.grid(axis="y", alpha=0.2)
    add_values(ax, bars, decimals=0)
    finish(fig, "04_validation_error_breakdown.png")


def learning_curve_plot() -> None:
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(ROUNDS, TRAIN_WMAE, color=BLUE, lw=2.5, label="Training weighted MAE")
    ax.plot(ROUNDS, VALID_WMAE, color=ORANGE, lw=2.5, label="Validation weighted MAE")
    ax.scatter([2999], [2902.28931], color=RED, zorder=5)
    ax.annotate("Best recorded validation = 2,902.29\nround 2,999", (2999, 2902.28931),
                xytext=(2050, 5000), arrowprops={"arrowstyle": "->", "color": RED})
    ax.set_title("Training dynamics", fontsize=17, weight="bold")
    ax.set_xlabel("Boosting round")
    ax.set_ylabel("Weighted MAE")
    ax.grid(alpha=0.2)
    ax.legend()
    finish(fig, "05_learning_curve.png")


def generalization_gap_plot() -> None:
    gap = VALID_WMAE - TRAIN_WMAE
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.plot(ROUNDS, gap, color=RED, lw=2.5)
    ax.fill_between(ROUNDS, gap, alpha=0.15, color=RED)
    ax.set_title("Train–validation generalization gap", fontsize=17, weight="bold")
    ax.set_xlabel("Boosting round")
    ax.set_ylabel("Validation WMAE − training WMAE")
    ax.grid(alpha=0.2)
    ax.annotate(f"Final gap = {gap[-1]:,.1f}", (ROUNDS[-1], gap[-1]),
                xytext=(2100, gap[-1] + 140), arrowprops={"arrowstyle": "->", "color": RED})
    finish(fig, "06_generalization_gap.png")


def split_plot() -> None:
    labels = ["Training\n2010-02-05 → 2011-10-28", "Validation (52 weeks)\n2011-11-04 → 2012-10-26"]
    values = [267184, 154386]
    fig, ax = plt.subplots(figsize=(10, 5.5))
    bars = ax.bar(labels, values, color=[BLUE, ORANGE], width=0.55)
    ax.set_title("Chronological baseline split", fontsize=17, weight="bold")
    ax.set_ylabel("Rows")
    ax.set_ylim(0, 300000)
    ax.grid(axis="y", alpha=0.2)
    add_values(ax, bars, decimals=0)
    finish(fig, "07_chronological_split.png")


def flow_plot() -> None:
    fig, ax = plt.subplots(figsize=(15, 4.2))
    ax.axis("off")
    labels = [
        "4 raw tables\ntrain/test + features + stores",
        "Merge\nStore + Date",
        "22 baseline features\ncalendar + store + external",
        "52-week\nchronological split",
        "Weighted XGBoost\nholiday weight = 5",
        "WMAE + W&B\nmodel artifacts",
    ]
    xs = np.linspace(0.08, 0.92, len(labels))
    colors = [SLATE, BLUE, GREEN, ORANGE, RED, BLUE]
    for index, (x, label, color) in enumerate(zip(xs, labels, colors)):
        ax.text(x, 0.5, label, ha="center", va="center", color="white", fontsize=10,
                bbox={"boxstyle": "round,pad=0.65", "facecolor": color, "edgecolor": "none"})
        if index < len(labels) - 1:
            ax.annotate("", xy=(xs[index + 1] - 0.065, 0.5), xytext=(x + 0.065, 0.5),
                        arrowprops={"arrowstyle": "->", "lw": 2, "color": SLATE})
    ax.set_title("XGBoost baseline workflow", fontsize=18, weight="bold", pad=20)
    finish(fig, "08_baseline_workflow.png")


def dashboard_plot() -> None:
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    names = [name for name, _ in sorted(FEATURE_IMPORTANCE.items(), key=lambda item: item[1], reverse=True)[:10]][::-1]
    values = [FEATURE_IMPORTANCE[name] for name in names]
    axes[0, 0].barh(names, values, color=BLUE)
    axes[0, 0].set_title("Top 10 features")
    axes[0, 0].grid(axis="x", alpha=0.2)
    bars = axes[0, 1].bar(["Median", "XGBoost"], [3305.52, 2902.289158], color=[SLATE, BLUE])
    axes[0, 1].set_title("Validation WMAE")
    axes[0, 1].set_ylim(0, 3700)
    add_values(axes[0, 1], bars, decimals=0)
    axes[1, 0].plot(ROUNDS, TRAIN_WMAE, label="Train", color=BLUE)
    axes[1, 0].plot(ROUNDS, VALID_WMAE, label="Validation", color=ORANGE)
    axes[1, 0].set_title("Learning curve")
    axes[1, 0].set_xlabel("Boosting round")
    axes[1, 0].legend()
    error_labels = ["WMAE", "MAE", "Holiday MAE", "Non-holiday MAE"]
    error_values = [2902.289158, 2727.652513, 3464.407683, 2665.610790]
    bars = axes[1, 1].bar(error_labels, error_values, color=[BLUE, GREEN, RED, ORANGE])
    axes[1, 1].set_title("Error breakdown")
    axes[1, 1].tick_params(axis="x", rotation=15)
    add_values(axes[1, 1], bars, decimals=0)
    fig.suptitle("XGBoost baseline — presentation dashboard", fontsize=21, weight="bold")
    finish(fig, "09_baseline_dashboard.png")


if __name__ == "__main__":
    feature_importance_plot()
    feature_family_plot()
    benchmark_plot()
    error_breakdown_plot()
    learning_curve_plot()
    generalization_gap_plot()
    split_plot()
    flow_plot()
    dashboard_plot()
    print(f"Generated figures in {OUTPUT_DIR}")
