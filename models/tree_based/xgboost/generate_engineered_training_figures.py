"""Generate presentation figures from the saved engineered XGBoost run."""

import base64
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).parent
OUTPUT_DIR = ROOT / "figures" / "engineered_training"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BLUE = "#2563EB"
GREEN = "#16A34A"
ORANGE = "#F59E0B"
RED = "#DC2626"
PURPLE = "#7C3AED"
SLATE = "#475569"

ROUNDS = np.array(
    [0, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500,
     550, 600, 650, 700, 750, 800, 850, 900, 950, 999]
)
TRAIN_WMAE = np.array(
    [13414.15828, 5176.80626, 2579.47121, 1849.10774, 1598.94062,
     1491.66614, 1435.06379, 1370.44785, 1297.33719, 1246.20746,
     1194.45798, 1158.44644, 1130.56477, 1102.97022, 1076.72783,
     1056.10084, 1039.56496, 1025.34896, 1012.45543, 1003.94159,
     996.93600]
)
VALID_WMAE = np.array(
    [13083.76261, 4821.76372, 2361.13073, 1802.67798, 1679.42912,
     1649.47251, 1640.32872, 1632.46740, 1626.14847, 1620.88781,
     1616.06384, 1613.50238, 1614.69514, 1613.11083, 1613.61898,
     1612.97826, 1613.11187, 1612.87355, 1612.77506, 1612.76034,
     1612.12657]
)

IMPORTANCE = {
    "Store_Dept_Sales_median": 510.444580,
    "Store_Dept_Sales_mean": 178.900513,
    "SalesLag52": 38.913094,
    "Type_Dept_Sales_median": 38.527721,
    "IsThanksgivingWeek": 34.454632,
    "Year": 32.712978,
    "Type_Dept_Sales_mean": 28.672062,
    "Holiday_TotalMarkDown": 27.998964,
    "Holiday_MarkDown3": 27.912550,
    "Holiday_MarkDown5": 26.441568,
    "Holiday_MarkDown4": 26.255085,
    "AbsWeeksToNearestThanksgiving": 26.136375,
    "Holiday_MarkDown2": 25.151472,
    "IsChristmasWeek": 24.477732,
    "Store_Dept_Sales_std": 23.991045,
    "DayOfYear": 23.615070,
    "Holiday_MarkDown1": 23.477947,
    "Dept_Sales_median": 22.207544,
    "Dept_Sales_std": 21.614994,
    "WeekOfYear": 21.359562,
}


def finish(fig: plt.Figure, filename: str) -> None:
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / filename, dpi=230, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def label_bars(ax, bars, decimals=0) -> None:
    for bar in bars:
        value = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value,
            f"{value:,.{decimals}f}",
            ha="center", va="bottom", fontsize=10,
        )


def learning_curve() -> None:
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(ROUNDS, TRAIN_WMAE, lw=2.6, color=BLUE, label="Training weighted MAE")
    ax.plot(ROUNDS, VALID_WMAE, lw=2.6, color=ORANGE, label="Validation weighted MAE")
    ax.scatter([999], [1612.12657], color=RED, zorder=5)
    ax.annotate(
        "Best recorded validation\nWMAE = 1,612.13",
        xy=(999, 1612.12657), xytext=(660, 3600),
        arrowprops={"arrowstyle": "->", "color": RED}, color=RED, weight="bold",
    )
    ax.set_title("Engineered XGBoost training curve", fontsize=18, weight="bold")
    ax.set_xlabel("Boosting round")
    ax.set_ylabel("Weighted MAE")
    ax.grid(alpha=0.2)
    ax.legend()
    finish(fig, "01_learning_curve.png")


def result_evolution() -> None:
    labels = ["Static baseline\n52 weeks", "Engineered model\n52 weeks", "Final candidate\n32 weeks"]
    values = [2902.2892, 1935.97, 1612.1265]
    fig, ax = plt.subplots(figsize=(10, 6.5))
    bars = ax.bar(labels, values, color=[SLATE, ORANGE, GREEN], width=0.6)
    ax.set_title("XGBoost validation WMAE evolution", fontsize=18, weight="bold")
    ax.set_ylabel("WMAE — lower is better")
    ax.set_ylim(0, 3300)
    ax.grid(axis="y", alpha=0.2)
    label_bars(ax, bars, decimals=1)
    ax.text(
        0.5, -0.18,
        "The first two results are directly comparable (52 weeks). The final candidate uses a 32-week holdout.",
        transform=ax.transAxes, ha="center", color=RED, fontsize=10, weight="bold",
    )
    finish(fig, "02_wmae_result_evolution.png")


def feature_importance() -> None:
    items = sorted(IMPORTANCE.items(), key=lambda item: item[1])
    fig, ax = plt.subplots(figsize=(12, 8.5))
    ax.barh([item[0] for item in items], [item[1] for item in items], color=BLUE)
    ax.set_title("Engineered XGBoost — top 20 features by gain", fontsize=18, weight="bold")
    ax.set_xlabel("Gain importance")
    ax.grid(axis="x", alpha=0.2)
    ax.text(
        0.99, 0.02,
        "Importance is predictive contribution, not causal effect.",
        transform=ax.transAxes, ha="right", color=SLATE, fontsize=9,
    )
    finish(fig, "03_feature_importance.png")


def holiday_error() -> None:
    labels = ["Holiday MAE", "Non-holiday MAE"]
    values = [1821.024048, 1578.360596]
    difference = values[0] - values[1]
    fig, ax = plt.subplots(figsize=(8.5, 6))
    bars = ax.bar(labels, values, color=[RED, GREEN], width=0.58)
    ax.set_title("Error by holiday status", fontsize=18, weight="bold")
    ax.set_ylabel("MAE — lower is better")
    ax.set_ylim(0, 2100)
    ax.grid(axis="y", alpha=0.2)
    label_bars(ax, bars, decimals=1)
    ax.text(0.5, 1950, f"Holiday MAE is {difference:,.1f} higher", ha="center", color=RED, weight="bold")
    finish(fig, "04_holiday_vs_non_holiday_error.png")


def metric_profile() -> None:
    labels = ["WMAE", "MAE", "RMSE", "Median benchmark"]
    values = [1612.126497, 1585.959717, 3423.168123, 2388.061568]
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(labels, values, color=[BLUE, GREEN, SLATE, ORANGE], width=0.62)
    ax.set_title("Engineered XGBoost validation profile", fontsize=18, weight="bold")
    ax.set_ylabel("Error")
    ax.set_ylim(0, 3800)
    ax.grid(axis="y", alpha=0.2)
    label_bars(ax, bars, decimals=0)
    ax.text(0.5, 3600, "32.49% WMAE improvement over the split-specific median benchmark",
            ha="center", color=GREEN, weight="bold")
    finish(fig, "05_validation_metric_profile.png")


def extract_notebook_diagnostics() -> bool:
    notebook = json.loads((ROOT / "model_experiment_XGBoost.ipynb").read_text())
    for cell in notebook.get("cells", []):
        source = "".join(cell.get("source", []))
        if "diagnostics/overview" not in source:
            continue
        for output in cell.get("outputs", []):
            png = output.get("data", {}).get("image/png")
            if png:
                encoded = "".join(png) if isinstance(png, list) else png
                (OUTPUT_DIR / "06_actual_prediction_residual_diagnostics.png").write_bytes(
                    base64.b64decode(encoded)
                )
                return True
    return False


if __name__ == "__main__":
    learning_curve()
    result_evolution()
    feature_importance()
    holiday_error()
    metric_profile()
    extracted = extract_notebook_diagnostics()
    print(f"Generated figures in {OUTPUT_DIR}; notebook diagnostics extracted={extracted}")
