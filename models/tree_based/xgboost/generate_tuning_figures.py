"""Generate four presentation figures from the real 20-trial XGBoost study."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import csv


OUTPUT_DIR = Path(__file__).parent / "figures" / "hyperparameter_tuning"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BLUE = "#2563EB"
GREEN = "#16A34A"
ORANGE = "#F59E0B"
RED = "#DC2626"
SLATE = "#475569"
LIGHT = "#F1F5F9"

PARAMS = [
    "learning_rate", "max_depth", "min_child_weight", "subsample",
    "colsample_bytree", "gamma", "reg_alpha", "reg_lambda", "max_bin",
]

# Trial, WMAE, learning_rate, max_depth, min_child_weight, subsample,
# colsample_bytree, gamma, reg_alpha, reg_lambda, max_bin.
ROWS = [
    [0, 1935.973116, 0.0253627693, 12, 8.960785, 0.859530, 0.704607, 2.275054e-07, 3.332365e-08, 9.842316, 256],
    [1, 2144.234952, 0.1113546925, 11, 1.889120, 0.713639, 0.714192, 4.431943e-06, 5.282115e-04, 0.986094, 256],
    [2, 2034.539299, 0.0206669803, 7, 3.920674, 0.924812, 0.719886, 2.973788e-04, 2.146501e-03, 0.127904, 128],
    [3, 2055.287183, 0.1056864905, 12, 11.265467, 0.756615, 0.684185, 8.956175e-03, 9.148975e-05, 0.190903, 512],
    [4, 1976.959160, 0.0190226275, 9, 2.544166, 0.832024, 0.841349, 4.055490e-07, 5.324289, 6.075806, 128],
    [5, 2181.232104, 0.0988255831, 4, 1.798786, 0.665830, 0.763866, 2.404873e-05, 2.767842e-06, 8.071419, 512],
    [6, 2067.003293, 0.0141932803, 11, 1.250238, 0.995410, 0.920286, 5.353302e-07, 1.121241e-08, 7.523176, 512],
    [7, 2067.208187, 0.0120200867, 7, 1.414976, 0.952086, 0.868154, 7.559133e-06, 3.732718e-08, 0.519493, 256],
    [8, 2039.375508, 0.0906699452, 8, 1.430855, 0.899636, 0.916275, 7.630158e-04, 8.683696e-02, 1.368480, 128],
    [9, 2333.660417, 0.0130747804, 4, 6.729597, 0.760025, 0.828000, 7.850432e-01, 1.752387e-06, 0.879637, 128],
    [10, 2007.928739, 0.0293040252, 12, 4.353547, 0.802560, 0.761995, 1.071838e-08, 1.498503e-08, 14.784700, 256],
    [11, 1993.545567, 0.0115306266, 11, 2.828878, 0.836868, 0.842497, 4.253054e-07, 1.421550e-01, 9.570444, 128],
    [12, 1949.323197, 0.0165262065, 10, 9.132018, 0.758809, 0.737791, 1.342851e-04, 3.831075e-08, 1.544825, 256],
    [13, 1977.264017, 0.0273878324, 8, 11.273049, 0.863754, 0.794438, 5.782725e-05, 5.159200e-08, 1.202684, 256],
    [14, 1943.708285, 0.0194056528, 11, 14.962821, 0.898558, 0.701173, 4.085447e-06, 1.494538e-07, 12.674728, 128],
    [15, 1986.388318, 0.0349943910, 11, 17.024705, 0.983686, 0.746694, 1.242198e-08, 2.846528e-06, 11.374418, 128],
    [16, 1966.179455, 0.0146215613, 10, 8.627794, 0.844598, 0.738151, 2.952083e-05, 1.575951e-06, 4.010283, 128],
    [17, 1958.923260, 0.0385993929, 12, 4.695231, 0.943074, 0.652140, 9.886750e-07, 8.057000e-07, 2.836609, 256],
    [18, 2003.608105, 0.0305882331, 10, 17.155302, 0.877277, 0.817421, 3.902620e-03, 3.892123e-05, 18.595053, 512],
    [19, 1944.468909, 0.0261248423, 9, 15.937947, 0.840054, 0.652404, 3.406781e-06, 4.094483e-04, 3.889632, 512],
]

COLUMNS = ["trial", "wmae", *PARAMS]
TRIALS = [dict(zip(COLUMNS, row)) for row in ROWS]


def finish(fig: plt.Figure, filename: str) -> None:
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / filename, dpi=230, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def leaderboard() -> None:
    ranked = sorted(TRIALS, key=lambda row: row["wmae"], reverse=True)
    colors = [GREEN if row["trial"] == 0 else BLUE for row in ranked]
    fig, ax = plt.subplots(figsize=(12, 8))
    bars = ax.barh([f"Trial {int(row['trial'])}" for row in ranked],
                   [row["wmae"] for row in ranked], color=colors)
    ax.set_xlim(1850, 2380)
    ax.set_title("Optuna trial leaderboard — 52-week validation", fontsize=18, weight="bold")
    ax.set_xlabel("Validation WMAE — lower is better")
    ax.grid(axis="x", alpha=0.2)
    for bar, row in zip(bars, ranked):
        value = row["wmae"]
        ax.text(value + 5, bar.get_y() + bar.get_height() / 2, f"{value:,.1f}", va="center", fontsize=9)
    ax.text(0.99, 0.02, "Green = selected best trial", transform=ax.transAxes,
            ha="right", color=GREEN, weight="bold")
    finish(fig, "01_trial_leaderboard.png")


def optimization_history() -> None:
    ordered = sorted(TRIALS, key=lambda row: row["trial"])
    trials = np.array([row["trial"] for row in ordered])
    wmae = np.array([row["wmae"] for row in ordered])
    cumulative_best = np.minimum.accumulate(wmae)
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.scatter(trials, wmae, color=BLUE, s=60, label="Trial WMAE", zorder=3)
    ax.plot(trials, cumulative_best, color=GREEN, lw=2.8, label="Best so far")
    ax.scatter([0], [wmae[0]], color=RED, s=100, zorder=4)
    ax.annotate("Trial 0 remained champion\nWMAE = 1,935.97", (0, wmae[0]),
                xytext=(3, 2250), arrowprops={"arrowstyle": "->", "color": RED}, color=RED, weight="bold")
    ax.set_title("Optuna optimization history", fontsize=18, weight="bold")
    ax.set_xlabel("Trial number")
    ax.set_ylabel("Validation WMAE")
    ax.set_xticks(range(20))
    ax.grid(alpha=0.2)
    ax.legend()
    finish(fig, "02_optimization_history.png")


def hyperparameter_importance() -> None:
    # A transparent small-study sensitivity proxy: normalized absolute Spearman
    # correlation with WMAE. This is not causal and may miss non-linear effects.
    def ranks(values):
        order = np.argsort(values, kind="mergesort")
        result = np.empty(len(values), dtype=float)
        result[order] = np.arange(len(values), dtype=float)
        for value in np.unique(values):
            indices = np.flatnonzero(values == value)
            result[indices] = result[indices].mean()
        return result

    target_rank = ranks(np.array([row["wmae"] for row in TRIALS]))
    score_map = {}
    for parameter in PARAMS:
        values = np.array([row[parameter] for row in TRIALS], dtype=float)
        if parameter in {"learning_rate", "gamma", "reg_alpha", "reg_lambda"}:
            values = np.log10(values)
        score_map[parameter] = abs(np.corrcoef(ranks(values), target_rank)[0, 1])
    total = sum(score_map.values())
    scores = sorted(((name, value / total) for name, value in score_map.items()), key=lambda item: item[1])
    fig, ax = plt.subplots(figsize=(11, 6.5))
    ax.barh([item[0] for item in scores], [item[1] for item in scores], color=ORANGE)
    ax.set_title("Hyperparameter sensitivity across 20 trials", fontsize=18, weight="bold")
    ax.set_xlabel("Normalized |Spearman correlation with WMAE|")
    ax.grid(axis="x", alpha=0.2)
    ax.text(
        0.99, 0.02,
        "Small-study sensitivity estimate; association, not causal importance.",
        transform=ax.transAxes, ha="right", color=RED, fontsize=9, weight="bold",
    )
    finish(fig, "03_hyperparameter_importance.png")


def best_configuration_comparison() -> None:
    rows = [
        ["learning_rate", "0.03", "0.01–0.12 (log)", "0.025363"],
        ["max_depth", "8", "4–12", "12"],
        ["min_child_weight", "5", "1–20 (log)", "8.9608"],
        ["subsample", "0.85", "0.65–1.00", "0.8595"],
        ["colsample_bytree", "0.85", "0.65–1.00", "0.7046"],
        ["gamma", "0", "1e-8–5 (log)", "2.28e-7"],
        ["reg_alpha", "0", "1e-8–10 (log)", "3.33e-8"],
        ["reg_lambda", "1", "0.1–20 (log)", "9.8423"],
        ["max_bin", "256 (default)", "128 / 256 / 512", "256"],
    ]
    fig, ax = plt.subplots(figsize=(13, 7))
    ax.axis("off")
    table = ax.table(
        cellText=rows,
        colLabels=["Hyperparameter", "Baseline", "Tuning search space", "Best trial 0"],
        cellLoc="center", colLoc="center", loc="center",
        colWidths=[0.24, 0.18, 0.30, 0.20],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.8)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("white")
        if row == 0:
            cell.set_facecolor(SLATE)
            cell.set_text_props(color="white", weight="bold")
        elif col == 3:
            cell.set_facecolor("#DCFCE7")
            cell.set_text_props(color="#166534", weight="bold")
        elif row % 2 == 0:
            cell.set_facecolor(LIGHT)
    ax.set_title("Baseline vs tuned XGBoost configuration", fontsize=19, weight="bold", pad=18)
    ax.text(0.5, 0.03, "All nine hyperparameters that participated in Optuna tuning",
            transform=ax.transAxes, ha="center", color=SLATE, fontsize=11)
    finish(fig, "04_best_configuration_comparison.png")


if __name__ == "__main__":
    leaderboard()
    optimization_history()
    hyperparameter_importance()
    best_configuration_comparison()
    with (OUTPUT_DIR / "xgboost_optuna_20_trials.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(TRIALS)
    print(f"Generated tuning figures in {OUTPUT_DIR}")
