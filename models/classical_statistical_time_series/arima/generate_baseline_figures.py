"""Generate presentation figures from the recorded aggregate ARIMA baseline."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).parent
REPO = ROOT.parents[2]
OUT = ROOT / "figures" / "baseline"
OUT.mkdir(parents=True, exist_ok=True)
BLUE, GREEN, ORANGE, RED, PURPLE, SLATE = "#2563EB", "#16A34A", "#F59E0B", "#DC2626", "#7C3AED", "#475569"


def save(fig, name, rect=None):
    fig.tight_layout(rect=rect)
    fig.savefig(OUT / name, dpi=230, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def load_weekly():
    train = pd.read_csv(REPO / "data" / "train.csv", parse_dates=["Date"])
    weekly = train.groupby("Date", as_index=False)["Weekly_Sales"].sum().sort_values("Date")
    return train, weekly


def benchmark():
    labels = ["Seasonal naive\n52-week", "Aggregate ARIMA\n(1,1,1)"]
    values = [1800.1736, 1856.8605]
    fig, ax = plt.subplots(figsize=(9, 6))
    bars = ax.bar(labels, values, color=[GREEN, ORANGE])
    ax.set(title="ARIMA baseline validation benchmark", ylabel="WMAE — lower is better", ylim=(0, 2150))
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 35, f"{value:,.2f}", ha="center", weight="bold")
    ax.text(0.5, 350, "ARIMA is 3.15% worse than seasonal naive", ha="center", color=RED, weight="bold")
    ax.grid(axis="y", alpha=.2)
    save(fig, "01_validation_benchmark.png")


def weekly_holdout():
    _, weekly = load_weekly()
    split = len(weekly) - 39
    fig, ax = plt.subplots(figsize=(15, 6))
    ax.plot(weekly.Date.iloc[:split], weekly.Weekly_Sales.iloc[:split] / 1e6, color=BLUE, lw=2, label="Training weekly total")
    ax.plot(weekly.Date.iloc[split:], weekly.Weekly_Sales.iloc[split:] / 1e6, color=ORANGE, lw=2.4, label="Validation actual")
    ax.axvline(weekly.Date.iloc[split], color=RED, ls="--", lw=2, label="Validation starts: 2012-02-03")
    ax.set(title="Aggregate weekly sales and chronological holdout", xlabel="Date", ylabel="Total Weekly_Sales (million)")
    ax.grid(alpha=.2); ax.legend()
    save(fig, "02_weekly_total_and_holdout.png")


def architecture():
    fig, ax = plt.subplots(figsize=(17, 5)); ax.axis("off")
    steps = [
        "421,570 row-level\nsales records",
        "Aggregate by Date\n143 weekly totals",
        "ARIMA(1,1,1)\non weekly total",
        "Forecast 39\nweekly totals",
        "Last-year Store–Dept\nshare per week",
        "Row-level\npredictions",
    ]
    xs = np.linspace(.07, .93, len(steps))
    colors = [BLUE, BLUE, PURPLE, ORANGE, GREEN, GREEN]
    for i, (x, label, color) in enumerate(zip(xs, steps, colors)):
        ax.text(x, .5, label, ha="center", va="center", transform=ax.transAxes,
                bbox=dict(boxstyle="round,pad=.7", fc="white", ec=color, lw=2.5))
        if i < len(steps) - 1:
            ax.annotate("", xy=(xs[i + 1] - .065, .5), xytext=(x + .065, .5), xycoords=ax.transAxes,
                        arrowprops=dict(arrowstyle="->", lw=2.2, color=SLATE))
    ax.set_title("Aggregate ARIMA baseline architecture", fontsize=20, weight="bold")
    save(fig, "03_training_inference_flow.png")


def order_diagram():
    fig, ax = plt.subplots(figsize=(13, 6)); ax.axis("off")
    items = [
        (2.2, "AR(1)", "Uses one previous\nweekly-total value", BLUE),
        (6.5, "I(1)", "Differences once to\nremove level/trend", ORANGE),
        (10.8, "MA(1)", "Uses one previous\nforecast error", PURPLE),
    ]
    for x, title, body, color in items:
        ax.text(x, 3.1, title, ha="center", fontsize=19, weight="bold", color=color)
        ax.text(x, 2.0, body, ha="center", va="center", fontsize=13,
                bbox=dict(boxstyle="round,pad=.7", fc="white", ec=color, lw=2.5))
    ax.text(6.5, .5, "ARIMA order = (p, d, q) = (1, 1, 1)", ha="center", fontsize=18, weight="bold")
    ax.set_xlim(0, 13); ax.set_ylim(0, 5)
    ax.set_title("What the baseline ARIMA(1,1,1) contains", fontsize=20, weight="bold")
    save(fig, "04_arima_order_explained.png")


def allocation():
    fig, ax = plt.subplots(figsize=(15, 6)); ax.axis("off")
    ax.text(.12, .62, "ARIMA weekly total\n$48M", ha="center", va="center", transform=ax.transAxes,
            bbox=dict(boxstyle="round,pad=.8", fc="#FFF7ED", ec=ORANGE, lw=3))
    rows = [("Store 1–Dept 1", "2.0%", "$0.96M"), ("Store 1–Dept 2", "1.2%", "$0.58M"), ("All other rows", "96.8%", "$46.46M")]
    ys = [.82, .52, .22]
    for (name, share, pred), y in zip(rows, ys):
        ax.annotate("", xy=(.45, y), xytext=(.21, .62), xycoords=ax.transAxes,
                    arrowprops=dict(arrowstyle="->", lw=2, color=SLATE))
        ax.text(.58, y, f"{name}\nlast-year share = {share}\nprediction = {pred}", ha="center", va="center", transform=ax.transAxes,
                bbox=dict(boxstyle="round,pad=.55", fc="#F0FDF4", ec=GREEN, lw=2))
    ax.text(.84, .52, "Shares sum to 100%\nfor each forecast week", ha="center", va="center", transform=ax.transAxes,
            bbox=dict(boxstyle="round,pad=.7", fc="#EFF6FF", ec=BLUE, lw=2.5))
    ax.annotate("", xy=(.77, .52), xytext=(.70, .52), xycoords=ax.transAxes,
                arrowprops=dict(arrowstyle="->", lw=2, color=SLATE))
    ax.set_title("How aggregate ARIMA becomes Store–Dept predictions", fontsize=20, weight="bold")
    save(fig, "05_last_year_share_allocation.png")


def metric_table():
    rows = [
        ["Validation WMAE", "1,856.86"], ["Validation MAE", "1,843.29"],
        ["Validation RMSE", "3,919.59"], ["Seasonal-naive WMAE", "1,800.17"],
        ["Difference vs seasonal naive", "+3.15% worse"], ["ARIMA order", "(1, 1, 1)"],
        ["Validation horizon", "39 weeks"], ["Allocation", "last-year share"],
    ]
    fig, ax = plt.subplots(figsize=(10, 7)); ax.axis("off")
    table = ax.table(cellText=rows, colLabels=["Recorded item", "Value"], cellLoc="left", colLoc="left", loc="center")
    table.auto_set_font_size(False); table.set_fontsize(12); table.scale(1, 1.7)
    for (r, _), cell in table.get_celld().items():
        cell.set_edgecolor("#CBD5E1"); cell.set_facecolor(BLUE if r == 0 else ("#F8FAFC" if r % 2 == 0 else "white"))
        if r == 0: cell.set_text_props(color="white", weight="bold")
    ax.set_title("Aggregate ARIMA baseline — recorded configuration", fontsize=20, weight="bold")
    save(fig, "06_baseline_metrics_table.png")


def split():
    fig, ax = plt.subplots(figsize=(14, 3))
    ax.barh([0], [104], color=BLUE, label="Training: 104 weeks")
    ax.barh([0], [39], left=[104], color=ORANGE, label="Validation: final 39 weeks")
    ax.set(xlim=(0, 143), xlabel="Weekly index", title="Chronological split — no random shuffling")
    ax.set_yticks([]); ax.legend(loc="lower center", bbox_to_anchor=(.5, -.58), ncol=2)
    save(fig, "07_chronological_split.png")


def limitations():
    labels = ["One aggregate\ntime series", "No Store/Dept\ndynamics in ARIMA", "No explicit\n52-week seasonality", "Allocation depends\non last year"]
    fig, ax = plt.subplots(figsize=(14, 5)); ax.axis("off")
    xs = np.linspace(.12, .88, 4)
    for x, label in zip(xs, labels):
        ax.text(x, .5, label, ha="center", va="center", transform=ax.transAxes, fontsize=13,
                bbox=dict(boxstyle="round,pad=.8", fc="#FEF2F2", ec=RED, lw=2.5))
    ax.set_title("Why aggregate ARIMA did not beat seasonal naive", fontsize=20, weight="bold")
    save(fig, "08_baseline_limitations.png")


if __name__ == "__main__":
    benchmark(); weekly_holdout(); architecture(); order_diagram(); allocation(); metric_table(); split(); limitations()
    print(f"Generated ARIMA baseline figures in {OUT}")
