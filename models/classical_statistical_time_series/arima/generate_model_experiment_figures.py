"""Generate presentation figures from recorded ARIMA and ARIMAX experiments."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).parent
OUT = ROOT / "figures" / "model_experiment"
OUT.mkdir(parents=True, exist_ok=True)
BLUE, GREEN, ORANGE, RED, PURPLE, SLATE = "#2563EB", "#16A34A", "#F59E0B", "#DC2626", "#7C3AED", "#475569"

ORDERS = [(p, d, q) for p in range(3) for d in range(2) for q in range(3)]
LAST_YEAR = [1836.5913, 1835.7033, 1852.3582, 2941.1402, 2183.7088, 1846.0585,
             1851.9874, 1870.4175, 1829.8800, 2776.0371, 1856.8605, 2310.0948,
             1873.4015, 1885.6444, 1834.6869, 2715.9486, 2132.7671, 1889.0005]
BLENDED = [2002.8350, 2003.6289, 2023.9889, 3149.6627, 2425.3955, 1994.0740,
           2023.7461, 2046.9084, 1995.6378, 2995.4548, 2050.0330, 2551.2186,
           2049.9196, 2061.1298, 2002.0611, 2939.1328, 2373.5821, 2040.4990]
ARIMAX_LAST = [2563.6915, 2686.9430, 2768.5633, 3840.8970, 3545.5354, 3863.5144,
               2793.6988, 2900.3397, 2997.0660, 3630.4139, 3656.7977, 4306.2966,
               2936.6539, 2948.3561, 2691.2308, 3660.7590, 3625.5305, 3956.4040]
SEASONAL_NAIVE, BASELINE = 1800.1736, 1856.8605


def save(fig, name, rect=None):
    fig.tight_layout(rect=rect)
    fig.savefig(OUT / name, dpi=230, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def results_frame():
    return pd.DataFrame({"order": [str(x) for x in ORDERS], "p": [x[0] for x in ORDERS],
                         "d": [x[1] for x in ORDERS], "q": [x[2] for x in ORDERS],
                         "last_year_share": LAST_YEAR, "blended_share": BLENDED,
                         "arimax_last_year": ARIMAX_LAST})


def evolution():
    labels = ["Seasonal naive", "ARIMA baseline\n(1,1,1)", "Best pure ARIMA\n(1,0,2)", "Best ARIMAX\n(0,0,0)"]
    values = [SEASONAL_NAIVE, BASELINE, 1829.8800, 2563.6915]
    fig, ax = plt.subplots(figsize=(12, 6)); bars = ax.bar(labels, values, color=[GREEN, ORANGE, BLUE, RED])
    ax.set(title="Recorded ARIMA experiment progression", ylabel="Validation WMAE — lower is better", ylim=(0, 2900))
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, value + 45, f"{value:,.2f}", ha="center", weight="bold")
    ax.grid(axis="y", alpha=.2); save(fig, "01_experiment_progression.png")


def leaderboard():
    df = results_frame().sort_values("last_year_share").head(10).sort_values("last_year_share", ascending=True)
    fig, ax = plt.subplots(figsize=(11, 7)); bars = ax.barh(df.order, df.last_year_share, color=[GREEN] + [BLUE]*9)
    ax.axvline(SEASONAL_NAIVE, color=RED, ls="--", lw=2, label=f"Seasonal naive {SEASONAL_NAIVE:,.2f}")
    ax.set(title="Pure ARIMA order-search leaderboard", xlabel="Validation WMAE — lower is better", ylabel="ARIMA order")
    ax.set_xlim(1780, 1905)
    for bar, value in zip(bars, df.last_year_share): ax.text(value + 2, bar.get_y()+bar.get_height()/2, f"{value:,.1f}", va="center", fontsize=9)
    ax.grid(axis="x", alpha=.2); ax.legend(); save(fig, "02_order_search_leaderboard.png")


def all_orders():
    df = results_frame(); x = np.arange(len(df))
    fig, ax = plt.subplots(figsize=(16, 7)); ax.plot(x, df.last_year_share, marker="o", lw=2, color=BLUE, label="Last-year share")
    ax.plot(x, df.blended_share, marker="o", lw=2, color=ORANGE, label="Blended share")
    ax.axhline(SEASONAL_NAIVE, color=GREEN, ls="--", lw=2, label="Seasonal naive")
    best = int(df.last_year_share.idxmin()); ax.scatter(best, df.last_year_share.iloc[best], color=RED, s=130, zorder=5)
    ax.set_xticks(x); ax.set_xticklabels(df.order, rotation=55, ha="right", fontsize=9)
    ax.set(title="All 18 ARIMA orders × two allocation strategies", xlabel="ARIMA order (p,d,q)", ylabel="Validation WMAE")
    ax.grid(alpha=.2); ax.legend(); save(fig, "03_all_orders_and_allocations.png")


def allocation_effect():
    df = results_frame(); delta = df.blended_share - df.last_year_share
    fig, ax = plt.subplots(figsize=(14, 6)); bars = ax.bar(df.order, delta, color=RED)
    ax.set(title="Blended allocation was worse for every ARIMA order", xlabel="ARIMA order", ylabel="WMAE increase vs last-year share")
    ax.set_xticks(np.arange(len(df))); ax.set_xticklabels(df.order, rotation=55, ha="right", fontsize=9); ax.grid(axis="y", alpha=.2)
    ax.text(.5, .92, f"Average penalty: +{delta.mean():,.1f} WMAE", transform=ax.transAxes, ha="center", weight="bold", color=RED)
    save(fig, "04_allocation_strategy_effect.png")


def differencing():
    df = results_frame(); means = df.groupby("d")["last_year_share"].mean(); medians = df.groupby("d")["last_year_share"].median()
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    axes[0].bar(["d=0\nno differencing", "d=1\none difference"], means.values, color=[GREEN, RED]); axes[0].set_title("Mean WMAE across orders")
    axes[1].bar(["d=0\nno differencing", "d=1\none difference"], medians.values, color=[GREEN, RED]); axes[1].set_title("Median WMAE across orders")
    for ax, values in zip(axes, [means.values, medians.values]):
        ax.set_ylabel("Validation WMAE"); ax.grid(axis="y", alpha=.2)
        for i, value in enumerate(values): ax.text(i, value + 35, f"{value:,.1f}", ha="center", weight="bold")
    fig.suptitle("Differencing sensitivity in the pure ARIMA search", fontsize=19, weight="bold")
    save(fig, "05_differencing_sensitivity.png")


def arima_vs_arimax():
    df = results_frame(); x = np.arange(len(df)); width=.38
    fig, ax = plt.subplots(figsize=(16, 7)); ax.bar(x-width/2, df.last_year_share, width, color=BLUE, label="Pure ARIMA")
    ax.bar(x+width/2, df.arimax_last_year, width, color=RED, label="ARIMAX with aggregate exogenous features")
    ax.set_xticks(x); ax.set_xticklabels(df.order, rotation=55, ha="right", fontsize=9)
    ax.set(title="Adding aggregate exogenous variables degraded every tested order", xlabel="Order", ylabel="Validation WMAE")
    ax.grid(axis="y", alpha=.2); ax.legend(); save(fig, "06_arima_vs_arimax_by_order.png")


def best_metrics():
    rows = [["Selected order", "(1, 0, 2)"], ["Allocation", "last_year_share"],
            ["Validation WMAE", "1,829.88"], ["Validation MAE", "1,840.65"],
            ["Validation RMSE", "3,937.05"], ["Baseline ARIMA WMAE", "1,856.86"],
            ["Gain vs ARIMA baseline", "26.98 WMAE / 1.45%"],
            ["Gap vs seasonal naive", "+29.71 WMAE / 1.65% worse"]]
    fig, ax = plt.subplots(figsize=(10, 7)); ax.axis("off")
    table=ax.table(cellText=rows,colLabels=["Best experiment item","Recorded value"],cellLoc="left",colLoc="left",loc="center")
    table.auto_set_font_size(False); table.set_fontsize(12); table.scale(1,1.7)
    for (r,_),cell in table.get_celld().items():
        cell.set_edgecolor("#CBD5E1"); cell.set_facecolor(BLUE if r==0 else ("#F8FAFC" if r%2==0 else "white"))
        if r==0: cell.set_text_props(color="white",weight="bold")
    ax.set_title("Best pure ARIMA experiment",fontsize=20,weight="bold"); save(fig,"07_best_model_table.png")


def search_space():
    fig, ax=plt.subplots(figsize=(15,5)); ax.axis("off")
    steps=["p ∈ {0,1,2}\nAR lags","d ∈ {0,1}\ndifferencing","q ∈ {0,1,2}\nMA lags","18 ARIMA orders","2 allocation methods","36 evaluated\ncombinations"]
    xs=np.linspace(.07,.93,len(steps))
    for i,(x,label) in enumerate(zip(xs,steps)):
        ax.text(x,.5,label,ha="center",va="center",transform=ax.transAxes,bbox=dict(boxstyle="round,pad=.65",fc="#EFF6FF",ec=BLUE,lw=2))
        if i<len(steps)-1: ax.annotate("",xy=(xs[i+1]-.06,.5),xytext=(x+.06,.5),xycoords=ax.transAxes,arrowprops=dict(arrowstyle="->",lw=2,color=SLATE))
    ax.set_title("Pure ARIMA experiment search design",fontsize=20,weight="bold"); save(fig,"08_search_space_flow.png")


def selection_decision():
    fig, ax=plt.subplots(figsize=(14,6)); ax.axis("off")
    ax.text(.16,.55,"Best tuned ARIMA\n(1,0,2)\nWMAE 1,829.88",ha="center",va="center",transform=ax.transAxes,bbox=dict(boxstyle="round,pad=.8",fc="#EFF6FF",ec=BLUE,lw=3))
    ax.text(.50,.55,"Seasonal naive\nWMAE 1,800.17",ha="center",va="center",transform=ax.transAxes,bbox=dict(boxstyle="round,pad=.8",fc="#F0FDF4",ec=GREEN,lw=3))
    ax.annotate("still +29.71 worse",xy=(.40,.55),xytext=(.27,.55),xycoords=ax.transAxes,ha="center",color=RED,arrowprops=dict(arrowstyle="->",lw=2,color=RED))
    ax.text(.84,.55,"Conclusion\nOrder tuning helps, but aggregate\nARIMA is not the winning model",ha="center",va="center",transform=ax.transAxes,bbox=dict(boxstyle="round,pad=.8",fc="#FEF2F2",ec=RED,lw=3))
    ax.annotate("",xy=(.73,.55),xytext=(.60,.55),xycoords=ax.transAxes,arrowprops=dict(arrowstyle="->",lw=2,color=SLATE))
    ax.set_title("Model-selection decision",fontsize=20,weight="bold"); save(fig,"09_model_selection_decision.png")


def limitations():
    fig, ax=plt.subplots(figsize=(15,5)); ax.axis("off")
    labels=["143 weekly totals\nonly", "Store–Dept detail enters\nafter forecasting", "No explicit seasonal\nARIMA component", "ARIMAX aggregate features\ndistorted the signal"]
    for x,label in zip(np.linspace(.12,.88,4),labels):
        ax.text(x,.5,label,ha="center",va="center",transform=ax.transAxes,fontsize=12,bbox=dict(boxstyle="round,pad=.75",fc="#FEF2F2",ec=RED,lw=2.4))
    ax.set_title("Why tuning produced only a small improvement",fontsize=20,weight="bold"); save(fig,"10_experiment_limitations.png")


if __name__ == "__main__":
    evolution(); leaderboard(); all_orders(); allocation_effect(); differencing(); arima_vs_arimax(); best_metrics(); search_space(); selection_decision(); limitations()
    print(f"Generated ARIMA experiment figures in {OUT}")
