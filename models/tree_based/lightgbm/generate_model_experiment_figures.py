"""Generate presentation figures for model_experiment_LightGBM.ipynb."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


OUT = Path(__file__).parent / "figures" / "model_experiment"
OUT.mkdir(parents=True, exist_ok=True)
BLUE, GREEN, ORANGE, RED, PURPLE, SLATE = "#2563EB", "#16A34A", "#F59E0B", "#DC2626", "#7C3AED", "#475569"

CURVES = {
    0: ([100,200,300,400,500,600,700,800,900], [1704.94,1479.79,1380.23,1319.87,1270.43,1228,1196.87,1171.9,1149.68], [1661.8,1635.55,1622.85,1617.34,1611.43,1609.67,1606.94,1605.11,1604.7]),
    1: ([100,200,300,400,500,600,700,800,900,1000,1100,1200], [1726.61,1495.58,1381.97,1317.38,1264.26,1221.79,1187.84,1163.28,1139.68,1118.02,1099.13,1082.94], [1652.04,1614.45,1596.74,1590.66,1583.77,1579.54,1576.97,1573.22,1571.26,1570.8,1568.78,1567.76]),
    2: ([100,200,300,400,500,600,700,800], [1781.54,1569.06,1463.78,1400.58,1359.8,1321.3,1294.66,1270.41], [1679.75,1635.46,1622.19,1617.88,1614.8,1613.42,1610.44,1607.18]),
    3: ([100,200,300,400,500,600,700,800,900], [1789.96,1536.76,1423.68,1371.02,1331.76,1294.34,1269.77,1248.69,1227.55], [1692.9,1653.98,1638.11,1631.23,1627.63,1624.33,1620.68,1618.32,1616.75]),
}
TRIAL_WMAE = [1604.5135, 1567.7045, 1607.1228, 1616.6889]


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=230, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def learning_curves():
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    for ax, (trial, (rounds, train, valid)) in zip(axes.flat, CURVES.items()):
        ax.plot(rounds, train, color=BLUE, lw=2.3, label="Train weighted MAE")
        ax.plot(rounds, valid, color=ORANGE, lw=2.3, label="Validation weighted MAE")
        ax.set(title=f"Optuna trial {trial}", xlabel="Boosting round", ylabel="Weighted MAE")
        ax.grid(alpha=.2); ax.legend()
    fig.suptitle("LightGBM model experiment — loss reduction by trial", fontsize=18, weight="bold")
    save(fig, "01_learning_curves_all_trials.png")


def trial_comparison():
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar([f"Trial {i}" for i in range(4)], TRIAL_WMAE, color=[BLUE, GREEN, BLUE, BLUE])
    ax.set_ylim(1540, 1640); ax.set_ylabel("Validation WMAE — lower is better")
    ax.set_title("Optuna trial leaderboard", fontsize=18, weight="bold")
    for bar, value in zip(bars, TRIAL_WMAE): ax.text(bar.get_x()+bar.get_width()/2, value+2, f"{value:.1f}", ha="center")
    save(fig, "02_optuna_trial_leaderboard.png")


def experiment_evolution():
    labels = ["Unsafe\nlag/rolling", "Safe\nSalesLag52", "Corrected\nsafe FE"]
    validation = [1573.4988, 1633.3693, 1615.4495]
    kaggle = [6200, 3600, 3490]
    colors = [RED, GREEN, GREEN]
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    axes[0].bar(labels, validation, color=colors); axes[0].set_ylim(1530, 1670); axes[0].set_title("Validation WMAE")
    axes[1].bar(labels, kaggle, color=colors); axes[1].set_title("Kaggle WMAE")
    for ax, vals in zip(axes, [validation, kaggle]):
        ax.set_ylabel("WMAE — lower is better"); ax.grid(axis="y", alpha=.2)
        for i,v in enumerate(vals): ax.text(i,v+(4 if ax is axes[0] else 80),f"{v:.1f}",ha="center")
    fig.suptitle("Feature-engineering experiment evolution\nRed = inference-unsafe, green = inference-safe", fontsize=17, weight="bold")
    save(fig, "03_validation_vs_kaggle_experiments.png")


def lag_transition():
    features = ["lag_1","lag_4","lag_13","lag_52","rolling_mean_4","rolling_std_4","rolling_mean_13","rolling_std_13","SalesLag52","SalesLag52_available"]
    matrix = np.array([[1,0]]*8 + [[0,1],[0,1]])
    fig, ax = plt.subplots(figsize=(9, 8)); ax.imshow(matrix, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks([0,1],["Old unsafe experiment","Final safe experiment"],rotation=10); ax.set_yticks(range(len(features)),features)
    for r in range(10):
        for c in range(2): ax.text(c,r,"USED" if matrix[r,c] else "REMOVED",ha="center",va="center",fontsize=9)
    ax.set_title("Removal of inference-unsafe lag and rolling features", fontsize=16, weight="bold")
    save(fig, "04_lag_rolling_to_safe_lag52.png")


def feature_groups():
    groups = {
        "Raw/store": 14, "Calendar/cyclical": 10, "Holiday proximity": 12,
        "Markdown/interactions": 19, "Historical aggregates": 16, "Safe lag": 2,
    }
    fig, ax = plt.subplots(figsize=(11, 6.5)); bars=ax.barh(list(groups),list(groups.values()),color=[SLATE,BLUE,ORANGE,PURPLE,GREEN,RED])
    ax.set_title("Engineered feature groups supplied to LightGBM",fontsize=18,weight="bold"); ax.set_xlabel("Feature count / design scope")
    for b,v in zip(bars,groups.values()): ax.text(v+.3,b.get_y()+b.get_height()/2,str(v),va="center")
    ax.text(.5,-.16,"This is feature-set composition, not numerical model importance.",transform=ax.transAxes,ha="center",color=RED,weight="bold")
    save(fig,"05_engineered_feature_groups.png")


def workflow():
    fig, ax = plt.subplots(figsize=(16, 4)); ax.axis("off")
    steps=["Merge raw data","32-week\ntime split","Safe FE\nfit on train","Diagnostic\nimportance","Optuna\n4 trials","Best iteration\n844","Full-data refit\n+ Registry"]
    xs=np.linspace(.07,.93,len(steps))
    for i,(x,label) in enumerate(zip(xs,steps)):
        ax.text(x,.5,label,ha="center",va="center",transform=ax.transAxes,bbox=dict(boxstyle="round,pad=.7",fc="#EFF6FF",ec=BLUE,lw=2))
        if i<len(steps)-1: ax.annotate("",xy=(xs[i+1]-.06,.5),xytext=(x+.06,.5),xycoords=ax.transAxes,arrowprops=dict(arrowstyle="->",lw=2,color=SLATE))
    ax.set_title("LightGBM model-experiment training workflow",fontsize=18,weight="bold")
    save(fig,"06_training_workflow.png")


if __name__ == "__main__":
    learning_curves(); trial_comparison(); experiment_evolution(); lag_transition(); feature_groups(); workflow()
    print(f"Generated model-experiment figures in {OUT}")
