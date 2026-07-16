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


def save(fig, name, rect=None):
    fig.tight_layout(rect=rect)
    fig.savefig(OUT / name, dpi=230, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def learning_curves():
    # Trial 1 is the best Optuna trial (validation WMAE = 1,567.70).
    # The notebook logged checkpoints every 100 rounds, not every iteration.
    rounds, train, valid = CURVES[1]
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    axes[0].plot(rounds, train, color=BLUE, lw=2.5, marker="o", label="Train weighted MAE")
    axes[0].plot(rounds, valid, color=ORANGE, lw=2.5, marker="o", label="Validation weighted MAE")
    axes[0].fill_between(rounds, train, valid, color=RED, alpha=.08, label="Generalization gap")
    axes[0].set(title="Full learning curve", xlabel="Boosting round", ylabel="Weighted MAE")
    axes[0].grid(alpha=.2); axes[0].legend()

    axes[1].plot(rounds, valid, color=ORANGE, lw=2.8, marker="o")
    axes[1].scatter([1200], [1567.76], color=GREEN, s=90, zorder=5)
    axes[1].annotate(
        "Best logged checkpoint\n1,567.76",
        xy=(1200, 1567.76), xytext=(820, 1585),
        arrowprops={"arrowstyle": "->", "color": GREEN}, color=GREEN, weight="bold",
    )
    axes[1].set_ylim(1560, 1660)
    axes[1].set(title="Validation curve — zoomed scale", xlabel="Boosting round", ylabel="Validation weighted MAE")
    axes[1].grid(alpha=.2)

    fig.suptitle("Best LightGBM Optuna trial (Trial 1) — checkpoints logged every 100 rounds", fontsize=17, weight="bold")
    fig.text(.5, -.01, "Validation improves by about 84 WMAE, then plateaus while training error keeps falling.", ha="center", color=RED, weight="bold")
    save(fig, "01_best_optuna_trial_learning_curve.png")


def trial_comparison():
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar([f"Trial {i}" for i in range(4)], TRIAL_WMAE, color=[BLUE, GREEN, BLUE, BLUE])
    ax.set_ylim(1540, 1640); ax.set_ylabel("Validation WMAE — lower is better")
    ax.set_title("Optuna trial leaderboard", fontsize=18, weight="bold")
    for bar, value in zip(bars, TRIAL_WMAE): ax.text(bar.get_x()+bar.get_width()/2, value+2, f"{value:.1f}", ha="center")
    save(fig, "02_optuna_trial_leaderboard.png")


def experiment_evolution():
    labels = ["Unsafe\nlag/rolling", "Safe\nSalesLag52", "Corrected\nsafe FE", "Final tuned +\nfull-data refit"]
    validation = [1573.4988, 1633.3693, 1615.4495, 1575.1545]
    kaggle = [6200, 3600, 3490, 2809]
    colors = [RED, GREEN, GREEN, PURPLE]
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    axes[0].bar(labels, validation, color=colors); axes[0].set_ylim(1530, 1670); axes[0].set_title("Validation WMAE")
    axes[1].bar(labels, kaggle, color=colors); axes[1].set_title("Kaggle WMAE")
    for ax, vals in zip(axes, [validation, kaggle]):
        ax.set_ylabel("WMAE — lower is better"); ax.grid(axis="y", alpha=.2)
        for i,v in enumerate(vals): ax.text(i,v+(4 if ax is axes[0] else 80),f"{v:.1f}",ha="center")
    fig.suptitle("Feature-engineering experiment evolution\nRed = inference-unsafe, green/purple = inference-safe", fontsize=17, weight="bold")
    save(fig, "03_validation_vs_kaggle_experiments.png")


def lag_transition():
    fig, ax = plt.subplots(figsize=(16, 7)); ax.axis("off")
    stages = [
        ("1. არაუსაფრთხო lag მოდელი", [
            "lag_1, lag_4, lag_13, lag_52",
            "rolling mean/std: 4 და 13",
            "ფიქსირებული n_estimators = 100",
            "validation შედეგი ძლიერი ჩანდა",
            "Kaggle ≈ 6200",
        ], "#FEE2E2", RED),
        ("2. უსაფრთხო SalesLag52 მოდელი", [
            "ამოვიღეთ მოკლე lag-ები და rolling",
            "დავამატეთ exact-date SalesLag52",
            "დავამატეთ SalesLag52_available",
            "კვლავ ფიქსირებული n_estimators = 100",
            "Kaggle ≈ 3490–3600",
        ], "#DCFCE7", GREEN),
        ("3. საბოლოო საუკეთესო მოდელი", [
            "დავტოვეთ inference-safe features",
            "დავამატეთ aggregate count features",
            "მაქს. 1200 round + early stopping",
            "საუკეთესო iteration = 844 + full-data refit",
            "Kaggle = 2809",
        ], "#EDE9FE", PURPLE),
    ]
    xs = [.18, .50, .82]
    for i, (x, (title, lines, face, edge)) in enumerate(zip(xs, stages)):
        text = title + "\n\n" + "\n".join(f"• {line}" for line in lines)
        ax.text(x, .52, text, transform=ax.transAxes, ha="center", va="center", fontsize=13,
                bbox=dict(boxstyle="round,pad=1.2", facecolor=face, edgecolor=edge, linewidth=3))
        if i < 2:
            ax.annotate("", xy=(xs[i+1]-.15,.52), xytext=(x+.15,.52), xycoords=ax.transAxes,
                        arrowprops=dict(arrowstyle="->", lw=3, color=SLATE))
    ax.set_title("LightGBM მოდელის განვითარება: უსაფრთხო features და მეტი boosting capacity", fontsize=19, weight="bold", pad=25)
    ax.text(.5,.05,"მთავარი გაუმჯობესება: მიუწვდომელი future-sales features ამოვიღეთ, ხოლო 100 ხის ლიმიტი early stopping-ით და full-data refit-ით ჩავანაცვლეთ.",transform=ax.transAxes,ha="center",color=SLATE,weight="bold",fontsize=12)
    save(fig, "04_three_stage_model_evolution.png")


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


def best_hyperparameters():
    fig, ax = plt.subplots(figsize=(15, 9)); ax.axis("off")
    groups = [
        ("Boosting", [
            "boosting_type = gbdt",
            "objective = regression_l1",
            "learning_rate = 0.0694",
            "num_leaves = 313",
            "max_depth = 12",
        ], .20, "#EFF6FF", BLUE),
        ("Regularization & sampling", [
            "min_child_samples = 86",
            "subsample = 0.7812",
            "colsample_bytree = 0.7765",
            "reg_alpha = 0.00131",
            "reg_lambda = 0.09842",
            "min_split_gain = 0.12022",
        ], .50, "#F0FDF4", GREEN),
        ("Training budget", [
            "maximum rounds = 1200",
            "early_stopping_rounds = 80",
            "best iteration = 844",
            "final refit rounds = 844",
            "holiday sample weight = 5",
            "random_state = 42",
        ], .80, "#F5F3FF", PURPLE),
    ]
    for title, lines, x, face, edge in groups:
        text = title + "\n\n" + "\n".join(lines)
        ax.text(x, .54, text, transform=ax.transAxes, ha="center", va="center", fontsize=14,
                linespacing=1.55, bbox=dict(boxstyle="round,pad=1.2", facecolor=face, edgecolor=edge, linewidth=3))
    ax.set_title("Best LightGBM hyperparameters selected by Optuna", fontsize=21, weight="bold", pad=25)
    ax.text(.5,.08,"Optuna searched with a maximum budget of 1,200 rounds; early stopping selected 844 rounds for the final full-data model.",transform=ax.transAxes,ha="center",color=SLATE,weight="bold",fontsize=13)
    save(fig,"08_best_hyperparameters.png")


def hyperparameter_sensitivity():
    # The final run requested more trials, but only four completed before timeout.
    trials = {
        "learning_rate": [0.088655, 0.069402, 0.092937, 0.072979],
        "num_leaves": [253, 313, 163, 206],
        "max_depth": [11, 12, 13, 12],
        "min_child_samples": [83, 86, 100, 58],
        "subsample": [0.813683, 0.781204, 0.792468, 0.808429],
        "colsample_bytree": [0.824907, 0.776519, 0.780910, 0.812282],
        "reg_alpha": [0.005531, 0.001307, 0.002327, 0.008168],
        "reg_lambda": [0.039523, 0.098423, 0.005013, 0.064079],
        "min_split_gain": [0.0, 0.120223, 0.104951, 0.039935],
    }
    wmae = np.array(TRIAL_WMAE)
    fig, axes = plt.subplots(3, 3, figsize=(16, 13))
    for ax, (name, values) in zip(axes.flat, trials.items()):
        values = np.asarray(values)
        order = np.argsort(values)
        ax.plot(values[order], wmae[order], color=SLATE, alpha=.45, lw=1.5)
        ax.scatter(values, wmae, color=BLUE, s=75)
        ax.scatter([values[1]], [wmae[1]], color=GREEN, s=120, zorder=4, label="Best: Trial 1")
        for trial_number, (x, y) in enumerate(zip(values, wmae)):
            ax.annotate(f"T{trial_number}", (x, y), xytext=(4,5), textcoords="offset points", fontsize=9)
        ax.set(title=name, ylabel="Validation WMAE")
        ax.grid(alpha=.2); ax.legend(fontsize=8)
    fig.suptitle("LightGBM hyperparameter sensitivity — 4 completed trials", fontsize=19, weight="bold")
    fig.text(.5,.018,"20 trials were requested, but only 4 completed before timeout. Treat these plots as directional, not statistically robust sensitivity estimates.",ha="center",color=RED,weight="bold",fontsize=12)
    save(fig,"09_hyperparameter_sensitivity_4_of_20_trials.png", rect=[0,.06,1,.96])


def hyperparameter_table():
    rows = [
        ["boosting_type", "gbdt", "Gradient-boosted decision trees"],
        ["objective", "regression_l1", "Optimize absolute error"],
        ["learning_rate", "0.0694", "Contribution of each new tree"],
        ["num_leaves", "313", "Maximum leaves per tree"],
        ["max_depth", "12", "Maximum tree depth"],
        ["min_child_samples", "86", "Minimum rows in a leaf"],
        ["subsample", "0.7812", "Row fraction per boosting step"],
        ["colsample_bytree", "0.7765", "Feature fraction per tree"],
        ["reg_alpha", "0.00131", "L1 regularization"],
        ["reg_lambda", "0.09842", "L2 regularization"],
        ["min_split_gain", "0.12022", "Minimum gain required for a split"],
        ["maximum boosting rounds", "1200", "Optuna/validation training budget"],
        ["early_stopping_rounds", "80", "Stop after no validation improvement"],
        ["best_iteration", "844", "Rounds used for final full-data refit"],
        ["holiday sample weight", "5", "Match Kaggle WMAE weighting"],
        ["random_state", "42", "Reproducibility"],
    ]
    fig, ax = plt.subplots(figsize=(15, 11)); ax.axis("off")
    table = ax.table(
        cellText=rows,
        colLabels=["Hyperparameter", "Best value", "Purpose"],
        colWidths=[.30, .20, .50],
        cellLoc="left", colLoc="left", loc="center",
    )
    table.auto_set_font_size(False); table.set_fontsize(12); table.scale(1, 1.75)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#CBD5E1")
        if row == 0:
            cell.set_facecolor(BLUE); cell.set_text_props(color="white", weight="bold")
        elif row % 2 == 0:
            cell.set_facecolor("#F8FAFC")
        if row in (12, 13, 14):
            cell.set_facecolor("#F5F3FF")
    ax.set_title("Best LightGBM hyperparameters", fontsize=22, weight="bold", pad=25)
    fig.text(.5,.035,"Maximum budget = 1,200 rounds; early stopping selected 844 rounds for the final full-data model.",ha="center",color=PURPLE,weight="bold",fontsize=13)
    save(fig,"10_best_hyperparameters_table.png", rect=[0,.06,1,.96])


if __name__ == "__main__":
    learning_curves(); trial_comparison(); experiment_evolution(); lag_transition(); feature_groups(); workflow(); best_hyperparameters(); hyperparameter_sensitivity(); hyperparameter_table()
    print(f"Generated model-experiment figures in {OUT}")
