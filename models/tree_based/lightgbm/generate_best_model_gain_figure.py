"""Reproduce the final LightGBM refit and export top-20 gain importance."""

import json
from pathlib import Path

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).parent
DATA = ROOT.parents[2] / "data"
OUT = ROOT / "figures" / "model_experiment"
OUT.mkdir(parents=True, exist_ok=True)

# Reuse the exact transformer definitions from the experiment notebook.
notebook = json.loads((ROOT / "model_experiment_LightGBM.ipynb").read_text())
feature_definition_source = "".join(notebook["cells"][4]["source"])
namespace = {}
exec(compile(feature_definition_source, "model_experiment_LightGBM.ipynb:cell4", "exec"), namespace)

train = pd.read_csv(DATA / "train.csv", parse_dates=["Date"])
features = pd.read_csv(DATA / "features.csv", parse_dates=["Date"])
stores = pd.read_csv(DATA / "stores.csv")
frame = train.merge(stores, on="Store", how="left").merge(
    features, on=["Store", "Date", "IsHoliday"], how="left"
).sort_values(["Date", "Store", "Dept"]).reset_index(drop=True)

with_lag = namespace["add_safe_lag_52"](frame, observed_history=frame)
pipeline = namespace["make_walmart_lgbm_feature_pipeline"](drop_target_and_date=True)
X = pipeline.fit_transform(with_lag)
y = frame["Weekly_Sales"]
weights = np.where(frame["IsHoliday"].to_numpy(), 5.0, 1.0)
categoricals = X.select_dtypes(include="category").columns.tolist()

# Best Optuna configuration, refitted for the validation-selected 844 rounds.
params = {
    "objective": "regression_l1", "metric": "mae", "boosting_type": "gbdt",
    "n_estimators": 844, "learning_rate": 0.06940238065869553,
    "num_leaves": 313, "max_depth": 12, "min_child_samples": 86,
    "subsample": 0.7812037280884873, "subsample_freq": 1,
    "colsample_bytree": 0.7765190684571545,
    "reg_alpha": 0.0013066739238053278, "reg_lambda": 0.09842315738502598,
    "min_split_gain": 0.12022300234864176, "random_state": 42,
    "n_jobs": -1, "verbosity": -1, "device_type": "cpu",
}
model = lgb.LGBMRegressor(**params)
model.fit(X, y, sample_weight=weights, categorical_feature=categoricals)

gain = model.booster_.feature_importance(importance_type="gain")
importance = (
    pd.DataFrame({"feature": X.columns, "gain": gain})
    .sort_values("gain", ascending=False)
    .reset_index(drop=True)
)
importance.to_csv(OUT / "top20_gain_importance.csv", index=False)

top = importance.head(20).sort_values("gain")
fig, ax = plt.subplots(figsize=(12, 9))
bars = ax.barh(top["feature"], top["gain"], color="#7C3AED")
ax.set_title("LightGBM final best model — Top 20 features by gain", fontsize=18, weight="bold")
ax.set_xlabel("Total gain across tree splits")
ax.grid(axis="x", alpha=.2)
for bar, value in zip(bars, top["gain"]):
    ax.text(value, bar.get_y() + bar.get_height()/2, f" {value:,.0f}", va="center", fontsize=9)
fig.text(.5, .01, "Reproduced full-data refit: safe feature pipeline, 844 boosting rounds, holiday weight = 5", ha="center", color="#475569")
fig.tight_layout(rect=[0, .03, 1, 1])
fig.savefig(OUT / "07_top20_features_by_gain.png", dpi=230, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(importance.head(20).to_string(index=False))
print(f"Saved figure and CSV in {OUT}")
