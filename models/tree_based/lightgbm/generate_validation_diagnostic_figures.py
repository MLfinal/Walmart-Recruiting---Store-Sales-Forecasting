"""Train the final LightGBM configuration on the time split and export diagnostics."""

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
BLUE, GREEN, ORANGE, RED, PURPLE, SLATE = "#2563EB", "#16A34A", "#F59E0B", "#DC2626", "#7C3AED", "#475569"

notebook = json.loads((ROOT / "model_experiment_LightGBM.ipynb").read_text())
namespace = {}
exec(compile("".join(notebook["cells"][4]["source"]), "LightGBM.ipynb:cell4", "exec"), namespace)

train = pd.read_csv(DATA / "train.csv", parse_dates=["Date"])
features = pd.read_csv(DATA / "features.csv", parse_dates=["Date"])
stores = pd.read_csv(DATA / "stores.csv")
frame = train.merge(stores, on="Store", how="left").merge(
    features, on=["Store", "Date", "IsHoliday"], how="left"
).sort_values(["Date", "Store", "Dept"]).reset_index(drop=True)

validation_dates = np.sort(frame["Date"].unique())[-32:]
fit = frame.loc[~frame["Date"].isin(validation_dates)].copy()
valid = frame.loc[frame["Date"].isin(validation_dates)].copy()
fit_lag = namespace["add_safe_lag_52"](fit, observed_history=fit)
valid_lag = namespace["add_safe_lag_52"](valid, observed_history=fit)
pipeline = namespace["make_walmart_lgbm_feature_pipeline"](drop_target_and_date=True)
X_fit = pipeline.fit_transform(fit_lag)
X_valid = pipeline.transform(valid_lag)
y_fit, y_valid = fit["Weekly_Sales"], valid["Weekly_Sales"]
w_fit = np.where(fit["IsHoliday"], 5.0, 1.0)
categoricals = X_fit.select_dtypes(include="category").columns.tolist()

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
model.fit(X_fit, y_fit, sample_weight=w_fit, categorical_feature=categoricals)
prediction = np.clip(model.predict(X_valid), 0, None)
result = valid[["Store", "Dept", "Date", "IsHoliday", "Weekly_Sales"]].copy()
result["Prediction"] = prediction
result["AbsoluteError"] = np.abs(result["Weekly_Sales"] - result["Prediction"])
result["Weight"] = np.where(result["IsHoliday"], 5.0, 1.0)


def wmae(part):
    return np.average(part["AbsoluteError"], weights=part["Weight"])


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=230, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# 1. Holiday versus non-holiday WMAE.
groups = ["Non-holiday", "Holiday", "Overall"]
values = [wmae(result[~result["IsHoliday"]]), wmae(result[result["IsHoliday"]]), wmae(result)]
fig, ax = plt.subplots(figsize=(10, 6)); bars = ax.bar(groups, values, color=[BLUE, RED, PURPLE])
ax.set(title="Final LightGBM validation error by holiday status", ylabel="WMAE — lower is better")
for bar, value in zip(bars, values): ax.text(bar.get_x()+bar.get_width()/2,value+20,f"{value:,.1f}",ha="center",weight="bold")
ax.grid(axis="y",alpha=.2); save(fig,"11_holiday_vs_nonholiday_wmae.png")

# 2. Exact holiday-event errors. The 32-week holdout contains Labor Day only.
event_dates = {name: pd.to_datetime(dates) for name, dates in namespace["WalmartHolidayFeatureTransformer"].HOLIDAY_DATES.items()}
event_values, event_labels = [], []
for name in ["SuperBowl", "LaborDay", "Thanksgiving", "Christmas"]:
    part = result[result["Date"].isin(event_dates[name])]
    event_labels.append(name + ("" if len(part) else "\n(no rows)"))
    event_values.append(wmae(part) if len(part) else np.nan)
fig, ax = plt.subplots(figsize=(11, 6)); x=np.arange(4); bars=ax.bar(x,np.nan_to_num(event_values),color=[BLUE,ORANGE,RED,GREEN])
ax.set_xticks(x,event_labels); ax.set(title="Validation WMAE by Walmart holiday event",ylabel="WMAE — lower is better")
for i,(bar,value) in enumerate(zip(bars,event_values)):
    ax.text(bar.get_x()+bar.get_width()/2,(value+20 if np.isfinite(value) else 50),f"{value:,.1f}" if np.isfinite(value) else "N/A",ha="center",weight="bold")
ax.text(.5,-.16,"The chronological validation window is 2012-03-23 to 2012-10-26, so only Labor Day is represented.",transform=ax.transAxes,ha="center",color=SLATE,weight="bold")
ax.grid(axis="y",alpha=.2); save(fig,"12_error_by_holiday_event.png")

# 3. Weekly validation WMAE.
weekly = result.groupby("Date").apply(wmae, include_groups=False)
fig, ax = plt.subplots(figsize=(13, 6)); ax.plot(weekly.index,weekly.values,color=BLUE,lw=2.2,marker="o",ms=4)
ax.axhline(wmae(result),color=RED,ls="--",label=f"Overall WMAE = {wmae(result):,.1f}")
ax.set(title="Final LightGBM validation WMAE over the last 32 weeks",xlabel="Validation week",ylabel="Weekly WMAE")
ax.grid(alpha=.2); ax.legend(); save(fig,"13_weekly_validation_wmae.png")

# 4. Actual versus predicted sales.
sample = result.sample(min(20000,len(result)),random_state=42)
low=min(sample["Weekly_Sales"].min(),sample["Prediction"].min()); high=max(sample["Weekly_Sales"].max(),sample["Prediction"].max())
fig, ax = plt.subplots(figsize=(9, 8)); ax.scatter(sample["Weekly_Sales"],sample["Prediction"],s=8,alpha=.18,color=BLUE)
ax.plot([low,high],[low,high],color=RED,ls="--",lw=2,label="Perfect prediction")
ax.set(title="Final LightGBM: actual vs predicted validation sales",xlabel="Actual Weekly_Sales",ylabel="Predicted Weekly_Sales")
ax.grid(alpha=.2); ax.legend(); save(fig,"14_actual_vs_predicted_validation.png")

result.to_csv(OUT / "final_validation_predictions.csv",index=False)
print(f"Validation WMAE: {wmae(result):.4f}")
print(f"Saved diagnostic figures in {OUT}")
