"""Create the final rich W&B report from existing project runs.

Authentication is read only from WANDB_API_KEY. No credential is persisted by this file.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import wandb
import wandb_workspaces.reports.v2 as wr

ENTITY = "kende23-n-a"
PROJECT = "Walmart-Recruiting---Store-Sales-Forecasting"
REPORT_TITLE = "Walmart Store Sales Forecasting — სრული მოდელური კვლევა"
REPORT_TAG = "final-report-v1"

MODELS = [
    {
        "model": "XGBoost",
        "family": "Tree-based",
        "local": 1612.1265,
        "private": 2806.0,
        "public": np.nan,
        "runtime": 35.0,
        "registered": 1,
    },
    {
        "model": "LightGBM",
        "family": "Tree-based",
        "local": 1575.1545,
        "private": 2809.0,
        "public": np.nan,
        "runtime": 24.0,
        "registered": 1,
    },
    {
        "model": "TimesFM v3",
        "family": "Foundation",
        "local": 1588.8029,
        "private": 2853.40612,
        "public": 2742.68603,
        "runtime": 29.7035,
        "registered": 1,
    },
    {
        "model": "TFT",
        "family": "Deep learning",
        "local": 2379.5014,
        "private": 3058.9828,
        "public": 2979.8606,
        "runtime": 240.0,
        "registered": 1,
    },
    {
        "model": "DLinear",
        "family": "Deep learning",
        "local": 1506.2825,
        "private": 3500.0,
        "public": np.nan,
        "runtime": 18.0,
        "registered": 1,
    },
    {
        "model": "SARIMAX",
        "family": "Classical",
        "local": 2563.6915,
        "private": 3525.0,
        "public": np.nan,
        "runtime": 8.0,
        "registered": 1,
    },
    {
        "model": "SARIMA",
        "family": "Classical",
        "local": 1831.6176,
        "private": 3842.0,
        "public": np.nan,
        "runtime": 4.0,
        "registered": 1,
    },
    {
        "model": "N-BEATS",
        "family": "Deep learning",
        "local": 2157.9829,
        "private": 4700.0,
        "public": np.nan,
        "runtime": 65.0,
        "registered": 1,
    },
    {
        "model": "Prophet v4",
        "family": "Classical",
        "local": 1367.4470,
        "private": np.nan,
        "public": np.nan,
        "runtime": 3.36,
        "registered": 1,
    },
    {
        "model": "ARIMA",
        "family": "Classical",
        "local": 1829.8800,
        "private": np.nan,
        "public": np.nan,
        "runtime": 3.0,
        "registered": 1,
    },
    {
        "model": "XGB+SARIMA",
        "family": "Hybrid",
        "local": 2111.4121,
        "private": np.nan,
        "public": np.nan,
        "runtime": 20.0,
        "registered": 1,
    },
]

COLORS = {
    "Tree-based": "#2a9d8f",
    "Deep learning": "#457b9d",
    "Classical": "#e9c46a",
    "Foundation": "#9b5de5",
    "Hybrid": "#e76f51",
}


def style_axis(ax: plt.Axes, title: str, xlabel: str = "", ylabel: str = "") -> None:
    ax.set_title(title, fontsize=15, fontweight="bold", pad=14)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(axis="both", alpha=0.2)
    ax.spines[["top", "right"]].set_visible(False)


def save_figure(fig: plt.Figure, directory: Path, name: str) -> Path:
    path = directory / f"{name}.png"
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def create_charts(directory: Path) -> dict[str, Path]:
    frame = pd.DataFrame(MODELS)
    charts: dict[str, Path] = {}

    ranked = frame.dropna(subset=["private"]).sort_values("private")
    fig, ax = plt.subplots(figsize=(11, 6))
    bars = ax.barh(
        ranked["model"], ranked["private"], color=[COLORS[x] for x in ranked["family"]]
    )
    ax.invert_yaxis()
    ax.bar_label(bars, fmt="%.1f", padding=4)
    style_axis(ax, "Final / Private Kaggle WMAE Ranking", "WMAE — lower is better")
    ax.set_xlim(2500, ranked["private"].max() * 1.05)
    charts["private_ranking"] = save_figure(fig, directory, "private_ranking")

    public_private = frame.dropna(subset=["public", "private"])
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(public_private))
    width = 0.35
    first = ax.bar(
        x - width / 2, public_private["public"], width, label="Public", color="#00b4d8"
    )
    second = ax.bar(
        x + width / 2,
        public_private["private"],
        width,
        label="Private",
        color="#023e8a",
    )
    ax.set_xticks(x, public_private["model"])
    ax.bar_label(first, fmt="%.1f", padding=3)
    ax.bar_label(second, fmt="%.1f", padding=3)
    ax.legend()
    style_axis(ax, "Public vs Private Kaggle Score", ylabel="WMAE — lower is better")
    charts["public_private"] = save_figure(fig, directory, "public_private")

    local = frame.sort_values("local")
    fig, ax = plt.subplots(figsize=(11, 7))
    bars = ax.barh(
        local["model"], local["local"], color=[COLORS[x] for x in local["family"]]
    )
    ax.invert_yaxis()
    ax.bar_label(bars, fmt="%.1f", padding=3)
    style_axis(
        ax, "Best Documented Local Validation WMAE", "WMAE — splits differ by family"
    )
    charts["local_ranking"] = save_figure(fig, directory, "local_ranking")

    comparable = frame.dropna(subset=["private"])
    fig, ax = plt.subplots(figsize=(9, 6))
    for family, group in comparable.groupby("family"):
        ax.scatter(
            group["local"],
            group["private"],
            s=100,
            label=family,
            color=COLORS[family],
            alpha=0.9,
        )
        for row in group.itertuples():
            ax.annotate(
                row.model,
                (row.local, row.private),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=9,
            )
    ax.legend()
    style_axis(
        ax,
        "Local Validation vs Private Kaggle Transfer",
        "Local WMAE",
        "Private Kaggle WMAE",
    )
    charts["local_vs_kaggle"] = save_figure(fig, directory, "local_vs_kaggle")

    improvements = pd.DataFrame(
        [
            ("XGBoost", 2902.2892, 1612.1265),
            ("LightGBM", 3184.2771, 1575.1545),
            ("DLinear", 1523.2097, 1506.2825),
            ("N-BEATS", 2157.9829, 2157.9829),
            ("TimesFM", 1672.2525, 1588.8029),
            ("ARIMA", 1856.8605, 1829.8800),
        ],
        columns=["model", "baseline", "best"],
    )
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(improvements))
    width = 0.36
    ax.bar(
        x - width / 2,
        improvements["baseline"],
        width,
        label="Baseline",
        color="#adb5bd",
    )
    ax.bar(x + width / 2, improvements["best"], width, label="Best", color="#2a9d8f")
    ax.set_xticks(x, improvements["model"], rotation=20)
    ax.legend()
    style_axis(ax, "Baseline to Best Local Improvement", ylabel="WMAE")
    charts["baseline_improvement"] = save_figure(fig, directory, "baseline_improvement")

    timesfm = pd.DataFrame(
        [
            ("v1 zero-shot", 1672.2525),
            ("v2 blend", 1620.5430),
            ("v3 XReg blend", 1588.8029),
            ("v4 LoRA standalone", 8396.0651),
        ],
        columns=["version", "wmae"],
    )
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(
        timesfm["version"],
        timesfm["wmae"],
        color=["#90e0ef", "#48cae4", "#9b5de5", "#e63946"],
    )
    ax.bar_label(bars, fmt="%.1f", padding=3)
    ax.tick_params(axis="x", rotation=15)
    style_axis(
        ax, "TimesFM Experiment Progression and LoRA Failure", ylabel="Validation WMAE"
    )
    charts["timesfm_versions"] = save_figure(fig, directory, "timesfm_versions")

    candidates = pd.DataFrame(
        [
            ("v3 blend", 1588.8029),
            ("raw", 1672.2525),
            ("residual", 1720.1709),
            ("seasonal naive", 1799.0451),
            ("XReg", 1939.0755),
            ("pure XReg", 1992.9980),
        ],
        columns=["candidate", "wmae"],
    ).sort_values("wmae")
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(candidates["candidate"], candidates["wmae"], color="#9b5de5")
    ax.invert_yaxis()
    ax.bar_label(bars, fmt="%.1f", padding=3)
    style_axis(ax, "TimesFM v3 Candidate Comparison", "Validation WMAE")
    charts["timesfm_candidates"] = save_figure(fig, directory, "timesfm_candidates")

    dlinear = pd.DataFrame(
        [
            ("baseline", 1523.2097),
            ("v1 calibration", 1506.2825),
            ("v2 104w+calendar", 1961.4508),
            ("v3 gated calendar", 1511.9733),
            ("v4 embeddings", 1542.8344),
            ("v5 tuned", 1507.4388),
            ("v6 external", 1548.0344),
        ],
        columns=["version", "wmae"],
    )
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(
        dlinear["version"], dlinear["wmae"], marker="o", linewidth=2.5, color="#457b9d"
    )
    for row in dlinear.itertuples():
        ax.annotate(
            f"{row.wmae:.1f}",
            (row.version, row.wmae),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
        )
    ax.tick_params(axis="x", rotation=25)
    style_axis(ax, "DLinear Version Progression", ylabel="Best validation WMAE")
    charts["dlinear_versions"] = save_figure(fig, directory, "dlinear_versions")

    nbeats = pd.DataFrame(
        [
            ("baseline", 2157.9829),
            ("lower LR", 2186.5015),
            ("78w context", 2662.8061),
            ("holiday loss", 2185.1366),
            ("best partial Optuna", 2191.4117),
        ],
        columns=["version", "wmae"],
    )
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(nbeats["version"], nbeats["wmae"], color="#457b9d")
    ax.bar_label(bars, fmt="%.1f", padding=3)
    ax.tick_params(axis="x", rotation=20)
    style_axis(
        ax, "N-BEATS: Added Complexity Did Not Beat Baseline", ylabel="Validation WMAE"
    )
    charts["nbeats_versions"] = save_figure(fig, directory, "nbeats_versions")

    coverage = pd.DataFrame(
        [("TFT model", 77248), ("Seasonal fallback", 37816)], columns=["source", "rows"]
    )
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.pie(
        coverage["rows"],
        labels=coverage["source"],
        autopct="%1.2f%%",
        startangle=90,
        colors=["#457b9d", "#adb5bd"],
    )
    ax.set_title("TFT Final Submission Coverage", fontsize=15, fontweight="bold")
    charts["tft_coverage"] = save_figure(fig, directory, "tft_coverage")

    classical = pd.DataFrame(
        [
            ("Prophet", 1367.4470),
            ("ARIMA", 1829.8800),
            ("SARIMA", 1831.6176),
            ("SARIMAX", 2563.6915),
        ],
        columns=["model", "local"],
    )
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(classical["model"], classical["local"], color="#e9c46a")
    ax.bar_label(bars, fmt="%.1f", padding=3)
    style_axis(ax, "Classical Models — Best Local WMAE", ylabel="WMAE")
    charts["classical_local"] = save_figure(fig, directory, "classical_local")

    return charts


def log_standardized_runs() -> None:
    for index, item in enumerate(MODELS):
        run = wandb.init(
            entity=ENTITY,
            project=PROJECT,
            id=f"rptm{index:04d}",
            resume="allow",
            name=f"report-summary-{item['model'].lower().replace(' ', '-').replace('+', '-')}",
            group=REPORT_TAG,
            job_type="report_standardized_metrics",
            tags=[REPORT_TAG, "report-data", item["family"].lower().replace(" ", "-")],
            config={"report/model": item["model"], "report/family": item["family"]},
            reinit="finish_previous",
        )
        metrics = {
            "report/local_wmae": item["local"],
            "report/runtime_minutes": item["runtime"],
            "report/pipeline_registered": item["registered"],
        }
        if np.isfinite(item["private"]):
            metrics["report/private_kaggle_wmae"] = item["private"]
        if np.isfinite(item["public"]):
            metrics["report/public_kaggle_wmae"] = item["public"]
        run.log(metrics)
        run.summary.update(metrics)
        run.finish()


def log_visual_assets(charts: dict[str, Path]) -> None:
    run = wandb.init(
        entity=ENTITY,
        project=PROJECT,
        id="rptmedia",
        resume="allow",
        name="final-report-visual-assets",
        group=REPORT_TAG,
        job_type="report_visual_assets",
        tags=[REPORT_TAG, "report-media"],
        reinit="finish_previous",
    )
    for name, path in charts.items():
        run.log({f"report_charts/{name}": wandb.Image(str(path))})
    table = wandb.Table(dataframe=pd.DataFrame(MODELS).replace({np.nan: None}))
    run.log({"report_tables/model_comparison": table})
    run.finish()


def selected_runset(name: str, run_names: list[str]) -> wr.Runset:
    quoted = ", ".join(repr(value) for value in run_names)
    return wr.Runset(
        entity=ENTITY,
        project=PROJECT,
        name=name,
        filters=f"Metric('displayName') in [{quoted}]",
    )


def build_report() -> wr.Report:
    standardized = wr.Runset(
        entity=ENTITY,
        project=PROJECT,
        name="Standardized final results",
        filters="Metric('jobType') in ['report_standardized_metrics']",
    )
    media = wr.Runset(
        entity=ENTITY,
        project=PROJECT,
        name="Curated report charts",
        filters="Metric('jobType') in ['report_visual_assets']",
    )
    dlinear = selected_runset(
        "DLinear versions",
        [
            "dlinear_baseline_39w",
            "dlinear_v1_52w_series_calibration",
            "dlinear_v2_104w_series_calendar",
            "dlinear_v3_52w_gated_calendar",
            "dlinear_v4_52w_store_dept_embeddings",
            "dlinear_v5_52w_tuned_series_calibration",
            "dlinear_v6_52w_external_covariates",
        ],
    )
    nbeats = selected_runset(
        "N-BEATS experiments",
        [
            "NBEATS_Baseline_Training_No_FE",
            "NBEATS_Experiment_01_Lower_LR_Early_Stopping",
            "NBEATS_Experiment_02_Context_78",
            "NBEATS_Experiment_03_Holiday_Weighted_Loss",
        ],
    )
    tft = selected_runset(
        "TFT experiments",
        [
            "tft_baseline_39w",
            "tft_v1_external_covariates",
            "tft_v2_log_target_external_covariates",
            "tft_v3_seasonal_residual_external_covariates",
        ],
    )
    timesfm = selected_runset(
        "TimesFM experiments",
        [
            "timesfm_v1_zero_shot_all_series_39w",
            "timesfm_v2_zero_shot_residual_calibrated_blend",
            "timesfm_v3_xreg_covariates_corrected_blend",
            "timesfm_v3_1_xreg_ablation_stability_audit",
            "timesfm_v4_lora_xreg_hybrid",
        ],
    )
    prophet = selected_runset(
        "Prophet experiments",
        [
            "prophet_baseline_top300_validation",
            "prophet_v1_external_covariates_all_series_validation",
            "prophet_v2_seasonal_residual_all_series_validation",
            "prophet_v3_seasonal_naive_prophet_blend_all_series_validation",
            "prophet_v4_event_aware_holiday_windows_all_series_validation",
            "prophet_v5_clean_covariates_regularized_calendar_all_series_validation",
            "prophet_v6_historical_alpha_tuning_all_series",
        ],
    )
    classical = selected_runset(
        "ARIMA and SARIMA family",
        [
            "ARIMA_Baseline_Aggregate",
            "ARIMA_Order_Allocation_Experiment",
            "ARIMAX_Order_Exog_Allocation_Experiment",
            "SARIMA_Baseline_Aggregate",
            "SARIMA_Order_Allocation_Experiment",
            "SARIMAX_Order_Exog_Allocation_Experiment",
        ],
    )
    hybrid = selected_runset("XGBoost + SARIMA", ["xgboost-sarima-baseline"])
    pipeline_runs = wr.Runset(
        entity=ENTITY,
        project=PROJECT,
        name="Pipeline registration and inference",
        filters="Metric('jobType') in ['model_registration', 'model-registration', 'pipeline_packaging', 'inference', 'timesfm_registry_pipeline_inference', 'prophet_raw_pipeline_inference']",
    )

    blocks = [
        wr.TableOfContents(),
        wr.CalloutBlock(
            "მთავარი შედეგი: XGBoost private/final WMAE 2806. TimesFM საუკეთესო non-tree მოდელია — public 2742.68603, private 2853.40612."
        ),
        wr.H1("1. ამოცანა, მონაცემები და შეფასება"),
        wr.MarkdownBlock(
            text="""ჩვენ ვპროგნოზირებთ ყოველკვირეულ `Weekly_Sales`-ს Store–Dept დონეზე. Train შეიცავს 421,570 row-ს, 45 store-ს, 81 department-ს და 3,331 historical series-ს; test — 115,064 row-ს და 3,169 series-ს. მთავარი metric არის WMAE: holiday კვირის absolute error-ს weight 5 აქვს, სხვა კვირას — 1. ამიტომ ყველა model-selection გადაწყვეტილება დაბალ WMAE-ს ეყრდნობა. Random split არ გამოიყენება; validation ყოველთვის ქრონოლოგიურია."""
        ),
        wr.H2("Executive scorecard"),
        wr.PanelGrid(
            runsets=[standardized],
            hide_run_sets=True,
            panels=[
                wr.BarPlot(
                    title="Private / final Kaggle WMAE",
                    metrics=["report/private_kaggle_wmae"],
                    orientation="h",
                    max_runs_to_show=12,
                    layout=wr.Layout(w=12, h=8),
                ),
                wr.BarPlot(
                    title="Best documented local WMAE",
                    metrics=["report/local_wmae"],
                    orientation="h",
                    max_runs_to_show=12,
                    layout=wr.Layout(w=12, h=8),
                ),
                wr.ScatterPlot(
                    title="Local validation vs private Kaggle",
                    x="report/local_wmae",
                    y="report/private_kaggle_wmae",
                    regression=True,
                    layout=wr.Layout(w=12, h=8),
                ),
                wr.BarPlot(
                    title="Approximate runtime comparison",
                    metrics=["report/runtime_minutes"],
                    orientation="h",
                    max_runs_to_show=12,
                    layout=wr.Layout(w=12, h=8),
                ),
            ],
        ),
        wr.H2("Curated cross-model graphics"),
        wr.PanelGrid(
            runsets=[media],
            hide_run_sets=True,
            panels=[
                wr.MediaBrowser(
                    title="Private ranking",
                    media_keys=["report_charts/private_ranking"],
                    num_columns=1,
                    layout=wr.Layout(w=12, h=8),
                ),
                wr.MediaBrowser(
                    title="Public vs private",
                    media_keys=["report_charts/public_private"],
                    num_columns=1,
                    layout=wr.Layout(w=12, h=8),
                ),
                wr.MediaBrowser(
                    title="Local ranking",
                    media_keys=["report_charts/local_ranking"],
                    num_columns=1,
                    layout=wr.Layout(w=12, h=8),
                ),
                wr.MediaBrowser(
                    title="Transfer gap",
                    media_keys=["report_charts/local_vs_kaggle"],
                    num_columns=1,
                    layout=wr.Layout(w=12, h=8),
                ),
            ],
        ),
        wr.H1("2. Tree-based მოდელები — XGBoost და LightGBM"),
        wr.MarkdownBlock(
            text="""Tree-based family-მ mixed tabular context ყველაზე ეფექტურად გამოიყენა: Store/Dept IDs, calendar, holiday proximity, Type/Size, markdown missingness, economic variables და leakage-safe SalesLag52. ძველი LightGBM short-lag setup local validation-ზე ოპტიმისტური იყო და Kaggle-ზე ჩავარდა; safe Lag52 + full refit + raw-input parity-მ LightGBM 2809-მდე მიიყვანა. XGBoost 2806-ით ფორმალურად ლიდერია, სხვაობა მხოლოდ 3 WMAE-ია."""
        ),
        wr.PanelGrid(
            runsets=[selected_runset("XGBoost baseline", ["xgboost-static-baseline"])],
            panels=[
                wr.LinePlot(
                    title="XGBoost learning curve",
                    x="epoch",
                    y=["validation_0-mae", "validation_1-mae"],
                    smoothing_factor=0.2,
                    layout=wr.Layout(w=12, h=7),
                ),
                wr.RunComparer(diff_only="split", layout=wr.Layout(w=12, h=7)),
            ],
        ),
        wr.PanelGrid(
            runsets=[
                wr.Runset(
                    entity=ENTITY,
                    project=PROJECT,
                    name="LightGBM tuning",
                    filters="Metric('displayName') in ['lightgbm-optuna-trial-0', 'lightgbm-optuna-trial-1', 'lightgbm-optuna-trial-2', 'lightgbm-optuna-trial-3']",
                )
            ],
            panels=[
                wr.BarPlot(
                    title="LightGBM trial validation WMAE",
                    metrics=["best_validation_weighted_mae"],
                    max_runs_to_show=20,
                    layout=wr.Layout(w=12, h=7),
                ),
                wr.ParameterImportancePlot(
                    with_respect_to="best_validation_weighted_mae",
                    layout=wr.Layout(w=12, h=7),
                ),
            ],
        ),
        wr.PanelGrid(
            runsets=[media],
            hide_run_sets=True,
            panels=[
                wr.MediaBrowser(
                    title="Baseline → champion improvement",
                    media_keys=["report_charts/baseline_improvement"],
                    num_columns=1,
                    layout=wr.Layout(w=24, h=9),
                )
            ],
        ),
        wr.H1("3. Deep learning — DLinear, N-BEATS და TFT"),
        wr.MarkdownBlock(
            text="""DLinear local champion იყო (1506.2825), მაგრამ Kaggle-ზე დაახლოებით 3500 მიიღო. N-BEATS-ის დამატებითი capacity baseline-ს ვერ აჯობა და Kaggle დაახლოებით 4700 იყო. TFT ყველაზე ძვირი და რთული neural workflow აღმოჩნდა, მაგრამ known-future/static covariates და seasonal fallback უკეთ გადაიტანა: public 2979.86060, private 3058.98280."""
        ),
        wr.H2("DLinear — versions and training curves"),
        wr.PanelGrid(
            runsets=[dlinear],
            panels=[
                wr.LinePlot(
                    title="DLinear validation WMAE by epoch",
                    x="epoch",
                    y=["validation/wmae"],
                    smoothing_factor=0.15,
                    max_runs_to_show=10,
                    layout=wr.Layout(w=12, h=8),
                ),
                wr.LinePlot(
                    title="DLinear normalized train/validation loss",
                    x="epoch",
                    y=["train/normalized_wmae_loss", "validation/normalized_wmae_loss"],
                    smoothing_factor=0.15,
                    max_runs_to_show=10,
                    layout=wr.Layout(w=12, h=8),
                ),
            ],
        ),
        wr.PanelGrid(
            runsets=[media],
            hide_run_sets=True,
            panels=[
                wr.MediaBrowser(
                    title="DLinear experiment progression",
                    media_keys=["report_charts/dlinear_versions"],
                    num_columns=1,
                    layout=wr.Layout(w=24, h=9),
                )
            ],
        ),
        wr.H2("N-BEATS — why tuning did not beat baseline"),
        wr.PanelGrid(
            runsets=[nbeats],
            panels=[
                wr.LinePlot(
                    title="N-BEATS validation WMAE",
                    x="epoch",
                    y=["validation/weighted_mae"],
                    smoothing_factor=0.15,
                    max_runs_to_show=10,
                    layout=wr.Layout(w=12, h=8),
                ),
                wr.LinePlot(
                    title="N-BEATS validation MAE",
                    x="epoch",
                    y=["validation/mae"],
                    smoothing_factor=0.15,
                    max_runs_to_show=10,
                    layout=wr.Layout(w=12, h=8),
                ),
            ],
        ),
        wr.PanelGrid(
            runsets=[media],
            hide_run_sets=True,
            panels=[
                wr.MediaBrowser(
                    title="N-BEATS experiments",
                    media_keys=["report_charts/nbeats_versions"],
                    num_columns=1,
                    layout=wr.Layout(w=24, h=9),
                )
            ],
        ),
        wr.H2("TFT — covariates, residual learning and fallback"),
        wr.PanelGrid(
            runsets=[tft],
            panels=[
                wr.LinePlot(
                    title="TFT validation loss",
                    x="epoch",
                    y=["val_loss"],
                    smoothing_factor=0.15,
                    max_runs_to_show=10,
                    layout=wr.Layout(w=12, h=8),
                ),
                wr.LinePlot(
                    title="TFT training loss",
                    x="epoch",
                    y=["train_loss_epoch"],
                    smoothing_factor=0.15,
                    max_runs_to_show=10,
                    layout=wr.Layout(w=12, h=8),
                ),
            ],
        ),
        wr.PanelGrid(
            runsets=[media],
            hide_run_sets=True,
            panels=[
                wr.MediaBrowser(
                    title="TFT model vs fallback coverage",
                    media_keys=["report_charts/tft_coverage"],
                    num_columns=1,
                    layout=wr.Layout(w=24, h=9),
                )
            ],
        ),
        wr.H1("4. Classical forecasting — ARIMA, SARIMA, SARIMAX, Prophet"),
        wr.MarkdownBlock(
            text="""ARIMA/SARIMA aggregate weekly total-ს პროგნოზირებდა და Store–Dept rows-ზე historical shares-ით ანაწილებდა, ამიტომ row-level heterogeneity იკარგებოდა. ARIMA local 1829.8800; SARIMA local 1831.6176 და Kaggle 3842; SARIMAX local 2563.6915, მაგრამ Kaggle 3525. Prophet per-series trend/seasonality/event structure-ით local 1367.4470-მდე მივიდა — საუკეთესო local score, თუმცა documented leaderboard score არ გვაქვს."""
        ),
        wr.PanelGrid(
            runsets=[classical],
            panels=[
                wr.BarPlot(
                    title="Classical validation WMAE",
                    metrics=[
                        "validation/wmae",
                        "validation/weighted_mae",
                        "best_validation_wmae",
                    ],
                    max_runs_to_show=12,
                    layout=wr.Layout(w=12, h=8),
                ),
                wr.RunComparer(diff_only="split", layout=wr.Layout(w=12, h=8)),
            ],
        ),
        wr.PanelGrid(
            runsets=[prophet],
            panels=[
                wr.BarPlot(
                    title="Prophet version WMAE",
                    metrics=[
                        "validation/wmae",
                        "validation/raw_prophet_wmae",
                        "validation/blend_wmae",
                    ],
                    max_runs_to_show=12,
                    layout=wr.Layout(w=12, h=8),
                ),
                wr.RunComparer(diff_only="split", layout=wr.Layout(w=12, h=8)),
            ],
        ),
        wr.PanelGrid(
            runsets=[media],
            hide_run_sets=True,
            panels=[
                wr.MediaBrowser(
                    title="Classical best local comparison",
                    media_keys=["report_charts/classical_local"],
                    num_columns=1,
                    layout=wr.Layout(w=24, h=9),
                )
            ],
        ),
        wr.H1("5. Foundation model — TimesFM"),
        wr.MarkdownBlock(
            text="""TimesFM v1 zero-shot WMAE 1672.2525 იყო; v2 seasonal/raw/residual calibration blend — 1620.5430; v3 leakage-safe XReg + corrected blend — 1588.8029. v3.1 audit-მა XReg-ის 27.17 WMAE contribution და temporal instability აჩვენა. LoRA standalone 8396.0651-მდე გაუარესდა და calibration-მა weight 0 მისცა. Registry champion Kaggle public 2742.68603 და private 2853.40612 — საუკეთესო non-tree private/final result."""
        ),
        wr.PanelGrid(
            runsets=[timesfm],
            panels=[
                wr.BarPlot(
                    title="TimesFM final validation WMAE",
                    metrics=[
                        "validation/wmae",
                        "validation/xreg_wmae",
                        "validation/lora_wmae",
                    ],
                    max_runs_to_show=10,
                    layout=wr.Layout(w=12, h=8),
                ),
                wr.LinePlot(
                    title="LoRA train vs validation WMAE",
                    x="epoch",
                    y=["train_wmae", "lora_validation_wmae"],
                    layout=wr.Layout(w=12, h=8),
                ),
            ],
        ),
        wr.PanelGrid(
            runsets=[media],
            hide_run_sets=True,
            panels=[
                wr.MediaBrowser(
                    title="TimesFM version progression",
                    media_keys=["report_charts/timesfm_versions"],
                    num_columns=1,
                    layout=wr.Layout(w=12, h=8),
                ),
                wr.MediaBrowser(
                    title="TimesFM candidate comparison",
                    media_keys=["report_charts/timesfm_candidates"],
                    num_columns=1,
                    layout=wr.Layout(w=12, h=8),
                ),
            ],
        ),
        wr.H1("6. Hybrid — XGBoost + SARIMA"),
        wr.MarkdownBlock(
            text="""Hybrid experiment complementary improvement არ ყოფილა. Standalone XGBoost WMAE 2111.412 იყო; SARIMA numerically exploded დაახლოებით 8.13e47-მდე; 0.90/0.10 blend-იც 8.13e46 დარჩა. Grid-ში pure XGBoost weight 1.00 არ შედიოდა. სწორი გადაწყვეტილება იყო xgb_weight=1, sarima_weight=0. ეს negative result აჩვენებს, რომ residual model მხოლოდ stable complementary structure-ისას არის სასარგებლო."""
        ),
        wr.PanelGrid(
            runsets=[hybrid],
            panels=[
                wr.BarPlot(
                    title="Hybrid component failure",
                    metrics=[
                        "validation/xgboost_wmae",
                        "validation/sarima_wmae",
                        "validation/hybrid_wmae",
                    ],
                    layout=wr.Layout(w=12, h=8),
                ),
                wr.RunComparer(diff_only="split", layout=wr.Layout(w=12, h=8)),
            ],
        ),
        wr.H1("7. Pipelines, Registry and inference lineage"),
        wr.MarkdownBlock(
            text="""Final pipeline მხოლოდ trained estimator არ არის. იგი raw test.csv-ს იღებს, schema-ს ამოწმებს, training preprocessing-ს იმეორებს, mappings/history/fallback-ს ინახავს, prediction order-ს აღადგენს და finite output-ს ამოწმებს. Champion artifacts W&B Model Registry-შია; inference notebooks პირდაპირ Registry URI-ს იყენებს და submission/manifests-ს ცალკე artifacts-ად ლოგავს."""
        ),
        wr.PanelGrid(
            runsets=[pipeline_runs],
            panels=[
                wr.BarPlot(
                    title="Pipeline / inference runtime",
                    metrics=[
                        "pipeline/contract_minutes",
                        "inference/minutes",
                        "fit/elapsed_minutes",
                    ],
                    max_runs_to_show=20,
                    layout=wr.Layout(w=12, h=8),
                ),
                wr.RunComparer(diff_only="split", layout=wr.Layout(w=12, h=8)),
            ],
        ),
        wr.H1("8. საბოლოო დასკვნა"),
        wr.MarkdownBlock(
            text="""**Overall champion: XGBoost — 2806 private/final WMAE.** LightGBM 2809-ით პრაქტიკულად თანაბარია. TimesFM v3 public 2742.68603 / private 2853.40612-ით საუკეთესო non-tree model და overall მესამეა. TFT private 3058.98280-ით საუკეთესო deep-learning model-ია. Prophet საუკეთესო local score-ს აჩვენებს, მაგრამ leaderboard evidence არ აქვს. მთავარი ცოდნა: feature availability, leakage control, chronological validation და train/inference parity architecture complexity-ზე არანაკლებ მნიშვნელოვანია."""
        ),
        wr.CalloutBlock(
            "EDA → baseline → controlled experiments → W&B tracking → champion selection → raw pipeline → Registry → independent inference → Kaggle."
        ),
    ]

    return wr.Report(
        entity=ENTITY,
        project=PROJECT,
        title=REPORT_TITLE,
        description="ხუთი modeling family, 553 W&B run, local/Kaggle comparison, failures, pipelines და final champion.",
        blocks=blocks,
        width="fluid",
    )


def main() -> None:
    if not os.environ.get("WANDB_API_KEY"):
        raise RuntimeError(
            "WANDB_API_KEY must be provided through the environment for this session."
        )
    with tempfile.TemporaryDirectory(prefix="walmart-report-") as temp:
        charts = create_charts(Path(temp))
        if os.environ.get("REPORT_SKIP_LOGGING") != "1":
            log_standardized_runs()
            log_visual_assets(charts)
        report = build_report()
        report.save()
        print(
            {
                "report_url": report.url,
                "charts": len(charts),
                "standardized_models": len(MODELS),
            }
        )


if __name__ == "__main__":
    main()
