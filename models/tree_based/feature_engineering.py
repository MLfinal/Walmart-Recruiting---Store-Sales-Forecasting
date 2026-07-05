from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline


MARKDOWN_COLS = ("MarkDown1", "MarkDown2", "MarkDown3", "MarkDown4", "MarkDown5")
NUMERIC_EXTERNAL_COLS = ("CPI", "Unemployment", "Temperature", "Fuel_Price")


def _existing_columns(frame: pd.DataFrame, columns: Iterable[str]) -> list[str]:
    return [col for col in columns if col in frame.columns]


class WalmartFeatureCleaner(BaseEstimator, TransformerMixin):
    """Basic cleaning before feature engineering."""

    def __init__(
        self,
        date_col: str = "Date",
        markdown_cols: tuple[str, ...] = MARKDOWN_COLS,
        numeric_impute_cols: tuple[str, ...] = NUMERIC_EXTERNAL_COLS,
        add_markdown_missing_indicators: bool = True,
        markdown_fill_value: float = 0.0,
        numeric_impute_strategy: str = "median",
        category_cols: tuple[str, ...] = ("Store", "Dept", "Type"),
    ):
        self.date_col = date_col
        self.markdown_cols = markdown_cols
        self.numeric_impute_cols = numeric_impute_cols
        self.add_markdown_missing_indicators = add_markdown_missing_indicators
        self.markdown_fill_value = markdown_fill_value
        self.numeric_impute_strategy = numeric_impute_strategy
        self.category_cols = category_cols

    def fit(self, X: pd.DataFrame, y=None):
        if self.numeric_impute_strategy not in {"median", "mean", "none"}:
            raise ValueError("numeric_impute_strategy must be 'median', 'mean', or 'none'.")

        self.numeric_fill_values_ = {}
        numeric_cols = _existing_columns(X, self.numeric_impute_cols)
        if self.numeric_impute_strategy != "none":
            for col in numeric_cols:
                if self.numeric_impute_strategy == "median":
                    self.numeric_fill_values_[col] = X[col].median()
                else:
                    self.numeric_fill_values_[col] = X[col].mean()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        frame = X.copy()

        if self.date_col in frame.columns:
            frame[self.date_col] = pd.to_datetime(frame[self.date_col])

        for col in _existing_columns(frame, self.markdown_cols):
            if self.add_markdown_missing_indicators:
                frame[f"{col}_missing"] = frame[col].isna().astype("int8")
            frame[col] = frame[col].fillna(self.markdown_fill_value)

        for col, value in getattr(self, "numeric_fill_values_", {}).items():
            if col in frame.columns:
                frame[col] = frame[col].fillna(value)

        for col in _existing_columns(frame, self.category_cols):
            frame[col] = frame[col].astype("category")

        if "IsHoliday" in frame.columns:
            frame["IsHoliday"] = frame["IsHoliday"].astype("int8")

        return frame


class CalendarFeatureTransformer(BaseEstimator, TransformerMixin):
    """Create date-derived features."""

    def __init__(
        self,
        date_col: str = "Date",
        start_date: str = "2010-02-05",
        add_cyclical_features: bool = True,
        drop_date: bool = False,
    ):
        self.date_col = date_col
        self.start_date = start_date
        self.add_cyclical_features = add_cyclical_features
        self.drop_date = drop_date

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        frame = X.copy()
        date = pd.to_datetime(frame[self.date_col])
        iso = date.dt.isocalendar()

        frame["Year"] = date.dt.year.astype("int16")
        frame["Month"] = date.dt.month.astype("int8")
        frame["WeekOfYear"] = iso.week.astype("int8")
        frame["Quarter"] = date.dt.quarter.astype("int8")
        frame["DayOfYear"] = date.dt.dayofyear.astype("int16")
        frame["DaysFromStart"] = (date - pd.Timestamp(self.start_date)).dt.days.astype("int16")

        if self.add_cyclical_features:
            frame["WeekSin"] = np.sin(2 * np.pi * frame["WeekOfYear"] / 52.0)
            frame["WeekCos"] = np.cos(2 * np.pi * frame["WeekOfYear"] / 52.0)
            frame["MonthSin"] = np.sin(2 * np.pi * frame["Month"] / 12.0)
            frame["MonthCos"] = np.cos(2 * np.pi * frame["Month"] / 12.0)

        if self.drop_date:
            frame = frame.drop(columns=[self.date_col])

        return frame


class WalmartHolidayFeatureTransformer(BaseEstimator, TransformerMixin):
    """Add Walmart competition holiday flags and proximity features."""

    HOLIDAY_DATES = {
        "SuperBowl": ("2010-02-12", "2011-02-11", "2012-02-10", "2013-02-08"),
        "LaborDay": ("2010-09-10", "2011-09-09", "2012-09-07", "2013-09-06"),
        "Thanksgiving": ("2010-11-26", "2011-11-25", "2012-11-23", "2013-11-29"),
        "Christmas": ("2010-12-31", "2011-12-30", "2012-12-28", "2013-12-27"),
    }

    def __init__(
        self,
        date_col: str = "Date",
        add_holiday_flags: bool = True,
        add_proximity_features: bool = True,
    ):
        self.date_col = date_col
        self.add_holiday_flags = add_holiday_flags
        self.add_proximity_features = add_proximity_features

    def fit(self, X: pd.DataFrame, y=None):
        self.holiday_dates_ = {
            name: pd.to_datetime(list(dates)) for name, dates in self.HOLIDAY_DATES.items()
        }
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        frame = X.copy()
        date = pd.to_datetime(frame[self.date_col])

        for name, holiday_dates in self.holiday_dates_.items():
            if self.add_holiday_flags:
                frame[f"Is{name}Week"] = date.isin(holiday_dates).astype("int8")

            if self.add_proximity_features:
                distances = np.vstack([(date - holiday).dt.days.to_numpy() for holiday in holiday_dates])
                nearest_distance = distances[np.abs(distances).argmin(axis=0), np.arange(len(date))]
                frame[f"DaysToNearest{name}"] = np.abs(nearest_distance).astype("int16")
                frame[f"WeeksToNearest{name}"] = (np.abs(nearest_distance) / 7.0).astype("float32")

        return frame


class MarkdownFeatureTransformer(BaseEstimator, TransformerMixin):
    """Create promotion/markdown features."""

    def __init__(
        self,
        markdown_cols: tuple[str, ...] = MARKDOWN_COLS,
        add_total_markdown: bool = True,
        add_has_markdown: bool = True,
        add_log_markdowns: bool = True,
        add_holiday_interaction: bool = True,
        holiday_col: str = "IsHoliday",
    ):
        self.markdown_cols = markdown_cols
        self.add_total_markdown = add_total_markdown
        self.add_has_markdown = add_has_markdown
        self.add_log_markdowns = add_log_markdowns
        self.add_holiday_interaction = add_holiday_interaction
        self.holiday_col = holiday_col

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        frame = X.copy()
        markdown_cols = _existing_columns(frame, self.markdown_cols)

        if self.add_total_markdown and markdown_cols:
            frame["TotalMarkDown"] = frame[markdown_cols].sum(axis=1)

        if self.add_has_markdown:
            for col in markdown_cols:
                frame[f"Has{col}"] = (frame[col] > 0).astype("int8")
            if "TotalMarkDown" in frame.columns:
                frame["HasAnyMarkDown"] = (frame["TotalMarkDown"] > 0).astype("int8")

        if self.add_log_markdowns:
            for col in markdown_cols:
                frame[f"{col}_log1p"] = np.log1p(frame[col].clip(lower=0))
            if "TotalMarkDown" in frame.columns:
                frame["TotalMarkDown_log1p"] = np.log1p(frame["TotalMarkDown"].clip(lower=0))

        if self.add_holiday_interaction and self.holiday_col in frame.columns:
            if "TotalMarkDown" in frame.columns:
                frame["Holiday_TotalMarkDown"] = frame[self.holiday_col] * frame["TotalMarkDown"]
            for col in markdown_cols:
                frame[f"Holiday_{col}"] = frame[self.holiday_col] * frame[col]

        return frame


class InteractionFeatureTransformer(BaseEstimator, TransformerMixin):
    """Create categorical interaction columns such as Store_Dept."""

    def __init__(
        self,
        interactions: tuple[tuple[str, ...], ...] = (("Store", "Dept"), ("Type", "Dept")),
        separator: str = "_",
        as_category: bool = True,
    ):
        self.interactions = interactions
        self.separator = separator
        self.as_category = as_category

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        frame = X.copy()

        for cols in self.interactions:
            if all(col in frame.columns for col in cols):
                new_col = self.separator.join(cols)
                values = frame[list(cols)].astype(str).agg(self.separator.join, axis=1)
                frame[new_col] = values.astype("category") if self.as_category else values

        return frame


class LagRollingFeatureTransformer(BaseEstimator, TransformerMixin):
    """Create leakage-safe lag and rolling features within Store/Dept groups.

    The input frame must contain target_col. For Kaggle test inference, create
    these features only from available historical sales or use a recursive
    prediction loop.
    """

    def __init__(
        self,
        group_cols: tuple[str, ...] = ("Store", "Dept"),
        date_col: str = "Date",
        target_col: str = "Weekly_Sales",
        lags: tuple[int, ...] = (1, 4, 13, 52),
        rolling_windows: tuple[int, ...] = (4, 13),
        rolling_stats: tuple[str, ...] = ("mean", "std"),
        min_periods: int = 1,
    ):
        self.group_cols = group_cols
        self.date_col = date_col
        self.target_col = target_col
        self.lags = lags
        self.rolling_windows = rolling_windows
        self.rolling_stats = rolling_stats
        self.min_periods = min_periods

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.target_col not in X.columns:
            raise ValueError(
                f"{self.target_col!r} is required for lag/rolling features. "
                "For test data, append historical sales first or use recursive inference."
            )

        frame = X.copy().sort_values(list(self.group_cols) + [self.date_col])
        grouped = frame.groupby(list(self.group_cols), observed=True)[self.target_col]

        for lag in self.lags:
            frame[f"lag_{lag}"] = grouped.shift(lag)

        for window in self.rolling_windows:
            shifted = grouped.shift(1)
            rolling = shifted.groupby([frame[col] for col in self.group_cols], observed=True).rolling(
                window=window,
                min_periods=self.min_periods,
            )
            if "mean" in self.rolling_stats:
                frame[f"rolling_mean_{window}"] = rolling.mean().reset_index(level=list(range(len(self.group_cols))), drop=True)
            if "std" in self.rolling_stats:
                frame[f"rolling_std_{window}"] = rolling.std().reset_index(level=list(range(len(self.group_cols))), drop=True)
            if "min" in self.rolling_stats:
                frame[f"rolling_min_{window}"] = rolling.min().reset_index(level=list(range(len(self.group_cols))), drop=True)
            if "max" in self.rolling_stats:
                frame[f"rolling_max_{window}"] = rolling.max().reset_index(level=list(range(len(self.group_cols))), drop=True)

        return frame.sort_index()


class HistoricalAggregateTransformer(BaseEstimator, TransformerMixin):
    """Add target aggregates computed only from fit data."""

    def __init__(
        self,
        groupings: tuple[tuple[str, ...], ...] = (("Store",), ("Dept",), ("Store", "Dept"), ("Type", "Dept")),
        target_col: str = "Weekly_Sales",
        stats: tuple[str, ...] = ("mean", "median", "std"),
        fill_missing_with_global: bool = True,
    ):
        self.groupings = groupings
        self.target_col = target_col
        self.stats = stats
        self.fill_missing_with_global = fill_missing_with_global

    def fit(self, X: pd.DataFrame, y=None):
        if self.target_col not in X.columns:
            raise ValueError(f"{self.target_col!r} must be present when fitting aggregates.")

        self.global_stats_ = X[self.target_col].agg(list(self.stats)).to_dict()
        self.aggregate_frames_ = []

        for grouping in self.groupings:
            existing_grouping = tuple(col for col in grouping if col in X.columns)
            if not existing_grouping:
                continue
            prefix = "_".join(existing_grouping)
            agg = (
                X.groupby(list(existing_grouping), observed=True)[self.target_col]
                .agg(list(self.stats))
                .reset_index()
            )
            rename = {stat: f"{prefix}_{self.target_col}_{stat}" for stat in self.stats}
            agg = agg.rename(columns=rename)
            self.aggregate_frames_.append((existing_grouping, agg, rename))

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        frame = X.copy()

        for grouping, agg, rename in self.aggregate_frames_:
            frame = frame.merge(agg, on=list(grouping), how="left", validate="many_to_one")
            if self.fill_missing_with_global:
                for stat, col in rename.items():
                    frame[col] = frame[col].fillna(self.global_stats_[stat])

        return frame


class ColumnDropper(BaseEstimator, TransformerMixin):
    """Drop columns before fitting the model."""

    def __init__(self, columns: tuple[str, ...] = ("Date", "Weekly_Sales"), errors: str = "ignore"):
        self.columns = columns
        self.errors = errors

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X.drop(columns=list(self.columns), errors=self.errors)


class FeatureImportanceSelector(BaseEstimator, TransformerMixin):
    """Select features with non-trivial model importance."""

    def __init__(self, estimator, threshold: float = 0.0, fit_params: dict | None = None):
        self.estimator = estimator
        self.threshold = threshold
        self.fit_params = fit_params

    def fit(self, X: pd.DataFrame, y):
        fit_params = self.fit_params or {}
        self.estimator.fit(X, y, **fit_params)
        importances = getattr(self.estimator, "feature_importances_", None)
        if importances is None:
            raise ValueError("estimator must expose feature_importances_ after fit.")

        self.feature_importances_ = pd.Series(importances, index=X.columns).sort_values(ascending=False)
        self.selected_features_ = self.feature_importances_[
            self.feature_importances_ > self.threshold
        ].index.tolist()
        if not self.selected_features_:
            raise ValueError("No features passed the importance threshold.")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X[self.selected_features_].copy()


def make_walmart_lgbm_feature_pipeline(
    include_lag_features: bool = True,
    drop_target_and_date: bool = True,
) -> Pipeline:
    """Build a default feature pipeline for LightGBM experiments."""

    steps = [
        ("clean", WalmartFeatureCleaner()),
        ("calendar", CalendarFeatureTransformer()),
        ("holiday", WalmartHolidayFeatureTransformer()),
        ("markdown", MarkdownFeatureTransformer()),
        ("interactions", InteractionFeatureTransformer()),
        ("aggregates", HistoricalAggregateTransformer()),
    ]

    if include_lag_features:
        steps.append(("lags_rollings", LagRollingFeatureTransformer()))

    if drop_target_and_date:
        steps.append(("drop_columns", ColumnDropper()))

    return Pipeline(steps)
