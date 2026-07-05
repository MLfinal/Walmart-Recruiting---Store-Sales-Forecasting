# LightGBM Feature Engineering and Feature Selection

This document explains the feature engineering and feature selection code currently placed in `model_experiment_LightGBM.ipynb`.

The notebook defines sklearn-style transformers using:

```python
BaseEstimator, TransformerMixin
```

That means each feature step follows the same structure:

```python
transformer.fit(train_data)
transformed_train = transformer.transform(train_data)
transformed_valid = transformer.transform(valid_data)
```

This is important because some features must be learned only from training data. If we calculate them using validation or test rows, we can accidentally leak future information into the model.

## 1. Feature Cleaning

Implemented by:

```python
WalmartFeatureCleaner
```

This transformer prepares the raw merged Walmart data before creating new features.

### Date Conversion

The `Date` column is converted to pandas datetime:

```python
frame["Date"] = pd.to_datetime(frame["Date"])
```

Why:

- calendar features need proper datetime format;
- sorting by time must be reliable;
- lag and rolling features depend on correct time order.

### Markdown Missing Indicators

For each markdown column:

```python
MarkDown1
MarkDown2
MarkDown3
MarkDown4
MarkDown5
```

the transformer creates:

```python
MarkDown1_missing
MarkDown2_missing
...
```

Why:

- markdown values are missing for a large part of the dataset;
- missing markdown does not always mean random missing data;
- in this competition, markdown data starts appearing later in time;
- the fact that markdown is missing may itself be useful information.

### Markdown Filling

After creating the missing indicators, missing markdown values are filled with `0`.

Why:

- a missing markdown amount is treated as no recorded promotion;
- tree models can split on the missing indicator separately;
- numerical operations like total markdown and log markdown need numeric values.

### Numeric Imputation

The transformer can fill missing values in:

```python
CPI
Unemployment
Temperature
Fuel_Price
```

Default strategy:

```python
median
```

Why median:

- median is more robust than mean when there are outliers;
- these columns are mostly complete, so this is mainly a safety step;
- it prevents errors in later sklearn tools that may not accept missing values.

Important:

LightGBM can handle missing numeric values itself. So this step is not always mandatory, but it keeps the pipeline consistent.

### Categorical Type Conversion

The transformer converts:

```python
Store
Dept
Type
```

to pandas `category`.

Why:

- LightGBM can use categorical columns directly;
- this avoids unnecessary one-hot encoding;
- store and department identities are important signals.

`IsHoliday` is converted to integer `0` or `1`.

## 2. Calendar Features

Implemented by:

```python
CalendarFeatureTransformer
```

This transformer creates date-based features from `Date`.

### Year

```python
Year
```

Why:

- sales behavior can change by year;
- economic conditions and store behavior may drift over time.

### Month

```python
Month
```

Why:

- sales have strong monthly seasonality;
- November and December are especially important for Walmart.

### Week Of Year

```python
WeekOfYear
```

Why:

- the dataset is weekly;
- holiday effects are strongly tied to specific weeks;
- Thanksgiving, Christmas, Super Bowl, and Labor Day happen around predictable weeks.

### Quarter

```python
Quarter
```

Why:

- captures broader seasonal periods;
- useful when monthly differences are too detailed.

### Day Of Year

```python
DayOfYear
```

Why:

- gives the model a continuous position inside the year;
- helps model gradual seasonal movement.

### Days From Start

```python
DaysFromStart
```

Why:

- gives the model a time trend feature;
- helps capture long-term changes across the training period.

### Cyclical Week Features

```python
WeekSin
WeekCos
```

These encode `WeekOfYear` on a circle:

```python
sin(2 * pi * WeekOfYear / 52)
cos(2 * pi * WeekOfYear / 52)
```

Why:

- week 52 and week 1 are close in time;
- plain numeric week values make them look far apart;
- sine and cosine make the yearly cycle continuous.

Important:

Both `WeekSin` and `WeekCos` are needed. Looking at only one of them can make different weeks look similar. Together they describe the week's position on the yearly cycle.

For LightGBM, these are optional. Tree models can already use `WeekOfYear`, but keeping cyclical features lets LightGBM decide whether they help.

### Cyclical Month Features

```python
MonthSin
MonthCos
```

Why:

- month 12 and month 1 are close in the yearly cycle;
- this gives another smooth seasonal representation.

## 3. Holiday Features

Implemented by:

```python
WalmartHolidayFeatureTransformer
```

The Walmart competition gives special weight to holiday weeks, so holiday features are important.

The transformer uses known competition holiday weeks:

```python
SuperBowl
LaborDay
Thanksgiving
Christmas
```

### Holiday Week Flags

For each holiday, it creates flags like:

```python
IsSuperBowlWeek
IsLaborDayWeek
IsThanksgivingWeek
IsChristmasWeek
```

Why:

- `IsHoliday` says only whether a row is a holiday week;
- it does not say which holiday;
- different holidays affect sales differently;
- Thanksgiving and Christmas usually have much stronger sales behavior than other holidays.

### Holiday Proximity Features

For each holiday, it creates:

```python
DaysToNearestSuperBowl
WeeksToNearestSuperBowl
DaysToNearestLaborDay
WeeksToNearestLaborDay
...
```

Why:

- sales can change before or after a holiday, not only during the exact holiday week;
- Christmas shopping may start before Christmas week;
- Thanksgiving effects can affect nearby weeks.

These features help LightGBM learn pre-holiday and post-holiday behavior.

## 4. Markdown Features

Implemented by:

```python
MarkdownFeatureTransformer
```

Markdown columns represent promotional discount information.

### Total Markdown

Creates:

```python
TotalMarkDown
```

by summing:

```python
MarkDown1 + MarkDown2 + MarkDown3 + MarkDown4 + MarkDown5
```

Why:

- individual markdown columns may be sparse;
- total promotional intensity can be more useful than each markdown alone;
- it gives the model a single promotion-size feature.

### Markdown Presence Flags

Creates:

```python
HasMarkDown1
HasMarkDown2
...
HasAnyMarkDown
```

Why:

- the existence of a promotion may matter separately from its size;
- tree models often benefit from simple binary split features;
- zero promotion vs some promotion is an important distinction.

### Log Markdown Features

Creates:

```python
MarkDown1_log1p
...
TotalMarkDown_log1p
```

Why:

- markdown values can be highly skewed;
- a very large markdown amount can dominate the raw scale;
- `log1p` compresses large values while keeping zero valid.

For tree models this is not strictly required, but it can still help create better split points.

### Holiday Markdown Interactions

Creates:

```python
Holiday_TotalMarkDown
Holiday_MarkDown1
Holiday_MarkDown2
...
```

Why:

- promotions may have different effects during holiday weeks;
- the same markdown amount can matter more near Thanksgiving or Christmas;
- this makes promotion plus holiday context explicit.

LightGBM can learn interactions by itself, but explicit interaction features can still help.

## 5. Categorical Interaction Features

Implemented by:

```python
InteractionFeatureTransformer
```

Default interactions:

```python
Store_Dept
Type_Dept
```

### Store Department Interaction

```python
Store_Dept
```

Why:

- each store and department pair behaves like its own small time series;
- department 1 in store 1 may behave differently from department 1 in store 20;
- this gives LightGBM a direct identifier for each store-department combination.

### Type Department Interaction

```python
Type_Dept
```

Why:

- department behavior can differ by store type;
- Type A, B, and C stores can have different sales scales;
- this helps with generalization when a specific store-department pair has limited history.

The created interaction columns are converted to pandas `category`, so LightGBM can treat them as categorical features.

## 6. Lag and Rolling Features

Implemented by:

```python
LagRollingFeatureTransformer
```

These are usually the strongest features for weekly sales forecasting.

The transformer groups by:

```python
Store
Dept
```

and sorts by:

```python
Date
```

This ensures every lag is calculated only within the same store-department time series.

### Lag Features

Default lags:

```python
lag_1
lag_4
lag_13
lag_52
```

Meaning:

- `lag_1`: previous week sales;
- `lag_4`: approximately previous month sales;
- `lag_13`: approximately previous quarter sales;
- `lag_52`: same week last year.

Why:

- sales are highly autocorrelated;
- last week's sales are usually very predictive;
- `lag_52` captures yearly seasonality.

Important leakage note:

Lag features require `Weekly_Sales`. For validation, this is okay only if the validation rows are transformed in a way that does not use future target values. For Kaggle test data, true future `Weekly_Sales` is not available, so test-time lag features require either:

- using only historical train sales where available;
- recursive prediction;
- or disabling lag features for a simpler baseline.

### Rolling Mean Features

Default rolling windows:

```python
rolling_mean_4
rolling_mean_13
```

Why:

- rolling mean smooths noisy weekly sales;
- it captures recent demand level;
- 4 weeks gives short-term trend;
- 13 weeks gives quarterly trend.

### Rolling Standard Deviation Features

Default rolling std features:

```python
rolling_std_4
rolling_std_13
```

Why:

- some store-department series are stable;
- others are volatile;
- volatility can help the model understand uncertainty and sales behavior.

### Shift Before Rolling

Rolling features use shifted sales:

```python
shift(1).rolling(...)
```

Why:

- the current row's `Weekly_Sales` must not be used to predict itself;
- this prevents target leakage;
- only past weeks are allowed.

## 7. Historical Aggregate Features

Implemented by:

```python
HistoricalAggregateTransformer
```

This transformer calculates target statistics from training data only.

Default groupings:

```python
Store
Dept
Store + Dept
Type + Dept
```

Default statistics:

```python
mean
median
std
```

### Store Sales Aggregates

Examples:

```python
Store_Weekly_Sales_mean
Store_Weekly_Sales_median
Store_Weekly_Sales_std
```

Why:

- some stores are generally larger than others;
- store-level averages capture store scale.

### Department Sales Aggregates

Examples:

```python
Dept_Weekly_Sales_mean
Dept_Weekly_Sales_median
Dept_Weekly_Sales_std
```

Why:

- some departments sell much more than others;
- department identity is one of the strongest signals.

### Store Department Aggregates

Examples:

```python
Store_Dept_Weekly_Sales_mean
Store_Dept_Weekly_Sales_median
Store_Dept_Weekly_Sales_std
```

Why:

- each store-department pair has its own baseline sales level;
- this is useful for global models trained across all stores and departments.

### Type Department Aggregates

Examples:

```python
Type_Dept_Weekly_Sales_mean
Type_Dept_Weekly_Sales_median
Type_Dept_Weekly_Sales_std
```

Why:

- useful fallback when a specific store-department pair has weak or missing history;
- captures behavior by store type and department.

### Missing Aggregate Fill

If a validation or test group was not seen during training, aggregate values are filled using the global target statistic.

Why:

- prevents missing values for new or rare groups;
- gives the model a reasonable fallback.

Important leakage note:

These aggregates must be fitted only on training data. Do not calculate them using the full train + validation dataset before validation, because that would leak validation target information.

## 8. Column Dropping

Implemented by:

```python
ColumnDropper
```

Default dropped columns:

```python
Date
Weekly_Sales
```

Why:

- `Weekly_Sales` is the target and must not be inside model features;
- raw `Date` is not directly usable by LightGBM unless converted;
- after calendar features are created, raw date can usually be removed.

## 9. Feature Selection

Implemented by:

```python
FeatureImportanceSelector
```

This is a model-based feature selection transformer.

It works like this:

1. Fit an estimator on the engineered features.
2. Read the estimator's `feature_importances_`.
3. Keep features whose importance is greater than a threshold.
4. Transform future data by keeping only those selected columns.

Example logic:

```python
selected_features = feature_importances[feature_importances > threshold]
```

### Why Use Model-Based Feature Selection?

LightGBM can handle many features, so aggressive feature selection is not necessary at the beginning.

But feature selection can help:

- remove useless zero-importance features;
- reduce training time;
- simplify explanation;
- reduce noise if validation performance improves.

### Recommended Threshold

Start with:

```python
threshold = 0.0
```

This removes only features that LightGBM never used.

Do not remove too many features immediately. If a feature has low importance but helps in combination with another feature, removing it can hurt performance.

### Better Validation Rule

Feature selection should be accepted only if validation score improves or stays the same.

Recommended process:

1. Train LightGBM with all engineered features.
2. Save validation WMAE.
3. Select non-zero-importance features.
4. Retrain LightGBM with selected features.
5. Compare validation WMAE.
6. Keep selected features only if validation WMAE improves or the simpler model is preferred.

## 10. Default Pipeline

Implemented by:

```python
make_walmart_lgbm_feature_pipeline
```

Default order:

```text
1. WalmartFeatureCleaner
2. CalendarFeatureTransformer
3. WalmartHolidayFeatureTransformer
4. MarkdownFeatureTransformer
5. InteractionFeatureTransformer
6. HistoricalAggregateTransformer
7. LagRollingFeatureTransformer
8. ColumnDropper
```

This order matters.

Cleaning comes first because later steps need valid dates, numeric markdown values, and categorical columns.

Calendar and holiday features come before modeling because they use `Date`.

Markdown features come after markdown cleaning because they depend on non-missing markdown values.

Historical aggregates are fitted before dropping `Weekly_Sales` because they need the target column.

Lag and rolling features are also created before dropping `Weekly_Sales`.

Finally, `Date` and `Weekly_Sales` are dropped before model training.

## 11. What LightGBM Receives

After the pipeline, LightGBM receives a table containing:

- original clean numeric features;
- categorical features such as `Store`, `Dept`, `Type`;
- calendar features;
- holiday flags and proximity features;
- markdown amount, presence, log, and interaction features;
- store-department interaction categories;
- lag and rolling sales features;
- historical target aggregates.

The target remains:

```python
Weekly_Sales
```

and should be passed separately as `y_train`.

## 12. Main Leakage Risks

The most important risks are:

```text
1. Historical aggregates calculated on validation/test target values.
2. Rolling features using the current row's target.
3. Test lag features using unknown future sales.
4. Random train/validation split for a forecasting task.
```

To avoid these:

- fit aggregate features only on training data;
- use `shift(1)` before rolling statistics;
- use time-based validation;
- be careful with lag features during test inference.

## 13. Practical Recommendation

For the first LightGBM experiment:

1. Use all engineered features.
2. Use a time-based validation split.
3. Evaluate with weighted MAE.
4. Inspect feature importance.
5. Remove only zero-importance features.
6. Retrain and compare validation WMAE.

The most important feature groups are expected to be:

```text
1. lag features
2. rolling sales features
3. Store/Dept identity and aggregates
4. holiday features
5. markdown features
6. calendar features
```

For this Walmart dataset, historical sales features usually matter more than complex feature-selection methods.
