# LightGBM Feature Engineering და Feature Selection

ეს დოკუმენტი ხსნის, რა feature engineering და feature selection ლოგიკაა დამატებული `model_experiment_LightGBM.ipynb`-ში.

Notebook-ში feature engineering დაწერილია sklearn-ის სტილში:

```python
BaseEstimator, TransformerMixin
```

ანუ თითოეული ნაბიჯი მუშაობს ასე:

```python
transformer.fit(train_data)
train_transformed = transformer.transform(train_data)
valid_transformed = transformer.transform(valid_data)
```

ეს მნიშვნელოვანია, რადგან ზოგი feature მხოლოდ train მონაცემებიდან უნდა ვისწავლოთ. თუ validation ან test row-ების target ინფორმაციას გამოვიყენებთ, მივიღებთ target leakage-ს და validation score ხელოვნურად კარგი გამოვა.

## Kaggle შედეგის მოკლე ანალიზი

LightGBM-ის ძველმა lag/rolling feature engineering-მა validation-ზე ძალიან ძლიერი შედეგი აჩვენა, მაგრამ Kaggle-ზე ცუდად generalized:

```text
ძველი unsafe LightGBM setup ≈ 6200 Kaggle score
ახალი safe SalesLag52 LightGBM setup ≈ 3600 Kaggle score
XGBoost final pipeline ≈ 2806 Kaggle score
```

ჩემი აზრით, მთავარი მიზეზი feature availability იყო. `lag_1`, `lag_4`, `lag_13` და rolling feature-ები validation-ზე რეალურ `Weekly_Sales` მნიშვნელობებზე იყო აგებული, მაგრამ Kaggle test future-ში ეს ინფორმაცია არ არსებობს. ამიტომ validation score ზედმეტად კარგი გამოდიოდა, submission-ზე კი model იშლებოდა.

Safe ვერსიაში დავტოვეთ `SalesLag52`, რადგან ერთი წლის წინანდელი sales history inference დროსაც ხელმისაწვდომია. ამის შემდეგ validation WMAE ოდნავ გაუარესდა, მაგრამ Kaggle score მნიშვნელოვნად გაუმჯობესდა. საბოლოოდ XGBoost მაინც უკეთესი დარჩა, რადგან მისი raw-input pipeline და feature engineering უფრო სტაბილურად დაემთხვა Kaggle test setup-ს.

## 1. Feature Cleaning

კლასი:

```python
WalmartFeatureCleaner
```

ეს transformer ამზადებს merged Walmart მონაცემებს feature engineering-მდე.

### Date-ის გარდაქმნა

`Date` სვეტი გარდაიქმნება pandas datetime ფორმატში:

```python
frame["Date"] = pd.to_datetime(frame["Date"])
```

რატომ:

- calendar feature-ებს სჭირდება სწორი თარიღის ფორმატი;
- დროით დალაგება უნდა იყოს საიმედო;
- lag და rolling feature-ები სწორ ქრონოლოგიაზეა დამოკიდებული.

### Markdown Missing Indicator-ები

Markdown სვეტებია:

```python
MarkDown1
MarkDown2
MarkDown3
MarkDown4
MarkDown5
```

თითოეულისთვის იქმნება missing indicator:

```python
MarkDown1_missing
MarkDown2_missing
...
```

რატომ:

- markdown სვეტებში ბევრი missing მნიშვნელობაა;
- missing markdown ყოველთვის random missing data არ არის;
- ამ competition-ში markdown მონაცემები დროის გარკვეული მომენტიდან ჩნდება;
- თვითონ missing ფაქტიც შეიძლება სასარგებლო signal იყოს.

### Markdown-ის 0-ით შევსება

missing indicator-ების შექმნის შემდეგ markdown სვეტებში missing მნიშვნელობები ივსება `0`-ით.

რატომ:

- missing markdown ვთარგმნით როგორც no recorded promotion;
- missing indicator ცალკე ინახავს ინფორმაციას, რომ originally მნიშვნელობა არ იყო;
- `TotalMarkDown` და `log1p` feature-ებს numeric value სჭირდება.

### Numeric Imputation

Transformer-ს შეუძლია missing მნიშვნელობების შევსება ამ სვეტებში:

```python
CPI
Unemployment
Temperature
Fuel_Price
```

default strategy არის:

```python
median
```

რატომ median:

- median mean-ზე უფრო robust-ია outlier-ების მიმართ;
- ეს სვეტები ძირითადად შევსებულია, ამიტომ ეს უფრო safety step-ია;
- ზოგი sklearn tool missing value-ებს ვერ იღებს.

შენიშვნა:

LightGBM-ს შეუძლია numeric missing values თვითონაც დაამუშაოს. ამიტომ ეს ნაბიჯი ყოველთვის აუცილებელი არ არის, მაგრამ pipeline-ს უფრო სტაბილურს ხდის.

### Categorical Type Conversion

ეს სვეტები გადადის pandas `category` ტიპში:

```python
Store
Dept
Type
```

რატომ:

- LightGBM pandas category dtype-ს categorical feature-ად იყენებს;
- one-hot encoding აუცილებელი აღარ არის;
- store და department identity ამ dataset-ში ძალიან მნიშვნელოვანი signal-ია.

`IsHoliday` გადადის integer ფორმატში:

```python
0 / 1
```

## 2. Calendar Features

კლასი:

```python
CalendarFeatureTransformer
```

ეს transformer `Date` სვეტიდან ქმნის calendar feature-ებს.

### Year

```python
Year
```

რატომ:

- გაყიდვების pattern შეიძლება წლიდან წლამდე იცვლებოდეს;
- ეკონომიკური მდგომარეობა და store behavior დროში იცვლება.

### Month

```python
Month
```

რატომ:

- გაყიდვებში არის monthly seasonality;
- Walmart-ისთვის ნოემბერი და დეკემბერი განსაკუთრებით მნიშვნელოვანია.

### WeekOfYear

```python
WeekOfYear
```

რატომ:

- dataset weekly frequency-ზეა;
- holiday effect-ები ხშირად კონკრეტულ კვირებზეა მიბმული;
- Thanksgiving, Christmas, Super Bowl და Labor Day predictable week-ებში მოდის.

### Quarter

```python
Quarter
```

რატომ:

- უფრო ფართო seasonal period-ს იჭერს;
- სასარგებლოა მაშინ, როცა month-level feature ზედმეტად დეტალურია.

### DayOfYear

```python
DayOfYear
```

რატომ:

- აძლევს მოდელს წლის შიგნით continuous პოზიციას;
- ეხმარება gradual seasonality-ის დაჭერაში.

### DaysFromStart

```python
DaysFromStart
```

რატომ:

- დროის trend feature-ია;
- ეხმარება long-term ცვლილებების დაჭერაში.

### WeekSin და WeekCos

```python
WeekSin
WeekCos
```

ეს არის cyclical encoding:

```python
sin(2 * pi * WeekOfYear / 52)
cos(2 * pi * WeekOfYear / 52)
```

რატომ:

- week 52 და week 1 რეალურად ერთმანეთთან ახლოსაა;
- თუ მხოლოდ `WeekOfYear = 1, 2, ..., 52` გვაქვს, მოდელი week 52-ს და week 1-ს შორს დაინახავს;
- sine/cosine representation კვირებს წრეზე ალაგებს.

მნიშვნელოვანი დეტალი:

ორივე feature ერთად უნდა გამოიყენო. მხოლოდ `WeekSin`-ის ნახვით ზოგ week-ს შეიძლება მსგავსი value ჰქონდეს, მაგრამ `WeekSin + WeekCos` ერთად კვირის განსხვავებულ პოზიციას აღწერს.

მაგალითად:

```text
Week 1:
WeekSin დაახლოებით 0.12
WeekCos დაახლოებით 0.99

Week 25:
WeekSin დაახლოებით 0.12
WeekCos დაახლოებით -0.99
```

ანუ week 1 და week 25 ერთი და იგივე არ არის, რადგან cosine განსხვავებულია.

LightGBM-ისთვის ეს optional feature-ებია. Tree model-ები `WeekOfYear`-საც კარგად იყენებენ, მაგრამ `WeekSin` და `WeekCos` ვტოვებთ, რომ LightGBM-მა თვითონ გადაწყვიტოს, ეხმარება თუ არა.

### MonthSin და MonthCos

```python
MonthSin
MonthCos
```

რატომ:

- month 12 და month 1 წლიურ ციკლში ახლოსაა;
- model-ს აძლევს smooth yearly seasonality representation-ს.

## 3. Holiday Features

კლასი:

```python
WalmartHolidayFeatureTransformer
```

Walmart competition-ში holiday weeks უფრო მაღალი წონით ფასდება, ამიტომ holiday feature-ები მნიშვნელოვანია.

გამოყენებულია competition-ის მთავარი holiday weeks:

```python
SuperBowl
LaborDay
Thanksgiving
Christmas
```

### Holiday Week Flags

იქმნება ასეთი binary feature-ები:

```python
IsSuperBowlWeek
IsLaborDayWeek
IsThanksgivingWeek
IsChristmasWeek
```

რატომ:

- `IsHoliday` მხოლოდ იმას ამბობს, holiday week არის თუ არა;
- არ ამბობს რომელი holiday არის;
- სხვადასხვა holiday გაყიდვებზე სხვადასხვანაირად მოქმედებს;
- Thanksgiving და Christmas, როგორც წესი, ყველაზე ძლიერი sales spike-ებია.

### Holiday Proximity Features

თითოეული holiday-სთვის იქმნება:

```python
DaysToNearestSuperBowl
WeeksToNearestSuperBowl
DaysToNearestLaborDay
WeeksToNearestLaborDay
...
```

რატომ:

- holiday effect მხოლოდ ზუსტად holiday week-ზე არ ჩნდება;
- Christmas shopping შეიძლება უფრო ადრე დაიწყოს;
- Thanksgiving-ის გავლენა მის ახლო კვირებზეც შეიძლება გავრცელდეს.

ეს feature-ები ეხმარება LightGBM-ს pre-holiday და post-holiday behavior-ის სწავლაში.

## 4. Markdown Features

კლასი:

```python
MarkdownFeatureTransformer
```

Markdown სვეტები promotional discount information-ს აღწერს.

### TotalMarkDown

იქმნება:

```python
TotalMarkDown
```

ფორმულა:

```python
MarkDown1 + MarkDown2 + MarkDown3 + MarkDown4 + MarkDown5
```

რატომ:

- individual markdown columns sparse შეიძლება იყოს;
- total promotional intensity ხშირად უფრო მარტივი signal-ია;
- მოდელს ეძლევა promotion size-ის ერთი aggregate feature.

### Markdown Presence Flags

იქმნება:

```python
HasMarkDown1
HasMarkDown2
...
HasAnyMarkDown
```

რატომ:

- promotion-ის არსებობა შეიძლება მის რაოდენობაზე დამოუკიდებლადაც მნიშვნელოვანი იყოს;
- tree model-ებს binary split feature-ები ხშირად ეხმარება;
- zero promotion vs some promotion მნიშვნელოვანი განსხვავებაა.

### Log Markdown Features

იქმნება:

```python
MarkDown1_log1p
...
TotalMarkDown_log1p
```

რატომ:

- markdown values skewed შეიძლება იყოს;
- ძალიან დიდი markdown raw scale-ს აბინძურებს;
- `log1p` დიდ მნიშვნელობებს compress-ს უკეთებს და zero-საც სწორად ამუშავებს.

Tree model-ისთვის log transformation ყოველთვის აუცილებელი არ არის, მაგრამ ზოგჯერ უკეთეს split point-ებს ქმნის.

### Holiday Markdown Interactions

იქმნება:

```python
Holiday_TotalMarkDown
Holiday_MarkDown1
Holiday_MarkDown2
...
```

რატომ:

- promotion holiday week-ში შეიძლება სხვანაირად მუშაობდეს;
- იგივე markdown amount Thanksgiving/Christmas პერიოდში უფრო მნიშვნელოვანი იყოს;
- interaction feature მოდელს პირდაპირ აძლევს promotion + holiday context-ს.

LightGBM interactions-ს თვითონაც სწავლობს, მაგრამ explicit interaction ზოგჯერ მაინც ეხმარება.

## 5. Categorical Interaction Features

კლასი:

```python
InteractionFeatureTransformer
```

default interactions:

```python
Store_Dept
Type_Dept
```

### Store_Dept

```python
Store_Dept
```

რატომ:

- თითოეული Store + Dept წყვილი ფაქტობრივად ცალკე time series-ია;
- Dept 1 Store 1-ში შეიძლება სრულიად სხვანაირად იქცეოდეს, ვიდრე Dept 1 Store 20-ში;
- ეს feature LightGBM-ს აძლევს კონკრეტული store-department წყვილის identity-ს.

### Type_Dept

```python
Type_Dept
```

რატომ:

- department behavior შეიძლება store type-ის მიხედვით იცვლებოდეს;
- Type A, B და C stores განსხვავებული scale-ისაა;
- ეხმარება generalization-ს, განსაკუთრებით მაშინ, როცა კონკრეტულ Store_Dept წყვილს ცოტა history აქვს.

ეს interaction columns გადადის pandas `category` ტიპში, რათა LightGBM-მა categorical feature-ებად გამოიყენოს.

## 6. Lag და Rolling Features

კლასი:

```python
LagRollingFeatureTransformer
```

ეს feature-ები weekly sales forecasting-ში, როგორც წესი, ყველაზე ძლიერია.

Transformer grouping-ს აკეთებს:

```python
Store
Dept
```

და ალაგებს:

```python
Date
```

ანუ lag და rolling feature-ები ითვლება მხოლოდ ერთი Store + Dept time series-ის შიგნით.

### Lag Features

default lags:

```python
lag_1
lag_4
lag_13
lag_52
```

მნიშვნელობა:

- `lag_1`: წინა კვირის sales;
- `lag_4`: დაახლოებით წინა თვის sales;
- `lag_13`: დაახლოებით წინა კვარტლის sales;
- `lag_52`: წინა წლის იგივე კვირის sales.

რატომ:

- sales time series ძლიერად autocorrelated არის;
- წინა კვირის sales ხშირად ძალიან predictive-ია;
- `lag_52` yearly seasonality-ს იჭერს.

Leakage note:

Lag feature-ებს სჭირდება `Weekly_Sales`. Validation-ზე ეს ფრთხილად უნდა გაკეთდეს, რომ validation target-ები future information-ად არ გამოვიყენოთ. Kaggle test-ზე მომავალი `Weekly_Sales` უცნობია, ამიტომ test-time lag feature-ებისთვის საჭიროა:

- მხოლოდ train history-ის გამოყენება;
- recursive prediction;
- ან lag feature-ების გამორთვა simpler baseline-ისთვის.

### Rolling Mean Features

default rolling windows:

```python
rolling_mean_4
rolling_mean_13
```

რატომ:

- rolling mean noisy weekly sales-ს ასწორებს;
- recent demand level-ს იჭერს;
- 4 weeks short-term trend-ია;
- 13 weeks quarterly trend-ია.

### Rolling Standard Deviation Features

default rolling std:

```python
rolling_std_4
rolling_std_13
```

რატომ:

- ზოგი Store_Dept სერია სტაბილურია;
- ზოგი volatile არის;
- volatility feature ეხმარება მოდელს sales behavior-ის უკეთ დაჭერაში.

### Shift Before Rolling

Rolling features ითვლება shifted target-ზე:

```python
shift(1).rolling(...)
```

რატომ:

- current row-ის `Weekly_Sales` არ უნდა გამოვიყენოთ current row-ის prediction-ში;
- ეს იცავს target leakage-სგან;
- მოდელს მხოლოდ წარსული კვირების ინფორმაცია უნდა ჰქონდეს.

## 7. Historical Aggregate Features

კლასი:

```python
HistoricalAggregateTransformer
```

ეს transformer target statistics-ს ითვლის მხოლოდ train data-დან.

default groupings:

```python
Store
Dept
Store + Dept
Type + Dept
```

default stats:

```python
mean
median
std
```

### Store Aggregates

მაგალითები:

```python
Store_Weekly_Sales_mean
Store_Weekly_Sales_median
Store_Weekly_Sales_std
```

რატომ:

- ზოგი store უფრო დიდია და ზოგადად მეტი sales აქვს;
- store-level average store scale-ს იჭერს.

### Department Aggregates

მაგალითები:

```python
Dept_Weekly_Sales_mean
Dept_Weekly_Sales_median
Dept_Weekly_Sales_std
```

რატომ:

- ზოგი department ბევრად მეტს ყიდის, ვიდრე სხვა;
- department identity ერთ-ერთი ყველაზე ძლიერი signal-ია.

### Store_Dept Aggregates

მაგალითები:

```python
Store_Dept_Weekly_Sales_mean
Store_Dept_Weekly_Sales_median
Store_Dept_Weekly_Sales_std
```

რატომ:

- თითოეულ Store_Dept pair-ს თავისი baseline sales level აქვს;
- global model-ს, რომელიც ყველა store/dept-ზე ერთად trainდება, ეს feature ძალიან ეხმარება.

### Type_Dept Aggregates

მაგალითები:

```python
Type_Dept_Weekly_Sales_mean
Type_Dept_Weekly_Sales_median
Type_Dept_Weekly_Sales_std
```

რატომ:

- fallback signal-ია, როცა კონკრეტული Store_Dept history სუსტია ან საერთოდ არ არსებობს;
- store type + department behavior-ს იჭერს.

### Missing Aggregate Fill

თუ validation/test-ში ისეთი group გამოჩნდა, რომელიც train-ში არ იყო, aggregate value ივსება global statistic-ით.

რატომ:

- new/rare group-ებზე missing values არ გვრჩება;
- model იღებს reasonable fallback-ს.

Leakage note:

ეს aggregates აუცილებლად train split-ზე უნდა fit-დეს. თუ full train + validation-ზე დაითვლება, validation target information გაჟონავს feature-ებში.

## 8. Column Dropping

კლასი:

```python
ColumnDropper
```

default dropped columns:

```python
Date
Weekly_Sales
```

რატომ:

- `Weekly_Sales` target-ია და feature-ებში არ უნდა დარჩეს;
- raw `Date` LightGBM-ს პირდაპირ არ სჭირდება, რადგან მისგან calendar features უკვე შევქმენით;
- model training-მდე target და raw date უნდა ამოვიღოთ.

## 9. Feature Selection

კლასი:

```python
FeatureImportanceSelector
```

ეს არის model-based feature selection.

როგორ მუშაობს:

1. estimator trainდება engineered features-ზე.
2. კითხულობს estimator-ის `feature_importances_`.
3. ინახავს მხოლოდ იმ feature-ებს, რომელთა importance threshold-ზე მეტია.
4. future transform-ზე აბრუნებს მხოლოდ selected columns-ს.

ლოგიკა:

```python
selected_features = feature_importances[feature_importances > threshold]
```

### რატომ Model-Based Feature Selection?

LightGBM ბევრ feature-ს კარგად უმკლავდება, ამიტომ დასაწყისში aggressive feature selection საჭირო არ არის.

მაგრამ feature selection შეიძლება დაგვეხმაროს:

- zero-importance feature-ების მოშორებაში;
- training time-ის შემცირებაში;
- model explanation-ის გამარტივებაში;
- noise-ის შემცირებაში, თუ validation score გაუმჯობესდება.

### Recommended Threshold

საწყისად გამოიყენე:

```python
threshold = 0.0
```

ეს აშორებს მხოლოდ იმ feature-ებს, რომლებიც model-მა საერთოდ არ გამოიყენა.

ძალიან ბევრი feature თავიდანვე არ წაშალო. ზოგ feature-ს დაბალი individual importance აქვს, მაგრამ სხვა feature-თან interaction-ში შეიძლება სასარგებლო იყოს.

### Validation Rule

Feature selection უნდა დავტოვოთ მხოლოდ მაშინ, თუ validation score გაუმჯობესდა ან იგივე დარჩა.

რეკომენდებული პროცესი:

1. Train LightGBM ყველა engineered feature-ით.
2. შეინახე validation WMAE.
3. აირჩიე non-zero-importance features.
4. თავიდან train LightGBM selected features-ით.
5. შეადარე validation WMAE.
6. selected feature set დატოვე მხოლოდ მაშინ, თუ WMAE გაუმჯობესდა ან simpler model გჭირდება.

## 10. Default Pipeline

ფუნქცია:

```python
make_walmart_lgbm_feature_pipeline
```

default order:

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

ეს order მნიშვნელოვანია.

Cleaning პირველია, რადგან შემდეგ ნაბიჯებს სჭირდება სწორი date, numeric markdown values და categorical columns.

Calendar და holiday features იქმნება raw `Date`-დან.

Markdown features იქმნება markdown cleaning-ის შემდეგ, რადგან missing values უკვე 0-ითაა შევსებული.

Historical aggregates უნდა შეიქმნას `Weekly_Sales` drop-მდე, რადგან target column სჭირდება.

Lag და rolling features-ებსაც `Weekly_Sales` სჭირდება.

ბოლოს `Date` და `Weekly_Sales` იშლება model training-მდე.

## 11. რას იღებს LightGBM

Pipeline-ის შემდეგ LightGBM იღებს feature table-ს, სადაც არის:

- cleaned numeric features;
- categorical features: `Store`, `Dept`, `Type`;
- calendar features;
- holiday flags და proximity features;
- markdown amount, presence, log და interaction features;
- Store_Dept და Type_Dept categorical interactions;
- lag და rolling sales features;
- historical target aggregates.

Target ცალკეა:

```python
Weekly_Sales
```

და model-ს უნდა გადაეცეს როგორც `y_train`.

## 12. მთავარი Leakage Risks

ყველაზე მნიშვნელოვანი leakage risks:

```text
1. Historical aggregates validation/test target values-ზე დათვლა.
2. Rolling features current row target-ის გამოყენებით.
3. Test lag features unknown future sales-ით.
4. Forecasting task-ზე random train/validation split.
```

თავიდან ასაცილებლად:

- aggregate transformer fit გააკეთე მხოლოდ train split-ზე;
- rolling statistics დათვალე `shift(1)`-ის შემდეგ;
- გამოიყენე time-based validation;
- test inference-ზე lag feature-ებს ძალიან ფრთხილად მოექეცი.

## 13. პრაქტიკული რეკომენდაცია

პირველი LightGBM experiment-ისთვის:

1. გამოიყენე ყველა engineered feature.
2. validation გააკეთე time-based split-ით.
3. metric-ად გამოიყენე weighted MAE.
4. ნახე feature importance.
5. წაშალე მხოლოდ zero-importance features.
6. თავიდან train გააკეთე და შეადარე validation WMAE.

მოსალოდნელად ყველაზე მნიშვნელოვანი feature groups იქნება:

```text
1. lag features
2. rolling sales features
3. Store/Dept identity და aggregates
4. holiday features
5. markdown features
6. calendar features
```

ამ Walmart dataset-ში historical sales features, როგორც წესი, უფრო მნიშვნელოვანია, ვიდრე რთული feature selection მეთოდები.

## Feature engineering-ის საბოლოო აუდიტირებული როლი

საბოლოო tree-based flow-ში feature მხოლოდ მაშინ ითვლება დასაშვებად, თუ validation-ისა და Kaggle test-ის forecast origin-ზე ხელმისაწვდომია. ამიტომ safe set მოიცავს calendar/holiday/store/markdown/external მონაცემებს და observed history-დან `SalesLag52`-ს; future target-ზე დამოკიდებული `lag_1/4/13` და rolling statistics აკრძალულია. ამ parity-მ LightGBM Kaggle score ძველი დაახლოებით `6200`-დან საბოლოო `2809`-მდე მიიყვანა. იგივე პრინციპი XGBoost-ის `2806` champion შედეგის მთავარი საფუძველიცაა.
