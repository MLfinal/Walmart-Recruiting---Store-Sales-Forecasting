# LightGBM ექსპერიმენტის აღწერა

ეს დოკუმენტი ხსნის, რა არის გაკეთებული `model_experiment_LightGBM.ipynb` notebook-ში, რატომ დაემატა თითოეული feature engineering და feature selection ნაბიჯი, და როგორ წარიმართა LightGBM-ის Optuna/W&B training ექსპერიმენტი.

## Notebook-ის სტრუქტურა

Notebook დაყოფილია რამდენიმე მთავარ ეტაპად:

1. Environment setup და Google Drive-ის mount.
2. მონაცემების წაკითხვა და merge.
3. Time-based train/validation split.
4. Feature engineering transformer-ების definition.
5. Feature engineering run-ის W&B-ში დალოგვა.
6. Feature selection run-ის W&B-ში დალოგვა.
7. Optuna LightGBM training run-ების W&B-ში დალოგვა.

ეს სტრუქტურა ერგება პროექტის მოთხოვნას: preprocessing/feature engineering, feature selection და model training ცალ-ცალკე ეტაპებადაა გამოყოფილი და W&B-ში ცალკე run-ებად ილოგება.

## ახალი LightGBM correction: რატომ ვცვლით FE/FS-ს

ბოლო LightGBM submission-მა Kaggle-ზე მოსალოდნელზე ცუდი შედეგი მოგვცა:

```text
LightGBM Kaggle score: 3600
XGBoost Kaggle score: 2806
SARIMAX Kaggle score: 3525
```

ეს ნიშნავს, რომ LightGBM-ის წინა feature engineering validation-ზე კარგი ჩანდა, მაგრამ public test-ზე კარგად არ გადაგვყვა. ამიტომ ამ ეტაპზე ვთვლი, რომ ბოლო FE/FS version ოპტიმალური არ იყო. პრობლემა უფრო feature/inference consistency-შია, ვიდრე LightGBM model architecture-ში.

ამიტომ `model_experiment_LightGBM.ipynb`-ში დავამატეთ XGBoost notebook-თან უფრო ახლოს მდგომი feature engineering:

- `SalesLag52` აღარ ივსება median-ით; missing value რჩება `NaN`, რომ LightGBM-მა native missing handling გამოიყენოს.
- `SalesLag52_available` ინახავს, არსებობს თუ არა same Store-Dept 52-week lag.
- `Store_Dept` და `Type_Dept` interaction features გადავიყვანეთ numeric encoding-ზე, XGBoost-ის მსგავსად.
- historical aggregate feature-ებს დაემატა count feature-ებიც:
  - `Store_Sales_count`
  - `Dept_Sales_count`
  - `Store_Dept_Sales_count`
  - `Type_Dept_Sales_count`
- shifted expanding target aggregate logic უფრო ახლოსაა XGBoost-ის time-safe encoder-თან.
- registered LightGBM pipeline ახლა raw `test.csv` rows-საც იღებს და თვითონ merge-ს აკეთებს stored `features.csv`/`stores.csv` tables-თან.

რატომ არის ეს უკეთესი მიმართულება:

- XGBoost-ის უკეთესი Kaggle score გვაჩვენებს, რომ row-level FE უფრო სწორად იყო მოწყობილი.
- LightGBM-საც იგივე signal-ები სჭირდება: 52-week lag, Store-Dept identity, calendar/holiday features, markdown interaction და historical aggregate encodings.
- დამატებითი hyperparameter tuning აზრს კარგავს, თუ train/test feature generation ბოლომდე სანდო არ არის.

ამ ახალი ვერსიის მიზანი არ არის ძველი ექსპერიმენტის წაშლა. ძველი result რჩება შედარებისთვის, მაგრამ final comparison-ში უფრო სანდოდ უნდა ჩაითვალოს corrected FE/FS setup, რადგან ის validation-ს და real Kaggle inference-ს ერთნაირ feature availability-ზე აყენებს.

Corrected FE/FS setup-ის ხელახალი Optuna run-მა validation-ზე ასეთი შედეგი მოგვცა:

```text
Best trial: 49
Validation Weighted MAE: 1615.4495
Validation MAE: 1608.3943
```

ეს validation-ზე უკეთესია წინა safe `SalesLag52` run-ზე (`1633.3693 -> 1615.4495`) და თითქმის იგივე დონეზეა ძველ unsafe validation score-თან, მაგრამ unsafe lag/rolling leakage-ის გარეშე. ამიტომ ამ შედეგს უფრო სანდოდ ვთვლი, ვიდრე ძველ `1573.4988` validation score-ს.

## Baseline LightGBM შედეგი

Baseline notebook არის `baseline_lightgbm.ipynb`. მისი მიზანია გვქონდეს მარტივი საწყისი შედეგი, რომელსაც შევადარებთ feature engineering + feature selection + Optuna ექსპერიმენტს.

Baseline-ში არ გამოიყენება advanced feature engineering:

- არ არის calendar feature-ები;
- არ არის holiday proximity feature-ები;
- არ არის markdown interaction feature-ები;
- არ არის lag/rolling feature-ები;
- არ არის historical aggregate feature-ები;
- არ არის feature selection;
- არ არის Optuna tuning.

მხოლოდ მინიმალური preprocessing გაკეთდა, რაც LightGBM-ს training-ისთვის სჭირდება:

- `Date` სვეტის drop;
- markdown missing value-ების `0`-ით შევსება;
- numeric missing value-ების train median-ით შევსება;
- `Type` categorical feature-ად გადაყვანა;
- chronological last-32-weeks validation split.

მოწოდებული baseline training log:

```text
[50]  train's l1: 5929.85  validation's l1: 5563.04
[100] train's l1: 4701.74  validation's l1: 4349.25
[150] train's l1: 4260.91  validation's l1: 4018.80
[200] train's l1: 3950.84  validation's l1: 3759.19
[250] train's l1: 3726.14  validation's l1: 3587.61
[300] train's l1: 3552.86  validation's l1: 3461.04
[350] train's l1: 3444.44  validation's l1: 3389.69
[400] train's l1: 3358.04  validation's l1: 3330.70
[450] train's l1: 3273.90  validation's l1: 3266.29
[500] train's l1: 3178.69  validation's l1: 3188.88
```

Baseline-ის საუკეთესო დალოგილი validation L1 ამ run-ში არის:

```text
Validation L1: 3188.88
Iteration: 500
```

ეს შედეგი აჩვენებს, რომ raw merged feature-ებით LightGBM სწავლობს general sales pattern-ებს, მაგრამ error ჯერ კიდევ მაღალია. Validation L1 თანმიმდევრულად იკლებს 50-დან 500 iteration-მდე, რაც ნიშნავს, რომ baseline model ბოლომდე არ არის saturated და მეტი tree ან უკეთესი feature-ები შეიძლება დაეხმაროს.

მნიშვნელოვანია, რომ train L1 და validation L1 ძალიან ახლოს არის ბოლო iteration-ზე:

```text
train L1: 3178.69
validation L1: 3188.88
```

ეს არ ჰგავს ძლიერ overfitting-ს. უფრო ჩანს, რომ baseline model under-featured არის: მას არ აქვს seasonality, holiday proximity, lag/rolling და historical aggregate signals, ამიტომ ვერ იჭერს Walmart sales-ის მთავარ structure-ს.

შედარებისთვის, corrected engineered LightGBM Optuna run-ის საუკეთესო შედეგი იყო:

```text
Validation Weighted MAE: 1615.4495
Validation MAE: 1608.3943
```

პირდაპირი შედარებისას სიფრთხილეა საჭირო, რადგან baseline log-ში მოცემულია LightGBM-ის `validation_l1`, ხოლო final experiment-ის მთავარი metric არის weighted MAE. მაგრამ მაინც ჩანს ძლიერი improvement: engineered model-ის validation error დაახლოებით ორჯერ დაბალია baseline validation L1-ზე.

დასკვნა:

- baseline LightGBM კარგი საწყისი წერტილია;
- baseline არ ჩანს overfitted;
- მთავარი პრობლემა feature signal-ის ნაკლებობაა;
- feature engineering-მა, feature selection-მა და Optuna tuning-მა მნიშვნელოვანი improvement მოიტანა;
- საბოლოო model selection მაინც უნდა გაკეთდეს `Validation Weighted MAE`-ით, რადგან competition metric ეს არის.

## მონაცემების გაყოფა

Notebook-ში გამოყენებულია chronological validation split და არა random split.

```python
VALIDATION_WEEKS = 32
validation_dates = np.sort(df_train_merged["Date"].unique())[-VALIDATION_WEEKS:]
```

ბოლო 32 unique weekly date მიდის validation-ში. ყველა წინა კვირა რჩება training-ში.

რატომ:

- ამოცანა არის time-series forecasting.
- random split აურევს წარსულსა და მომავალს.
- time-based split უკეთ ასახავს რეალურ forecasting სიტუაციას: მოდელი სწავლობს წარსულზე და პროგნოზირებს მომავალს.

## Metric Evaluation

ამ ამოცანაში მთავარი შეფასების metric არის **Weighted Mean Absolute Error (WMAE)**, რადგან Walmart Kaggle competition სწორედ ამ metric-ით აფასებს submission-ს.

გამოყენებული metric-ები:

| Metric | სად გამოიყენება | რატომ |
| --- | --- | --- |
| `Weighted MAE` / `WMAE` | მთავარი validation score და Optuna objective | ემთხვევა competition metric-ს და holiday week-ებს უფრო დიდ წონას აძლევს |
| `MAE` | დამატებითი validation diagnostic | აჩვენებს საშუალო absolute error-ს ყველა row-ზე თანაბარი წონით |
| `train_l1` / `validation_l1` | LightGBM training curves/W&B charts | გვაჩვენებს, როგორ მცირდება L1 loss train და validation ეტაპებზე |
| `RMSE` | ზოგადი diagnostic, საჭიროების შემთხვევაში | დიდ შეცდომებს უფრო მკაცრად სჯის, მაგრამ competition-ის მთავარი metric არ არის |

### WMAE ფორმულა

Competition metric ითვლება ასე:

```python
WMAE = sum(weights * abs(y_true - y_pred)) / sum(weights)
```

სადაც:

```python
weight = 5  # holiday week
weight = 1  # non-holiday week
```

ანუ holiday rows ხუთჯერ უფრო მნიშვნელოვანია, ვიდრე ჩვეულებრივი კვირები.

### რატომ არის WMAE საუკეთესო metric ამ task-ისთვის

Walmart sales forecasting-ში ყველა კვირა ერთნაირად მნიშვნელოვანი არ არის. Holiday კვირებში, მაგალითად Thanksgiving, Christmas, Labor Day და Super Bowl period-ში, გაყიდვები მკვეთრად იცვლება და ბიზნესისთვის პროგნოზის შეცდომა უფრო ძვირია.

ამიტომ `MAE` მარტო საკმარისი არ არის: ის holiday და non-holiday rows-ს ერთნაირად აფასებს. მაგალითად, თუ model holiday week-ზე ძალიან ცუდად ცდება, მაგრამ ordinary week-ებზე კარგად მუშაობს, ჩვეულებრივი MAE ამას სრულად ვერ ასახავს. WMAE კი holiday შეცდომას უფრო დიდ მნიშვნელობას ანიჭებს და ზუსტად competition-ის business logic-ს მიჰყვება.

ამის გამო Optuna optimization-ის დროს საუკეთესო model შეირჩა არა უბრალო MAE-ით, არამედ validation WMAE-ით:

```python
weighted_mae = np.sum(np.abs(y_val - y_pred_val) * sample_weights_val) / np.sum(sample_weights_val)
```

### რატომ მაინც ვლოგავთ MAE-საც

MAE მაინც სასარგებლოა, რადგან გვაჩვენებს model-ის საშუალო შეცდომას ყველა row-ზე თანაბრად. თუ WMAE და MAE ძალიან განსხვავდება, ეს ნიშნავს, რომ model holiday rows-ზე სხვანაირად იქცევა, ვიდრე ordinary rows-ზე.

ამ corrected notebook run-ში საუკეთესო trial-ისთვის დალოგილი იყო:

```text
Validation Weighted MAE: 1615.4495
Validation MAE: 1608.3943
```

WMAE ოდნავ მაღალია MAE-ზე, რაც ბუნებრივია, რადგან holiday rows უფრო მძიმე წონით ითვლება და ისინი პროგნოზირებისთვის უფრო რთულია.

### რატომ არ ავირჩიეთ RMSE მთავარ metric-ად

RMSE დიდ შეცდომებს უფრო მკაცრად სჯის, რადგან error-ს კვადრატში იღებს. ეს diagnostic-ად შეიძლება სასარგებლო იყოს, მაგრამ ამ competition-ში leaderboard WMAE-ით ითვლება. ამიტომ თუ model-ს RMSE უკეთესი აქვს, მაგრამ WMAE უარესი, ასეთი model ამ კონკრეტული დავალებისთვის არ არის საუკეთესო.

დასკვნა:

- მთავარი model selection metric არის `Validation Weighted MAE`.
- `MAE` გამოიყენება როგორც დამატებითი sanity check.
- `train_l1` და `validation_l1` გამოიყენება learning curve-ების შესაფასებლად.
- `RMSE` შეიძლება დაგვეხმაროს დიდი შეცდომების აღმოჩენაში, მაგრამ არ არის მთავარი optimization target.

## Feature Engineering

Feature engineering დაწერილია sklearn-ის სტილით:

```python
BaseEstimator, TransformerMixin
```

ეს დიზაინი შეირჩა იმიტომ, რომ თითოეული transformation შეიძლება train data-ზე fit-დეს და შემდეგ იგივე წესით validation/test data-ზე transform გაკეთდეს.

## 1. WalmartFeatureCleaner

ეს transformer აკეთებს საწყის cleaning-ს feature-ების შექმნამდე.

რას აკეთებს:

- `Date` სვეტს გარდაქმნის datetime ფორმატში.
- `MarkDown1`-დან `MarkDown5`-მდე ქმნის missing indicator-ებს.
- markdown missing value-ებს ავსებს `0`-ით.
- საჭიროების შემთხვევაში numeric missing value-ებს ავსებს median-ით.
- `Store`, `Dept`, `Type` სვეტებს გარდაქმნის pandas `category` ტიპში.
- `IsHoliday` სვეტს გარდაქმნის integer მნიშვნელობად.

რატომ დაემატა:

- markdown missingness ამ dataset-ში ინფორმაციულია, რადგან markdown data მხოლოდ გარკვეული დროის შემდეგ ჩნდება.
- markdown-ის `0`-ით შევსება მოდელს აძლევს საშუალებას missing promotion ჩათვალოს როგორც no recorded promotion, ხოლო missing indicator ინახავს ინფორმაციას, რომ მნიშვნელობა თავიდან არ არსებობდა.
- LightGBM categorical feature-ებს კარგად ამუშავებს, თუ ისინი სწორად არის მონიშნული.

## 2. CalendarFeatureTransformer

ეს transformer `Date` სვეტიდან ქმნის calendar feature-ებს:

- `Year`
- `Month`
- `WeekOfYear`
- `Quarter`
- `DayOfYear`
- `DaysFromStart`
- `WeekSin`
- `WeekCos`
- `MonthSin`
- `MonthCos`

რატომ დაემატა:

- Walmart sales-ს ძლიერი seasonality აქვს.
- Weekly sales ხშირად დამოკიდებულია month-ზე, week of year-ზე და holiday season-ზე.
- `DaysFromStart` მოდელს აძლევს trend feature-ს.
- `WeekSin` და `WeekCos` week of year-ს წრიულად encode-ს უკეთებს, ამიტომ week 52 და week 1 ერთმანეთთან ახლოს გამოდის.

Tree model-ებისთვის `WeekOfYear` თვითონაც სასარგებლოა, მაგრამ cyclical feature-ები მოდელს seasonality-ის ალტერნატიულ representation-ს აძლევს.

## 3. WalmartHolidayFeatureTransformer

ეს transformer ამატებს competition-ის მთავარ holiday feature-ებს:

- Super Bowl
- Labor Day
- Thanksgiving
- Christmas

ქმნის holiday-week flag-ებს და proximity feature-ებს, მაგალითად:

- `IsChristmasWeek`
- `DaysToNearestChristmas`
- `WeeksToNearestChristmas`

რატომ დაემატა:

- Kaggle metric holiday weeks-ს უფრო მაღალ წონას ანიჭებს.
- სხვადასხვა holiday გაყიდვებზე სხვადასხვანაირად მოქმედებს.
- sales შეიძლება შეიცვალოს holiday-მდე ან holiday-ის შემდეგაც, არა მხოლოდ ზუსტად holiday week-ში.

## 4. MarkdownFeatureTransformer

ეს transformer ქმნის promotion-related feature-ებს:

- `TotalMarkDown`
- `HasMarkDown1`-დან `HasMarkDown5`-მდე
- `HasAnyMarkDown`
- log-transformed markdown features
- holiday-markdown interaction features

რატომ დაემატა:

- promotion-ებს შეუძლია sales-ზე გავლენის მოხდენა, განსაკუთრებით holiday period-ში.
- `TotalMarkDown` აჩვენებს promotion-ის საერთო ძალას.
- binary markdown flags მოდელს ეხმარება განასხვავოს no promotion და some promotion.
- log markdown feature-ები ამცირებს ძალიან დიდი markdown value-ების გავლენას.
- holiday interaction feature-ები მოდელს აძლევს საშუალებას holiday weeks-ში promotion სხვანაირად დაინახოს.

## 5. InteractionFeatureTransformer

ეს transformer ქმნის categorical interaction feature-ებს:

- `Store_Dept`
- `Type_Dept`

რატომ დაემატა:

- თითოეული store-department pair რეალურად ცალკე time series-ს ჰგავს.
- ერთი და იგივე department სხვადასხვა store-ში განსხვავებულად იყიდება.
- store type department-level behavior-ზე მოქმედებს.

## 6. HistoricalAggregateTransformer

ეს transformer training data-დან ქმნის target-based historical aggregate feature-ებს:

- Store-level sales mean/median/std.
- Department-level sales mean/median/std.
- Store-department sales mean/median/std.
- Type-department sales mean/median/std.

რატომ დაემატა:

- `Store` და `Dept` identity ამ dataset-ში ერთ-ერთი ყველაზე ძლიერი signal-ია.
- Historical average-ები მოდელს baseline sales level-ს აძლევს.
- ეს aggregate feature-ები ეხმარება sparse ან cold-start store-department pair-ებს.

მნიშვნელოვანი შენიშვნა:

ეს aggregate-ები validation/test-ზე ითვლება მხოლოდ training data-ზე fit-ებული mapping-ებიდან. Training rows-ზე კი გამოიყენება expanding historical aggregate logic, რომ row-მ თავისივე `Weekly_Sales` მნიშვნელობა არ გამოიყენოს. ეს XGBoost notebook-ის time-safe aggregate იდეასთან არის დაახლოებული და target leakage-ის რისკს ამცირებს.

## 7. LagRollingFeatureTransformer ძველი ექსპერიმენტი

პირველ LightGBM training ექსპერიმენტში აქტიურად გამოიყენებოდა lag/rolling feature engineering.

ეს transformer ქმნიდა:

- `lag_1`
- `lag_4`
- `lag_13`
- `lag_52`
- rolling mean feature-ებს
- rolling standard deviation feature-ებს

რატომ დაემატა თავდაპირველად:

- Weekly sales ძლიერად autocorrelated არის.
- წინა კვირის, წინა თვის, წინა კვარტლის და წინა წლის იგივე კვირის sales ძლიერი predictor-ებია.
- Rolling mean noisy weekly sales-ს ასწორებს.
- Rolling standard deviation volatility-ს აღწერს.

ამ ძველი feature set-ით საუკეთესო დალოგილი Optuna validation result იყო:

```text
Validation Weighted MAE: 1573.4988
Validation MAE: 1543.4832
Best trial: 46
```

მაგრამ ამ feature set-ს ჰქონდა მნიშვნელოვანი რისკი: validation dataframe-ში `Weekly_Sales` უკვე ცნობილი იყო, ამიტომ `lag_1`, `lag_4`, `lag_13` და rolling feature-ები validation-ზე ზედმეტად optimistic score-ს იძლეოდა. Kaggle test-ზე future `Weekly_Sales` უცნობია, ამიტომ იგივე feature-ები პირდაპირ ვერ შეიქმნება.

ამის გამო Kaggle score ბევრად გაუარესდა, მიუხედავად იმისა, რომ validation WMAE ძალიან კარგი ჩანდა.

## 8. Safe SalesLag52 feature მიმდინარე retrain-ისთვის

ახალი LightGBM feature engineering გადაკეთდა XGBoost-ის მსგავსად და აქტიურ pipeline-ში დარჩა მხოლოდ უსაფრთხო yearly lag:

- `SalesLag52`
- `SalesLag52_available`

`SalesLag52` ითვლება მხოლოდ observed history-დან:

```python
history_date = current_date - 52 weeks
```

ანუ validation/test row არ იყენებს თავისივე `Weekly_Sales` მნიშვნელობას და არ იყენებს future sales-ს. თუ 52 კვირით უკან შესაბამისი row არ არსებობს, `SalesLag52` რჩება `NaN`, ხოლო `SalesLag52_available = 0` ინახავს ინფორმაციას, რომ რეალური lag ვერ მოიძებნა. ეს უკეთ ერგება LightGBM-ს, რადგან LightGBM missing values-ს native-ად ამუშავებს და median-fill-ით ხელოვნურ signal-ს აღარ ვუმატებთ.

რატომ არის ეს უფრო სწორი:

- Kaggle inference-ზე იგივე logic მუშაობს, რაც validation-ზე.
- model აღარ ეყრდნობა unknown future sales-ს.
- validation score ნაკლებად optimistic უნდა იყოს.
- Kaggle score უფრო ახლოს უნდა მივიდეს validation score-თან.

ამ ცვლილების შემდეგ LightGBM ხელახლა გავუშვით safe setup-ზე. ძველი Optuna result (`1573.4988`) არ უნდა წაიშალოს, რადგან comparison-ისთვის საჭიროა, მაგრამ final model selection-ისთვის ის სანდო აღარ არის unsafe lag/rolling feature-ების გამო.

## Feature Engineering W&B Run

Feature engineering ეტაპი W&B-ში ილოგება როგორც:

```text
LightGBM_Feature_Engineering
```

ლოგავს:

- training row count
- validation row count
- engineered feature count
- categorical feature count
- train missing value count
- validation missing value count
- feature metadata table

ეს preprocessing ეტაპს W&B-ში ხილულს ხდის, ანუ მარტო final model training არ ილოგება.

## Feature Selection

Feature selection ილოგება როგორც:

```text
LightGBM_Feature_Selection
```

გამოყენებულია model-based feature selection:

1. LightGBM trainდება engineered feature-ებზე.
2. იღებს `feature_importances_` მნიშვნელობებს.
3. ინარჩუნებს feature-ებს, რომელთა importance მეტია `0`-ზე.
4. შლის zero-importance feature-ებს.
5. Optuna training უკვე selected feature-ებზე ეშვება.

რატომ დაემატა:

- LightGBM ბევრ feature-ს კარგად უმკლავდება, მაგრამ zero-importance feature-ები noise-ს და complexity-ს ზრდის.
- Tree model-ისთვის model-based feature selection ბუნებრივი და სწრაფი მიდგომაა.
- feature reduction ეტაპი უფრო explainable ხდება.

Run-ში ილოგება:

- input feature count
- selected feature count
- dropped feature count
- selected ratio
- full feature importance table
- selected feature list
- dropped feature list

დალოგილი run-ის მიხედვით feature selection-მა feature set შეამცირა 82 feature-დან 47 selected feature-მდე.

## Training პროცესი

Training იყენებს LightGBM-ს Optuna hyperparameter optimization-თან ერთად.

Objective function თითო Optuna trial-ზე train-ს უკეთებს ერთ LightGBM model-ს და თითოეული trial W&B-ში ილოგება ასე:

```text
lightgbm-optuna-trial-{trial_number}
```

თითოეული trial ლოგავს:

- training MAE curve
- validation MAE curve
- validation weighted MAE
- validation MAE
- feature importance table
- actual vs predicted plot
- trial hyperparameters

Model objective არის:

```python
objective = "mae"
```

ეს სწორია, რადგან competition metric დაფუძნებულია weighted absolute error-ზე.

Validation metric ითვლება როგორც weighted MAE:

```python
weighted_mae = sum(abs(y_true - y_pred) * weights) / sum(weights)
```

Holiday rows იღებს weight `5`, non-holiday rows იღებს weight `1`.

## Hyperparameter Search Space

Optuna ცვლიდა შემდეგ LightGBM hyperparameter-ებს:

| Hyperparameter | Search Range / ქცევა | მნიშვნელობა |
| --- | --- | --- |
| `learning_rate` | `0.01`-დან `0.1`-მდე, log scale | თითო boosting iteration-ის step size |
| `num_leaves` | `20`-დან `256`-მდე | ერთ tree-ში leaf-ების მაქსიმალური რაოდენობა |
| `max_depth` | `5`-დან `20`-მდე | tree-ის მაქსიმალური სიღრმე |
| `min_child_samples` | `20`-დან `100`-მდე | leaf-ში მინიმალური row count |
| `subsample` | `0.7`-დან `1.0`-მდე | row sampling ratio |
| `subsample_freq` | fixed `1` | row sampling-ის ჩართვა ყოველ iteration-ზე |
| `colsample_bytree` | `0.7`-დან `1.0`-მდე | feature sampling ratio |
| `reg_alpha` | `0.0`-დან `0.1`-მდე | L1 regularization |
| `reg_lambda` | `0.0`-დან `0.1`-მდე | L2 regularization |
| `n_estimators` | fixed `100` | boosting round-ების რაოდენობა |

## საუკეთესო Trial corrected FE/FS run-ში

ეს section ეხება მხოლოდ ახალ corrected FE/FS run-ს. ძველი results ქვემოთ ცალკე რჩება historical comparison-ისთვის და არ იშლება.

მოწოდებული corrected FE/FS Optuna logs-ის მიხედვით საუკეთესო run იყო:

```text
Trial: 49
Validation Weighted MAE: 1615.4495
Validation MAE: 1608.3943
```

საუკეთესო hyperparameter-ები:

```python
{
    "learning_rate": 0.08865537207323315,
    "num_leaves": 253,
    "max_depth": 11,
    "min_child_samples": 83,
    "subsample": 0.8136830368318766,
    "colsample_bytree": 0.8249071686333185,
    "reg_alpha": 0.005530932818058136,
    "reg_lambda": 0.0395233307315138,
}
```

ამ trial-მა საუკეთესო შედეგი აჩვენა, რადგან ჰქონდა:

- შედარებით მაღალი learning rate;
- ძალიან დიდი `num_leaves`;
- საშუალო სიღრმის trees;
- ზომიერი row sampling;
- ზომიერი feature sampling;
- მსუბუქი, მაგრამ ნულზე მაღალი regularization.

ეს combination მოდელს აძლევს საშუალებას დაიჭიროს რთული store-department-seasonality interactions, მაგრამ `max_depth = 11`, `min_child_samples = 83`, `subsample = 0.8137` და regularization ერთად ამცირებს overfitting-ის რისკს. ეს განსაკუთრებით მნიშვნელოვანია corrected setup-ში, სადაც unsafe short lag/rolling feature-ები აღარ გვაქვს და model უფრო მეტად ეყრდნობა safe yearly lag-ს, historical aggregate-ებს და interaction feature-ებს.

### ძველ შედეგებთან მოკლე შედარება

| ეტაპი | Best trial | Validation WMAE | Kaggle score | როგორ ვკითხულობ |
| --- | ---: | ---: | ---: | --- |
| ძველი unsafe lag/rolling FE | 46 | `1573.4988` | დაახლოებით `6200` | validation leakage-ის გამო ზედმეტად optimistic |
| პირველი safe `SalesLag52` FE | 42 | `1633.3693` | დაახლოებით `3600` | Kaggle-ზე ბევრად უკეთესი, რადგან feature availability უფრო რეალურია |
| corrected XGBoost-aligned FE | 49 | `1615.4495` | დაახლოებით `3490` | validation და Kaggle ორივე გაუმჯობესდა წინა safe setup-თან შედარებით |

ამიტომ ახალი result ძველ unsafe score-ს პირდაპირ არ უნდა შევადაროთ როგორც “worse/better model”, რადგან ძველი `1573.4988` validation leakage-ით იყო გაძლიერებული. სწორი შედარება არის safe setup-თან: `1633.3693 -> 1615.4495`, რაც გაუმჯობესებაა leakage-ის დაბრუნების გარეშე.

## Trial-ების შედარება corrected FE/FS run-ში

Representative trial results:

| Trial | Weighted MAE | ძირითადი pattern |
| --- | ---: | --- |
| 0 | 2785.06 | დაბალი learning rate და 100 round-ში ნელი learning |
| 1 | 2014.69 | shallow depth-მა და უკეთესმა learning rate-მა დიდი improvement მისცა |
| 10 | 1841.82 | მაღალი learning rate და პატარა `num_leaves`; ძლიერი early improvement |
| 18 | 1681.48 | balanced learning rate/leaves; first strong corrected setup |
| 25 | 1648.08 | მაღალი learning rate და საშუალო tree capacity |
| 31 | 1626.35 | მაღალი learning rate, 187 leaves, shallow depth; თითქმის საუკეთესო |
| 43 | 1624.28 | `num_leaves = 210`, `max_depth = 10`; კარგი balance |
| 45 | 1619.26 | უფრო დიდი leaves და `max_depth = 12`; trial 49-მდე საუკეთესო |
| 49 | 1615.45 | საუკეთესო overall; დიდი leaves + medium depth + moderate sampling |

## Hyperparameter-ების ქცევა

### Learning Rate

ყველაზე მნიშვნელოვანი pattern იყო learning rate.

დაბალი learning rate-ები, დაახლოებით `0.01`-დან `0.02`-მდე, ცუდად მუშაობდა, რადგან `n_estimators = 100` იყო fixed.

მაგალითები:

- Trial 4: `learning_rate = 0.0116`, WMAE `5603.77`
- Trial 9: `learning_rate = 0.0101`, WMAE `6260.35`
- Trial 47: `learning_rate = 0.0163`, WMAE `4106.75`

ეს models ჯერ კიდევ სწავლობდნენ, მაგრამ 100 boosting round საკმარისი არ იყო.

მაღალი learning rate-ები, დაახლოებით `0.06`-დან `0.09`-მდე, ბევრად უკეთესი იყო:

- Trial 16: `0.0583`, WMAE `1636.31`
- Trial 31: `0.0994`, WMAE `1626.35`
- Trial 45: `0.0878`, WMAE `1619.26`
- Trial 49: `0.0887`, WMAE `1615.45`

დასკვნა:

რადგან `n_estimators = 100`, higher learning rate აუცილებელი აღმოჩნდა. თუ learning rate დაბალი იქნება, მაშინ `n_estimators` უნდა გაიზარდოს.

### Number of Leaves

საუკეთესო trial-ები ძირითადად იყენებდნენ ბევრ leaves-ს:

- Trial 31: `num_leaves = 187`
- Trial 43: `num_leaves = 210`
- Trial 45: `num_leaves = 237`
- Trial 49: `num_leaves = 253`

ეს ნიშნავს, რომ მოდელს სჭირდება complex tree structure, რადგან Walmart sales დამოკიდებულია store, department, week, holiday, markdown და historical sales interaction-ებზე.

მაგრამ მარტო დიდი tree capacity საკმარისი არ არის. Trial 0-ს ჰქონდა `num_leaves = 245`, მაგრამ learning rate დაბალი იყო, ამიტომ შედეგი უარესი გამოვიდა.

### Max Depth

საუკეთესო trial-ებში tree depth მაღალი იყო:

- Trial 31: `max_depth = 10`
- Trial 43: `max_depth = 10`
- Trial 45: `max_depth = 12`
- Trial 49: `max_depth = 11`

ეს აჩვენებს, რომ corrected setup-ში უკიდურესად ღრმა trees აღარ იყო საჭირო. ძველ unsafe feature set-ში საუკეთესო trial-ები `max_depth = 18-20` არეალში იყო, მაგრამ ახალ setup-ში `max_depth = 10-12` უკეთესად generalize-და. ჩემი ინტერპრეტაციით, XGBoost-ის მსგავსად დალაგებულმა interaction/aggregate feature-ებმა model-ს ნაკლები სიღრმითაც მისცა საჭირო signal.

### Min Child Samples

საუკეთესო trial-ს ჰქონდა:

```python
min_child_samples = 83
```

ეს საკმაოდ მაღალი მნიშვნელობაა. ანუ ბევრი leaves მიუხედავად, model პატარა leaf-ებს ერიდებოდა. ეს regularization-ს ქმნის და overfitting-ს ამცირებს.

### Subsample

საუკეთესო trial-ებში row sampling მაღალი იყო:

- Trial 31: `subsample = 0.9036`
- Trial 43: `subsample = 0.8680`
- Trial 45: `subsample = 0.8649`
- Trial 49: `subsample = 0.8137`

ეს აჩვენებს, რომ corrected setup-ში ზომიერი row sampling უკეთესად მუშაობდა. ძველ run-ში საუკეთესო trial-ები თითქმის მთელ dataset-ს იყენებდნენ, მაგრამ ახალ run-ში `0.81-0.90` sampling უკეთეს regularization-ს ქმნის.

### Column Sampling

საუკეთესო trial-ს ჰქონდა:

```python
colsample_bytree = 0.8249
```

სხვა ძლიერი trial-ები იყენებდნენ დაახლოებით `0.80`-დან `0.95`-მდე მნიშვნელობებს.

ეს ნიშნავს, რომ feature-ების დიდი ნაწილი სასარგებლო იყო, მაგრამ ყველა feature-ის გამოყენება თითო tree-ში აუცილებელი არ იყო. Column sampling გარკვეულ regularization-ს ქმნის.

### Regularization

საუკეთესო trial-ში regularization ძალიან სუსტი იყო:

```python
reg_alpha = 0.00553
reg_lambda = 0.03952
```

ეს ნიშნავს, რომ ძლიერი L1/L2 penalty საჭირო არ იყო, მაგრამ ნულზე მაღალი regularization დაეხმარა corrected feature set-ს. `reg_lambda` ძველ best run-ზე უფრო მაღალია, რაც logical არის, რადგან model ახლა უფრო safe, მაგრამ შედარებით ნაკლებად direct target-derived feature-ებზე სწავლობს.

## Training Curves

W&B curves აჩვენებს, რომ training L1 და validation L1 უმეტეს trial-ში იკლებდა 100 iteration-ის განმავლობაში.

ეს ნიშნავს:

- model სტაბილურად სწავლობდა;
- validation error ბევრ run-ში ბოლომდე იკლებდა;
- `n_estimators = 100` ზოგი configuration-ისთვის შეიძლება ცოტაა;
- early stopping არ იყო გამოყენებული, ამიტომ run summary-ში ბოლო iteration არის `99`.

დაბალი learning rate-ის მქონე trial-ებში curves 100 iteration-ის შემდეგაც შორს იყო convergence-დან. მაღალი learning rate-ის მქონე trial-ებში curves უფრო სწრაფად flatten-დებოდა და უკეთეს score-ს აღწევდა.

## ძველი საუკეთესო validation შედეგი unsafe lag/rolling feature-ებით

პირველი LightGBM ექსპერიმენტის საუკეთესო დალოგილი validation result იყო:

```text
Validation Weighted MAE: 1573.4988
Best trial: 46
```

საუკეთესო run:

```text
lightgbm-optuna-trial-46
```

ეს შედეგი validation-ზე ძალიან ძლიერი ჩანდა, მაგრამ Kaggle-ზე იგივე feature engineering-მა დაახლოებით `6200` score მისცა. ამის მთავარი მიზეზი იყო ის, რომ validation-ზე `lag_1`, `lag_4`, `lag_13` და rolling feature-ები რეალურ validation `Weekly_Sales` მნიშვნელობებზე იყო აგებული. Kaggle test-ზე future `Weekly_Sales` არ გვაქვს, ამიტომ იგივე feature availability არ არსებობს.

ამიტომ `1573.4988` აღარ უნდა ჩაითვალოს final reliable validation score-ად. ის დარჩა როგორც ძველი ექსპერიმენტის შედეგი და საჭიროა comparison-ისთვის.

## ახალი Safe SalesLag52 ექსპერიმენტი

შემდეგ LightGBM გადავაკეთეთ XGBoost-ის მსგავსად:

- ამოვიღეთ unsafe short lag feature-ები: `lag_1`, `lag_4`, `lag_13`;
- ამოვიღეთ rolling mean/std feature-ები;
- დავტოვეთ safe yearly lag: `SalesLag52`;
- დავამატეთ `SalesLag52_available`;
- historical aggregates training rows-ზე გადავიდა expanding/time-safe logic-ზე;
- inference pipeline-ში registered model თავად ქმნის `SalesLag52`-ს stored observed history-დან.

ახალი Optuna run-ის შედეგი:

```text
Number of finished trials: 50
Best trial: 42
Validation Weighted MAE: 1633.3693
Validation MAE: 1620.2000
Kaggle score: 3600
```

საუკეთესო hyperparameter-ები:

```python
{
    "learning_rate": 0.0712359757088356,
    "num_leaves": 227,
    "max_depth": 16,
    "min_child_samples": 77,
    "subsample": 0.799399470346009,
    "colsample_bytree": 0.8558722105232142,
    "reg_alpha": 0.031819794172038236,
    "reg_lambda": 0.01422823668758164,
}
```

Best trial-ის final training log:

```text
[25]  train's l1: 4182.68  validation's l1: 3779.92
[50]  train's l1: 2231.64  validation's l1: 1931.73
[75]  train's l1: 1773.73  validation's l1: 1652.98
[100] train's l1: 1643.72  validation's l1: 1623.56
```

### შედარება ძველ feature engineering-თან

| ექსპერიმენტი | Validation WMAE | Kaggle score | შეფასება |
| --- | ---: | ---: | --- |
| ძველი lag/rolling FE | `1573.4988` | დაახლოებით `6200` | validation ძალიან optimistic იყო |
| ახალი safe `SalesLag52` FE | `1633.3693` | დაახლოებით `3600` | validation ოდნავ უარესია, მაგრამ Kaggle ბევრად უკეთესი გახდა |

Validation-ზე ახალი feature engineering ოდნავ უარესია:

```text
1573.4988 -> 1633.3693
```

ეს მოსალოდნელი იყო, რადგან მოდელს წავართვით ძლიერი, მაგრამ unsafe target-derived feature-ები. ძველი `lag_1`, `lag_4`, `lag_13` და rolling feature-ები validation period-ის რეალურ sales-ს ირიბად აწვდიდა მოდელს. ამიტომ validation score ხელოვნურად უკეთესი გამოდიოდა.

Kaggle-ზე კი ახალი feature engineering მნიშვნელოვნად უკეთესია:

```text
6200 -> 3600
```

ეს ნიშნავს, რომ ახალი validation/inference setup უფრო რეალისტურია. მოდელი ახლა ეყრდნობა ისეთ feature-ებს, რომლებიც test set-ზეც ხელმისაწვდომია:

- calendar features;
- holiday/proximity features;
- markdown features;
- store/dept/type interactions;
- historical aggregates training history-დან;
- safe `SalesLag52`.

### რატომ გახდა Kaggle უკეთესი, მიუხედავად იმისა რომ validation გაუარესდა

ძველი validation score უკეთესი იყო, მაგრამ ის არ ასახავდა რეალურ Kaggle inference-ს. Validation-ზე model ხედავდა recent sales behavior-ს `lag_1`, `lag_4`, `lag_13` და rolling feature-ებით, Kaggle test-ზე კი ეს recent true sales არ არსებობს.

ახალ ვერსიაში validation და Kaggle უფრო ერთნაირ წესს იყენებს: ორივეგან `SalesLag52` მოდის მხოლოდ observed history-დან. ამიტომ validation score ნაკლებად ლამაზია, მაგრამ Kaggle score უკეთესად ემთხვევა რეალურ performance-ს.

დასკვნა:

- ძველი feature engineering იყო validation-ზე ძლიერი, მაგრამ test-ზე unreliable.
- ახალი feature engineering არის უფრო honest და Kaggle-safe.
- final comparison-ში ახალი Safe `SalesLag52` model უნდა ჩაითვალოს უფრო სანდოდ, რადგან Kaggle score ბევრად გაუმჯობესდა.
- შემდეგი გაუმჯობესება უნდა იყოს validation procedure-ის კიდევ უფრო დაახლოება Kaggle horizon-თან და possibly full-data retrain final submission-მდე.

## Corrected XGBoost-aligned LightGBM retrain

პირველი safe `SalesLag52` run-ის შემდეგ LightGBM feature engineering კიდევ ერთხელ მივუახლოვეთ XGBoost notebook-ს. მთავარი ცვლილებები იყო:

- `SalesLag52` missing value აღარ ივსება median-ით და რჩება `NaN`;
- LightGBM-ს ვაძლევთ native missing value handling-ის გამოყენების საშუალებას;
- `Store_Dept` და `Type_Dept` interaction feature-ები numeric encoding-ზე გადავიდა;
- historical aggregate transformer-ში დაემატა count feature-ები;
- registered pipeline-ს შეუძლია raw `test.csv` მიიღოს და stored `features.csv`/`stores.csv` თვითონ merge-ოს.

ამ corrected setup-ის Optuna შედეგი:

```text
Number of finished trials: 50
Best trial: 49
Validation Weighted MAE: 1615.4495
Validation MAE: 1608.3943
```

საუკეთესო hyperparameter-ები:

```python
{
    "learning_rate": 0.08865537207323315,
    "num_leaves": 253,
    "max_depth": 11,
    "min_child_samples": 83,
    "subsample": 0.8136830368318766,
    "colsample_bytree": 0.8249071686333185,
    "reg_alpha": 0.005530932818058136,
    "reg_lambda": 0.0395233307315138,
}
```

Best trial-ის final training log:

```text
[25]  train's l1: 3310.53  validation's l1: 2895.50
[50]  train's l1: 1983.45  validation's l1: 1706.42
[75]  train's l1: 1744.14  validation's l1: 1636.23
[100] train's l1: 1641.02  validation's l1: 1625.82
```

### რას ნიშნავს ეს შედეგი

ეს result უკეთესია წინა safe LightGBM run-ზე:

```text
Safe SalesLag52 WMAE: 1633.3693
Corrected XGBoost-aligned WMAE: 1615.4495
Improvement: 17.9198 WMAE
```

ეს გაუმჯობესება პატარაა, მაგრამ მნიშვნელოვანი მიმართულებაა. მთავარი მიზეზი ის არის, რომ model-ს დავუტოვეთ test-safe feature availability, მაგრამ feature representation გავაუმჯობესეთ:

- median-filled `SalesLag52`-ის ნაცვლად `NaN` უფრო სწორად გამოხატავს missing lag-ს;
- count aggregate feature-ები model-ს ეუბნება, რამდენად სანდოა historical mean/median/std;
- numeric interaction-ები tree model-ისთვის უფრო ბუნებრივი split-ებია, ვიდრე string category interaction;
- raw-input registry pipeline ამცირებს train/test preprocessing mismatch-ის რისკს.

შედარება სამ LightGBM ეტაპს შორის:

| ექსპერიმენტი | Validation WMAE | Kaggle score | შეფასება |
| --- | ---: | ---: | --- |
| ძველი unsafe lag/rolling FE | `1573.4988` | დაახლოებით `6200` | validation leakage; test-ზე unreliable |
| პირველი safe `SalesLag52` FE | `1633.3693` | დაახლოებით `3600` | Kaggle-safe, მაგრამ ჯერ კიდევ არასაკმარისად ძლიერი |
| corrected XGBoost-aligned FE | `1615.4495` | დაახლოებით `3490` | საუკეთესო reliable LightGBM result ამ ეტაპზე |

ჩემი შეფასებით, ეს ბოლო result უკვე უკეთესია, რადგან ძველი `1573.4988` score-თან შედარებით leakage არ აქვს, ხოლო წინა safe run-თან შედარებით representation უკეთესია. Kaggle score-იც გაუმჯობესდა `3600`-დან `3490`-მდე, რაც ადასტურებს, რომ XGBoost-ის მსგავს feature representation-ს LightGBM-შიც ჰქონდა რეალური სარგებელი.

## შენიშვნები და რისკები

მიმდინარე corrected feature engineering-მა validation გააუმჯობესა `1633.3693`-დან `1615.4495`-მდე. Kaggle-ზეც score გაუმჯობესდა `3600`-დან `3490`-მდე. ეს სწორ მიმართულებას აჩვენებს, მაგრამ ჯერ კიდევ ჩამორჩება XGBoost-ის დაახლოებით `2806` score-ს.

მთავარი დარჩენილი რისკები:

1. Validation split არის last 32 weeks, მაგრამ Kaggle test horizon და distribution შეიძლება უფრო რთული იყოს.
2. `n_estimators = 100` fixed არის; საუკეთესო trial-ების curves ხშირად 100 iteration-მდე ჯერ კიდევ იკლებდა, ამიტომ `n_estimators`/early stopping tuning შეიძლება დაგვეხმაროს.
3. Historical aggregates ჯერ კიდევ ძლიერ signal-ს იძლევა, ამიტომ final inference-ზე უნდა დავრწმუნდეთ, რომ registry pipeline მხოლოდ training history-ს იყენებს.
4. Full-data retrain final submission-მდე ჯერ ცალკე უნდა გადაწყდეს, რადგან validation comparison-ისთვის time split აუცილებელია.

## Kaggle submission ანალიზი

LightGBM-ის წინა safe `SalesLag52` Kaggle submission-ზე მივიღე:

```text
Kaggle score: 3600
```

ეს შედეგი ძველ LightGBM submission-ზე უკეთესი იყო, მაგრამ XGBoost-ზე უარესი:

```text
ძველი LightGBM unsafe lag/rolling setup ≈ 6200
ახალი LightGBM safe SalesLag52 setup ≈ 3600
corrected XGBoost-aligned LightGBM setup ≈ 3490
XGBoost ≈ 2806
```

ჩემი შეფასებით, LightGBM-ის მთავარი გაკვეთილი იყო validation leakage-ის კონტროლი. ძველი feature engineering validation-ზე ძალიან კარგად ჩანდა, მაგრამ Kaggle-ზე ცუდად გადავიდა, რადგან validation feature-ებში შედიოდა ისეთი lag/rolling ინფორმაცია, რომელიც test future-ში რეალურად არ გვექნება.

ახალ setup-ში ეს პრობლემა შევამცირეთ:

- ამოვიღეთ `lag_1`, `lag_4`, `lag_13`;
- ამოვიღეთ rolling mean/std feature-ები;
- დავტოვეთ მხოლოდ safe `SalesLag52`;
- inference pipeline-ში `SalesLag52` stored observed history-დან იქმნება.

ამის შემდეგ validation ძველ unsafe score-თან შედარებით გაუარესდა:

```text
Old validation WMAE = 1573.4988
Safe validation WMAE = 1633.3693
```

მაგრამ Kaggle მნიშვნელოვნად გაუმჯობესდა:

```text
Old Kaggle ≈ 6200
New Kaggle ≈ 3600
```

ეს ჩემთვის კარგი ნიშანია: validation score ნაკლებად ლამაზია, მაგრამ უფრო honest გახდა. ანუ model ახლა იმ feature-ებს იყენებს, რომლებიც test set-ზეც რეალურად ხელმისაწვდომია.

შემდეგ corrected XGBoost-aligned FE/FS run-მა validation კიდევ გააუმჯობესა:

```text
Safe validation WMAE = 1633.3693
Corrected validation WMAE = 1615.4495
```

ამ უკანასკნელისთვის Kaggle score მივიღეთ:

```text
Safe Kaggle score = 3600
Corrected Kaggle score = 3490
```

ანუ corrected feature engineering-მა Kaggle-ზეც გააუმჯობესა შედეგი. improvement არ არის ძალიან დიდი, მაგრამ მიმართულება სწორია: validation WMAE გაუმჯობესდა `17.92`-ით, ხოლო Kaggle score დაახლოებით `110` point-ით. ეს ნიშნავს, რომ `SalesLag52`-ის `NaN` handling, numeric interaction-ები, aggregate count feature-ები და raw-input registry pipeline რეალურად დაეხმარა generalization-ს.

რატომ ვერ აჯობა XGBoost-ს:

- XGBoost-ის raw-input pipeline თავიდანვე უფრო self-contained იყო.
- XGBoost-ის feature engineering უფრო მკაცრად იყო მორგებული Kaggle inference-ზე.
- LightGBM-ში feature selection-მა შეიძლება ზოგი feature ამოიღო, რომელიც test generalization-ზე სასარგებლო იქნებოდა.
- LightGBM `n_estimators = 100` fixed იყო; რამდენიმე trial-ში validation curve 100 iteration-მდე კიდევ იკლებდა, ამიტომ training ბოლომდე optimal არ ჩანს.
- LightGBM-ის validation score ჯერ კიდევ Kaggle-ზე ბევრად უკეთესია, რაც ნიშნავს, რომ validation-test mismatch ბოლომდე არ მოგვარებულა.

საბოლოო დასკვნა:

LightGBM ძალიან ძლიერი validation model იყო და corrected setup-მა Kaggle-ზეც გააუმჯობესა შედეგი `3600 -> 3490`. მიუხედავად ამისა, XGBoost-ის `2806` score ჯერ კიდევ მნიშვნელოვნად უკეთესია. ამიტომ ამ ეტაპზე LightGBM-ში მთავარი დასკვნა არის: feature engineering-ის გასწორებამ real improvement მოგვცა, მაგრამ model/pipeline ჯერ კიდევ ვერ generalize-დება ისე კარგად, როგორც XGBoost. შემდეგი გაუმჯობესება უნდა იყოს validation split-ის უფრო Kaggle-like გაკეთება, `n_estimators`/early stopping tuning და feature selection-ის გადამოწმება, რადგან შესაძლოა LightGBM-ს test generalization-ისთვის ზოგი feature ზედმეტად aggressive selection-მა დააკლო.
