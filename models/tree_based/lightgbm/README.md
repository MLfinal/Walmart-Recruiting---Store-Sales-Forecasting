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

ეს aggregate-ები უნდა დაითვალოს მხოლოდ training data-ზე. თუ validation target values მოხვდება aggregate-ში, მივიღებთ target leakage-ს.

## 7. LagRollingFeatureTransformer

ეს transformer ქმნის:

- `lag_1`
- `lag_4`
- `lag_13`
- `lag_52`
- rolling mean feature-ებს
- rolling standard deviation feature-ებს

რატომ დაემატა:

- Weekly sales ძლიერად autocorrelated არის.
- წინა კვირის, წინა თვის, წინა კვარტლის და წინა წლის იგივე კვირის sales ძლიერი predictor-ებია.
- Rolling mean noisy weekly sales-ს ასწორებს.
- Rolling standard deviation volatility-ს აღწერს.

მნიშვნელოვანი რისკი:

Lag და rolling feature-ებმა შეიძლება leakage შექმნას, თუ validation rows წინა validation-period `Weekly_Sales` მნიშვნელობებს იყენებს. ეს validation score-ს გააუმჯობესებს, მაგრამ Kaggle test inference-ზე future `Weekly_Sales` უცნობია, ამიტომ ეს საკითხი საბოლოო submission-მდე ფრთხილად უნდა მოგვარდეს.

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

## საუკეთესო Trial

მოწოდებული Optuna logs-ის მიხედვით საუკეთესო run იყო:

```text
Trial: 46
Validation Weighted MAE: 1573.4988
Validation MAE: 1543.4832
```

საუკეთესო hyperparameter-ები:

```python
{
    "learning_rate": 0.08117866851143801,
    "num_leaves": 196,
    "max_depth": 19,
    "min_child_samples": 98,
    "subsample": 0.9817092354210323,
    "colsample_bytree": 0.8206871721053576,
    "reg_alpha": 0.00031014058676548666,
    "reg_lambda": 0.01501347092737337,
}
```

ამ trial-მა საუკეთესო შედეგი აჩვენა, რადგან ჰქონდა:

- შედარებით მაღალი learning rate;
- დიდი `num_leaves`;
- ღრმა trees;
- მაღალი row sampling;
- ზომიერი feature sampling;
- ძალიან მსუბუქი regularization.

ეს combination მოდელს აძლევს საშუალებას დაიჭიროს რთული store-department-seasonality interactions, მაგრამ sampling მაინც ამცირებს overfitting-ის რისკს.

## Trial-ების შედარება

Representative trial results:

| Trial | Weighted MAE | ძირითადი pattern |
| --- | ---: | --- |
| 0 | 2861.46 | დაბალი learning rate და ძალიან დიდი tree capacity; 100 round-ში ბოლომდე ვერ ისწავლა |
| 1 | 1988.89 | უკეთესი learning rate და shallow depth; დიდი improvement |
| 4 | 5673.74 | ძალიან დაბალი learning rate; fixed 100 trees-ისთვის ძალიან ნელია |
| 10 | 1713.95 | მაღალი learning rate და პატარა `num_leaves`; ძლიერი improvement |
| 16 | 1636.31 | learning rate, leaves, depth და sampling-ის კარგი balance |
| 33 | 1603.35 | ძლიერი high-learning-rate configuration |
| 39 | 1573.57 | trial 46-მდე საუკეთესო; მაღალი learning rate და large trees |
| 46 | 1573.50 | საუკეთესო overall |
| 49 | 3695.29 | დაბალი learning rate; underfit |

## Hyperparameter-ების ქცევა

### Learning Rate

ყველაზე მნიშვნელოვანი pattern იყო learning rate.

დაბალი learning rate-ები, დაახლოებით `0.01`-დან `0.02`-მდე, ცუდად მუშაობდა, რადგან `n_estimators = 100` იყო fixed.

მაგალითები:

- Trial 4: `learning_rate = 0.0116`, WMAE `5673.74`
- Trial 9: `learning_rate = 0.0101`, WMAE `6329.98`
- Trial 49: `learning_rate = 0.0184`, WMAE `3695.29`

ეს models ჯერ კიდევ სწავლობდნენ, მაგრამ 100 boosting round საკმარისი არ იყო.

მაღალი learning rate-ები, დაახლოებით `0.06`-დან `0.09`-მდე, ბევრად უკეთესი იყო:

- Trial 16: `0.0583`, WMAE `1636.31`
- Trial 33: `0.0714`, WMAE `1603.35`
- Trial 39: `0.0881`, WMAE `1573.57`
- Trial 46: `0.0812`, WMAE `1573.50`

დასკვნა:

რადგან `n_estimators = 100`, higher learning rate აუცილებელი აღმოჩნდა. თუ learning rate დაბალი იქნება, მაშინ `n_estimators` უნდა გაიზარდოს.

### Number of Leaves

საუკეთესო trial-ები ძირითადად იყენებდნენ ბევრ leaves-ს:

- Trial 39: `num_leaves = 210`
- Trial 46: `num_leaves = 196`
- Trial 48: `num_leaves = 224`

ეს ნიშნავს, რომ მოდელს სჭირდება complex tree structure, რადგან Walmart sales დამოკიდებულია store, department, week, holiday, markdown და historical sales interaction-ებზე.

მაგრამ მარტო დიდი tree capacity საკმარისი არ არის. Trial 0-ს ჰქონდა `num_leaves = 245`, მაგრამ learning rate დაბალი იყო, ამიტომ შედეგი უარესი გამოვიდა.

### Max Depth

საუკეთესო trial-ებში tree depth მაღალი იყო:

- Trial 39: `max_depth = 20`
- Trial 46: `max_depth = 19`
- Trial 48: `max_depth = 18`

ეს აჩვენებს, რომ deeper trees ეხმარებოდა nonlinear interaction-ების დაჭერაში.

### Min Child Samples

საუკეთესო trial-ს ჰქონდა:

```python
min_child_samples = 98
```

ეს საკმაოდ მაღალი მნიშვნელობაა. ანუ deep tree და ბევრი leaves მიუხედავად, model პატარა leaf-ებს ერიდებოდა. ეს regularization-ს ქმნის და overfitting-ს ამცირებს.

### Subsample

საუკეთესო trial-ებში row sampling მაღალი იყო:

- Trial 39: `subsample = 0.9727`
- Trial 46: `subsample = 0.9817`
- Trial 48: `subsample = 0.9849`

ეს აჩვენებს, რომ model-ს dataset-ის დიდი ნაწილი სჭირდებოდა stable historical patterns-ის დასაჭერად.

### Column Sampling

საუკეთესო trial-ს ჰქონდა:

```python
colsample_bytree = 0.8207
```

სხვა ძლიერი trial-ები იყენებდნენ დაახლოებით `0.80`-დან `0.95`-მდე მნიშვნელობებს.

ეს ნიშნავს, რომ feature-ების დიდი ნაწილი სასარგებლო იყო, მაგრამ ყველა feature-ის გამოყენება თითო tree-ში აუცილებელი არ იყო. Column sampling გარკვეულ regularization-ს ქმნის.

### Regularization

საუკეთესო trial-ში regularization ძალიან სუსტი იყო:

```python
reg_alpha = 0.00031
reg_lambda = 0.01501
```

ეს ნიშნავს, რომ selected features და `min_child_samples` უკვე ქმნიდა საკმარის regularization-ს. ძლიერი L1/L2 penalty საუკეთესო შედეგისთვის საჭირო არ აღმოჩნდა.

## Training Curves

W&B curves აჩვენებს, რომ training L1 და validation L1 უმეტეს trial-ში იკლებდა 100 iteration-ის განმავლობაში.

ეს ნიშნავს:

- model სტაბილურად სწავლობდა;
- validation error ბევრ run-ში ბოლომდე იკლებდა;
- `n_estimators = 100` ზოგი configuration-ისთვის შეიძლება ცოტაა;
- early stopping არ იყო გამოყენებული, ამიტომ run summary-ში ბოლო iteration არის `99`.

დაბალი learning rate-ის მქონე trial-ებში curves 100 iteration-ის შემდეგაც შორს იყო convergence-დან. მაღალი learning rate-ის მქონე trial-ებში curves უფრო სწრაფად flatten-დებოდა და უკეთეს score-ს აღწევდა.

## მიმდინარე საუკეთესო შედეგი

საუკეთესო დალოგილი validation result:

```text
Validation Weighted MAE: 1573.4988
Best trial: 46
```

საუკეთესო run:

```text
lightgbm-optuna-trial-46
```

## შენიშვნები და რისკები

Notebook კარგია experiment tracking-ისთვის, მაგრამ ორი მნიშვნელოვანი რისკი ჯერ რჩება:

1. Lag/rolling validation leakage შეიძლება არსებობდეს, რადგან validation rows შეიცავს რეალურ `Weekly_Sales` მნიშვნელობებს.
2. საუკეთესო model ჯერ არ არის შენახული reusable pipeline-ად და არ არის დარეგისტრირებული inference-ისთვის.

Final Kaggle submission-მდე pipeline უნდა გახდეს raw test data-ზე უსაფრთხოდ გასაშვები, ხოლო საუკეთესო model უნდა შეინახოს artifact/model registry-ში.

