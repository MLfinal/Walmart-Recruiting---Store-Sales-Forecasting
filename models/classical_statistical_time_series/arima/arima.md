# ARIMA baseline ექსპერიმენტის აღწერა

ეს დოკუმენტი აღწერს `baseline_arima.ipynb` notebook-ში გაკეთებულ ARIMA baseline-ს: რა მიდგომა ავირჩიე, რა ნაბიჯები შესრულდა, რა metric-ები დალოგდა W&B-ში და რატომ მივიღეთ ასეთი შედეგი.

## მიზანი

ARIMA baseline-ის მიზანი იყო სწრაფი classical/statistical reference model-ის შექმნა. აქ არ ვცდილობდით რთულ feature engineering-ს ან ბევრი მოდელის tuning-ს. მთავარი კითხვა იყო:

```text
შეუძლია თუ არა ძალიან მარტივ ARIMA baseline-ს Walmart weekly sales-ის reasonable forecast?
```

ამიტომ notebook deliberately მარტივია და არ აკეთებს დიდ training-ს.

## Notebook-ის სტრუქტურა

`baseline_arima.ipynb` რამდენიმე ძირითად ეტაპადაა დაყოფილი:

1. ბიბლიოთეკების დაყენება და import.
2. configuration-ის განსაზღვრა.
3. `train.csv`, `test.csv`, `features.csv`, `stores.csv` ფაილების წაკითხვა.
4. chronological train/validation split.
5. weekly total sales-ზე aggregate ARIMA model-ის training.
6. weekly forecast-ის Store-Dept დონეზე განაწილება historical share-ებით.
7. validation metric-ების დათვლა.
8. W&B-ში metric-ების დალოგვა.
9. optional Kaggle submission artifact-ის შექმნა.

## რატომ გამოვიყენეთ aggregate ARIMA

Walmart dataset-ში forecast საჭიროა `Store` + `Dept` + `Date` დონეზე. სრული per-series ARIMA approach ნიშნავს, რომ ათასობით ცალკე time series-ზე უნდა გავწვრთნათ ARIMA model.

ეს baseline-ისთვის ზედმეტად მძიმე იქნებოდა:

- ბევრი Store-Dept series მოკლეა ან noisy;
- ყველა series-ზე ARIMA fitting ნელია;
- convergence warning-ების რისკი მაღალია;
- baseline-ის მიზანი სწრაფი reference result იყო და არა final champion model.

ამიტომ ავირჩიე უფრო მსუბუქი მიდგომა:

```text
ყველა გაყიდვა ერთ weekly total time series-ად ვაგრეგირებთ
        ↓
ვწვრთნით ერთ ARIMA(1, 1, 1) model-ს
        ↓
ვპროგნოზირებთ future weekly total sales-ს
        ↓
weekly total forecast-ს ვანაწილებთ Store-Dept rows-ზე historical share-ებით
```

## გამოყენებული ARIMA პარამეტრები

გამოყენებული order იყო:

```text
ARIMA order: (1, 1, 1)
```

ეს ნიშნავს:

- `p = 1`: ერთი autoregressive lag;
- `d = 1`: პირველი differencing trend-ის დასასტაბილურებლად;
- `q = 1`: ერთი moving-average term.

ეს არ არის tune-ებული საუკეთესო order. ეს არის მარტივი baseline order, რომელიც სწრაფად ეშვება და გვაძლევს პირველ classical/statistical benchmark-ს.

## Validation split

Validation გაკეთდა chronological split-ით:

```text
validation_start = 2012-02-03
validation_end   = 2012-10-26
```

ანუ ბოლო 39 კვირა წავიდა validation-ში.

რატომ chronological split:

- forecasting task-ში future data train-ში არ უნდა მოხვდეს;
- random split time-series ამოცანაში leakage-ს ქმნის;
- Kaggle test-იც future period-ია, ამიტომ validation-იც future holdout უნდა იყოს.

## Forecast allocation logic

ARIMA model პროგნოზირებს მხოლოდ total weekly sales-ს. მაგრამ competition ითხოვს prediction-ს თითოეული `Store` + `Dept` + `Date` row-სთვის.

ამისთვის გამოვიყენეთ allocation step:

1. თითო validation/test row-სთვის ვეძებთ 52 კვირით ძველ გაყიდვას იგივე `Store` + `Dept`-ზე.
2. თუ last-year sales არსებობს, ის ხდება allocation base.
3. თუ არ არსებობს, fallback-ად ვიყენებთ ამ Store-Dept series-ის საშუალო historical sales-ს.
4. თითო date-ზე ყველა row-ის allocation base ჯამდება.
5. row-level share ითვლება ასე:

```text
row_share = row_allocation_base / total_allocation_base_for_that_week
```

6. საბოლოო prediction:

```text
row_prediction = weekly_arima_forecast * row_share
```

ეს approach ძალიან მარტივია: ARIMA სწავლობს მხოლოდ overall weekly trend-ს, ხოლო Store-Dept structure historical shares-იდან მოდის.

## Metric Evaluation

მთავარი metric იყო **Weighted Mean Absolute Error (WMAE)**, რადგან Walmart Kaggle competition სწორედ ამ metric-ს იყენებს.

WMAE formula:

```text
WMAE = sum(weight * abs(y_true - y_pred)) / sum(weight)
```

სადაც:

```text
holiday week weight = 5
normal week weight  = 1
```

დამატებით დალოგდა:

| Metric | მნიშვნელობა |
| --- | --- |
| `validation/wmae` | მთავარი validation score |
| `validation/mae` | ჩვეულებრივი absolute error |
| `validation/rmse` | დიდი შეცდომების diagnostic |
| `baseline/seasonal_naive_wmae` | 52-კვირიანი seasonal naive baseline |
| `improvement_vs_seasonal_naive_pct` | ARIMA-ს გაუმჯობესება ან გაუარესება seasonal naive-სთან შედარებით |

## W&B run

Run დალოგდა W&B-ში:

```text
Run name: ARIMA_Baseline_Aggregate
Run URL: https://wandb.ai/kende23-n-a/Walmart-Recruiting---Store-Sales-Forecasting/runs/balfjn48
```

Run summary:

```text
arima_order: (1, 1, 1)
baseline/seasonal_naive_wmae: 1800.17359
validation/wmae: 1856.86053
validation/mae: 1843.28763
validation/rmse: 3919.59437
improvement_vs_seasonal_naive_pct: -3.14897
validation_start: 2012-02-03
validation_end: 2012-10-26
```

## შედეგის ანალიზი

ARIMA baseline-ის validation result იყო:

```text
Validation WMAE: 1856.86053
```

Seasonal naive baseline იყო:

```text
Seasonal naive WMAE: 1800.17359
```

ARIMA-მ seasonal naive-ზე უარესი შედეგი აჩვენა:

```text
Improvement vs seasonal naive: -3.14897%
```

ანუ ამ baseline setup-ში ARIMA დაახლოებით 3.15%-ით უარესია 52-კვირიან seasonal naive forecast-ზე.

## რატომ გამოვიდა ARIMA seasonal naive-ზე უარესი

ჩემი შეფასებით, მთავარი მიზეზი ისაა, რომ Walmart weekly sales ძალიან ძლიერი yearly seasonality-ით მუშაობს. ბევრი Store-Dept series-ისთვის ყველაზე ძლიერი signal არის:

```text
ამავე კვირის გაყიდვები ერთი წლის წინ
```

Seasonal naive პირდაპირ ამ signal-ს იყენებს. ARIMA baseline კი weekly total sales-ზეა trained და შემდეგ row-level prediction-ს historical share-ებით ანაწილებს. ამ დროს რამდენიმე მნიშვნელოვანი ინფორმაცია იკარგება:

1. **Store-Dept-specific seasonality იკარგება.**  
   Aggregate ARIMA ხედავს მთლიან Walmart sales-ს, მაგრამ ვერ ხედავს, რომ კონკრეტულ department-ს კონკრეტულ store-ში შეიძლება განსხვავებული seasonal pattern ჰქონდეს.

2. **Holiday behavior სუსტადაა დაჭერილი.**  
   WMAE holiday weeks-ს 5-ჯერ მეტ წონას აძლევს. ARIMA(1,1,1) არ იყენებს explicit holiday feature-ს, ამიტომ Thanksgiving/Christmas-like spikes-ს კარგად ვერ სწავლობს.

3. **External covariates არ გამოიყენება.**  
   `features.csv` და `stores.csv` იკითხება, მაგრამ baseline ARIMA მათ model-ში არ იყენებს. არ არის markdown, fuel price, CPI, temperature, store type, store size.

4. **Allocation step ძალიან მარტივია.**  
   Total weekly forecast შეიძლება reasonable იყოს, მაგრამ row-level distribution თუ არასწორია, final WMAE მაინც იზრდება. Kaggle metric row-level prediction-ს აფასებს და არა total weekly sales-ს.

5. **ARIMA order არ არის tune-ებული.**  
   `(1,1,1)` ავიღეთ როგორც სწრაფი baseline. შესაძლოა სხვა order უკეთესი იყოს, მაგრამ baseline მიზანი დიდი search არ ყოფილა.

## რა ვისწავლეთ

ეს ექსპერიმენტი სასარგებლოა, რადგან აჩვენებს, რომ მხოლოდ aggregate statistical forecasting საკმარისი არ არის ამ competition-ისთვის.

მთავარი დასკვნები:

- ARIMA სწრაფი და მარტივია, მაგრამ row-level Walmart task-ისთვის ზედმეტად coarse approach გამოდის.
- Strong yearly seasonality იმდენად მნიშვნელოვანია, რომ seasonal naive უკეთესი baseline აღმოჩნდა.
- Walmart competition-ში Store-Dept-level structure, holiday effects და tabular covariates ძალიან მნიშვნელოვანია.
- Classical model-ის გასაუმჯობესებლად საჭიროა ან per-series SARIMA/SARIMAX, ან hybrid approach, სადაც yearly lag და covariates ცალკე შედის.

## შედარება სხვა baseline-ებთან

ამ ეტაპზე ARIMA baseline-ის შედეგი:

```text
ARIMA validation WMAE: 1856.86
Seasonal naive WMAE: 1800.17
```

ეს ნიშნავს, რომ ARIMA ჯერ არ არის final candidate. ის უფრო diagnostic baseline-ია, რომელიც გვიჩვენებს, რომ simple aggregate time-series model ამ dataset-ზე საკმარისი არ არის.

Tree-based models უკეთ მუშაობენ, რადგან მათ შეუძლიათ ერთდროულად გამოიყენონ:

- store/dept identity;
- calendar features;
- holiday indicators;
- markdown/promotion data;
- historical lag signals;
- interactions between tabular variables.

## შემდეგი ნაბიჯები

თუ ARIMA/SARIMA მიმართულებას გავაგრძელებთ, შემდეგი გაუმჯობესებები იქნება:

1. per-series seasonal naive + ARIMA residual model;
2. SARIMA yearly seasonality-ით;
3. SARIMAX external regressors-ით, მაგალითად markdown და holiday flags;
4. Store-Dept clustering და თითო cluster-ზე separate ARIMA/SARIMA;
5. direct comparison Kaggle submission-ზე.

ამ ეტაპზე კი `baseline_arima.ipynb` რჩება სწრაფ baseline-ად და არა final model candidate-ად.
