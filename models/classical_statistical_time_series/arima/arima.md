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

## ARIMA order search ექსპერიმენტი

baseline-ის შემდეგ `model_experiment_ARIMA.ipynb`-ში დავამატეთ controlled brute-force search მხოლოდ ARIMA order-ზე და allocation strategy-ზე. აქ არ არის ARIMAX და არ არის SARIMA. მიზანი იყო გვენახა, გაუმჯობესდებოდა თუ არა pure aggregate ARIMA უკეთესი `(p, d, q)` order-ით.

გამოყენებული grid:

```python
p = [0, 1, 2]
d = [0, 1]
q = [0, 1, 2]
```

სულ გამოვიდა:

```text
18 ARIMA order
```

ყოველი order შემოწმდა ორი allocation strategy-ით:

```text
last_year_share
blended_share
```

ანუ ჯამში გაკეთდა 36 validation comparison.

### საუკეთესო შედეგი

საუკეთესო შედეგი მივიღეთ:

```text
order: (1, 0, 2)
allocation: last_year_share
Validation WMAE: 1829.879987
Validation MAE: 1840.653606
Validation RMSE: 3937.045372
Improvement vs seasonal naive: -1.650196%
```

ეს უკეთესია baseline ARIMA-ზე:

```text
baseline ARIMA(1,1,1): 1856.86053
best searched ARIMA(1,0,2): 1829.87999
```

გაუმჯობესება:

```text
1856.86 -> 1829.88
```

ანუ order search-მა ARIMA baseline დაახლოებით `26.98` WMAE-ით გააუმჯობესა.

მაგრამ seasonal naive-ს მაინც ვერ აჯობა:

```text
seasonal naive: 1800.17359
best ARIMA:     1829.87999
```

best ARIMA ჯერ კიდევ დაახლოებით `29.71` WMAE-ით უარესია seasonal naive-ზე.

### Top results

საუკეთესო რამდენიმე trial:

| Rank | Trial | Order | Allocation | Validation WMAE | Improvement vs seasonal naive |
| ---: | ---: | --- | --- | ---: | ---: |
| 1 | 8 | `(1, 0, 2)` | `last_year_share` | `1829.879987` | `-1.650196%` |
| 2 | 14 | `(2, 0, 2)` | `last_year_share` | `1834.686858` | `-1.917219%` |
| 3 | 1 | `(0, 0, 1)` | `last_year_share` | `1835.703255` | `-1.973680%` |
| 4 | 0 | `(0, 0, 0)` | `last_year_share` | `1836.591341` | `-2.023013%` |
| 5 | 5 | `(0, 1, 2)` | `last_year_share` | `1846.058549` | `-2.548918%` |

### რატომ გახდა `(1, 0, 2)` საუკეთესო

ჩემი აზრით, `(1, 0, 2)` საუკეთესო იმიტომ გამოვიდა, რომ ამ aggregate weekly sales series-ზე differencing (`d=1`) ზედმეტად აგრესიული აღმოჩნდა.

საუკეთესო result-ს აქვს:

```text
d = 0
```

ეს ნიშნავს, რომ model-ს უკეთ აწყობდა level-ის შენარჩუნება და არა weekly total sales-ის differenced series-ზე მუშაობა. Walmart total weekly sales-ში yearly/holiday pattern და level information ძალიან მნიშვნელოვანია. როცა `d=1` ვიყენებთ, trend/level ნაწილი იშლება და model შეიძლება კარგავს იმ signal-ს, რომელიც row-level allocation-საც სჭირდება.

ეს ჩანს შედეგებშიც:

```text
(1,1,1) last_year_share WMAE = 1856.86
(1,0,2) last_year_share WMAE = 1829.88
```

ანუ იგივე aggregate approach-ში `d=0` უკეთესი აღმოჩნდა.

`p=1` ეხმარება ბოლო weekly total sales-ის autoregressive signal-ის დაჭერაში, ხოლო `q=2` ეხმარება short-term error/noise correction-ს. მაგრამ ეს მაინც aggregate-level model-ია, ამიტომ Store-Dept სპეციფიკურ seasonal pattern-ს სრულად ვერ სწავლობს.

### რატომ იყო `last_year_share` ყოველთვის უკეთესი `blended_share`-ზე

ყველა საუკეთესო trial იყენებს:

```text
allocation = last_year_share
```

`blended_share` ყველა შემთხვევაში აშკარად უარესია. მაგალითად:

```text
(1,0,2) last_year_share WMAE = 1829.88
(1,0,2) blended_share   WMAE = 1995.64
```

ეს ნიშნავს, რომ Walmart dataset-ში Store-Dept distribution-ისთვის ერთი წლის წინანდელი share უფრო ძლიერი signal-ია, ვიდრე recent average-ის დამატება.

ჩემი ინტერპრეტაცია:

- weekly sales-ს აქვს ძლიერი yearly seasonality;
- ბევრი department seasonal demand-ს იმეორებს წლიდან წლამდე;
- recent average შეიძლება seasonal structure-ს აბუნდოვანებდეს;
- validation period-ში last-year distribution უკეთ ემთხვევა future distribution-ს.

ამიტომ `blended_share`-მა ვერ გააუმჯობესა result. პირიქით, recent share-ის 30%-მა შეიტანა noise და WMAE გაზარდა.

### რატომ მაინც ვერ აჯობა seasonal naive-ს

მიუხედავად იმისა, რომ order search-მა baseline ARIMA გააუმჯობესა, საუკეთესო ARIMA მაინც უარესია seasonal naive-ზე:

```text
seasonal naive WMAE = 1800.17
best ARIMA WMAE     = 1829.88
```

მთავარი მიზეზი არის ის, რომ seasonal naive row-level forecast-ს პირდაპირ იღებს იგივე Store-Dept-ის 52 კვირით ძველი sales-იდან. ეს ძალიან ძლიერი baseline-ია ამ competition-ში.

ARIMA კი აკეთებს ორეტაპიან approximation-ს:

```text
total weekly sales forecast
        ↓
row-level allocation
```

ამ პროცესში Store-Dept-level information იკარგება. Aggregate forecast შეიძლება მისაღები იყოს, მაგრამ თუ allocation ცოტათი მაინც ცდება, Kaggle/WMAE row-level metric ამას მკაცრად სჯის.

ამიტომ best ARIMA result-ის შეფასება ასეთია:

- baseline ARIMA-ზე უკეთესია;
- `blended_share`-ზე აშკარად უკეთესია;
- seasonal naive-ზე მაინც უარესია;
- final model candidate არ არის, მაგრამ useful diagnostic experiment არის.

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
