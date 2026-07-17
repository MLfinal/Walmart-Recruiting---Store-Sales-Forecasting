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

## ARIMAX ექსპერიმენტის ანალიზი

ARIMA order search-ის შემდეგ ცალკე `model_experiment_ARIMAX.ipynb` notebook-ში გავტესტეთ ARIMAX, ანუ ARIMA external regressors-ით. აქ SARIMA არ გამოგვიყენებია. Seasonal order გამორთულია:

```python
seasonal_order = (0, 0, 0, 0)
```

ARIMAX-ის იდეა იყო, რომ aggregate weekly sales-ს გარდა model-ს ენახა weekly-level external signals:

- holiday share;
- average temperature;
- fuel price;
- CPI;
- unemployment;
- markdown totals;
- calendar week/month signals.

### ARIMAX-ის საუკეთესო შედეგი

ARIMAX experiment-ის საუკეთესო შედეგი იყო:

```text
best_order: (0, 0, 0)
best_allocation: last_year_share
best_use_exog: True
Validation WMAE: 2563.691454
Validation MAE: 2517.848644
Validation RMSE: 5135.390683
Improvement vs seasonal naive: -42.413569%
```

ეს შედეგი ბევრად უარესია როგორც seasonal naive-ზე, ისე ARIMA baseline-ზე და tuned ARIMA-ზე.

შედარება:

| Model | Setup | Validation WMAE | Seasonal naive-სთან შედარება |
| --- | --- | ---: | ---: |
| Seasonal naive | 52-week row-level lag | `1800.17359` | reference |
| Baseline ARIMA | `(1,1,1)` + `last_year_share` | `1856.86053` | `-3.14897%` |
| Tuned ARIMA | `(1,0,2)` + `last_year_share` | `1829.87999` | `-1.65020%` |
| ARIMAX | `(0,0,0)` + exog + `last_year_share` | `2563.69145` | `-42.41357%` |

ამ ცხრილიდან ჩანს, რომ ARIMAX ამ ფორმით არ გაუმჯობესდა. პირიქით, external regressors-მა model-ის performance მნიშვნელოვნად გააუარესა.

### რატომ გახდა ARIMAX ცუდი

ჩემი შეფასებით, მთავარი მიზეზი ისაა, რომ ჩვენ ARIMAX-ში external features aggregate weekly დონეზე შევიყვანეთ, ხოლო final metric row-level Store-Dept prediction-ს აფასებს.

ARIMAX ხედავს ასეთ data-ს:

```text
weekly total sales + weekly averaged/summed external variables
```

მაგრამ competition ითხოვს:

```text
Store + Dept + Date level Weekly_Sales
```

ამ დონეებს შორის დიდი information loss არის.

მთავარი მიზეზები:

1. **External regressors ზედმეტად aggregate იყო.**  
   მაგალითად average temperature ან total markdown across all stores ვერ ამბობს, კონკრეტულ Store-Dept-ს რა მოუვა. Walmart sales ძალიან local და department-specific არის.

2. **Markdown signals row-level demand-ს კარგად ვერ დაემთხვა.**  
   Markdown-ები store/date დონეზეა, ჩვენ კი weekly aggregate-ად ვაქციეთ. შედეგად model-ში შევიდა noisy signal, რომელიც total weekly sales-ს შეიძლება სუსტად უკავშირდებოდეს, მაგრამ Store-Dept allocation-ს ვერ აუმჯობესებს.

3. **ARIMAX-მა historical yearly pattern ჩაანაცვლა სუსტი exog signal-ით.**  
   ამ dataset-ში ყველაზე ძლიერი signal არის 52-კვირიანი seasonal behavior. ARIMAX external variables-ს ზედმეტად ეყრდნობა, მაშინ როცა validation period-ისთვის row-level yearly structure უფრო მნიშვნელოვანია.

4. **Best ARIMAX order გახდა `(0,0,0)`.**  
   ეს ძალიან მნიშვნელოვანი სიგნალია. `(0,0,0)` ნიშნავს, რომ AR/MA dynamics პრაქტიკულად არ დაეხმარა და model ძირითადად exogenous regression-like behavior-ზე გადავიდა. რადგან exog features noisy იყო, result გაუარესდა.

5. **Allocation bottleneck მაინც დარჩა.**  
   თუნდაც aggregate weekly forecast გაუმჯობესებულიყო, row-level WMAE მაინც allocation-ზეა დამოკიდებული. ARIMAX-მა allocation პრობლემა არ გადაჭრა.

### რატომ იყო `last_year_share` ისევ საუკეთესო

ARIMAX-შიც საუკეთესო allocation იყო:

```text
last_year_share
```

მაგალითად საუკეთესო order-ზე:

```text
(0,0,0) + last_year_share WMAE = 2563.69
(0,0,0) + blended_share   WMAE = 2744.69
```

ეს იმავე დასკვნას ადასტურებს, რაც pure ARIMA-ში ვნახეთ: Store-Dept distribution-ისთვის 52 კვირით ძველი sales share უფრო სანდოა, ვიდრე recent average-თან blending.

### ARIMAX vs ARIMA

ARIMAX-ის დამატებამ არ გააუმჯობესა ARIMA:

```text
Tuned ARIMA WMAE:  1829.88
Best ARIMAX WMAE: 2563.69
```

გაუარესება:

```text
2563.69 - 1829.88 = 733.81 WMAE
```

ეს ძალიან დიდი სხვაობაა. ამიტომ ამ კონკრეტული implementation-ით ARIMAX არ უნდა ჩაითვალოს improvement-ად.

### რა ვისწავლეთ ARIMAX-იდან

ARIMAX-ის შედეგი არ ნიშნავს, რომ external features ყოველთვის ცუდია. ეს ნიშნავს, რომ ამ ფორმით aggregate ARIMAX არ იყო სწორი representation.

External features უფრო კარგად მუშაობს tree-based models-ში, რადგან ისინი raw row-level feature-ებს იყენებენ:

```text
Store + Dept + Date + markdown + holiday + store metadata
```

ARIMAX-ში კი ეს ყველაფერი დაიკუმშა ერთ weekly aggregate time series-ად. ამან დაკარგა ის დეტალი, რომელიც Kaggle metric-ისთვის აუცილებელია.

ამ ეტაპზე conclusions:

- ARIMAX არ გაუმჯობესდა;
- best ARIMAX ბევრად უარესია baseline ARIMA-ზე;
- tuned pure ARIMA უკეთესია ARIMAX-ზე;
- seasonal naive მაინც საუკეთესო classical baseline რჩება;
- ARIMAX-ის ამ ვერსიას final candidate-ად არ ავირჩევდი.

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

## საბოლოო აუდიტირებული შეჯამება

ARIMA aggregate weekly total-ს პროგნოზირებს და შემდეგ Store–Dept rows-ზე ანაწილებს. Flow: total aggregation → chronological 39-week validation → `(p,d,q)` grid → `last_year_share`/`blended_share` comparison → WMAE → full-data refit → raw-input allocation pipeline. Baseline `(1,1,1)` იყო `1856.8605`; საუკეთესო tuned `(1,0,2)` + last-year share გახდა `1829.8800`. Seasonal naive `1800.1736` მაინც უკეთესი დარჩა. დადასტურებული Kaggle score არ არის, ამიტომ ARIMA champion არ არის.

## Repaired ARIMAX v2 — ახალი training და შედეგები

ძველი aggregate ARIMAX-ის validation WMAE იყო:

```text
2563.6915
```

ძველი ვერსიის მთავარი პრობლემა იყო რვა correlation-selected raw aggregate feature, განსხვავებული feature scales და raw total-sales target. ახალი experiment-ის მიზანი იყო ARIMAX-ის representation-ისა და numerical stability-ის გაუმჯობესება ისე, რომ future data leakage არ დაგვემატებინა.

### Validation setup

Validation setup უცვლელი დარჩა, რათა ძველ ARIMA/ARIMAX run-ებთან შედარება სამართლიანი ყოფილიყო:

```text
Total weekly dates: 143
Training weeks:     104
Validation weeks:    39
Validation start:   2012-02-03
Validation end:     2012-10-26
Holiday weight:     5
Allocation:         last_year_share
```

Seasonal naive reference:

```text
WMAE = 1800.1736
```

### Target engineering

ARIMAX-ს raw aggregate sales-ის ნაცვლად `log1p`-გარდაქმნილი target მივაწოდეთ:

```python
model_target = np.log1p(np.clip(total_weekly_sales, 0.0, None))
```

ფორმულა:

```text
z_t = log(1 + TotalSales_t)
```

ამის მიზანი იყო:

- holiday spikes-ის გავლენის შემცირება;
- variance-ის უფრო სტაბილურად ქცევა;
- ათობით მილიონიანი target scale-ის შეკუმშვა;
- optimizer-ის numerical stability-ის გაუმჯობესება.

Forecast-ის შემდეგ prediction საწყის sales scale-ზე დავაბრუნეთ:

```python
forecast_sales = np.expm1(forecast_log)
```

```text
ŷ_t = exp(ẑ_t) - 1
```

### Exogenous feature engineering

ძველი raw/correlation-selected feature set შევცვალეთ ექვსი compact future-known feature-ით:

```text
week_sin
week_cos
week_sin_2
week_cos_2
holiday_share
log_total_markdown
```

#### პირველი annual Fourier harmonic

```text
week_sin = sin(2π × week_of_year / 52)
week_cos = cos(2π × week_of_year / 52)
```

ეს ორი feature annual seasonal position-ს cyclic ფორმაში აღწერს. Week 52 და Week 1 ერთმანეთთან ახლოს რჩება, განსხვავებით raw `month` ან `weekofyear` feature-ისგან.

#### მეორე annual Fourier harmonic

```text
week_sin_2 = sin(4π × week_of_year / 52)
week_cos_2 = cos(4π × week_of_year / 52)
```

მეორე harmonic მოდელს აძლევს საშუალებას წელიწადში ერთზე მეტი seasonal rise/fall დაიჭიროს. ეს კვლავ linear regressors-ია, მაგრამ მათი კომბინაცია უფრო მოქნილ seasonal curve-ს ქმნის.

#### Holiday share

```python
holiday_share = features.groupby("Date")["IsHoliday"].mean()
```

ეს feature აღნიშნავს კვირის aggregate holiday signal-ს. Walmart dataset-ში holiday indicator Store-Date დონეზეა მოცემული, ხოლო ჩვენი target ერთი Walmart-wide weekly total-ია, ამიტომ იგივე weekly aggregate level გამოვიყენეთ.

#### Log total markdown

ჯერ ყველა Store-Date Markdown მნიშვნელობა კვირის მიხედვით დავაჯამეთ:

```text
total_markdown_t = Σ MarkDown1..5
```

შემდეგ:

```text
log_total_markdown_t = log(1 + total_markdown_t)
```

`log1p` საჭიროა, რადგან raw Markdown totals ძალიან skewed იყო და რამდენიმე promotion week-ს ძალიან დიდი მნიშვნელობა ჰქონდა.

### Train-only feature scaling

ყველა exogenous feature training statistics-ით დავასტანდარტეთ:

```text
x_scaled = (x - μ_train) / σ_train
```

Validation-ზე გამოყენებულია მხოლოდ training mean/std. Validation statistics scaling-ში არ შეგვიტანია, ამიტომ preprocessing leakage-safe დარჩა.

Full-data registry refit-ის დროს mean/std თავიდან გამოითვლება მთელ train.csv-ზე და pipeline-ში ინახება, რათა inference-ზე ზუსტად იგივე transformation შესრულდეს.

### რა ამოვიღეთ ძველი ARIMAX-იდან

ახალ feature set-ში აღარ გვაქვს:

```text
raw month
is_december
raw MarkDown3
raw MarkDown5
raw total_markdown
Temperature
correlation-based top-8 selection
```

ამის მიზეზებია:

- raw `month` cyclic seasonality-ს არასწორ linear ordering-ს აძლევდა;
- `is_december`, `month`, Markdown და holiday features ერთმანეთთან redundant შეიძლებოდა ყოფილიყო;
- raw Markdown ძალიან skewed იყო;
- მხოლოდ 104 training კვირაზე 8 noisy regression coefficient არასტაბილური იყო;
- Temperature aggregate mean Store/Dept-specific effect-ს კარგავდა.

### ARIMAX order search

Search space გავაფართოვეთ:

```text
p ∈ {0,1,2,3,4}
d ∈ {0,1}
q ∈ {0,1,2,3,4}
```

სულ:

```text
5 × 2 × 5 = 50 ARIMAX orders
```

Allocation strategy მხოლოდ `last_year_share` იყო, რადგან წინა experiment-ში `blended_share` ყველა order-ზე უარესი აღმოჩნდა.

Optimizer-ის მაქსიმალური iterations გაიზარდა:

```text
maxiter: 200 → 300
```

### საუკეთესო ახალი ARIMAX

საუკეთესო validation result:

```text
Order:             ARIMAX(1,0,1)
Exogenous inputs:  6 engineered features
Allocation:        last_year_share
Validation WMAE:   1824.4816
Kaggle WMAE:       3200
```

`ARIMAX(1,0,1)` ნიშნავს:

```text
p = 1 → ერთი AR lag
d = 0 → differencing არ გამოიყენება
q = 1 → ერთი previous innovation/error lag
```

Exogenous regression terms ემატება ARMA dynamics-ს:

```text
y_t = c + φ₁y_(t-1) + θ₁ε_(t-1) + βᵀX_t + ε_t
```

ამ experiment-ში equation log-total-sales scale-ზე fit-დება.

### შედარება ძველ ARIMAX-თან

```text
Old ARIMAX:       2563.6915
Repaired ARIMAX:  1824.4816
Improvement:       739.2099 WMAE
```

პროცენტულად:

```text
(2563.6915 - 1824.4816) / 2563.6915 ≈ 28.83%
```

ეს გვიჩვენებს, რომ ძველი ARIMAX-ის ცუდი შედეგი მხოლოდ external variables-ის გამოყენების პრობლემა არ ყოფილა. Target transformation, compact feature engineering და train-only scaling მნიშვნელოვანი იყო.

### შედარება pure ARIMA-სთან

```text
Best pure ARIMA:      1829.8800
Best repaired ARIMAX: 1824.4816
Difference:              5.3984
```

Validation-ზე repaired ARIMAX pure ARIMA-ზე დაახლოებით `0.30%`-ით უკეთესი გახდა. გაუმჯობესება რეალურია, მაგრამ ძალიან მცირეა.

### შედარება Seasonal Naive-სთან

```text
Seasonal naive:       1800.1736
Repaired ARIMAX:      1824.4816
Gap:                    24.3080
```

Repaired ARIMAX seasonal naive-ზე კვლავ დაახლოებით `1.35%`-ით უარესია. ანუ yearly Store–Dept lag-52 signal ჯერ კიდევ უფრო ძლიერია, ვიდრე aggregate ARIMAX forecast + allocation pipeline.

### საუკეთესო validation orders

```text
(1,0,1) → 1824.4816
(0,1,1) → 1830.1113
(1,1,2) → 1832.4395
(1,0,2) → 1835.2707
(0,1,2) → 1854.0596
(3,1,3) → 1863.6190
(2,1,3) → 1864.3769
(2,1,4) → 1867.3321
(1,1,4) → 1868.0885
(2,1,1) → 1870.3414
```

### რატომ აფეთქდა ზოგი order

რამდენიმე order-მა ძალიან მაღალი WMAE მიიღო:

```text
(0,0,0) → 15952.32
(0,0,1) → 15952.29
(0,0,2) → 15951.82
(0,0,3) → 15915.47
(0,0,4) → 15929.20
```

ამ შემთხვევებში dynamic structure საკმარისი არ იყო log-total-sales level-ის გრძელ 39-week horizon-ზე დასასტაბილურებლად. მხოლოდ exogenous regression და მცირე MA component aggregate level-ს ცუდად extrapolate-ებდა.

სხვა unstable configurations-ში მთავარი რისკები იყო:

- მხოლოდ 104 training observation;
- მაღალი `p` და `q` order;
- ბევრი AR/MA coefficient მცირე sample-ზე;
- 39-step forecast sensitivity;
- near-cancellation ან poorly identified AR/MA terms;
- validation period-ის exogenous distribution shift.

50 order-იდან მხოლოდ ნაწილია `2000` WMAE-ზე უკეთესი. ეს ნიშნავს, რომ ARIMAX order selection ძალიან მგრძნობიარეა და training likelihood მარტო სანდო selection criterion არ არის.

### Validation და Kaggle generalization gap

```text
Validation WMAE: 1824.48
Kaggle WMAE:     3200
Gap:             1375.52
```

Kaggle error validation-ზე დაახლოებით `75.4%`-ით მაღალია. ამიტომ საუკეთესო validation ARIMAX-ს champion-ად ვერ ჩავთვლით.

Gap-ის შესაძლო მიზეზებია:

1. ერთი 39-week validation split order/feature selection-ს კონკრეტულ პერიოდზე არგებს.
2. Kaggle test period-ის seasonal და promotion distribution განსხვავებულია.
3. Aggregate forecast შეიძლება reasonable იყოს, მაგრამ last-year row allocation test-ზე უფრო მეტად ცდებოდეს.
4. Store–Dept-specific external-feature effects aggregate regressors-ში დაკარგულია.
5. ARIMAX validation ranking მცირე განსხვავებებითაა შექმნილი; best order-ის `5.4`-იანი gain pure ARIMA-სთან შედარებით robust improvement არ არის.

### საბოლოო pipeline flow

```text
train.csv Weekly_Sales
        ↓ group by Date / sum
Walmart weekly total series
        ↓ log1p
log aggregate target

features.csv
        ↓ weekly aggregation
holiday share + total markdown
        ↓ Fourier features + log1p markdown
6 exogenous variables
        ↓ train-only mean/std scaling
standardized exogenous matrix

log target + exogenous matrix
        ↓ ARIMAX(1,0,1)
39 weekly aggregate forecasts
        ↓ expm1
weekly total-sales forecasts
        ↓ last-year Store–Dept shares
row-level Weekly_Sales predictions
```

### Repaired ARIMAX figures

Figures ინახება:

```text
models/classical_statistical_time_series/arima/figures/arimax_repaired/
```

დამატებულია:

1. result progression;
2. top-15 order leaderboard;
3. `(p,d,q)` heatmaps `d=0` და `d=1`-ისთვის;
4. order stability distribution;
5. differencing sensitivity;
6. feature-engineering flow;
7. selected-feature table;
8. target/scaling transformations;
9. repaired ARIMAX architecture;
10. best-result table;
11. validation-vs-Kaggle gap;
12. unstable-order explanation.

### Repaired ARIMAX-ის საბოლოო შეფასება

Repaired ARIMAX ძველ ARIMAX-ზე მნიშვნელოვნად უკეთესია და validation-ზე tuned pure ARIMA-ს მცირედით აჯობა. მიუხედავად ამისა:

- seasonal naive კვლავ უკეთესია;
- Kaggle score `3200` მნიშვნელოვნად უარესია validation score-ზე;
- aggregate representation Store–Dept-level interactions-ს კვლავ კარგავს;
- best validation gain pure ARIMA-სთან შედარებით მხოლოდ `5.4 WMAE`-ია;
- საბოლოო საუკეთესო Kaggle კანდიდატად tree-based models რჩება.

ამიტომ repaired ARIMAX უნდა შეფასდეს როგორც წარმატებული architecture repair და ძლიერი statistical experiment, მაგრამ არა როგორც final champion.
