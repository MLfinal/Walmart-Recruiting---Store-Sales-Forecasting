# SARIMA baseline ექსპერიმენტის აღწერა

ეს დოკუმენტი აღწერს `baseline_sarima.ipynb` notebook-ში გაკეთებულ SARIMA baseline-ს: რა მიდგომა ავირჩიე, რა ნაბიჯები შესრულდა, რა metric-ები დალოგდა W&B-ში და რატომ მივიღეთ ასეთი შედეგი.

## მიზანი

SARIMA baseline-ის მიზანი იყო სწრაფი classical/statistical reference model-ის შექმნა. აქ არ ვცდილობდით რთულ feature engineering-ს ან ბევრი მოდელის tuning-ს. მთავარი კითხვა იყო:

```text
შეუძლია თუ არა ძალიან მარტივ SARIMA baseline-ს Walmart weekly sales-ის reasonable forecast?
```

ამიტომ notebook deliberately მარტივია და არ აკეთებს დიდ training-ს.

## Notebook-ის სტრუქტურა

`baseline_sarima.ipynb` რამდენიმე ძირითად ეტაპადაა დაყოფილი:

1. ბიბლიოთეკების დაყენება და import.
2. configuration-ის განსაზღვრა.
3. `train.csv`, `test.csv`, `features.csv`, `stores.csv` ფაილების წაკითხვა.
4. chronological train/validation split.
5. weekly total sales-ზე aggregate SARIMA model-ის training.
6. weekly forecast-ის Store-Dept დონეზე განაწილება historical share-ებით.
7. validation metric-ების დათვლა.
8. W&B-ში metric-ების დალოგვა.
9. optional Kaggle submission artifact-ის შექმნა.

## რატომ გამოვიყენეთ aggregate SARIMA

Walmart dataset-ში forecast საჭიროა `Store` + `Dept` + `Date` დონეზე. სრული per-series SARIMA approach ნიშნავს, რომ ათასობით ცალკე time series-ზე უნდა გავწვრთნათ SARIMA model.

ეს baseline-ისთვის ზედმეტად მძიმე იქნებოდა:

- ბევრი Store-Dept series მოკლეა ან noisy;
- ყველა series-ზე SARIMA fitting ნელია;
- convergence warning-ების რისკი მაღალია;
- baseline-ის მიზანი სწრაფი reference result იყო და არა final champion model.

ამიტომ ავირჩიე უფრო მსუბუქი მიდგომა:

```text
ყველა გაყიდვა ერთ weekly total time series-ად ვაგრეგირებთ
        ↓
ვწვრთნით ერთ SARIMA(1, 1, 1) model-ს
        ↓
ვპროგნოზირებთ future weekly total sales-ს
        ↓
weekly total forecast-ს ვანაწილებთ Store-Dept rows-ზე historical share-ებით
```

## გამოყენებული SARIMA პარამეტრები

გამოყენებული order იყო:

```text
SARIMA order: (1, 1, 1)
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

SARIMA model პროგნოზირებს მხოლოდ total weekly sales-ს. მაგრამ competition ითხოვს prediction-ს თითოეული `Store` + `Dept` + `Date` row-სთვის.

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
row_prediction = weekly_sarima_forecast * row_share
```

ეს approach ძალიან მარტივია: SARIMA სწავლობს მხოლოდ overall weekly trend-ს, ხოლო Store-Dept structure historical shares-იდან მოდის.

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
| `improvement_vs_seasonal_naive_pct` | SARIMA-ს გაუმჯობესება ან გაუარესება seasonal naive-სთან შედარებით |

## W&B run

Run დალოგდა W&B-ში:

```text
Run name: SARIMA_Baseline_Aggregate
Run URL: https://wandb.ai/kende23-n-a/Walmart-Recruiting---Store-Sales-Forecasting/runs/3vktfzge
```

Run summary:

```text
sarima_order: (1, 1, 1)
baseline/seasonal_naive_wmae: 1800.17359
validation/wmae: 1856.86053
validation/mae: 1843.28763
validation/rmse: 3919.59437
improvement_vs_seasonal_naive_pct: -3.14897
validation_start: 2012-02-03
validation_end: 2012-10-26
```

## შედეგის ანალიზი

SARIMA baseline-ის validation result იყო:

```text
Validation WMAE: 1856.86053
```

Seasonal naive baseline იყო:

```text
Seasonal naive WMAE: 1800.17359
```

SARIMA-მ seasonal naive-ზე უარესი შედეგი აჩვენა:

```text
Improvement vs seasonal naive: -3.14897%
```

ანუ ამ baseline setup-ში SARIMA დაახლოებით 3.15%-ით უარესია 52-კვირიან seasonal naive forecast-ზე.

მნიშვნელოვანი დეტალი: ეს პირველი SARIMA baseline ჯერ რეალურად **seasonal component-ს არ იყენებს**. Notebook-ში გამოყენებულია მხოლოდ non-seasonal order:

```text
order = (1, 1, 1)
seasonal_order = disabled / not tuned yet
```

ამიტომ შედეგიც ზუსტად ემთხვევა ARIMA baseline-ის result-ს. ეს მოსალოდნელი იყო, რადგან ამ ეტაპზე მოდელი იგივე aggregate weekly sales-ს სწავლობს და იგივე `last_year_share` allocation logic-ს იყენებს. განსხვავება მხოლოდ naming/architecture folder-შია, მაგრამ seasonal part ჯერ არ დაგვიმატებია.

შედეგის ინტერპრეტაცია:

- baseline SARIMA validation WMAE არის `1856.86053`;
- seasonal naive არის `1800.17359`;
- SARIMA baseline seasonal naive-ზე `56.69` WMAE-ით უარესია;
- relative გაუარესება არის `-3.14897%`;
- მთავარი მიზეზი ისაა, რომ aggregate model Store-Dept-level yearly seasonality-ს პირდაპირ ვერ ხედავს.

ამ baseline-ის მიზანი იყო starting point-ის მიღება. რეალური SARIMA improvement უნდა გამოჩნდეს მხოლოდ მაშინ, როცა დავამატებთ seasonal order-ს, მაგალითად weekly retail data-სთვის წლიურ სეზონურობაზე მორგებულ კომპონენტს. ამ dataset-ში 52-week seasonality ძლიერია, ამიტომ შემდეგი meaningful ნაბიჯი სწორედ seasonal component-ის დამატება და შედარებაა.

## SARIMA order search ექსპერიმენტი

baseline-ის შემდეგ `model_experiment_SARIMA.ipynb`-ში დავამატეთ controlled brute-force search მხოლოდ SARIMA order-ზე და allocation strategy-ზე. აქ არ არის SARIMAX და seasonal component ჯერ ცალკე tuning-ს საჭიროებს. მიზანი იყო გვენახა, გაუმჯობესდებოდა თუ არა pure aggregate SARIMA უკეთესი `(p, d, q)` order-ით.

გამოყენებული grid:

```python
p = [0, 1, 2]
d = [0, 1]
q = [0, 1, 2]
```

სულ გამოვიდა:

```text
18 SARIMA order
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

ეს უკეთესია baseline SARIMA-ზე:

```text
baseline SARIMA(1,1,1): 1856.86053
best searched SARIMA(1,0,2): 1829.87999
```

გაუმჯობესება:

```text
1856.86 -> 1829.88
```

ანუ order search-მა SARIMA baseline დაახლოებით `26.98` WMAE-ით გააუმჯობესა.

მაგრამ seasonal naive-ს მაინც ვერ აჯობა:

```text
seasonal naive: 1800.17359
best SARIMA:     1829.87999
```

best SARIMA ჯერ კიდევ დაახლოებით `29.71` WMAE-ით უარესია seasonal naive-ზე.

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

მიუხედავად იმისა, რომ order search-მა baseline SARIMA გააუმჯობესა, საუკეთესო SARIMA მაინც უარესია seasonal naive-ზე:

```text
seasonal naive WMAE = 1800.17
best SARIMA WMAE     = 1829.88
```

მთავარი მიზეზი არის ის, რომ seasonal naive row-level forecast-ს პირდაპირ იღებს იგივე Store-Dept-ის 52 კვირით ძველი sales-იდან. ეს ძალიან ძლიერი baseline-ია ამ competition-ში.

SARIMA კი აკეთებს ორეტაპიან approximation-ს:

```text
total weekly sales forecast
        ↓
row-level allocation
```

ამ პროცესში Store-Dept-level information იკარგება. Aggregate forecast შეიძლება მისაღები იყოს, მაგრამ თუ allocation ცოტათი მაინც ცდება, Kaggle/WMAE row-level metric ამას მკაცრად სჯის.

ამიტომ best SARIMA result-ის შეფასება ასეთია:

- baseline SARIMA-ზე უკეთესია;
- `blended_share`-ზე აშკარად უკეთესია;
- seasonal naive-ზე მაინც უარესია;
- final model candidate არ არის, მაგრამ useful diagnostic experiment არის.

## SARIMAX ექსპერიმენტის ანალიზი

SARIMA order search-ის შემდეგ ცალკე `model_experiment_SARIMAX.ipynb` notebook-ში გავტესტეთ SARIMAX, ანუ SARIMA external regressors-ით. აქ SARIMA არ გამოგვიყენებია. Seasonal order გამორთულია:

```python
seasonal_order = (0, 0, 0, 0)
```

SARIMAX-ის იდეა იყო, რომ aggregate weekly sales-ს გარდა model-ს ენახა weekly-level external signals:

- holiday share;
- average temperature;
- fuel price;
- CPI;
- unemployment;
- markdown totals;
- calendar week/month signals.

### SARIMAX-ის საუკეთესო შედეგი

SARIMAX experiment-ის საუკეთესო შედეგი იყო:

```text
best_order: (0, 0, 0)
best_allocation: last_year_share
best_use_exog: True
Validation WMAE: 2563.691454
Validation MAE: 2517.848644
Validation RMSE: 5135.390683
Improvement vs seasonal naive: -42.413569%
```

ეს შედეგი ბევრად უარესია როგორც seasonal naive-ზე, ისე SARIMA baseline-ზე და tuned SARIMA-ზე.

შედარება:

| Model | Setup | Validation WMAE | Seasonal naive-სთან შედარება |
| --- | --- | ---: | ---: |
| Seasonal naive | 52-week row-level lag | `1800.17359` | reference |
| Baseline SARIMA | `(1,1,1)` + `last_year_share` | `1856.86053` | `-3.14897%` |
| Tuned SARIMA | `(1,0,2)` + `last_year_share` | `1829.87999` | `-1.65020%` |
| SARIMAX | `(0,0,0)` + exog + `last_year_share` | `2563.69145` | `-42.41357%` |

ამ ცხრილიდან ჩანს, რომ SARIMAX ამ ფორმით არ გაუმჯობესდა. პირიქით, external regressors-მა model-ის performance მნიშვნელოვნად გააუარესა.

### რატომ გახდა SARIMAX ცუდი

ჩემი შეფასებით, მთავარი მიზეზი ისაა, რომ ჩვენ SARIMAX-ში external features aggregate weekly დონეზე შევიყვანეთ, ხოლო final metric row-level Store-Dept prediction-ს აფასებს.

SARIMAX ხედავს ასეთ data-ს:

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

3. **SARIMAX-მა historical yearly pattern ჩაანაცვლა სუსტი exog signal-ით.**  
   ამ dataset-ში ყველაზე ძლიერი signal არის 52-კვირიანი seasonal behavior. SARIMAX external variables-ს ზედმეტად ეყრდნობა, მაშინ როცა validation period-ისთვის row-level yearly structure უფრო მნიშვნელოვანია.

4. **Best SARIMAX order გახდა `(0,0,0)`.**  
   ეს ძალიან მნიშვნელოვანი სიგნალია. `(0,0,0)` ნიშნავს, რომ AR/MA dynamics პრაქტიკულად არ დაეხმარა და model ძირითადად exogenous regression-like behavior-ზე გადავიდა. რადგან exog features noisy იყო, result გაუარესდა.

5. **Allocation bottleneck მაინც დარჩა.**  
   თუნდაც aggregate weekly forecast გაუმჯობესებულიყო, row-level WMAE მაინც allocation-ზეა დამოკიდებული. SARIMAX-მა allocation პრობლემა არ გადაჭრა.

### რატომ იყო `last_year_share` ისევ საუკეთესო

SARIMAX-შიც საუკეთესო allocation იყო:

```text
last_year_share
```

მაგალითად საუკეთესო order-ზე:

```text
(0,0,0) + last_year_share WMAE = 2563.69
(0,0,0) + blended_share   WMAE = 2744.69
```

ეს იმავე დასკვნას ადასტურებს, რაც pure SARIMA-ში ვნახეთ: Store-Dept distribution-ისთვის 52 კვირით ძველი sales share უფრო სანდოა, ვიდრე recent average-თან blending.

### SARIMAX vs SARIMA

SARIMAX-ის დამატებამ არ გააუმჯობესა SARIMA:

```text
Tuned SARIMA WMAE:  1829.88
Best SARIMAX WMAE: 2563.69
```

გაუარესება:

```text
2563.69 - 1829.88 = 733.81 WMAE
```

ეს ძალიან დიდი სხვაობაა. ამიტომ ამ კონკრეტული implementation-ით SARIMAX არ უნდა ჩაითვალოს improvement-ად.

### რა ვისწავლეთ SARIMAX-იდან

SARIMAX-ის შედეგი არ ნიშნავს, რომ external features ყოველთვის ცუდია. ეს ნიშნავს, რომ ამ ფორმით aggregate SARIMAX არ იყო სწორი representation.

External features უფრო კარგად მუშაობს tree-based models-ში, რადგან ისინი raw row-level feature-ებს იყენებენ:

```text
Store + Dept + Date + markdown + holiday + store metadata
```

SARIMAX-ში კი ეს ყველაფერი დაიკუმშა ერთ weekly aggregate time series-ად. ამან დაკარგა ის დეტალი, რომელიც Kaggle metric-ისთვის აუცილებელია.

ამ ეტაპზე conclusions:

- SARIMAX არ გაუმჯობესდა;
- best SARIMAX ბევრად უარესია baseline SARIMA-ზე;
- tuned pure SARIMA უკეთესია SARIMAX-ზე;
- seasonal naive მაინც საუკეთესო classical baseline რჩება;
- SARIMAX-ის ამ ვერსიას final candidate-ად არ ავირჩევდი.

### Kaggle submission შედეგები: SARIMA vs SARIMAX

Validation-ზე SARIMAX ცუდად გამოვიდა, მაგრამ Kaggle public leaderboard-ზე შედეგი სხვანაირი მივიღე:

| Model | Kaggle score |
| --- | ---: |
| SARIMA | `3842` |
| SARIMAX | `3525` |

ანუ Kaggle-ზე SARIMAX-მა SARIMA-ზე უკეთესი score მისცა:

```text
3842 - 3525 = 317
```

ეს დაახლოებით `8.25%` გაუმჯობესებაა SARIMA submission-თან შედარებით.

ჩემი შეფასებით, ეს გაუმჯობესება უფრო **feature engineering / external regressors-ის ეფექტია**, ვიდრე training algorithm-ის ან SARIMA order-ის სიძლიერე. ორივე მოდელი მაინც aggregate time-series approach-ს იყენებს და ორივეს აქვს იგივე მთავარი bottleneck: weekly total forecast შემდეგ Store-Dept rows-ზე historical share-ებით ნაწილდება. განსხვავება ისაა, რომ SARIMAX-ში aggregate forecast-ს დამატებით ეძლევა `features.csv`-დან მიღებული external signal-ები:

- holiday indicator;
- markdown information;
- temperature;
- fuel price;
- CPI;
- unemployment;
- calendar-derived weekly signals.

Kaggle test period-ში ეს features, როგორც ჩანს, გარკვეულწილად დაეხმარა future weekly total sales-ის უკეთ დაჭერას. განსაკუთრებით holiday და markdown-related signal-ები შეიძლება public test period-ზე უფრო სასარგებლო აღმოჩნდა, ვიდრე ჩვენს validation split-ზე.

მნიშვნელოვანია, რომ ეს არ ნიშნავს, თითქოს SARIMAX validation-ზე უკეთესი იყო. პირიქით:

```text
Validation tuned SARIMA WMAE:  1829.88
Validation SARIMAX WMAE:       2563.69
Kaggle SARIMA score:           3842
Kaggle SARIMAX score:          3525
```

აქ ჩანს validation/Kaggle mismatch. ამის რამდენიმე მიზეზი შეიძლება იყოს:

1. **Validation period და Kaggle test period ერთნაირად არ იქცევა.**  
   Validation იყო 2012 წლის ბოლო 39 კვირა, ხოლო Kaggle test სხვა future პერიოდს აფასებს. External regressors validation-ზე noisy აღმოჩნდა, მაგრამ test-ზე უფრო სასარგებლო signal მისცა.

2. **SARIMAX-ის features aggregate დონეზე სუსტია, მაგრამ მთლიან weekly total-ს მაინც ეხმარება.**  
   Validation row-level WMAE ძალიან მკაცრად სჯის allocation-ის შეცდომებს. Kaggle-ზე კი თუ weekly total forecast ცოტათი უკეთ დაემთხვა, ეს საბოლოო submission score-ში მაინც გამოჩნდა.

3. **SARIMA სრულიად historical pattern-ზეა დამოკიდებული.**  
   Pure SARIMA-ს არ აქვს ინფორმაცია markdown-ზე, fuel price-ზე, CPI-ზე ან holiday context-ზე. თუ test period-ში external conditions განსხვავდება historical pattern-ისგან, SARIMA ამას ვერ ხედავს, SARIMAX კი ნაწილობრივ ხედავს.

4. **გაუმჯობესება მოდის არა უკეთესი per-row modeling-დან, არამედ better aggregate forecast-დან.**  
   SARIMAX კვლავ არ სწავლობს Store-Dept-level behavior-ს. ამიტომ მისი Kaggle score მაინც მაღალია tree-based models-თან შედარებით, მაგრამ SARIMA-ზე უკეთესი გამოვიდა, რადგან aggregate forecast-ში დამატებითი features დაეხმარა.

ამიტომ საბოლოო ინტერპრეტაცია ასეთია:

- validation-ზე SARIMAX unreliable აღმოჩნდა;
- Kaggle-ზე SARIMAX-მა SARIMA-ზე უკეთესი public score დადო;
- გაუმჯობესების მთავარი მიზეზი იყო external features, არა model architecture-ის ძირეული უპირატესობა;
- ორივე მოდელი მაინც სუსტია, რადგან Store-Dept-level demand-ს პირდაპირ არ სწავლობს;
- SARIMAX useful experiment არის, მაგრამ final candidate მაინც არ არის tree-based models-თან შედარებით.

## რატომ გამოვიდა SARIMA seasonal naive-ზე უარესი

ჩემი შეფასებით, მთავარი მიზეზი ისაა, რომ Walmart weekly sales ძალიან ძლიერი yearly seasonality-ით მუშაობს. ბევრი Store-Dept series-ისთვის ყველაზე ძლიერი signal არის:

```text
ამავე კვირის გაყიდვები ერთი წლის წინ
```

Seasonal naive პირდაპირ ამ signal-ს იყენებს. SARIMA baseline კი weekly total sales-ზეა trained და შემდეგ row-level prediction-ს historical share-ებით ანაწილებს. ამ დროს რამდენიმე მნიშვნელოვანი ინფორმაცია იკარგება:

1. **Store-Dept-specific seasonality იკარგება.**  
   Aggregate SARIMA ხედავს მთლიან Walmart sales-ს, მაგრამ ვერ ხედავს, რომ კონკრეტულ department-ს კონკრეტულ store-ში შეიძლება განსხვავებული seasonal pattern ჰქონდეს.

2. **Holiday behavior სუსტადაა დაჭერილი.**  
   WMAE holiday weeks-ს 5-ჯერ მეტ წონას აძლევს. SARIMA(1,1,1) არ იყენებს explicit holiday feature-ს, ამიტომ Thanksgiving/Christmas-like spikes-ს კარგად ვერ სწავლობს.

3. **External covariates არ გამოიყენება.**  
   `features.csv` და `stores.csv` იკითხება, მაგრამ baseline SARIMA მათ model-ში არ იყენებს. არ არის markdown, fuel price, CPI, temperature, store type, store size.

4. **Allocation step ძალიან მარტივია.**  
   Total weekly forecast შეიძლება reasonable იყოს, მაგრამ row-level distribution თუ არასწორია, final WMAE მაინც იზრდება. Kaggle metric row-level prediction-ს აფასებს და არა total weekly sales-ს.

5. **SARIMA order არ არის tune-ებული.**  
   `(1,1,1)` ავიღეთ როგორც სწრაფი baseline. შესაძლოა სხვა order უკეთესი იყოს, მაგრამ baseline მიზანი დიდი search არ ყოფილა.

## რა ვისწავლეთ

ეს ექსპერიმენტი სასარგებლოა, რადგან აჩვენებს, რომ მხოლოდ aggregate statistical forecasting საკმარისი არ არის ამ competition-ისთვის.

მთავარი დასკვნები:

- SARIMA სწრაფი და მარტივია, მაგრამ row-level Walmart task-ისთვის ზედმეტად coarse approach გამოდის.
- Strong yearly seasonality იმდენად მნიშვნელოვანია, რომ seasonal naive უკეთესი baseline აღმოჩნდა.
- Walmart competition-ში Store-Dept-level structure, holiday effects და tabular covariates ძალიან მნიშვნელოვანია.
- Classical model-ის გასაუმჯობესებლად საჭიროა ან per-series SARIMA/SARIMAX, ან hybrid approach, სადაც yearly lag და covariates ცალკე შედის.

## შედარება სხვა baseline-ებთან

ამ ეტაპზე SARIMA baseline-ის შედეგი:

```text
SARIMA validation WMAE: 1856.86
Seasonal naive WMAE: 1800.17
```

ეს ნიშნავს, რომ SARIMA ჯერ არ არის final candidate. ის უფრო diagnostic baseline-ია, რომელიც გვიჩვენებს, რომ simple aggregate time-series model ამ dataset-ზე საკმარისი არ არის.

Tree-based models უკეთ მუშაობენ, რადგან მათ შეუძლიათ ერთდროულად გამოიყენონ:

- store/dept identity;
- calendar features;
- holiday indicators;
- markdown/promotion data;
- historical lag signals;
- interactions between tabular variables.

## შემდეგი ნაბიჯები

თუ SARIMA/SARIMA მიმართულებას გავაგრძელებთ, შემდეგი გაუმჯობესებები იქნება:

1. per-series seasonal naive + SARIMA residual model;
2. SARIMA yearly seasonality-ით;
3. SARIMAX external regressors-ით, მაგალითად markdown და holiday flags;
4. Store-Dept clustering და თითო cluster-ზე separate SARIMA/SARIMA;
5. direct comparison Kaggle submission-ზე.

ამ ეტაპზე კი `baseline_sarima.ipynb` რჩება სწრაფ baseline-ად და არა final model candidate-ად.
