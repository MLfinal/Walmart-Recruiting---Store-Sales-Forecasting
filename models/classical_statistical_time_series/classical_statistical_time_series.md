# კლასიკური სტატისტიკური time-series მოდელების შედარება

ამ folder-ში სამი classical მიმართულება შევამოწმეთ: `ARIMA`, `SARIMA` და `Prophet`. მიზანი იყო გაგვეგო, რამდენად შორს მიდის ტრადიციული time-series forecasting Walmart-ის weekly Store–Dept გაყიდვებზე, სანამ უფრო მოქნილ tree-based და deep-learning მოდელებს შევადარებდით.

სამივე მიმართულებაში დავიცავით ერთი ძირითადი პრინციპი: ვასწავლით მხოლოდ წარსულს და ვამოწმებთ მომავალ 39 კვირაზე. მთავარი metric ყველგან არის Kaggle-ის **WMAE** — holiday კვირის შეცდომას აქვს weight `5`, სხვა კვირას `1`; ამიტომ დაბალი WMAE უკეთესია.

## საერთო ექსპერიმენტული ჩარჩო

ყველა family-ში notebook-ების flow ერთი ლოგიკით არის აგებული:

```text
CSV-ების წაკითხვა და Date/calendar preparation
→ chronological validation split
→ seasonal-naive reference
→ baseline
→ controlled experiments / feature ცვლილება
→ WMAE და diagnostics W&B-ზე
→ full-history best-pipeline packaging
→ W&B Model Registry
→ raw test inference და Kaggle submission
```

Train/validation separation მნიშვნელოვანია: ბოლო `39` train კვირა (`2012-02-03`–`2012-10-26`) არის future holdout, ხოლო model-ს ეს target-ები training დროს არ უნახავს. Prophet-ისა და ARIMA/SARIMA-ის notebooks სხვადასხვა internal preparation/allocation გზას იყენებენ; ამიტომ მათ README-ებში seasonal-naive reference განსხვავებულია (`1604.27` Prophet-ში, `1800.17` aggregate ARIMA/SARIMA-ში). ეს ნიშნავს, რომ family-ების WMAE შედარება სასარგებლო შედეგობრივი სურათია, მაგრამ აბსოლუტური ranking-ისთვის ერთიანი evaluator-ის გადამოწმება ყოველთვის ყველაზე მკაცრი წესია.

## რა განსხვავებაა მოდელებს შორის

| Family | რას სწავლობს | სეზონურობა / მოვლენები | Store–Dept დონე | მთავარი შეზღუდვა |
| --- | --- | --- | --- | --- |
| ARIMA | ერთი aggregate weekly total series | `p,d,q` autoregressive/moving-average structure | total forecast შემდეგ historical share-ით ნაწილდება | კონკრეტული Store–Dept pattern იკარგება |
| SARIMA | ARIMA-ს seasonal გაფართოება | ამ განხორციელებაში seasonal order გამორთულია | იგივე aggregate + allocation | რეალური SARIMA seasonality არ ყოფილა ჩართული |
| Prophet | ცალკე model თითო Store–Dept series-ზე | trend, yearly seasonality და event calendar | პირდაპირ per-series forecast | მოკლე/noisy series და მკვეთრი retail spikes რთულია |

ARIMA და SARIMA ერთი aggregation philosophy-ით მუშაობენ:

```text
ყველა Store–Dept-ის გაყიდვების weekly total
→ ერთი statistical model
→ total forecast
→ last-year Store–Dept share-ებით row-level allocation
```

ეს სწრაფია და ახსნადია, მაგრამ competition-ის target row-level-ია. თუ total სწორიცაა, historical share-ს ერთი პატარა აცდენაც ათასობით row-ზე ნაწილდება. სწორედ ამიტომ aggregate models განსაკუთრებით სუსტდებიან holiday spike-ებისა და department-specific ცვლილებების დროს.

Prophet ამ bottleneck-ს არ იყენებს: თითო series-ს პირდაპირ forecast-ს უკეთებს და მხოლოდ sparse/cold-start series-ზე გადადის 52-week seasonal-naive fallback-ზე. ამის ფასი არის ათასობით ცალკე fit, თუმცა შედეგი გაცილებით უკეთესი გამოვიდა.

## Baseline-ების შედარება

| მოდელი | Baseline configuration | Validation WMAE | Seasonal-naive reference | დასკვნა |
| --- | --- | ---: | ---: | --- |
| ARIMA | aggregate `ARIMA(1,1,1)` + `last_year_share` | `1856.8605` | `1800.1736` | naive-ზე `3.15%` უარესი |
| SARIMA | იგივე `(1,1,1)`, seasonal component disabled | `1856.8605` | `1800.1736` | ARIMA-სთან იდენტური, რადგან seasonal ნაწილი გამორთულია |
| Prophet | per-series trend + yearly seasonality + generic holiday | `1625.4781` | `1604.2697` | naive-ზე `1.32%` უარესი, მაგრამ ბევრად ძლიერი საწყისი classical model |

Baseline-ებმა ერთი მნიშვნელოვანი რამ აჩვენა: ამ მონაცემში 52-week seasonal naive ძალიან ძლიერია. ის იმავე Store–Dept-ის წინა წლის შესაბამის კვირას პირდაპირ იყენებს, მაშინ როცა aggregate მოდელები ამ ინფორმაციას ჯერ total-ში აერთიანებენ და შემდეგ share-ებით აბრუნებენ. Prophet კი per-series დონეს ინარჩუნებს, თუმცა მისი smooth trend/seasonality ჯერ ვერ იმეორებს ყველა promotion-driven ან abrupt spike-ს.

## ARIMA — რა ვცადეთ და რა ვისწავლეთ

`baseline_arima.ipynb`-ის შემდეგ `model_experiment_ARIMA.ipynb`-ში გადავარჩიეთ `18` order (`p∈{0,1,2}`, `d∈{0,1}`, `q∈{0,1,2}`) და ორი allocation მეთოდი: `last_year_share` და `blended_share`.

| ARIMA ვერსია | Setup | Validation WMAE |
| --- | --- | ---: |
| baseline | `(1,1,1)` + last-year share | `1856.8605` |
| საუკეთესო pure ARIMA | `(1,0,2)` + last-year share | `1829.8800` |
| საუკეთესო ARIMAX | `(0,0,0)` + external regressors + last-year share | `2563.6915` |

Order search-მა baseline `26.98` WMAE-ით გააუმჯობესა, მაგრამ 52-week naive-ს მაინც `29.71`-ით ჩამორჩა. `last_year_share` მუდმივად სჯობდა `blended_share`-ს: მიმდინარე/ბოლო პერიოდების საშუალო share-მა distribution-ში noise შეიტანა.

ARIMAX-ში დავამატეთ aggregate holiday, Markdown, temperature, fuel price, CPI, unemployment და calendar signal-ები. შედეგი მნიშვნელოვნად გაუარესდა. მიზეზი არ არის ის, რომ external features თავისთავად ცუდია; ისინი aggregate total-ს მიეწოდა, მაშინ როცა შეფასება Store–Dept row-ზე ხდება. features ვერ აბრუნებს იმ granular structure-ს, რომელიც aggregation-მა დაკარგა.

## SARIMA — რა განსხვავებული იყო სინამდვილეში

SARIMA notebook-ებში იგივე baseline/order/allocation ექსპერიმენტები გაკეთდა, შემდეგ კი SARIMAX-ში exogenous features დაემატა.

| SARIMA ვერსია | Setup | Validation WMAE |
| --- | --- | ---: |
| baseline | `(1,1,1)` + last-year share | `1856.8605` |
| საუკეთესო `model_sarima` | `(1,0,2)` + last-year share | `1831.6176` |
| საუკეთესო SARIMAX | `(0,0,0)` + exogenous features + last-year share | `2563.6915` |

საუკეთესო SARIMA tuned ARIMA-ს მხოლოდ `1.74` WMAE-ით ჩამორჩა. ეს architecture-ის უპირატესობა/ნაკლოვანება არ არის: ამ განხორციელებაში `seasonal_order` რეალურად გამორთული იყო, ამიტომ SARIMA პრაქტიკულად ARIMA-like მოდელია. ანალოგიურად SARIMAX და ARIMAX ერთსა და იმავე validation შედეგამდე მივიდნენ.

Kaggle-ზე SARIMAX-ის public score `3525` უკეთესი იყო SARIMA-ის `3842` score-ზე (`8.25%` გაუმჯობესება). ეს validation-ს არ აუქმებს: სავარაუდოდ test პერიოდის aggregate total-ს external signals ოდნავ დაეხმარა, მაგრამ row-level allocation limitation ორივე მიდგომას დარჩა. ამიტომ SARIMAX საინტერესო diagnostic/feature experiment-ია და არა ამ ოჯახის champion.

## Prophet — განსხვავებული, წარმატებული მიმართულება

Prophet-ში aggregate forecast არ გამოგვიყენებია. `baseline_prophet.ipynb` და `model_experiment_prophet.ipynb` თითო Store–Dept pair-ზე მუშაობს. sparse history-ისას fallback არის 52-week seasonal naive; ამიტომ model-ის failure ან არასაკმარისი data არ ქმნის დაუცველ prediction-ს.

| Prophet ვერსია | მთავარი ცვლილება | Validation WMAE | შეფასება |
| --- | --- | ---: | --- |
| baseline | generic holiday, raw Prophet | `1625.4781` | naive-ზე ოდნავ უარესი |
| v1 | external covariates, მაგრამ backward-fill leakage | `4236.6848` | invalid comparison; არ გამოიყენება არჩევანში |
| v2 | seasonal-naive residual Prophet | `3808.4181` | over-prediction, უარყოფითი შედეგი |
| v3 | `0.5 × Prophet + 0.5 × SeasonalNaive52` | `1402.2612` | ორივე კომპონენტზე უკეთესი |
| v4 | event-aware holiday windows + იგივე blend | **`1367.4470`** | Prophet family-ის champion |
| v5 | Markdown/context covariates + regularisation | `1415.5392` | v4-ზე უარესი |
| v6 | historical blend-alpha tuning; `alpha=0.45` | `1373.0902` | v4-ის `0.50` ოდნავ უკეთესია |

v4-ში `IsHoliday` ერთი საერთო flag აღარ იყო. Super Bowl, Labor Day, Thanksgiving და Christmas ცალკე events-ად განვაცხადეთ; Thanksgiving/Christmas-ს `[-7, 0]` day window მივეცით, რათა წინასადღესასწაულო გაყიდვაც დაჭერილიყო. ეს leakage-safe feature engineering-ია, რადგან event calendar წინასწარ ცნობილია.

საბოლოო prediction:

```text
0.50 × event-aware raw Prophet
+ 0.50 × same Store–Dept-ის SeasonalNaive52
```

ამ blend-ის ძალა complementary errors-ია: seasonal naive ძალიან კარგად ინარჩუნებს კონკრეტული series-ის წლიურ დონეს, Prophet კი trend/event ცვლილებას ასწორებს. external numeric covariates ამ per-series setting-ში უფრო ხმაურიანი აღმოჩნდა, ვიდრე event calendar.

## საბოლოო შედეგობრივი სურათი

| Family-ის საუკეთესო valid model | Validation WMAE | შედარებითი შეფასება |
| --- | ---: | --- |
| Prophet v4 event-aware 50/50 blend | **`1367.4470`** | აშკარად საუკეთესო classical შედეგი; საკუთარი naive reference-ზე `14.76%` უკეთესი |
| ARIMA tuned `(1,0,2)` | `1829.8800` | baseline-ზე უკეთესი, მაგრამ seasonal-naive-ზე უარესი |
| SARIMA tuned `(1,0,2)` | `1831.6176` | ARIMA-სთან თითქმის იდენტური; true seasonal component არ გამოუყენებია |
| ARIMAX / SARIMAX | `2563.6915` | external features aggregate representation-ში არასტაბილური |

ამიტომ არჩევანი ერთმნიშვნელოვანია: classical family-ის **champion არის Prophet v4**, არა იმიტომ, რომ სახელით უფრო ახალია, არამედ იმიტომ, რომ იგი პირდაპირ ინარჩუნებს Store–Dept granularity-ს და event calendar-ს seasonal-naive signal-თან სწორად აერთიანებს.

## Reproducibility, W&B და inference

სამივე family-ის best pipelines W&B-ზე artifact/Model Registry სახითაა რეგისტრირებული და inference notebook-ები მათ ხელახლა training-ის გარეშე იყენებენ.

| Family | Registry pipeline | Raw input |
| --- | --- | --- |
| ARIMA | `Walmart_ARIMA_Pipeline:champion` | `test.csv` |
| SARIMA | `Walmart_SARIMA_Pipeline:champion` | `test.csv` |
| Prophet | `Walmart_Prophet_Raw_Pipeline:champion` | `test.csv` |

ARIMA/SARIMA pipeline-ში ინახება aggregate forecasting და row-level allocation logic. Prophet pipeline-ში ინახება full-history-ზე fit-ებული per-series model JSON-ები, event configuration, stored history, 52-week fallback და `alpha=0.50`. ყველა inference notebook ამოწმებს prediction count-სა და finiteness/non-negativity-ს, ქმნის Kaggle CSV-ს და W&B-ზე ტვირთავს submission artifact-ს (CSV, manifest და შესაბამის diagnostics).

Prophet-ის pipeline registration run არის `rlm39vch`; raw registry inference run არის `1w7kftox`. ამ inference-ში `115064` row შეიქმნა და Kaggle upload წარმატებით დასრულდა. ARIMAX/SARIMAX-ს raw `features.csv`-ც სჭირდება, რადგან external regressors pipeline-ში შედის; ისინი ამ comparison-ში ჩემპიონებად არ შეგვირჩევია.

## დასკვნა

ამ სამმა მიმართულებამ განსხვავებული ცოდნა მოგვცა. ARIMA-მ გვაჩვენა, რომ order tuning aggregate forecast-ს მცირე გაუმჯობესებას აძლევს, მაგრამ allocation bottleneck რჩება. SARIMA-მ დაადასტურა, რომ მხოლოდ სახელის შეცვლა seasonality-ს არ ამატებს: seasonal order უნდა იყოს რეალურად ჩართული. ARIMAX/SARIMAX-მ გვასწავლა, რომ covariate-ის ღირებულება მისი granularity-სა და forecasting representation-ზეა დამოკიდებული.

Prophet v4-მა კი დაადასტურა, რომ ამ dataset-ისთვის ყველაზე ეფექტური classical recipe არის per-series modeling, წინასწარ ცნობილი event calendar და ძლიერი 52-week seasonal reference-ის blend. მიუხედავად ამისა, ეს family უფრო interpretable statistical benchmark-ია, ვიდრე პროექტის საერთო საუკეთესო მოდელი: tree-based და შესაბამისი deep-learning მიდგომები Store/Dept/features-ის ურთიერთქმედებებს უფრო პირდაპირ სწავლობენ.
