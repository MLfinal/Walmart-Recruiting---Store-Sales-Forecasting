# TimesFM — foundation model ექსპერიმენტები

## მიზანი

TimesFM-ის პირველი ექსპერიმენტის მიზანი იყო pretrained foundation model-ის სუფთა **zero-shot** შესაძლებლობის გაზომვა Walmart weekly sales-ზე. ამ run-ში model არ დაგვიტრენინგებია, არ გაგვიკეთებია fine-tuning, feature engineering, hyperparameter search ან prediction blending. Google-ის უკვე გაწვრთნილ `TimesFM 2.5 200M` checkpoint-ს მხოლოდ historical sales sequence მივაწოდეთ და შევამოწმეთ, რამდენად კარგად იწინასწარმეტყველებდა ბოლო 39 კვირას ამ dataset-ის წინასწარი ნახვის გარეშე.

## რას წარმოადგენს TimesFM

TimesFM არის decoder-only time-series foundation model. ჩვეულებრივი project-specific neural network-ისგან განსხვავებით, იგი თავიდან არ სწავლობს მხოლოდ Walmart-ის მონაცემს: pretrained checkpoint უკვე დიდი რაოდენობის სხვადასხვა ტიპის time series-ზეა გაწვრთნილი და ახალი sequence-ის pattern-ს inference დროს ამოიცნობს.

ჩვენ გამოვიყენეთ:

```text
checkpoint       = google/timesfm-2.5-200m-transformers
model parameters = 231,289,280
mode             = zero-shot
fine-tuned       = False
context          = 104 კვირა
horizon          = 39 კვირა
hardware         = Tesla T4
```

TimesFM-ს არ მივაწოდეთ `Store`, `Dept`, `IsHoliday`, Markdown, store metadata ან სხვა external covariate. თითო Store–Dept მისთვის დამოუკიდებელი რიცხვითი sequence იყო. ამიტომ ეს შედეგი ზომავს მხოლოდ pretrained model-ის მიერ target history-დან ამოცნობილ trend-სა და temporal pattern-ს.

## Notebook-ის flow

`model_experiment_TimesFM.ipynb` შემდეგი თანმიმდევრობით მუშაობს:

```text
environment და TimesFM dependencies
→ configuration და W&B authentication
→ train.csv loading/validation
→ chronological 104/39 split
→ Store–Dept weekly sequences
→ TimesFM 2.5 checkpoint download
→ batched zero-shot inference
→ original-row WMAE evaluation
→ seasonal-naive comparison და plots
→ W&B metrics, tables და evaluation artifact
```

### მონაცემის მომზადება

სრული `train.csv` შეიცავდა:

```text
rows        = 421,570
all series  = 3,331
date range  = 2010-02-05 → 2012-10-26
```

ბოლო `39` unique კვირა validation-ად გამოვყავით:

```text
history weeks      = 104
validation weeks   = 39
validation period  = 2012-02-03 → 2012-10-26
validation rows    = 115,588
validation series  = 3,204
```

`3,331` არის მთელ train history-ში არსებული Store–Dept pair-ების რაოდენობა, ხოლო `3,204` — ის pair-ები, რომლებსაც final validation პერიოდში რეალური rows ჰქონდათ. Metric სწორედ ამ `115,588` რეალურ row-ზე დაითვალა; validation-ში არარსებული series ხელოვნურად არ დაგვიმატებია.

TimesFM თანაბრად დაშორებულ time points-ს მოითხოვს. ამიტომ თითო series გლობალურ weekly calendar-ზე გადავიყვანეთ და არარსებული historical observation `0`-ით შევავსეთ. ეს მნიშვნელობები მხოლოდ model context-ის რეგულარიზაციისთვის გამოიყენება; validation score კვლავ original rows-ზე ითვლება.

### Seasonal-naive reference

შედარებისთვის თითო Store–Dept-ზე გამოვიყენეთ 52-კვირიანი reference:

```text
SeasonalNaive52(t) = WeeklySales(t - 52 weeks)
```

ეს Walmart-ისთვის ძლიერი benchmark-ია, რადგან store/department yearly pattern და holiday season ხშირად მეორდება. TimesFM-ის შედეგის შეფასება მხოლოდ აბსოლუტური WMAE-ით არ შემოვიფარგლეთ — შევამოწმეთ, აჯობა თუ არა pretrained model-მა ამ მარტივ, მაგრამ ძლიერ წესს.

## Zero-shot inference

checkpoint-ის ჩატვირთვას `47.89` წამი დასჭირდა. ყველა `3,204` series დაიყო batch-ებად და T4-ზე inference შესრულდა. არცერთ series-ს არ დასჭირდა fallback:

```text
TimesFM series       = 3,204
fallback series      = 0
forecast time        = 10.90 წუთი
peak GPU memory      = 13.26 GB
```

T4-ის 16 GB მეხსიერებიდან დაახლოებით 13.26 GB გამოიყენა, ამიტომ batch size-ის ზედმეტად გაზრდა უსაფრთხო არ იქნებოდა. საბოლოო run ტექნიკურად სტაბილური იყო: ყველა validation series-ზე შეიქმნა forecast და NaN/Inf prediction არ დარჩენილა.

## შედეგი

პირველი zero-shot run-ის metrics:

| Metric | შედეგი |
| --- | ---: |
| TimesFM validation WMAE | **`1672.2525`** |
| TimesFM validation MAE | `1611.6392` |
| SeasonalNaive52 WMAE | `1798.9705` |
| TimesFM improvement vs seasonal naive | **`7.0439%`** |

სხვაობა:

```text
1798.9705 - 1672.2525 = 126.7180 WMAE
```

ეს მნიშვნელოვანი შედეგია: model-ს Walmart-ზე ერთი optimization step-იც არ გაუვლია და მხოლოდ sales history მიიღო, თუმცა 52-week seasonal reference `126.72` WMAE-ით დაამარცხა. შესაბამისად, pretrained representation რეალურად შეიცავს Walmart-ისთვის გამოსადეგ trend/seasonality signal-ს და zero-shot forecast შემთხვევითი baseline არ არის.

ამავე დროს `1672.25` ჯერ არ არის პროექტის საუკეთესო validation შედეგი. მაგალითად, XGBoost-ისა და Prophet-ის საუკეთესო documented runs უფრო დაბალ WMAE-ს აჩვენებს, თუმცა მათი validation window/evaluator და feature availability ზუსტად ერთნაირი არ არის. ამიტომ TimesFM-ის ყველაზე სამართლიანი პირველი დასკვნა მისივე იმავე-row seasonal reference-თან `7.04%` გაუმჯობესებაა.

## Plot-ების ინტერპრეტაცია

Weekly validation MAE plot-ზე TimesFM კვირების უმეტესობაში seasonal naive-ზე დაბლაა. ყველაზე რთულ პერიოდებში ორივე forecast-ის error ერთდროულად იზრდება, რაც მიუთითებს საერთო რთულ retail spikes-ზე და არა მხოლოდ TimesFM-ის ტექნიკურ failure-ზე. რამდენიმე ადრეულ კვირაში TimesFM naive-ზე უარესია, მაგრამ დანარჩენ horizon-ზე მიღებული სტაბილური მოგება საბოლოო WMAE-ს ამცირებს.

Actual-vs-prediction scatter ძირითად დიაგონალთან ახლოსაა: დაბალი და საშუალო გაყიდვების დიდი ნაწილი სწორ scale-ზეა. dispersion იზრდება მაღალი-volume rows-ზე; ჩანს როგორც underprediction, ისე რამდენიმე მკვეთრი outlier. ეს მოსალოდნელია, რადგან zero-shot model-ს არ ჰქონია Store/Dept identity, holiday event ან promotion context.

## W&B logging

Run სრულად დაილოგა W&B project-ში:

```text
run name = timesfm_v1_zero_shot_all_series_39w
run id   = 3h3c62o0
job type = zero_shot_validation
group    = timesfm-experiments
```

W&B-ზე ინახება:

- სრული configuration და checkpoint ID;
- WMAE, MAE და seasonal-naive comparison;
- TimesFM/fallback coverage;
- model load time, inference time და peak GPU memory;
- validation prediction table;
- weekly diagnostic table;
- weekly error და actual-vs-prediction plot;
- reproducibility-სთვის prediction SHA-256;
- evaluation artifact `timesfm-v1-zero-shot-validation`.

Prediction hash:

```text
993ab7f1791c521c27452e02a3add26ee8d2c7778ca00e42c80906579d966064
```

Artifact-ში შევინახეთ validation predictions CSV, weekly diagnostics CSV, metrics JSON და diagnostic PNG. ეს ჯერ `evaluation` artifact-ია და არა Model Registry pipeline: zero-shot baseline-ის გაზომვის ეტაპზე best configuration ჯერ არჩეული არ ყოფილა.

## პირველი ექსპერიმენტის დასკვნა

TimesFM zero-shot baseline წარმატებულია როგორც foundation-model proof of concept. იგი სწრაფად გაეშვა T4-ზე, მოიცვა ყველა `3,204` validation series fallback-ის გარეშე და training-ის გარეშე აჯობა იმავე evaluator-ის seasonal naive-ს. მთავარი აღმოჩენა არის არა მხოლოდ `1672.25` WMAE, არამედ ის, რომ external features-ისა და Walmart-specific fitting-ის გარეშე model-მა უკვე სასარგებლო complementary forecast შექმნა.

ამ run-მა ასევე აჩვენა გაუმჯობესების ძირითადი სივრცე: model ამჟამად ვერ ხედავს Store/Dept ურთიერთობებს, holiday event-ს, Markdown-ს და სხვა known-future context-ს, ხოლო high-sales rows-ზე error dispersion ჯერ მაღალია.

## Experiment v2 — seasonal residual და historical calibration

v1-მა დაამტკიცა, რომ raw zero-shot TimesFM სასარგებლო forecast-ს ქმნის. v2-ში model-ის weights კვლავ არ შეგვიცვლია: არც training და არც fine-tuning არ ჩატარებულა. შევამოწმეთ, TimesFM უკეთ შეძლებდა თუ არა მთლიან `Weekly_Sales`-ის ნაცვლად წინა წლის შესაბამის კვირასთან ცვლილების პროგნოზირებას.

Residual განვსაზღვრეთ ასე:

```text
Residual(t) = WeeklySales(t) - WeeklySales(t - 52)
```

TimesFM-ის residual forecast ისევ გაყიდვების scale-ზე დავაბრუნეთ:

```text
TimesFMResidualPrediction(t)
= SeasonalNaive52(t) + ForecastedResidual(t)
```

შემდეგ სამი განსხვავებული signal გავაერთიანეთ:

```text
w_naive   × SeasonalNaive52
+ w_raw    × RawTimesFM
+ w_resid  × TimesFMResidualPrediction
```

Weights იყო არაუარყოფითი, მათი ჯამი `1`, ხოლო grid-ის ნაბიჯი `0.05`. ყველაზე მნიშვნელოვანი methodological წესი იყო ის, რომ weights final validation-ზე არ აგვირჩევია. არსებული 143 კვირა სამ chronological ნაწილად გაიყო:

```text
calibration context       = პირველი 84 კვირა
weight calibration        = 20 კვირა, 2011-09-16 → 2012-01-27
untouched validation      = 39 კვირა, 2012-02-03 → 2012-10-26
```

Residual sequence მხოლოდ 52-week difference-ის შექმნის შემდეგ იწყება. ამიტომ calibration residual context იყო `32` კვირა, final residual context კი `52` კვირა. ეს ბევრად მოკლე history-ა, ვიდრე raw TimesFM-ის `84/104` კვირა და residual-only შედეგის შეფასებისას მნიშვნელოვანი გარემოებაა.

### Inference-ის შესრულება

v2-ში ერთი checkpoint ოთხჯერ გამოვიყენეთ:

| Pass | Context | Horizon | დრო |
| --- | ---: | ---: | ---: |
| raw calibration | 84 | 20 | `9.61` წუთი |
| residual calibration | 32 | 20 | `9.65` წუთი |
| raw final | 104 | 39 | `9.65` წუთი |
| residual final | 52 | 39 | `9.65` წუთი |

სრული TimesFM inference გაგრძელდა `38.56` წუთი. Checkpoint cache-იდან `14.09` წამში ჩაიტვირთა, peak GPU memory კი `6.67 GB` იყო. ყველა `3,204` validation series კვლავ სრულად დამუშავდა.

### Calibration-ის მიერ არჩეული weights

20-კვირიან historical calibration-ზე grid search-მა აირჩია:

```text
w_naive     = 0.45
w_raw       = 0.10
w_residual  = 0.45
reported grid WMAE = 1908.5810
```

Raw TimesFM-ის მხოლოდ `0.10` weight აჩვენებს, რომ calibration regime-ში მისი დამოუკიდებელი forecast არასტაბილური იყო. Seasonal baseline და residual reconstruction თითქმის თანაბარი წონით გაერთიანდა, ხოლო raw forecast მცირე complementary correction-ად დარჩა.

### Final validation შედეგები

არჩეული weights უცვლელად გადავიტანეთ ბოლო 39 კვირაზე:

| Candidate | Calibration WMAE | Final validation WMAE | Final MAE |
| --- | ---: | ---: | ---: |
| Calibrated blend | `1935.8667` | **`1620.5430`** | `1621.6440` |
| Raw TimesFM | `3601.4726` | `1672.2525` | `1611.6392` |
| TimesFM residual | `2072.9576` | `1720.1709` | `1736.1453` |
| SeasonalNaive52 | `2106.7113` | `1799.0451` | `1796.0549` |

v2 blend-ის გაუმჯობესება v1 raw TimesFM-თან შედარებით:

```text
1672.2525 → 1620.5430
improvement = 51.7095 WMAE = 3.0922%
```

იმავე final evaluator-ის seasonal naive-სთან შედარებით გაუმჯობესება დაახლოებით `9.92%` გახდა. შესაბამისად, residual-only forecast v1-ზე უკეთესი არ ყოფილა, მაგრამ მისი error raw TimesFM-ის error-ისგან საკმარისად განსხვავდებოდა და blend-ში სასარგებლო კომპონენტი გახდა.

Weekly plot-ზეც calibrated blend უმეტეს კვირაში raw, residual და seasonal curves-ის უკიდურეს შეცდომებს არბილებს. ყველაზე რთულ spike-ზე ყველა candidate უარესდება, თუმცა blend-ის peak შედარებით დაბალია. ეს არის ensemble-ის რეალური სარგებელი: იგი არც ერთ forecast-ს სრულად არ ენდობა.

### Calibration logging-ის აღმოჩენილი შეუსაბამობა

Run-ში grid search-ის საუკეთესო calibration WMAE არის `1908.5810`, ხოლო შემდეგ იმავე blend column-ის ხელახლა დათვლილი WMAE — `1935.8667`. მიზეზი execution order ან target leakage არ ყოფილა. Weight-search objective blend-ს clipping-მდე აფასებდა, ხოლო `TimesFM_CalibratedBlend` column-ის შექმნისას prediction `0`–`300000` დიაპაზონში იჭრებოდა.

`Weekly_Sales`/seasonal history-ში მცირე უარყოფითი values არსებობს, ამიტომ clipping-მდე და clipping-ის შემდეგ WMAE ზუსტად ერთნაირი ვერ დარჩა. Final `1620.5430` score უკვე რეალურად შენახულ, clipped prediction-ზეა დათვლილი და სწორია, მაგრამ weight selection objective deployment postprocessing-ს იდეალურად არ ემთხვეოდა. ეს leakage არ არის; ეს calibration implementation-ის მცირე inconsistency-ია, რომელიც შემდეგ weight search-ში clipping-ის grid-ის შიგნით გადატანით უნდა გასწორდეს.

### W&B run და artifact

```text
run name = timesfm_v2_zero_shot_residual_calibrated_blend
run id   = hewjpg8r
artifact = timesfm-v2-zero-shot-residual-calibration
```

W&B-ზე დაილოგა ოთხივე candidate-ის calibration/final score, არჩეული weights, top-50 weight combinations, calibration და validation predictions, weekly comparison plot, runtime/GPU metrics, config და prediction hash:

```text
73970198d411d7724645aff8e3acf65871991b5697f628a14472ecce44d7867f
```

## v1 და v2-ის ერთიანი დასკვნა

| Version | მიდგომა | Final WMAE | მთავარი ცოდნა |
| --- | --- | ---: | --- |
| v1 | raw TimesFM zero-shot | `1672.2525` | pretrained forecast seasonal naive-ს training-ის გარეშე სჯობს |
| v2 residual-only | SeasonalNaive52 + forecasted residual | `1720.1709` | მოკლე residual history დამოუკიდებლად საკმარისი არ არის |
| v2 calibrated blend | naive + raw + residual | **`1620.5430`** | განსხვავებული zero-shot errors-ის გაერთიანება საუკეთესოა |

ორი run-ის შემდეგ TimesFM-ის საუკეთესო valid შედეგია v2 calibrated blend `1620.54`. ჯერ არცერთი TimesFM parameter არ შეცვლილა და external covariate არ გამოგვიყენებია. მიღებული გაუმჯობესება მთლიანად representation-ის შეცვლამ, historical calibration-მა და complementary forecasts-ის გაერთიანებამ შექმნა.

## Experiment v3 — TimesFM XReg და corrected calibration

v3-ში პირველად მივეცით TimesFM-ს target history-ის გარდა known-future context. TimesFM-ის neural weights კვლავ არ დაგვიტრენინგებია; გამოვიყენეთ TimesFM 2.5-ის ოფიციალური XReg mechanism, რომელიც context-ში target/covariate ურთიერთობას მსუბუქი regression correction-ით აკავშირებს pretrained forecast-თან.

v2-ის დაახლოებით 39-წუთიანი raw/residual inference აღარ გაგვიმეორებია. W&B-დან ჩამოვტვირთეთ artifact:

```text
timesfm-v2-zero-shot-residual-calibration:latest
calibration rows = 59,317
validation rows  = 115,588
```

ამით v3 notebook-ში v2-ის ზუსტად იგივე predictions გამოვიყენეთ და მხოლოდ ახალი XReg component დავითვალეთ. ეს ამცირებს runtime-ს და გამორიცხავს იმის რისკს, რომ comparison checkpoint/API-ის შემთხვევითი განსხვავების გამო შეიცვალოს.

### დამატებული covariates

Dynamic numerical covariates:

```text
Temperature, Fuel_Price, CPI, Unemployment
ამ ოთხივე feature-ის missing indicator
log1p(total Markdown1–5)
Markdown missing count
week-of-year sine/cosine
month sine/cosine
```

Dynamic categorical covariates:

```text
IsHoliday
event: none / Super Bowl / Labor Day / Thanksgiving / Christmas
week of year
```

Static covariates:

```text
Store, Dept, Type, Size
```

ყველა dynamic array მოიცავდა როგორც context-ს, ისე შესაბამის forecast horizon-ს. Markdown-ის missing value გახდა `0` და ცალკე missing indicator; Temperature/Fuel/CPI/Unemployment დამუშავდა Store-level forward fill-ით და მხოლოდ context-period median-ით. Future period-იდან backward fill არ გამოგვიყენებია, ამიტომ feature preparation leakage-safe დარჩა.

### XReg mode comparison

Calibration-ზე შევადარეთ ორივე ოფიციალური composition order:

| XReg mode | Combined WMAE | Pure XReg WMAE |
| --- | ---: | ---: |
| `timesfm + xreg` | **`3242.5574`** | `3213.9526` |
| `xreg + timesfm` | `3259.0939` | `3173.4844` |

Final forecast-ისთვის combined prediction-ის მიხედვით ავირჩიეთ `timesfm + xreg`. ორივე mode calibration-ზე საკმაოდ მაღალი error-ით მუშაობდა, მაგრამ comparison ერთსა და იმავე historical period-ზე შესრულდა და არჩევანი final validation-მდე გაკეთდა.

არჩეული XReg mode-ის final შედეგი:

```text
TimesFM + XReg WMAE = 1939.0755
Pure XReg WMAE      = 1992.9980
```

ანუ XReg standalone forecast raw TimesFM-ზე ბევრად უარესი აღმოჩნდა. ბევრი external feature-ის არსებობა თავისთავად უკეთეს forecast-ს არ ნიშნავს: XReg-ის regression correction linear/limited-ია, Store–Dept histories მოკლეა, feature effects department-specific და ხშირად არაწრფივია. განსაკუთრებით Markdown/holiday effect სხვადასხვა department-ზე განსხვავებულად მოქმედებს.

### v2 clipping inconsistency-ის გასწორება

v3-ში ყველა weight combination ზუსტად deployment order-ით შეფასდა:

```text
weighted candidate sum
→ clip [0, 300000]
→ calibration WMAE
```

არჩეული blend materialize-ის შემდეგ assertion ამოწმებს, რომ ხელახლა დათვლილი calibration WMAE grid-search score-ს `1e-9` tolerance-ით ემთხვევა. v3-ში calibration logging-ის ძველი `1908.58 / 1935.87` შეუსაბამობა აღარ არსებობს.

### Corrected blend weights

ოთხკომპონენტიანმა search-მა აირჩია:

```text
SeasonalNaive52 weight   = 0.40
Raw TimesFM weight       = 0.05
Residual TimesFM weight  = 0.45
TimesFM XReg weight      = 0.10
calibration WMAE         = 1918.6194
```

XReg-ს მხოლოდ `10%` მიეცა. ეს ზუსტად შეესაბამება standalone diagnostics-ს: XReg საკმარისად ძლიერი არ არის ძირითადი forecast-ისთვის, მაგრამ განსხვავებული covariate-driven error pattern მცირე correction-ის სახით სასარგებლოა. Raw TimesFM-ის weight `10%`-დან `5%`-მდე შემცირდა, seasonal/residual structure კი კვლავ blend-ის `85%` დარჩა.

### Final validation შედეგები

| Candidate | Calibration WMAE | Final validation WMAE | Final MAE |
| --- | ---: | ---: | ---: |
| TimesFM v3 corrected blend | `1918.6194` | **`1588.8029`** | `1593.3578` |
| Raw TimesFM | `3601.4726` | `1672.2525` | `1611.6392` |
| Residual TimesFM | `2072.9576` | `1720.1709` | `1736.1453` |
| SeasonalNaive52 | `2106.7113` | `1799.0451` | `1796.0549` |
| TimesFM XReg | `3242.5574` | `1939.0755` | `1903.4276` |
| Pure XReg | — | `1992.9980` | `1958.3267` |

v2-დან გაუმჯობესება:

```text
v2 blend = 1620.5430
v3 blend = 1588.8029
difference = 31.7401 WMAE
relative improvement = 1.9586%
```

v1 raw zero-shot-თან შედარებით საერთო გაუმჯობესება უკვე `83.45` WMAE, დაახლოებით `4.99%`-ია. Seasonal naive-სთან შედარებით v3 დაახლოებით `11.69%`-ით უკეთესია.

მნიშვნელოვანია, რომ v3-v2 მოგება მთლიანად XReg-ს არ უნდა მივაწეროთ: v3-ში ერთდროულად დაემატა XReg candidate და გასწორდა clipping-aware weight selection. მიღებული `1588.80` არის ამ ორი ცვლილების ერთობლივი final result. მიუხედავად ამისა, არჩეული `10%` XReg weight ადასტურებს, რომ calibration-მა covariate forecast მთლიანად არ უარყო.

Weekly plot-ზე v3 blend უმეტეს კვირაში raw TimesFM-სა და seasonal naive-ზე დაბალია. XReg curve რამდენიმე პერიოდში მკვეთრად უარესდება და standalone მაღალი WMAE სწორედ ამ არასტაბილურობიდან მოდის. Blend ამ spikes-ს `10%` weight-ით ზღუდავს, მაგრამ იმ კვირებში, სადაც covariates სასარგებლოა, მცირე correction-ს ინარჩუნებს.

### Runtime და W&B

TimesFM model cache-იდან `3.11` წამში ჩაიტვირთა. ორი calibration mode და ერთი final XReg pass ჯამში მხოლოდ `1.01` წუთს გაგრძელდა:

```text
xreg + timesfm calibration  = 0.381 წუთი
timesfm + xreg calibration  = 0.301 წუთი
selected final XReg         = 0.329 წუთი
```

W&B run:

```text
run name = timesfm_v3_xreg_covariates_corrected_blend
run id   = eyzk1bc6
artifact = timesfm-v3-xreg-corrected-calibration
```

Artifact-ში ინახება calibration/validation predictions, ორივე XReg mode-ის table, ოთხკომპონენტიანი weight grid, candidate scores, metrics/config, diagnostic plot და prediction hash:

```text
1f679bf22ecf973baacc07019a1bb7257d0e4f8cba62def07f6e9bf59f7b7592
```

## v1–v3 მდგომარეობა

| Version | მთავარი ცვლილება | საუკეთესო WMAE | v1-თან ცვლილება |
| --- | --- | ---: | ---: |
| v1 | raw TimesFM zero-shot | `1672.2525` | reference |
| v2 | seasonal/raw/residual historical blend | `1620.5430` | `3.09%` უკეთესი |
| v3 | XReg covariates + corrected four-way blend | **`1588.8029`** | `4.99%` უკეთესი |

სამი run-ის შემდეგ TimesFM family-ის champion არის v3 blend. Foundation model-ის weights ჯერ ერთხელაც არ განახლებულა: მთელი გაუმჯობესება zero-shot temporal forecast-ის, yearly residual representation-ის, leakage-safe covariates-ის და historical calibration-ის სწორად გაერთიანებიდან მივიღეთ.

## v3.1 — XReg ablation და temporal stability audit

v3.1 ახალი forecasting model არ არის. მისი მიზანი იყო v3-ის `1588.80` გაუმჯობესების წყაროს იზოლირება: რამდენი მოიტანა clipping fix-მა და რამდენი — XReg-მა. Notebook-მა W&B-დან ჩამოტვირთა v3-ის უკვე შენახული calibration/validation predictions და არც TimesFM inference გაუშვია, არც training.

```text
source artifact   = timesfm-v3-xreg-corrected-calibration:latest
calibration rows  = 59,317, 20 კვირა
validation rows   = 115,588, 39 კვირა
```

### Corrected blend XReg-ის გარეშე

პირველად იგივე clipping-aware search მხოლოდ სამ ძველ კომპონენტზე ჩავატარეთ:

```text
SeasonalNaive52   = 0.40
Raw TimesFM       = 0.10
Residual TimesFM  = 0.50

calibration WMAE = 1933.8741
validation WMAE  = 1615.9719
```

ეს v2-ის ძველ `1620.5430` შედეგზე `4.5711` WMAE-ით უკეთესია. შესაბამისად, v2→v3 გაუმჯობესების მცირე ნაწილი მართლაც calibration implementation-ის გასწორებამ შექმნა: weight search და final prediction ახლა ორივე clipping-ის შემდეგ ფასდება.

### Corrected blend XReg-ით

ოთხკომპონენტიანმა search-მა კვლავ ზუსტად v3 weights აირჩია:

```text
SeasonalNaive52   = 0.40
Raw TimesFM       = 0.05
Residual TimesFM  = 0.45
TimesFM XReg      = 0.10

calibration WMAE = 1918.6194
validation WMAE  = 1588.8029
```

პირდაპირი ablation:

| Setup | Calibration WMAE | Final validation WMAE |
| --- | ---: | ---: |
| corrected blend, XReg-ის გარეშე | `1933.8741` | `1615.9719` |
| corrected blend, XReg-ით | **`1918.6194`** | **`1588.8029`** |

XReg-ის ზუსტი marginal contribution final validation-ზე:

```text
1615.9719 - 1588.8029 = 27.1690 WMAE
relative gain = 1.6813%
```

ამგვარად v2→v3 სრული `31.7401` WMAE მოგება ორ ნაწილად იყოფა:

```text
clipping-aware calibration fix ≈ 4.5711
XReg complementary component   ≈ 27.1690
total                           ≈ 31.7401
```

ეს ადასტურებს, რომ XReg-ის `10%` weight შემთხვევითი დეკორაცია არ ყოფილა: untouched final validation-ზე მან რეალური დამატებითი მოგება შექმნა, მიუხედავად იმისა, რომ standalone XReg forecast სუსტი იყო.

### Time-ordered stability folds

XReg-ის ეფექტის სტაბილურობისთვის 20-კვირიან calibration period-ში ორი expanding-time audit გაკეთდა. თითო fold-ში weights მხოლოდ უფრო ადრეულ კვირებზე შეირჩა და შემდეგ მომავალ 5 კვირაზე შეფასდა.

| Fold | Weight-fit პერიოდი | Holdout | No-XReg WMAE | With-XReg WMAE | XReg gain |
| --- | --- | --- | ---: | ---: | ---: |
| fold 1 | 2011-09-16 → 2011-11-18 | 2011-11-25 → 2011-12-23 | `3343.1626` | **`2978.6975`** | **`+364.4651`** |
| fold 2 | 2011-09-16 → 2011-12-23 | 2011-12-30 → 2012-01-27 | **`2058.7755`** | `2071.0250` | `-12.2495` |

Fold 1-ში, რომელიც Thanksgiving/Christmas პერიოდს მოიცავს, XReg ძალიან სასარგებლო აღმოჩნდა. ეს ლოგიკურია: XReg ხედავს holiday/event/Markdown/calendar covariates-ს. Fold 2-ში კი მან მცირე ზიანი მოიტანა. ორი fold-იდან მხოლოდ ერთზე იყო holdout improvement დადებითი.

Weights-იც regime-ის მიხედვით იცვლებოდა:

```text
fold 1 with XReg: naive 0.20, raw 0.20, residual 0.40, XReg 0.20
fold 2 with XReg: naive 0.35, raw 0.00, residual 0.50, XReg 0.15
```

ეს ნიშნავს, რომ XReg-ს აქვს სასარგებლო holiday/context signal, მაგრამ მისი ღირებულება დროში მუდმივი არ არის. იგი უნდა დარჩეს მცირე, კონტროლირებულ ensemble component-ად და არა primary forecast-ად.

### Audit W&B run

```text
run name = timesfm_v3_1_xreg_ablation_stability_audit
run id   = f4pnn5ga
artifact = timesfm-v3-1-xreg-ablation-audit
```

Artifact-ში ინახება ორივე complete weight search, final ablation, temporal fold results, audited validation predictions, metrics და plot.

Audit notebook-ის install cell-მა `pandas 3.0.3` და `numpy 2.5.1` დააყენა, რის გამოც Colab-მა `google-colab`/`numba` dependency warning დაბეჭდა. Audit მაინც სრულად დასრულდა და მისი გამოთვლები მხოლოდ pandas/numpy-ზე მუშაობდა, ამიტომ შედეგი არ დაზიანებულა. შემდეგ notebook-ში ამ packages-ს აღარ განვაახლებთ იძულებით; დავაყენებთ მხოლოდ საჭირო W&B/model dependencies-ს, რათა Colab environment-ის pinned versions შევინარჩუნოთ.

v3 artifact CSV-ის ხელახლა წაკითხვის შემდეგ prediction hash გახდა:

```text
556efcd0202ded86e8cc8af53603fc4980a74df94485c57616b1cf5668b6ccab
```

v3-ის original hash-თან byte-level განსხვავება CSV serialization/reload precision-იდან მოდის; final WMAE `1588.8029448973086` ზუსტად განმეორდა, ამიტომ forecast-ის შინაარსობრივი reproducibility დადასტურდა.

## v3.1-ის საბოლოო დასკვნა

v3.1-მა ორი რამ დაადასტურა:

1. v3-ის მოგების დიდი ნაწილი (`27.17` WMAE) ნამდვილად XReg-ის complementary contribution იყო და არა მხოლოდ clipping bug fix.
2. XReg დროში არასტაბილურია: holiday-heavy fold-ზე ძლიერია, შემდეგ fold-ზე მცირედ აზიანებს შედეგს.

ამიტომ TimesFM family-ის champion უცვლელია — v3 corrected four-way blend `1588.8029`. v3.1 არის ამ არჩევანის audit და არა ახალი champion model.

## Experiment v4 — LoRA adaptation + capped XReg hybrid

v4 იყო პირველი ექსპერიმენტი, სადაც TimesFM-ის forecast Walmart data-ზე neural training-ით მოვარგეთ. სრული 231M-parameter model არ გაგვიწვრთნია: pretrained base weights frozen დარჩა და `all-linear` layers-ზე მხოლოდ მცირე LoRA adapters დაემატა.

```text
total parameters      = 232,672,192
trainable LoRA params = 1,382,912
trainable share       = 0.5944%
```

LoRA და XReg end-to-end ერთად არ სწავლობს. ოფიციალური XReg wrapper საკუთარ base TimesFM checkpoint-ს იყენებს და PEFT adapter-ს პირდაპირ ვერ იღებს. ამიტომ v4 hybrid-ის flow იყო:

```text
historical Weekly_Sales
→ TimesFM 2.5 + trained LoRA adapter
→ Walmart-adapted forecast

v3 artifact
→ previously validated TimesFM XReg forecast

LoRA + XReg + raw + residual + SeasonalNaive52
→ capped calibration blend
```

### Chronological splits

Final 39-week validation კვლავ training/calibration-ის გარეთ დარჩა:

```text
LoRA training targets end = 2011-06-10
LoRA validation           = 2011-06-17 → 2011-09-09, 13 კვირა
blend calibration         = 2011-09-16 → 2012-01-27, 20 კვირა
final validation          = 2012-02-03 → 2012-10-26, 39 კვირა
```

სრული training matrix მოიცავდა `3331` Store–Dept series-ს. LoRA dataset-ში შეიქმნა `6000` random windows; validation-ში target observation ჰქონდა `3108` series-ს. Missing Store–Dept observations loss-ში zero weight-ით გამოირიცხა, ხოლო holiday target weeks weight `5`-ით შეფასდა.

### LoRA configuration

```text
context length         = 32
training horizon       = 13
epochs requested       = 6
batch size             = 8
gradient accumulation  = 4
learning rate          = 5e-5
LoRA rank / alpha      = 4 / 8
LoRA dropout           = 0.05
optimizer              = AdamW
early stopping patience= 2
```

Context `32` ავირჩიეთ, რათა ადრეულ history-ში საკმარისი random training windows შექმნილიყო. Base TimesFM internal normalization-ს იყენებს, ამიტომ sales values გარედან არ დაგვისკალავს. Loss იყო original-scale holiday-weighted absolute error.

### Training curve და early stopping

| Epoch | Train WMAE | LoRA-validation WMAE | Elapsed |
| ---: | ---: | ---: | ---: |
| 1 | `3811.3475` | **`2123.3362`** | `5.84` წუთი |
| 2 | `3690.0201` | `2143.4547` | `11.17` წუთი |
| 3 | `3613.0545` | `2163.2184` | `16.52` წუთი |

Train error ყოველ epoch-ზე მცირდებოდა, მაგრამ validation error პირველივე epoch-ის შემდეგ იზრდებოდა. Early stopping epoch 3-ზე ჩაირთო და best adapter epoch 1-დან აღადგინა. ეს არის მკაფიო overfitting/domain-overadaptation pattern: adapter training windows-ს უკეთ ერგებოდა, თუმცა მომავალ 13 კვირაზე generalization უარესდებოდა.

რეალური training მოსალოდნელ 2–4 საათზე ბევრად სწრაფი აღმოჩნდა:

```text
training time          = 16.52 წუთი
LoRA calibration pass  = 6.10 წუთი
LoRA final pass        = 6.08 წუთი
```

### Final LoRA შედეგი

Best adapter-მა უფრო გვიან პერიოდებზე ძალიან ცუდი forecast შექმნა:

```text
LoRA calibration WMAE = 9926.4555
LoRA final WMAE       = 8396.0651
LoRA final MAE        = 8296.9159
```

LoRA-validation `2123.34` და final `8396.07` შორის დიდი სხვაობა აჩვენებს, რომ 32-week context/13-week target-ზე ადრეული history-ით მიღებული adaptation 39-week horizon-ზე არ გადავიდა. შესაძლო მიზეზებია:

- training context-ში სრული 52-week annual cycle არ ეტეოდა;
- LoRA მხოლოდ ადრეულ regime-ზე ისწავლა;
- 13-week training horizon და 39-week final horizon განსხვავდება;
- short random windows-ზე adapter-მა local scale/pattern ზედმეტად შეცვალა და pretrained generalization დააზიანა.

ეს numerical crash არ ყოფილა: predictions finite იყო და evaluation ბოლომდე შესრულდა. პრობლემა იყო forecast quality, არა pipeline failure.

### Capped blend-ის გადაწყვეტილება

v4 calibration search-ში candidates იყო:

```text
SeasonalNaive52
Raw TimesFM
Residual TimesFM
LoRA TimesFM
TimesFM XReg, maximum weight 0.10
```

არჩეული weights:

```text
SeasonalNaive52   = 0.40
Raw TimesFM       = 0.05
Residual TimesFM  = 0.45
LoRA TimesFM      = 0.00
TimesFM XReg      = 0.10
```

Calibration-მა LoRA სრულად უარყო და `0%` weight მისცა. დანარჩენი weights ზუსტად v3 configuration-ს დაუბრუნდა. ამიტომ v4 final blend prediction და hash v3.1-ს დაემთხვა:

```text
v4 blend WMAE = 1588.8029
v3 WMAE       = 1588.8029
improvement   = 0.0%
prediction SHA-256 = 556efcd0202ded86e8cc8af53603fc4980a74df94485c57616b1cf5668b6ccab
```

Final candidate ranking:

| Candidate | Calibration WMAE | Final WMAE |
| --- | ---: | ---: |
| v4 blend / v3 configuration | **`1918.6194`** | **`1588.8029`** |
| Raw TimesFM | `3601.4726` | `1672.2525` |
| Residual TimesFM | `2072.9576` | `1720.1709` |
| SeasonalNaive52 | `2106.7113` | `1799.0451` |
| TimesFM XReg | `3242.5574` | `1939.0755` |
| TimesFM LoRA | `9926.4555` | `8396.0651` |

### W&B artifacts

```text
run name            = timesfm_v4_lora_xreg_hybrid
run id              = o5x6bw14
adapter artifact    = timesfm-v4-lora-adapter
evaluation artifact = timesfm-v4-lora-xreg-evaluation
```

W&B-ზე ინახება epoch-level train/validation WMAE, learning rate, elapsed time, best epoch, trainable parameter count, LoRA adapter files, calibration weight search, candidate scores, validation predictions და diagnostic plots.

## v4-ის დასკვნა

LoRA training ტექნიკურად წარმატებით შესრულდა და adapter reproducibly შეინახა, მაგრამ model quality მკვეთრად გააუარესა. Corrected calibration-მა უსაფრთხოდ დაიცვა final forecast და LoRA-ს `0%` weight მისცა. ამიტომ v4 ახალი champion არ არის და TimesFM family-ის საუკეთესო მოდელად კვლავ v3 corrected blend `1588.8029` რჩება.

ეს უარყოფითი შედეგიც მნიშვნელოვანი ექსპერიმენტული დასკვნაა: pretrained foundation model-ის parameter-efficient adaptation ავტომატურად გაუმჯობესებას არ ნიშნავს. მოკლე context/horizon-ზე fine-tuning-მა ამ შემთხვევაში TimesFM-ის ძლიერი zero-shot generalization დააზიანა.

## საბოლოო მოდელის არჩევა და pipeline-ის შეფუთვა

LoRA-ს შედეგის შემდეგ საბოლოო pipeline-ის საფუძვლად v3 ავირჩიეთ. ეს არჩევანი მხოლოდ ერთი run-ის მიხედვით არ გაკეთებულა: v3-ს ჰქონდა TimesFM family-ში საუკეთესო final-validation WMAE `1588.8029`, v3.1 audit-მა მისი შედეგი გაიმეორა, ხოლო v4 calibration-მა LoRA-ს `0%` weight მისცა და ისევ ზუსტად v3-ის კომბინაციას დაუბრუნდა.

```text
SeasonalNaive52       = 0.40
Raw TimesFM           = 0.05
Residual TimesFM      = 0.45
TimesFM + XReg        = 0.10
LoRA                  = 0.00
```

ამიტომ pipeline-ში არ შეგვიტანია მხოლოდ ყველაზე ახალი ექსპერიმენტი; შევინახეთ ის მოდელი და კავშირები, რომლებმაც chronological validation-ზე რეალურად საუკეთესო შედეგი აჩვენა.

### რას ნიშნავს ამ შემთხვევაში სრული pipeline

`model_experiment_TimesFM_final_pipeline.ipynb`-ში შევქმენით `TimesFMRawPipeline` კლასი. მისი public contract არის:

```python
predictions = pipeline.predict(raw_test)
```

Input არის Kaggle-ის raw `test.csv` ფორმატის DataFrame, ხოლო output — იმავე row order-ში დალაგებული final `Weekly_Sales` prediction. ამ ერთ object-ში ინახება:

- Walmart-ის სრული training history;
- `features.csv`-ის external covariates და `stores.csv`-ის metadata;
- Store–Dept series-ის აგების, weekly alignment-ისა და missing-value დამუშავების წესები;
- `SeasonalNaive52`, raw TimesFM და seasonal-residual TimesFM forecast-ის ლოგიკა;
- v3-ში გამოყენებული leakage-safe XReg feature engineering;
- საბოლოო ოთხი blend weight, clipping limits და model configuration;
- raw rows-ის forecast matrix-იდან საწყის Kaggle row order-ში დაბრუნების ლოგიკა.

XReg-ის dynamic numerical covariates მოიცავს `Temperature`, `Fuel_Price`, `CPI`, `Unemployment` მნიშვნელობებსა და მათ missing indicators-ს, Markdown-ის log-total/missing-count features-ს და calendar sin/cos ნიშნებს. Dynamic categorical ნაწილში შედის holiday/event/week-of-year ინფორმაცია, static ნაწილში კი Store, Dept, Type და Size. Future target ან validation-derived feature pipeline-ში არ შედის.

TimesFM-ის `925 MB` pretrained neural weights pickle-ში ხელახლა არ დუბლირდება. Pipeline ინახავს pinned checkpoint ID-ს — `google/timesfm-2.5-200m-pytorch` — და პირველ inference-ზე ზუსტად ამ checkpoint-ს lazy-load რეჟიმში ტვირთავს. შესაბამისად, W&B artifact შეიცავს Walmart-specific fitted state-სა და სრულ inference orchestration-ს, ხოლო უცვლელი foundation-model weights reproducibly მოდის Hugging Face-დან. პირველი გაშვებისთვის ინტერნეტი საჭიროა; შემდეგ checkpoint Colab/Hugging Face cache-ში რჩება.

### serialization და raw-input contract test

Pipeline artifact-ის ატვირთვამდე notebook სრულ `test.csv`-ზე იძახებს `pipeline.predict(test_raw)`-ს. ეს ტესტი ამოწმებს არა მხოლოდ pickle-ის შექმნას, არამედ იმავე გზას, რომელსაც inference გამოიყენებს: raw schema → feature engineering → სამი TimesFM path → XReg → blend → original row order.

Contract pass-ის შემდეგ notebook:

1. pipeline-ს `walmart_timesfm_v3_raw_pipeline.pkl` ფაილად ინახავს;
2. pickle-ს თავიდან კითხულობს და metadata-ს ამოწმებს;
3. წერს manifest-ს configuration-ით, source validation WMAE-ით, runtime-ითა და prediction SHA-256 hash-ით;
4. W&B artifact-ში ამატებს pipeline pickle-ს, manifest-ს, registry reference-სა და contract predictions-ს;
5. artifact-ს აკავშირებს `wandb-registry-model/Walmart_TimesFM_Raw_Pipeline` registry collection-თან `champion` და `latest` aliases-ით.

Source experiment-თან კავშირის შესანარჩუნებლად registration run ასევე იყენებს `timesfm-v3-xreg-corrected-calibration:latest` evaluation artifact-ს და config-ში აფიქსირებს champion validation WMAE-ს `1588.8029448973086`.

### contract test-მა აღმოჩენილი integration პრობლემები

Pipeline-ის პირველი Colab contract runs-მა სამი თანმიმდევრული API/shape პრობლემა გამოავლინა. ეს training-ის ან v3 შედეგის შეცდომა არ ყოფილა — პრობლემა ექსპერიმენტის ცალკეულ ფუნქციებში არსებული ლოგიკის reusable class-ში გადატანისას გაჩნდა.

**1. `forecast_with_covariates()`-ს გადაეცა `horizon=`.** TimesFM `2.0.2`-ის `TimesFM_2p5.forecast_with_covariates()` ამ argument-ს არ იღებს; XReg horizon-ს future covariate arrays-ის სიგრძიდან ადგენს. ამიტომ call-იდან `horizon=horizon` ამოვიღეთ.

```text
TypeError: unexpected keyword argument 'horizon'
```

**2. model XReg-compatible რეჟიმში არ იყო compiled.** ჩვეულებრივი `forecast()` მუშაობდა, მაგრამ XReg მოითხოვს backcast-ის დაბრუნებასაც. `_load_model()`-ის `ForecastConfig`-ში დავამატეთ:

```python
return_backcast=True
```

```text
ValueError: For XReg, `return_backcast` must be set to True
```

**3. `return_backcast=True`-ის შემდეგ ordinary forecast-ის shape შეიცვალა.** Raw და residual calls უკვე მხოლოდ `39` future values-ს კი არა, `224` context + `39` future values-ს, სულ `263` columns-ს აბრუნებდა. Seasonal forecast-ის shape იყო `(3169, 39)`, TimesFM output-ის კი `(3169, 263)`, რის გამოც blend-მდე residual reconstruction ვერ შესრულდა.

```text
ValueError: operands could not be broadcast together with shapes (3169,39) (3169,263)
```

გამოსწორებისას raw, residual და XReg arrays point-forecast ფორმატში გადავიყვანეთ და მხოლოდ ბოლო horizon columns დავტოვეთ:

```python
forecast = forecast[:, -horizon:]
```

ამის გარდა blend-მდე დავამატეთ explicit shape contract. `seasonal`, `raw`, `residual` და `xreg` ოთხივე component-ის მოსალოდნელი shape არის `(series_count, horizon)`; შეუსაბამობისას pipeline ახლა ყველა მიღებულ shape-ს ერთ მკაფიო error-ში აჩვენებს. ამით ჩუმი broadcasting ან არასწორი row mapping აღარ დაიშვება.

ეს debugging sequence მნიშვნელოვანი იყო, რადგან მხოლოდ pickle-ის წარმატებით შექმნა pipeline-ის სისწორეს ვერ დაადასტურებდა. Full raw-input execution-მა გვაჩვენა, რომ model loading, compile configuration, XReg API და output semantics ერთ reproducible inference contract-ში ნამდვილად უნდა ემთხვეოდეს.

### წარმატებული final contract და W&B Registry registration

სამივე შესწორების შემდეგ notebook თავიდან გავუშვით და complete raw-input contract წარმატებით დასრულდა:

```text
raw test rows          = 115,064
Store–Dept series      = 3,169
stored history rows    = 421,570
stored features rows   = 8,190
contract runtime       = 1.465 წუთი
pipeline size          = 13.984 MB
prediction minimum     = 0.0
prediction mean        = 16,563.8952
prediction maximum     = 293,424.1790
prediction SHA-256     = 97529caaa57ed8334c3eaf7e9cdf1f7a95d0bb4efee4c09166433bf02a9dfefe
```

Pipeline pickle-ის reload metadata ზუსტად დაემთხვა original object-ს. Contract-მა ყველა `115,064` raw test row-ზე finite prediction დააბრუნა, სწორი row count შეინარჩუნა და output range clipping limits-ში დატოვა. ამის შემდეგ model artifact წარმატებით აიტვირთა და Registry-ს დაუკავშირდა:

```text
W&B run name      = timesfm_v3_raw_pipeline_registration
W&B run id        = j1cbzk3d
artifact name     = timesfm-v3-raw-input-pipeline
registry target   = wandb-registry-model/Walmart_TimesFM_Raw_Pipeline
registry URI      = wandb-registry-model/Walmart_TimesFM_Raw_Pipeline:champion
aliases           = champion, latest
registered        = True
```

Registration run-ში ლოგირდება pipeline size/runtime, verified row count, prediction range და hash, raw-input contract status, champion validation WMAE `1588.8029`, manifest, registry reference, contract predictions და serialized pipeline. Source evaluation artifact-თან lineage-იც შენარჩუნებულია: `timesfm-v3-xreg-corrected-calibration:latest` → registered v3 raw pipeline.

ამ ეტაპზე TimesFM-ის training-to-inference ჯაჭვი დასრულებულია: v3 experiment-მა აირჩია საუკეთესო forecast composition, v3.1-მა selection შეამოწმა, v4-მა LoRA უარყო, final packaging run-მა კი champion-ის მთლიანი raw-data processing და prediction flow ერთ Registry model-ში შეინახა. დამოუკიდებელ inference notebook-ს აღარ სჭირდება feature engineering-ის ხელახლა დაწერა — იგი Registry-დან იღებს `:champion` pipeline-ს და raw `test.csv`-ზე პირდაპირ `predict()`-ს იძახებს.

## საბოლოო inference — Registry champion-იდან submission-მდე

`timesfm_inference.ipynb` არის დამოუკიდებელი inference notebook. იგი experiment notebook-ს, local checkpoint-ს ან წინასწარ დამუშავებულ feature table-ს არ იყენებს. Notebook-ის flow არის:

```text
raw test.csv
→ W&B Registry: Walmart_TimesFM_Raw_Pipeline:champion
→ cloudpickle pipeline load
→ pipeline.predict(test_raw)
→ validation + Kaggle schema check
→ submission CSV + manifest
→ W&B prediction artifact
```

Notebook თავიდან მხოლოდ `test.csv`-ს კითხულობს და ამოწმებს raw schema-ს: `Store`, `Dept`, `Date`, `IsHoliday`. შემდეგ W&B run-ში `use_artifact()`-ით იღებს Registry champion-ს. Downloaded object-ის type აუცილებლად `TimesFMRawPipeline` უნდა იყოს; feature engineering-ის, history matrix-ის, XReg covariates-ის ან blending-ის კოდი inference notebook-ში ხელახლა არ იწერება.

Registry-დან რეალურად ჩაიტვირთა:

```text
resolved artifact      = Walmart_TimesFM_Raw_Pipeline:champion
pipeline type          = TimesFMRawPipeline
model checkpoint       = google/timesfm-2.5-200m-pytorch
stored history rows    = 421,570
stored feature rows    = 8,190
seasonal period        = 52
xreg mode              = timesfm + xreg
weights                = 0.40 seasonal / 0.05 raw / 0.45 residual / 0.10 xreg
```

ეს output ადასტურებს, რომ inference-მ ნამდვილად Registry model გამოიყენა და არა შემთხვევით დარჩენილი local object.

### inference run-ის შედეგი

```text
W&B run name          = timesfm_v3_champion_registry_inference
W&B run id            = nyqjuuwe
raw test rows         = 115,064
prediction runtime    = 29.703 წუთი
prediction minimum    = 0.0
prediction mean       = 16,563.8942
prediction maximum    = 293,424.1742
zero predictions      = 97
all values finite     = True
prediction SHA-256    = 0e09858d9ae377569318d3e80a85ae088fc937cb493640f444ce7123115788b7
```

Fresh inference runtime registration contract-ზე გრძელი იყო, რადგან ამ run-ში Registry-დან ახლად ჩატვირთულ pipeline-ს pretrained `925 MB` TimesFM checkpoint-ის initialization და სრული raw/residual/XReg execution დასჭირდა. Registration contract-ის `1.465` წუთი უკვე warm Colab process/cache-ზე გაიზომა, ამიტომ ეს ორი დრო სხვადასხვა runtime state-ს აღწერს და პირდაპირ model-speed comparison არ არის.

Inference prediction hash registration contract-ის hash-ს byte-level-ზე არ დაემთხვა. მიუხედავად ამისა, row count და range იგივეა, mean მხოლოდ დაახლოებით `0.0011`-ით, maximum კი დაახლოებით `0.0048`-ით განსხვავდება. ეს GPU/linear-solver floating-point execution-ის ძალიან მცირე numerical variation-ია და არა სხვა pipeline/configuration-ის გამოყენება. Registry metadata, component weights, input rows და output order უცვლელი დარჩა.

### submission validation და W&B logging

Prediction-ის შემდეგ notebook-მა Kaggle ID ააგო `Store_Dept_Date` ფორმატით და, ხელმისაწვდომობის შემთხვევაში, ყველა ID და row order `sampleSubmission.csv`-ს შეადარა. Duplicate ID, row-count mismatch და non-finite prediction არ აღმოჩნდა. საბოლოო ფაილი წარმატებით შეიქმნა:

```text
/content/drive/MyDrive/walmart_competition_inference/timesfm/timesfm_v3_champion_submission.csv
rows = 115,064
```

Inference logging cell-ში W&B-ზე იგზავნება prediction distribution histogram, პირველი `1,000` row-ის preview table, runtime, min/mean/max, zero count, prediction hash, resolved Registry artifact, submission CSV და JSON manifest. Prediction artifact-ის სახელია `timesfm-v3-champion-inference`, aliases — `latest` და `champion-pipeline`. ამგვარად lineage სრულად იკითხება:

```text
v3 evaluation artifact
→ registered raw-input champion pipeline
→ inference run nyqjuuwe
→ submission prediction artifact
```

### Kaggle upload-ის შედეგი

CSV-ის შექმნა და model inference წარმატებული იყო, მაგრამ optional Kaggle CLI upload ამ Colab runtime-ში authentication-ის არქონის გამო ვერ შესრულდა:

```text
You must authenticate before you can call the Kaggle API.
```

ეს forecast-ის ან pipeline-ის failure არ არის; submission CSV უკვე Google Drive-ში სწორად იყო შენახული. თავდაპირველ final cell-ში ამ auxiliary upload failure-ზე `RuntimeError` ჩნდებოდა და `run.finish()`-მდე execution წყდებოდა. Finalized notebook-ში ეს flow გავასწორეთ: Kaggle failure ინახება W&B summary-ში როგორც `kaggle/submitted=False` და error text, მაგრამ valid inference output არ იკარგება, W&B run ყოველთვის იხურება და `inference_complete=True` იბეჭდება. Kaggle upload შეგვიძლია ცალკე, credentials-ის გამართვის შემდეგ, უკვე შექმნილი CSV-ით შევასრულოთ.

## TimesFM-ის საბოლოო შედეგი

TimesFM-ის საუკეთესო შეფასება დარჩა v3 corrected blend WMAE `1588.8029`. Raw zero-shot-თან შედარებით blend-მა seasonal structure, residual dynamics და მცირე XReg contribution გააერთიანა; v3.1 audit-მა XReg-ის რეალური, თუმცა დროში არასტაბილური სარგებელი აჩვენა; LoRA-მ generalization მკვეთრად გააუარესა და calibration-მა იგი სწორად გამორიცხა.

Registry champion pipeline-ით შექმნილი `timesfm_v3_champion_submission.csv` Kaggle-ზე წარმატებით შეფასდა:

```text
submission = timesfm_v3_champion_submission.csv
message    = TimesFM v3 W&B Registry champion pipeline
status     = Complete (after deadline)
public     = 2742.68603
private    = 2853.40612
```

Public score `2742.68603` პროექტის XGBoost `2806` და LightGBM `2809` recorded scores-ზე დაბალია, მაგრამ final/private score `2853.40612` ორივე tree model-ზე მაღალია. ამიტომ TimesFM public leaderboard-ზე განსაკუთრებით ძლიერი აღმოჩნდა, ხოლო private/final ranking-ში XGBoost და LightGBM კვლავ წინ რჩება. TimesFM-მა TFT-ის private `3058.98280` შედეგს `205.57668` WMAE-ით აჯობა და მთლიან პროექტში მესამე საუკეთესო documented private/final submission გახდა.

საბოლოოდ გვაქვს არა მხოლოდ საუკეთესო validation prediction, არამედ სრული reproducible lifecycle: გაშვებული ექსპერიმენტები და diagnostics, W&B artifacts, audited champion selection, raw-input pipeline, W&B Model Registry registration და Registry-დან შესრულებული დამოუკიდებელი inference. TimesFM-ის model family ამ ეტაპზე დასრულებულია.

## საბოლოო აუდიტირებული შეჯამება

TimesFM pretrained decoder-only time-series foundation model-ია. Flow: global weekly calendar → per-series context → zero-shot forecast → seasonal-naive/residual/XReg candidates → calibration-only nonnegative blend → untouched final validation → audit → optional LoRA experiment → Registry packaging. v1 zero-shot `1672.2525`, v2 blend `1620.5430`, audited v3 XReg blend **`1588.8029`**; LoRA standalone `8396.0651` იყო და blend-მა weight `0` მისცა. Registry champion submission-ის Kaggle public score არის **`2742.68603`**, private — **`2853.40612`**. Public-ზე TimesFM tree models-საც უსწრებს; private/final ranking-ში იგი XGBoost/LightGBM-ის შემდეგ მესამეა და საუკეთესო non-tree submission-ია.
