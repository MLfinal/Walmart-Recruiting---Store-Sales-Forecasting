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
