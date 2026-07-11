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
