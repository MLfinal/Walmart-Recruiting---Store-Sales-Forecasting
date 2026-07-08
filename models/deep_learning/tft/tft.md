# Temporal Fusion Transformer

ფოლდერი: `models/deep_learning/tft`

TFT-ზე მუშაობა დავიწყეთ როგორც deep learning ალტერნატივაზე, რომელსაც DLinear-ზე მეტი feature-ის გამოყენება შეუძლია. იდეა იყო, რომ Temporal Fusion Transformer-ს შეეძლო ერთად დაენახა historical sales, Store/Dept identity, holiday/calendar features და external covariates. ყველა შედეგს ვუყურებდით WMAE-ით, რადგან Kaggle-ის metric სწორედ WMAE-ია.

## საწყისი full-data მცდელობა

თავდაპირველად TFT გავუშვით full Store-Dept data-ზე. მოდელი თვითონ დიდი არ იყო — დაახლოებით `26.2K` trainable parameter — მაგრამ dataset ძალიან მძიმე გამოვიდა:

```text
3331 Store-Dept series
52-week encoder
39-week decoder
~1504 train batches per epoch
```

Colab-ზე ეს პრაქტიკულად ძალიან ნელი აღმოჩნდა. ერთი epoch-ის projected time დაახლოებით `10–13` წუთამდე ადიოდა, და training-ისას გამოჩნდა warning:

```text
Loss is not finite. Resetting it to 1e9
```

აქედან მივხვდით, რომ TFT-ს პირდაპირ full-data რეჟიმში გაშვება არ იყო კარგი საწყისი ნაბიჯი. ჯერ გვჭირდებოდა პატარა, სწრაფი baseline, რომ დაგვემტკიცებინა: notebook მუშაობს, W&B logging მუშაობს, checkpoint/artifact ინახება და validation WMAE ითვლება.

## fast baseline

შემდეგ baseline შევამცირეთ top active Store-Dept series-ზე:

```text
top_n_series = 300
encoder_weeks = 26
batch_size = 512
max_epochs = 5
max_time_minutes = 10
limit_train_batches = 20
limit_val_batches = 5
hidden_size = 8
attention_head_size = 1
hidden_continuous_size = 4
learning_rate = 1e-4
```

Feature set იყო მარტივი:

- `Weekly_Sales` history;
- `Store`, `Dept` static categoricals;
- `IsHoliday`;
- week/month sine-cosine calendar features.

ეს baseline წარმატებით გაეშვა და W&B-ზე დალოგა metrics, plots, prediction table, checkpoint და artifact.

W&B run:

```text
https://wandb.ai/kende23-n-a/Walmart-Recruiting---Store-Sales-Forecasting/runs/w43bg7sh
```

შედეგი top-300 subset-ზე:

```text
seasonal_naive_wmae = 6026.29
tft_baseline_wmae = 7801.90
improvement_vs_seasonal_naive_pct = -29.46%
best_val_loss = 7290.13
prediction_rows = 11700
```

ეს შედეგი ცუდი იყო, მაგრამ baseline-მა თავისი როლი შეასრულა: pipeline და W&B logging დადასტურდა.

## v1 — external covariates

შემდეგი ნაბიჯი იყო TFT-სთვის ისეთი feature-ების მიცემა, რისთვისაც ეს architecture უფრო შესაფერისია. v1-ში დავამატეთ `features.csv` და `stores.csv`:

```text
features.csv:
Temperature
Fuel_Price
MarkDown1-5
CPI
Unemployment

stores.csv:
Type
Size
```

ამავე დროს sample გავზარდეთ:

```text
top_n_series = 500
encoder_weeks = 39
hidden_size = 16
attention_head_size = 2
hidden_continuous_size = 8
max_epochs = 8
limit_train_batches = 40
```

v1 უკვე ბევრად უკეთესი იყო baseline-ზე:

```text
seasonal_naive_wmae = 4969.77
tft_v1_wmae = 6200.95
improvement_vs_seasonal_naive_pct = -24.77%
best_val_loss = 5841.39
prediction_rows = 19500
```

აქ ვისწავლეთ ორი რამ: external covariates დაეხმარა, რადგან WMAE `7801.90`-დან `6200.95`-მდე ჩამოვიდა, მაგრამ TFT მაინც ვერ აჯობა seasonal naive-ს. ანუ მოდელმა feature-ებიდან რაღაც ისწავლა, მაგრამ full sales level საკმარისად კარგად ვერ დაიჭირა.

## v2 — log target

v2-ში აღარ დაგვიმატებია ახალი feature. შევცვალეთ target-ის ფორმა. v1 პირდაპირ raw sales-ზე სწავლობდა:

```text
target = Weekly_Sales
```

v2-ში target გახდა:

```text
SalesLog = log1p(max(Weekly_Sales, 0))
```

prediction შემდეგ original scale-ზე დავაბრუნეთ:

```text
Prediction = expm1(PredictionLog)
Prediction = clip(Prediction, lower=0)
```

იდეა იყო, რომ log transform შეამსუბუქებდა scale-ის პრობლემას. training loss მართლაც ბევრად პატარა numeric scale-ზე გადავიდა, მაგრამ Kaggle WMAE გაუარესდა:

```text
seasonal_naive_wmae = 4969.77
tft_v1_wmae = 6200.95
tft_v2_wmae = 6524.68
improvement_vs_seasonal_naive_pct = -31.29%
improvement_vs_v1_pct = -5.22%
best_val_loss = 0.1205
prediction_rows = 19500
```

ამ run-მა გვაჩვენა, რომ პატარა `val_loss` არ ნიშნავს კარგ WMAE-ს. ჩვენი მთავარი metric არის original sales scale-ზე დათვლილი WMAE, და ამ metric-ით log target rejected გახდა.

## v3 — seasonal residual-ის პირველი მცდელობა

v1 და v2 ორივე full sales level-ის პროგნოზს ცდილობდა. რადგან seasonal naive ძალიან ძლიერი reference აღმოჩნდა, v3-ში შევცვალეთ ამოცანა:

```text
SeasonalNaive52 = same Store-Dept sales 52 weeks earlier
ResidualSales = Weekly_Sales - SeasonalNaive52
TFT target = ResidualSales
Final prediction = SeasonalNaive52 + PredictedResidual
```

იდეა სწორი იყო: TFT-ს აღარ უნდა ესწავლა მთელი sales level, უნდა ესწავლა მხოლოდ correction seasonal baseline-ზე.

მაგრამ პირველი v3 run invalid აღმოჩნდა. შედეგი იყო:

```text
seasonal_naive_wmae = 4969.77
tft_v3_wmae = 53035.36
improvement_vs_seasonal_naive_pct = -967.16%
improvement_vs_v1_pct = -755.28%
improvement_vs_v2_pct = -712.84%
best_val_loss = 5230.30
prediction_rows = 19500
```

ეს იმდენად ცუდი იყო, რომ output preview შევამოწმეთ. იქ გამოჩნდა მთავარი პრობლემა:

```text
SeasonalNaive52 = 0.0
```

validation rows-ზე, სადაც 52-week seasonal value აუცილებლად უნდა არსებობდეს, seasonal baseline ნული იყო. ამიტომ reconstruction რეალურად გახდა:

```text
Prediction ≈ PredictedResidual
```

იმის ნაცვლად, რომ ყოფილიყო:

```text
Prediction = real SeasonalNaive52 + PredictedResidual
```

ასე მივხვდით, რომ შედეგი არ ასახავდა residual TFT-ის ხარისხს. ეს იყო implementation bug. seasonal baseline row-wise shifted column-ით შეიქმნა და validation reconstruction-ში სწორად არ დალაგდა.

## v3 fixed notebook და სწორი residual run

ამის შემდეგ notebook გავასწორეთ. ახლა seasonal baseline იქმნება Store-Dept sales panel-იდან და არა row order-ზე დაყრდნობილი shift-ით.

სწორი ლოგიკაა:

```text
sales_panel = Store-Dept × Date matrix
SeasonalNaive52(date_t) = sales_panel(date_t - 52 weeks)
ResidualSales = Weekly_Sales - SeasonalNaive52
Prediction = SeasonalNaive52 + PredictedResidual
```

notebook-ში დამატებულია sanity checks:

```text
validation_seasonal_min
validation_seasonal_mean
validation_seasonal_max
eval_seasonal_min
eval_seasonal_mean
eval_seasonal_max
seasonal_naive_wmae_check
```

თუ validation seasonal baseline ისევ ნული გამოვა, notebook error-ს აგდებს და invalid evaluation აღარ გაგრძელდება.

fixed run-ის დროს sanity check-ებმა აჩვენა, რომ seasonal baseline უკვე რეალური იყო:

```text
validation_seasonal_min = 2702.18
validation_seasonal_mean = 53437.45
validation_seasonal_max = 241120.00
eval_seasonal_min = 2702.18
eval_seasonal_mean = 53437.45
eval_seasonal_max = 241120.00
seasonal_naive_wmae_check = 4969.77
```

ეს უკვე სწორი evaluation იყო, რადგან validation-ში `SeasonalNaive52` აღარ იყო ნული. W&B run:

```text
https://wandb.ai/kende23-n-a/Walmart-Recruiting---Store-Sales-Forecasting/runs/nqjm85gh
```

შედეგი:

```text
seasonal_naive_wmae = 4969.77
tft_v3_fixed_wmae = 5212.71
improvement_vs_seasonal_naive_pct = -4.89%
improvement_vs_v1_pct = +15.94%
improvement_vs_v2_pct = +20.11%
best_val_loss = 5230.30
prediction_rows = 19500
```

ამ run-მა გვაჩვენა, რომ residual idea რეალურად მუშაობს უკეთ, ვიდრე v1 და v2. WMAE `6200.95`-დან `5212.71`-მდე ჩამოვიდა v1-თან შედარებით და `6524.68`-დან `5212.71`-მდე v2-თან შედარებით. მაგრამ seasonal naive მაინც უკეთესია: `4969.77` vs `5212.71`. ანუ TFT-მ ისწავლა useful correction, მაგრამ correction ზედმეტად აგრესიულია და seasonal baseline-ს ბოლომდე ვერ აჯობა.

ამჟამინდელი `model_experiment_TFT.ipynb` არის fixed residual version. ძველი invalid v3 შედეგი README-ში დარჩა როგორც training story-ის ნაწილი, რადგან ზუსტად იმ output-მა გვაჩვენა implementation bug.

## შედეგების მოკლე ცხრილი

| Run | Subset | იდეა | WMAE | დასკვნა |
|---|---:|---|---:|---|
| seasonal naive | top 300 | 52-week lookup | 6026.29 | reference |
| baseline | top 300 | small TFT, calendar only | 7801.90 | pipeline/logging OK, model weak |
| seasonal naive | top 500 | 52-week lookup | 4969.77 | reference |
| v1 | top 500 | raw sales + external covariates | 6200.95 | external covariates დაეხმარა baseline-თან შედარებით |
| v2 | top 500 | log target + external covariates | 6524.68 | worse than v1 |
| v3 invalid | top 500 | residual, broken seasonal base | 53035.36 | invalid implementation |
| v3 fixed | top 500 | residual + correct 52-week lookup | 5212.71 | best valid TFT so far, მაგრამ seasonal naive-ზე უარესი |

ამ ეტაპზე საუკეთესო valid TFT არის v3 fixed. მთავარი გაკვეთილი ასეთია: TFT-სთვის full sales-ის სწავლა რთული აღმოჩნდა, log target-მა WMAE გააუარესა, ხოლო seasonal residual-მა მოდელი ბევრად დააახლოვა seasonal naive-სთან. საბოლოო პრობლემა დარჩა ის, რომ residual correction ჯერ კიდევ ზედმეტ შეცდომას ამატებს და reference baseline-ს `4.89%`-ით ჩამორჩება.
