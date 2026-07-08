# Temporal Fusion Transformer

ფოლდერი: `models/deep_learning/tft`

ამ ეტაპზე შექმნილია პირველი TFT baseline notebook:

```text
baseline_tft.ipynb
```

Baseline-ის მიზანია მივიღოთ პირველი leakage-safe TFT შედეგი იგივე protocol-ით, რასაც DLinear-ში ვიყენებდით:

- validation არის `train.csv`-ის ბოლო 39 კვირა;
- metric არის Kaggle-style WMAE original sales scale-ზე;
- W&B-ზე ილოგება config, training curves, validation WMAE, plots, prediction table, checkpoint და summary artifact;
- MLflow არ გამოიყენება.

## პირველი full-data მცდელობა

პირველი TFT baseline გავუშვით full Store-Dept data-ზე, მაგრამ Colab-ზე პრაქტიკულად ძალიან ნელი აღმოჩნდა.

მთავარი პრობლემა model size არ იყო. model-ს ჰქონდა დაახლოებით `26.2K` trainable parameters, რაც პატარაა. პრობლემა იყო sliding-window dataset:

```text
3331 Store-Dept series
52-week encoder
39-week decoder
~1504 train batches per epoch
```

Colab run-ში პირველი epoch-ის პროგრესი ძალიან ნელა მიდიოდა და projected time ერთ epoch-ზე დაახლოებით `10–13` წუთი იყო. მომხმარებლის დაკვირვებით run-მა დაახლოებით `1` საათი წაიღო მხოლოდ რამდენიმე batch/progress step-ზე, ამიტომ full-data baseline ამ ფორმით არ არის მისაღები.

ასევე გამოჩნდა warning:

```text
Loss is not finite. Resetting it to 1e9
```

ეს ნიშნავს, რომ raw target/normalization/training setup ზოგ batch-ზე unstable იყო. ამიტომ baseline notebook შევცვალეთ ისე, რომ ჯერ დაადასტუროს pipeline და W&B logging, და არა full-scale TFT performance.

## განახლებული fast baseline

ახლანდელი `baseline_tft.ipynb` არის Colab-safe sanity baseline. მიზანია მაქსიმუმ დაახლოებით `10` წუთში მივიღოთ პირველი TFT result და W&B logs.

ძირითადი ცვლილებები:

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

ასევე `Weekly_Sales` target იჭრება `>= 0`:

```text
Weekly_Sales = clip(lower=0)
```

ეს ამცირებს non-finite loss-ის რისკს და შეესაბამება submission logic-ს, რადგან საბოლოო პროგნოზიც არ უნდა იყოს negative.

მნიშვნელოვანი caveat:

```text
ეს baseline უკვე აღარ არის full-data final TFT score.
ეს არის fast TFT pipeline/logging baseline.
```

შედეგი უნდა შევადაროთ ფრთხილად: DLinear/XGBoost full validation-ზეა, ხოლო fast TFT baseline მხოლოდ top active Store-Dept series-ზე train/validate ხდება. Full-data ან larger-sample TFT უნდა გაკეთდეს მხოლოდ მაშინ, თუ fast baseline მუშაობს სტაბილურად და W&B logging სწორია.

Baseline feature set:

- target history: `Weekly_Sales`;
- static categoricals: `Store`, `Dept`;
- known future categorical: `IsHoliday`;
- known future calendar reals: `time_idx`, week sine/cosine, month sine/cosine.

External covariates (`features.csv`, `stores.csv`) ამ baseline-ში შეგნებულად არ არის დამატებული. ისინი უნდა დაემატოს შემდეგ TFT experiment-ში, baseline result-ის შემდეგ.

## Baseline result

Fast baseline წარმატებით გაეშვა და W&B logging/artifacts შეიქმნა.

W&B run:

https://wandb.ai/kende23-n-a/Walmart-Recruiting---Store-Sales-Forecasting/runs/w43bg7sh

Run setup:

```text
top_n_series = 300
encoder_weeks = 26
hidden_size = 8
trainable params = 8.4K
train_batches_total = 38
validation_batches_total = 1
max_epochs = 5
```

Validation result იმავე top-300 subset-ზე:

```text
seasonal_naive_wmae = 6026.29
tft_baseline_wmae = 7801.90
improvement_vs_seasonal_naive_pct = -29.46%
best_val_loss = 7290.13
best_checkpoint = epoch 3
prediction_rows = 11700
```

Interpretation:

- baseline-ის მთავარი მიზანი იყო runtime/logging/pipeline validation და არა final score;
- model გაუშვა, checkpoint შეინახა, W&B-ზე metrics/plots/prediction table/artifact დალოგდა;
- WMAE seasonal naive-ზე ბევრად უარესია, ამიტომ ეს baseline rejected როგორც predictive model;
- პრობლემა მოსალოდნელია: ძალიან პატარა TFT, ცოტა training batches, მხოლოდ top-300 sample, და ჯერ external covariates არ გვაქვს.

ამ ეტაპზე baseline დასრულებულია. საჭირო არ არის მის დამატებით tuning-ზე დროის დახარჯვა. შემდეგი სწორი ნაბიჯია v1 experiment, სადაც TFT-ს მივცემთ იმ known covariates-ს, რისთვისაც ეს architecture უკეთესად არის შექმნილი.

## v1: external covariates

ფაილი:

```text
model_experiment_TFT.ipynb
```

v1 baseline-ისგან განსხვავდება ასე:

```text
top_n_series: 300 → 500
encoder_weeks: 26 → 39
hidden_size: 8 → 16
attention_head_size: 1 → 2
hidden_continuous_size: 4 → 8
max_epochs: 5 → 8
max_time_minutes: 10 → 20
limit_train_batches: 20 → 40
```

v1-ში დამატებული known/static features:

```text
features.csv:
- Temperature
- Fuel_Price
- MarkDown1
- MarkDown2
- MarkDown3
- MarkDown4
- MarkDown5
- CPI
- Unemployment

stores.csv:
- Type
- Size
```

Feature logic:

- `MarkDown1-5` missing values ივსება `0.0`-ით, რადგან markdown-ის არარსებობა ხშირად ნიშნავს promotion value არ გვაქვს;
- `Temperature`, `Fuel_Price`, `CPI`, `Unemployment` ივსება Store-level forward/backward fill-ით, შემდეგ median fallback-ით;
- `Type` არის static categorical;
- `Size` არის known/static real covariate;
- target ისევ იჭრება `Weekly_Sales >= 0`, რომ non-finite/negative prediction პრობლემები შემცირდეს.

v1-ის კითხვა:

```text
ეხმარება თუ არა TFT-ს external known covariates enough, რომ fast baseline-ზე და seasonal naive-ზე უკეთესი გახდეს?
```

თუ v1 მაინც seasonal naive-ზე უარესია, შემდეგი ნაბიჯი არ უნდა იყოს full-data expensive training. ჯერ უნდა შევცვალოთ target normalization/loss strategy ან sample/window strategy.

### v1 result

v1 წარმატებით გაეშვა და W&B/artifacts შეინახა.

Run setup:

```text
top_n_series = 500
encoder_weeks = 39
hidden_size = 16
attention_head_size = 2
trainable params = 41.1K
train_batches_total = 65
effective_train_batches_per_epoch = 40
validation_batches_total = 1
max_epochs = 8
best_checkpoint = epoch 7
best_val_loss = 5841.39
```

Validation result იმავე top-500 subset-ზე:

```text
seasonal_naive_wmae = 4969.77
tft_v1_wmae = 6200.95
improvement_vs_seasonal_naive_pct = -24.77%
prediction_rows = 19500
```

Interpretation:

- v1 baseline-ზე უკეთესია: `7801.90 → 6200.95`;
- external covariates-მა აშკარად დაეხმარა, მაგრამ seasonal naive-ს მაინც ვერ აჯობა;
- TFT ჯერ კიდევ ვერ სწავლობს საკმარისად კარგ absolute sales scale-ს;
- რადგან v1 top-500 subset-ზეც seasonal naive-ზე უარესია, full-data expensive training ჯერ არ არის გამართლებული.

## v2: log target strategy

ფაილი:

```text
model_experiment_TFT_v2.ipynb
```

v2-ის მიზანია არა ახალი covariates დამატება, არამედ target/training stability-ის გამოსწორება.

v1 raw target-ზე სწავლობდა:

```text
target = Weekly_Sales
```

v2 სწავლობს log-transformed target-ზე:

```text
target = SalesLog = log1p(max(Weekly_Sales, 0))
```

შემდეგ validation prediction ბრუნდება original scale-ზე:

```text
prediction = expm1(prediction_log)
prediction = clip(prediction, lower=0)
```

WMAE ისევ ითვლება original `Weekly_Sales` scale-ზე, ამიტომ metric რჩება Kaggle-compatible.

v2 keeps:

```text
top_n_series = 500
encoder_weeks = 39
features.csv covariates
stores.csv covariates
same 39-week validation
```

v2 changes:

```text
target strategy: raw sales → log1p sales
max_epochs: 8 → 10
limit_train_batches: 40 → 50
max_time_minutes: 20 → 25
```

v2-ის კითხვა:

```text
თუ target scale უფრო სტაბილური გახდა, შეძლებს თუ არა TFT seasonal naive-სთან მიახლოებას ან მის გადასწრებას?
```

Decision rule:

- თუ v2 ისევ seasonal naive-ზე უარესია, შემდეგი ნაბიჯი უნდა იყოს sample/window strategy ან TFT-ის შეჩერება;
- თუ v2 v1-ს ძლიერად აჯობებს, შემდეგ v3-ში შეიძლება top series count/training budget გავზარდოთ.

### v2 result

v2 წარმატებით გაეშვა. ამ run-ში `model_experiment_TFT.ipynb` უკვე v2 notebook-ად არის შენახული.

Run setup:

```text
top_n_series = 500
encoder_weeks = 39
target = SalesLog = log1p(max(Weekly_Sales, 0))
hidden_size = 16
attention_head_size = 2
trainable params = 41.1K
max_epochs = 10
best_checkpoint = epoch 7
best_val_loss = 0.1205
```

Validation result იმავე top-500 subset-ზე:

```text
seasonal_naive_wmae = 4969.77
tft_v1_wmae = 6200.95
tft_v2_wmae = 6524.68
improvement_vs_seasonal_naive_pct = -31.29%
improvement_vs_v1_pct = -5.22%
prediction_rows = 19500
```

Interpretation:

- log target-მა training loss numeric scale-ზე გაალამაზა (`best_val_loss = 0.1205`), მაგრამ Kaggle WMAE original scale-ზე გააუარესა;
- v2 v1-ზე `5.22%`-ით უარესია;
- v2 seasonal naive-ზე `31.29%`-ით უარესია;
- ამიტომ log target rejected ამ setup-ში;
- TFT-ის პრობლემა მხოლოდ raw target scale არ ყოფილა. მთავარი პრობლემა უფრო likely არის model-selection objective, WMAE weighting, და ის, რომ TFT მთელ forecast level-ს სწავლობს, როცა seasonal naive უკვე ძალიან ძლიერი referenceა.

ამ ეტაპზე საუკეთესო TFT run არის v1, მაგრამ ისიც seasonal naive-ს ვერ ჯობნის:

```text
best_observed_tft = v1
best_observed_tft_wmae = 6200.95
seasonal_naive_on_same_subset = 4969.77
```

## შემდეგ რა უნდა ვცადოთ TFT-ში

TFT-ის ცუდი შედეგი ჯერ არ ნიშნავს, რომ architecture მთლიანად useless არის. უფრო ზუსტი დასკვნაა: ჩვენი current controlled TFT setup ჯერ ვერ ჯობნის simple seasonal reference-ს. მიზეზები სავარაუდოდ არის:

- training sample შეზღუდულია top 300/500 series-ზე;
- batches/epochs ხელოვნურად შეზღუდულია Colab runtime-ის გამო;
- TFT raw/log target-ზე ჯერ ვერ სწავლობს Store-Dept scale-ს კარგად;
- WMAE holiday weighting loss-ში პირდაპირ არ არის ჩაშენებული ისე, როგორც DLinear-ში გვქონდა;
- model selection ხდება PyTorch Forecasting `val_loss`-ით, ხოლო final metric არის ჩვენი custom WMAE original scale-ზე.

ყველაზე აზრიანი შემდეგი მცდელობები:

1. **v3: weighted validation/selection fix**  
   checkpoint selection არ უნდა ენდობოდეს მხოლოდ `val_loss`-ს. v2-მ კარგად აჩვენა ეს: `val_loss` პატარა იყო, მაგრამ WMAE ცუდი გამოვიდა. უნდა დავამატოთ training-ის შემდეგ რამდენიმე saved checkpoint-ის custom WMAE comparison ან callback, რომელიც best model-ს Kaggle WMAE-ით აირჩევს.

2. **v4: stronger seasonal baseline hybrid**  
   TFT-ს ვასწავლოთ არა raw sales, არამედ seasonal naive residual:

   ```text
   target = Weekly_Sales - sales_52_weeks_ago
   prediction = seasonal_naive + predicted_residual
   ```

   Walmart-ში yearly seasonality ძლიერია. TFT შეიძლება უკეთ სწავლობდეს correction-ს, ვიდრე მთელ sales level-ს.

3. **v5: bigger but still controlled training**  
   მხოლოდ თუ v3/v4-ში გაუმჯობესება ჩანს:

   ```text
   top_n_series = 1000
   limit_train_batches = 100+
   max_epochs = 15-20
   ```

ამ ორი გზიდან უკეთესი შემდეგი ნაბიჯია **seasonal residual TFT**. მიზეზი: v1/v2 ორივე ვერ სწავლობს full sales level-ს ისე კარგად, როგორც 52-week seasonal naive. ამიტომ უფრო სწორია, TFT-ს ვასწავლოთ correction seasonal naive-ზე.

თუ seasonal residual TFT-იც seasonal naive-ზე უარესია, TFT უნდა დავხუროთ როგორც non-competitive model და inference არ უნდა დავწეროთ.
