# DLinear

ფოლდერი: `models/deep_learning/DLinear`

მიზანი არის DLinear-ის ეტაპობრივი განვითარება: ჯერ სუფთა baseline, შემდეგ მცირე კონტროლირებული ცვლილებები. ყველა run ფასდება იგივე წესით: ბოლო 39 კვირა validation-ად და Kaggle-ის WMAE, სადაც holiday row-ის წონაა 5, დანარჩენის 1.

## რატომ DLinear

DLinear არის მარტივი deep learning time-series მოდელი. ის input sequence-ს ყოფს ორ ნაწილად:

- `trend` — moving average-ით მიღებული გლუვი ნაწილი;
- `seasonal/residual` — input minus trend.

შემდეგ ორივე ნაწილი linear layer-ით გადადის forecast horizon-ზე და ჯამდება. ჩვენთვის ეს არის კარგი neural baseline, რადგან XGBoost/LightGBM-ისგან განსხვავებით ჯერ მხოლოდ historical sales sequence-ს უყურებს და არა ბევრ tabular feature-ს.

## Notebook-ების საერთო სტრუქტურა

ორივე notebook ერთნაირი ლოგიკითაა აგებული:

1. install/import/config  
   ყენდება PyTorch/W&B, ფიქსირდება seed, ირჩევა device, იწერება `CONFIG`. მთავარი პარამეტრებია `validation_weeks=39`, `input_weeks`, learning rate, patience, W&B project/entity.

2. data loading  
   Drive-იდან იტვირთება `train.csv` და `test.csv`. მოთხოვნილი columns მოწმდება: `Store`, `Dept`, `Date`, `Weekly_Sales`, `IsHoliday`.

3. weekly panel  
   `train.csv` გარდაიქმნება matrix-ად: row = `(Store, Dept)`, column = weekly date, value = `Weekly_Sales`. DLinear fixed-length sequence-ს ითხოვს, ამიტომ ეს panel არის მთავარი data format. missing sales baseline-ში ივსება `0.0`-ით.

4. split  
   ბოლო 39 კვირა არის validation. დანარჩენი ისტორია გამოიყენება training windows-ის შესაქმნელად. ეს leakage-safe split-ია, რადგან validation future period training-ში არ ხვდება.

5. metric/loss  
   `wmae()` ითვლის Kaggle metric-ს original sales scale-ზე. `weighted_mae_loss()` იგივე holiday weights-ს იყენებს normalized target-ზე training-ის დროს.

6. datasets  
   `WindowDataset` ქმნის sliding windows-ს training-ისთვის: past `input_weeks` → next 39 weeks. აბრუნებს `x`, `y`, `weights`, `mean`, `std`, `series_idx` და საჭიროებისას future calendar features-ს.  
   `ValidationDataset` თითო Store-Dept სერიაზე ქმნის ერთ example-ს: validation-მდე არსებული ბოლო `input_weeks` → ბოლო 39 validation კვირა.

7. model classes  
   `MovingAverage` აკეთებს trend smoothing-ს.  
   `SeriesDecomposition` აბრუნებს `(seasonal, trend)`.  
   `DLinear` ან მისი experiment version forecast-ს აგებს trend/seasonal linear projections-ით.  
   experiment notebook-ში ემატება `series_bias` ან calendar branch.

8. train/evaluate  
   train loop აკეთებს forward/backward/gradient clipping-ს. `evaluate_model()` აბრუნებს predictions original scale-ზე და ითვლის validation WMAE-ს. `ReduceLROnPlateau` ამცირებს learning rate-ს, early stopping აჩერებს run-ს თუ validation აღარ უმჯობესდება.

9. W&B/artifacts  
   W&B-ზე ილოგება config, epoch metrics, validation WMAE, improvement percentages, prediction table, scatter plot, checkpoint და summary JSON.

## შედეგები

| Run | Input | დამატება | Best epoch | Validation WMAE | vs seasonal naive | vs previous |
|---|---:|---|---:|---:|---:|---:|
| Seasonal naive | 52w lookup | იგივე კვირა 1 წლით ადრე | — | 1604.27 | — | — |
| Baseline | 52w | pure DLinear | 8 | 1523.21 | +5.05% | — |
| v1 | 52w | Store-Dept calibration | 11 | 1506.28 | +6.11% | +1.11% |
| v2 | 65w | free calendar branch | 36 | 1961.45 | -22.27% | -30.22% |
| v3 | 52w | gated calendar branch | 10 | 1511.97 | +5.75% | +22.92% vs v2 |

შენიშვნა: v1-ის იდეა იყო უფრო გრძელი context, მაგრამ რეალურად 104-week context ამ split-ზე შეუძლებელია training windows-ისთვის: pre-validation history არის 104 კვირა, ხოლო target horizon არის 39 კვირა. საჭიროა `input_weeks + 39 <= 104`. ამიტომ working v1 უნდა ჩაითვალოს როგორც 52-week DLinear + Store-Dept calibration.

## Train setup, რომელიც ყველა run-ში ერთნაირია

ყველა DLinear run-ში ერთი და იგივე evaluation protocol გვაქვს:

- validation: ბოლო 39 კვირა `train.csv`-დან;
- target horizon: 39 კვირა;
- metric: WMAE original sales scale-ზე;
- holiday weight: `5.0`;
- non-holiday weight: `1.0`;
- Store-Dept სერიების რაოდენობა: `3331`;
- seasonal naive reference: იგივე Store-Dept გაყიდვა 52 კვირით ადრე;
- W&B-ში ილოგება: `train/normalized_wmae_loss`, `validation/normalized_wmae_loss`, `validation/wmae`, improvement percentages, learning rate, prediction table, scatter plot, checkpoint, summary JSON.

Data flow notebook-ში ასეთია:

```text
train.csv
→ sort by Store/Dept/Date
→ pivot Store-Dept × Date panel
→ split: fit dates + last 39 validation dates
→ WindowDataset / ValidationDataset
→ DLinear model
→ evaluate_model()
→ W&B metrics + artifacts
```

## Baseline

ფაილი: `baseline_dlinear.ipynb`

Baseline-ში model input არის მხოლოდ past sales:

```text
past 52 Weekly_Sales
→ per-window normalization
→ DLinear
→ 39-week forecast
```

`IsHoliday` baseline-ში model feature არ არის. ის გამოიყენება მხოლოდ WMAE loss/metric weights-ში. შედეგი: `1523.21` WMAE, რაც seasonal naive-ზე 5.05%-ით უკეთესია. ეს ადასტურებს, რომ DLinear historical sequence-დან სასარგებლო pattern-ს სწავლობს.

## v1: Store-Dept calibration

ფაილი: `model_experiment_DLinear.ipynb`

v1-ის მიზანი იყო გაგვერკვია: თუ DLinear-ს დავუმატებთ კონკრეტული Store-Dept სერიის bias correction-ს, historical sequence baseline გაუმჯობესდება თუ არა.

### v1 config/logs

| ველი | მნიშვნელობა |
|---|---:|
| `validation_weeks` | `39` |
| effective `input_weeks` | `52` |
| `batch_size` | `512` |
| `learning_rate` | `8e-4` |
| `weight_decay` | `2e-4` |
| `series_bias_weight_decay` | `1e-3` |
| `moving_avg_kernel` | `25` |
| training windows | `46634` |
| validation series | `3331` |
| seasonal naive WMAE | `1604.27` |
| best epoch | `11` |
| early stopping | epoch `23` |
| best validation WMAE | `1506.28` |

best epoch-ის ძირითადი log:

```text
epoch = 11
train/normalized_wmae_loss = 2.3409955
validation/normalized_wmae_loss = 0.6877526
validation/wmae = 1506.2825
validation/improvement_vs_seasonal_naive_pct = 6.1079
improvement_vs_baseline_pct = 1.1113
```

### v1-ში რა feature/input გვქონდა

Model feature-ები:

- `Weekly_Sales` history: ბოლო 52 კვირა თითო Store-Dept სერიაზე;
- `series_idx`: integer id თითოეული `(Store, Dept)` სერიისთვის;
- per-window `mean/std`: normalization-ისთვის.

Feature არ იყო:

- `IsHoliday` როგორც model input;
- week/month calendar;
- markdowns;
- CPI/Fuel/Unemployment;
- Store type/size.

`IsHoliday` მხოლოდ weights-ში გამოიყენებოდა:

```text
holiday week → weight 5
normal week  → weight 1
```

### v1-ში როგორ დავამატეთ Store-Dept calibration

Baseline DLinear აკეთებს:

```text
past sales → decomposition(trend, seasonal) → linear forecast → 39-week forecast
```

v1-ში დაემატა `series_bias`:

```text
series_idx → Embedding(n_series, pred_len)
```

ანუ თითო Store-Dept სერიას აქვს 39-ზომიანი learnable vector — თითო correction forecast horizon-ის თითო კვირისთვის.

ლოგიკა:

```text
past sales → DLinear forecast
series_idx → series_bias[series_idx]
forecast = DLinear forecast + series_bias
```

რატომ: Walmart-ში Store-Dept სერიებს განსხვავებული scale და systematic bias აქვთ. per-window normalization scale-ს ამცირებს, მაგრამ კონკრეტული department/store-ის მუდმივ გადახრას სრულად ვერ იჭერს.

### v1 result interpretation

v1-მა baseline გააუმჯობესა:

```text
1523.21 → 1506.28
absolute improvement = 16.93 WMAE
relative improvement = 1.11%
```

ეს არ არის დიდი ნახტომი, მაგრამ სწორი signal-ია: Store-Dept identity correction ეხმარება. ამავე დროს, v1 მაინც მხოლოდ target history-ზე დგას, ამიტომ promotion/calendar/economic effects-ს ვერ ხედავს.

## v2: calendar features

v2-ის მიზანი იყო გაგვერკვია: known future calendar features გააუმჯობესებს თუ არა holiday/seasonal კვირების პროგნოზს.

თავიდან ვცადეთ `input_weeks=104`, მაგრამ ამ split-ზე ეს შეუძლებელია. Colab-ში მივიღეთ:

```text
ValueError: Not enough history for configured windows.
```

მიზეზი:

```text
total train weeks = 143
validation weeks = 39
pre-validation history = 143 - 39 = 104
needed for one training window = input_weeks + prediction_weeks
104 + 39 = 143 > 104
max_start = 104 - 104 - 39 = -39
```

ანუ validation-მდე არსებულ history-ში ვერ იქმნება ასეთი training pair:

```text
past 104 weeks → next 39 weeks
```

### v2 config/logs

W&B run:

https://wandb.ai/kende23-n-a/Walmart-Recruiting---Store-Sales-Forecasting/runs/1e085ojw

| ველი | მნიშვნელობა |
|---|---:|
| `validation_weeks` | `39` |
| `input_weeks` | `65` |
| `batch_size` | `512` |
| `learning_rate` | `7e-4` |
| `weight_decay` | `2e-4` |
| `series_bias_weight_decay` | `1e-3` |
| `calendar_weight_decay` | `1e-4` |
| `moving_avg_kernel` | `25` |
| training windows | `3331` |
| validation series | `3331` |
| seasonal naive WMAE | `1604.27` |
| best epoch | `36` |
| early stopping | epoch `50` |
| best validation WMAE | `1961.45` |

summary log:

```text
experiment = dlinear_v2_104w_series_calendar
best_epoch = 36
best_validation_wmae = 1961.4508
baseline_dlinear_39w_wmae = 1523.2097
dlinear_v1_104w_calibration_wmae = 1506.2825
seasonal_naive_wmae = 1604.2697
improvement_vs_v1_pct = -30.2180
```

ბოლო epoch-ების behavior:

```text
epoch 46 validation/wmae = 1962.99
epoch 47 validation/wmae = 1963.99
epoch 48 validation/wmae = 1963.63
epoch 49 validation/wmae = 1963.18
epoch 50 validation/wmae = 1963.96
early stopping: best epoch 36
```

### v2-ში ზუსტად რა features დავამატეთ

v2-ში DLinear-ს პირველად მივეცით future calendar covariates. თითო forecast კვირისთვის შეიქმნა 6 feature:

- `IsHoliday`;
- `week_sin`, `week_cos`;
- `month_sin`, `month_cos`;
- `horizon_position`.

feature construction:

```text
is_holiday = 0/1
week_sin = sin(2π * ISO_week / 52)
week_cos = cos(2π * ISO_week / 52)
month_sin = sin(2π * month / 12)
month_cos = cos(2π * month / 12)
horizon_position = linear position inside available dates
```

რატომ sin/cos: week/month ციკლურია. მაგალითად week 52 და week 1 ერთმანეთთან ახლოსაა, მაგრამ უბრალო integer encoding-ით შორს გამოჩნდებოდა.

რატომ leakage-safe: ეს features ცნობილია prediction-მდე, რადგან მომავალ test dates და holiday flag უკვე გვაქვს `test.csv`-ში.

### v2-ში როგორ დაემატა calendar branch

Dataset-ში `future_calendar` დაემატა target window-ის იგივე თარიღებზე:

```text
target_positions = forecast weeks
future_calendar = calendar_features[target_positions]
```

Model-ში calendar ნაწილი იყო small MLP:

```text
future_calendar[6]
→ Linear(6, 16)
→ ReLU
→ Linear(16, 1)
→ calendar_adjustment per forecast week
```

საბოლოო forecast:

```text
DLinear forecast
+ series_bias
+ calendar_adjustment
```

### v2 result interpretation

v2 მკვეთრად გაუარესდა:

```text
v1: 1506.28
v2: 1961.45
change = -30.22% vs v1
```

ეს არ ნიშნავს, რომ calendar features ცუდია. უფრო სწორი დასკვნაა, რომ v2 implementation იყო ზედმეტად აგრესიული:

1. `input_weeks=65`-მა training windows შეამცირა  
   pre-validation history არის 104 კვირა. ფორმულა:

   ```text
   windows_per_series = 104 - input_weeks - 39 + 1
   ```

   v1/baseline-ში:

   ```text
   104 - 52 - 39 + 1 = 14 windows per series
   14 * 3331 = 46634 train windows
   ```

   v2-ში:

   ```text
   104 - 65 - 39 + 1 = 1 window per series
   1 * 3331 = 3331 train windows
   ```

   ანუ v2-ს ჰქონდა დაახლოებით 14-ჯერ ნაკლები training examples.

2. `calendar_adjustment` პირდაპირ ემატებოდა forecast-ს  
   calendar branch-ს არ ჰქონდა gate/scale control. ამიტომ model-ს შეეძლო calendar correction ზედმეტად გაეზარდა და დაეზიანებინა უკვე კარგი DLinear + series_bias forecast.

3. train loss უმჯობესდებოდა, validation კი ცუდი იყო  
   ეს მიუთითებს არა უბრალოდ undertraining-ზე, არამედ ცუდ generalization-ზე validation horizon-ზე.

დასკვნა: v2 არის failed ablation, მაგრამ სასარგებლო. მან გვაჩვენა, რომ calendar signal უნდა დაემატოს კონტროლით და training-window რაოდენობა არ უნდა შევამციროთ.

## v3: gated calendar residual

v3-ის მიზანი იყო v2-ის failure გაგვესწორებინა ისე, რომ calendar features მთლიანად არ გადაგვეგდო.

### v3 config/logs

W&B run:

https://wandb.ai/kende23-n-a/Walmart-Recruiting---Store-Sales-Forecasting/runs/pru2kfaq

| ველი | მნიშვნელობა |
|---|---:|
| `validation_weeks` | `39` |
| `input_weeks` | `52` |
| `batch_size` | `512` |
| `learning_rate` | `6e-4` |
| `weight_decay` | `2e-4` |
| `series_bias_weight_decay` | `1e-3` |
| `calendar_weight_decay` | `1e-4` |
| `calendar_gate_l1` | `1e-4` |
| `moving_avg_kernel` | `25` |
| training windows per series | `14` |
| total training windows | `46634` |
| validation series | `3331` |
| seasonal naive WMAE | `1604.27` |
| best epoch | `10` |
| early stopping | epoch `24` |
| best validation WMAE | `1511.97` |
| best calendar gate | `0.05366` |

best epoch-ის log:

```text
epoch = 10
train/normalized_wmae_loss = 2.3617122
validation/normalized_wmae_loss = 0.6861886
validation/wmae = 1511.9733
validation/improvement_vs_seasonal_naive_pct = 5.7532
validation/improvement_vs_baseline_pct = 0.7377
validation/improvement_vs_v1_pct = -0.3778
model/calendar_gate = 0.05366
learning_rate = 0.0006
```

W&B summary:

```text
best_epoch = 10
best_validation_wmae = 1511.97327
baseline_dlinear_39w_wmae = 1523.20972
dlinear_v1_calibration_wmae = 1506.28247
dlinear_v2_calendar_wmae = 1961.45081
best_improvement_vs_v1_pct = -0.3778
final_calendar_gate = 0.05366
```

Run history-დან:

- `validation/wmae` დაეცა `1626.93`-დან `1511.97`-მდე;
- best result მივიღეთ epoch `10`-ზე;
- epoch 10-ის შემდეგ train loss მცირდებოდა, მაგრამ validation აღარ გაუმჯობესდა;
- learning rate შემცირდა `0.0006 → 0.0003 → 0.00015`;
- `calendar_gate` გაიზარდა, მაგრამ best checkpoint-ში პატარა დარჩა (`0.05366`).

### v3-ში რა შეიცვალა v2-თან შედარებით

1. დავაბრუნეთ `input_weeks=52`

v2:

```text
input_weeks = 65
windows_per_series = 104 - 65 - 39 + 1 = 1
total windows = 3331
```

v3:

```text
input_weeks = 52
windows_per_series = 104 - 52 - 39 + 1 = 14
total windows = 46634
```

ამით v3-ს დაუბრუნდა საკმარისი training data.

2. calendar branch გავხადეთ gated

v2:

```text
forecast = DLinear forecast + series_bias + calendar_adjustment
```

v3:

```text
forecast = DLinear forecast + series_bias + tanh(calendar_gate) * calendar_adjustment
```

`calendar_gate` იწყება 0-დან:

```text
tanh(0) = 0
```

ამიტომ training-ის დასაწყისში model თითქმის v1-ს ჰგავს:

```text
forecast ≈ DLinear forecast + series_bias
```

calendar feature-ები model-ზე გავლენას იღებს მხოლოდ მაშინ, თუ training gate-ს გაზრდის.

3. gate-ს დავამატეთ პატარა regularization

```text
loss = weighted_mae_loss + calendar_gate_l1 * abs(calendar_gate)
```

ეს calendar correction-ს აკავებს, თუ ის რეალურად არ აუმჯობესებს loss-ს.

### v3-ში features

v3-ში calendar features იგივე დარჩა რაც v2-ში:

- `is_holiday`;
- `week_sin`;
- `week_cos`;
- `month_sin`;
- `month_cos`;
- `horizon_position`.

განსხვავება feature list-ში არ არის; განსხვავება არის how strongly model can use those features.

v3 calendar path:

```text
future_calendar[6]
→ Linear(6, 16)
→ ReLU
→ Linear(16, 1)
→ calendar_adjustment
→ tanh(calendar_gate) * calendar_adjustment
```

### v3 result interpretation

v3 ბევრად უკეთესია v2-ზე:

```text
v2: 1961.45
v3: 1511.97
absolute improvement = 449.48 WMAE
relative improvement = 22.92%
```

მიზეზი:

- v3-ში training windows დაბრუნდა `3331 → 46634`;
- calendar correction აღარ ემატება uncontrolled ფორმით;
- gate პატარა დარჩა, ამიტომ calendar branch forecast-ს მხოლოდ მსუბუქად ასწორებს.

v3 baseline-ზე უკეთესია:

```text
baseline: 1523.21
v3: 1511.97
relative improvement = 0.74%
```

მაგრამ v3 ჯერ კიდევ ოდნავ უარესია v1-ზე:

```text
v1: 1506.28
v3: 1511.97
difference vs v1 = -0.38%
```

სწორი დასკვნა:

- v3-მა v2-ის პრობლემა რეალურად გაასწორა;
- calendar features ამ gated ფორმით უსაფრთხოა, მაგრამ ჯერ არ სჯობს v1 calibration-only model-ს;
- ამ ეტაპზე საუკეთესო DLinear run კვლავ v1-ია (`1506.28`), ხოლო v3 არის კარგი corrected calendar experiment.

## XGBoost-თან შედარება

XGBoost იყენებს ძლიერ tabular feature engineering-ს: calendar, store/dept metadata, lags, aggregates და სხვა. DLinear ამ ეტაპზე ძირითადად sequence model-ია. ამიტომ XGBoost-ის უპირატესობა მოსალოდნელია, რადგან Walmart sales heavily depends on store/dept identity, holidays, markdowns და სხვა known/external signals.

DLinear-ის ამ ეტაპის პროგრესი:

- pure sequence baseline მუშაობს;
- series calibration აუმჯობესებს;
- naive calendar branch აზიანებს;
- შემდეგი საჭირო ნაბიჯია controlled calendar ან richer covariates, მაგრამ აუცილებლად ablation-ებით.

## შემდეგი v4: Store/Dept embeddings

v4-ში calendar features დროებით ამოვიღეთ, რადგან v3-მ v1 ვერ აჯობა. ვტესტავთ იმ მიმართულებას, რომელმაც უკვე გაამართლა: identity signal.

v4 architecture:

```text
past 52 Weekly_Sales
→ DLinear forecast

series_idx
→ Store-Dept series_bias

Store id → store_embedding
Dept id  → dept_embedding
concat(store_embedding, dept_embedding)
→ identity_head MLP
→ 39-week identity_adjustment

forecast = DLinear forecast + series_bias + identity_adjustment
```

რა შეიცვალა v3-თან შედარებით:

- calendar branch ამოღებულია;
- `input_weeks=52` რჩება, ანუ ისევ გვაქვს `46634` training windows;
- `series_bias` რჩება, რადგან v1-მა აჩვენა რომ ეს ეხმარება;
- დაემატა ცალკე Store embedding და Dept embedding;
- embeddings გადადის პატარა MLP-ში, რომელიც აკეთებს horizon-level adjustment-ს.

v4-ის კითხვა:

```text
სჯობს თუ არა separate Store/Dept identity modeling უბრალო Store-Dept series_bias-ს?
```

თუ v4 აჯობებს v1-ს (`1506.28`), identity modeling არის სწორი გზა. თუ ვერ აჯობებს, მაშინ შემდეგი ძლიერი ნაბიჯი იქნება hyperparameter tuning v1/v4-ზე.
