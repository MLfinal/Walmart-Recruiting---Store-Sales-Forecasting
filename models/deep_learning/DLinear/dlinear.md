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
| v4 | 52w | Store/Dept embeddings | 7 | 1542.83 | +3.83% | -2.43% vs v1 |
| v5 | 52w | tuned v1 optimization | 15 | 1507.44 | +6.04% | -0.08% vs v1 |
| v6 | 52w | external covariates | 7 | 1548.03 | +3.51% | -2.77% vs v1 |

შენიშვნა: v1-ის იდეა იყო უფრო გრძელი context, მაგრამ რეალურად 104-week context ამ split-ზე შეუძლებელია training windows-ისთვის: pre-validation history არის 104 კვირა, ხოლო target horizon არის 39 კვირა. საჭიროა `input_weeks + 39 <= 104`. ამიტომ working v1 უნდა ჩაითვალოს როგორც 52-week DLinear + Store-Dept calibration.

მნიშვნელოვანი დასკვნა: v1 არ არის “ყველაზე სუსტი” model. v1 არის პირველი manual improvement baseline-ზე, რომელმაც დაამატა Store-Dept calibration და validation WMAE მკაფიოდ გააუმჯობესა. შემდეგი versions უფრო რთული იყო, მაგრამ validation-ზე noise/overfit შემოიტანა და v1-ს ვერ აჯობა. ამიტომ final wording არის: **best observed DLinear run = v1**, არა “დამტკიცებულად globally optimal DLinear”.

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

## v4: Store/Dept embeddings

v4-ში calendar features დროებით ამოვიღეთ, რადგან v3-მ v1 ვერ აჯობა. ვტესტავთ იმ მიმართულებას, რომელმაც უკვე გაამართლა: identity signal.

W&B run:

https://wandb.ai/kende23-n-a/Walmart-Recruiting---Store-Sales-Forecasting/runs/la5ta7ky

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

### v4 config/logs

| ველი | მნიშვნელობა |
|---|---:|
| `validation_weeks` | `39` |
| `input_weeks` | `52` |
| `batch_size` | `512` |
| `learning_rate` | `6e-4` |
| `weight_decay` | `2e-4` |
| `series_bias_weight_decay` | `1e-3` |
| `identity_weight_decay` | `1e-4` |
| `store_embedding_dim` | `8` |
| `dept_embedding_dim` | `12` |
| `identity_hidden_dim` | `32` |
| stores | `45` |
| depts | `81` |
| total training windows | `46634` |
| validation series | `3331` |
| best epoch | `7` |
| early stopping | epoch `21` |
| best validation WMAE | `1542.83` |

best epoch-ის log:

```text
epoch = 7
train/normalized_wmae_loss = 2.3371136
validation/normalized_wmae_loss = 0.6916478
validation/wmae = 1542.8344
validation/improvement_vs_seasonal_naive_pct = 3.8295
validation/improvement_vs_baseline_pct = -1.2884
validation/improvement_vs_v1_pct = -2.4266
learning_rate = 0.0006
```

Run summary:

```text
best_epoch = 7
best_validation_wmae = 1542.83435
baseline_dlinear_39w_wmae = 1523.20972
dlinear_v1_calibration_wmae = 1506.28247
dlinear_v3_gated_calendar_wmae = 1511.97327
best_improvement_vs_v1_pct = -2.42663
```

### v4 result interpretation

v4 გაუარესდა:

```text
v1: 1506.28
v4: 1542.83
difference = -2.43% vs v1
```

ასევე v4 baseline-ზეც უარესია:

```text
baseline: 1523.21
v4: 1542.83
```

ეს ნიშნავს, რომ separate Store/Dept embedding MLP ამ ფორმით არ დაეხმარა. სავარაუდო მიზეზები:

- `series_bias` უკვე იჭერს Store-Dept identity correction-ს პირდაპირ;
- Store/Dept embedding MLP ამატებს capacity-ს, მაგრამ validation-ზე generalization არ გაუმჯობესდა;
- train loss მცირდებოდა, validation კი საუკეთესო იყო ადრე, epoch 7-ზე, რის შემდეგაც validation WMAE გაუარესდა;
- Store/Dept decomposition შეიძლება ზედმეტად უხეშია, რადგან Walmart-ში კონკრეტული Store-Dept pair უფრო მნიშვნელოვანია, ვიდრე Store და Dept ცალ-ცალკე.

დასკვნა: v4 rejected. საუკეთესო observed architecture კვლავ v1 რჩება.

## შემდეგი v5: v1 refinement

რადგან v2/v3/v4 feature branches-მა v1 ვერ აჯობა, v5 აღარ ამატებს ახალ features-ს. v5 აბრუნებს საუკეთესო იდეას:

```text
DLinear + Store-Dept series_bias
```

და ცდის უფრო ფრთხილ training setup-ს:

- calendar features არ არის;
- Store/Dept embeddings არ არის;
- `input_weeks=52`;
- `series_bias` რჩება;
- learning rate მცირდება `8e-4 → 5e-4`;
- patience იზრდება, რომ training-ს მეტი შანსი ჰქონდეს;
- scheduler რჩება.

v5-ის კითხვა:

```text
შეგვიძლია თუ არა v1-ს ვაჯობოთ არა ახალი feature-ით, არამედ უფრო სტაბილური optimization-ით?
```

W&B run:

https://wandb.ai/kende23-n-a/Walmart-Recruiting---Store-Sales-Forecasting/runs/3oypefvl

### v5 config/logs

| ველი | მნიშვნელობა |
|---|---:|
| `validation_weeks` | `39` |
| `input_weeks` | `52` |
| `batch_size` | `512` |
| `learning_rate` | `5e-4` |
| `weight_decay` | `2e-4` |
| `series_bias_weight_decay` | `1e-3` |
| `moving_avg_kernel` | `25` |
| `epochs` | `120` |
| `patience` | `18` |
| total training windows | `46634` |
| validation series | `3331` |
| best epoch | `15` |
| early stopping | epoch `33` |
| best validation WMAE | `1507.44` |

best epoch-ის log:

```text
epoch = 15
train/normalized_wmae_loss = 2.3483218
validation/normalized_wmae_loss = 0.6868140
validation/wmae = 1507.4388
validation/improvement_vs_seasonal_naive_pct = 6.0358
validation/improvement_vs_baseline_pct = 1.0354
validation/improvement_vs_v1_pct = -0.0768
learning_rate = 0.0005
```

Run summary:

```text
best_epoch = 15
best_validation_wmae = 1507.43884
baseline_dlinear_39w_wmae = 1523.20972
dlinear_v1_calibration_wmae = 1506.28247
dlinear_v4_identity_wmae = 1542.83435
best_improvement_vs_v1_pct = -0.07677
```

Run history:

- validation WMAE სწრაფად გაუმჯობესდა `1647.75 → 1507.44`;
- საუკეთესო შედეგი იყო epoch `15`;
- epoch 15-ის შემდეგ train loss განაგრძობდა შემცირებას, მაგრამ validation აღარ გაუმჯობესდა;
- learning rate შემცირდა `0.0005 → 0.00025 → 0.000125 → 0.0000625`;
- early stopping მოხდა epoch `33`-ზე.

### v5 result interpretation

v5 თითქმის გაუტოლდა v1-ს, მაგრამ ვერ აჯობა:

```text
v1: 1506.28
v5: 1507.44
difference = -0.08% vs v1
```

v5 baseline-ზე უკეთესია:

```text
baseline: 1523.21
v5: 1507.44
improvement = +1.04%
```

დასკვნა:

- lower learning rate-მა training უფრო სტაბილური გახადა;
- v5 ბევრად სჯობს v4-ს და v3-ს უახლოვდება/ჯობნის, მაგრამ v1-ს ვერ აჭარბებს;
- DLinear-ის საუკეთესო validation run რჩება v1 (`1506.28`);
- შემდგომი მნიშვნელოვანი გაუმჯობესება ალბათ უკვე random hyperparameter tuning-ს ან external covariates-ს მოითხოვს.

## DLinear experiments summary

ამ ეტაპზე DLinear-ის ხელით გაკეთებული sequential experiments საკმარისია:

- baseline დაამტკიცა, რომ pure sequence DLinear მუშაობს;
- v1 დაამტკიცა, რომ Store-Dept calibration ეხმარება;
- v2/v3 აჩვენებს, რომ calendar features საჭიროა ძალიან ფრთხილად;
- v4 აჩვენებს, რომ separate Store/Dept embeddings ამ ფორმით არ დაეხმარა;
- v5 აჩვენებს, რომ მარტივი optimization tweak v1-ს თითქმის უტოლდება, მაგრამ არ სჯობს.

## საბოლოო v6: external covariates

v6 არის ბოლო manual DLinear experiment inference-მდე. აქ ვტესტავთ იმ feature-ებს, რომლებიც ჯერ არ გვიცდია, მაგრამ Walmart-ის ამოცანაში შეიძლება მნიშვნელოვანი იყოს.

v6 base ისევ v1-ია:

```text
past 52 Weekly_Sales
→ DLinear forecast

series_idx
→ Store-Dept series_bias
```

დამატებულია future covariates:

`features.csv`-დან:

- `Temperature`;
- `Fuel_Price`;
- `MarkDown1`;
- `MarkDown2`;
- `MarkDown3`;
- `MarkDown4`;
- `MarkDown5`;
- `CPI`;
- `Unemployment`;
- `IsHoliday`.

`stores.csv`-დან:

- `Type_A`;
- `Type_B`;
- `Type_C`;
- `Size`.

როგორ დაემატა:

```text
(Store, Date) features
→ numeric scaling fitted only on fit dates
→ future_covariates for each 39-week target window
→ covariate_head MLP
→ gated covariate_adjustment

forecast = DLinear forecast
         + series_bias
         + tanh(covariate_gate) * covariate_adjustment
```

რატომ gated: v2/v3-მა გვაჩვენა, რომ external branch პირდაპირ თუ ემატება forecast-ს, შეიძლება დააზიანოს prediction. ამიტომ v6-ში covariate branch იწყება თითქმის 0 გავლენით და მხოლოდ training-ისას იღებს წონას.

leakage control:

- validation split იგივეა: ბოლო 39 კვირა;
- target-derived features არ ემატება;
- covariate scaler fit ხდება მხოლოდ pre-validation fit dates-ზე;
- validation future covariates გამოიყენება იმიტომ, რომ იგივე ტიპის future covariates test period-შიც გვაქვს `features.csv`-ში.

v6-ის კითხვა:

```text
შეიძლება თუ არა external Walmart covariates-მა DLinear-ს v1-ზე უკეთესი შედეგი მისცეს?
```

W&B run:

https://wandb.ai/kende23-n-a/Walmart-Recruiting---Store-Sales-Forecasting/runs/y7yrkwl0

### v6 config/logs

| ველი | მნიშვნელობა |
|---|---:|
| `validation_weeks` | `39` |
| `input_weeks` | `52` |
| `batch_size` | `512` |
| `learning_rate` | `5e-4` |
| `weight_decay` | `2e-4` |
| `series_bias_weight_decay` | `1e-3` |
| `covariate_weight_decay` | `1e-4` |
| `covariate_gate_l1` | `1e-4` |
| `covariate_hidden_dim` | `32` |
| total training windows | `46634` |
| validation series | `3331` |
| best epoch | `7` |
| early stopping | epoch `25` |
| best validation WMAE | `1548.03` |
| best covariate gate | `-0.11790` |

best epoch-ის log:

```text
epoch = 7
train/normalized_wmae_loss = 2.3866920
validation/normalized_wmae_loss = 0.6892855
validation/wmae = 1548.0344
validation/improvement_vs_seasonal_naive_pct = 3.5054
validation/improvement_vs_baseline_pct = -1.6298
validation/improvement_vs_v1_pct = -2.7719
model/covariate_gate = -0.11790
learning_rate = 0.0005
```

Run summary:

```text
best_epoch = 7
best_validation_wmae = 1548.03442
dlinear_v1_calibration_wmae = 1506.28247
dlinear_v5_tuned_wmae = 1507.43884
best_improvement_vs_v1_pct = -2.77185
final_covariate_gate = -0.11790
```

v6-ში გამოყენებული features:

```text
Temperature, Fuel_Price,
MarkDown1, MarkDown2, MarkDown3, MarkDown4, MarkDown5,
CPI, Unemployment,
Size,
Type_A, Type_B, Type_C,
IsHoliday
```

### v6 result interpretation

v6-მ ვერ გააუმჯობესა v1:

```text
v1: 1506.28
v6: 1548.03
difference = -2.77% vs v1
```

v6 baseline-ზეც უარესია:

```text
baseline: 1523.21
v6: 1548.03
```

ეს ნიშნავს, რომ external covariates ამ ფორმით DLinear-ს არ დაეხმარა. სავარაუდო მიზეზები:

- DLinear-ის მარტივი forecast head უკეთ მუშაობს target-history pattern-ზე, ვიდრე heterogeneous tabular covariates-ზე;
- covariate branch-მა train loss შეამცირა, მაგრამ validation WMAE სწრაფად გაუარესდა epoch 7-ის შემდეგ;
- tree-based models უკეთ ამუშავებენ markdown/CPI/fuel/store metadata ტიპის tabular signal-ს;
- covariate gate non-zero გახდა, მაგრამ ეს signal validation-ზე noise აღმოჩნდა.

დასკვნა: v6 rejected. DLinear-ის საუკეთესო observed architecture კვლავ v1 რჩება.

## Hyperparameter tuning

manual feature experiments დასრულებულია. შემდეგი ნაბიჯი არის არა ახალი feature branch, არამედ საუკეთესო observed manual architecture-ის tuning:

```text
best observed manual architecture = v1 = DLinear + Store-Dept series_bias
```

Tuning notebook ცდის:

- `input_weeks`: `39`, `52`;
- `learning_rate`;
- `weight_decay`;
- `series_bias_weight_decay`;
- `moving_avg_kernel`;
- `batch_size`;
- patience/scheduler იგივე ლოგიკით.

Colab-ისთვის tuning loop განზრახ მსუბუქია:

- default `n_trials=8`;
- max `35` epoch თითო trial-ზე;
- `num_workers=0`, რომ DataLoader thread-ებზე არ გაიჭედოს;
- Optuna pruning იყენებს validation WMAE-ს;
- W&B-ზე აღარ იქმნება ცალკე run თითო trial-ზე, ილოგება ერთი tuning run და trial summary table.

თუ tuning-მა v1-ს აჯობა, inference გაკეთდება tuned checkpoint-ით. თუ tuning ვერ აჯობებს, inference გაკეთდება v1 checkpoint/artifact-ით.

აქ ერთი მნიშვნელოვანი technical caveat გვაქვს: tuning-მა დატესტა v1 architecture, მაგრამ exact v1 hyperparameter recipe არ იყო guarantee-ით ჩასმული როგორც fixed first trial. ანუ Optuna-მ სცადა v1-style models, მაგრამ არ გაუშვია ზუსტად იგივე configuration:

```text
v1:
input_weeks = 52
batch_size = 512
learning_rate = 0.0008
weight_decay = 0.0002
series_bias_weight_decay = 0.001
moving_avg_kernel = 25
```

tuning search space-ში ეს მნიშვნელობები ნაწილობრივ იყო შესაძლებელი, მაგრამ sampled best trial სხვა იყო. ამიტომ tuning result სწორად უნდა წავიკითხოთ ასე:

```text
tuning did not beat the already saved v1 checkpoint
```

და არა ასე:

```text
tuning mathematically proved v1 hyperparameters are optimal
```

### tuning results

W&B tuning run:

https://wandb.ai/kende23-n-a/Walmart-Recruiting---Store-Sales-Forecasting/runs/gh0dvxo2

Best tuned artifact logging run:

https://wandb.ai/kende23-n-a/Walmart-Recruiting---Store-Sales-Forecasting/runs/4jlq0kyc

Optuna-მ გაუშვა `8` trial. საუკეთესო trial იყო `0`.

Best tuning result:

```text
best_validation_wmae = 1508.9519
manual_v1_wmae = 1506.2825
improvement_vs_manual_v1_pct = -0.1772
```

Best hyperparameters:

```text
input_weeks = 52
batch_size = 512
learning_rate = 0.00036199292763563996
weight_decay = 0.00007705594012729586
series_bias_weight_decay = 0.00012551115172973836
moving_avg_kernel = 13
best_epoch = 15
```

Trial ranking:

| trial | WMAE | input | batch | lr | moving avg | state |
|---:|---:|---:|---:|---:|---:|---|
| 0 | 1508.95 | 52 | 512 | 0.000362 | 13 | complete |
| 1 | 1509.71 | 52 | 512 | 0.000373 | 13 | complete |
| 4 | 1553.09 | 52 | 1024 | 0.000410 | 25 | pruned |
| 2 | 1584.46 | 39 | 1024 | 0.000520 | 25 | complete |

Interpretation:

- tuning-მ v1-ს ვერ აჯობა;
- საუკეთესო tuned run ძალიან ახლოსაა v1-თან, მაგრამ validation-ზე მაინც `0.18%`-ით სუსტია;
- `input_weeks=52` ისევ დადასტურდა როგორც უკეთესი არჩევანი, ვიდრე `39`;
- `moving_avg_kernel=13` tuning-ში საუკეთესო sampled trial-ში გამოვიდა, მაგრამ exact v1 setup tuning-ში fixed trial-ად არ ყოფილა;
- tuning უფრო exploratory იყო: `8` trial, `35` max epoch, `patience=7`. v1 checkpoint-ს ჰქონდა საკუთარი training setup და saved result.

Final DLinear model choice:

```text
best observed DLinear run = manual v1
WMAE = 1506.28
```

Inference default იქნება manual v1 artifact-ზე. tuned artifact დარჩება fallback/comparison option-ად.

## Inference notebook

Final inference ფაილი:

```text
models/deep_learning/DLinear/dlinear_inference.ipynb
```

Inference იყენებს არა ბოლო trained run-ს, არამედ საუკეთესო validation model-ს:

```text
selected model = manual v1
architecture = DLinear + Store-Dept series_bias
validation WMAE = 1506.28
```

რატომ არა tuned model:

```text
tuned best WMAE = 1508.95
manual v1 WMAE = 1506.28
```

Tuning-ის საუკეთესო hyperparameters იყო:

```text
input_weeks = 52
batch_size = 512
learning_rate = 0.00036199292763563996
weight_decay = 0.00007705594012729586
series_bias_weight_decay = 0.00012551115172973836
moving_avg_kernel = 13
best_epoch = 15
```

მაგრამ ამ configuration-მა saved v1 checkpoint-ს ვერ აჯობა, ამიტომ final inference default რჩება `manual_v1`. ეს არის practical model-selection decision: inference იყენებს საუკეთესო observed validation checkpoint-ს.

Notebook-ის flow:

1. packages/install + Drive mount;
2. config-ში ირჩევა `model_choice = "manual_v1"` ან `"tuned_best"`;
3. W&B run იწყება `job_type="dlinear_inference"`;
4. notebook W&B artifact-იდან ტვირთავს selected checkpoint-ს;
5. checkpoint-იდან იღებს `series_index`, model weights-ს და train config-ს;
6. `input_weeks` და `pred_len` აღდგება weight tensor shape-ებიდან, რომ ძველ 104w naming/config confusion-მა inference არ გატეხოს;
7. `train.csv`-დან იქმნება Store-Dept sales panel;
8. ბოლო `input_weeks` კვირა ნორმალიზდება თითო series-ზე;
9. model აკეთებს test horizon forecast-ს;
10. prediction იჭრება `>= 0`, რადგან negative weekly sales submission-ისთვის ცუდი სიგნალია;
11. იქმნება Kaggle submission csv;
12. W&B-ზე ილოგება inference metrics, preview table, histogram, submission artifact და manifest;
13. selected model artifact ლინკდება W&B Model Registry-ში, რომ final inference model ცალკე გამოჩნდეს.

Inference W&B-ზე ლოგავს:

- selected model artifact URI;
- checkpoint path/name;
- `input_weeks`, `pred_len`, `moving_avg_kernel`;
- number of Store-Dept series;
- submission row count;
- missing predictions count;
- prediction min/mean/max;
- submission preview table;
- prediction distribution histogram;
- submission csv artifact;
- inference manifest json;
- model registry link, თუ W&B permission/registry path სწორია.

Final expectation:

```text
Experiment phase: finished
Best observed DLinear: manual v1
Next DLinear step: run dlinear_inference.ipynb on Colab and check W&B inference run/artifacts
```

## Kaggle submission ანალიზი

Final Kaggle submission-ზე DLinear-ის score მივიღე:

```text
Kaggle score: 3500
```

ჩემი შეფასებით, ეს შედეგი validation score-თან შედარებით სუსტად გამოიყურება, რადგან validation-ზე DLinear ძალიან ძლიერად ჩანდა:

```text
Best validation WMAE = 1506.28
```

მაგრამ Kaggle-ზე score `3500` გახდა. ეს განსხვავება ჩემთვის ნიშნავს, რომ validation setup ბოლომდე არ იმეორებდა Kaggle test-ის სირთულეს. DLinear კარგად იჭერს trend/seasonality-ს თითო Store-Dept series-ზე, მაგრამ Kaggle horizon-ზე forecast უკვე უფრო შორს მიდის და uncertainty იზრდება.

რატომ გამოვიდა DLinear უკეთესი, ვიდრე N-BEATS:

- DLinear უფრო მარტივი და სტაბილური არქიტექტურაა.
- Walmart data-ში ბევრი series-ს აქვს ძლიერი yearly/linear pattern, რაც DLinear-ს კარგად ერგება.
- მოდელი არ ცდილობს ზედმეტად რთული nonlinear representation-ის სწავლას მცირე ისტორიული ფანჯრიდან.
- Store-Dept `series_bias` ეხმარება თითოეული series-ის საშუალო sales level-ის დაჭერაში.

რატომ ვერ აჯობა XGBoost-ს:

- DLinear ძირითადად time-series signal-ს ეყრდნობა, ხოლო XGBoost უკეთ იყენებს tabular context-ს: holidays, markdowns, store metadata, external features.
- Kaggle test-ზე promotion/holiday behavior შეიძლება ისეთი იყოს, რასაც მხოლოდ linear decomposition ვერ დაიჭერს.
- DLinear forecast horizon-ზე error გროვდება, განსაკუთრებით იმ Store-Dept series-ებში, სადაც sales არასტაბილურია.

ჩემი დასკვნა:

DLinear აღმოჩნდა საუკეთესო deep learning მიმართულებიდან, მაგრამ tree-based მოდელებს მაინც ჩამორჩა. ამ dataset-ზე tabular feature engineering უფრო მნიშვნელოვანი აღმოჩნდა, ვიდრე მხოლოდ neural forecasting architecture. DLinear-ის `3500` score მისაღები და შედარებით სტაბილურია, მაგრამ საბოლოო Kaggle submission-ისთვის XGBoost უკეთესი candidate ჩანს.
