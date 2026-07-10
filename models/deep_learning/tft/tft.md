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

## v4 — residual blending

v3 fixed-ის შემდეგ უკვე ვიცოდით, რომ TFT-ის residual correction სასარგებლო იყო, მაგრამ `alpha = 1.0` ანუ სრული correction seasonal naive-ს აუარესებდა. ამიტომ v4-ში training architecture არ შეგვიცვლია. იგივე residual TFT გავუშვით და validation-ზე დავამატეთ post-processing comparison:

```text
Prediction(alpha) = SeasonalNaive52 + alpha * PredictedResidual
```

ამით ერთდროულად შევამოწმეთ, რამდენად უნდა ვენდოთ TFT-ის correction-ს:

```text
alpha = 0.00 -> pure seasonal naive
alpha = 0.25 -> TFT correction-ის 25%
alpha = 0.50 -> TFT correction-ის 50%
alpha = 0.75 -> TFT correction-ის 75%
alpha = 1.00 -> full residual TFT, იგივე v3 reconstruction
```

notebook-ში v4 უკვე ცალკე experiment-ადაა დალოგილი:

```text
run_name = tft_v4_residual_blending_external_covariates
artifact_name = tft-v4-residual-blending-external-covariates
```

W&B run:

```text
https://wandb.ai/kende23-n-a/Walmart-Recruiting---Store-Sales-Forecasting/runs/bk5a1tph
```

v4-ის full residual შედეგი პრაქტიკულად v3-ის მსგავსი გამოვიდა:

```text
validation_wmae_full_residual_alpha_1 = 5213.25
seasonal_naive_wmae = 4969.77
improvement_vs_seasonal_naive_pct = -4.90%
best_val_loss = 5230.72
prediction_rows = 19500
```

მაგრამ blending-მა შეცვალა სურათი. საუკეთესო აღმოჩნდა `alpha = 0.50`:

```text
best_blend_alpha = 0.50
best_blend_wmae = 4728.60
best_blend_improvement_vs_seasonal_naive_pct = +4.85%
best_blend_improvement_vs_full_residual_pct = +9.30%
```

alpha comparison:

| alpha | WMAE | seasonal naive-სთან შედარება |
|---:|---:|---:|
| 0.50 | 4728.60 | +4.85% |
| 0.25 | 4757.41 | +4.27% |
| 0.75 | 4886.35 | +1.68% |
| 0.00 | 4969.77 | reference |
| 1.00 | 5213.25 | -4.90% |

ეს მნიშვნელოვანი შედეგია: TFT-ის correction მთლიანად რომ გამოვიყენეთ, მოდელი seasonal naive-ზე უარესი იყო; მაგრამ correction-ის ნახევარი უკვე უკეთესი აღმოჩნდა. ანუ TFT useful signal-ს პოულობს, უბრალოდ raw output ზედმეტად ძლიერად ცვლის seasonal baseline-ს. v4-ში მოდელის “ჭკუა” გამოვიყენეთ უფრო ფრთხილად და პირველად TFT-მ top-500 validation-ზე seasonal naive-ს აჯობა.

## v5 — fine residual blending

v4-ში coarse alpha grid-მა გვაჩვენა, რომ საუკეთესო იყო `alpha = 0.50`. ამიტომ v5-ში model architecture ისევ არ შეგვიცვლია. იგივე residual TFT დავტოვეთ და მხოლოდ blending search გავხადეთ უფრო წვრილი იმ ზონის გარშემო, სადაც v4-მ კარგი შედეგი მოგვცა:

```text
v4 alpha grid = [0.00, 0.25, 0.50, 0.75, 1.00]
v5 alpha grid = [0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65]
```

notebook-ში v5 ცალკე W&B run-ად გაეშვა:

```text
run_name = tft_v5_fine_residual_blending_external_covariates
artifact_name = tft-v5-fine-residual-blending-external-covariates
```

W&B run:

```text
https://wandb.ai/kende23-n-a/Walmart-Recruiting---Store-Sales-Forecasting/runs/r5nf8rv9
```

full residual ისევ seasonal naive-ზე უარესი დარჩა:

```text
validation_wmae_full_residual_alpha_1 = 5212.70
seasonal_naive_wmae = 4969.77
improvement_vs_seasonal_naive_pct = -4.89%
best_val_loss = 5230.36
prediction_rows = 19500
```

მაგრამ fine blending-მა v4-ზე უკეთესი alpha იპოვა:

```text
best_blend_alpha = 0.40
best_blend_wmae = 4717.71
best_blend_improvement_vs_seasonal_naive_pct = +5.07%
best_blend_improvement_vs_full_residual_pct = +9.50%
```

alpha comparison:

| alpha | WMAE | seasonal naive-სთან შედარება |
|---:|---:|---:|
| 0.40 | 4717.71 | +5.07% |
| 0.45 | 4719.21 | +5.04% |
| 0.35 | 4723.93 | +4.95% |
| 0.50 | 4728.06 | +4.86% |
| 0.55 | 4744.74 | +4.53% |
| 0.60 | 4768.61 | +4.05% |
| 0.65 | 4800.98 | +3.40% |

v5-მა დააზუსტა v4-ის დასკვნა: TFT-ის residual correction საჭიროა, მაგრამ არა სრულად. საუკეთესო ზონა `0.40–0.45` აღმოჩნდა. ეს ნიშნავს, რომ seasonal baseline რჩება პროგნოზის მთავარი ნაწილი, ხოლო TFT უკეთ მუშაობს როგორც correction model, რომელსაც კონტროლირებადი წონა აქვს.

## v6 — serious full-data train

v5-ის შემდეგ გაჩნდა სწორი კითხვა: შეიძლება TFT უბრალოდ საკმარისად დიდხანს ან საკმარისად დიდ data-ზე არ ვწვრთნიდით? ამიტომ v6-ში უკვე პატარა top-500 experiment აღარ გავაკეთეთ. მიზანი იყო შეგვემოწმებინა, გაუმჯობესდება თუ არა residual TFT, თუ მას მივცემთ მეტ series-ს, მეტ yearly context-ს, მეტ capacity-ს და ბევრად დიდ training budget-ს.

v6-ის configuration:

```text
top_n_series = 4000
actual n_series = 3331
encoder_weeks = 52
validation_weeks = 39
batch_size = 512
max_epochs = 50
patience = 8
max_time_minutes = 240
limit_train_batches = 1.0
limit_val_batches = 1.0
hidden_size = 24
hidden_continuous_size = 12
attention_head_size = 2
dropout = 0.10
blend_alphas = [0.30, 0.35, 0.40, 0.45, 0.50]
```

`top_n_series = 4000` რეალურად full-data რეჟიმია, რადგან dataset-ში `3331` Store-Dept series აღმოჩნდა. training data აღარ შეიკვეცა:

```text
train_rows_before = 421570
train_rows_after = 421570
training_samples = 116736
validation_samples = 2938
train_batches_total = 228
validation_batches_total = 6
model_parameters = 81965
```

W&B run:

```text
https://wandb.ai/kende23-n-a/Walmart-Recruiting---Store-Sales-Forecasting/runs/bsywwipw
```

Notebook output-ის მიხედვით run-მა მიაღწია:

```text
epoch = 8
trainer/global_step = 2051
best_checkpoint = tft-v6-epoch=00-val_loss=10628.5859.ckpt
best_val_loss = 10628.5859
seasonal_naive_wmae = 1604.27
prediction_rows = 114582
```

ზუსტი wall-clock duration notebook output-ში არ არის შენახული. რაც ჩანს: v6 აღარ იყო 10–25 წუთიანი პატარა run; მან full-data რეჟიმში 9 epoch-მდე (`0`-დან `8`-მდე) და `2051` optimizer step-მდე მივიდა. Training-ს ჰქონდა `240` წუთიანი hard cap, მაგრამ output-იდან ზუსტად ვერ ვამბობთ, early stopping-მა გააჩერა თუ runtime cap-მა.

მთავარი შედეგი ის არის, რომ v6 valid model result არ გახდა. Evaluation-ში ყველა WMAE გახდა `NaN`:

```text
validation_wmae_full_residual_alpha_1 = NaN
best_blend_alpha = 0.30
best_blend_wmae = NaN
best_blend_improvement_vs_seasonal_naive_pct = NaN
```

alpha comparison-იც მთლიანად invalid გამოვიდა:

| alpha | WMAE |
|---:|---:|
| 0.30 | NaN |
| 0.35 | NaN |
| 0.40 | NaN |
| 0.45 | NaN |
| 0.50 | NaN |

ეს ნიშნავს, რომ v6-ის prediction/evaluation chain-ში non-finite values გაჩნდა. რადგან `prediction_rows = 114582` შეიქმნა, alignment მთლიანად არ ჩამოშლილა; პრობლემა უფრო likely არის prediction values-ში ან reconstruction-ში, სადაც ერთი ან მეტი `NaN` მთელ WMAE-ს `NaN`-ად აქცევს. ამ run-ს leaderboard-style comparison-ში ვერ ჩავთვლით.

v6-მა მაინც მნიშვნელოვანი ინფორმაცია მოგვცა:

- full-data serious train ავტომატურად უკეთესი არ გახდა;
- larger model (`41K` → `82K` parameter) და full panel საკმარისი არ იყო stable residual prediction-ისთვის;
- best validation checkpoint epoch `0`-ზე დარჩა, ხოლო შემდეგი validation loss-ები გაუარესდა;
- full-data validation-ზე seasonal naive ბევრად ძლიერი reference გამოვიდა (`1604.27`), ამიტომ top-500 შედეგებთან პირდაპირი შედარება არ შეიძლება;
- v6-ის მთავარი ტექნიკური პრობლემა იყო guard-ების არქონა: თუ prediction-ში ერთი `NaN` მაინც გაჩნდება, WMAE მთლიანად `NaN` ხდება.

## v7 — stable serious residual blending

v6-ის შემდეგ არ იყო სწორი უბრალოდ იგივე full-data train-ის გამეორება. ამიტომ v7-ში მიზანი იყო არა მაქსიმალურად დიდი run, არამედ ისეთი serious run, რომელიც არ დაიშლება და მოგვცემს სანდო WMAE-ს. იდეა იგივე დარჩა: TFT არ პროგნოზირებს პირდაპირ sales-ს, ის პროგნოზირებს 52-week seasonal baseline-ის correction-ს.

v7-ში შევცვალეთ:

```text
top_n_series = 2000
train_rows_before = 421570
train_rows_after = 285226
encoder_weeks = 52
validation_weeks = 39
batch_size = 512
max_epochs = 45
patience = 7
max_time_minutes = 240
learning_rate = 5e-5
hidden_size = 24
hidden_continuous_size = 12
attention_head_size = 2
dropout = 0.15
gradient_clip_val = 0.05
residual_clip = 100000
prediction_clip_max = 300000
blend_alphas = [0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
```

`top_n_series = 2000` compromise იყო: v5-ის top-500-ზე ბევრად დიდი data მივეცით, მაგრამ v6-ის full `3331` series აღარ ავიღეთ, რადგან იქ noisy/sparse series-ებმა და non-finite prediction-ებმა run გააფუჭა. ასევე დავამატეთ stability checks:

```text
raw_prediction_nan_count
raw_prediction_posinf_count
raw_prediction_neginf_count
raw_prediction_finite_count
postprocessed_prediction_min / mean / max
```

ამით უკვე ზუსტად ვხედავთ, prediction valid არის თუ არა. v7-ში prediction აღარ გაფუჭდა:

```text
raw_prediction_nan_count = 0
raw_prediction_posinf_count = 0
raw_prediction_neginf_count = 0
raw_prediction_finite_count = 78000
raw_prediction_total_count = 78000
raw_prediction_min_finite = -20219.15
raw_prediction_mean_finite = 250.06
raw_prediction_max_finite = 25161.58
```

W&B run:

```text
https://wandb.ai/kende23-n-a/Walmart-Recruiting---Store-Sales-Forecasting/runs/iz6z9sd1
```

Training early stopped / finished around epoch `8`. საუკეთესო checkpoint იყო:

```text
best_checkpoint = tft-v7-epoch=01-val_loss=2830.1372.ckpt
best_val_loss = 2830.1372
model_parameters = 81.6K
training_samples = 79679
validation_samples = 2000
train_batches_total = 155
validation_batches_total = 4
```

ეს ნიშნავს, რომ მოდელმა საუკეთესო validation loss ძალიან ადრე მიიღო. ამიტომ გრძელი `max_epochs = 45` პრაქტიკულად ბოლომდე არ დასჭირდა; early stopping-მა ცუდი epoch-ების გაგრძელება შეწყვიტა. ეს crash არ არის — ეს ნორმალური training control-ია. მაგრამ ისიც გვაჩვენებს, რომ მოცემული architecture/data setup სწრაფად აღწევს თავის ლიმიტს.

v7-ის validation შედეგები:

```text
seasonal_naive_wmae = 2488.03
validation_wmae_full_residual_alpha_1 = 2736.14
full_residual_improvement_vs_seasonal_naive_pct = -9.97%
best_blend_alpha = 0.35
best_blend_wmae = 2379.50
best_blend_improvement_vs_seasonal_naive_pct = +4.36%
best_blend_improvement_vs_full_residual_pct = +13.03%
prediction_rows = 78000
```

full residual ისევ ცუდია, რადგან TFT correction-ს ზედმეტად აძლიერებს. მაგრამ blending-მა ეს გააკონტროლა: alpha `0.35` საუკეთესო აღმოჩნდა და seasonal naive-ზე `4.36%` improvement მოიტანა.

alpha comparison:

| alpha | WMAE | seasonal naive-სთან შედარება |
|---:|---:|---:|
| 0.35 | 2379.50 | +4.36% |
| 0.30 | 2381.50 | +4.28% |
| 0.40 | 2381.96 | +4.26% |
| 0.25 | 2388.23 | +4.01% |
| 0.45 | 2388.87 | +3.99% |
| 0.50 | 2400.04 | +3.54% |

v7-ის ყველაზე მნიშვნელოვანი დასკვნა არის ის, რომ TFT-ს შეუძლია ბევრად უკეთესი absolute WMAE აჩვენოს, როცა subset უფრო ძლიერი/მაღალ volume series-ებზეა აგებული. მაგრამ ეს რიცხვი პირდაპირ v5-ს არ უნდა შევადაროთ, რადგან validation population შეიცვალა: v5 იყო top-500, v7 არის top-2000. ამიტომ `4717.71 → 2379.50` არ ნიშნავს უბრალოდ ორჯერ უკეთეს მოდელს; ეს ნიშნავს, რომ top-2000 validation set-ზე seasonal baseline უკვე `2488.03` იყო და TFT-blending-მა მასზე დამატებით მცირე improvement გააკეთა.

## შედეგების მოკლე ცხრილი

| Run | Subset | იდეა | WMAE | დასკვნა |
|---|---:|---|---:|---|
| seasonal naive | top 300 | 52-week lookup | 6026.29 | reference |
| baseline | top 300 | small TFT, calendar only | 7801.90 | pipeline/logging OK, model weak |
| seasonal naive | top 500 | 52-week lookup | 4969.77 | reference |
| v1 | top 500 | raw sales + external covariates | 6200.95 | external covariates დაეხმარა baseline-თან შედარებით |
| v2 | top 500 | log target + external covariates | 6524.68 | worse than v1 |
| v3 invalid | top 500 | residual, broken seasonal base | 53035.36 | invalid implementation |
| v3 fixed | top 500 | residual + correct 52-week lookup | 5212.71 | residual signal useful, მაგრამ full correction ზედმეტია |
| v4 | top 500 | residual blending, alpha=0.50 | 4728.60 | best TFT so far და seasonal naive-ზე უკეთესი |
| v5 | top 500 | fine residual blending, alpha=0.40 | 4717.71 | best TFT so far |
| v6 | full 3331 series | serious full-data train | NaN | invalid evaluation / non-finite predictions |
| v7 | top 2000 | stable serious residual blending, alpha=0.35 | 2379.50 | valid, no NaN, seasonal naive-ზე უკეთესი |

ამ ეტაპზე საუკეთესო valid TFT run არის v7, მაგრამ შედარებისას context აუცილებელია: top-500 და top-2000 validation population ერთი და იგივე difficulty არ არის. მთავარი გაკვეთილი ასეთია: TFT-სთვის პირდაპირ full sales-ის სწავლა სუსტი აღმოჩნდა, log target-მა WMAE გააუარესა, seasonal residual-მა მოდელი seasonal naive-სთან დააახლოვა, ხოლო residual blending-მა correction-ის ძალა გააკონტროლა. v6-მა გვაჩვენა, რომ full-data serious train stability guard-ის გარეშე სანდო არ არის. v7-ში guard-ები დაემატა, prediction finite დარჩა და top-2000 subset-ზე საუკეთესო blending WMAE `2379.50` მივიღეთ.

## inference — v7 checkpoint + seasonal fallback

TFT-ზე training phase აქ დავასრულეთ და inference notebook ცალკე ფაილად დაიწერა:

```text
models/deep_learning/tft/tft_inference.ipynb
```

Inference არ წვრთნის მოდელს. flow ასეთია:

1. Colab-ზე იტვირთება raw `train.csv`, `test.csv`, `features.csv`, `stores.csv`;
2. თავიდან ითვლება იგივე deterministic top-2000 Store-Dept selection, რაც v7-ში გვქონდა;
3. `train.csv`-დან კეთდება 52-week seasonal baseline test horizon-ისთვის;
4. W&B-დან ჩამოდის v7 model artifact:

```text
kende23-n-a/Walmart-Recruiting---Store-Sales-Forecasting/tft-v7-stable-serious-residual-blending:experiment-v7
```

5. checkpoint-იდან იტვირთება `tft_v7_best.ckpt`;
6. TFT პროგნოზირებს მხოლოდ იმ Store-Dept series-ებს, რომლებიც top-2000-ში მოხვდა;
7. პროგნოზი reconstruct ხდება ასე:

```text
Weekly_Sales = SeasonalNaive52 + 0.35 * PredictedResidual
```

8. Store-Dept pair-ები, რომლებიც top-2000-ში არ არის, არ გადის TFT-ში და იღებს fallback-ს:

```text
Weekly_Sales = SeasonalNaive52
```

ეს მნიშვნელოვანია, რადგან TFT categorical encoder-ს unseen Store-Dept category-ების უსაფრთხოდ მიღება არ შეუძლია. ამიტომ inference-ში model coverage და fallback coverage ცალკე ლოგდება.

W&B inference run ინახავს:

- Kaggle submission csv-ს;
- detailed prediction csv-ს;
- TFT-covered prediction csv-ს;
- inference manifest-ს;
- prediction histogram-ს;
- TFT vs fallback coverage plot-ს;
- prediction NaN/Inf diagnostics-ს;
- row coverage summary-ს;
- model artifact lineage-ს v7 checkpoint-ზე.

ამით TFT-ის final artifact story სრულდება: training run ინახავს checkpoint-ს, inference run იყენებს ამ checkpoint-ს და W&B-ში აბრუნებს submission-სა და ყველა საჭირო diagnostic ფაილს.

### inference run result

Inference notebook წარმატებით გაეშვა და W&B-ზე დალოგდა:

```text
https://wandb.ai/kende23-n-a/Walmart-Recruiting---Store-Sales-Forecasting/runs/rr6jcmci
```

გამოყენებული configuration:

```text
model_artifact_uri = kende23-n-a/Walmart-Recruiting---Store-Sales-Forecasting/tft-v7-stable-serious-residual-blending:experiment-v7
checkpoint = /content/artifacts/tft_v7_model/tft_v7_best.ckpt
top_n_series = 2000
encoder_weeks = 52
blend_alpha = 0.35
```

data coverage:

```text
train_rows_before = 421570
train_rows_after_top_filter = 285226
test_rows_total = 115064
test_rows_tft_covered_before_dataset_index = 77844
test_series_total = 3169
test_series_tft_covered = 2000
test_series_fallback_only = 1169
```

`TimeSeriesDataSet`-მა 18 group prediction index-იდან ამოაგდო, რადგან მათთვის encoder/prediction window საკმარისად არ შედგა:

```text
prediction_samples = 1982
prediction_batches = 4
```

ეს არ არის failure. ეს ნიშნავს, რომ top-2000-ში მყოფი 18 Store-Dept pair მაინც ვერ გამოიყენა TFT-მა inference window-ისთვის. ასეთ rows-ზე notebook ავტომატურად seasonal fallback-ზე გადავიდა.

prediction diagnostics:

```text
raw_prediction_nan_count = 0
raw_prediction_posinf_count = 0
raw_prediction_neginf_count = 0
raw_prediction_finite_count = 77298
raw_prediction_total_count = 77298
raw_prediction_min_finite = -21457.74
raw_prediction_mean_finite = -61.26
raw_prediction_max_finite = 23676.27
```

ეს არის მთავარი ხარისხის check: v6-ისგან განსხვავებით, inference-ში `NaN` ან `Inf` prediction არ გაჩნდა.

final submission summary:

```text
submission_rows = 115064
tft_rows = 77248
fallback_rows = 37816
tft_row_coverage = 67.13%
prediction_min = 0.0
prediction_mean = 16458.39
prediction_max = 300000.0
registry_status = linked
```

output files:

```text
/content/drive/MyDrive/walmart_competition_inference/tft/tft_v7_submission.csv
/content/drive/MyDrive/walmart_competition_inference/tft/tft_v7_detailed_predictions.csv
/content/drive/MyDrive/walmart_competition_inference/tft/tft_v7_inference_manifest.json
```

Kaggle-ზე ამ submission-ის ატვირთვის შემდეგ მივიღეთ:

```text
file = tft_v7_submission.csv
status = Complete after deadline
public_score = 2979.86060
private_score = 3058.98280
```

ეს score validation WMAE-სგან განსხვავებულია, რადგან Kaggle test period უკვე სხვა დროის მონაკვეთია და იქ submission-ის დაახლოებით `32.87%` seasonal fallback-ით არის შევსებული. მიუხედავად ამისა, შედეგი usable final TFT submission-ად ჩაითვალა: notebook-მა სრულად შექმნა Kaggle-format ფაილი, W&B-ზე დალოგა inference lineage და Kaggle-მაც ფაილი წარმატებით მიიღო.

inference-ის საბოლოო ლოგიკა ასეთია: სადაც v7 TFT-ს სანდოდ შეუძლია პროგნოზი, ვიყენებთ blended residual correction-ს; სადაც TFT coverage არ გვაქვს, ვიყენებთ 52-week seasonal naive-ს. ამიტომ submission ყოველთვის სრულად ივსება, ხოლო W&B-ში ცალკე ჩანს, prediction-ის რა ნაწილი მოდის TFT-დან და რა ნაწილი fallback-იდან.

## Raw-input pipeline და Model Registry-ის გამოსწორება

თავდაპირველი v7 inference checkpoint-ს W&B artifact-იდან იწერდა, მაგრამ შემდეგ notebook თვითონ თავიდან კითხულობდა `train.csv`, `features.csv`, `stores.csv`-ს, ხელახლა ქმნიდა top-2000 selection-ს, `TimeSeriesDataSet`-ს, seasonal fallback-ს და blending logic-ს. ეს reproducible იყო, მაგრამ `DESCRIPTION.md`-ის ყველაზე მკაცრ მოთხოვნას ბოლომდე არ აკმაყოფილებდა:

```text
Model Registry artifact უნდა იყოს სრული pipeline,
რომელსაც პირდაპირ raw test.csv-ზე შეუძლია predict.
```

ამიტომ pipeline packaging training/experiment ეტაპზე გადავიტანეთ. `model_experiment_TFT.ipynb`-ში best v7 checkpoint-იდან შეიქმნა `TFTRawPipeline`, რომელიც ერთ artifact-ში აერთიანებს:

- TFT v7 model weights;
- fitted `TimeSeriesDataSet` encoders/scalers;
- სრული `421570`-row training history;
- `features.csv` და `stores.csv` tables;
- deterministic top-2000 series selection;
- 52-week residual target და seasonal-naive fallback;
- selected blend `alpha = 0.35`;
- raw input contract: `Store`, `Dept`, `Date`, `IsHoliday`.

pipeline registration flow:

```text
tft-v7 source checkpoint artifact
→ TFTRawPipeline
→ save/reload contract test on raw test.csv
→ W&B pipeline artifact
→ Walmart_TFT_Raw_Pipeline:champion
```

Contract test-ში pipeline-ს სრული raw test set ორჯერ მიეწოდა: ერთხელ memory-ში, ერთხელ save/reload შემდეგ. შედეგები იყო finite, non-negative და იდენტური. Registration metadata:

```text
pipeline_type        = TFTRawPipeline
top_n_series         = 2000
stored_history_rows  = 421570
blend_alpha          = 0.35
contract_rows        = 115064
registry target      = Walmart_TFT_Raw_Pipeline:champion
```

ახალი `tft_inference.ipynb` უკვე აღარ იწერს checkpoint-ს პირდაპირ და აღარ აწყობს TFT preprocessing-ს notebook-ში. მისი flow არის:

```python
pipeline = download("wandb-registry-model/Walmart_TFT_Raw_Pipeline:champion")
predictions = pipeline.predict(raw_test)
```

Registry-based inference წარმატებით გაეშვა:

```text
registry artifact     = Walmart_TFT_Raw_Pipeline:champion
pipeline type         = TFTRawPipeline
raw test rows         = 115064
submission rows       = 115064
prediction min/mean/max = 0.0 / 16458.3956 / 300000.0
prediction SHA-256    = 96e1018a7573866f81b77c29362bae112aa498fb1b61fedd88aea979d8f7af53
```

18 top-2000 series inference dataset-ში საკმარისი window-ის გამო ვერ მოხვდა. ეს crash არ არის: მათზეც pipeline-მა seasonal-naive fallback გამოიყენა. საბოლოო output ყოველთვის სრულია.

ერთ ეტაპზე მხოლოდ W&B submission artifact logging გაჩერდა, რადგან იგივე artifact სახელი ადრე სხვა artifact type-ით არსებობდა. Model prediction და Registry pipeline ამ დროს უკვე წარმატებული იყო. საბოლოო inference run-ში artifact type/name consistency გამოსწორდა და submission CSV, manifest და histogram W&B-ზე წარმატებით დაილოგა.
