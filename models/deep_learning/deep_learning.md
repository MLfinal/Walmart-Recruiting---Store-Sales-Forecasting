# Deep Learning მოდელების შედარება

ამ ფოლდერში სამი deep learning მიმართულება გავტესტე:

```text
models/deep_learning/DLinear
models/deep_learning/N-BEATS
models/deep_learning/tft
```

სამივე მოდელის საერთო მიზანი ერთი იყო: Walmart Store Sales Forecasting ამოცანაზე historical weekly sales-იდან future horizon-ის პროგნოზი დაგვეგენერირებინა და შედეგები W&B-ზე სრულად დაგველოგა. DLinear/TFT-ში ძირითადი validation horizon იყო `39` კვირა, ხოლო N-BEATS-ის README-ში მთავარი validation setup `32` კვირაზეა აღწერილი. ამიტომ მათი ლოგიკა, data format, feature usage და საბოლოო ქცევა მნიშვნელოვნად განსხვავდება.

## მოკლე დასკვნა

Deep learning მიმართულებაში საუკეთესო practical შედეგი მივიღე TFT inference submission-ით:

| Model | Best validation result | Kaggle result | ჩემი საბოლოო შეფასება |
|---|---:|---:|---|
| DLinear | `1506.28` WMAE | დაახლოებით `3500` | validation-ზე ყველაზე სუფთა და სტაბილური, მაგრამ Kaggle-ზე სუსტდება |
| N-BEATS | `2157.98` WMAE | დაახლოებით `4700` | ყველაზე სუსტი deep learning მოდელი ამ dataset-ზე |
| TFT | `2379.50` WMAE top-2000 subset-ზე | public `2979.86060`, private `3058.98280` | საუკეთესო Kaggle deep learning submission, მაგრამ fallback-ს ეყრდნობა |

აქ validation რიცხვები პირდაპირ ერთ ხაზზე არ უნდა შევადაროთ:

- DLinear შეფასდა ყველა `3331` Store-Dept series-ზე და ბოლო `39` validation კვირაზე;
- N-BEATS README-ში validation horizon ძირითადად `32` კვირაა;
- TFT-ის საუკეთესო valid run არის top-2000 subset-ზე, ხოლო ადრე top-300/top-500 ექსპერიმენტებიც გვქონდა.

ამიტომ საბოლოო ranking-ს Kaggle submission-ის მიხედვით ვაკეთებ:

```text
TFT     ≈ 3058 private
DLinear ≈ 3500
N-BEATS ≈ 4700
```

ამ ranking-ში TFT საუკეთესო deep learning submission გახდა, DLinear მეორეა, N-BEATS კი ყველაზე სუსტი.

## რა არის სამივეში საერთო

სამივე მოდელში ერთი ძირითადი იდეა მეორდება: Walmart-ის data ბუნებრივად არის ბევრი პატარა weekly time series:

```text
ერთი Store + Dept = ერთი time series
```

ყველა deep learning notebook-ში საბოლოოდ ვცდილობდი row-based CSV data გადამექცია sequence/panel format-ად:

```text
train.csv
→ Store-Dept × Date panel ან TimeSeriesDataSet
→ historical window
→ forecast horizon
→ WMAE
→ W&B logging/artifacts
```

სამივეში WMAE იყო მთავარი metric, რადგან Kaggle-ის metric-იც weighted MAE-ა:

```text
holiday week  -> weight 5
normal week   -> weight 1
```

სამივეში W&B გამოვიყენე არა მხოლოდ final metric-ისთვის, არამედ model story-ისთვისაც:

- config;
- validation metrics;
- prediction tables;
- plots;
- checkpoints/pipelines;
- inference artifacts;
- submission files;
- manifest/metadata;
- model registry ან artifact lineage.

## მთავარი განსხვავება data flow-ში

### DLinear

DLinear-ში data ყველაზე მარტივია:

```text
Store-Dept × Date sales panel
→ ბოლო 52 კვირა
→ DLinear decomposition
→ 39 კვირის forecast
```

DLinear თითქმის არ იყენებს external feature-ებს. მისი მთავარი strength არის target history. საუკეთესო v1-ში დავამატე Store-Dept calibration:

```text
series_idx → Embedding(n_series, pred_len)
forecast = DLinear forecast + series_bias
```

ანუ მოდელი სწავლობს არა მხოლოდ საერთო temporal pattern-ს, არამედ თითო Store-Dept pair-ის systematic correction-საც.

### N-BEATS

N-BEATS-იც sequence model-ია, მაგრამ DLinear-ზე უფრო nonlinear/stack-based architecture აქვს. მასაც ძირითადად historical sales sequence მივეცი:

```text
past sales window
→ N-BEATS blocks
→ forecast horizon
```

N-BEATS-ში feature engineering ფაქტობრივად model input-ში არ დამატებულა. ვცადე training setup-ის ცვლილებები:

- lower learning rate;
- early stopping;
- longer context;
- holiday-aware weighted loss;
- Optuna tuning.

მაგრამ მთავარი architecture ისევ historical sequence-ზე იყო დამოკიდებული.

### TFT

TFT ყველაზე complex model-ია:

```text
static categoricals: Store, Dept, Type
known future reals: calendar + external covariates
known future categoricals: IsHoliday
unknown target/residual
LSTM + attention + variable selection
```

TFT-ს raw sales direct prediction-ზე კარგი შედეგი არ ჰქონდა. საუკეთესო approach გახდა residual forecasting:

```text
SeasonalNaive52 = same Store-Dept sales 52 weeks earlier
ResidualSales = Weekly_Sales - SeasonalNaive52
TFT predicts PredictedResidual
Final prediction = SeasonalNaive52 + alpha * PredictedResidual
```

ამით TFT-ს აღარ ვთხოვდით მთლიანი sales scale-ის პროგნოზს. მას ვთხოვდით seasonal baseline-ის correction-ს.

## DLinear-ის გზა

DLinear დავიწყე როგორც სუფთა neural baseline. baseline-ში model input იყო მხოლოდ past `52` weekly sales. შედეგი:

```text
Seasonal naive WMAE = 1604.27
DLinear baseline WMAE = 1523.21
```

ეს უკვე კარგი signal იყო: უბრალო linear decomposition-მაც seasonal naive-ს აჯობა.

შემდეგ v1-ში დავამატე Store-Dept calibration:

```text
best validation WMAE = 1506.28
best epoch = 11
validation series = 3331
training windows = 46634
```

v1 საუკეთესო DLinear run დარჩა. მიზეზი მარტივია: Walmart-ში თითო Store-Dept pair-ს თავისი scale და bias აქვს. `series_bias` ამ მუდმივ გადახრას კარგად იჭერს.

შემდეგი მცდელობები:

| Run | იდეა | WMAE | დასკვნა |
|---|---|---:|---|
| baseline | pure DLinear | `1523.21` | seasonal naive-ზე უკეთესი |
| v1 | Store-Dept calibration | `1506.28` | საუკეთესო DLinear |
| v2 | calendar branch | `1961.45` | ცუდად გააუარესა |
| v3 | gated calendar branch | `1511.97` | თითქმის კარგი, მაგრამ v1-ზე სუსტი |
| v4 | Store/Dept embeddings | `1542.83` | identity branch ზედმეტი/ხმაურიანი გამოვიდა |
| v5 | v1 optimization refinement | `1507.44` | v1-ს თითქმის გაუტოლდა, მაგრამ ვერ აჯობა |
| v6 | external covariates | `1548.03` | covariates-მა DLinear-ში noise შემოიტანა |
| tuning | Optuna v1 architecture | `1508.95` | v1-ზე ოდნავ სუსტი |

აქ მთავარი lesson ჩემთვის იყო: DLinear-ს ზედმეტი feature branch ადვილად აფუჭებს. ეს model ძლიერი აღმოჩნდა მაშინ, როცა მას მარტივი historical signal + Store-Dept calibration მივეცი. Calendar/external covariates უფრო რთული neural branch-ებით არ დაეხმარა.

Inference-ში საბოლოოდ manual v1 ავირჩიე, რადგან tuning-მა saved v1 checkpoint-ს ვერ აჯობა:

```text
manual v1 WMAE = 1506.28
tuned best WMAE = 1508.95
```

Kaggle-ზე DLinear-ის score დაახლოებით `3500` მივიღე. ეს validation-ზე ბევრად უარესია. ჩემი დასკვნაა, რომ DLinear validation split-ზე ძალიან კარგად იჭერს yearly/series pattern-ს, მაგრამ Kaggle future horizon-ზე promotion/holiday/external behavior უფრო რთულია, ვიდრე pure historical linear decomposition.

## N-BEATS-ის გზა

N-BEATS თავიდან საინტერესო იყო, რადგან deep forecasting-ში ცნობილი architecture-ია და theoretically trend/seasonality-ს nonlinear basis-ებით სწავლობს. მაგრამ ჩვენს Walmart setup-ში ის ყველაზე სუსტი გამოვიდა.

Baseline result:

```text
Best epoch = 2
Best validation WMAE = 2157.9829
Best validation MAE = 2128.4329
Seasonal naive WMAE = 3902.8521
```

Baseline seasonal naive-ზე უკეთესი იყო, მაგრამ training curve-მა ადრევე აჩვენა პრობლემა:

```text
epoch 1 validation WMAE = 2222.4863
epoch 2 validation WMAE = 2157.9829
epoch 3 validation WMAE = 2191.0672
epoch 30 validation WMAE = 2244.5165
```

Training loss მცირდებოდა, validation კი epoch 2-ის შემდეგ უარესდებოდა. ანუ N-BEATS ძალიან ადრე იწყებდა overfitting/generalization degradation-ს.

შემდეგი ექსპერიმენტები:

| Experiment | ცვლილება | Best WMAE | დასკვნა |
|---|---|---:|---|
| baseline | default N-BEATS | `2157.98` | საუკეთესო N-BEATS |
| exp1 | lower LR + early stopping | `2186.50` | training კონტროლი უკეთესი, score უარესი |
| exp2 | context length `78` | `2662.81` | გრძელი context მკვეთრად უარესი |
| exp3 | holiday-aware weighted loss | `2185.14` | ლოგიკური იდეა, მაგრამ baseline ვერ აჯობა |
| partial Optuna | tuning | `2191.41` best partial | baseline მაინც უკეთესი |

N-BEATS-ის მთავარი პრობლემა ჩემთვის იყო ის, რომ model historical target sequence-ზეა დამოკიდებული და business context-ს თითქმის ვერ ხედავს. Walmart-ში კი holiday, markdown, store/dept identity, yearly lag და sparse series behavior ძალიან მნიშვნელოვანია.

Inference/registry flow-ში N-BEATS pipeline-ად შეფუთვის იდეა იყო:

```text
walmart_nbeats_raw_pipeline.pkl
pipeline.predict(raw_test_df)
W&B Registry target = wandb-registry-model/Walmart_NBEATS_Pipeline
```

მაგრამ final Kaggle score დაახლოებით `4700` გამოვიდა. ეს ადასტურებს, რომ N-BEATS ამ dataset-ზე final candidate არ არის.

ჩემი შეფასება: N-BEATS კარგი იყო როგორც deep learning baseline/negative result. მან გვაჩვენა, რომ უფრო complex neural sequence architecture ავტომატურად უკეთესს არ ნიშნავს. ამ data-ზე feature context ბევრად მნიშვნელოვანია.

## TFT-ის გზა

TFT ყველაზე დიდ potential-ად ჩანდა, რადგან შეუძლია static categorical, known future features, external covariates და attention ერთ architecture-ში გააერთიანოს. მაგრამ ყველაზე რთულად სამართავიც ის აღმოჩნდა.

პირველი full-data TFT ძალიან ნელი იყო: Colab-ზე epoch ძალიან დიდხანს მიდიოდა. ამიტომ baseline-ში top subset-ებზე გადავედი.

საწყისი შედეგები:

| Run | Subset | იდეა | WMAE | დასკვნა |
|---|---:|---|---:|---|
| baseline | top 300 | small TFT, calendar only | `7801.90` | pipeline/logging OK, model weak |
| v1 | top 500 | raw sales + external covariates | `6200.95` | baseline-ზე უკეთესი, მაგრამ seasonal naive-ზე სუსტი |
| v2 | top 500 | log target | `6524.68` | log target გააუარესა |
| v3 invalid | top 500 | residual, broken seasonal base | `53035.36` | implementation invalid |
| v3 fixed | top 500 | correct residual | `5212.71` | residual signal useful |
| v4 | top 500 | residual blending, alpha `0.50` | `4728.60` | seasonal naive-ზე უკეთესი |
| v5 | top 500 | fine alpha, alpha `0.40` | `4717.71` | best top-500 TFT |
| v6 | full 3331 | serious full-data | `NaN` | non-finite predictions |
| v7 | top 2000 | stable residual blending, alpha `0.35` | `2379.50` | best valid TFT setup |

TFT-ში ყველაზე მნიშვნელოვანი turning point იყო residual blending. Full residual correction ცუდი იყო:

```text
Prediction = SeasonalNaive52 + PredictedResidual
```

მაგრამ scaled correction კარგად მუშაობდა:

```text
Prediction = SeasonalNaive52 + alpha * PredictedResidual
```

v7-ში საუკეთესო alpha იყო:

```text
alpha = 0.35
best_blend_wmae = 2379.50
seasonal_naive_wmae = 2488.03
improvement_vs_seasonal_naive = +4.36%
```

v6-მა full `3331` series-ზე valid result ვერ მისცა:

```text
best_val_loss = 10628.5859
validation_wmae = NaN
best_blend_wmae = NaN
```

პრობლემა იყო non-finite prediction/evaluation chain. ამიტომ v7-ში დავამატე guards:

```text
raw_prediction_nan_count
raw_prediction_posinf_count
raw_prediction_neginf_count
np.nan_to_num
prediction clipping
residual clipping
```

v7 inference-ში W&B run:

```text
https://wandb.ai/kende23-n-a/Walmart-Recruiting---Store-Sales-Forecasting/runs/rr6jcmci
```

Inference summary:

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

TFT inference-ის მნიშვნელოვანი ნაწილი fallback logic-ია. v7 model მხოლოდ top-2000 series-ზე იყო trained. დანარჩენ Store-Dept pairs-ზე პირდაპირ TFT-ის გამოყენება unsafe იქნებოდა, რადგან categorical encoder unseen groups-ს ვერ მიიღებს. ამიტომ final logic გახდა:

```text
თუ Store-Dept covered by TFT:
    Weekly_Sales = SeasonalNaive52 + 0.35 * PredictedResidual
სხვა შემთხვევაში:
    Weekly_Sales = SeasonalNaive52
```

Kaggle result:

```text
file = tft_v7_submission.csv
public_score = 2979.86060
private_score = 3058.98280
status = Complete after deadline
```

ეს საუკეთესო deep learning Kaggle result გამოვიდა, მაგრამ უნდა აღინიშნოს: submission-ის დაახლოებით `32.87%` fallback-ით იყო შევსებული. ანუ final TFT submission სინამდვილეში hybrid-ია: TFT correction + seasonal naive fallback.

## similarity/difference ერთ ცხრილში

| საკითხი | DLinear | N-BEATS | TFT |
|---|---|---|---|
| ძირითადი input | historical sales | historical sales | sales/residual + covariates |
| Store-Dept identity | `series_bias` | implicit/pipeline level | static categoricals |
| external covariates | v6-ში ვცადე, არ დაეხმარა | პრაქტიკულად არა | ძირითადი ნაწილი |
| holiday handling | WMAE/loss weights | WMAE/weighted loss experiment | known categorical + WMAE |
| seasonality approach | moving average decomposition | learned basis blocks | explicit `SeasonalNaive52` residual |
| საუკეთესო idea | Store-Dept calibration | baseline configuration | residual blending |
| მთავარი პრობლემა | Kaggle horizon-ზე სუსტდება | overfitting early | მძიმე/unstable training |
| inference ფორმა | checkpoint artifact | raw-input pipeline | checkpoint + fallback |
| Kaggle quality | საშუალო | სუსტი | საუკეთესო DL |

## რატომ DLinear validation-ზე უკეთესი ჩანს, მაგრამ Kaggle-ზე TFT ჯობნის

DLinear validation WMAE `1506.28` ძალიან ძლიერია, მაგრამ ეს იყო last-39-week validation on train data, სადაც yearly/series pattern კარგად მუშაობდა. Kaggle test horizon სხვა future period-ია და იქ uncertainty იზრდება.

TFT validation score `2379.50` მხოლოდ top-2000 subset-ზეა და პირდაპირ DLinear-ის `3331` all-series validation-ს არ ედრება. მაგრამ Kaggle-ზე TFT-ის hybrid inference უკეთ გამოვიდა:

```text
TFT private ≈ 3058.98
DLinear ≈ 3500
```

ჩემი ახსნა ასეთია:

- DLinear ძალიან კარგია clean historical continuation-ზე;
- TFT-ს შეუძლია known future calendar/external context და residual correction გამოიყენოს;
- TFT fallback იცავს ისეთ rows-ზე, სადაც model coverage არ არის;
- Kaggle test-ზე hybrid seasonal baseline + controlled neural correction უფრო robust აღმოჩნდა.

## რატომ N-BEATS ჩამორჩა

N-BEATS-ში რამდენიმე რამ არ მუშაობდა:

- საუკეთესო epoch ძალიან ადრე მოდიოდა;
- training loss მცირდებოდა, validation უარესდებოდა;
- longer context `78` კვირა მკვეთრად ცუდი იყო;
- holiday-weighted loss baseline-ს ვერ აჯობდა;
- Optuna-ს partial tuning-მაც baseline ვერ გადალახა;
- Kaggle score `4700` ყველაზე ცუდი იყო.

ჩემი დასკვნაა, რომ N-BEATS ამ project-ში ვერ იღებს საკმარის business signal-ს. ის historical pattern-ს სწავლობს, მაგრამ Walmart sales-ში მხოლოდ historical shape არ კმარა.

## W&B logging-ის შედარება

სამივე მოდელში W&B გვქონდა როგორც experiment notebook-ის ცენტრალური ნაწილი, მაგრამ logging-ის ხასიათი განსხვავდებოდა.

DLinear-ში W&B-ზე ვლოგავდი:

- epoch-level train/validation normalized WMAE;
- original-scale validation WMAE;
- improvement vs seasonal naive;
- prediction table;
- scatter plot;
- checkpoint;
- summary JSON;
- inference submission artifact;
- Model Registry link.

N-BEATS-ში W&B-ზე ვლოგავდი:

- baseline evaluation;
- training history;
- validation predictions;
- weekly errors;
- Optuna trial table;
- best trial diagnostics;
- pipeline artifact/registry logic.

TFT-ში W&B განსაკუთრებით მნიშვნელოვანი გახდა, რადგან model unstable იყო:

- training run metrics;
- best checkpoint;
- residual blending table;
- alpha comparison;
- prediction NaN/Inf diagnostics;
- inference row coverage;
- fallback coverage;
- histogram/plots;
- final submission artifact;
- registry link.

TFT-ში W&B diagnostics-მა პირდაპირ დაგვანახა v6-ის პრობლემა: `NaN` WMAE. ამის შემდეგ v7-ში guard-ები დავამატე და inference-ში უკვე `raw_prediction_nan_count = 0` მივიღე.

## საბოლოო ranking

ჩემი საბოლოო deep learning ranking ასეთია:

### 1. TFT

საუკეთესო Kaggle deep learning result:

```text
public = 2979.86060
private = 3058.98280
```

TFT ყველაზე რთული იყო, მაგრამ residual blending + fallback strategy-მ საუკეთესო final submission მოგვცა.

### 2. DLinear

საუკეთესო validation deep learning model:

```text
validation WMAE = 1506.28
Kaggle ≈ 3500
```

DLinear იყო ყველაზე სუფთა და stable neural model. თუ მხოლოდ validation notebook-ს შევხედავ, DLinear ყველაზე დამაჯერებელია. მაგრამ Kaggle-ზე TFT hybrid submission უკეთესი გამოვიდა.

### 3. N-BEATS

ყველაზე სუსტი final result:

```text
validation WMAE = 2157.98
Kaggle ≈ 4700
```

N-BEATS useful იყო როგორც comparison baseline, მაგრამ final candidate-ად არ ავირჩევდი.

## მთავარი გაკვეთილი

ამ სამმა deep learning მოდელმა ერთი მნიშვნელოვანი რამ აჩვენა: Walmart forecasting-ში neural architecture alone არ არის საკმარისი. საუკეთესო შედეგი არ მოვიდა ყველაზე “deep” ან ყველაზე complex model-იდან პირდაპირი ფორმით. შედეგი მოვიდა მაშინ, როცა neural model სწორად შევზღუდე:

```text
seasonal baseline
+ small learned correction
+ fallback
+ W&B diagnostics
```

DLinear-მ დამანახა, რომ მარტივი historical model ძალიან ძლიერი baseline შეიძლება იყოს. N-BEATS-მა დამანახა, რომ architecture popularity არ ნიშნავს task fit-ს. TFT-მ დამანახა, რომ complex model-ს შეუძლია უკეთესი final submission, მაგრამ მხოლოდ მაშინ, როცა მას explicit seasonal structure, blending და safety fallback აქვს.

საბოლოოდ deep learning ნაწილში ჩემთვის ყველაზე სწორი დასკვნა ასეთია:

```text
best validation stability = DLinear
best Kaggle deep learning submission = TFT
weakest approach = N-BEATS
most important idea = seasonal baseline + controlled correction
```
