# Tree-based models — XGBoost და LightGBM შედარება

ეს დოკუმენტი აჯამებს `models/tree_based/` ფოლდერში გაკეთებულ ორ ძირითად tabular/tree-based მიმართულებას:

- [`xgboost/`](./xgboost/) — XGBoost baseline, engineered training, raw-input pipeline და inference;
- [`lightgbm/`](./lightgbm/) — LightGBM baseline, feature engineering/selection, Optuna training, registry pipeline და inference.

ორივე მოდელი ერთი ამოცანისთვის გამოიყენება: Walmart-ის weekly sales forecasting. ორივე მუშაობს tabular ფორმატზე, იყენებს `train.csv`, `features.csv`, `stores.csv`, `test.csv` მონაცემებს და ორივეს საბოლოო მიზანია Kaggle-ისთვის `Weekly_Sales` prediction-ის გენერირება. მთავარი metric ყველგან არის WMAE, რადგან competition holiday week-ებს 5-ჯერ მეტ წონას აძლევს.

## საერთო სურათი

| ნაწილი | XGBoost | LightGBM |
|---|---|---|
| Baseline | static global XGBoost, 22 feature | simple LightGBM baseline |
| Validation | chronological split; 52 კვირა baseline/tuning-ში, შემდეგ 32 კვირა final candidate-ში | chronological split; 32 კვირა |
| Feature engineering | feature-rich raw-input transformer, 81 feature | sklearn-style feature pipeline, 82 feature → 47 selected |
| Tuning | Optuna, 20 trial | Optuna, 50 trial |
| Best validation result | `1612.13` WMAE final candidate, 32 კვირა | `1573.50` WMAE best Optuna trial, 32 კვირა |
| Registered artifact | `Walmart_XGBoost_Pipeline:champion` | `Walmart_LightGBM_Pipeline:champion` |
| Inference style | raw `test.csv` პირდაპირ pipeline-ში | train history + test ერთად transform lag/rolling feature-ებისთვის |
| Inference run | `7991kiez` | `ychwtp0l` |

რიცხვების შედარებისას ყველაზე ფრთხილი ნაწილი validation window-ია. LightGBM-ის მთავარი შედეგი 32-კვირიან split-ზეა. XGBoost-ს აქვს როგორც 52-კვირიანი comparison (`1935.97`), ისე 32-კვირიანი final candidate (`1612.13`). ამიტომ LightGBM `1573.50` და XGBoost `1612.13` ერთმანეთთან უფრო ახლოს შესადარებელია, ვიდრე XGBoost-ის 52-კვირიანი `1935.97`.

## 1. Baseline ეტაპი

Baseline-ის მიზანი ორივე მოდელში ერთი იყო: ჯერ შეგვექმნა მარტივი tree model, რომელიც გვეტყოდა, რამდენად ძლიერია tabular approach რთული feature engineering-ის გარეშე.

### XGBoost baseline

XGBoost baseline-მა გამოიყენა შედარებით static feature set:

- Store/Dept identity;
- calendar information;
- holiday flag;
- store metadata;
- economic/external columns;
- მარტივი preprocessing.

ამ ეტაპზე ჯერ არ გვქონდა full historical feature pipeline, Optuna tuning ან raw-input production pipeline. მთავარი კითხვა იყო: შეუძლია თუ არა ერთ global XGBoost model-ს Store/Dept series-ების საერთო pattern-ის სწავლა?

დალოგილი baseline result:

```text
Static XGBoost baseline, 52-week validation
WMAE = 2902.29
MAE = 2727.65
Holiday MAE = 3464.41
Non-holiday MAE = 2665.61
```

ამავე 52-კვირიან split-ზე median benchmark იყო:

```text
Median benchmark WMAE = 3305.52
```

ანუ XGBoost baseline უკვე სჯობდა მარტივ historical median-ს. ეს იყო პირველი ნიშანი, რომ tree-based model-ი tabular feature-ებით სასარგებლო მიმართულებაა.

### LightGBM baseline

LightGBM baseline notebook უფრო მინიმალური ექსპერიმენტია, ვიდრე `model_experiment_LightGBM.ipynb`. მისი როლი იყო LightGBM-ის სწრაფი sanity check:

- time-based split;
- Walmart tables merge;
- baseline feature preparation;
- LightGBM training;
- W&B baseline logging;
- feature importance/diagnostic plots.

ამ baseline-ის ზუსტი metric local notebook output-ში აღარ არის შენახული, ამიტომ ამ დოკუმენტში რიცხვს არ ვიგონებთ. მაგრამ baseline-ის ლოგიკა იგივე იყო, რაც XGBoost-ის baseline-ში: ჯერ მარტივი LightGBM უნდა დაგვენახა, შემდეგ კი feature engineering და tuning უნდა შეგვედარებინა მასთან.

### Baseline შედარება

XGBoost baseline-ის შედეგი დოკუმენტირებულად ძლიერია, რადგან 52 კვირაზე median benchmark-ს `12.20%`-ით აჯობა. LightGBM baseline უფრო checkpoint-ის ფუნქციას ასრულებდა: გვჭირდებოდა დაგვემტკიცებინა, რომ LightGBM training/W&B flow მუშაობს, სანამ უფრო რთულ pipeline-ს ავაწყობდით.

ორივე baseline-მა ერთი საერთო დასკვნა მოგვცა: tree model-ს შეუძლია Walmart-ის Store/Dept structure-ის გამოყენება, მაგრამ ძლიერი შედეგისთვის მხოლოდ static feature-ები საკმარისი არ არის. მთავარი improvement history-aware feature engineering-იდან მოდის.

## 2. Training და feature engineering

ორივე მოდელში training-ის განვითარება ერთი მიმართულებით წავიდა: raw Walmart tables → chronological split → feature engineering → Optuna tuning → W&B artifacts. განსხვავება იყო იმაში, როგორ აშენდა feature pipeline და როგორ მოექცა თითოეული მოდელი categorical/history feature-ებს.

## საერთო feature families

ორივე მოდელმა გამოიყენა ერთი და იგივე macro idea:

| Feature family | რატომ არის მნიშვნელოვანი |
|---|---|
| Store/Dept identity | თითოეული Store-Dept წყვილი ცალკე sales series-ს ჰგავს |
| Store metadata | `Type`, `Size` store-level demand scale-ს ხსნის |
| Calendar features | week/month/seasonality demand pattern-ს იჭერს |
| Holiday features | WMAE holiday rows-ს მეტ წონას აძლევს |
| Markdown features | promotion context განსაკუთრებით holiday პერიოდში მუშაობს |
| Historical aggregates | Store/Dept-ის ტიპურ sales level-ს აძლევს მოდელს |
| Lag/rolling/yearly history | წინა წლის და წინა პერიოდების demand strongest signal-ია |

ყველაზე მნიშვნელოვანი საერთო გაკვეთილი იყო: sales level-ის გარეშე მოდელი სუსტია. Calendar, markdown და economic columns სასარგებლოა, მაგრამ Store/Dept historical level და yearly seasonality ყველაზე ძლიერ signal-ს იძლევა.

## XGBoost training logic

XGBoost-ის engineered notebook უფრო production-style მიმართულებით განვითარდა. მთავარი მიზანი იყო არა მხოლოდ კარგი validation score, არამედ ისეთი pipeline, რომელიც inference-ზე raw `test.csv`-ს მიიღებდა.

XGBoost feature-rich მოდელში მივიღეთ:

```text
81 engineered feature
SalesLag52
Store/Dept aggregate statistics
holiday proximity/context
markdown interactions
calendar/cyclical features
external economic/store metadata
```

Optuna tuning:

```text
20 trials
Best 52-week validation WMAE = 1935.97
Best iteration = 2219
MAE = 1791.90
RMSE = 4398.17
Holiday MAE = 2399.72
Non-holiday MAE = 1740.71
```

საუკეთესო 52-კვირიანი XGBoost trial-ის ხასიათი:

```text
learning_rate = 0.02536
max_depth = 12
subsample = 0.8595
colsample_bytree = 0.7046
reg_lambda = 9.8423
```

აქ XGBoost-ს სჭირდებოდა დაბალი learning rate და ბევრი boosting round. ეს ნიშნავს, რომ მოდელი ნელა, მაგრამ სტაბილურად სწავლობდა რთულ Store/Dept/holiday/history interaction-ებს.

შემდეგი final-candidate run უკვე 32-კვირიან validation-ზე შესრულდა:

```text
Final candidate XGBoost, 32-week validation
WMAE = 1612.13
MAE = 1585.96
Holiday MAE = 1821.02
Non-holiday MAE = 1578.36
Best iteration = 999
```

ეს final-candidate შედეგი უკეთესი რიცხვია, მაგრამ 52-კვირიან `1935.97`-თან პირდაპირ არ უნდა შევადაროთ, რადგან split განსხვავებულია. 32-კვირიანი split უფრო ახლოა LightGBM-ის setup-თან.

## LightGBM training logic

LightGBM notebook-ში feature engineering უფრო modular sklearn-style transformer-ებად დაიწერა:

- `WalmartFeatureCleaner`;
- `CalendarFeatureTransformer`;
- `WalmartHolidayFeatureTransformer`;
- `MarkdownFeatureTransformer`;
- `InteractionFeatureTransformer`;
- `HistoricalAggregateTransformer`;
- `LagRollingFeatureTransformer`.

LightGBM-ისთვის დამატებით მნიშვნელოვანი იყო pandas categorical dtype, რადგან LightGBM categorical feature-ებს native-ად ამუშავებს. ამის გამო `Store`, `Dept`, `Type` one-hot encoding-ის გარეშე გამოიყენება.

Feature engineering stage:

```text
train rows = 326,856
validation rows = 94,714
feature count = 82
categorical feature count = 5
```

Feature selection:

```text
input feature count = 82
selected feature count = 47
dropped feature count = 35
selected ratio = 0.573
```

LightGBM-ში feature selection model-based იყო: ჯერ LightGBM train გახდა ყველა engineered feature-ზე, შემდეგ zero-importance feature-ები ამოიღო. ეს XGBoost-ისგან განსხვავდება: XGBoost final pipeline-ში feature-rich 81-column design პირდაპირ გამოიყენებოდა, ხოლო LightGBM-ში feature set უფრო aggressively შემცირდა.

Optuna tuning:

```text
50 trials
Best trial = 46
Validation Weighted MAE = 1573.4988
Validation MAE = 1543.4832
```

საუკეთესო LightGBM hyperparameter-ები:

```python
{
    "learning_rate": 0.08117866851143801,
    "num_leaves": 196,
    "max_depth": 19,
    "min_child_samples": 98,
    "subsample": 0.9817092354210323,
    "colsample_bytree": 0.8206871721053576,
    "reg_alpha": 0.00031014058676548666,
    "reg_lambda": 0.01501347092737337,
}
```

LightGBM-ის pattern XGBoost-ისგან განსხვავებული იყო. აქ `n_estimators = 100` fixed იყო, ამიტომ low learning rate-ები აშკარად underfit გახდა. საუკეთესო trial-ები იყენებდნენ უფრო მაღალ learning rate-ს (`0.06–0.09`), ბევრ leaves-ს და deep trees-ს. ანუ LightGBM სწრაფად სწავლობდა 100 boosting round-ში, ხოლო XGBoost-ის საუკეთესო 52-week setup უფრო ნელ learning rate-ს და ბევრ iteration-ს ეყრდნობოდა.

## Training result comparison

| მოდელი | Validation | Feature setup | Tuning | WMAE | შენიშვნა |
|---|---:|---|---:|---:|---|
| XGBoost median benchmark | 52 კვირა | Store/Dept median | არა | `3305.52` | simple reference |
| XGBoost baseline | 52 კვირა | 22 static feature | არა | `2902.29` | median-ზე უკეთესი |
| XGBoost engineered | 52 კვირა | 81 feature | 20 trial | `1935.97` | baseline-ზე დიდი improvement |
| XGBoost final candidate | 32 კვირა | 81 feature | selected params | `1612.13` | LightGBM split-თან უფრო შესადარებელი |
| LightGBM engineered | 32 კვირა | 82 → 47 selected feature | 50 trial | `1573.50` | საუკეთესო tree-based validation score |
| LightGBM registered pipeline | 32 კვირა | selected features + saved pipeline | champion artifact | `1575.94` | registry-ში შენახული champion version |

ამ ცხრილში მთავარი comparison არის:

```text
XGBoost final candidate: 1612.13
LightGBM best training trial: 1573.50
LightGBM registered champion: 1575.94
```

LightGBM ოდნავ უკეთესი გამოვიდა 32-კვირიან validation-ზე. სავარაუდო მიზეზები:

1. LightGBM categorical feature-ებს native-ად უკეთ იყენებს ამ dataset-ში.
2. Feature selection-მა noisy/zero-importance feature-ები მოაშორა.
3. 50 Optuna trial-მა search space უფრო ფართოდ დაფარა, ვიდრე XGBoost-ის 20 trial.
4. LightGBM-ის leaf-wise tree growth კარგად ერგება Store/Dept-level heterogeneous patterns-ს.

მაგრამ XGBoost-საც ძლიერი მხარე ჰქონდა: raw-input pipeline უფრო self-contained გამოვიდა. XGBoost pipeline საკუთარ თავში ინახავს external feature table-ს, stores metadata-ს, observed training history-ს, aggregate mappings-ს და final feature order-ს, ამიტომ inference notebook მხოლოდ `test.csv`-ს კითხულობს.

## 3. W&B run-ები და artifacts

ორივე მოდელში W&B არ იყო მხოლოდ chart logger. ის გამოიყენებოდა როგორც experiment lineage და artifact storage.

### XGBoost მთავარი run-ები

| Run | როლი | შედეგი |
|---|---|---|
| `pc46skfo` | static baseline | `2902.29` WMAE, 52 კვირა |
| `mosu9yww` | Optuna tuning summary | 20 trial leaderboard |
| `d2j50ses` | engineered best model | `1935.97` WMAE, 52 კვირა |
| `1v4zt3kx` / `unahq2uk` | final-candidate validation | `1612.13` WMAE, 32 კვირა |
| `5rxamhq3` | final-candidate full refit | submission/refit artifacts |
| `7991kiez` | inference | champion pipeline → submission |

XGBoost-ში W&B job types უფრო detailed იყო:

- `data-preparation`;
- `hyperparameter-tuning`;
- `tuning-summary`;
- `train-best-model`;
- `full-refit`;
- `model-registration`;
- `inference`.

### LightGBM მთავარი run-ები

| Run | როლი | შედეგი |
|---|---|---|
| `llzq416u` | feature engineering | 82 feature, train/validation profile |
| `i4n1gdp7` | feature selection | 82 → 47 selected feature |
| `lightgbm-optuna-trial-46` | best Optuna trial | `1573.50` WMAE |
| `LightGBM_Best_Model_Registry` | registry packaging | `Walmart_LightGBM_Pipeline` |
| `ychwtp0l` | inference | champion pipeline → submission |

LightGBM-ის inference artifact-ში ჩანს:

```text
artifact = Walmart_LightGBM_Pipeline:champion
artifact_version = v0
validation_weighted_mae = 1575.9437
selected_features = 47
submission_rows = 115064
```

## 4. Inference comparison

ორივე მოდელმა საბოლოოდ inference notebook მიიღო, მაგრამ inference-ის დიზაინი განსხვავებულია.

## XGBoost inference

XGBoost inference notebook კითხულობს მხოლოდ raw `test.csv`-ს:

```text
test.csv
    ↓
W&B Registry champion pipeline
    ↓
internal feature engineering
    ↓
XGBoost predict
    ↓
submission_xgboost_champion.csv
```

Registry:

```text
wandb-registry-model/Walmart_XGBoost_Pipeline:champion
```

Pipeline file:

```text
walmart_xgboost_raw_pipeline.pkl
```

ეს pipeline inference-ისთვის ყველაზე სუფთაა, რადგან test notebook-ში feature engineering class-ების გადაწერა აღარ ხდება. `cloudpickle`-ით შენახული pipeline შეიცავს notebook-ში განსაზღვრულ custom class-ებსაც. დამატებით fresh-process smoke test ამოწმებს, რომ pipeline ცალკე Python process-შიც იტვირთება.

XGBoost inference run:

```text
run = 7991kiez
submission path = /content/drive/MyDrive/walmart_competition_inference/xgboost/submission_xgboost_champion.csv
```

## LightGBM inference

LightGBM inference notebook ასევე Registry-დან ტვირთავს champion pipeline-ს:

```text
wandb-registry-model/Walmart_LightGBM_Pipeline:champion
```

Pipeline file:

```text
lightgbm_best_pipeline.pkl
```

LightGBM-ის მნიშვნელოვანი განსხვავება ის არის, რომ registered feature pipeline ჯერ კიდევ historical `Weekly_Sales`-ს იყენებს lag/rolling feature-ებისთვის. ამიტომ inference-ში notebook აკეთებს:

```text
train history rows + test rows
    ↓
feature_pipeline.transform(combined_rows)
    ↓
keep only transformed test rows
    ↓
pipeline.model.predict(selected_features)
    ↓
submission_lightgbm_registry.csv
```

ეს სწორია, რადგან lag/rolling feature-ებს history სჭირდება. მაგრამ operationally XGBoost-ზე ოდნავ ნაკლებად self-contained არის, რადგან inference notebook-ს `train.csv`, `features.csv`, `stores.csv` და `test.csv` ერთად სჭირდება. XGBoost-ის inference flow ამ მხრივ უფრო მარტივია: registered pipeline-ში მეტი state არის ჩაშენებული.

LightGBM inference run:

```text
run = ychwtp0l
submission path = /content/artifacts/lightgbm_inference/submission_lightgbm_registry.csv
prediction_mean = 14768.19
prediction_std = 17675.85
prediction_min = 0
prediction_max = 179720.57
```

## 5. მთავარი მსგავსებები

ორივე tree-based მოდელში საერთო იყო:

1. **Chronological validation.** Random split არ გამოიყენება, რადგან forecasting-ში მომავალი არ უნდა მოხვდეს training-ში.
2. **WMAE metric.** Holiday rows იღებს weight `5`, non-holiday rows weight `1`.
3. **Global model.** თითო Store/Dept-ზე ცალკე მოდელი არ გვიწერია; ერთი მოდელი სწავლობს ყველა series-ის pattern-ს.
4. **Feature engineering არის მთავარი.** მოდელის არქიტექტურა საკმარისი არ იყო; result-ის დიდი ნაწილი history, aggregates, holiday და markdown feature-ებიდან მოვიდა.
5. **W&B artifacts.** ორივე მოდელში ინახება model/pipeline, config, feature metadata, diagnostics და submission.
6. **Registry-based inference.** საბოლოო inference მოდელს training notebook-იდან კი არ აშენებს, არამედ W&B Registry-დან იღებს champion artifact-ს.

## 6. მთავარი განსხვავებები

| თემა | XGBoost | LightGBM |
|---|---|---|
| Categorical handling | encoded/engineered feature pipeline | pandas `category` native support |
| Feature count | 81 final features | 82 engineered → 47 selected |
| Tuning behavior | low learning rate + many rounds worked well | high learning rate needed because 100 estimators fixed |
| Best tree shape | deep trees, strong L2 regularization | many leaves, deep trees, high min_child_samples |
| Inference portability | more self-contained raw-input sklearn Pipeline | needs train history + test together for lag/rolling transform |
| Registry artifact | raw pipeline with feature transformer + model | bundle with feature_pipeline, selected_features, model |
| Best documented score | `1612.13` on 32-week validation | `1573.50` best trial; `1575.94` registered champion |

## 7. რატომ მიიღო LightGBM-მ უკეთესი validation score

LightGBM-ის უკეთესი score სავარაუდოდ ერთი მიზეზით არ აიხსნება. რამდენიმე ფაქტორი ერთად მოქმედებს:

- LightGBM native categorical handling ამ dataset-ში სასარგებლოა, რადგან `Store`, `Dept`, `Type` ძალიან ძლიერი identity signal-ებია.
- Feature selection-მა feature space გაასუფთავა: 82-დან 47 feature დარჩა.
- Optuna-ს 50 trial უფრო ფართო search იყო.
- LightGBM-ის leaf-wise growth კარგად იჭერს heterogeneous demand patterns-ს.
- Validation split 32 კვირაა, სადაც historical/lag/rolling feature-ები ძალიან ძლიერად მუშაობს.

მაგრამ ეს არ ნიშნავს, რომ XGBoost სუსტია. XGBoost feature-rich მოდელმა 52-კვირიან validation-ზე baseline `2902.29`-დან `1935.97`-მდე ჩამოიყვანა WMAE, ხოლო 32-კვირიან final candidate-ზე `1612.13` მიიღო. XGBoost-ის pipeline design უფრო გამართულია raw test portability-ის მხრივ.

## 8. საბოლოო დასკვნა

Tree-based მიმართულებაში ორივე მოდელმა ერთი და იგივე ძირითადი ამბავი აჩვენა: Walmart forecasting-ში მთავარი signal არის Store/Dept historical level + yearly seasonality + holiday/promotion context. Model choice მნიშვნელოვანია, მაგრამ feature design კიდევ უფრო მნიშვნელოვანია.

ამ ეტაპზე validation score-ით LightGBM ლიდერობს:

```text
LightGBM best trial WMAE = 1573.50
LightGBM registered champion WMAE = 1575.94
XGBoost final candidate WMAE = 1612.13
```

inference portability-ით XGBoost-ის pipeline უფრო სუფთაა, რადგან raw `test.csv`-ს პირდაპირ იღებს. LightGBM-ს უკეთესი score აქვს, მაგრამ inference-ში train history-საც იყენებს lag/rolling feature-ების უსაფრთხოდ შესაქმნელად.

ამიტომ tree-based comparison-ის practical conclusion ასეთია:

- **Best validation score:** LightGBM.
- **Cleanest raw-input inference contract:** XGBoost.
- **Common winning idea:** historical aggregates + yearly lag + holiday-aware features.
- **Production requirement:** W&B Registry artifact უნდა იყოს inference-ის წყარო, არა ხელით გადაწერილი preprocessing.

