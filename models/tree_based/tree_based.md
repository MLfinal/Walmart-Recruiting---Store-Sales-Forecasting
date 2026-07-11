# Tree-based models — XGBoost და LightGBM შედარება

ეს დოკუმენტი აჯამებს `models/tree_based/` ფოლდერში გაკეთებულ ორ ძირითად tabular/tree-based მიმართულებას:

- [`xgboost/`](./xgboost/) — XGBoost baseline, engineered training, raw-input pipeline და inference;
- [`lightgbm/`](./lightgbm/) — LightGBM baseline, feature engineering/selection, Optuna training, registry pipeline და inference.

ორივე მოდელი ერთი ამოცანისთვის გამოიყენება: Walmart-ის weekly sales forecasting. ორივე მუშაობს tabular ფორმატზე, იყენებს `train.csv`, `features.csv`, `stores.csv`, `test.csv` მონაცემებს და ორივეს საბოლოო მიზანია Kaggle-ისთვის `Weekly_Sales` prediction-ის გენერირება. მთავარი metric ყველგან არის WMAE, რადგან competition holiday week-ებს 5-ჯერ მეტ წონას აძლევს.

## Final Kaggle submission შედარება

Tree-based მიმართულებაში final Kaggle submission-ებმა ასეთი შედეგი მისცა:

```text
XGBoost Kaggle score                  = 2806
LightGBM safe SalesLag52 score       ≈ 3600
LightGBM corrected XGBoost-aligned   ≈ 3490
LightGBM latest FE/FS score           = 3500
LightGBM final GPU FE/FS + refit      = 2809
```

საბოლოო Kaggle შედეგით XGBoost ფორმალურად უკეთესია, მაგრამ სხვაობა მხოლოდ `3` WMAE-ია. ორივე gradient boosted tree family-ს ეკუთვნის და ამ პროექტში პრაქტიკულად ერთ performance დონეზე მივიდა. შედეგების ევოლუცია pipeline/design ცვლილებებმა განსაზღვრა:

- XGBoost-ის feature engineering თავიდანვე უფრო Kaggle-safe იყო;
- XGBoost იყენებდა safe `SalesLag52` yearly lag-ს და არა future sales-ზე დამოკიდებულ short lag/rolling feature-ებს;
- XGBoost raw-input pipeline უფრო self-contained იყო: test inference-ში preprocessing logic, training history, aggregate mappings და feature order ერთ artifact-ში იყო მოქცეული;
- validation/inference behavior უკეთ ემთხვეოდა ერთმანეთს;
- LightGBM-ის პირველმა ვერსიამ validation-ზე უკეთესი WMAE აჩვენა, მაგრამ Kaggle-ზე ცუდად generalized, რადგან unsafe lag/rolling feature-ები validation score-ს ზედმეტად ალამაზებდა;
- final LightGBM-მა იგივე safe FE პრინციპები, ყველა engineered feature, full-data refit და სრული observed history გააერთიანა, რის შემდეგაც Kaggle score `2809` გახდა.

LightGBM-ის safe `SalesLag52` retrain-მა Kaggle score გააუმჯობესა:

```text
LightGBM old unsafe setup ≈ 6200
LightGBM safe SalesLag52 setup ≈ 3600
```

Final შედეგში XGBoost-ის უპირატესობა მინიმუმამდე შემცირდა:

```text
XGBoost = 2806
LightGBM final GPU FE/FS + refit = 2809
```

ისტორიულად XGBoost-ის feature engineering და inference contract უკეთ იმეორებდა რეალურ Kaggle test სიტუაციას, რის გამოც ძველი LightGBM `3490–3500` დონეზე რჩებოდა. Final LightGBM-ში full-data refit-ისა და სრული observed history-ის დამატების შემდეგ ეს განსხვავება თითქმის გაქრა: `2806` და `2809` პრაქტიკულად თანაბარი შედეგებია.

პრაქტიკული არჩევანი:

- **Formal tree-based champion:** XGBoost, მხოლოდ 3 WMAE უპირატესობით.
- **Practical result:** XGBoost და final LightGBM ერთ performance დონეზეა.
- **LightGBM status:** პრაქტიკულად XGBoost-ის თანაბარი final candidate; სხვაობა მხოლოდ 3 WMAE.

## საერთო სურათი

| ნაწილი | XGBoost | LightGBM |
|---|---|---|
| Baseline | static global XGBoost, 22 feature | simple LightGBM baseline |
| Validation | chronological split; 52 კვირა baseline/tuning-ში, შემდეგ 32 კვირა final candidate-ში | chronological split; 32 კვირა |
| Feature engineering | feature-rich raw-input transformer, 81 feature | sklearn-style safe feature pipeline, final run-ში 80 feature |
| Feature selection | ყველა final feature ინარჩუნებს | final run-ში importance diagnostic-ია და ყველა 80 feature რჩება |
| Tuning | Optuna, 20 trial | final time-budgeted GPU Optuna: 4 completed trial |
| Best validation result | `1612.13` WMAE final candidate, 32 კვირა | final retrain: `1575.15`; Optuna trial best: `1567.70` |
| Kaggle score | `2806` | final: `2809`; previous: ≈`3490` / `3500` |
| Registered artifact | `Walmart_XGBoost_Pipeline:champion` | `Walmart_LightGBM_Pipeline:champion` |
| Inference style | raw `test.csv` პირდაპირ pipeline-ში | registered bundle ქმნის safe `SalesLag52`-ს stored history-დან |
| Inference run | `7991kiez` | latest registry inference run |

რიცხვების შედარებისას ყველაზე ფრთხილი ნაწილი validation window და feature availability-ია. LightGBM-ის ძველი `1573.50` validation result unsafe lag/rolling feature-ებით იყო მიღებული და Kaggle-ზე არ დადასტურდა. Final safe LightGBM-ის `1575.15` retrain validation-ს უკვე `2809` Kaggle score მოჰყვა. Kaggle-ის მიხედვით XGBoost ფორმალურად ლიდერობს, მაგრამ სხვაობა მხოლოდ 3 WMAE-ია.

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
| LightGBM engineered old | 32 კვირა | unsafe lag/rolling + selected features | 50 trial | `1573.50` | validation optimistic იყო |
| LightGBM safe retrain | 32 კვირა | safe `SalesLag52` + selected features | 50 trial | `1633.37` | Kaggle-ზე უკეთ generalized |
| LightGBM corrected XGBoost-aligned | 32 კვირა | XGBoost-aligned safe features | 50 trial | `1615.45` | Kaggle score ≈`3490` |
| LightGBM latest FE/FS | — | ახალი FE/FS setup | — | არ არის მოწოდებული | Kaggle score `3500`; improvement არ არის |
| LightGBM final GPU FE/FS + refit | 32 კვირა | all safe engineered features + full-data refit | 4 completed trial | `1575.15` | Kaggle score `2809` |

ამ ცხრილში მთავარი comparison არის:

```text
XGBoost final candidate: 1612.13
LightGBM old unsafe validation: 1573.50
LightGBM safe validation: 1633.37
LightGBM corrected validation: 1615.45
XGBoost Kaggle: 2806
LightGBM safe Kaggle: ≈3600
LightGBM corrected XGBoost-aligned Kaggle: ≈3490
LightGBM latest FE/FS Kaggle: 3500
LightGBM final GPU FE/FS + refit Kaggle: 2809
```

ძველ LightGBM-ს validation-ზე უკეთესი რიცხვი ჰქონდა, მაგრამ ეს შედეგი Kaggle-ზე არ დადასტურდა. safe retrain-მა leakage შეამცირა, corrected XGBoost-aligned run-მა Kaggle score დაახლოებით `3490`-მდე ჩამოიყვანა, ხოლო მომდევნო FE/FS run-მა `3500` მიიღო. საბოლოო GPU FE/FS + full-data refit-მა ეს plateau დაარღვია და Kaggle score `2809`-მდე გააუმჯობესა.

Final LightGBM-ის ძლიერი მხარეები:

1. LightGBM categorical feature-ებს native-ად კარგად იყენებს.
2. Importance diagnostic-ად გამოიყენება, მაგრამ final model ყველა 80 engineered feature-ს ინარჩუნებს.
3. Time-budgeted Optuna-მ მხოლოდ 4 trial-ში იპოვა ძლიერი configuration (`1567.70` trial WMAE).
4. Leaf-wise tree growth კარგად ერგება Store/Dept-level heterogeneous patterns-ს.

XGBoost-ის raw-input pipeline კვლავ კარგი reference-ია: ის საკუთარ თავში ინახავს external feature table-ს, stores metadata-ს, observed training history-ს, aggregate mappings-ს და final feature order-ს. Final LightGBM bundle-იც იმავე production contract-ს მიუახლოვდა და raw `test.csv`-ს პირდაპირ ამუშავებს.

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
| `lightgbm-optuna-trial-46` | old unsafe Optuna best | `1573.50` validation WMAE, Kaggle-ზე unreliable |
| `lightgbm-optuna-trial-42` | safe Optuna best | `1633.37` validation WMAE, Kaggle `3600` |
| `xdhdtwd1` | final GPU trial 0 / seeded configuration | `1604.51` validation WMAE, best iteration 833 |
| `1ytk7503` | final GPU Optuna best trial 1 | `1567.70` validation WMAE, trial best iteration 1195 |
| `LightGBM_Best_Model_Registry` | registry packaging | `Walmart_LightGBM_Pipeline` |
| latest final inference run | inference | full-data registry pipeline → Kaggle `2809` submission |

ძველი LightGBM inference artifact-ის historical example:

```text
artifact = Walmart_LightGBM_Pipeline:champion
artifact_version = v0
validation_weighted_mae = 1633.3693
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

LightGBM-ის ძველი inference flow train history + test rows ერთად transform-ს აკეთებდა, რადგან lag/rolling feature-ებს history სჭირდებოდა. safe retrain-ის შემდეგ registered bundle უფრო სუფთა გახდა: ის stored observed history-ს ინახავს და `predict()` დროს თვითონ ქმნის safe `SalesLag52` feature-ს.

```text
raw test rows + stores/features merge
    ↓
registered LightGBM bundle
    ↓
stored observed history → SalesLag52
    ↓
feature_pipeline.transform(...)
    ↓
selected feature order
    ↓
model.predict(...)
    ↓
submission_lightgbm_registry.csv
```

ეს ძველ flow-ზე უკეთესია, რადგან inference აღარ ეყრდნობა future `Weekly_Sales`-ზე დამოკიდებულ rolling/short-lag feature-ებს. მიუხედავად ამისა, XGBoost-ის pipeline მაინც უფრო სუფთა და უფრო ადრე დამტკიცებული raw-input contract იყო.

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
| Feature count | 81 final features | final run: 80 engineered feature, filtering-ის გარეშე |
| Tuning behavior | low learning rate + many rounds worked well | targeted GPU tuning, მაქს. 1200 round + early stopping |
| Best tree shape | deep trees, strong L2 regularization | many leaves, deep trees, high min_child_samples |
| Inference portability | more self-contained raw-input sklearn Pipeline | registry bundle owns history and creates safe `SalesLag52` |
| Registry artifact | raw pipeline with feature transformer + model | bundle with feature_pipeline, selected_features, model |
| Best documented score | `1612.13` validation; `2806` Kaggle | `1575.15` final validation; `2809` Kaggle |

## 7. როგორ გაუთანაბრდა final LightGBM XGBoost-ს Kaggle-ზე

LightGBM-ის ძველი validation/Kaggle mismatch ერთი მიზეზით არ აიხსნება. ისტორიულად რამდენიმე ფაქტორი ერთად მოქმედებდა:

- LightGBM native categorical handling ამ dataset-ში სასარგებლოა, რადგან `Store`, `Dept`, `Type` ძალიან ძლიერი identity signal-ებია.
- ძველ run-ში feature selection-მა feature space 82-დან 47-მდე შეამცირა, მაგრამ final run-ში ყველა 80 safe feature შევინარჩუნეთ.
- final targeted Optuna-მ წინა ძლიერ region-ში მხოლოდ 4 trial გამოიყენა.
- LightGBM-ის leaf-wise growth კარგად იჭერს heterogeneous demand patterns-ს.
- ძველ setup-ში validation split-ზე historical/lag/rolling feature-ები ზედმეტად ძლიერად მუშაობდა, რადგან validation `Weekly_Sales` უკვე ცნობილი იყო.

XGBoost feature-rich მოდელმა 52-კვირიან validation-ზე baseline `2902.29`-დან `1935.97`-მდე ჩამოიყვანა WMAE, 32-კვირიან final candidate-ზე `1612.13` მიიღო და Kaggle-ზე `2806`-მდე მივიდა.

ძველ ეტაპებზე XGBoost-ის უპირატესობის მთავარი მიზეზები იყო:

- safe yearly history feature;
- ნაკლები validation-test mismatch;
- raw-input pipeline-ის უკეთესი contract;
- უფრო მკაცრი leakage control;
- inference-ზე preprocessing drift-ის ნაკლები რისკი.

Final LightGBM-მა ეს უპირატესობები გადაიღო: safe yearly history, ყველა labeled row-ზე refit, სრული observed history, raw-input Registry bundle და feature order-ის ფიქსაცია. ამასთან, fixed 100-tree underfitting შეიცვალა მაქსიმუმ 1200 round-ითა და early stopping-ით. შედეგად LightGBM Kaggle-ზე `3500`-დან `2809`-მდე გაუმჯობესდა და XGBoost-ს პრაქტიკულად გაუთანაბრდა.

## 8. საბოლოო დასკვნა

Tree-based მიმართულებაში ორივე მოდელმა ერთი და იგივე ძირითადი ამბავი აჩვენა: Walmart forecasting-ში მთავარი signal არის Store/Dept historical level + yearly seasonality + holiday/promotion context. Model choice მნიშვნელოვანია, მაგრამ feature design კიდევ უფრო მნიშვნელოვანია.

ამ ეტაპზე Kaggle score-ით XGBoost ლიდერობს:

```text
XGBoost Kaggle score = 2806
LightGBM final GPU FE/FS Kaggle score = 2809
```

Final LightGBM-მა Kaggle-ზე `2809` მიიღო და XGBoost-ის `2806` შედეგს მხოლოდ 3 point-ით ჩამორჩა. ფორმალურად XGBoost კვლავ ლიდერია, მაგრამ პრაქტიკულად ორი final tree-based model ერთ დონეზეა. LightGBM-ის მთავარი improvement მოვიდა full-data refit-იდან, safe `SalesLag52`-ის სრული observed history-დან, fixed 100-tree limit-ის მოხსნიდან და training/inference contract-ის გასწორებიდან.

ამიტომ tree-based comparison-ის practical conclusion ასეთია:

- **Formal best Kaggle score:** XGBoost `2806`, მხოლოდ 3 WMAE უპირატესობით.
- **Latest and best LightGBM:** final GPU FE/FS + full-data refit, Kaggle `2809`.
- **Practical champion tier:** XGBoost და LightGBM ერთად.
- **Best reliable LightGBM validation:** `1575.15` retrain WMAE (`1567.70` Optuna trial best).
- **Common winning idea:** historical aggregates + yearly lag + holiday-aware features.
- **Production requirement:** W&B Registry artifact უნდა იყოს inference-ის წყარო, არა ხელით გადაწერილი preprocessing.
