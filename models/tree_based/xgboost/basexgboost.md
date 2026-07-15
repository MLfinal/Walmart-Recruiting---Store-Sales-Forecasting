# Baseline XGBoost — სრული ახსნა

ეს დოკუმენტი აღწერს [`baseline_xgboost.ipynb`](./baseline_xgboost.ipynb)-ის მიმდინარე, Colab-ში გაშვებულ ვერსიას: რას აკეთებს თითოეული cell, რას ნიშნავს კოდის თითოეული მნიშვნელოვანი ხაზი და რა შედეგი დაბრუნდა გაშვებისას.

## მოკლე შედეგი

- მოდელი: გლობალური `XGBRegressor`
- სამიზნე: `Weekly_Sales`
- validation: ბოლო 52 კვირა
- მთავარი მეტრიკა: Kaggle-ის WMAE
- median benchmark WMAE: **3305.52**
- XGBoost validation WMAE: **2902.29**
- გაუმჯობესება benchmark-თან შედარებით: **12.20%**
- საუკეთესო იტერაცია: **2999**
- W&B run: [xgboost-static-baseline](https://wandb.ai/kende23-n-a/Walmart-Recruiting---Store-Sales-Forecasting/runs/pc46skfo)

## Kaggle submission შენიშვნა

Final XGBoost pipeline-მა Kaggle-ზე მიიღო:

```text
Kaggle score: 2806
```

ეს baseline notebook-ის validation WMAE-ს პირდაპირ არ უდრის, რადგან final submission უკვე უფრო განვითარებული feature set-ით და registry pipeline-ით იყო გაშვებული. ჩემთვის მთავარი დასკვნა არის ის, რომ XGBoost-მა ყველაზე კარგად გადაიტანა validation logic Kaggle test-ზე. LightGBM safe retrain-მა `3600` score მიიღო, DLinear-მა `3500`, N-BEATS-მა კი `4700`; ამ შედარებაში XGBoost ყველაზე საიმედო აღმოჩნდა.

## მონაცემებისა და მოდელის ნაკადი

```text
train/test CSV
      │
      ├── features.csv ── Temperature, Fuel_Price, MarkDown, CPI...
      └── stores.csv   ── Type, Size
              │
              ▼
       build_features()
              │
              ▼
   ქრონოლოგიური train/validation split
              │
       ┌──────┴──────┐
       ▼             ▼
 median benchmark   XGBoost
                       │
                       ├── validation metrics → W&B
                       ├── model + metadata → W&B Artifact
                       └── full-data refit → Kaggle submission
```

## მნიშვნელოვანი შენიშვნები

1. ახალ Colab runtime-ში **Cell 7 (`drive.mount`) Cell 6-მდე უნდა გაეშვას**. მიმდინარე notebook-ში მონაცემების წაკითხვის cell mount-cell-ზე ადრე დგას. არსებული გაშვება წარმატებული იყო, რადგან Drive უკვე mounted მდგომარეობაში აღმოჩნდა.
2. `CONFIG["run_final_refit"]` არის `True`, ამიტომ საბოლოო მოდელის cell რეალურად გაეშვა და submission შეიქმნა. Cell 23-ის ტექსტი, რომელიც `False` მნიშვნელობას აღწერს, მიმდინარე კონფიგურაციას აღარ ემთხვევა.
3. საუკეთესო იტერაცია არის `2999`, ანუ ზუსტად ბოლო შესაძლო იტერაცია. შესაბამისად, `early stopping` არ გააქტიურდა და validation loss ჯერ კიდევ მცირდებოდა. ეს ნიშნავს, რომ `n_estimators=3000` შესაძლოა ზედა ზღვარი იყოს და არა რეალური ოპტიმუმი.
4. prediction-ისას მიღებული CUDA warning ნიშნავს, რომ მოდელი GPU-ზეა, ხოლო `pandas` მონაცემები CPU-ზე. XGBoost დროებით იყენებს `DMatrix` fallback-ს. ეს შეიძლება იყოს უფრო ნელი და მოიხმაროს მეტი მეხსიერება, მაგრამ გამოთვლილ პროგნოზს ან WMAE-ს არ აუქმებს.
5. Notebook-ში execution count-ები შენახული არ არის, თუმცა output-ები შენახულია. ამიტომ ქვემოთ აღწერილი შედეგები უშუალოდ output-ებიდანაა აღებული.

---

## Cell 0 — Colab badge

```html
<a href="..."><img ... /></a>
```

- `<a>` ქმნის ბმულს, რომელმაც notebook Google Colab-ში უნდა გახსნას.
- `target="_parent"` ბმულს მიმდინარე browser context-ში ხსნის.
- `<img>` აჩვენებს “Open in Colab” badge-ს.
- მიმდინარე URL repository root-ში ეძებს `baseline_xgboost.ipynb`-ს, რეალური ფაილი კი `models/tree_based/xgboost/`-შია. თუ badge არ იმუშავებს, URL-ში სრული path უნდა ჩაიწეროს.

## Cell 1 — მიზანი და მეთოდოლოგია

Markdown cell აღწერს ექსპერიმენტის დიზაინს:

- გამოიყენება ერთი გლობალური XGBoost მოდელი ყველა `Store`/`Dept` სერიისთვის;
- random split-ის ნაცვლად გამოიყენება დროითი split;
- holiday rows იღებს წონას 5, დანარჩენი rows — 1;
- იგივე წონები გამოიყენება training-სა და validation-ში;
- W&B ინახავს კონფიგურაციას, metric-ებსა და artifacts-ს;
- lag features შეგნებულად არ გამოიყენება.

Lag-ების არგამოყენების მიზეზი მნიშვნელოვანია: თუ validation-ის მომდევნო კვირის lag-ში რეალურ validation target-ს ჩავდებთ, მოდელი მიიღებს ინფორმაციას, რომელიც Kaggle test-ის პროგნოზირებისას არ ექნება. ეს target leakage-ს და ზედმეტად კარგ validation score-ს გამოიწვევს.

## Cell 2 — ბიბლიოთეკების დაყენება

```python
%pip install -q "xgboost>=3.0,<4" "wandb>=0.19,<1"
```

- `%pip` არის Jupyter/Colab magic command.
- `-q` ამცირებს installation log-ის მოცულობას.
- `"xgboost>=3.0,<4"` აყენებს XGBoost 3.x ვერსიას.
- `"wandb>=0.19,<1"` აყენებს W&B SDK-ის თავსებად 0.x ვერსიას.

ამ cell-ს შენახული output არ აქვს. ფაქტობრივ გარემოში მოგვიანებით დაფიქსირდა `xgboost==3.3.0` და `wandb==0.28.0`.

## Cell 3 — imports, seed და ვერსიები

```python
from __future__ import annotations
```

Type hint-ებს, მაგალითად `Path | None`, შეფასების ნაცვლად annotation-ის სახით ინახავს და type declaration-ებთან მუშაობას ამარტივებს.

```python
import json
import math
import os
import platform
import random
```

- `json` — feature list-ისა და metric-ების JSON ფაილებში ჩაწერა;
- `math` — RMSE-სთვის square root;
- `os` — environment variable-იდან W&B key-ის წაკითხვა;
- `platform` — Python version;
- `random` — Python random seed.

```python
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import wandb
import xgboost as xgb
```

- `matplotlib` — დიაგნოსტიკური გრაფიკები;
- `numpy` — მასივები, weights და ციკლური features;
- `pandas` — CSV და tabular transformations;
- `sklearn` — version და evaluation metrics;
- `wandb` — experiment tracking;
- `xgboost` — მოდელი.

```python
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error
from wandb.integration.xgboost import WandbCallback
```

- `Path` — filesystem path-ების უსაფრთხო აგება;
- `mean_absolute_error` — ჩვეულებრივი MAE;
- `mean_squared_error` — RMSE-ს საფუძველი;
- `WandbCallback` — training iteration-ების W&B-ში ავტომატური logging.

```python
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
```

ადგენს reproducibility seed-ს Python-ისა და NumPy-სთვის. XGBoost-ს იგივე seed მოგვიანებით ცალკე გადაეცემა.

```python
pd.set_option("display.max_columns", 100)
```

DataFrame-ის ჩვენებისას მაქსიმუმ 100 სვეტს აჩენს, რათა features არ დაიმალოს.

```python
print({...})
```

ბეჭდავს გარემოს ვერსიებს. მიღებული შედეგი:

```text
Python 3.12.13
pandas 2.2.2
scikit-learn 1.6.1
xgboost 3.3.0
wandb 0.28.0
```

## Cell 4 — ექსპერიმენტის კონფიგურაცია

`CONFIG` არის ერთი dictionary, რომელშიც მოდელისა და pipeline-ის პარამეტრები ინახება.

```python
"seed": SEED
```

Reproducibility-ის seed, მნიშვნელობა `42`.

```python
"validation_weeks": 52
```

ბოლო 52 კვირა validation-ად გამოიყოფა.

```python
"holiday_weight": 5.0
```

Kaggle WMAE-ში holiday row-ის წონა არის 5; ჩვეულებრივი row-ის — 1.

```python
"objective": "reg:absoluteerror"
```

XGBoost ოპტიმიზაციას absolute error-ზე აკეთებს, რაც MAE/WMAE მიზანს შეესაბამება.

```python
"eval_metric": "mae"
```

ყოველი boosting iteration-ის შემდეგ ითვლება MAE. რადგან validation weights ცალკე გადაეცემა, validation-ის ეს MAE შეწონილი MAE-ა.

```python
"n_estimators": 3000
```

boosting trees-ის მაქსიმალური რაოდენობა.

```python
"learning_rate": 0.03
```

თითოეული ახალი tree-ის contribution მცირეა. დაბალი learning rate ჩვეულებრივ მეტ tree-ს მოითხოვს.

```python
"max_depth": 8
```

ერთი tree-ის მაქსიმალური სიღრმე. მეტი სიღრმე რთულ interactions-ს იჭერს, მაგრამ overfitting-ის რისკსაც ზრდის.

```python
"min_child_weight": 5
```

ზღუდავს ძალიან მცირე მხარდაჭერის მქონე leaf/split-ებს და გარკვეულ regularization-ს ქმნის.

```python
"subsample": 0.85
```

ყოველი tree training rows-ის შემთხვევით 85%-ს იყენებს.

```python
"colsample_bytree": 0.85
```

ყოველი tree features-ის შემთხვევით 85%-ს იყენებს.

```python
"reg_alpha": 0.0
"reg_lambda": 1.0
```

- `reg_alpha` — L1 regularization;
- `reg_lambda` — L2 regularization.

```python
"early_stopping_rounds": 100
```

თუ validation metric 100 iteration-ის განმავლობაში არ გაუმჯობესდება, training უნდა შეწყდეს.

```python
"tree_method": "hist"
```

იყენებს histogram-based tree builder-ს, რომელიც დიდ tabular dataset-ზე სწრაფია.

```python
"device": "cuda"
```

training GPU-ზე სრულდება.

```python
"run_final_refit": True
```

validation მოდელის შემდეგ სრული train dataset-ით საბოლოო მოდელიც იწვრთნება.

```python
"log_dataset_artifact": True
```

ოთხივე raw CSV W&B dataset artifact-ში იტვირთება.

```python
WANDB_ENTITY = "kende23-n-a"
WANDB_PROJECT = "Walmart-Recruiting---Store-Sales-Forecasting"
```

განსაზღვრავს W&B team/entity-სა და project-ს.

```python
wandb.login(key=os.environ.get("WANDB_API_KEY"), relogin=False)
```

- ეძებს `WANDB_API_KEY` environment variable-ს;
- თუ key ხელმისაწვდომია, იყენებს მას;
- `relogin=False` უკვე არსებულ login-ს არ ცვლის.

Output-ში იყო warning:

```text
Calling wandb.login() after wandb.init() has no effect.
```

ეს ნიშნავს, რომ runtime-ში W&B run უკვე ინიციალიზებული იყო, სავარაუდოდ cell-ების განმეორებით ან სხვა თანმიმდევრობით გაშვების გამო. სუფთა runtime-ში login უნდა შესრულდეს `wandb.init()`-მდე.

ბოლო `CONFIG` ხაზი dictionary-ს notebook output-ად აჩვენებს და ადასტურებს რეალურ პარამეტრებს.

## Cell 5 — მონაცემების სექციის სათაური

Markdown heading მხოლოდ შემდეგი cell-ების დანიშნულებას გამოყოფს: data loading და schema validation.

## Cell 6 — მონაცემების ჩატვირთვა და შემოწმება

```python
import pandas as pd
from pathlib import Path
```

ეს imports Cell 3-ში უკვე შესრულებულია, ამიტომ აქ განმეორებითია, თუმცა cell-ს დამოუკიდებლად გასაშვებად ხდის.

```python
DATA_DIR = Path("/content/drive/MyDrive/walmart_competition_data")
OUTPUT_DIR = Path("/content/artifacts/xgboost_baseline")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
```

- `DATA_DIR` მიუთითებს Google Drive-ზე შენახულ CSV-ებზე;
- `OUTPUT_DIR` მიუთითებს Colab-ის დროებით local disk-ზე;
- `parents=True` საჭირო parent directories-საც ქმნის;
- `exist_ok=True` უკვე არსებული directory-ის შემთხვევაში error-ს არ აგდებს.

```python
train_raw = pd.read_csv(..., parse_dates=["Date"])
test_raw = pd.read_csv(..., parse_dates=["Date"])
features = pd.read_csv(..., parse_dates=["Date"])
stores = pd.read_csv(...)
```

- კითხულობს ოთხ CSV-ს;
- პირველ სამ ცხრილში `Date` პირდაპირ datetime ტიპად გარდაიქმნება;
- `stores.csv`-ში date არ არის.

```python
required_train = {...}
required_test = {...}
required_features = {...}
required_stores = {...}
```

თითოეული ფაილისთვის ადგენს აუცილებელი სვეტების set-ს.

```python
assert required_*.issubset(...columns)
```

ამოწმებს, რომ ყველა საჭირო column ნამდვილად არსებობს. რომელიმე column-ის არყოფნისას execution შეწყდება.

```python
assert not train_raw.duplicated(["Store", "Dept", "Date"]).any()
assert not test_raw.duplicated(["Store", "Dept", "Date"]).any()
```

ამოწმებს, რომ ერთი `Store`/`Dept`/`Date` კომბინაცია ორჯერ არ მეორდება.

```python
display(pd.DataFrame({...}))
```

აგებს მოკლე data summary-ს. მიღებული შედეგი:

| dataset | rows | min date | max date |
|---|---:|---|---|
| train | 421,570 | 2010-02-05 | 2012-10-26 |
| test | 115,064 | 2012-11-02 | 2013-07-26 |
| features | 8,190 | 2010-02-05 | 2013-07-26 |
| stores | 45 | — | — |

## Cell 7 — Google Drive mount

```python
from google.colab import drive
drive.mount("/content/drive")
```

- import-ს უკეთებს Colab Drive integration-ს;
- Google Drive-ს `/content/drive` path-ზე აერთებს.

Output:

```text
Mounted at /content/drive
```

Fresh runtime-ში ეს cell Cell 6-მდე უნდა გაეშვას, წინააღმდეგ შემთხვევაში CSV path ჯერ არ იარსებებს.

## Cell 8 — feature engineering-ის აღწერა

Markdown განმარტავს სამ პრინციპს:

- გამოიყენება მხოლოდ train/test-ში ხელმისაწვდომი ინფორმაცია;
- numeric missing values XGBoost-ს პირდაპირ შეუძლია დაამუშაოს;
- `Type` რიცხვად გარდაიქმნება, target encoding კი არ გამოიყენება.

## Cell 9 — feature engineering

```python
TYPE_MAP = {"A": 0, "B": 1, "C": 2}
```

მაღაზიის ტიპებს integer code-ებად გარდაქმნის.

```python
def build_features(sales_frame, external, store_metadata) -> pd.DataFrame:
```

ქმნის reusable ფუნქციას, რომელიც ერთნაირ transformation-ს აკეთებს train-სა და test-ზე.

```python
frame = sales_frame.copy()
```

ორიგინალ DataFrame-ს არ ცვლის; ქმნის სამუშაო ასლს.

```python
external_no_holiday = external.drop(columns="IsHoliday")
```

`features.csv`-დან შლის `IsHoliday`-ს, რადგან იგივე column უკვე train/test-შია. ამით merge-ის შემდეგ `_x`/`_y` duplicate columns არ იქმნება.

```python
frame.merge(... on=["Store", "Date"], how="left", validate="many_to_one")
```

- თითოეულ sales row-ს `Store`/`Date` მიხედვით უერთებს external features-ს;
- `left` ყველა sales row-ს ინარჩუნებს;
- `many_to_one` ამოწმებს, რომ external table-ში თითო `Store`/`Date` key მხოლოდ ერთხელაა.

```python
frame.merge(store_metadata, on="Store", how="left", validate="many_to_one")
```

თითოეულ row-ს უერთებს store `Type`-სა და `Size`-ს.

```python
iso = frame["Date"].dt.isocalendar()
```

Date-დან ISO calendar fields-ს იღებს; აქ საჭიროა კვირის ნომერი.

```python
frame["Year"] = ...astype("int16")
frame["Month"] = ...astype("int8")
frame["WeekOfYear"] = ...astype("int8")
frame["Quarter"] = ...astype("int8")
```

ქმნის year/month/week/quarter features-ს და memory-ს მცირე integer types-ით ამცირებს.

```python
frame["DaysFromStart"] = (
    frame["Date"] - pd.Timestamp("2010-02-05")
).dt.days.astype("int16")
```

თარიღს გარდაქმნის dataset-ის დასაწყისიდან გასული დღეების რაოდენობად. ეს მოდელს გრძელვადიანი trend-ის დანახვაში ეხმარება.

```python
frame["WeekSin"] = np.sin(2 * np.pi * WeekOfYear / 52.0)
frame["WeekCos"] = np.cos(2 * np.pi * WeekOfYear / 52.0)
```

კვირის ნომერს ციკლურად encode-ავს. ასე 52-ე და 1-ლი კვირა ერთმანეთთან ახლოს წარმოდგება, განსხვავებით უბრალო integer encoding-ისგან.

```python
frame["TotalMarkDown"] = frame[markdown_columns].sum(axis=1, min_count=1)
```

- `axis=1` ერთი row-ის ხუთ markdown მნიშვნელობას აჯამებს;
- `min_count=1` ყველა markdown-ის missing მნიშვნელობისას შედეგსაც missing-ად ტოვებს, ნაცვლად მცდარი ნულისა.

```python
frame["Type"] = frame["Type"].map(TYPE_MAP).astype("int8")
frame["IsHoliday"] = frame["IsHoliday"].astype("int8")
```

ორ categorical/boolean column-ს numeric ტიპად გარდაქმნის.

```python
return frame.sort_values(["Date", "Store", "Dept"]).reset_index(drop=True)
```

- rows-ს ქრონოლოგიურად და key-ების მიხედვით ალაგებს;
- ძველ index-ს შლის და ახალ თანმიმდევრულ index-ს ქმნის.

```python
train = build_features(train_raw, features, stores)
test = build_features(test_raw, features, stores)
```

ზუსტად ერთ transformation-ს იყენებს ორივე dataset-ზე.

```python
NON_FEATURE_COLUMNS = {"Date", "Weekly_Sales"}
FEATURE_COLUMNS = [...]
```

- `Date` პირდაპირ model matrix-ში არ შედის, რადგან მისგან calendar features უკვე შეიქმნა;
- `Weekly_Sales` target-ია და feature-ში მისი დატოვება პირდაპირ leakage იქნებოდა;
- ყველა დანარჩენი column feature ხდება.

Output:

```text
ნიშან-თვისებების რაოდენობა: 22
```

22 feature არის:

```text
Store, Dept, IsHoliday, Temperature, Fuel_Price,
MarkDown1–MarkDown5, CPI, Unemployment, Type, Size,
Year, Month, WeekOfYear, Quarter, DaysFromStart,
WeekSin, WeekCos, TotalMarkDown
```

## Cell 10 — split-ის აღწერა

Markdown ამბობს, რომ ბოლო 52 კვირა validation-ია. ამ split-ში validation იწყება 2011-11-04-ზე, ამიტომ მოიცავს 2011 წლის Thanksgiving/Christmas პერიოდს.

## Cell 11 — WMAE და ქრონოლოგიური split

```python
def weighted_mae(y_true, y_pred, is_holiday, holiday_weight=5.0):
```

ქმნის Kaggle-ის metric function-ს.

```python
weights = np.where(..., holiday_weight, 1.0)
```

holiday row-ს აძლევს 5-ს, სხვა row-ს 1-ს.

```python
np.abs(np.asarray(y_true) - np.asarray(y_pred))
```

თითოეული row-ის absolute prediction error-ს ითვლის.

```python
np.average(..., weights=weights)
```

ასრულებს ზუსტ ფორმულას:

```text
WMAE = Σ(weight × absolute error) / Σ(weight)
```

```python
validation_start = train["Date"].max() - pd.Timedelta(weeks=51)
```

მაქსიმალური თარიღიდან 51 კვირას აკლებს. საზღვრის ჩათვლით ეს 52 weekly dates-ს იძლევა.

```python
train_mask = train["Date"] < validation_start
valid_mask = ~train_mask
```

- საზღვრამდე rows training-ში მიდის;
- დანარჩენი validation-ში.

```python
X_train, y_train, X_valid, y_valid
```

DataFrame-ს feature matrices-ად და target vectors-ად ჰყოფს.

```python
w_train = np.where(...)
w_valid = np.where(...)
```

training და validation row weights-ს ქმნის.

```python
split_summary = {...}
```

W&B config/artifact metadata-სთვის split-ის ზომებსა და თარიღებს ინახავს.

მიღებული შედეგი:

| ნაწილი | rows | დასაწყისი | დასასრული |
|---|---:|---|---|
| train | 267,184 | 2010-02-05 | 2011-10-28 |
| validation | 154,386 | 2011-11-04 | 2012-10-26 |

Feature count: **22**.

## Cell 12 — W&B სექციის სათაური

Markdown განმარტავს, რომ raw CSV-ები versioned dataset artifact-ად ჩაიწერება.

## Cell 13 — W&B run და dataset artifact

```python
run = wandb.init(...)
```

იწყებს ახალ W&B experiment run-ს.

- `entity` — team/account;
- `project` — W&B project;
- `job_type="train"` — run-ის დანიშნულება;
- `name="xgboost-static-baseline"` — dashboard-ში ხილული სახელი;
- `tags` — filtering-ისთვის;
- `config={**CONFIG, **split_summary, "features": FEATURE_COLUMNS}` — აერთიანებს hyperparameters-ს, split metadata-ს და feature list-ს;
- `save_code=True` — W&B-ს code snapshot-ის შენახვას სთხოვს.

```python
if CONFIG["log_dataset_artifact"]:
```

dataset upload მხოლოდ flag-ის `True` მნიშვნელობისას სრულდება.

```python
dataset_artifact = wandb.Artifact(...)
```

ქმნის `walmart-recruiting-raw-data` სახელის `dataset` artifact-ს.

```python
for filename in [...]:
    dataset_artifact.add_file(...)
```

artifact-ში ამატებს `train.csv`, `test.csv`, `features.csv` და `stores.csv` ფაილებს.

```python
run.log_artifact(dataset_artifact, aliases=["latest"])
```

artifact-ს run-ის output-ად ტვირთავს და მიმდინარე ვერსიას `latest` alias-ს ანიჭებს.

ბოლო `run` ხაზი notebook-ში აქტიური W&B Run object-ის representation-ს აჩვენებს.

შექმნილი run ID: `pc46skfo`.

## Cell 14 — sanity benchmark-ის აღწერა

Median benchmark საჭიროა იმის სანახავად, რეალურად სჯობს თუ არა XGBoost ძალიან მარტივ ისტორიულ წესს.

## Cell 15 — median benchmark

```python
history = train.loc[train_mask, ["Store", "Dept", "Weekly_Sales"]]
valid_keys = train.loc[valid_mask, ["Store", "Dept"]]
```

- `history` მხოლოდ training პერიოდის target-ებს შეიცავს;
- `valid_keys` validation-ის Store/Dept key-ებს.

```python
pair_median = history.groupby(["Store", "Dept"])["Weekly_Sales"].median()
```

თითო Store/Dept სერიის ისტორიულ median sales-ს ითვლის.

```python
dept_median = history.groupby("Dept")["Weekly_Sales"].median()
global_median = float(history["Weekly_Sales"].median())
```

ქმნის fallback-ებს:

1. department median;
2. მთელი training set-ის median.

```python
pair_median.get((store, dept), np.nan)
```

თითო validation row-ისთვის ეძებს შესაბამის Store/Dept median-ს. უცნობი pair-ისას აბრუნებს `NaN`-ს.

```python
fillna(valid_keys["Dept"].map(dept_median)).fillna(global_median)
```

უცნობ pair-ს ჯერ department median-ით, შემდეგ global median-ით ავსებს.

```python
naive_wmae = weighted_mae(...)
```

benchmark-ს Kaggle-ის ზუსტ metric-ზე აფასებს.

```python
if "run" in globals() and run is not None:
```

W&B logging-ს მხოლოდ აქტიური `run` object-ის არსებობისას ასრულებს. ეს cell-ს უფრო გამძლეს ხდის out-of-order execution-ის მიმართ.

შედეგი:

```text
Median benchmark validation WMAE = 3305.52
```

## Cell 16 — training-ის აღწერა

Markdown აღნიშნავს, რომ XGBoost validation MAE ამ შემთხვევაში WMAE-ა, რადგან `sample_weight_eval_set` holiday weights-ს შეიცავს.

## Cell 17 — XGBoost training

```python
model_params = {
    key: CONFIG[key]
    for key in [...]
}
```

`CONFIG`-დან მხოლოდ XGBoost-სთვის საჭირო პარამეტრებს არჩევს.

```python
callbacks = []
if "run" in globals() and wandb.run is not None:
```

callback list თავიდან ცარიელია. W&B callback მხოლოდ აქტიური run-ისას ემატება.

```python
WandbCallback(
    log_model=False,
    log_feature_importance=True,
    importance_type="gain",
    define_metric=True,
)
```

- `log_model=False` — callback model artifact-ს ავტომატურად არ ქმნის, რადგან მოდელი მოგვიანებით ხელით ინახება;
- `log_feature_importance=True` — feature importance-ს W&B-ში აგზავნის;
- `importance_type="gain"` — importance split-ებით მიღებული loss reduction-ით იზომება;
- `define_metric=True` — W&B-ს best-step summary metric-ების განსაზღვრაში ეხმარება.

```python
model = xgb.XGBRegressor(
    **model_params,
    random_state=CONFIG["seed"],
    n_jobs=-1,
    callbacks=callbacks,
)
```

- ქმნის regression model-ს;
- `**model_params` dictionary-ს keyword arguments-ად შლის;
- `random_state` reproducibility-ს აკონტროლებს;
- `n_jobs=-1` CPU-side სამუშაოებისთვის ყველა core-ს იყენებს;
- callbacks iteration logging-ს ასრულებს.

```python
model.fit(
    X_train,
    y_train,
    sample_weight=w_train,
    eval_set=[(X_train, y_train), (X_valid, y_valid)],
    sample_weight_eval_set=[w_train, w_valid],
    verbose=50,
)
```

- training features და target გადაეცემა;
- `sample_weight=w_train` holiday errors-ს training objective-ში ხუთმაგ წონას აძლევს;
- `eval_set` training და validation curve-ებს ითვლის;
- `sample_weight_eval_set` ორივე curve-ს შესაბამის weights-ს აძლევს;
- `verbose=50` log-ს ყოველ 50 iteration-ზე ბეჭდავს.

Log-ში:

- iteration 0 validation WMAE იყო `13924.37`;
- iteration 1000-ზე — `3316.91`;
- iteration 2000-ზე — `3055.15`;
- iteration 2999-ზე — `2902.29`.

საბოლოო output:

```text
საუკეთესო იტერაცია: 2999
საუკეთესო validation WMAE: 2902.29
```

რადგან best iteration ბოლო iteration-ია, early stopping არ მომხდარა.

## Cell 18 — evaluation სექციის სათაური

Markdown გამოყოფს prediction, metrics და diagnostics ნაწილს.

## Cell 19 — validation metrics

```python
valid_pred = model.predict(X_valid)
```

validation rows-ის პროგნოზებს ქმნის.

```python
valid_holiday = ...to_numpy(dtype=bool)
```

holiday mask-ს boolean NumPy array-ად გარდაქმნის.

`metrics` dictionary:

- `validation/wmae` — Kaggle metric;
- `validation/mae` — ყველა row-ის ჩვეულებრივი, შეუწონავი MAE;
- `validation/rmse` — დიდ errors-ს კვადრატის გამო უფრო მკაცრად სჯის;
- `validation/holiday_mae` — მხოლოდ holiday rows;
- `validation/non_holiday_mae` — მხოლოდ non-holiday rows;
- `validation/improvement_over_median_pct` — XGBoost-ის პროცენტული გაუმჯობესება median benchmark-ზე;
- `model/best_iteration` — საუკეთესო tree iteration;
- `model/best_score` — XGBoost-ის მიერ დაფიქსირებული საუკეთესო weighted validation MAE.

```python
run.log(metrics)
run.summary.update(metrics)
```

metric-ებს W&B history-სა და run summary-ში წერს.

```python
validation_results = train.loc[..., [...]].copy()
```

დიაგნოსტიკისთვის ინახავს Store, Dept, Date, holiday flag-სა და actual sales-ს.

```python
validation_results["Prediction"] = valid_pred
validation_results["AbsoluteError"] = ...
```

ამატებს prediction-სა და თითო row-ის absolute error-ს.

```python
validation_results["Date"] = ...strftime("%Y-%m-%d")
```

Date-ს ცხრილებისა და serialization-ისთვის სტრიქონად გარდაქმნის.

მიღებული metric-ები:

| metric | მნიშვნელობა |
|---|---:|
| WMAE | **2902.2892** |
| MAE | 2727.6525 |
| RMSE | 6654.0033 |
| holiday MAE | 3464.4077 |
| non-holiday MAE | 2665.6108 |
| median-ზე გაუმჯობესება | **12.1986%** |
| best iteration | 2999 |

CUDA warning-ის მიზეზი CPU `pandas` input და GPU model-ის device mismatch-ია. XGBoost prediction-ს მაინც ასრულებს.

## Cell 20 — feature importance და გრაფიკები

```python
pd.DataFrame({
    "feature": FEATURE_COLUMNS,
    "importance": model.feature_importances_,
})
```

ყოველ feature-ს მოდელის importance მნიშვნელობას უკავშირებს.

```python
.sort_values("importance", ascending=False).reset_index(drop=True)
```

features-ს მნიშვნელოვნების კლებადობით ალაგებს და index-ს ასუფთავებს.

ტოპ features:

| feature | importance |
|---|---:|
| Dept | 0.1653 |
| Size | 0.1625 |
| Type | 0.1513 |
| Store | 0.0788 |
| Unemployment | 0.0528 |
| WeekCos | 0.0528 |
| CPI | 0.0526 |
| WeekOfYear | 0.0442 |

`MarkDown1`, `MarkDown2`, `MarkDown3` და `MarkDown5` ნაჩვენებ top-20-ში `0.0` importance-ით დასრულდა. ეს ნიშნავს, რომ ამ fitted მოდელში შესაბამის features-ს gain არ მიუღია; ეს არ ამტკიცებს, რომ markdown ზოგადად უსარგებლოა.

```python
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
```

ქმნის ერთ figure-ს ორი subplot-ით.

```python
top = importance.head(20).sort_values("importance")
axes[0].barh(...)
```

პირველ subplot-ზე top-20 feature importance-ის horizontal bar chart-ს ხატავს. ხელახალი ascending sort ყველაზე მნიშვნელოვან feature-ს ზედა მხარეს აჩენს.

```python
sample = validation_results.sample(
    min(15000, len(validation_results)),
    random_state=SEED,
)
```

scatter plot-ის სიმძიმის შესამცირებლად მაქსიმუმ 15,000 validation row-ს reproducibly არჩევს.

```python
axes[1].scatter(actual, prediction, alpha=0.15, s=8)
```

რეალურ და პროგნოზირებულ sales-ს ადარებს. გამჭვირვალობა overlapping points-ს უკეთ აჩვენებს.

```python
limits = [...]
axes[1].plot(limits, limits, "r--", linewidth=1)
```

ამატებს იდეალური პროგნოზის `y=x` წითელ წყვეტილ ხაზს.

```python
plt.tight_layout()
```

subplot-ებს ისე ალაგებს, რომ labels ერთმანეთს არ გადაეფაროს.

```python
run.log({"validation/diagnostics": wandb.Image(fig)})
plt.show()
```

figure-ს W&B-ში ტვირთავს და notebook-შიც აჩვენებს.

## Cell 21 — model saving-ის აღწერა

Markdown განმარტავს, რომ:

- model ინახება XGBoost JSON ფორმატში;
- feature order ცალკე ინახება;
- metrics და importance ცალკე files-ად ინახება;
- ყველაფერი ერთ W&B model artifact-ში ერთიანდება.

Feature order კრიტიკულია: inference-ისას columns იგივე თანმიმდევრობით უნდა გადაეცეს.

## Cell 22 — validation model artifact

```python
if "metrics" not in globals():
```

თუ Cell 19-ის `metrics` ცვლადი runtime-ში არ არსებობს, მინიმალურ metric dictionary-ს თავიდან ქმნის. ეს out-of-order execution-ისგან ნაწილობრივი დაცვაა.

```python
model_path = ...
feature_path = ...
importance_path = ...
metrics_path = ...
```

განსაზღვრავს ოთხ output file-ს:

- `xgboost_baseline_validation.json`;
- `feature_columns.json`;
- `feature_importance.csv`;
- `validation_metrics.json`.

```python
model.save_model(model_path)
```

trained validation model-ს XGBoost JSON ფორმატში წერს.

```python
feature_path.write_text(json.dumps(FEATURE_COLUMNS, indent=2))
```

feature names/order-ს human-readable JSON-ში ინახავს.

```python
importance.to_csv(importance_path, index=False)
```

feature importance table-ს CSV-ში წერს.

```python
metrics_path.write_text(json.dumps(metrics, indent=2))
```

validation metric-ებს JSON-ში წერს.

```python
model_artifact = wandb.Artifact(...)
```

ქმნის `xgboost-static-baseline` ტიპის `model` artifact-ს და metadata-ში metric-ებსა და split summary-ს ათავსებს.

```python
model_artifact.add_dir(str(OUTPUT_DIR))
run.log_artifact(... aliases=["validation", "latest"])
```

მთელ output directory-ს artifact-ში ამატებს და ორ alias-ს აძლევს.

Output ადასტურებს, რომ ფაილები ჩაიწერა:

```text
/content/artifacts/xgboost_baseline
```

## Cell 23 — final refit-ის აღწერა

Markdown აღწერს optional full-data training-ს. მიმდინარე notebook-ში flag უკვე `True` არის, ამიტომ ეს ნაბიჯი optional აღარ იყო და რეალურად შესრულდა.

ტექსტში მითითებული Kaggle ფორმატი შინაარსობრივად ნიშნავს ორ column-ს:

```text
Id,Weekly_Sales
```

სადაც `Id`-ის თითოეული მნიშვნელობა არის `Store_Dept_Date`.

## Cell 24 — final model და submission

```python
if CONFIG["run_final_refit"]:
```

მთელი block მხოლოდ `True` flag-ისას სრულდება.

```python
final_rounds = int(model.best_iteration) + 1
```

XGBoost iteration zero-based არის. Best iteration `2999` ნიშნავს 3000 tree-ს, ამიტომ ემატება 1.

```python
final_params = {
    ...
    if key not in {"n_estimators", "early_stopping_rounds"}
}
```

validation model-ის params-ს იღებს, მაგრამ:

- `n_estimators` შემდეგ `final_rounds`-ით განისაზღვრება;
- early stopping აღარ არის საჭირო, რადგან final training-ს validation set არ აქვს.

```python
final_model = xgb.XGBRegressor(...)
```

ქმნის საბოლოო model-ს 3000 boosting round-ით.

```python
all_weights = np.where(train["IsHoliday"], 5.0, 1.0)
```

სრული labeled dataset-ის holiday weights-ს ქმნის.

```python
final_model.fit(
    train[FEATURE_COLUMNS],
    train["Weekly_Sales"],
    sample_weight=all_weights,
    verbose=False,
)
```

საბოლოო მოდელს ყველა 421,570 training row-ზე წვრთნის. `verbose=False` training log-ს მალავს.

```python
test_pred = final_model.predict(test[FEATURE_COLUMNS])
```

115,064 Kaggle test row-ის sales prediction-ს ქმნის.

```python
"Id": Store + "_" + Dept + "_" + Date
```

Kaggle-ის row identifier-ს ქმნის, მაგალითად:

```text
1_1_2012-11-02
```

```python
"Weekly_Sales": test_pred
```

submission-ის მეორე column-ში model prediction-ს წერს.

```python
final_model.save_model(final_model_path)
submission.to_csv(submission_path, index=False)
```

ინახავს:

- `xgboost_baseline_final.json`;
- `submission_xgboost_baseline.csv`.

```python
final_artifact = wandb.Artifact(...)
```

ქმნის საბოლოო model artifact-ს metadata-ით:

- `n_estimators=3000`;
- `training_rows=421570`.

```python
add_file(final_model_path)
add_file(feature_path)
add_file(submission_path)
```

artifact-ში აერთიანებს model-ს, feature order-სა და Kaggle submission-ს.

```python
aliases=["production-candidate", "latest"]
```

artifact-ს production candidate-ისა და latest version-ის alias-ებს აძლევს.

Output:

```text
საბოლოო ფაილი შენახულია:
/content/artifacts/xgboost_baseline/submission_xgboost_baseline.csv
```

`else` branch მხოლოდ flag-ის `False` მნიშვნელობისას დაბეჭდავდა, რომ refit გამოტოვებულია.

## Cell 25 — W&B დასრულების აღწერა

Markdown გვახსენებს, რომ run აუცილებლად უნდა დასრულდეს, რათა buffered metrics და artifacts server-ზე სრულად აიტვირთოს.

## Cell 26 — W&B run-ის დასრულება

```python
run.finish()
```

ასრულებს run-ს, დარჩენილ მონაცემებს სინქრონიზაციას უკეთებს და local W&B process-ს ხურავს.

საბოლოო W&B output:

- run name: `xgboost-static-baseline`;
- run ID: `pc46skfo`;
- synced files: 5;
- media files: 2;
- artifact files: 40;
- best iteration: 2999;
- best score: 2902.28931.

## შედეგის ინტერპრეტაცია

XGBoost median benchmark-ს დაახლოებით 12.2%-ით აუმჯობესებს, ამიტომ baseline რეალურ დამატებით signal-ს სწავლობს. ყველაზე ძლიერი features არის department identity, store size/type და store identity. Calendar და macroeconomic features დამატებით signal-ს იძლევა.

Holiday MAE (`3464.41`) non-holiday MAE-ზე (`2665.61`) მაღალია, ანუ holiday demand კვლავ უფრო რთული საპროგნოზოა. WMAE ჩვეულებრივ MAE-ზე მაღალია, რადგან რთულ holiday errors-ს ხუთმაგი წონა აქვს.

Train WMAE (`1728.14`) validation WMAE-ზე (`2902.29`) მნიშვნელოვნად დაბალია. ეს მოსალოდნელ generalization gap-ს აჩვენებს და hyperparameter tuning-ის საჭიროებას მიუთითებს. ამასთან, validation metric ბოლო iteration-მდე უმჯობესდებოდა, ამიტომ შემდეგ ექსპერიმენტში ღირს:

- `n_estimators`-ის გაზრდა და early stopping-ის რეალურად ამუშავება;
- shallower trees ან უფრო ძლიერი regularization-ის შედარება;
- holiday proximity features;
- leakage-safe lag-52 ან multi-horizon forecasting;
- რამდენიმე chronological fold-ით backtesting.

## შექმნილი ფაილები

Colab-ში შეიქმნა:

```text
/content/artifacts/xgboost_baseline/
├── xgboost_baseline_validation.json
├── xgboost_baseline_final.json
├── feature_columns.json
├── feature_importance.csv
├── validation_metrics.json
└── submission_xgboost_baseline.csv
```

`/content` დროებითი storage-ია და runtime-ის წაშლისას ქრება. მიმდინარე გაშვებაში model/submission W&B artifact-ადაც აიტვირთა, ამიტომ მათი აღდგენა W&B-დან შეიძლება.

## Baseline-იდან საბოლოო მოდელამდე — აუდიტირებული შეჯამება

Baseline-მა დაადასტურა global XGBoost flow და მიიღო `2902.2892` validation WMAE. შემდეგ engineered notebook-მა დაამატა leakage-safe `SalesLag52`, target aggregates, holiday proximity, markdown interactions, tuning და full-data refit. საბოლოო `2806` Kaggle score baseline notebook-ის score არ არის; ის განვითარებული Registry pipeline-ის შედეგია. ამიტომ baseline გამოიყენება reference-ად, ხოლო engineered XGBoost — champion-ად.
