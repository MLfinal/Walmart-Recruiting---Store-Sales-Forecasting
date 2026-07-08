# N-BEATS მოდელის ექსპერიმენტი

ეს ფაილი ხსნის, რა ნაბიჯებია გაკეთებული `model_experiment_N-BEATS.ipynb` notebook-ში და რატომ არის თითოეული ნაბიჯი საჭირო Walmart Store Sales Forecasting ამოცანისთვის.

## მიმდინარე baseline run-ის ანალიზი

`baseline_N-BEATS.ipynb` გაეშვა როგორც საწყისი baseline მოდელი დამატებითი feature engineering-ის გარეშე. ამ run-ში მოდელი სწავლობდა 30 epoch-ს, მაგრამ საუკეთესო validation შედეგი მიიღო ძალიან ადრე:

```text
Best epoch: 2
Best validation WMAE: 2157.9829
Best validation MAE: 2128.4329
```

Epoch-ების მიხედვით ჩანს, რომ training loss მუდმივად მცირდება:

```text
epoch 1  train L1 = 0.52564
epoch 30 train L1 = 0.41464
```

მაგრამ validation metric გაუმჯობესდა მხოლოდ მე-2 epoch-მდე:

```text
epoch 1 validation WMAE = 2222.4863
epoch 2 validation WMAE = 2157.9829
epoch 3 validation WMAE = 2191.0672
epoch 30 validation WMAE = 2244.5165
```

ეს ნიშნავს, რომ baseline N-BEATS მოდელი training data-ზე სწავლას აგრძელებს, მაგრამ validation period-ზე უკეთესი აღარ ხდება. ასეთი ქცევა მიუთითებს, რომ მოდელი სავარაუდოდ:

- ძალიან ადრე იწყებს overfitting-ს;
- ან train period-ის pattern-ებს კარგად სწავლობს, მაგრამ ბოლო 32 კვირაზე კარგად ვერ generalize-დება;
- ან learning rate/model capacity ისეთი კონფიგურაციით არის არჩეული, რომ საუკეთესო generalization ძალიან ადრე მიიღება.

ამიტომ ამ baseline-ის შემდეგ პირველ რიგში feature engineering-ის დამატება არ არის ყველაზე სწორი ნაბიჯი. ჯერ უკეთესია training setup-ის შემოწმება:

- lower learning rate, მაგალითად `3e-4`;
- early stopping, მაგალითად patience `5`;
- best epoch-ის შენახვა validation WMAE-ის მიხედვით;
- შემდეგ უკვე ერთი feature/preprocessing ცვლილების დამატება და შედარება baseline-თან.

Baseline reference:

```text
N-BEATS baseline best WMAE = 2157.9829 at epoch 2
```

შემდეგი ექსპერიმენტები უნდა შედარდეს ამ რიცხვთან. თუ ახალი ცვლილება მიიღებს უფრო დაბალ validation WMAE-ს, ცვლილება გაუმჯობესებად ჩაითვლება. თუ WMAE გაიზრდება, ცვლილება არ აუმჯობესებს მოდელს.

## 1. გარემოს მომზადება

Notebook-ის დასაწყისში ყენდება საჭირო ბიბლიოთეკები:

- `torch` - N-BEATS ნეირონული ქსელის ასაწყობად და სასწავლებლად.
- `wandb` - ექსპერიმენტების, მეტრიკების, არტიფაქტების და Model Registry-ისთვის.
- `pandas` და `numpy` - მონაცემების დამუშავებისთვის.
- `matplotlib` - validation დიაგნოსტიკური გრაფიკებისთვის.

შემდეგ ხდება Google Drive-ის mount, რადგან Colab-ში მონაცემები ხშირად ინახება Drive-ზე. თუ notebook ლოკალურად გაეშვება, Drive mount უბრალოდ გამოტოვდება.

## 2. კონფიგურაცია

Notebook-ში არის საერთო `CONFIG`, სადაც ინახება მოდელის და preprocessing-ის მთავარი პარამეტრები:

- `validation_weeks = 32` - validation ნაწილად გამოიყენება ბოლო 32 კვირა.
- `context_length = 52` - მოდელი შესავალად იღებს წინა 52 კვირის გაყიდვებს.
- `forecast_horizon = 32` - მოდელი ერთდროულად პროგნოზირებს შემდეგ 32 კვირას.
- `max_epochs = 30` - თითოეული hyperparameter combination სწავლობს 30 epoch-ს.
- `holiday_weight = 5.0` - holiday კვირებზე შეცდომა უფრო მნიშვნელოვანია, როგორც competition metric-ში.

ასევე მითითებულია W&B project/entity:

- `WANDB_ENTITY = "kende23-n-a"`
- `WANDB_PROJECT = "Walmart-Recruiting---Store-Sales-Forecasting"`

ეს საჭიროა, რომ preprocessing, training, grid search და best model სწორ W&B project-ში დალოგდეს.

## 3. მონაცემების ჩატვირთვა

Notebook კითხულობს ოთხ ფაილს:

- `train.csv`
- `test.csv`
- `features.csv`
- `stores.csv`

N-BEATS ამ notebook-ში გამოიყენება როგორც univariate forecasting model. ანუ მისი მთავარი input არის თითოეული `(Store, Dept)` სერიის `Weekly_Sales` ისტორია. მიუხედავად ამისა, იტვირთება ყველა ფაილი, რადგან ისინი საჭიროა experiment metadata-სთვის და მომავალ inference პროცესში შეიძლება დაგვჭირდეს.

ჩატვირთვის შემდეგ მოწმდება, რომ `train.csv` შეიცავს აუცილებელ სვეტებს:

- `Store`
- `Dept`
- `Date`
- `Weekly_Sales`
- `IsHoliday`

ეს validation იცავს notebook-ს ჩუმი შეცდომებისგან, თუ data path არასწორია ან dataset დაზიანებულია.

## 4. დროითი დალაგება

მონაცემები ლაგდება ასე:

```python
Store, Dept, Date
```

Time series ამოცანაში დალაგება აუცილებელია, რადგან მოდელმა უნდა ისწავლოს წარსულიდან მომავლის პროგნოზირება. თუ თარიღები არასწორი თანმიმდევრობით იქნება, sliding windows არასწორ ისტორიულ პერიოდებს გამოიყენებს.

## 5. Store-Dept სერიებად გადაკეთება

თავდაპირველად dataset არის tabular ფორმატში, სადაც თითო row არის ერთი store/dept/date ჩანაწერი. N-BEATS-ს სჭირდება რეგულარული time series tensor.

ამიტომ notebook ქმნის pivot table-ს:

- index: `(Store, Dept)`
- columns: `Date`
- values: `Weekly_Sales`

ამის შემდეგ თითოეული row არის ერთი კონკრეტული `(Store, Dept)` გაყიდვების ისტორია კვირების მიხედვით.

## 6. სრული კვირეული დროითი grid

Notebook ქმნის სრულ weekly date range-ს:

```python
pd.date_range(min_date, max_date, freq="W-FRI")
```

Walmart dataset-ში კვირები პარასკევით მთავრდება, ამიტომ გამოიყენება `W-FRI`. ეს უზრუნველყოფს, რომ ყველა სერიას ერთი და იგივე date columns ჰქონდეს.

ეს საჭიროა იმისთვის, რომ neural network input tensor-ს ჰქონდეს სტაბილური ზომა.

## 7. მოკლე სერიების მოცილება

სერია უნდა შეიცავდეს მინიმუმ:

```python
context_length + forecast_horizon
```

ანუ:

```python
52 + 32 = 84 კვირა
```

თუ კონკრეტულ `(Store, Dept)` სერიას ამაზე ნაკლები ისტორია აქვს, მას ვერ გამოვიყენებთ, რადგან მოდელს სჭირდება 52 კვირის input და 32 კვირის target.

ამიტომ ძალიან მოკლე სერიები იფილტრება.

## 8. Missing weeks შევსება

სერიების გაფილტვრის შემდეგ missing values ივსება:

- forward fill
- backward fill
- ბოლოს დარჩენილი missing value-ები `0.0`

ეს კეთდება მხოლოდ მას შემდეგ, რაც ძალიან მოკლე სერიები უკვე ამოღებულია. მიზანი არის რეგულარული matrix-ის მიღება, სადაც ყველა `(Store, Dept)` სერიას ერთნაირი რაოდენობის კვირა აქვს.

Tree-based მოდელებში missing value ზოგჯერ შეიძლება პირდაპირ დავტოვოთ, მაგრამ neural network-ს tensor input-ში `NaN` ვერ ექნება. ამიტომ აქ შევსება აუცილებელია.

## 9. Validation split

Validation არ კეთდება random split-ით. გამოიყენება time-based split:

- train period - ყველა კვირა ბოლო 32 კვირამდე.
- validation period - ბოლო 32 კვირა.

ეს მნიშვნელოვანია, რადგან forecasting ამოცანაში მოდელმა არ უნდა ნახოს მომავალი პერიოდის ინფორმაცია. Random split გამოიწვევდა time leakage-ს და validation score ხელოვნურად უკეთესი გამოჩნდებოდა.

## 10. Target transform

`Weekly_Sales` მნიშვნელობები გადადის log scale-ზე:

```python
np.log1p(np.clip(sales, a_min=0.0, a_max=None))
```

ამის მიზეზებია:

- გაყიდვებს აქვს დიდი scale განსხვავება store/dept სერიებს შორის.
- neural network სტაბილურად სწავლობს, როცა target distribution ნაკლებად skewed არის.
- `log1p` ამცირებს ძალიან დიდი გაყიდვების გავლენას loss-ზე.

Negative sales იშვიათად გვხვდება refund/return ჩანაწერების გამო. Log transform უარყოფით რიცხვებზე პრობლემურია, ამიტომ მხოლოდ transform-ისთვის ხდება clipping `0`-ზე. Validation metric მაინც ითვლება original sales scale-ზე.

## 11. Per-series normalization

თითოეული `(Store, Dept)` სერია ცალკე ნორმალიზდება:

```python
(value - series_mean) / series_std
```

mean და standard deviation ითვლება მხოლოდ training period-ზე, validation period-ის გარეშე.

ეს მნიშვნელოვანია ორი მიზეზით:

- არ ხდება validation information leakage.
- სხვადასხვა store/dept სერია შეიძლება სრულიად განსხვავებულ sales scale-ზე იყოს.

მაგალითად, დიდი store-ის პოპულარული department და პატარა store-ის იშვიათი department ერთნაირ scale-ზე არ იყიდება. normalization ეხმარება N-BEATS-ს ზოგადი pattern-ების სწავლაში.

## 12. Sliding windows

Training samples იქმნება sliding window ლოგიკით:

- input: წინა 52 კვირა
- target: შემდეგი 32 კვირა

მაგალითად:

```text
weeks 1-52   -> predict weeks 53-84
weeks 2-53   -> predict weeks 54-85
weeks 3-54   -> predict weeks 55-86
...
```

ეს ზრდის training samples-ის რაოდენობას და მოდელს ასწავლის სხვადასხვა historical context-იდან 32 კვირიანი forecast-ის გაკეთებას.

Validation-ში კი თითოეული სერიისთვის input არის validation-მდე ბოლო 52 კვირა, target კი validation-ის ბოლო 32 კვირა.

## 13. Preprocessing-ის W&B logging

Preprocessing ცალკე W&B run-ში ლოგდება:

- გამოყენებული სერიების რაოდენობა.
- training windows რაოდენობა.
- train/validation date ranges.
- target transform.
- normalization სტრატეგია.
- preprocessing steps table.

ეს საჭიროა, რომ W&B-ში ცალკე ჩანდეს არა მხოლოდ training, არამედ feature engineering/preprocessing პროცესიც.

## 14. Baseline model

N-BEATS-მდე ითვლება seasonal naive baseline.

Baseline პროგნოზად იყენებს validation-მდე ბოლო 32 კვირას და ადარებს validation-ის ნამდვილ 32 კვირას.

ეს გვაძლევს reference score-ს:

- თუ N-BEATS baseline-ზე უკეთესია, მოდელი რეალურად სწავლობს სასარგებლო pattern-ებს.
- თუ baseline უკეთესია, training ან preprocessing უნდა გადაიხედოს.

## 15. N-BEATS არქიტექტურა

Notebook-ში იმპლემენტირებულია simplified generic N-BEATS:

- `NBeatsBlock`
- `NBeats`

თითო block ქმნის:

- `backcast` - input history-ის ახსნილი ნაწილი.
- `forecast` - მომავლის პროგნოზის ნაწილი.

N-BEATS მუშაობს residual პრინციპით:

1. პირველი block იღებს input-ს.
2. block აბრუნებს backcast-ს და forecast-ს.
3. residual = residual - backcast.
4. შემდეგი block სწავლობს დარჩენილი residual signal-ის ახსნას.
5. ყველა block-ის forecast ჯამდება საბოლოო პროგნოზად.

ეს დიზაინი ეხმარება მოდელს complex time series pattern-ების ეტაპობრივად სწავლაში.

## 16. Brute-force hyperparameter search

Notebook-ში დამატებულია brute-force grid search. ეს ნიშნავს, რომ წინასწარ განსაზღვრული hyperparameter values-ის ყველა კომბინაცია ისინჯება.

Grid არის:

```python
batch_size = [32, 64, 128]
learning_rate = [1e-3, 1e-2, 1e-1]
hidden_units = [128, 256]
num_blocks = [3, 4]
num_layers = [3, 4]
dropout = [0.0, 0.10]
weight_decay = [0.0, 1e-4]
```

სულ არის:

```text
3 * 3 * 2 * 2 * 2 * 2 * 2 = 288 combinations
```

თითოეული combination სწავლობს 30 epoch-ს.

ეს არის brute-force, რადგან ყველა შესაძლო კომბინაცია თანმიმდევრულად იტრეინება და არა Optuna/TPE/random search-ის მსგავსად შერჩევითად.

## 17. რატომ ეს hyperparameters

### `batch_size`

`batch_size` განსაზღვრავს, რამდენ training window-ს ხედავს მოდელი ერთ optimization step-ში.

- `32` - უფრო noisy gradient, ზოგჯერ უკეთესი generalization.
- `64` - საშუალო ვარიანტი.
- `128` - უფრო სტაბილური gradient და ხშირად უფრო სწრაფი training.

### `learning_rate`

`learning_rate` განსაზღვრავს, რა ზომის ნაბიჯებს დგამს optimizer.

- `1e-3` - conservative და ხშირად სტაბილური.
- `1e-2` - უფრო სწრაფი სწავლა, მაგრამ შეიძლება არასტაბილური იყოს.
- `1e-1` - აგრესიული learning rate, შეიძლება სწრაფად ისწავლოს ან diverge მოხდეს.

### `hidden_units`

ეს არის dense layer-ების სიგანე.

- `128` - პატარა მოდელი, ნაკლები overfitting risk.
- `256` - უფრო ძლიერი მოდელი, მეტი capacity.

### `num_blocks`

N-BEATS block-ების რაოდენობა.

- მეტი block ნიშნავს, რომ მოდელს შეუძლია residual signal-ის უფრო ეტაპობრივად დამუშავება.
- ძალიან ბევრი block ზრდის training time-ს და overfitting risk-ს.

### `num_layers`

თითო block-ში dense layer-ების რაოდენობა.

- `3` - მარტივი block.
- `4` - უფრო ღრმა block, მეტი non-linear capacity.

### `dropout`

Dropout გამოიყენება regularization-ისთვის.

- `0.0` - dropout არ გამოიყენება.
- `0.10` - layer-ის activation-ების ნაწილი ითიშება training დროს, რაც overfitting-ის შემცირებას ეხმარება.

### `weight_decay`

Weight decay არის L2 regularization AdamW optimizer-ში.

- `0.0` - regularization არ არის.
- `1e-4` - მცირე regularization, რომელიც შეიძლება დაეხმაროს generalization-ს.

## 18. Training loop

თითოეული grid combination-ისთვის:

1. იქმნება ახალი N-BEATS მოდელი.
2. იქმნება AdamW optimizer.
3. train loader იყენებს შესაბამის `batch_size`-ს.
4. მოდელი სწავლობს 30 epoch-ს.
5. ყოველ epoch-ზე ითვლება:
   - train L1 loss log-normalized scale-ზე.
   - validation Weighted MAE original sales scale-ზე.
   - validation MAE.
   - validation RMSE.
6. metrics იგზავნება W&B-ში.
7. თუ trial-ის epoch უკეთესია, ინახება trial-ის best state.

მთავარი selection metric არის:

```text
validation Weighted MAE
```

ეს ემთხვევა Walmart competition-ის ლოგიკას, სადაც holiday weeks უფრო მაღალი weight-ით ფასდება.

## 19. Best model selection

ყველა trial-ის დასრულების შემდეგ notebook არჩევს იმ trial-ს, რომელსაც აქვს ყველაზე დაბალი validation Weighted MAE.

ინახება:

- best trial id
- best W&B run name
- best epoch
- best validation Weighted MAE
- best hyperparameters
- model state dict
- preprocessing metadata
- series normalization parameters
- validation dates
- series index

ეს ყველაფერი ინახება `nbeats_best_model.pt` ფაილში.

## 20. Grid search summary W&B-ში

Grid search-ის დასრულების შემდეგ იქმნება ცალკე W&B summary run.

იქ ლოგდება:

- ყველა trial-ის შედეგების table.
- best trial id.
- best validation Weighted MAE.
- best hyperparameters.

ეს საჭიროა, რომ W&B-ში მარტივად შეადარო ყველა hyperparameter combination და ნახო, რომელმა იმუშავა საუკეთესოდ.

## 21. Validation diagnostics

Best model იტვირთება და validation period-ზე კეთდება პროგნოზები.

შემდეგ იქმნება:

- `validation_predictions.csv`
- `weekly_validation_errors.csv`
- actual vs predicted scatter plot
- weekly MAE plot

ეს დიაგნოსტიკა აჩვენებს:

- მოდელი ზოგადად სწორ scale-ზე პროგნოზირებს თუ არა.
- რომელ validation კვირებზე აქვს მაღალი შეცდომა.
- holiday weeks-ზე ხომ არ უჭირს.

ეს ნაწილი მნიშვნელოვანია, რადგან მხოლოდ ერთი final metric არ გვიჩვენებს, სად უშვებს მოდელი შეცდომებს.

## 22. Best model artifact

Best model bundle ლოგდება W&B artifact-ად.

Artifact-ში შედის:

- `nbeats_best_model.pt`
- `metadata.json`
- `config.json`
- `series_index.csv`
- `validation_predictions.csv`
- `weekly_validation_errors.csv`
- `nbeats_grid_search_results.csv`

ეს ნიშნავს, რომ W&B-ში ინახება არა მარტო model weights, არამედ ყველა საჭირო metadata, რომ მომავალში გავიგოთ როგორ შეიქმნა მოდელი.

## 23. W&B Model Registry

Best artifact რეგისტრირდება W&B Model Registry-ში სახელით:

```text
NBEATS-Best-Model
```

aliases:

- `best`
- `latest`
- `production-candidate`

Registry საჭიროა, რომ საუკეთესო მოდელი ცალკე იყოს გამოყოფილი training runs-ისგან და მომავალში მარტივად მოხდეს მისი გამოყენება, შედარება ან promotion.

## 24. რას არ აკეთებს ეს notebook

ეს notebook არ აკეთებს final `test.csv` submission-ს. ის ამ ეტაპზე ფოკუსირებულია model experiment-ზე:

- preprocessing
- validation split
- hyperparameter search
- model training
- validation evaluation
- W&B logging
- best model registration

Test prediction-ისთვის საჭიროა ცალკე inference/submission notebook ან დამატებითი cell, სადაც იგივე preprocessing logic გამოყენებული იქნება test period-ისთვის.

## 25. მნიშვნელოვანი შენიშვნა runtime-ზე

Grid search საკმაოდ მძიმეა:

```text
288 combinations * 30 epochs
```

ეს შეიძლება დიდხანს გაგრძელდეს, განსაკუთრებით CPU-ზე.

Notebook-ში არის debug კონტროლი:

```python
MAX_GRID_RUNS = None
```

თუ გინდა მხოლოდ შემოწმება, შეგიძლია დროებით დააყენო:

```python
MAX_GRID_RUNS = 5
```

ამ შემთხვევაში გაეშვება მხოლოდ პირველი 5 combination. სრული brute-force training-ისთვის უნდა დარჩეს:

```python
MAX_GRID_RUNS = None
```

## 26. სრული pipeline-ის შეჯამება

საბოლოოდ N-BEATS notebook-ის pipeline ასეთია:

1. Environment setup.
2. Data loading.
3. Data validation.
4. Time sorting.
5. Store-Dept სერიებად pivot.
6. Weekly date grid.
7. Short series filtering.
8. Missing weeks filling.
9. Last 32 weeks validation split.
10. Log target transform.
11. Per-series normalization.
12. Sliding windows creation.
13. Preprocessing logging in W&B.
14. Seasonal naive baseline.
15. N-BEATS model definition.
16. Brute-force hyperparameter search.
17. 30 epoch training per combination.
18. Best model selection by validation Weighted MAE.
19. Validation diagnostics.
20. Best model artifact logging.
21. W&B Model Registry registration.

ეს სტრუქტურა უზრუნველყოფს, რომ ექსპერიმენტი იყოს reproducible, time-series leakage-ის გარეშე, W&B-ში სრულად დალოგილი და საუკეთესო მოდელი ცალკე რეგისტრირებული.
