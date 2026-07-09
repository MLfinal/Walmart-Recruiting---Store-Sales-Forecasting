# N-BEATS ექსპერიმენტების ანალიზი

ეს ფაილი გამოიყენება N-BEATS მოდელების შედეგების ჩასაწერად. აქ უნდა დაემატოს თითოეული ექსპერიმენტის შედეგი: რა შეიცვალა, გაუმჯობესდა თუ არა validation metric და რა დასკვნა გამოვიტანეთ.

## Baseline model-ის ანალიზი

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

## Experiment 1: lower learning rate + early stopping

`model_experiment_N-BEATS.ipynb`-ში დამატებულია პირველი ექსპერიმენტი:

- preprocessing იგივე რჩება, რაც baseline-ში;
- feature engineering ჯერ არ ემატება;
- learning rate შემცირდა `1e-3`-დან `3e-4`-ზე;
- დაემატა early stopping `patience = 5`;
- validation metric კვლავ არის Weighted MAE.

ამ ექსპერიმენტის მიზანია შემოწმდეს, baseline-ის early overfitting/generalization პრობლემა მცირდება თუ არა უფრო ნელი სწავლით და early stopping-ით.

### შედეგი

Experiment 1 გაეშვა შემდეგი ცვლილებებით:

```text
learning_rate = 3e-4
early_stopping_patience = 5
max_epochs = 30
```

მოდელი გაჩერდა early stopping-ით მე-12 epoch-ზე:

```text
Early stopping triggered at epoch 12.
Best epoch was 7.
```

საუკეთესო შედეგი:

```text
Best epoch: 7
Best validation WMAE: 2186.5015
Best validation MAE: 2157.4897
```

Epoch-ების მიხედვით validation WMAE ასე იცვლებოდა:

```text
epoch 1  validation WMAE = 2237.9617
epoch 4  validation WMAE = 2200.6792
epoch 7  validation WMAE = 2186.5015
epoch 12 validation WMAE = 2228.5127
```

### Baseline-თან შედარება

Baseline-ის საუკეთესო შედეგი იყო:

```text
Baseline best WMAE = 2157.9829 at epoch 2
```

Experiment 1-ის საუკეთესო შედეგია:

```text
Experiment 1 best WMAE = 2186.5015 at epoch 7
```

შედარება:

```text
2186.5015 - 2157.9829 = +28.5186
```

რადგან Weighted MAE უფრო დაბალი უკეთესია, Experiment 1-მა baseline ვერ გააუმჯობესა. შედეგი დაახლოებით `28.52`-ით უარესია.

### ანალიზი

Lower learning rate-მა training პროცესი უფრო ნელი და შედარებით სტაბილური გახადა. Baseline-ში საუკეთესო epoch იყო `2`, ხოლო Experiment 1-ში საუკეთესო epoch გახდა `7`. ეს ნიშნავს, რომ მოდელი უფრო ნელა მივიდა საუკეთესო validation შედეგამდე.

მაგრამ საბოლოო generalization არ გაუმჯობესდა, რადგან საუკეთესო validation WMAE baseline-ზე მაღალია:

- baseline: `2157.9829`
- experiment 1: `2186.5015`

Training loss კვლავ მცირდებოდა:

```text
epoch 1  train L1 = 0.54343
epoch 12 train L1 = 0.44583
```

მაგრამ validation WMAE საუკეთესო შედეგის შემდეგ ისევ უარესდებოდა. ეს კვლავ მიუთითებს, რომ მოდელი training data-ზე სწავლას აგრძელებს, მაგრამ validation period-ზე უკეთესად ვერ generalize-დება.

### დასკვნა

Experiment 1-მა არ გააუმჯობესა baseline.

შედეგი:

```text
Not improved
```

Lower learning rate + early stopping დაეხმარა training-ის კონტროლს და ცუდი გვიანი epoch-ების შეჩერებას, მაგრამ validation score baseline-ზე უკეთესი არ გახდა. შემდეგი ნაბიჯისთვის მხოლოდ learning rate-ის შემცირება საკმარისი არ ჩანს.

შემდეგი ექსპერიმენტისთვის უფრო ლოგიკურია ერთი სხვა ცვლილების შემოწმება, მაგალითად:

- `context_length` გაზრდა `52`-დან `78` კვირამდე;
- ან model capacity-ის შემცირება, რადგან overfitting/generalization პრობლემა კვლავ ჩანს;
- ან holiday-aware loss/sample weighting, რადგან competition metric holiday weeks-ს უფრო დიდ წონას აძლევს.

## Experiment 2: context length 78

`model_experiment_N-BEATS.ipynb`-ში შემდეგი ექსპერიმენტისთვის დაემატა ერთი isolated ცვლილება:

```text
context_length = 78
```

Baseline-ში გამოყენებული იყო:

```text
context_length = 52
```

Experiment 1-ის lower learning rate + early stopping ცვლილება ამ ექსპერიმენტში არ რჩება, რადგან მან baseline ვერ გააუმჯობესა. Experiment 2-ში training setup დაბრუნებულია baseline-ის ლოგიკაზე:

```text
learning_rate = 1e-3
max_epochs = 30
early stopping = no
```

ამ ექსპერიმენტის მიზანია შემოწმდეს, ეხმარება თუ არა N-BEATS-ს უფრო გრძელი ისტორია. ანუ მოდელი იღებს არა ბოლო 52 კვირას, არამედ ბოლო 78 კვირას და პროგნოზირებს შემდეგ 32 კვირას.

თავდაპირველად განიხილებოდა `context_length = 104`, მაგრამ ამ validation split-ით training period-ში არ რჩება საკმარისი ისტორია სრული training window-ისთვის: input `104` კვირა + target `32` კვირა. ამიტომ leak-free ექსპერიმენტისთვის გამოყენებულია `78` კვირა, რომელიც baseline-ზე გრძელია და მაინც ქმნის training windows-ს validation-მდე.

შესადარებელი reference შედეგები:

```text
Baseline best WMAE = 2157.9829
Experiment 1 best WMAE = 2186.5015
```

Experiment 2 გაუმჯობესებულად ჩაითვლება მხოლოდ მაშინ, თუ მისი best validation WMAE იქნება `2157.9829`-ზე დაბალი.

### შედეგი

Experiment 2 გაეშვა შემდეგი ცვლილებით:

```text
context_length = 78
learning_rate = 1e-3
max_epochs = 30
early stopping = no
```

საუკეთესო validation შედეგი მიიღო მე-2 epoch-ზე:

```text
Best epoch: 2
Best validation WMAE: 2662.8061
Best validation MAE: 2623.1826
```

Epoch-ების მიხედვით validation WMAE იყო:

```text
epoch 1  validation WMAE = 2682.9611
epoch 2  validation WMAE = 2662.8061
epoch 3  validation WMAE = 2706.3169
epoch 10 validation WMAE = 2685.4876
epoch 20 validation WMAE = 2711.2920
epoch 30 validation WMAE = 2714.2796
```

Training loss კი მუდმივად მცირდებოდა:

```text
epoch 1  train L1 = 0.59188
epoch 30 train L1 = 0.35715
```

### Baseline-თან და Experiment 1-თან შედარება

Reference შედეგები:

```text
Baseline best WMAE     = 2157.9829
Experiment 1 best WMAE = 2186.5015
Experiment 2 best WMAE = 2662.8061
```

Baseline-თან სხვაობა:

```text
2662.8061 - 2157.9829 = +504.8232
```

Experiment 1-თან სხვაობა:

```text
2662.8061 - 2186.5015 = +476.3046
```

რადგან Weighted MAE უფრო დაბალი უკეთესია, Experiment 2 მკვეთრად უარესია როგორც baseline-ზე, ასევე Experiment 1-ზე.

### ანალიზი

`context_length = 78` იდეა იყო, რომ მოდელს უფრო გრძელი ისტორია ენახა და შეიძლება უკეთესად დაეჭირა seasonality ან department/store pattern-ები. მაგრამ შედეგმა აჩვენა, რომ ამ კონკრეტულ setup-ში უფრო გრძელი context არ ეხმარება.

მნიშვნელოვანი დაკვირვება:

- training loss ძალიან კარგად მცირდება;
- validation WMAE თავიდანვე მაღალია;
- საუკეთესო შედეგი ისევ ძალიან ადრე, მე-2 epoch-ზე მიიღება;
- epoch 3-ის შემდეგ validation ძირითადად `2680-2715` დიაპაზონში რჩება;
- ეს ნიშნავს, რომ მოდელი training windows-ზე უკეთეს fit-ს სწავლობს, მაგრამ validation period-ზე generalization მკვეთრად უარესდება.

ამის სავარაუდო მიზეზებია:

- `78` კვირიანი input ამცირებს training windows-ის რაოდენობას, რადგან თითო sample-ს მეტი ისტორია სჭირდება;
- გრძელი context მოდელს აძლევს მეტ ინფორმაციას, მაგრამ ასევე მეტ noise-ს;
- N-BEATS baseline architecture შეიძლება ვერ იყენებდეს დამატებით 26 კვირას ეფექტიანად;
- Walmart-ის ბოლო 32 კვირის validation period შეიძლება უფრო ახლო recent pattern-ებზე იყოს დამოკიდებული, ვიდრე 78 კვირიან გრძელ ისტორიაზე;
- უფრო გრძელი input ზრდის model fitting complexity-ს და overfitting/generalization პრობლემა უფრო ძლიერდება.

### დასკვნა

Experiment 2-მა არ გააუმჯობესა baseline.

შედეგი:

```text
Not improved
```

`context_length = 78` მნიშვნელოვნად უარესია baseline-ზე. ამიტომ ამ მიმართულებით გაგრძელება ამ ეტაპზე არ არის რეკომენდებული.

შემდეგი ექსპერიმენტისთვის უკეთესი იქნება არა context-ის გაზრდა, არამედ model capacity-ის შემცირება ან holiday-aware loss/sample weighting-ის დამატება. იმის გამო, რომ validation metric holiday weeks-ს უფრო დიდ წონას აძლევს, შემდეგი ლოგიკური ნაბიჯი შეიძლება იყოს holiday-aware weighted loss, სადაც forecast horizon-ის holiday კვირებს training loss-ში მეტი წონა ექნება.

## Experiment 3: holiday-aware weighted loss

`model_experiment_N-BEATS.ipynb`-ში შემდეგი ექსპერიმენტისთვის დაემატა holiday-aware training loss.

ამ ექსპერიმენტში baseline-ის ძირითადი setup დაბრუნებულია, მაგრამ training duration გაიზარდა 100 epoch-მდე:

```text
context_length = 52
learning_rate = 1e-3
max_epochs = 100
early stopping = no
```

Experiment 2-ის `context_length = 78` არ გამოიყენება, რადგან მან validation WMAE მნიშვნელოვნად გააუარესა.

### რა დაემატა

Training target horizon-ის თითოეულ კვირას ენიჭება weight:

```text
normal week  -> weight = 1
holiday week -> weight = 5
```

შემდეგ training loss ითვლება weighted L1-ით:

```text
weighted_l1 = sum(abs(prediction - target) * holiday_weight) / sum(holiday_weight)
```

ეს feature engineering არ ამატებს ახალ input feature-ს მოდელში. N-BEATS კვლავ იღებს მხოლოდ historical sales sequence-ს. ცვლილება არის training objective-ში: მოდელს უფრო მეტად ვასწავლით holiday target weeks-ზე სწორ პროგნოზს, რადგან competition metric-შიც holiday weeks უფრო მაღალ წონას იღებს.

### რატომ არის ეს ლოგიკური შემდეგი ნაბიჯი

წინა შედეგები:

```text
Baseline best WMAE     = 2157.9829
Experiment 1 best WMAE = 2186.5015
Experiment 2 best WMAE = 2662.8061
```

ორივე ექსპერიმენტმა აჩვენა, რომ მხოლოდ training speed-ის შეცვლა ან history-ის გაზრდა საკმარისი არ იყო. რადგან evaluation metric არის weighted და holiday weeks-ს უფრო დიდ მნიშვნელობას ანიჭებს, ლოგიკურია training loss-იც იგივე პრიორიტეტს მიჰყვეს.

Experiment 3 გაუმჯობესებულად ჩაითვლება მხოლოდ მაშინ, თუ მისი best validation WMAE იქნება baseline-ზე დაბალი:

```text
target: best WMAE < 2157.9829
```

### შედეგი

Experiment 3 გაეშვა 100 epoch-ზე holiday-aware weighted loss-ით.

საუკეთესო validation შედეგი მიიღო მე-5 epoch-ზე:

```text
Best epoch: 5
Best validation WMAE: 2185.1366
Best validation MAE: 2160.8167
```

Epoch-ების მიხედვით მნიშვნელოვანი წერტილები:

```text
epoch 1   validation WMAE = 2216.2192
epoch 5   validation WMAE = 2185.1366
epoch 10  validation WMAE = 2230.4306
epoch 30  validation WMAE = 2237.8199
epoch 50  validation WMAE = 2238.5973
epoch 100 validation WMAE = 2238.7149
```

Training weighted loss შემცირდა:

```text
epoch 1   train weighted L1 = 0.54881
epoch 100 train weighted L1 = 0.41390
```

### Baseline-თან და წინა ექსპერიმენტებთან შედარება

Reference შედეგები:

```text
Baseline best WMAE     = 2157.9829
Experiment 1 best WMAE = 2186.5015
Experiment 2 best WMAE = 2662.8061
Experiment 3 best WMAE = 2185.1366
```

Baseline-თან სხვაობა:

```text
2185.1366 - 2157.9829 = +27.1537
```

Experiment 1-თან სხვაობა:

```text
2185.1366 - 2186.5015 = -1.3649
```

Experiment 2-თან სხვაობა:

```text
2185.1366 - 2662.8061 = -477.6695
```

რადგან Weighted MAE უფრო დაბალი უკეთესია, Experiment 3:

- baseline-ზე უარესია;
- Experiment 1-ზე ოდნავ უკეთესია;
- Experiment 2-ზე ბევრად უკეთესია.

მაგრამ მთავარი reference არის baseline, ამიტომ Experiment 3 საბოლოოდ გაუმჯობესებად არ ითვლება.

### ანალიზი

Holiday-aware weighted loss იყო ლოგიკური იდეა, რადგან validation metric holiday weeks-ს უფრო დიდ weight-ს აძლევს. ამ ცვლილებამ მოდელი ოდნავ უკეთესი გახადა Experiment 1-თან შედარებით, მაგრამ baseline-ის საუკეთესო შედეგი ვერ გადალახა.

მნიშვნელოვანი დაკვირვება:

- საუკეთესო შედეგი ისევ ძალიან ადრე, მე-5 epoch-ზე მიიღება;
- training weighted loss შემდეგაც მცირდება;
- validation WMAE მე-5 epoch-ის შემდეგ უარესდება და დაახლოებით `2238`-თან სტაბილურდება;
- 100 epoch-მდე გაშვებამ დამატებითი გაუმჯობესება არ მოიტანა.

ეს ნიშნავს, რომ მხოლოდ holiday-aware weighted loss საკმარისი არ არის. მოდელი კვლავ უკეთ ერგება training windows-ს, მაგრამ validation period-ზე უკეთესად ვერ generalize-დება.

100 epoch-ის შედეგიც აჩვენებს, რომ მეტი epoch არ არის გამოსავალი. საუკეთესო epoch იყო `5`, ხოლო epoch `100` უკვე ბევრად უარესია:

```text
best WMAE at epoch 5   = 2185.1366
final WMAE at epoch 100 = 2238.7149
```

### დასკვნა

Experiment 3-მა baseline არ გააუმჯობესა.

შედეგი:

```text
Not improved vs baseline
```

თუმცა Experiment 3 ოდნავ უკეთესია Experiment 1-ზე, ამიტომ holiday-aware loss მთლიანად უინტერესო არ არის. უბრალოდ ამ fixed hyperparameters-ით baseline-ზე უკეთესი შედეგი ვერ მიიღო.

შემდეგი ნაბიჯი შეიძლება იყოს hyperparameter tuning, მაგრამ tuning-ის დროს სასურველია შევადაროთ ორი ვარიანტი:

- regular L1 loss;
- holiday-aware weighted L1 loss.

ასე გამოჩნდება, weighted loss სხვა hyperparameters-თან ერთად უკეთ მუშაობს თუ არა.

## Final Experiment: Optuna hyperparameter tuning

`model_experiment_N-BEATS.ipynb` გადაკეთდა final Optuna hyperparameter tuning ექსპერიმენტად.

Grid search-ის ნაცვლად გამოიყენება Optuna, რადგან სრული grid უკვე საკმაოდ მძიმე იქნებოდა. Optimizer-ის დამატების შემდეგ grid იქნებოდა 128 trial, ხოლო Optuna საშუალებას გვაძლევს ნაკლები trial-ით ვცადოთ უფრო პერსპექტიული კომბინაციები.

წინა ექსპერიმენტებიდან მთავარი დასკვნები იყო:

```text
Baseline best WMAE     = 2157.9829
Experiment 1 best WMAE = 2186.5015
Experiment 2 best WMAE = 2662.8061
Experiment 3 best WMAE = 2185.1366
```

საუკეთესო შედეგი კვლავ baseline-ს ჰქონდა. თუმცა Experiment 3-ში holiday-aware weighted loss ოდნავ უკეთესი იყო Experiment 1-ზე, ამიტომ Optuna search-ში კვლავ შედარდება ორივე loss type:

- regular L1 loss;
- holiday-aware weighted L1 loss.

### Optuna search setup

Final Optuna tuning იყენებს baseline-ის preprocessing-ს:

```text
context_length = 52
forecast_horizon = 32
validation_weeks = 32
```

Optuna ცდის შემდეგ search space-ს:

```text
optimizer     = sgd, adam
loss_type     = regular_l1, holiday_weighted_l1
batch_size    = 64, 128
learning_rate = 1e-4 ... 1e-2, log scale
hidden_units  = 128, 256
dropout       = 0.0 ... 0.20
weight_decay  = 1e-6 ... 1e-3, log scale
```

Trial-ების რაოდენობა notebook-ში არის:

```text
OPTUNA_N_TRIALS = 8
```

ფიქსირებული პარამეტრები:

```text
num_blocks = 4
num_layers = 4
max_epochs = 20
early_stopping_patience = 4
```

Early stopping დაემატა იმიტომ, რომ წინა ყველა ექსპერიმენტში საუკეთესო validation შედეგი ადრეულ epoch-ებზე მიიღებოდა. 100 epoch-მდე სწავლა ხშირად აღარ აუმჯობესებდა validation WMAE-ს.

### რას ლოგავს W&B-ში

Notebook ლოგავს:

- preprocessing run-ს;
- თითო Optuna trial-ს ცალკე W&B run-ად;
- train loss-ს;
- validation Weighted MAE-ს;
- validation MAE/RMSE-ს;
- Optuna summary table-ს;
- best trial diagnostics-ს;
- best validation predictions-ს და weekly errors-ს.

ამ ეტაპზე model artifact და Model Registry არ იქმნება, რადგან ჯერ გვინდა მხოლოდ validation-ზე საუკეთესო setup-ის პოვნა.

### მიზანი

Final Optuna tuning გაუმჯობესებულად ჩაითვლება მხოლოდ მაშინ, თუ საუკეთესო trial მიიღებს baseline-ზე დაბალ validation WMAE-ს:

```text
target: best Optuna WMAE < 2157.9829
```

შედეგი ჯერ გასაშვებია.

### Model Registry და inference flow

დავალების მოთხოვნის მიხედვით N-BEATS-ის საუკეთესო მოდელი უნდა შეინახოს pipeline-ად და inference დროს ჩაიტვირთოს W&B Model Registry-დან.

ამიტომ რეგისტრაციის ლოგიკა დაემატა:

```text
model_experiment_N-BEATS.ipynb
```

ამ notebook-ში Optuna tuning-ის შემდეგ ხდება სწორი model selection:

- თუ Optuna-ს საუკეთესო trial baseline-ს აჯობებს, რეგისტრირდება Optuna-ს საუკეთესო trial;
- თუ Optuna baseline-ს ვერ აჯობებს, რეგისტრირდება baseline configuration, რადგან ამ არქიტექტურის საუკეთესო შედეგი baseline-მა აჩვენა.

ეს მნიშვნელოვანია, რადგან ამჟამინდელი შედეგებით baseline არის საუკეთესო N-BEATS მოდელი:

```text
Baseline best WMAE = 2157.9829
Best partial Optuna WMAE = 2191.4117
```

ანუ baseline უკეთესია. ამიტომ registry-ში არ უნდა მოხვდეს სუსტი Optuna candidate, თუ მან baseline ვერ გადალახა.

არჩეული მოდელი იფუთება raw-input pipeline-ად:

```text
walmart_nbeats_raw_pipeline.pkl
```

Pipeline ინახავს:

- trained N-BEATS model weights;
- საუკეთესო trial-ის hyperparameters;
- per-series normalization statistics;
- Store-Dept historical sales matrix;
- forecasting logic-ს, რომელიც raw `test.csv` rows-ზე პირდაპირ მუშაობს.

Pipeline-ის prediction interface არის:

```python
pipeline.predict(raw_test_df)
```

ანუ inference notebook-ში feature engineering ხელახლა არ უნდა დაიწეროს. Pipeline თვითონ აკეთებს საჭირო preprocessing/forecasting logic-ს.

W&B Registry target:

```text
wandb-registry-model/Walmart_NBEATS_Pipeline
```

Aliases:

- `best-nbeats`
- `latest`
- `champion`, თუ არჩეულმა მოდელმა baseline reference-ს აჯობა;
- `candidate`, თუ baseline reference მაინც უკეთესია.

Inference უნდა მოხდეს ფაილში:

```text
N-BEATS_inference.ipynb
```

ეს notebook აკეთებს:

1. W&B Registry-დან pipeline artifact-ის ჩამოტვირთვას;
2. `test.csv`-ის raw ფორმით ჩატვირთვას;
3. `pipeline.predict(test_raw)` გამოძახებას;
4. Kaggle submission CSV-ის შექმნას;
5. submission artifact-ის W&B-ში დალოგვას.


### Partial Optuna run observation

პირველი Optuna მცდელობა შეწყდა, რადგან 25 trial და 100 epoch ძალიან დიდ დროს იღებდა. პირველმა რამდენიმე trial-მაც baseline ვერ გააუმჯობესა:

```text
Trial 0 best WMAE = 2191.4117
Trial 1 best WMAE = 2400.4774
Trial 2 best WMAE = 2199.4592
Trial 3 best WMAE = 2201.6325
Baseline best WMAE = 2157.9829
```

ამიტომ Optuna setup შემცირდა სწრაფი tuning pass-ისთვის:

```text
OPTUNA_N_TRIALS = 8
max_epochs = 20
early_stopping_patience = 4
MedianPruner warmup = 3 epochs
```

ეს ვერსია უფრო სწრაფად უნდა გაეშვას და მიზანია სწრაფად ვნახოთ, არის თუ არა N-BEATS-ის tuning-ში baseline-ზე უკეთესი signal.

## Kaggle submission ანალიზი

Final Kaggle submission-ზე N-BEATS-ის score მივიღე:

```text
Kaggle score: 4700
```

Validation-ზე საუკეთესო observed result იყო baseline configuration:

```text
Baseline best WMAE = 2157.9829
```

Kaggle-ზე score ბევრად გაუარესდა. ჩემი აზრით, მთავარი მიზეზი არის ის, რომ N-BEATS validation უფრო მარტივ და კონტროლირებულ forecast setup-ს ამოწმებდა, ხოლო Kaggle test-ზე მოდელს უფრო რთული future horizon დახვდა.

რატომ გაუარესდა Kaggle-ზე:

- N-BEATS prediction დიდად ეყრდნობა ბოლო historical window-ს.
- თუ test horizon გრძელია, forecast error დროთა განმავლობაში გროვდება.
- მოდელს არ აქვს explicit holiday, markdown, store metadata და external feature context ისეთი ფორმით, როგორც tree-based models-ს.
- Walmart sales-ში promotion/holiday effect ძალიან ძლიერია, მაგრამ pure historical neural model ამას ბოლომდე ვერ იჭერს.
- ბევრი Store-Dept series sparse/noisy არის, რაც N-BEATS-ს რთულს ხდის.

რატომ იყო baseline უკეთესი Optuna-ზე:

Optuna-ს პირველმა trial-ებმა baseline ვერ გააუმჯობესა:

```text
Best partial Optuna WMAE = 2191.4117
Baseline best WMAE = 2157.9829
```

ჩემი შეფასებით, tuning-მა ვერ იპოვა ისეთი configuration, რომელიც N-BEATS-ს ამ dataset-ზე რეალურად გააძლიერებდა. სავარაუდოდ პრობლემა მარტო hyperparameter-ებში არ არის; თვითონ architecture ამ კონკრეტულ tabular+holiday forecasting task-ს ნაკლებად ერგება, თუ მას external covariates არ დავუმატებთ.

შედარება სხვა მოდელებთან:

```text
N-BEATS Kaggle score = 4700
DLinear Kaggle score = 3500
XGBoost Kaggle score = 2806
```

N-BEATS ყველაზე სუსტი გამოვიდა ამ სამიდან. ჩემი დასკვნა არის, რომ Walmart competition-ში მხოლოდ neural sequence model საკმარისი არ არის. საჭიროა ისეთი feature-ები, რომლებიც ბიზნეს context-ს აღწერს: holiday proximity, markdowns, store/dept identity, yearly lag და historical aggregates.

საბოლოო დასკვნა:

N-BEATS კარგი baseline იყო deep learning ექსპერიმენტისთვის, მაგრამ final model candidate-ად არ ავირჩევდი. მისი Kaggle score აჩვენებს, რომ ამ ამოცანაში feature-rich tree-based approach უფრო საიმედოა.
