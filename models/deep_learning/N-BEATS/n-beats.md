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

- `context_length` გაზრდა `52`-დან `104` კვირამდე;
- ან model capacity-ის შემცირება, რადგან overfitting/generalization პრობლემა კვლავ ჩანს;
- ან holiday-aware loss/sample weighting, რადგან competition metric holiday weeks-ს უფრო დიდ წონას აძლევს.

## Experiment 2: context length 104

`model_experiment_N-BEATS.ipynb`-ში შემდეგი ექსპერიმენტისთვის დაემატა ერთი isolated ცვლილება:

```text
context_length = 104
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

ამ ექსპერიმენტის მიზანია შემოწმდეს, ეხმარება თუ არა N-BEATS-ს უფრო გრძელი ისტორია. ანუ მოდელი იღებს არა ბოლო 52 კვირას, არამედ ბოლო 104 კვირას და პროგნოზირებს შემდეგ 32 კვირას.

შესადარებელი reference შედეგები:

```text
Baseline best WMAE = 2157.9829
Experiment 1 best WMAE = 2186.5015
```

Experiment 2 გაუმჯობესებულად ჩაითვლება მხოლოდ მაშინ, თუ მისი best validation WMAE იქნება `2157.9829`-ზე დაბალი.

შედეგი ჯერ გასაშვებია.
