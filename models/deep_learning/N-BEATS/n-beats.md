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

შედეგი ჯერ გასაშვებია.
