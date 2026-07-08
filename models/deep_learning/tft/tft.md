# Temporal Fusion Transformer

ფოლდერი: `models/deep_learning/tft`

ამ ეტაპზე შექმნილია პირველი TFT baseline notebook:

```text
baseline_tft.ipynb
```

Baseline-ის მიზანია მივიღოთ პირველი leakage-safe TFT შედეგი იგივე protocol-ით, რასაც DLinear-ში ვიყენებდით:

- validation არის `train.csv`-ის ბოლო 39 კვირა;
- metric არის Kaggle-style WMAE original sales scale-ზე;
- W&B-ზე ილოგება config, training curves, validation WMAE, plots, prediction table, checkpoint და summary artifact;
- MLflow არ გამოიყენება.

## პირველი full-data მცდელობა

პირველი TFT baseline გავუშვით full Store-Dept data-ზე, მაგრამ Colab-ზე პრაქტიკულად ძალიან ნელი აღმოჩნდა.

მთავარი პრობლემა model size არ იყო. model-ს ჰქონდა დაახლოებით `26.2K` trainable parameters, რაც პატარაა. პრობლემა იყო sliding-window dataset:

```text
3331 Store-Dept series
52-week encoder
39-week decoder
~1504 train batches per epoch
```

Colab run-ში პირველი epoch-ის პროგრესი ძალიან ნელა მიდიოდა და projected time ერთ epoch-ზე დაახლოებით `10–13` წუთი იყო. მომხმარებლის დაკვირვებით run-მა დაახლოებით `1` საათი წაიღო მხოლოდ რამდენიმე batch/progress step-ზე, ამიტომ full-data baseline ამ ფორმით არ არის მისაღები.

ასევე გამოჩნდა warning:

```text
Loss is not finite. Resetting it to 1e9
```

ეს ნიშნავს, რომ raw target/normalization/training setup ზოგ batch-ზე unstable იყო. ამიტომ baseline notebook შევცვალეთ ისე, რომ ჯერ დაადასტუროს pipeline და W&B logging, და არა full-scale TFT performance.

## განახლებული fast baseline

ახლანდელი `baseline_tft.ipynb` არის Colab-safe sanity baseline. მიზანია მაქსიმუმ დაახლოებით `10` წუთში მივიღოთ პირველი TFT result და W&B logs.

ძირითადი ცვლილებები:

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

ასევე `Weekly_Sales` target იჭრება `>= 0`:

```text
Weekly_Sales = clip(lower=0)
```

ეს ამცირებს non-finite loss-ის რისკს და შეესაბამება submission logic-ს, რადგან საბოლოო პროგნოზიც არ უნდა იყოს negative.

მნიშვნელოვანი caveat:

```text
ეს baseline უკვე აღარ არის full-data final TFT score.
ეს არის fast TFT pipeline/logging baseline.
```

შედეგი უნდა შევადაროთ ფრთხილად: DLinear/XGBoost full validation-ზეა, ხოლო fast TFT baseline მხოლოდ top active Store-Dept series-ზე train/validate ხდება. Full-data ან larger-sample TFT უნდა გაკეთდეს მხოლოდ მაშინ, თუ fast baseline მუშაობს სტაბილურად და W&B logging სწორია.

Baseline feature set:

- target history: `Weekly_Sales`;
- static categoricals: `Store`, `Dept`;
- known future categorical: `IsHoliday`;
- known future calendar reals: `time_idx`, week sine/cosine, month sine/cosine.

External covariates (`features.csv`, `stores.csv`) ამ baseline-ში შეგნებულად არ არის დამატებული. ისინი უნდა დაემატოს შემდეგ TFT experiment-ში, baseline result-ის შემდეგ.
