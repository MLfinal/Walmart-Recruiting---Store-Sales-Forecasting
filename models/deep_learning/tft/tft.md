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

Baseline feature set:

- target history: `Weekly_Sales`;
- static categoricals: `Store`, `Dept`;
- known future categorical: `IsHoliday`;
- known future calendar reals: `time_idx`, week sine/cosine, month sine/cosine.

External covariates (`features.csv`, `stores.csv`) ამ baseline-ში შეგნებულად არ არის დამატებული. ისინი უნდა დაემატოს შემდეგ TFT experiment-ში, baseline result-ის შემდეგ.
