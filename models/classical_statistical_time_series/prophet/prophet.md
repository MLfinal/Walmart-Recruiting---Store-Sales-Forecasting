# Prophet baseline

ფოლდერი:

```text
models/classical_statistical_time_series/prophet
```

ამ ნაწილში დავიწყე classical statistical time-series models. პირველი baseline არის Prophet, რადგან ის კლასიკური/სტატისტიკური forecasting model-ია, რომელიც trend-ს, seasonality-ს, holiday effect-ს და changepoints-ს ცალკე კომპონენტებად სწავლობს.

Prophet განსხვავდება DLinear/N-BEATS/TFT-ისგან იმით, რომ აქ ერთი global neural model არ გვაქვს. Prophet ჩვეულებრივ ერთ time series-ზე fit-დება, ამიტომ Walmart-ის data-ზე ლოგიკა ასეთია:

```text
ერთი Store + Dept = ერთი Prophet model
```

ამიტომ baseline-ში თავიდანვე არ გავუშვი ყველა `3331` Store-Dept pair. Colab-friendly baseline-ისთვის ავიღე top `300` highest-volume series.

## Notebook

ფაილი:

```text
baseline_prophet.ipynb
```

Notebook-ის მთავარი flow:

1. Colab dependencies
   - `prophet`
   - `wandb`
   - `pandas`
   - `numpy`
   - `matplotlib`

2. data loading
   - `train.csv`
   - `test.csv`
   - `features.csv`
   - `stores.csv`

3. chronological validation split
   - ბოლო `39` კვირა validation;
   - იგივე horizon აქვს Kaggle test set-ს;
   - fit period მთავრდება `2012-01-27`;
   - validation period არის `2012-02-03` → `2012-10-26`.

4. top-series selection
   - თითო Store-Dept pair-ის total sales ითვლება;
   - ვიღებთ top `300` series-ს;
   - ამით Prophet baseline სწრაფად ეშვება და შეგვიძლია ჯერ იდეა შევამოწმოთ.

5. seasonal naive reference
   - თითო validation კვირაზე ვიყენებთ იგივე Store-Dept გაყიდვას 52 კვირით ადრე;
   - ეს არის მთავარი reference, რადგან Walmart weekly sales strongly yearly-seasonal არის.

6. Prophet fit loop
   - თითო selected Store-Dept pair-ზე ცალკე Prophet model fit-დება;
   - target column გადადის Prophet format-ში:

```text
Date         → ds
Weekly_Sales → y
```

7. WMAE evaluation
   - metric არის Kaggle-style WMAE;
   - holiday row weight = `5`;
   - normal row weight = `1`.

8. W&B logging
   - metrics;
   - prediction table;
   - series fit/fallback table;
   - weekly errors;
   - diagnostic plot;
   - artifact with CSVs/config/metrics.

## Baseline configuration

```text
validation_weeks = 39
top_n_series = 300
min_history_points = 52
growth = linear
yearly_seasonality = True
weekly_seasonality = False
daily_seasonality = False
seasonality_mode = additive
changepoint_prior_scale = 0.05
seasonality_prior_scale = 10.0
holidays_prior_scale = 10.0
prediction_clip_min = 0.0
prediction_clip_max = 300000.0
run_final_refit = False
```

`run_final_refit = False` intentional არის. baseline-ის მიზანი ჯერ validation result-ის შემოწმებაა და არა Kaggle submission. Final refit ცალკე გავააქტიურებთ მხოლოდ მაშინ, თუ Prophet-ის baseline ან შემდეგი experiment საკმარისად ღირსეული იქნება.

## Data split და coverage

Notebook output:

```text
all_train_rows = 421570
selected_train_rows = 42900
top_n_series = 300
selected_series = 300
fit_start = 2010-02-05
fit_end = 2012-01-27
validation_start = 2012-02-03
validation_end = 2012-10-26
test_start = 2012-11-02
test_end = 2013-07-26
```

ამით baseline მხოლოდ top-300 high-volume Store-Dept series-ს აფასებს. ამიტომ მისი WMAE პირდაპირ all-series DLinear/TFT/XGBoost validation numbers-ს არ ედრება. სწორი შედარება ამავე top-300 subset-ზე seasonal naive-სთანაა.

## Seasonal naive reference

Top-300 subset-ზე 52-week seasonal naive:

```text
seasonal_naive_wmae = 6026.2907
```

ეს reference ძალიან მნიშვნელოვანია. Prophet-ს თუ yearly seasonality კარგად ეხმარება, მას ამ რიცხვზე დაბალი WMAE უნდა მიეღო. თუ ვერ იღებს, ნიშნავს რომ უბრალო “same week last year” lookup უფრო ძლიერია ამ subset-ზე.

## Prophet baseline result

W&B run:

```text
https://wandb.ai/kende23-n-a/Walmart-Recruiting---Store-Sales-Forecasting/runs/42duaxjv
```

Notebook output:

```text
validation/wmae = 6455.6646
validation/mae = 5940.0752
validation/seasonal_naive_wmae = 6026.2907
validation/improvement_vs_seasonal_naive_pct = -7.1250
fit/series_total = 300
fit/series_fit_ok = 300
fit/series_fallback = 0
fit/elapsed_minutes = 0.7338
```

ანუ Prophet baseline-მა ყველა 300 selected series წარმატებით დააფიტა:

```text
fit = 300
fallback = 0
```

მაგრამ WMAE seasonal naive-ზე უარესი გამოვიდა:

```text
Prophet WMAE        = 6455.66
Seasonal naive WMAE = 6026.29
difference          = +429.37
relative change     = -7.13%
```

რადგან WMAE-ში lower is better, ეს ნიშნავს:

```text
Prophet baseline did not beat seasonal naive.
```

## რატომ შეიძლება Prophet baseline იყოს უარესი

ჩემი აზრით, მთავარი მიზეზი ის არის, რომ Walmart Store-Dept series-ები არ არის ჩვეულებრივი smooth business time series. ბევრი department/store-ს აქვს:

- sudden promotions;
- holiday spikes;
- sparse/noisy behavior;
- yearly same-week effects;
- level shifts;
- markdown-related changes;
- department-specific seasonality.

Prophet ცდილობს trend + smooth yearly seasonality + holiday component ააწყოს. მაგრამ Walmart-ში ძალიან ძლიერი signal არის კონკრეტული კვირა ერთი წლით ადრე:

```text
sales(Store, Dept, same week last year)
```

Seasonal naive სწორედ ამას აკეთებს. Prophet კი ამ signal-ს smooth seasonality-ად აქცევს და შეიძლება ძალიან აგრესიულად გაასწოროს ისეთი spikes, რომლებიც actually მნიშვნელოვანია WMAE-სთვის.

სხვა მიზეზი: Prophet თითო series-ს დამოუკიდებლად fit-ავს. ის არ სწავლობს cross-series information-ს:

```text
Store 1 Dept 2-ის pattern
არ ეხმარება
Store 10 Dept 2-ს
```

ამ მხრივ tree-based models ან global neural models უკეთესად იყენებენ shared structure-ს.

## რას ვლოგავთ W&B-ზე

Prophet baseline run W&B-ზე ინახავს:

- config;
- split summary;
- `validation/wmae`;
- `validation/mae`;
- `validation/seasonal_naive_wmae`;
- `validation/improvement_vs_seasonal_naive_pct`;
- fit status counts;
- elapsed minutes;
- validation prediction table;
- series info table;
- weekly errors table;
- diagnostic plot;
- artifact:
  - `prophet_baseline_validation_predictions.csv`;
  - `prophet_baseline_series_info.csv`;
  - `prophet_baseline_metrics.json`;
  - `prophet_baseline_config.json`;
  - `prophet_baseline_validation_diagnostics.png`.

ეს საკმარისია baseline story-სთვის: ვხედავთ არა მხოლოდ metric-ს, არამედ რამდენი series fit-და, fallback ხომ არ მოხდა, და როგორ ნაწილდება შეცდომა validation weeks-ზე.

## Baseline conclusion

ამ ეტაპზე Prophet baseline accepted როგორც working baseline, მაგრამ rejected როგორც strong model:

```text
status = working baseline, not competitive
```

მნიშვნელოვანი დასკვნები:

- Prophet notebook მუშაობს Colab-ზე;
- W&B logging სწორად მუშაობს;
- 300 Prophet model fit სწრაფად დასრულდა (`~0.73` წუთი);
- fallback არ დაგვჭირდა;
- მაგრამ Prophet seasonal naive-ზე `7.13%`-ით უარესია;
- ამიტომ შემდეგი experiment უნდა იყოს focused improvement, არა უბრალოდ all-series run.

ჩემი მოკლე დასკვნა:

```text
Prophet baseline proves pipeline works,
but simple 52-week seasonal naive is stronger on top-300 validation.
```

თუ Prophet-ზე გავაგრძელებთ მუშაობას, შემდეგი ლოგიკური ნაბიჯი იქნება არა blind scaling all `3331` series-ზე, არამედ ერთ-ერთი controlled change:

- multiplicative seasonality;
- lower/higher changepoint prior;
- yearly seasonality Fourier order tuning;
- explicit holiday windows;
- seasonal naive + Prophet residual correction.

ამ baseline-ის მიხედვით ყველაზე საინტერესო მიმართულება იქნება residual Prophet, რადგან უკვე ვნახეთ რომ pure Prophet ვერ ჯობნის yearly lookup-ს.
