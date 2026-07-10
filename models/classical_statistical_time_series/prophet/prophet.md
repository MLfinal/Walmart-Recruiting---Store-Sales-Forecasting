# Prophet

ფოლდერი:

```text
models/classical_statistical_time_series/prophet
```

ამ ნაწილში დავიწყე classical statistical time-series models. პირველი მოდელია Prophet: ის თითოეული დროითი რიგის გაყიდვას trend-, seasonality-, holiday- და changepoint-კომპონენტებად აღწერს. Walmart-ის შემთხვევაში ერთი რიგი არის ერთი `(Store, Dept)` წყვილი, ამიტომ Prophet global model არ არის:

```text
ერთი Store + Dept = ერთი დამოუკიდებელი Prophet model
```

ეს განსხვავდება DLinear/N-BEATS/TFT-ისგან, სადაც ერთი neural model ბევრ series-ს ერთად სწავლობს, და tree-based მოდელებისგანაც, სადაც ერთი supervised model ყველა row-ზე მუშაობს.

## Notebook-ის flow

ფაილი: `baseline_prophet.ipynb`.

Notebook რიგრიგობით აკეთებს შემდეგს:

1. Colab-ში აყენებს `prophet`, `wandb` და საჭირო ბიბლიოთეკებს; შემდეგ კითხულობს `train.csv`, `test.csv`, `features.csv`, `stores.csv`.
2. თარიღით ყოფს მონაცემს: ბოლო `39` train კვირა არის validation (`2012-02-03` → `2012-10-26`), რადგან Kaggle test horizon-იც 39 კვირაა. Prophet მხოლოდ წინა პერიოდზე (`2010-02-05` → `2012-01-27`) fit-დება.
3. აგებს Store-Dept გაყიდვების panel-ს და პარალელურად ითვლის 52-week seasonal naive-ს: validation კვირის პროგნოზია იმავე series-ის გაყიდვა ზუსტად 52 კვირით ადრე.
4. ქმნის Walmart holiday calendar-ს `train.csv` და `test.csv`-ის `IsHoliday` სვეტიდან და გადასცემს მას Prophet-ს.
5. `fit_predict_prophet_for_series()` თითო series-ის `Date → ds`, `Weekly_Sales → y` ფორმატს აგებს, fit-ავს Prophet-ს და აბრუნებს validation prediction-ებს.
6. თუ series-ს ძალიან ცოტა არანულოვანი ისტორია აქვს ან fit მოულოდნელად ჩავარდება, ფუნქცია იმ series-ზე იყენებს 52-week seasonal-naive fallback-ს. ეს ნიშნავს, რომ სრული run არასოდეს წყდება ერთი პრობლემური department-ის გამო.
7. ბოლოს ითვლება WMAE/MAE, იქმნება diagnostics და ყველაფერი იგზავნება W&B-ზე artifact-თან ერთად. `run_final_refit=False` იყო, რადგან ეს ეტაპი მხოლოდ validation baseline-ია და არა Kaggle submission.

## Metric და baseline logic

Kaggle-ის metric არის weighted MAE:

```text
holiday row weight = 5
ordinary row weight = 1
```

ამიტომ ვიყენებთ WMAE-ს ყველა მოდელში. Seasonal naive განსაკუთრებით ძლიერი reference-ია Walmart-ისთვის, რადგან ხშირად ყველაზე მნიშვნელოვანი ინფორმაციაა:

```text
იგივე Store + იგივე Dept + იგივე კვირა შარშან
```

Prophet-ის მიზანი იყო ამ მარტივ lookup-ს trend, smooth yearly seasonality და holiday effect-ით გადაეჯობა.

## პირველი ტექნიკური run — top 300 series

საწყისად run გავუშვით top-300 highest-volume series-ზე, რათა სწრაფად დაგვემტკიცებინა notebook, per-series loop და W&B pipeline.

W&B run: [42duaxjv](https://wandb.ai/kende23-n-a/Walmart-Recruiting---Store-Sales-Forecasting/runs/42duaxjv)

```text
series_total                 = 300
fit_ok                       = 300
fallback                     = 0
elapsed_minutes              = 0.7338
Prophet WMAE                 = 6455.6646
seasonal naive WMAE          = 6026.2907
relative change vs naive     = -7.1250%
```

ანუ ტექნიკურად ყველა model fit-და, მაგრამ Prophet seasonal naive-ზე უარესი იყო. ეს შედეგი მხოლოდ smoke test იყო: top-300 subset-ის WMAE ვერ შედარდებოდა სხვა მოდელების all-series შედეგს.

## სრული baseline — 3,331 series

შემდეგ baseline გავუშვით ყველა ხელმისაწვდომ `(Store, Dept)` series-ზე. ამ run-ში შეიცვალა მხოლოდ coverage:

```text
top_n_series          = 3331
selected_train_rows   = 421570
selected_series       = 3331
```

იგივე validation split, WMAE, Prophet configuration და seasonal-naive reference დარჩა, ამიტომ ეს უკვე სწორი all-series comparison-ია.

```text
fit period             = 2010-02-05 → 2012-01-27
validation period      = 2012-02-03 → 2012-10-26
test period            = 2012-11-02 → 2013-07-26
seasonal naive WMAE    = 1604.2697
Prophet WMAE           = 1625.4781
Prophet MAE            = 1558.5649
difference vs naive    = +21.2084 WMAE
relative change        = -1.3220%
elapsed time           = 7.9325 minutes
```

Fit status:

```text
fit                            = 3259
fallback_insufficient_history  = 72
fallback_fit_error             = 0
```

72 fallback series შეცდომა არ არის. მათ არ ჰქონდათ baseline-ისთვის საკმარისი არანულოვანი გაყიდვების ისტორია; ამიტომ მათზე უსაფრთხოდ დარჩა seasonal-naive პროგნოზი. დანარჩენი 3,259 series Prophet-მა წარმატებით დააფიტა. სრული run-ის ~8 წუთიანი runtime ასევე აჩვენებს, რომ per-series Prophet Colab-ზე პრაქტიკულად გასაშვებია.

## შედეგის ინტერპრეტაცია

```text
lower WMAE is better
1625.48 > 1604.27
```

ამიტომ pure Prophet baseline seasonal naive-ს ვერ აჯობა, თუმცა განსხვავება სრული მონაცემის შემთხვევაში მხოლოდ `1.32%`-ია; top-300 run-ზე განსხვავება `7.13%` იყო. სრული coverage-მა უფრო სანდო სურათი მოგვცა: Prophet pipeline მუშაობს, მაგრამ მისი smooth trend/seasonality decomposition ჯერ ვერ იმეორებს ისეთ მკვეთრ, department-specific yearly pattern-ს, რომელსაც 52-week lookup პირდაპირ იღებს.

ამის მიზეზია Walmart-ის sales signal-ის ბუნება: promotion, holiday spike, rare departments და store-specific level shifts ხშირად არ არის smooth. Prophet თითო series-ს იზოლირებულად სწავლობს, ამიტომ ერთი department-ის ან store-ის pattern სხვა series-ს არ ეხმარება. Seasonal naive კი პირდაპირ იყენებს წინა წლის შესაბამის observation-ს.

## Experiment v1 — external covariates

პირველ experiment-ში baseline Prophet-ისთვის დავამატეთ `features.csv`-ის ცხრა future-known covariate:

```text
Temperature, Fuel_Price, CPI, Unemployment,
MarkDown1, MarkDown2, MarkDown3, MarkDown4, MarkDown5
```

მათი join კეთდებოდა `(Store, Date)`-ით და Prophet-ში `add_regressor()`-ით გადადიოდა. იდეა იყო, რომ მხოლოდ calendar/holiday-ს ნაცვლად მოდელს store-level ეკონომიკური და promotion ინფორმაცია ჰქონოდა. Run ისევ ყველა 3,331 series-ზე და იმავე 39-week split-ზე შესრულდა.

```text
Prophet v1 WMAE             = 4236.6848
Prophet v1 MAE              = 4005.8699
seasonal naive WMAE         = 1604.2697
relative change vs naive    = -164.0881%
fit_ok / fallback           = 3259 / 72
elapsed time                = 9.5987 minutes
```

ეს შედეგი baseline-ზე ბევრად უარესია, მაგრამ v1-ის მთავარი დასკვნა უფრო მნიშვნელოვანი ტექნიკური აღმოჩენაა. Notebook output-ში 2010 წლის რიგებისთვის `MarkDown1`–`MarkDown5` უკვე არანულოვანი მნიშვნელობებით ჩანდა, მიუხედავად იმისა, რომ Markdown data რეალურად მოგვიანებით გახდა ხელმისაწვდომი. მიზეზი იყო იმპუტაციის `ffill().bfill()` ნაბიჯი: `bfill()` თითო Store-ში პირველი შემდგომი Markdown მნიშვნელობას ადრინდელ თარიღებზეც ავრცელებდა.

ამგვარად v1-ის covariate history ხელოვნურად იყო შევსებული მომავლის feature მნიშვნელობებით. target `Weekly_Sales` არ გაჟონა, თუმცა ასეთი backfill მაინც არღვევს honest temporal feature preparation-ს და ამ run-ს სამართლიან model comparison-ად არ ვითვლით. `4236.68` WMAE შენახულია W&B-ზე როგორც აღმოჩენილი, წარუმატებელი ექსპერიმენტი; იგი არ გამოიყენება Prophet-ის ხარისხის საბოლოო შეფასებად.

ამ run-მა ასევე აჩვენა, რომ all-series external-regressor loop ტექნიკურად მუშაობდა: fit error არ ყოფილა, 72 fallback ისევ მხოლოდ sparse-history series-ებზე მოვიდა, ხოლო runtime დაახლოებით 9.6 წუთი იყო.

## W&B-ზე შენახული ინფორმაცია

ყოველი run W&B-ზე ინახავს:

- სრულ configuration-სა და split summary-ს;
- `validation/wmae`, `validation/mae`, seasonal-naive WMAE-ს და პროცენტულ განსხვავებას;
- თითო series-ის fit/fallback status-ს და elapsed time-ს;
- validation prediction table-ს, series-info table-ს და weekly error table-ს;
- actual-vs-prediction და weekly-MAE diagnostic plot-ს;
- artifact-ს შემდეგი ფაილებით:
  - validation predictions CSV;
  - series status CSV;
  - metrics JSON;
  - config JSON;
  - diagnostic PNG.

ამიტომ W&B run-დან ჩანს არა მხოლოდ საბოლოო score, არამედ რამდენი model fit-და, fallback რატომ გამოიყენეს და რომელ validation კვირებზე გაიზარდა შეცდომა.

## დასკვნა

Prophet baseline დასრულებულია როგორც სრული, reproducible all-series benchmark:

```text
status = working, reproducible baseline; not stronger than seasonal naive
```

მან დაადასტურა, რომ per-series classical forecasting და W&B logging სწორად მუშაობს. ამავე დროს, `1625.48` WMAE-მ აჩვენა, რომ baseline Prophet-ს მარტო trend/yearly seasonality/holiday component-ებით ჯერ არ შეუძლია Walmart-ის ძლიერი კონკრეტული-კვირა-წინა-წლის signal-ის გადაჭარბება.

External-covariate v1-მა კი დაადასტურა, რომ feature imputation დროით უნდა შემოწმდეს: მომავალიდან backward fill არ შეიძლება. ამიტომ v1-ის მაღალი WMAE model-performance conclusion არ არის; ის არის მონაცემის მომზადების შეცდომის დაფიქსირებული შედეგი.
