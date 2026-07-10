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

## Experiment v2 — seasonal-residual Prophet

v1-ის შემდეგ external covariates სრულად ამოვიღეთ. v2-ის მიზანი იყო ძლიერი 52-week seasonal-naive პროგნოზის შეცვლა კი არა, მხოლოდ მისი კორექცია. თითო series-ისთვის notebook ასე მუშაობდა:

```text
residual(t)   = sales(t) - sales(t - 52)
final_pred(t) = sales(t - 52) + Prophet(residual(t))
```

კოდში `fit_predict_prophet_for_series()` ჯერ fit პერიოდის პირველ 52 კვირას lag-ის შესაქმნელად ტოვებს. დარჩენილი კვირებიდან ქმნის residual target-ს და Prophet-ს ამ target-ზე fit-ავს. Validation-ზე base ნაწილი ისევ რეალური, 52 კვირით ადრე არსებული გაყიდვაა; Prophet-ს მხოლოდ residual correction ემატება. ამიტომ v2-ში არც external feature და არც backfill არ გამოყენებულა — ეს temporal-ად სუფთა experiment იყო.

ყველა 3,331 series, იგივე ბოლო 39 კვირა და იგივე WMAE გამოვიყენეთ:

```text
Prophet v2 residual WMAE       = 3808.4181
Prophet v2 residual MAE        = 3855.4009
baseline Prophet WMAE          = 1625.4781
seasonal naive WMAE            = 1604.2697
vs baseline                    = +2182.9400 WMAE  (+134.30%)
vs seasonal naive              = -137.3926%
fit_ok / sparse fallback       = 3259 / 72
fit error                      = 0
elapsed time                   = 17.3492 minutes
```

ანუ v2 v1-ზე ოდნავ ნაკლებად ცუდი იყო, მაგრამ baseline Prophet-სა და seasonal naive-სთან შედარებით ძალიან ცუდი შედეგი მიიღო. მიზეზი თვითონ residual-ის მასშტაბია. თითო series-ზე fit პერიოდში მხოლოდ დაახლოებით 52 residual observation დარჩა; Prophet ამ მოკლე რიგზე trend/seasonality-ს სწავლობდა და validation-ში ხშირად დიდ დადებით კორექციას აბრუნებდა.

Notebook-ის პირველივე `(Store=1, Dept=1)` მაგალითში residual prediction-ები დაახლოებით `+6,625`, `+6,547`, `+7,781`, `+8,877`, `+13,098` იყო. ეს მნიშვნელობები seasonal-naive forecast-ს დაემატა. როცა base forecast უკვე ახლოსაა რეალურ გაყიდვასთან, ასეთი დიდი დამატება სისტემურ over-prediction-ს ქმნის. WMAE-ში holiday კვირებს weight `5` აქვს, ამიტომ holiday-ზე მსგავსი შეცდომა განსაკუთრებით ძლიერად აზიანებს საბოლოო score-ს.

ეს run ვალიდურია და W&B-ზე სრულად შენახულია, მაგრამ მისი დასკვნა ნათელია:

```text
Prophet უნდა იყოს კონტროლირებული დამატება seasonal naive-სთან,
არა თავისუფალი residual კორექტორი მოკლე residual ისტორიაზე.
```

## Experiment v3 — 50/50 seasonal-naive + direct Prophet blend

v3-ში residual target აღარ გამოგვიყენებია. თითო series-ზე ისევ baseline-ის პირდაპირი Prophet დავაფიტეთ, პარალელურად კი 52-week seasonal-naive პროგნოზი დავტოვეთ. საბოლოო prediction fixed 50/50 საშუალოა:

```text
final_prediction(t) =
0.50 × seasonal_naive_52(t) +
0.50 × direct_Prophet(t)
```

კოდში `fit_predict_prophet_for_series()` ინახავს ორივე კომპონენტს:

```text
SeasonalNaive52       → შარშანდელი იგივე კვირის გაყიდვა
RawProphetPrediction  → baseline Prophet-ის პირდაპირი forecast
Prediction            → მათი 50/50 blend
```

ასე W&B-სა და CSV-ში ცალკე ჩანს final blend, raw Prophet და seasonal-naive reference. Sparse-history 72 series-ზე Prophet fallback თვითონ seasonal naive-ა, ამიტომ ამ series-ებზე blend-იც ზუსტად seasonal naive რჩება და ზედმეტ რისკს არ ქმნის.

სრული 3,331-series run:

```text
blend WMAE                     = 1402.2612
blend MAE                      = 1373.0802
raw Prophet WMAE               = 1625.4781
seasonal naive WMAE            = 1604.2697
improvement vs seasonal naive  = 12.5919%
improvement vs raw Prophet     = 13.7324%
fit_ok / fallback              = 3259 / 72
fit error                      = 0
elapsed time                   = 7.5925 minutes
```

ეს არის Prophet ოჯახის პირველი შედეგი, რომელმაც ორივე დამოუკიდებელ კომპონენტს აჯობა:

```text
1402.26 < 1604.27 < 1625.48
blend       naive       raw Prophet
```

თავდაპირველად blend-ისგან მხოლოდ baseline Prophet-ზე უსაფრთხო გაუმჯობესებას ველოდით: absolute error convex-ია და ორი prediction-ის საშუალოს WMAE არ აღემატება მათი WMAE-ების საშუალოს. რეალური შედეგი ბევრად უკეთესი გამოვიდა. ეს ნიშნავს, რომ seasonal naive და raw Prophet ერთნაირ შეცდომას არ უშვებენ:

- seasonal naive ზუსტად ინარჩუნებს კონკრეტული Store-Dept-ის წინა წლის კვირის დონეს, მაგრამ ახალი trend/holiday ცვლილება შეიძლება გამორჩეს;
- raw Prophet trend-სა და holiday structure-ს ამატებს, მაგრამ smooth forecast-ით ზოგჯერ spike-ს ან კონკრეტულ department-level დონეს აცდენს;
- 50/50 blend-ში ამ განსხვავებული შეცდომების ნაწილი ერთმანეთს აუქმებს.

მაგალითად პირველ series-ზე seasonal naive და raw Prophet სხვადასხვა მხარეს იხრებოდნენ, ხოლო საშუალო ხშირად რეალურ გაყიდვასთან ახლოს აღმოჩნდა. ამიტომ blend-ის მოგება არ მოდის მხოლოდ „რისკის შემცირებიდან“; ის მოდის ორი განსხვავებული signal-ის გაერთიანებიდან.

v3 W&B run-ში მთავარი score არის `validation/wmae` — ეს final blend-ის WMAE-ა. დამატებით ინახება `validation/raw_prophet_wmae` და `validation/seasonal_naive_wmae`, რათა ზუსტად ჩანდეს, რომ გაუმჯობესება ერთ-ერთი კომპონენტის სახელის შეცვლა კი არა, რეალური blend effect-ია.

## Experiment v4 — event-aware holiday windows

v3-ში ყველა holiday Prophet-ისთვის ერთი `walmart_holiday` კატეგორია იყო. v4-ში seasonal-naive/Prophet 50/50 blend უცვლელი დავტოვეთ და შევცვალეთ მხოლოდ calendar feature engineering. `IsHoliday`-ის flagged კვირები თარიღის month-ით ოთხ ცნობილ Walmart event-ად დავყავით:

```text
February  → Super Bowl
September → Labor Day
November  → Thanksgiving
December  → Christmas
```

ყველა event ერთნაირად არ მოქმედებს. Thanksgiving და Christmas-ის გაყიდვების მნიშვნელოვანი ნაწილი event კვირამდე მოდის, ამიტომ Prophet holiday calendar-ში მათ დაემატა `lower_window = -7` დღე. Weekly data Friday-ზეა, ამიტომ ეს ზუსტად წინა Friday/week-ს მოიცავს. Super Bowl და Labor Day მხოლოდ თავად event კვირაზე დარჩა.

```text
Super Bowl     = [0, 0]
Labor Day      = [0, 0]
Thanksgiving  = [-7, 0]
Christmas     = [-7, 0]
```

ეს feature engineering სრულად leakage-safe-ია: holiday date და event type forecast horizon-მდე ცნობილია; არც target და არც future external value არ გამოყენებულა.

v4 full all-series result:

```text
v4 blend WMAE                = 1367.4470
v4 blend MAE                 = 1353.8687
v4 raw Prophet WMAE          = 1534.8594
seasonal naive WMAE          = 1604.2697
v3 blend WMAE                = 1402.2612
improvement vs v3            = 34.8143 WMAE  (2.48%)
improvement vs seasonal naive= 14.7620%
improvement vs raw Prophet   = 10.9073%
fit_ok / fallback            = 3259 / 72
fit error                    = 0
elapsed time                 = 8.9578 minutes
```

მთავარი შედეგი ისაა, რომ calendar representation-მა არა მხოლოდ blend, არამედ raw Prophet-იც გააძლიერა:

```text
raw Prophet, generic holiday   = 1625.4781
raw Prophet, event windows     = 1534.8594
improvement                    = 90.6187 WMAE
```

v3-ში blend-ის სიძლიერე იყო yearly lookup და smooth Prophet forecast-ის შეცდომების კომპენსაცია. v4-ში Prophet-ს უკვე უკეთ ესმის, რომ Thanksgiving/Christmas effect event-day-only კი არა, წინასწარი demand window-იცაა. ამის გამო მისი component seasonal naive-სთან უფრო სასარგებლო complementary signal გახდა და final blend `1367.45`-მდე ჩამოვიდა.

ამ ეტაპზე v4 არის Prophet family-ის მიმდინარე საუკეთესო valid model. W&B-ში ინახება event calendar table, event-window configuration, raw Prophet/seasonal-naive/blend metrics, validation predictions, status table, plot და validation artifact.

## Experiment v5 — clean promotion/context covariates + stronger regularization

v5-ში v4-ის ორი ძლიერი ნაწილი უცვლელი დავტოვეთ:

```text
event-aware holiday windows
0.50 × seasonal naive + 0.50 × direct Prophet blend
```

შემდეგ ერთად დავამატეთ `features.csv`-იდან ხუთი forecast-time-ზე ცნობილი feature და გავამკაცრეთ Prophet-ის regularization:

```text
MarkdownLogTotal       = log1p(MarkDown1 + ... + MarkDown5)
HasMarkdown            = markdown availability indicator
TemperatureDeviation13 = temperature - trailing 13-week Store mean
FuelPriceChange4       = 4-week fuel-price change
UnemploymentChange4    = 4-week unemployment change
```

Markdown columns-ში v1-ის შეცდომა არ განმეორებულა. ადრეული missing Markdown პირდაპირ `0`/no-promotion მდგომარეობად ჩაითვალა; `bfill()` საერთოდ არ გამოყენებულა. Temperature/Fuel/Unemployment features Store-ის შიგნით მხოლოდ მიმდინარე ან წინა feature dates-იდან შეიქმნა. ამიტომ v5 temporal leakage-ის გარეშე შესრულებული valid experiment-ია.

Model settings-იც შეიცვალა:

```text
yearly Fourier order       = 8
changepoint_prior_scale    = 0.03
seasonality_prior_scale    = 5.0
holidays_prior_scale       = 15.0
regressor_prior_scale      = 0.15
```

იდეა იყო, რომ limited per-series history-ზე regressors მხოლოდ სუსტი correction ყოფილიყო და Prophet-ს ზედმეტი თავისუფლება არ ჰქონოდა.

v5 full all-series result:

```text
v5 blend WMAE                = 1415.5392
v5 blend MAE                 = 1389.4946
v5 raw Prophet WMAE          = 1654.6635
seasonal naive WMAE          = 1604.2697
v4 blend WMAE                = 1367.4470
difference vs v4             = +48.0922 WMAE  (3.52% worse)
improvement vs seasonal naive= 11.7643%
improvement vs raw Prophet   = 14.4515%
fit_ok / fallback            = 3259 / 72
fit error                    = 0
elapsed time                 = 10.0468 minutes
```

v5 ჯერ კიდევ seasonal naive-ზე უკეთესია, მაგრამ v4-ზე უარესია:

```text
v4 event-aware blend      = 1367.45   ← current Prophet champion
v5 covariate blend        = 1415.54
seasonal naive            = 1604.27
```

მთავარი დიაგნოზი არის raw Prophet component-ის გაუარესება:

```text
v4 raw Prophet = 1534.86
v5 raw Prophet = 1654.66
```

ეს ნიშნავს, რომ ამ მოკლე, ინდივიდუალური Store-Dept series-ებისთვის Markdown/context regressors და ახალი regularization ერთად საკმარისად სტაბილური დამატებითი signal არ აღმოჩნდა. Markdown/promotion ეფექტი სავარაუდოდ department-specific და არაწრფივია; ერთი მცირე Prophet regression თითო series-ზე მას საიმედოდ ვერ სწავლობს. Seasonal-naive blend-მა ზიანი შეამცირა, მაგრამ v4-ის უკეთესი calendar-only component ვერ შეინარჩუნა.

v5-ის დასკვნა არ არის, რომ `features.csv` უსარგებლოა საერთოდ. ის tree-based/TFT global model-ებში შეიძლება უკეთ გამოიყენებოდეს. დასკვნა კონკრეტულად Prophet-ზეა: per-series Prophet-ისთვის v4-ის event calendar უფრო ძლიერი და სტაბილური feature engineering აღმოჩნდა, ვიდრე ამ ხუთი external regressor-ის ერთდროული დამატება.

## Experiment v6 — historical blend-weight tuning

v4-ის fixed `alpha=0.50` blend ძალიან კარგი აღმოჩნდა, მაგრამ ეს weight ხელით იყო არჩეული. v6-ში ახალი feature აღარ დაგვიმატებია. დავბრუნდით v4-ის event-aware calendar configuration-ზე და მხოლოდ blend weight გავტესტეთ:

```text
final prediction =
(1 - alpha) × SeasonalNaive52 +
alpha × raw Prophet
```

იმისთვის, რომ ბოლო 39-week validation-ზე alpha არ აგვერჩია, data სამ ნაწილად დაიყო:

```text
1. early history            → Prophet calibration fit
2. next 26 weeks            → historical alpha calibration
3. final 39 weeks           → untouched final validation
```

Calibration-ში ყველა `alpha = 0.00, 0.05, ..., 1.00` გაიტესტა 3,331 series-ზე. შედეგები:

```text
best calibration alpha = 0.45
calibration WMAE       = 1653.7729

next candidates:
alpha 0.40 = 1654.9587
alpha 0.50 = 1656.1512
```

ანუ ძველ historical period-ზე Prophet-ს seasonal-naive correction-ისთვის დაახლოებით 45% weight ყველაზე სტაბილურად გამოუვიდა. შემდეგ ეს `0.45` უცვლელად გადავიტანეთ საბოლოო 39-week validation-ზე.

```text
v6 tuned-alpha WMAE          = 1373.0902
v4 fixed-0.50 WMAE           = 1367.4470
difference vs v4             = +5.6432 WMAE  (0.41% worse)
v6 raw Prophet WMAE          = 1534.8594
seasonal naive WMAE          = 1604.2697
best alpha                   = 0.45
fit_ok / fallback            = 3259 / 72
total elapsed time           = 35.3230 minutes
```

v6 მაინც seasonal naive-ზე `14.41%`-ით უკეთესია, მაგრამ v4-ის fixed 50/50 blend-ს ვერ აჯობა. ეს მნიშვნელოვანი და სანდო დასკვნაა: calibration fold-ზე ნაპოვნი alpha final validation-ზე თითქმის იგივე რეგიონში მუშაობს, მაგრამ ამ კონკრეტულ საბოლოო პერიოდზე `0.50` ოდნავ უკეთესია.

ამიტომ tuning დასრულდა ამ არჩევანით:

```text
Prophet final champion = v4
event-aware holiday windows
+ 0.50 × raw Prophet
+ 0.50 × SeasonalNaive52
validation WMAE = 1367.4470
```

v6 არ არის წარუმატებელი run. მან დაადასტურა, რომ blend-ის სასარგებლო weight დაახლოებით `0.45–0.50` რეგიონშია და რომ v4-ის შედეგი შემთხვევითი ან აშკარად overfit weight არ იყო. მაგრამ final model selection-ში lowest untouched validation WMAE პრიორიტეტია, ამიტომ v4 რჩება საუკეთესო არჩევანად.

## Final v4 pipeline, artifact და Model Registry

v4 champion-ის არჩევის შემდეგ `model_experiment_prophet.ipynb`-ის ბოლოს შესრულდა final full-history refit და pipeline packaging. ეს უკვე validation experiment არ არის: მიზანია Kaggle inference-ისთვის self-contained artifact-ის შექმნა.

```text
v4 event-aware Prophet configuration
+ full train history
+ fitted Prophet models per valid Store-Dept series
+ 52-week seasonal-naive fallback
+ blend alpha = 0.50
→ ProphetRawPipeline
```

Final refit ყველა `3331` series-ზე შესრულდა. სრული ისტორიის გამოყენების შემდეგ უფრო მეტ series-ს ჰქონდა საკმარისი non-zero history, ვიდრე validation fit-ში:

```text
fitted Prophet models = 3283
fallback series        = 48
stored history rows    = 421570
blend alpha            = 0.50
```

Pipeline-ში მხოლოდ checkpoint/config არ ინახება. `model_to_json()`-ით ინახება უკვე full-history-ზე fit-ებული Prophet model თითო valid series-ზე. ამიტომ inference დროს Prophet models ხელახლა არ train-დება. Pipeline იღებს raw `test.csv`-ს, Prophet forecast-ს აკეთებს მხოლოდ fit-ებული series-ებისთვის და sparse/cold-start cases-ში stored history-დან 52-week seasonal-naive fallback-ს იყენებს.

Contract test შესრულდა packaging-ის დროს:

```text
pipeline.predict(raw_test)
→ save pipeline
→ reload pipeline
→ reloaded_pipeline.predict(raw_test)
→ identical, finite, non-negative predictions
```

W&B pipeline registration run:

```text
run = prophet_v4_raw_pipeline_registration
run id = rlm39vch
```

Registry target:

```text
wandb-registry-model/Walmart_Prophet_Raw_Pipeline:champion
```

ახლა Prophet inference-ის სწორ flow-ს აქვს მხოლოდ ერთი პასუხისმგებლობა:

```python
pipeline = download_from_registry(...)
predictions = pipeline.predict(raw_test)
```

ის აღარ კითხულობს `train.csv`, აღარ refit-ავს Prophet-ს და აღარ იმეორებს event/fallback preprocessing-ს notebook-ში. Submission, manifest, prediction histogram და submission artifact უკვე inference run-ზე ლოგდება.

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

მან დაადასტურა, რომ per-series classical forecasting და W&B logging სწორად მუშაობს. Direct baseline Prophet-ის `1625.48` WMAE-მ აჩვენა, რომ trend/yearly seasonality/holiday component მარტო ვერ ჯობდა Walmart-ის ძლიერი კონკრეტული-კვირა-წინა-წლის signal-ს. v3 blend-მა ეს `1402.26`-მდე ჩამოიყვანა, v4 event-aware holiday engineering-მა `1367.45`-მდე გააუმჯობესა, v5 external covariates-მა გააუარესა, ხოლო v6 historical tuning-მა v4-ის alpha არჩევანი თითქმის დაადასტურა. Prophet family-ის champion რჩება v4.

External-covariate v1-მა კი დაადასტურა, რომ feature imputation დროით უნდა შემოწმდეს: მომავალიდან backward fill არ შეიძლება. ამიტომ v1-ის მაღალი WMAE model-performance conclusion არ არის; ის არის მონაცემის მომზადების შეცდომის დაფიქსირებული შედეგი.
