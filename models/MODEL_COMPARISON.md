# ყველა მოდელის არქიტექტურული, სასწავლო და შედეგობრივი ანალიზი

ეს ანგარიში აჯამებს პროექტში გამოცდილი ყველა არაჰიბრიდული არქიტექტურის მიზანს, მონაცემთა გარდაქმნას, training flow-ს, validation მეთოდს, inference-სა და შედეგებს.

ჰიბრიდული XGBoost–SARIMA მიმართულება განზრახ გამოტოვებულია, რადგან მისი დოკუმენტაცია ცალკე ფაილში ინახება.

მთავარი არჩევის metric არის Kaggle-ის Weighted Mean Absolute Error — WMAE. ნაკლები მნიშვნელობა უკეთეს მოდელს ნიშნავს.

## 1. ამოცანა და შეფასების წესი

ყოველი row აღწერს კონკრეტული Store–Dept წყვილის ერთკვირიან Weekly_Sales მნიშვნელობას. ამოცანაა test.csv-ის ყველა Store–Dept–Date row-ზე გაყიდვების პროგნოზი.

Holiday კვირები ბიზნესურად უფრო მნიშვნელოვანია და კონკურსის metric მათ შეცდომას ხუთჯერ მეტ წონას ანიჭებს.

```text
WMAE = Σ(weight_i × |actual_i - prediction_i|) / Σ(weight_i)
weight_i = 5, თუ IsHoliday=True
weight_i = 1, სხვა შემთხვევაში
```

- WMAE არის საბოლოო model-selection metric.
- MAE diagnostic metric-ია, მაგრამ holiday weighting არ აქვს.
- RMSE დიდ შეცდომებს უფრო მკაცრად სჯის, თუმცა Kaggle ranking-ს არ განსაზღვრავს.
- R² variance explanation-ს აჩვენებს, მაგრამ submission-ის არჩევის კრიტერიუმი არაა.
- MAPE ამ dataset-ზე zero/near-zero/negative sales-ის გამო არასტაბილურია.
- Validation score და Kaggle score ცალ-ცალკე უნდა ჩაიწეროს.
- Submission file-ის შექმნა Kaggle score-ის არსებობას არ ნიშნავს.
- Leaderboard score მხოლოდ მაშინ ითვლება დადასტურებულად, როცა repository-ში კონკრეტული score წერია.

## 2. ერთიანი training flow

### ნაბიჯი 1: მონაცემების ჩატვირთვა

train.csv, test.csv, features.csv და stores.csv იკითხება schema/date validation-ით.

### ნაბიჯი 2: იდენტიფიკატორების შემოწმება

Store–Dept–Date uniqueness და row count მოწმდება.

### ნაბიჯი 3: ქრონოლოგიური დაყოფა

ბოლო forecast horizon validation-ად გამოიყოფა; random split აკრძალულია.

### ნაბიჯი 4: Feature availability audit

რჩება მხოლოდ ის ინფორმაცია, რომელიც forecast origin-ზე რეალურად ცნობილია.

### ნაბიჯი 5: Preprocessing fit

imputation, encoders, scalers და mappings მხოლოდ training partition-ზე fit-დება.

### ნაბიჯი 6: Representation

family-ის მიხედვით იქმნება tabular matrix, aggregate series, local series, neural window ან pretrained context.

### ნაბიჯი 7: Training

holiday weighting გამოიყენება loss-ში ან sample weights-ში, როცა architecture ამის საშუალებას იძლევა.

### ნაბიჯი 8: Tuning

parameter search მხოლოდ chronological validation WMAE-ს უყურებს.

### ნაბიჯი 9: Diagnostics

holiday/non-holiday, weekly, Store და Dept slices ცალკე მოწმდება.

### ნაბიჯი 10: Final refit

არჩეული configuration მთელ labeled train-ზე თავიდან fit-დება.

### ნაბიჯი 11: Packaging

preprocessing, model, history და fallback ერთ portable artifact/pipeline-ში ერთიანდება.

### ნაბიჯი 12: Contract test

raw test input-ზე prediction count, order, finiteness და reproducibility მოწმდება.

### ნაბიჯი 13: Registry

მხოლოდ valid საუკეთესო model იღებს champion alias-ს.

### ნაბიჯი 14: Inference

inference notebook raw test.csv-ს კითხულობს და Registry pipeline.predict-ს იძახებს.

### ნაბიჯი 15: Submission

იქმნება Id,Weekly_Sales ფორმატი და მხოლოდ განზრახ activation-ისას იტვირთება Kaggle-ზე.

## 3. Leakage და სამართლიანი შედარება

- Random split მომავალ rows-ს წარსულში ურევს და time-series leakage-ს ქმნის.
- lag_1, lag_4 და lag_13 unsafe-ა, თუ multi-step validation-ში რეალური future target-იდან იქმნება.
- SalesLag52 safe-ა მაშინ, როცა lookup მხოლოდ observed history-ს იყენებს.
- Target encoding training rows-ზე shifted expanding წესით უნდა დაითვალოს.
- Validation/test transformer-მა მხოლოდ training-ზე ნასწავლი mappings უნდა გამოიყენოს.
- Full-data refit validation score-ს აღარ ქმნის; ის მხოლოდ final production candidate-ს ქმნის.
- განსხვავებული horizon/series subset-ის local scores პირდაპირ apples-to-apples არაა.
- Kaggle score ყველაზე მკაცრი cross-model evidence-ია, რადგან ყველა submission ერთ test set-ზე ფასდება.

## 4. მოდელების დეტალური ანალიზი

## 4.1. XGBoost

**ოჯახი:** Tree-based

### არქიტექტურა

Level-wise gradient-boosted decision trees, რომლებიც residual error-ს თანმიმდევრულად ამცირებს.

### მონაცემის წარმოდგენა

Global Store–Dept row matrix

### გამოყენებული signal

Store, Dept, calendar, cyclical week, holiday identity/proximity, store metadata, markdowns, external variables, safe SalesLag52 და shifted aggregates.

### ნაბიჯ-ნაბიჯ training flow

Training მიჰყვება ზემოთ აღწერილ ერთიან flow-ს: chronological split, family-specific representation, WMAE selection, full-data refit, portable artifact და raw-input inference contract.

### შედეგები

| ეტაპი | შედეგი |
|---|---|
| Baseline/local reference | 2902.2892 |
| საუკეთესო local result | 1612.1265 |
| Kaggle result | 2806 |
| საბოლოო სტატუსი | Overall champion |

### რატომ იმუშავა

nonlinear interactions, global pooling, sparse/missing feature robustness და train/inference parity.

### მთავარი შეზღუდვა

Score კვლავ validation–test distribution shift-ს განიცდის; categorical IDs ordinal numeric representation-ადაა მოცემული.

### ანალიტიკური დასკვნა

XGBoost-ის არჩევა ან უარყოფა ეფუძნება არა მხოლოდ local WMAE-ს, არამედ Kaggle evidence-ს, feature availability-სა და inference reproducibility-ს. საბოლოო სტატუსი: **Overall champion**.

## 4.2. LightGBM

**ოჯახი:** Tree-based

### არქიტექტურა

Leaf-wise histogram gradient boosting, რომელიც ყველაზე დიდი gain-ის leaf-ს პირველ რიგში აფართოებს.

### მონაცემის წარმოდგენა

Global Store–Dept tabular matrix

### გამოყენებული signal

XGBoost-ის მსგავს safe calendar/store/markdown/external set-სა და SalesLag52-ს იყენებს.

### ნაბიჯ-ნაბიჯ training flow

Training მიჰყვება ზემოთ აღწერილ ერთიან flow-ს: chronological split, family-specific representation, WMAE selection, full-data refit, portable artifact და raw-input inference contract.

### შედეგები

| ეტაპი | შედეგი |
|---|---|
| Baseline/local reference | 3184.2771 |
| საუკეთესო local result | 1575.1545 final; 1567.7045 best trial |
| Kaggle result | 2809 |
| საბოლოო სტატუსი | Tree-based runner-up |

### რატომ იმუშავა

სწრაფი training, ძლიერი tabular accuracy, XGBoost-თან თითქმის თანაბარი leaderboard transfer.

### მთავარი შეზღუდვა

Leaf-wise growth overfit-ს უფრო ადვილად ქმნის; unsafe early feature set-მა misleading validation გამოიწვია.

### ანალიტიკური დასკვნა

LightGBM-ის არჩევა ან უარყოფა ეფუძნება არა მხოლოდ local WMAE-ს, არამედ Kaggle evidence-ს, feature availability-სა და inference reproducibility-ს. საბოლოო სტატუსი: **Tree-based runner-up**.

## 4.3. ARIMA

**ოჯახი:** Classical statistical

### არქიტექტურა

Autoregressive, differencing და moving-average terms aggregate weekly total-ზე.

### მონაცემის წარმოდგენა

ერთი aggregate weekly total series

### გამოყენებული signal

ძირითადად მხოლოდ target history; row forecast allocation history shares-ით.

### ნაბიჯ-ნაბიჯ training flow

Training მიჰყვება ზემოთ აღწერილ ერთიან flow-ს: chronological split, family-specific representation, WMAE selection, full-data refit, portable artifact და raw-input inference contract.

### შედეგები

| ეტაპი | შედეგი |
|---|---|
| Baseline/local reference | 1856.8605 |
| საუკეთესო local result | 1829.8800 |
| Kaggle result | არ არის დაფიქსირებული |
| საბოლოო სტატუსი | Diagnostic classical baseline |

### რატომ იმუშავა

მარტივი, interpretable და დაბალი computational cost.

### მთავარი შეზღუდვა

Aggregate total-იდან ათასობით Store–Dept row-ზე allocation information-ს კარგავს.

### ანალიტიკური დასკვნა

ARIMA-ის არჩევა ან უარყოფა ეფუძნება არა მხოლოდ local WMAE-ს, არამედ Kaggle evidence-ს, feature availability-სა და inference reproducibility-ს. საბოლოო სტატუსი: **Diagnostic classical baseline**.

## 4.4. SARIMA

**ოჯახი:** Classical statistical

### არქიტექტურა

ARIMA-ს seasonal extension, რომელიც პერიოდულ differencing/AR/MA structure-ს ამატებს.

### მონაცემის წარმოდგენა

Aggregate weekly total series

### გამოყენებული signal

Target seasonality და historical row shares.

### ნაბიჯ-ნაბიჯ training flow

Training მიჰყვება ზემოთ აღწერილ ერთიან flow-ს: chronological split, family-specific representation, WMAE selection, full-data refit, portable artifact და raw-input inference contract.

### შედეგები

| ეტაპი | შედეგი |
|---|---|
| Baseline/local reference | 1856.8605 reference |
| საუკეთესო local result | 1831.6176 |
| Kaggle result | 3842 |
| საბოლოო სტატუსი | SARIMAX-ზე სუსტი Kaggle result |

### რატომ იმუშავა

Yearly repetition-ის explicit მოდელირება შეუძლია.

### მთავარი შეზღუდვა

ამ implementation-ში aggregate allocation bottleneck დარჩა; ზოგი seasonal setup unstable იყო.

### ანალიტიკური დასკვნა

SARIMA-ის არჩევა ან უარყოფა ეფუძნება არა მხოლოდ local WMAE-ს, არამედ Kaggle evidence-ს, feature availability-სა და inference reproducibility-ს. საბოლოო სტატუსი: **SARIMAX-ზე სუსტი Kaggle result**.

## 4.5. SARIMAX

**ოჯახი:** Classical statistical

### არქიტექტურა

SARIMA external regressors-ით, რათა aggregate total-ზე known covariates შევიდეს.

### მონაცემის წარმოდგენა

Aggregate weekly total + external weekly regressors

### გამოყენებული signal

Temperature, Fuel_Price, CPI, Unemployment, markdown/holiday-derived aggregate signals.

### ნაბიჯ-ნაბიჯ training flow

Training მიჰყვება ზემოთ აღწერილ ერთიან flow-ს: chronological split, family-specific representation, WMAE selection, full-data refit, portable artifact და raw-input inference contract.

### შედეგები

| ეტაპი | შედეგი |
|---|---|
| Baseline/local reference | SARIMA 1831.6176 |
| საუკეთესო local result | 2563.6915 |
| Kaggle result | 3525 |
| საბოლოო სტატუსი | Classical Kaggle champion among scored models |

### რატომ იმუშავა

External signals-მა Kaggle aggregate horizon-ზე SARIMA გააუმჯობესა.

### მთავარი შეზღუდვა

Local validation გაუარესდა და row-level heterogeneity კვლავ allocation-ზეა დამოკიდებული.

### ანალიტიკური დასკვნა

SARIMAX-ის არჩევა ან უარყოფა ეფუძნება არა მხოლოდ local WMAE-ს, არამედ Kaggle evidence-ს, feature availability-სა და inference reproducibility-ს. საბოლოო სტატუსი: **Classical Kaggle champion among scored models**.

## 4.6. Prophet

**ოჯახი:** Classical statistical

### არქიტექტურა

Additive trend + weekly/yearly seasonality + explicit holiday/event effects თითო series-ზე.

### მონაცემის წარმოდგენა

ერთი local model თითო Store–Dept series-ზე

### გამოყენებული signal

Date, sales history, Walmart holiday calendar; experimented external covariates.

### ნაბიჯ-ნაბიჯ training flow

Training მიჰყვება ზემოთ აღწერილ ერთიან flow-ს: chronological split, family-specific representation, WMAE selection, full-data refit, portable artifact და raw-input inference contract.

### შედეგები

| ეტაპი | შედეგი |
|---|---|
| Baseline/local reference | 1625.4781 |
| საუკეთესო local result | 1367.4470 v4 |
| Kaggle result | upload documented, score არა |
| საბოლოო სტატუსი | Classical local-validation champion |

### რატომ იმუშავა

Holiday/event structure interpretable-ად და ეფექტურად დაიჭირა.

### მთავარი შეზღუდვა

ათასობით local fit ძვირია; unseen/short series fallback-ს საჭიროებს; leaderboard evidence აკლია.

### ანალიტიკური დასკვნა

Prophet-ის არჩევა ან უარყოფა ეფუძნება არა მხოლოდ local WMAE-ს, არამედ Kaggle evidence-ს, feature availability-სა და inference reproducibility-ს. საბოლოო სტატუსი: **Classical local-validation champion**.

## 4.7. DLinear

**ოჯახი:** Deep learning

### არქიტექტურა

Moving-average decomposition trend/seasonal ნაწილებად და linear projection თითო ნაწილისთვის.

### მონაცემის წარმოდგენა

Store–Dept sales panel, 52-week context, 39-week horizon

### გამოყენებული signal

ძირითადად target history; საუკეთესო ვერსიაში per-series bias embedding.

### ნაბიჯ-ნაბიჯ training flow

Training მიჰყვება ზემოთ აღწერილ ერთიან flow-ს: chronological split, family-specific representation, WMAE selection, full-data refit, portable artifact და raw-input inference contract.

### შედეგები

| ეტაპი | შედეგი |
|---|---|
| Baseline/local reference | 1523.2097 |
| საუკეთესო local result | 1506.2825 manual v1 |
| Kaggle result | დაახლოებით 3500 |
| საბოლოო სტატუსი | Deep-learning local champion |

### რატომ იმუშავა

მარტივი neural architecture, stable training და საუკეთესო local DL score.

### მთავარი შეზღუდვა

Long Kaggle horizon-ზე regime/holiday shifts მხოლოდ history-დან ვერ ახსნა.

### ანალიტიკური დასკვნა

DLinear-ის არჩევა ან უარყოფა ეფუძნება არა მხოლოდ local WMAE-ს, არამედ Kaggle evidence-ს, feature availability-სა და inference reproducibility-ს. საბოლოო სტატუსი: **Deep-learning local champion**.

## 4.8. N-BEATS

**ოჯახი:** Deep learning

### არქიტექტურა

Fully connected residual blocks, backcast/forecast decomposition და learned basis expansion.

### მონაცემის წარმოდგენა

Normalized historical sales windows

### გამოყენებული signal

ძირითადად target history; external covariates თითქმის არ გამოიყენება.

### ნაბიჯ-ნაბიჯ training flow

Training მიჰყვება ზემოთ აღწერილ ერთიან flow-ს: chronological split, family-specific representation, WMAE selection, full-data refit, portable artifact და raw-input inference contract.

### შედეგები

| ეტაპი | შედეგი |
|---|---|
| Baseline/local reference | დაახლოებით 2231 best epoch in baseline logs |
| საუკეთესო local result | 2157.98 documented champion; 2224.1179 latest selected run |
| Kaggle result | დაახლოებით 4700 |
| საბოლოო სტატუსი | Rejected final candidate |

### რატომ იმუშავა

Flexible nonlinear basis და generic univariate forecasting.

### მთავარი შეზღუდვა

Early overfit, weak covariate awareness და ყველაზე ცუდი Kaggle transfer.

### ანალიტიკური დასკვნა

N-BEATS-ის არჩევა ან უარყოფა ეფუძნება არა მხოლოდ local WMAE-ს, არამედ Kaggle evidence-ს, feature availability-სა და inference reproducibility-ს. საბოლოო სტატუსი: **Rejected final candidate**.

## 4.9. TFT

**ოჯახი:** Deep learning

### არქიტექტურა

Variable selection networks + recurrent encoder + gated residual network + interpretable multi-head attention.

### მონაცემის წარმოდგენა

Top-series panel with static IDs, known future covariates and observed residual history

### გამოყენებული signal

Store/Dept categoricals, calendar, holiday, external covariates და SeasonalNaive52 residual target.

### ნაბიჯ-ნაბიჯ training flow

Training მიჰყვება ზემოთ აღწერილ ერთიან flow-ს: chronological split, family-specific representation, WMAE selection, full-data refit, portable artifact და raw-input inference contract.

### შედეგები

| ეტაპი | შედეგი |
|---|---|
| Baseline/local reference | 7801.8986 top-300 |
| საუკეთესო local result | 2379.5014 top-2000 blend |
| Kaggle result | public 2979.86060; private 3058.98280 |
| საბოლოო სტატუსი | Deep-learning Kaggle champion |

### რატომ იმუშავა

Feature-rich temporal reasoning და საუკეთესო deep-learning leaderboard score.

### მთავარი შეზღუდვა

Heavy training, subset-dependent validation და 32.87% fallback reliance.

### ანალიტიკური დასკვნა

TFT-ის არჩევა ან უარყოფა ეფუძნება არა მხოლოდ local WMAE-ს, არამედ Kaggle evidence-ს, feature availability-სა და inference reproducibility-ს. საბოლოო სტატუსი: **Deep-learning Kaggle champion**.

## 4.10. TimesFM

**ოჯახი:** Foundation model

### არქიტექტურა

Pretrained decoder-only time-series foundation model, zero-shot/few-shot forecasting representation-ით.

### მონაცემის წარმოდგენა

Global weekly calendar-ზე დალაგებული per-series context

### გამოყენებული signal

Raw history, seasonal-naive/residual variants და v3-ში XReg complementary forecast.

### ნაბიჯ-ნაბიჯ training flow

Training მიჰყვება ზემოთ აღწერილ ერთიან flow-ს: chronological split, family-specific representation, WMAE selection, full-data refit, portable artifact და raw-input inference contract.

### შედეგები

| ეტაპი | შედეგი |
|---|---|
| Baseline/local reference | 1672.2525 zero-shot |
| საუკეთესო local result | 1588.8029 v3 |
| Kaggle result | public `2742.68603`; private `2853.40612` |
| საბოლოო სტატუსი | Foundation champion; third private/final overall |

### რატომ იმუშავა

Task-specific training-ის გარეშე ძლიერი forecast და complementary representation.

### მთავარი შეზღუდვა

Standalone XReg/LoRA unstable იყო; private score tree champions-ს ჩამორჩება.

### ანალიტიკური დასკვნა

TimesFM-ის არჩევა ან უარყოფა ეფუძნება არა მხოლოდ local WMAE-ს, არამედ Kaggle evidence-ს, feature availability-სა და inference reproducibility-ს. საბოლოო სტატუსი: **Foundation champion და საუკეთესო non-tree private/final submission**.

## 5. Tree-based მოდელების ერთმანეთთან შედარება

| კრიტერიუმი | XGBoost | LightGBM |
|---|---|---|
| Growth strategy | level-wise | leaf-wise |
| Best local WMAE | 1612.1265 | 1575.1545 |
| Best trial | documented engineered run | 1567.7045 |
| Kaggle | 2806 | 2809 |
| Difference | winner | +3 worse |
| Safe yearly lag | დიახ | დიახ |
| Global pooling | დიახ | დიახ |
| Final status | champion | runner-up |

- LightGBM local validation-ზე უკეთესია, მაგრამ Kaggle-ზე XGBoost 3 point-ით იგებს.
- 3 point დაახლოებით 0.11%-ია, ამიტომ პრაქტიკული განსხვავება მცირეა.
- ორივე model-ის წარმატების ძირითადი მიზეზი იგივე safe feature contract-ია.
- Unsafe lag experiments local metric-ს აუმჯობესებდა, მაგრამ Kaggle transfer-ს ანადგურებდა.
- Final ranking-ში actual leaderboard evidence local metric-ზე მაღლა დგას.

## 6. Classical time-series მოდელების შედარება

| მოდელი | Local WMAE | Kaggle | ძირითადი limitation |
|---|---|---|---|
| Prophet v4 | 1367.4470 | არ არის | leaderboard evidence missing |
| ARIMA | 1829.8800 | არ არის | aggregate allocation |
| SARIMA | 1831.6176 | 3842 | aggregate allocation |
| SARIMAX | 2563.6915 | 3525 | local/test mismatch |

- Prophet local-validation champion-ია, რადგან events და local trends პირდაპირ მოდელირდება.
- ARIMA/SARIMA total forecast კარგავს Store–Dept-specific behavior-ს.
- last_year_share blended_share-ზე უკეთესი იყო, რადგან recent mean distribution-ს noise შეჰქონდა.
- SARIMAX local validation-ზე უარესია, მაგრამ Kaggle-ზე SARIMA-ს აჯობა.
- Classical Kaggle champion არის SARIMAX მხოლოდ იმ მოდელებს შორის, რომლებსაც score აქვთ.

## 7. Deep-learning მოდელების შედარება

| მოდელი | Representation | Local WMAE | Kaggle | სტატუსი |
|---|---|---|---|---|
| DLinear | decomposed linear panel | 1506.2825 | ≈3500 | local champion |
| N-BEATS | residual FC blocks | ≈2157.98 | ≈4700 | rejected |
| TFT | covariate-aware attention residual | 2379.5014 subset | 3058.98280 private | Kaggle champion |

- DLinear-ის local evaluator ყველაზე სრული/სუფთა იყო, მაგრამ leaderboard transfer გაუარესდა.
- N-BEATS nonlinear capacity-მა Walmart-ის covariate/holiday structure ვერ ჩაანაცვლა.
- TFT external/known covariates-ს იყენებს და Kaggle-ზე საუკეთესო deep-learning result მიიღო.
- TFT-ის 32.87% fallback ნიშნავს, რომ submission pure TFT არ არის.
- Different subset/horizon local WMAE პირდაპირი ranking-ისთვის საკმარისი არაა.

## 8. Foundation-model ანალიზი

| ვერსია | მიდგომა | Final local WMAE |
|---|---|---|
| v1 | raw zero-shot | 1672.2525 |
| v2 | seasonal/raw/residual blend | 1620.5430 |
| v3 | audited XReg blend | 1588.8029 |
| v4 | LoRA candidate weight 0 | 1588.8029 blend |

- Zero-shot result seasonal naive-ს აჯობა და pretrained representation-ის ღირებულება დაამტკიცა.
- Calibration blend raw forecast-ზე უკეთესი იყო.
- XReg standalone სუსტი იყო, მაგრამ 10% complementary contribution სასარგებლო აღმოჩნდა.
- Audit-მა clipping inconsistency გამოასწორა და v3 gain-ის წყარო დაადასტურა.
- LoRA overfit/scale failure იყო; safe selection-მა მას weight 0 მისცა.
- Kaggle public `2742.68603` tree models-ის recorded scores-ზე უკეთესია, private `2853.40612` კი XGBoost/LightGBM-ს ჩამორჩება.

## 9. Local WMAE და Kaggle score რატომ განსხვავდება

- Validation horizon შეიძლება Kaggle horizon-ს ზუსტად არ ემთხვეოდეს.
- Validation rows/series subset შეიძლება სრული test coverage-ისგან განსხვავდებოდეს.
- Feature availability validation-ში ხშირად უფრო ოპტიმისტურია.
- Full-data refit model state-ს ცვლის.
- Test period-ში holiday/markdown distribution განსხვავდება.
- Unseen Store–Dept pairs fallback-ს მოითხოვს.
- Recursive multi-step forecast error horizon-ის ზრდასთან გროვდება.
- Clipping/postprocessing validation-სა და inference-ში ერთნაირი უნდა იყოს.
- Aggregate allocation local total accuracy-ს row-level error-ად გარდაქმნის.
- Leaderboard score actual competition distribution-ზე ერთადერთი პირდაპირი evidence-ია.

## 10. დადასტურებული Kaggle ranking

| ადგილი | მოდელი | Kaggle WMAE | შეფასება |
|---|---|---|---|
| 1 | XGBoost | 2806 | overall champion |
| 2 | LightGBM | 2809 | 3 point behind |
| 3 | TimesFM v3 | 2853.40612 private; 2742.68603 public | best non-tree/private foundation |
| 4 | TFT | 3058.98280 private | best deep learning |
| 5 | DLinear | ≈3500 | local DL champion |
| 6 | SARIMAX | 3525 | best scored classical |
| 7 | SARIMA | 3842 | aggregate seasonal |
| 8 | N-BEATS | ≈4700 | weakest recorded |

## 11. მოდელები leaderboard score-ის გარეშე

| მოდელი | საუკეთესო local WMAE | რატომ არ შედის Kaggle ranking-ში |
|---|---|---|
| Prophet v4 | 1367.4470 | upload/registry documented, score არა |
| ARIMA | 1829.8800 | submission path exists, score არა |

## 12. რატომ არის XGBoost საუკეთესო

- ყველაზე დაბალი დადასტურებული Kaggle WMAE აქვს: 2806.
- LightGBM-ს 3 point-ით აჯობა.
- TFT private score-ს დაახლოებით 252.98 point-ით აჯობა.
- DLinear/SARIMAX-ს დაახლოებით 700-ზე მეტი point-ით აჯობა.
- Global model ათასობით Store–Dept series-ს შორის information sharing-ს აკეთებს.
- SalesLag52 Walmart-ის ძლიერ yearly seasonality-ს leakage-ის გარეშე გადასცემს.
- Holiday proximity და markdown interactions spike behavior-ს tree splits-ით იჭერს.
- Boosting nonlinear Store×Dept×calendar relationships-ს ეფექტურად სწავლობს.
- Missing external values trees-ისთვის მართვადია.
- Full-data refit test forecast-მდე ყველა observed row-ს იყენებს.
- Raw-input transformer train/inference feature parity-ს იცავს.
- Registry pipeline preprocessing state-სა და model-ს ერთად ინახავს.
- Contract tests row order-სა და prediction reproducibility-ს ამოწმებს.
- Kaggle result ადასტურებს, რომ local design რეალურ test distribution-ზე გადაიტანა.

## 13. საბოლოო გადაწყვეტილებები family-ის მიხედვით

| ოჯახი | არჩეული მოდელი | Evidence | გაფრთხილება |
|---|---|---|---|
| Tree-based | XGBoost | Kaggle 2806 | LightGBM მხოლოდ 3 point-ით უკანაა |
| Classical local | Prophet v4 | local 1367.4470 | Kaggle score აკლია |
| Classical Kaggle | SARIMAX | Kaggle 3525 | local validation სუსტია |
| Deep learning | TFT | private 3058.98280 | 32.87% fallback |
| Foundation | TimesFM v3 | local 1588.8029; private 2853.40612 | best non-tree private result |
| Overall | XGBoost | lowest recorded Kaggle WMAE | final champion |

## 14. Reproducibility checklist

- [ ] 1. ფიქსირებული random seed ჩაიწეროს.
- [ ] 2. Package versions artifact-ში იყოს.
- [ ] 3. Input CSV hashes შეინახოს.
- [ ] 4. Chronological boundary ზუსტად ჩაიწეროს.
- [ ] 5. Feature columns ordered manifest-ში იყოს.
- [ ] 6. Transformer fit partition დადასტურდეს.
- [ ] 7. Target-derived features leakage audit გაიაროს.
- [ ] 8. Holiday weights train და validation-ში ემთხვეოდეს.
- [ ] 9. Best parameters W&B config-ში იყოს.
- [ ] 10. Validation prediction CSV შეინახოს.
- [ ] 11. Weekly/holiday slice metrics დაილოგოს.
- [ ] 12. Final refit rounds/epochs ჩაიწეროს.
- [ ] 13. Raw pipeline fresh-process load გაიაროს.
- [ ] 14. Prediction count test row count-ს ემთხვეოდეს.
- [ ] 15. Prediction order raw Id order-ს ემთხვეოდეს.
- [ ] 16. NaN/Inf აკრძალული იყოს.
- [ ] 17. Clipping rule explicit იყოს.
- [ ] 18. Fallback coverage დაილოგოს.
- [ ] 19. Registry alias მხოლოდ valid model-ს მიეცეს.
- [ ] 20. Kaggle score submission filename-ს დაუკავშირდეს.

## 15. Metric interpretation checklist

| Metric | როლი | რას ზომავს | გამოყენება |
|---|---|---|---|
| WMAE | Primary | holiday-weighted absolute error | model selection |
| MAE | Diagnostic | average absolute error | normal scale |
| RMSE | Diagnostic | squared-error-sensitive scale | outlier failures |
| R² | Diagnostic | explained variance | fit context |
| MAPE | Warning only | relative percentage error | unstable near zero |
| Kaggle score | Primary external | competition WMAE | final ranking |

## 16. შემდეგი ექსპერიმენტების პრიორიტეტი

1. XGBoost/LightGBM prediction blending მხოლოდ identical Kaggle-safe pipeline-ებით.
2. Holiday-specific blend weights rolling-origin validation-ზე.
3. Prophet submission score-ის რეალურად დაფიქსირება.
4. Unified 39-week evaluator ყველა family-ზე.
5. ერთიანი Store–Dept coverage და fallback policy.
6. OOF prediction correlation analysis ensemble diversity-სთვის.
7. XGBoost residual model holiday outliers-ზე.
8. TFT fallback rate-ის შემცირება full-series encoder strategy-ით.
9. TimesFM v3 forecast-ის tree-based blend-ში დამატება.
10. Model Registry promotion gate actual WMAE + stability checks-ით.

## 17. საბოლოო დასკვნა

Local validation-ის მიხედვით სხვადასხვა ოჯახს სხვადასხვა winner ჰყავს, მაგრამ validation setup-ების განსხვავების გამო ეს რიცხვები პირდაპირი overall ranking-ისთვის არასაკმარისია.

Actual Kaggle submission-ებს შორის ყველაზე დაბალი score არის XGBoost-ის 2806. LightGBM 2809-ით პრაქტიკულად თანაბარია, მაგრამ ფორმალურად მეორე ადგილზეა.

TFT საუკეთესო deep-learning submission-ია; Prophet საუკეთესო classical local model-ია; SARIMAX საუკეთესო scored classical submission-ია; TimesFM v3 საუკეთესო foundation model და private/final ranking-ში მესამეა (`2853.40612`).

პროექტის საბოლოო champion არის XGBoost, რადგან მას აქვს საუკეთესო external evidence, leakage-safe feature contract და reproducible raw-input Registry pipeline.
