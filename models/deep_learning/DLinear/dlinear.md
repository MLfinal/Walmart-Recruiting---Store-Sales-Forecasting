# DLinear — baseline და პირველი დასკვნები

ეს ფოლდერი არის DLinear მოდელის სამუშაო სივრცე Walmart-ის weekly sales forecasting ამოცანაზე.

ამ ეტაპზე დასრულებულია baseline და პირველი მცირე ექსპერიმენტი. შემდეგი ექსპერიმენტები დაემატება იგივე დოკუმენტში, რომ საბოლოოდ გვქონდეს ერთი თანმიმდევრული ისტორია: საიდან დავიწყეთ, რა შევცვალეთ, რა შედეგი მივიღეთ და რატომ.

## რატომ DLinear

DLinear ავირჩიეთ როგორც deep learning არქიტექტურის მარტივი baseline. მისი მთავარი იდეაა დროითი სერიის ორ ნაწილად გაყოფა:

- trend — გრძელვადიანი მიმართულება;
- seasonal/residual — ის ნაწილი, რაც trend-ის შემდეგ რჩება.

შემდეგ ორივე ნაწილი მარტივი linear projection-ით პროგნოზირდება მომავალ ჰორიზონტზე. ეს მიდგომა გაცილებით მარტივია, ვიდრე TFT, N-BEATS ან PatchTST, ამიტომ კარგი საწყისი წერტილია: თუ ასეთი მარტივი neural მოდელი უკვე მუშაობს, შემდეგ ექსპერიმენტებში შეგვიძლია უფრო თავდაჯერებულად დავამატოთ embeddings, covariates, tuning და სხვა არქიტექტურული ცვლილებები.

## baseline notebook

ფაილი:

`baseline_dlinear.ipynb`

baseline-ის მთავარი ლოგიკა:

- ვალიდაციად ავიღეთ ბოლო 39 კვირა;
- 39 კვირა ემთხვევა Kaggle test horizon-ს, ამიტომ მოდელი სწავლობს პირდაპირ 39-კვირიან პროგნოზს;
- ყველა Store-Dept წყვილი გადავიყვანეთ ერთიან weekly panel-ში;
- missing weekly sales baseline-ში შევავსეთ 0-ით;
- input window არის ბოლო 52 კვირა;
- მოდელი არის global univariate DLinear — ყველა Store-Dept სერიაზე ერთი საერთო მოდელი;
- შეფასების მეტრიკა არის WMAE, რადგან Kaggle-იც ამით აფასებს;
- holiday row-ებს მივანიჭეთ წონა 5, non-holiday row-ებს წონა 1;
- ყველაფერი დალოგილია W&B-ზე, MLflow არ გამოგვიყენებია.

baseline-ის მიზანი არ იყო მაქსიმალური score. მიზანი იყო სუფთა საწყისი წერტილი, რომელიც გვაძლევს პასუხს კითხვაზე: “რამდენად შორს მიდის ძალიან მარტივი DLinear historical target-ის გამოყენებით?”

## W&B run

baseline run:

https://wandb.ai/kende23-n-a/Walmart-Recruiting---Store-Sales-Forecasting/runs/eep8a1y5

W&B-ზე დალოგილია:

- training configuration;
- epoch-level train loss;
- validation WMAE;
- seasonal naive benchmark;
- improvement vs seasonal naive;
- validation prediction table;
- actual-vs-prediction scatter plot;
- model checkpoint artifact;
- summary JSON.

ეს მნიშვნელოვანია იმიტომ, რომ შემდეგ ექსპერიმენტებში ყველა ცვლილებას შევადარებთ არა მხოლოდ final WMAE-ით, არამედ train/validation behavior-ითაც.

## baseline შედეგი

39-კვირიან validation split-ზე მივიღეთ:

| მოდელი | Validation WMAE | შენიშვნა |
|---|---:|---|
| Seasonal naive | 1604.27 | იგივე Store-Dept გაყიდვა 52 კვირით ადრე |
| DLinear baseline | 1523.21 | best epoch = 8 |

DLinear baseline-მა seasonal naive-ს მოუგო დაახლოებით 5.05%-ით.

ეს ნიშნავს, რომ მოდელმა რაღაც სასარგებლო pattern ნამდვილად ისწავლა, მაგრამ improvement ჯერ ზომიერია. ეს ნორმალურია baseline-ისთვის, რადგან მოდელი არ იყენებს:

- store/dept embeddings-ს;
- markdown/promotion features-ს;
- economic covariates-ს;
- calendar features-ს;
- hyperparameter tuning-ს;
- model ensembling-ს.

## baseline notebook-ის კოდის სტრუქტურა

`baseline_dlinear.ipynb` აგებულია ასეთი თანმიმდევრობით:

1. setup/import/config

   პირველ უჯრებში ხდება Colab-ისთვის საჭირო package-ების დაყენება, random seed-ის დაფიქსირება, GPU/CPU device-ის არჩევა და ძირითადი configuration-ის აღწერა. აქვეა მითითებული `validation_weeks=39`, `input_weeks=52`, W&B project/entity და Drive-ის data path.

2. data loading და weekly panel

   `train.csv` და `test.csv` იტვირთება Drive-იდან. შემდეგ `train.csv` გარდაიქმნება Store-Dept weekly panel-ად, სადაც row არის კონკრეტული `(Store, Dept)` სერია, column კი კონკრეტული კვირა. DLinear-ს fixed-length sequence სჭირდება, ამიტომ ეს panel არის მთავარი input ფორმატი.

3. metric

   `wmae()` ფუნქცია ითვლის Kaggle-ის weighted MAE-ს. holiday კვირებს ენიჭება weight 5, სხვა კვირებს weight 1. იგივე weighting გამოიყენება training loss-ისთვისაც, რომ მოდელი იმავე objective-ს მიუახლოვდეს, რითაც საბოლოოდ ფასდება.

4. dataset classes

   `WindowDataset` ქმნის training examples-ს sliding window პრინციპით. თითოეულ example-ში გვაქვს:

   - `x`: წარსული 52 კვირა;
   - `y`: შემდეგი 39 კვირა;
   - `weights`: holiday-aware WMAE weights;
   - `mean/std`: თითოეული window-ის normalization-ისთვის.

   `ValidationDataset` ქმნის მხოლოდ ერთ validation example-ს თითო Store-Dept სერიისთვის: ბოლო 52 კვირა validation-მდე და target-ად ბოლო 39 კვირა.

5. model classes

   `MovingAverage` აგებს trend component-ს rolling average-ით.

   `SeriesDecomposition` ყოფს sequence-ს ორ ნაწილად: seasonal/residual და trend.

   `DLinear` ამ ორ ნაწილს ცალ-ცალკე linear layer-ით გადააგზავნის 39-კვირიან forecast horizon-ზე და ბოლოს აერთიანებს.

6. training/evaluation

   `weighted_mae_loss()` გამოიყენება training loss-ად normalized scale-ზე. `evaluate_model()` აბრუნებს პროგნოზს original sales scale-ზე და ითვლის validation WMAE-ს. training loop-ში არის early stopping და learning-rate scheduler.

7. W&B logging/artifacts

   W&B-ზე ინახება epoch-level metrics, validation prediction table, scatter plot, checkpoint და summary JSON. ეს გვაძლევს reproducible comparison-ს შემდეგ ექსპერიმენტებთან.

## განსხვავება XGBoost-თან

XGBoost-ის შემთხვევაში ძირითადი ძალა მოდიოდა feature engineering-იდან: calendar features, store/dept metadata, lag/aggregate features და სხვა tabular signal-ები. XGBoost პირდაპირ სწავლობს row-level დამოკიდებულებებს feature-ებზე.

DLinear baseline განსხვავებული ტიპის მოდელია:

- ის უყურებს Store-Dept-ის წარსულ sales sequence-ს;
- არ იცის Store type, size, markdowns, CPI, fuel price და სხვა external feature-ები;
- პროგნოზს აკეთებს direct multi-step ფორმით, ანუ ერთდროულად აბრუნებს 39 კვირას;
- მისი “feature engineering” ძირითადად sequence construction და scaling-ია.

ამიტომ DLinear-ის და XGBoost-ის რიცხვები პირდაპირ მხოლოდ მაშინ უნდა შევადაროთ, როცა split ერთნაირია. ჩვენი DLinear baseline არის 39-week validation-ზე, XGBoost-ის რამდენიმე შედეგი კი სხვა validation horizon-ზე იყო. მთავარი დასკვნა ამ ეტაპზე არის არა “რომელი ჯობია აბსოლუტურად”, არამედ:

- XGBoost უკეთ იყენებს tabular/covariate ინფორმაციას;
- DLinear უკვე მხოლოდ historical sales-ითაც აუმჯობესებს seasonal naive-ს;
- DLinear-ის შემდეგი ლოგიკური განვითარება არის temporal model-ში identity/calendar/covariate signal-ის დამატება.

## რას ველოდებით შემდეგი ექსპერიმენტებიდან

baseline-მა აჩვენა, რომ direct 39-week DLinear approach მუშაობს, მაგრამ ჯერ ძალიან შეზღუდულია. შემდეგ ეტაპზე ვცდით მცირე, კონტროლირებულ ცვლილებებს:

- უფრო გრძელი context window, მაგალითად 104 კვირა;
- Store-Dept identity calibration;
- უკეთესი normalization;
- learning rate / regularization tuning;
- შემდეგ ეტაპზე calendar და external covariates.

ამ ცვლილებების მიზანი იქნება გავიგოთ, DLinear-ს აკლია მხოლოდ capacity/context თუ საჭიროა უკვე feature-rich temporal architecture.

## experiment v1 — 104 კვირა + Store-Dept calibration

ფაილი:

`model_experiment_DLinear.ipynb`

პირველი ექსპერიმენტის იდეა იყო baseline-ისგან მხოლოდ მცირე, კონტროლირებული ცვლილება:

- input window გავზარდეთ 52 კვირიდან 104 კვირამდე;
- DLinear core იგივე დავტოვეთ;
- დავამატეთ Store-Dept-level calibration layer;
- validation split და WMAE იგივე დარჩა.

Store-Dept calibration ნიშნავს, რომ მოდელს აქვს პატარა learnable correction თითოეული სერიისთვის და თითოეული forecast horizon-ისთვის. ეს არ არის სრული store/dept embedding architecture; უფრო მარტივი bias adjustment-ია. ამით ვამოწმებთ, ეხმარება თუ არა მოდელს თითოეული სერიის ინდივიდუალური საშუალო/სისტემური გადახრის დაჭერა.

### experiment v1 შედეგი

| მოდელი | Validation WMAE | Improvement vs seasonal naive | Improvement vs DLinear baseline |
|---|---:|---:|---:|
| Seasonal naive | 1604.27 | — | — |
| DLinear baseline, 52w | 1523.21 | 5.05% | — |
| DLinear v1, 104w + calibration | 1506.28 | 6.11% | 1.11% |

best epoch იყო `11`, early stopping მოხდა `23` epoch-ზე.

### რას გვასწავლის v1

v1-მა baseline გააუმჯობესა, მაგრამ გაუმჯობესება მცირეა. ეს სასარგებლო სიგნალია:

- მარტო უფრო გრძელი context და series-level correction საკმარისი არ არის დიდი ნახტომისთვის;
- DLinear historical sales pattern-ს უკეთ იყენებს, მაგრამ მას ჯერ არ აქვს future calendar/promotion/economic ინფორმაცია;
- training loss ნელ-ნელა უმჯობესდებოდა, მაგრამ validation WMAE საუკეთესო იყო შედარებით ადრე, რაც მიუთითებს რომ მოდელი ადვილად იწყებს validation horizon-ზე ზედმეტად მორგებას.

ამიტომ შემდეგი ლოგიკური ნაბიჯი არ არის ძალიან დიდი არქიტექტურის დამატება. ჯერ უნდა დავამატოთ მხოლოდ ის feature-ები, რომლებიც future-ში წინასწარ ცნობილია და leakage-ს არ ქმნის.

## experiment v2 — known future calendar signal

შემდეგი ვერსია, რომელსაც ახლა გავუშვებთ, იგივე `model_experiment_DLinear.ipynb`-შია განახლებული.

v2 ამატებს მხოლოდ მარტივ calendar covariates-ს:

- `IsHoliday`;
- week-of-year sin/cos;
- month sin/cos;
- forecast horizon-ის normalized position.

ამ feature-ების გამოყენება უსაფრთხოა, რადგან test period-შიც წინასწარ ვიცით თარიღი და holiday flag. ეს არ არის full feature engineering, რადგან ჯერ არ ვიყენებთ markdowns, CPI, unemployment, fuel price, store size/type და სხვა tabular signal-ებს.

v2-ის მიზანია ვუპასუხოთ ერთ კონკრეტულ კითხვას:

თუ DLinear-ს მივცემთ future calendar context-ს, შეამცირებს თუ არა შეცდომას holiday და seasonal კვირებზე?

თუ v2 მნიშვნელოვნად გაუმჯობესდება, შემდეგ ღირს უფრო მდიდარი covariates-ის დამატება. თუ გაუმჯობესება მცირე იქნება, მაშინ DLinear-ისთვის მთავარი შეზღუდვა შეიძლება იყოს არქიტექტურა ან ის, რომ Walmart-ის ამოცანაზე tabular tree-based models უკეთ იყენებენ ხელმისაწვდომ signal-ს.

## ამ ეტაპის დასკვნა

DLinear baseline დასრულებულად ითვლება.

მიღებული შედეგი საკმარისია როგორც baseline:

- გვაქვს leakage-safe 39-week validation;
- გვაქვს WMAE, იგივე metric რაც Kaggle-ზე;
- გვაქვს seasonal naive benchmark;
- გვაქვს W&B run და artifacts;
- გვაქვს reproducible notebook Colab-ისთვის.

შემდეგი ფაილი არის experiment notebook, სადაც baseline-ს ეტაპობრივად ვაუმჯობესებთ და ყველა ახალი შედეგი დაემატება ამ README-ს.
