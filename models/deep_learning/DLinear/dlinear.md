# DLinear — baseline და პირველი დასკვნები

ეს ფოლდერი არის DLinear მოდელის სამუშაო სივრცე Walmart-ის weekly sales forecasting ამოცანაზე.

ამ ეტაპზე დასრულებულია მხოლოდ baseline. შემდეგი ექსპერიმენტები დაემატება იგივე დოკუმენტში, რომ საბოლოოდ გვქონდეს ერთი თანმიმდევრული ისტორია: საიდან დავიწყეთ, რა შევცვალეთ, რა შედეგი მივიღეთ და რატომ.

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

## ამ ეტაპის დასკვნა

DLinear baseline დასრულებულად ითვლება.

მიღებული შედეგი საკმარისია როგორც baseline:

- გვაქვს leakage-safe 39-week validation;
- გვაქვს WMAE, იგივე metric რაც Kaggle-ზე;
- გვაქვს seasonal naive benchmark;
- გვაქვს W&B run და artifacts;
- გვაქვს reproducible notebook Colab-ისთვის.

შემდეგი ფაილი უკვე არის experiment notebook, სადაც baseline-ს ეტაპობრივად გავაუმჯობესებთ და ყველა ახალი შედეგი დაემატება ამ README-ს.
