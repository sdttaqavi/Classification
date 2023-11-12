
**بخش اول: پیش پردازش داده ها**

در گام اول، ما عملیات پیش پردازش داده‌ها را بر روی مجموعه داده‌مان اعمال می‌کنیم. برای این کار، کتابخانه های مربوطه را فراخوانی می کنیم:
"""

import numpy as np
import pandas as pd
from sklearn import preprocessing

"""سپس مجموعه داده‌مان را فراخوانی می‌کنیم و مراحل بعدی را برای هر دو مجموعه داده مان انجام می دهیم"""

!gdown 1VymOuQtYUfXBV4cdUrJusrx2kVA6cVGU
Adult_TrainDataset = pd.read_csv('Adult_TrainDataset.csv')
Adult_TrainDataset

!gdown 1X_ndzy4zZC_kCugo5cOobDFL_jY2WIy9
Adult_TestDataset = pd.read_csv('Adult_TestDataset.csv')
Adult_TestDataset

"""---

به علت اینکه تاثیر زیادی در نتیجه گیری ما ندارند، حذف می کنیم Capital_Loss و Capital_Gain دو ستون

جایگذاری می کنیم تا بتوانیم در مرحله بعد تعداد مقادیر از دست رفته را شناسایی کنیم  NaN  و مقادیر ؟ با مقادیر
"""

Adult_TrainDataset.drop('Capital_Gain', axis=1, inplace=True)
Adult_TrainDataset.drop('Capital_Loss', axis=1, inplace=True)
Adult_TrainDataset.replace({'?':np.NaN},  inplace=True)
Adult_TrainDataset

Adult_TestDataset.drop('Capital_Gain', axis=1, inplace=True)
Adult_TestDataset.drop('Capital_Loss', axis=1, inplace=True)
Adult_TestDataset.replace({'?':np.NaN},  inplace=True)
Adult_TestDataset

"""---

از متد زیر استفاده می‌کنیم، تا اطلاعات مجموعه داده جدیدمان رابه دست بیاوریم این اطلاعات نشان می دهد که مجموعه داده اموزش ما ۳۲۵۶۱ و مجموعه داده تست ما ۱۶۲۸۱ نمونه دارد

 در هر دو مجموعه داده داراي مقادير گمشده هستند Occupation, Native_Country, Work_Class همچنین نشان می دهد که ستون های
"""

Adult_TrainDataset.info()

Adult_TestDataset.info()

"""---

استفاده می‌کنیم، تا مقادیر اماری مانند میانگین، انحراف معیار، مینیمم، صدک اول، میانه، صدک سوم و ماکسیمم هر ویژگی با مقادیر عددی را به دست اوریم   describe()  از متد
"""

Adult_TrainDataset.describe()

Adult_TestDataset.describe()

"""---

برای اینکه بدانیم مجموعا چند داده گمشده در هر ویژگی داریم، از این متد استفاده می کنیم:
"""

Adult_TrainDataset.isnull().sum()

Adult_TestDataset.isnull().sum()

"""
از ان جایی که ویژگی هایی که مقادیر گمشده دارند از نوع شی هستند، بنابراین با استفاده از کلاس استفاده شده زیر، مقادیر ازدست رفته را با مقادیری که بیشترین تکرار در هر ویژگی دارند را جایگزین می کنیم. بعد از این مرحله مقادیر گمشده ی دیگری نداریم"""

from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy='most_frequent')
Adult_TrainDataset['Native_Country'] = imputer.fit_transform(Adult_TrainDataset[['Native_Country']])
Adult_TrainDataset['Work_Class'] = imputer.fit_transform(Adult_TrainDataset[['Work_Class']])
Adult_TrainDataset['Occupation'] = imputer.fit_transform(Adult_TrainDataset[['Occupation']])

Adult_TrainDataset.isnull().sum()

from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy='most_frequent')
Adult_TestDataset['Native_Country'] = imputer.fit_transform(Adult_TestDataset[['Native_Country']])
Adult_TestDataset['Work_Class'] = imputer.fit_transform(Adult_TestDataset[['Work_Class']])
Adult_TestDataset['Occupation'] = imputer.fit_transform(Adult_TestDataset[['Occupation']])

Adult_TestDataset.isnull().sum()

"""---

تبدیل مقادیر غیر عددی به مقادیر عددی

1. One Hot Encoding
"""

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

Adult_TrainDataset.Work_Class = le.fit_transform(Adult_TrainDataset.Work_Class)
Adult_TrainDataset.Marital_Status = le.fit_transform(Adult_TrainDataset.Marital_Status)
Adult_TrainDataset.Occupation = le.fit_transform(Adult_TrainDataset.Occupation)
Adult_TrainDataset.Relationship = le.fit_transform(Adult_TrainDataset.Relationship)
Adult_TrainDataset.Race = le.fit_transform(Adult_TrainDataset.Race)
Adult_TrainDataset.Sex = le.fit_transform(Adult_TrainDataset.Sex)
Adult_TrainDataset.Native_Country = le.fit_transform(Adult_TrainDataset.Native_Country)
Adult_TrainDataset.Income = le.fit_transform(Adult_TrainDataset.Income)

Adult_TrainDataset.info()

"""2. Label Encoding"""

from sklearn.preprocessing import LabelEncoder
categorical_column = ['Education']
encoder = LabelEncoder()
for column in categorical_column:
    Adult_TrainDataset[column] = encoder.fit_transform(Adult_TrainDataset[column])
Adult_TrainDataset.info()

"""برای اینکه کدام یک از این متدها را انتخاب کنیم باید به نوع داده های خود دقت کنیم، ماهیت داده‌های خود و نوع متغیرهای طبقه‌بندی شده (اسمی یا ترتیبی)،  را در نظر بگیریم

 (one hot encoding)



*   هر دسته در یک متغیر طبقه بندی به یک ستون باینری تبدیل می شود

*   این روش برای متغیرهای اسمی مناسب است

  با معرفی ستون های اضافی فضای ویژگی بزرگتری ایجاد می کند که می تواند برای گرفتن روابط غیر خطی موثر باشد
    - از تعبیر نادرست ترتیب یا بزرگی در بین دسته ها جلوگیری می کند
    - با این حال، هنگامی که با تعداد زیادی دسته‌بندی منحصربه‌فرد سروکار داریم، می‌تواند منجر به یک مجموعه داده پراکنده با ابعاد بالا شود


  (Label Encoding)



*   به هر دسته یک برچسب عددی منحصر به فرد اختصاص داده شده است
*   این روش برای متغیرهای ترتیبی مناسب است

رابطه ترتیبی بین دسته ها را حفظ می کند. ابعاد داده ها را در مقایسه با روش  قبلی کاهش می دهد

 با این حال، برخی از الگوریتم‌های یادگیری ماشین ممکن است برچسب‌های کدگذاری شده را به‌عنوان معنی یا بزرگی عددی تفسیر کنند که منجر به جهت گیری‌های بالقوه یا فرضیات نادرست می‌شود

 بنابراین برای اینکه کدام روش را انتخاب کنیم به ماهیت داده ها و الزامات خاص مشکل بستگی دارد. به طور کلی، اگر متغیر طبقه‌ای هیچ ترتیب ذاتی نداشته باشد یا زمانی که تعداد زیادی دسته‌بندی منحصربه‌فرد وجود داشته باشد، یک

one hot encoding

 اغلب ترجیح داده می‌شود. این امر به ویژه برای الگوریتم هایی که می توانند داده های با ابعاد بالا را مدیریت کنند، مانند مدل های مبتنی بر درخت، صدق می کند. از جهت دیگر، اگر طبقه‌بندی دارای نظم ذاتی باشد یا مقیاس معناداری را نشان دهد،

Label Encoding

 ممکن است مناسب‌تر باشد. این متد اغلب در مورد متغیرهای ترتیبی استفاده می شود، جایی که رابطه ترتیبی باید حفظ شود






"""

from sklearn.preprocessing import LabelEncoder

let = LabelEncoder()

Adult_TestDataset.Work_Class = let.fit_transform(Adult_TestDataset.Work_Class)
Adult_TestDataset.Marital_Status = let.fit_transform(Adult_TestDataset.Marital_Status)
Adult_TestDataset.Occupation = let.fit_transform(Adult_TestDataset.Occupation)
Adult_TestDataset.Relationship = let.fit_transform(Adult_TestDataset.Relationship)
Adult_TestDataset.Race = let.fit_transform(Adult_TestDataset.Race)
Adult_TestDataset.Sex = let.fit_transform(Adult_TestDataset.Sex)
Adult_TestDataset.Native_Country = let.fit_transform(Adult_TestDataset.Native_Country)
Adult_TestDataset.Income = let.fit_transform(Adult_TestDataset.Income)

Adult_TestDataset.info()

from sklearn.preprocessing import LabelEncoder
categorical_columns2 = ['Education']
encoder2 = LabelEncoder()
for column2 in categorical_columns2:
    Adult_TestDataset[column2] = encoder2.fit_transform(Adult_TestDataset[column2])
Adult_TestDataset.info()

"""---

**شناسایی داده های پرت**

 در این بخش ما داده های پرت را از مجموعه دادهایی که داریم حذف می کنیم تا مدل
  بهتري را بسازیم. با استفاده دستور زیر داده های پرت را شناسایی می کنیم و بعد ان ها را از مجموعه داده مان حذف می کنیم
"""

from sklearn.ensemble import IsolationForest

outlier_detector = IsolationForest(contamination=0.05)
outlier_detector.fit(Adult_TrainDataset)
outlier_predictions = outlier_detector.predict(Adult_TrainDataset)
outlier_mask = outlier_detector.predict(Adult_TrainDataset) == -1
outliers = Adult_TrainDataset[outlier_mask]
print("Outliers:")
print(outliers)
num_outliers = len(outliers)
print("Number of outliers:", num_outliers)
Adult_TrainDataset = Adult_TrainDataset[~outlier_mask]
print("New Dataset without outliers:")
Adult_TrainDataset

"""با توجه به خروجی کد بالا، ما ۱۶۲۸ داده پرت داریم که در نهایت از مجموعه داده مان حذف کرده ایم و یک مجموعه داده جدید با ۳۰۹۳۳ نمونه داریم. این کار را برای مجموعه داده تست تکرار می کنیم:"""

from sklearn.ensemble import IsolationForest

outlier_detector2 = IsolationForest(contamination=0.05)
outlier_detector2.fit(Adult_TestDataset)
outlier_predictions2 = outlier_detector2.predict(Adult_TestDataset)
outlier_mask2 = outlier_detector2.predict(Adult_TestDataset) == -1
outliers2 = Adult_TestDataset[outlier_mask2]
print("Outliers:")
print(outliers2)
num_outliers2 = len(outliers2)
print("Number of outliers:", num_outliers2)
Adult_TestDataset = Adult_TestDataset[~outlier_mask2]
print("New Dataset without outliers:")
Adult_TestDataset

"""با توجه به خروجی کد بالا، ما ۸۱۴ داده پرت داریم که در نهایت از مجموعه داده مان حذف کرده ایم و یک مجموعه داده جدید با ۱۵۴۶۷ نمونه داریم


---

**بخش دوم: بصری سازی**

**دو مدل نمودار بصری سازی برای مجموعه داده اموزش**

در این مرحله ما از دو نوع نمودار برای تحلیل استفاده می کنیم:

۱. Scatter Plot

۲.  Line Chart
"""

import matplotlib.pyplot as plt

plt.scatter(Adult_TrainDataset['Sex'], Adult_TrainDataset['Age'], c=Adult_TrainDataset['Income'])

plt.xlabel('Sex')
plt.ylabel('Age')

cbar = plt.colorbar()
cbar.set_label('Income')

plt.show()

"""با توجه به نمودار بالا می توانیم این نتیجه را بگیریم که مردان، حدودا از بازه ۳۰ تا ۶۰ سال درامدی بیشتر از ۵۰۰۰۰ دارند و همچنین به صورت کلی اقایان، بیشتر درامد بیشتر از ۵۰۰۰۰ را به نسبت خانوم ها از ۳۰ تا ۹۰ سالگی کسب می کنند"""

import matplotlib.pyplot as plt

fig, ax = plt.subplots()

ax.scatter(Adult_TrainDataset['Age'], Adult_TrainDataset['Final_Weight'])

ax.set_xlabel('Age')
ax.set_ylabel('Final Weight')

ax.set_title('Scatter Plot of Age vs Final Weight')

plt.show()

"""با توجه به نمودار بالا می توانیم تقریبا این نتیجه را بگیریم که به صورت کلی از سن ۲۵ تا ۵۵ سالگی به طور میانگین افراد می توانند درامد نهایی بالاتری نسبت به سایر زمان ها داشته باشند"""

import matplotlib.pyplot as plt

x = Adult_TrainDataset['Hours-Per-Week']
y = Adult_TrainDataset['Income']

plt.plot(x, y)

plt.xlabel('Hours-Per-Week')
plt.ylabel('Income')
plt.title('Line Chart')

plt.show()

"""باتوجه به نمودار بالا می توانیم این نتیجه را بگیریم که اغلب افرادی که بین ۲۰ تا ۷۰ ساعت در هفته کار می کنند بیشتر می توانند درامدی بیشتر از ۵۰۰۰۰ را کسب کنند


---

**بخش سوم: ساخت و اموزش مدل ها**

**knn دسته بندی به روش**
"""

from sklearn.neighbors import KNeighborsClassifier

XTR = Adult_TrainDataset.drop(['Income','Age'], axis='columns')
YTR = Adult_TrainDataset.Income

XTE = Adult_TestDataset.drop(['Income','Age'], axis='columns')
YTE = Adult_TestDataset.Income

knn = KNeighborsClassifier(n_neighbors=10)
knn.fit(XTR, YTR)

trAcc = knn.score(XTR, YTR)
teAcc = knn.score(XTE, YTE)

print ('Train Accuracy: ', trAcc)
print ('Test Accuracy: ', teAcc)

"""k=10
در قسمت بالا ما، دقت مدل را به ازای

در هر دو مجموعه داده محاسبه کرده ایم
"""

import matplotlib.pyplot as plt

trAcc=[]
teAcc=[]
Ks=[]

for i in range(1,11):
    KNN = KNeighborsClassifier(n_neighbors = i)
    KNN.fit(XTR, YTR)
    trAcc.append(KNN.score(XTR, YTR))
    teAcc.append(KNN.score(XTE, YTE))
    Ks.append(i)

plt.plot(Ks, trAcc, label = 'Train')
plt.plot(Ks, teAcc, label = 'Test')
plt.xlabel('K')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

"""kدر قسمت بالا ما به ازای مقادیر مختلف


kنمودار ان را رسم کرده ایم تا به ازای


kهای مختلف مقدار دقت مدل را برای مجموعه داده تست و اموزشی مشاهده کنیم. در واقع به ازای
هر  


kیک مدل فیت می کنیم و خروجی ان را رسم می کنیم. همانطور که در نمودار بالا می بینیم با افزایش مقدار


kدقت مدل برای هر دو مجموعه داده با یک نسبتی به هم نزدیک می شوند. همچنین با افزایش مقدار


دقت مدل برای مجموعه داده اموزشی کاهش و دقت مدل برای مجموعه داده تست افزایش پیدا می کند
"""

from sklearn.metrics import confusion_matrix
import seaborn as sns

y_pred = knn.predict(XTE)
cm = confusion_matrix(YTE, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

""" ما باتوجه به این جدول می توانیم بگوییم: ما ۲۰۰ بار در
واقعیت و در پیش بینی یک داشتیم، ما ۳۷۲ بار در واقعیت صفر و در پیش بینی یک داشتیم، ما ۳۳۷۴ بار در واقعیت
یک و در پیش بینی صفر داشتیم و ما ۱۱۵۲۱ بار در واقعیت و پیش بینی صفر داشتیم. همچنین شدت رنگ به ما این امکان را می دهد که توزیع پیش بینی های درست و نادرست را در کلاس های مختلف به صورت بصری تفسیر کنیم. ما می توانیم این الگو را مشاهده کنیم و مناطقی را که مدل از نظر پیش بینی کلاس های خاص خوب یا ضعیف عمل کرده است را، شناسایی کنیم  
"""

from sklearn.metrics import classification_report
print(classification_report(YTE, y_pred))

"""*    precision

برای هر کلاس نرخ نسبت مثبت های واقعی پیش بینی شده به کل مثبت های پیش بینی شده را بیان می کند
که به عنوان مثال این مقدار برای کلاس صفر ۰.۷۷ درصد است.
 این به این معنی است که ۷۷ درصد از نمونه های پیش بینی شده از کلاس صفر در واقع مثبت واقعی هستند

*    Recall

برای هر کلاس نرخ نسبت پیش بینی مثبت های واقعی به کل مثبت های واقعی را نشان می دهد. به عنوان مثال، برای کلاس صفر ۰.۹۷ است، این به این معنی است که ۹۷ درصد از
نمونه های واقعی کلاس صفر به درستی به عنوان کلاس صفر طبقه بندی شده اند


*    F1-Score

برای هر کلاس، میانگین هارمونیک دو معیار دقت و یادآوری را محاسبه می کند. یک متر واحد را ارائه می دهد که دقت و یادآوری را متعادل کند. به عنوان مثال، برای کلاس صفر ۰.۸۶ است

*    Support

در این قسمت تعداد رخدادهایی که در هر کلاس برچسب های درست را دریافت کرده اند، نشان می دهد. به عنوان مثال، کلاس صفر دارای ۱۱۸۹۳ است، به این معنی که ۱۱۸۹۳ نمونه در مجموعه تست وجود دارد که متعلق به کلاس صفر هستند

*     Accuracy

نشان دهنده دقت کلی مدل طبقه بندی کننده است. نسبت نمونه هایی است که به درستی طبقه بندی شده اند به تعداد کل نمونه ها را نشان می دهد که در اینجا ۷۶ درصد است

*     Macro average

میانگین کلان، میانگین مقدار متریک را در همه کلاس ها محاسبه می کند. هنگام محاسبه میانگین با هر کلاس به صورت برابر رفتار می کند.

*     Weighted average

میانگین وزنی میانگین ارزش متریک را در همه کلاس‌ها محاسبه می‌کند، وزن آن‌ها بر اساس پشتیبانی یا تعداد نمونه‌ها وزن دهی می کند. درواقع عدم تعادل طبقاتی را در نظر می گیرد و به کلاس هایی که تعداد نمونه های بیشتری دارند وزن بیشتری می دهد

---

**SVM دسته بندی به روش**
"""

import sklearn.svm as sv

Clsfr = sv.SVC(kernel='rbf')
Clsfr.fit(XTR, YTR)

trAc = Clsfr.score(XTR, YTR)
teAc = Clsfr.score(XTE, YTE)

print ('Train Accuracy: ', trAc)
print ('Test Accuracy: ', teAc)

"""در قسمت بالا ما،  
دقت مدل را با هسته

rbf

  در هر دو مجموعه داده محاسبه کرده ایم
"""

Y_pred = Clsfr.predict(XTE)
cm = confusion_matrix(YTE, Y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

""" ما باتوجه به این جدول می توانیم بگوییم: ما ۰ بار در
واقعیت و در پیش بینی یک داشتیم، ما ۰ بار در واقعیت صفر و در پیش بینی یک داشتیم، ما ۳۵۷۴ بار در واقعیت
یک و در پیش بینی صفر داشتیم و ما ۱۱۸۹۳ بار در واقعیت و پیش بینی صفر داشتیم. همچنین شدت رنگ به ما این امکان را می دهد که توزیع پیش بینی های درست و نادرست را در کلاس های مختلف به صورت بصری تفسیر کنیم. ما می توانیم این الگو را مشاهده کنیم و مناطقی را که مدل از نظر پیش بینی کلاس های خاص خوب یا ضعیف عمل کرده است را، شناسایی کنیم  
"""

print(classification_report(YTE, Y_pred))

"""*    precision

برای هر کلاس نرخ نسبت مثبت های واقعی پیش بینی شده به کل مثبت های پیش بینی شده را بیان می کند
که به عنوان مثال این مقدار برای کلاس صفر ۰.۷۷ درصد است.
 این به این معنی است که ۷۷ درصد از نمونه های پیش بینی شده از کلاس صفر در واقع مثبت واقعی هستند

*    Recall

برای هر کلاس نرخ نسبت پیش بینی مثبت های واقعی به کل مثبت های واقعی را نشان می دهد. به عنوان مثال، برای کلاس صفر ۱ است، این به این معنی است که ۱۰۰ درصد از
نمونه های واقعی کلاس صفر به درستی به عنوان کلاس صفر طبقه بندی شده اند


*    F1-Score

برای هر کلاس، میانگین هارمونیک دو معیار دقت و یادآوری را محاسبه می کند. یک متر واحد را ارائه می دهد که دقت و یادآوری را متعادل کند. به عنوان مثال، برای کلاس صفر ۰.۸۷ است

*    Support

در این قسمت تعداد رخدادهایی که در هر کلاس برچسب های درست را دریافت کرده اند، نشان می دهد. به عنوان مثال، کلاس صفر دارای ۱۱۸۹۳ است، به این معنی که ۱۱۸۹۳ نمونه در مجموعه تست وجود دارد که متعلق به کلاس صفر هستند

*     Accuracy

نشان دهنده دقت کلی مدل طبقه بندی کننده است. نسبت نمونه هایی است که به درستی طبقه بندی شده اند به تعداد کل نمونه ها را نشان می دهد که در اینجا ۷۷ درصد است

*     Macro average

میانگین کلان، میانگین مقدار متریک را در همه کلاس ها محاسبه می کند. هنگام محاسبه میانگین با هر کلاس به صورت برابر رفتار می کند.

*     Weighted average

میانگین وزنی میانگین ارزش متریک را در همه کلاس‌ها محاسبه می‌کند، وزن آن‌ها بر اساس پشتیبانی یا تعداد نمونه‌ها وزن دهی می کند. درواقع عدم تعادل طبقاتی را در نظر می گیرد و به کلاس هایی که تعداد نمونه های بیشتری دارند وزن بیشتری می دهد

---

**درخت تصمیم**
"""

import sklearn.tree as tr

trAcc = []
teAcc = []
MD = []

for i in range(2, 12):
    DT = tr.DecisionTreeClassifier(max_depth = i)
    DT.fit(XTR, YTR)
    trAcc.append(DT.score(XTR, YTR))
    teAcc.append(DT.score(XTE, YTE))
    MD.append(i)

plt.plot(MD, trAcc, label = 'Train', marker = 'o')
plt.plot(MD, teAcc, label = 'Test', marker = 'o')
plt.xlabel('Max Depth')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

"""با توجه به نمودار بالا هرچه قدر عمق درخت تصمیم افزایش پیدا کند میزان دقت مدل برای مجموعه داده اموزشی افزایش پیدا می کند، در حالی که این افزایش عمق درخت تصمیم تاثیر چندانی در بهبود دقت مدل برای مجموعه داده تست ندارد و حتی ممکن است با این افزایش، دقت مدل کاهش پیدا کند"""

Y_pred0 = DT.predict(XTE)
cm = confusion_matrix(YTE, Y_pred0)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

""" ما باتوجه به این جدول می توانیم بگوییم: ما ۱۹۵۱ بار در
واقعیت و در پیش بینی یک داشتیم، ما ۱۱۰۴ بار در واقعیت صفر و در پیش بینی یک داشتیم، ما ۱۶۲۳ بار در واقعیت
یک و در پیش بینی صفر داشتیم و ما ۱۰۷۸۹ بار در واقعیت و پیش بینی صفر داشتیم. همچنین شدت رنگ به ما این امکان را می دهد که توزیع پیش بینی های درست و نادرست را در کلاس های مختلف به صورت بصری تفسیر کنیم. ما می توانیم این الگو را مشاهده کنیم و مناطقی را که مدل از نظر پیش بینی کلاس های خاص خوب یا ضعیف عمل کرده است را، شناسایی کنیم  
"""

print(classification_report(YTE, Y_pred0))

"""*    precision

برای هر کلاس نرخ نسبت مثبت های واقعی پیش بینی شده به کل مثبت های پیش بینی شده را بیان می کند
که به عنوان مثال این مقدار برای کلاس صفر ۰.۸۷ درصد است.
 این به این معنی است که ۸۷ درصد از نمونه های پیش بینی شده از کلاس صفر در واقع مثبت واقعی هستند

*    Recall

برای هر کلاس نرخ نسبت پیش بینی مثبت های واقعی به کل مثبت های واقعی را نشان می دهد. به عنوان مثال، برای کلاس صفر ۰.۹۱ است، این به این معنی است که ۹۱ درصد از
نمونه های واقعی کلاس صفر به درستی به عنوان کلاس صفر طبقه بندی شده اند


*    F1-Score

برای هر کلاس، میانگین هارمونیک دو معیار دقت و یادآوری را محاسبه می کند. یک متر واحد را ارائه می دهد که دقت و یادآوری را متعادل کند. به عنوان مثال، برای کلاس صفر ۰.۸۹ است

*    Support

در این قسمت تعداد رخدادهایی که در هر کلاس برچسب های درست را دریافت کرده اند، نشان می دهد. به عنوان مثال، کلاس صفر دارای ۱۱۸۹۳ است، به این معنی که ۱۱۸۹۳ نمونه در مجموعه تست وجود دارد که متعلق به کلاس صفر هستند

*     Accuracy

نشان دهنده دقت کلی مدل طبقه بندی کننده است. نسبت نمونه هایی است که به درستی طبقه بندی شده اند به تعداد کل نمونه ها را نشان می دهد که در اینجا ۸۲ درصد است

*     Macro average

میانگین کلان، میانگین مقدار متریک را در همه کلاس ها محاسبه می کند. هنگام محاسبه میانگین با هر کلاس به صورت برابر رفتار می کند.

*     Weighted average

میانگین وزنی میانگین ارزش متریک را در همه کلاس‌ها محاسبه می‌کند، وزن آن‌ها بر اساس پشتیبانی یا تعداد نمونه‌ها وزن دهی می کند. درواقع عدم تعادل طبقاتی را در نظر می گیرد و به کلاس هایی که تعداد نمونه های بیشتری دارند وزن بیشتری می دهد

---

**مقایسه عملکرد مدل سازی ها**

باتوجه به اینکه در قسمت های بالا میزان دقت عملکرد مدل به روش

KNN

برابر با ۷۶ درصد است، میزان دقت عملکرد مدل به روش

SVM

برابر با ۷۷ درصد و میزان دقت عملکرد مدل به روش درخت تصمیم برابر با ۸۲ درصد است. با توجه به این موضوع مدل درخت تصمیم بالاترین دقت را دارد که بهتر است این مدل را برای مدل سازی مان انتخاب کنیم
"""
