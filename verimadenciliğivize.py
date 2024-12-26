#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Kütüphanelerin import edilmesi
from sklearn.preprocessing import LabelEncoder #
from sklearn import preprocessing   #normalleştirme için
from sklearn.model_selection import train_test_split,GridSearchCV, cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.metrics import roc_auc_score,roc_curve
from sklearn.linear_model import LogisticRegression
from warnings import filterwarnings
import os

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from itertools import combinations
# !pip install missingno
import missingno as msno
from datetime import date
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler
#pip install xlrd
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 500)


filterwarnings('ignore')


# In[122]:


# Regular Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

#Sci-kit Model Development
from sklearn.model_selection import StratifiedKFold, train_test_split, cross_val_score, RandomizedSearchCV, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, BaggingClassifier, AdaBoostClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import KNNImputer
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif, RFE
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import PCA


#Sci-kit Metrics
from sklearn import metrics
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
#Sci-kit Scaler
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder

#XGBoost Classifier
#!pip install XGBoost
from xgboost import XGBClassifier


# In[2]:


train = pd.read_excel(r"C:\Users\HANDENUR\Downloads\ist405_ara_sinav_data.xls",engine='xlrd')
test=pd.read_excel(r"C:\Users\HANDENUR\Downloads\ist405_ogrenci_test.xlsx")


# In[3]:


train = train.replace([r"^\s*$", r"^na$", r"^NA$", r"^n\/a$", r"^N\/A$"], np.nan, regex=True)
test = test.replace([r"^\s*$", r"^na$", r"^NA$", r"^n\/a$", r"^N\/A$"], np.nan, regex=True)


# In[ ]:





# In[26]:


def check_df(dataframe, head=5):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(head))
    print("##################### Tail #####################")
    print(dataframe.tail(head))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Quantiles #####################")
    print(dataframe.describe().T)

check_df(train)


# In[5]:


check_df(test)


# In[ ]:





# In[4]:


# Kategorik değişkenlerin listesi
categorical_columns = [
    'cınsıyet', 'egıtımduzeyı', 'meslegı', 'sıgarakullanımı', 'tanı',
    'hastaneyeyattımı', 'ailedekoahveyaastımTanılıHastavarmı', 
    'varsakımde ANNE', 'varsakımde BABA', 'varsakımde KARDES', 'varsakımde DİĞER']

# Sayısal değişkenlerin listesi
numeric_columns = [
    'YAŞ', 'sıgarayıbırakannekadarGÜNıcmıs', 'sıgarabırakangundekacadetıcmıs',
    'nezamanbırakmısGÜN', 'sıgarayadevamedengundekacadetıcıyo', 'tanısuresıyıl',
    'tanısuresıay', 'acılservıseyatıssayısı', 'acılservısetoplamyatıssuresısaat',
    'acilservistoplamyatışsüresigün', 'yogumbakımayatıssayısı', 
    'yogumbakımatoplamyatıssuresısaat', 'yogumbakımatoplamyatıssüresıgun',
    'servıseyatıssayısı', 'servıseoplamyatıssuresısaat', 'servisetoplamyatıssüresıgun',
    'boy', 'vucutagırlıgı', 'kanbasıncısıstolık', 'kanbasıncıdıastolık', 'nabız', 
    'solunumsayısı', 'FEV1', 'PEF', 'FEV1 %', 'PEF %', 'FEV1/FVC Değeri']

# Tarih sütunları (varsa)
datetime_columns = ['basvurutarıhı']


# Veri tiplerini düzeltmek için fonksiyon
def adjust_dtypes(data, is_train=True):
    # Kategorik değişkenleri düzeltme
    for col in categorical_columns:
        if col in data.columns:  # Eğer veri setinde varsa
            data[col] = data[col].astype('object')
    
    # Sayısal değişkenleri düzeltme
    for col in numeric_columns:
        if col in data.columns:  # Eğer veri setinde varsa
            data[col] = pd.to_numeric(data[col], errors='coerce')  # Hatalı verileri NaN yapar
    
    # Tarih değişkenlerini düzeltme
    for col in datetime_columns:
        if col in data.columns:  # Eğer veri setinde varsa
            data[col] = pd.to_datetime(data[col], errors='coerce')
    
    
    return data

# Eğitim ve test kümelerine uygula
train = adjust_dtypes(train, is_train=True)
test = adjust_dtypes(test, is_train=False)

test.info()


# bağımlı değişken yani "tanı" sütunu test kümesinde yok. Zaten amaç da onu tahmin etmek

# In[7]:


train.info()


# ## eksik değer analizi

# In[5]:


def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]
    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")
    if na_name:
        return na_columns

na_columns = missing_values_table(train, na_name=True)


# In[73]:


na_columns = missing_values_table(test, na_name=True)


# In[8]:


plt.figure(figsize=(10, 6))
sns.heatmap(test.isnull(), cbar=False, cmap='viridis')
plt.title('Missing Data Heatmap')
plt.show()


# In[9]:


plt.figure(figsize=(10, 6))
sns.heatmap(train.isnull(), cbar=False, cmap='viridis')
plt.title('Missing Data Heatmap')
plt.show()


# # eksik değer doldurma
# 

# In[6]:


df_train=train.copy()
df_test=test.copy()


# In[11]:


df_train.info()


# In[7]:


# "sıgarakullanımı" sütununda 1 (Hiç içmemiş) ve 3 (Halen içiyor) olan satırlara 0 yazmak (sıgarayıbırakannekadarGÜNıcmıs)
df_train.loc[
    df_train['sıgarakullanımı'].isin([1, 3]), 'sıgarayıbırakannekadarGÜNıcmıs'] = 0

# Kontrol etmek için güncellenmiş değerleri görüntüleme
güncellenmis_1 = df_train[
    df_train['sıgarakullanımı'].isin([1, 3])][['sıgarakullanımı', 'sıgarayıbırakannekadarGÜNıcmıs']]

print(güncellenmis_1.head())

# "sıgarakullanımı" sütununda 1 (Hiç içmemiş) ve 3 (Halen içiyor) olan satırlara 0 yazmak (sıgarabırakangundekacadetıcmıs)
df_train.loc[
    df_train['sıgarakullanımı'].isin([1, 3]), 'sıgarabırakangundekacadetıcmıs'] = 0

# Kontrol etmek için güncellenmiş değerleri görüntüleme
güncellenmis_2 = df_train[
    df_train['sıgarakullanımı'].isin([1, 3])][['sıgarakullanımı', 'sıgarabırakangundekacadetıcmıs']]

print(güncellenmis_2.head())


# "sıgarakullanımı" sütununda 1 (Hiç içmemiş) ve 3 (Halen içiyor) olan satırlara 0 yazmak (nezamanbırakmısGÜN)
df_train.loc[
    df_train['sıgarakullanımı'].isin([1, 3]), 'nezamanbırakmısGÜN'] = 0

# Kontrol etmek için güncellenmiş değerleri görüntüleme
güncellenmis_3 = df_train[
    df_train['sıgarakullanımı'].isin([1, 3])][['sıgarakullanımı', 'nezamanbırakmısGÜN']]

print(güncellenmis_3.head())

# "sıgarayadevamedengundekacadetıcıyo" sütununda 1 (Hiç içmemiş) ve 2 (Bırakmış) olan satırlara 0 yazmak
df_train.loc[df_train['sıgarakullanımı'].isin([1, 2]), 'sıgarayadevamedengundekacadetıcıyo'] = 0

# Kontrol etmek için güncellenmiş değerleri görüntüleme
güncellenmis_4 = df_train[
    df_train['sıgarakullanımı'].isin([1, 2])][['sıgarakullanımı', 'sıgarayadevamedengundekacadetıcıyo']]

print(güncellenmis_4.head())


# In[8]:


# Eksik değerleri yalnızca ilgili sütunlarda median ile doldur
median_ile_doldur = [ 'tanısuresıyıl', 'tanısuresıay',
                  'kanbasıncısıstolık', 'kanbasıncıdıastolık', 'FEV1', 'PEF', 'PEF %']

for col in median_ile_doldur:
    if col in df_train.columns:  # Sütunun gerçekten veri kümesinde olup olmadığını kontrol et
        median_value = df_train[col].median()  # İlgili sütunun median al
        df_train[col].fillna(median_value, inplace=True)  # Eksik değerleri median ile doldur


# In[9]:


na_columns = missing_values_table(df_train, na_name=True)


# "nezamanbırakmısGÜN" değişkenini doldurmamıza rağmen eksik 2 değer kalmış. Detaylı incelediğimizde sigarayı bırakan(2) durumundaki kişilere ait.

# In[10]:


# Eksik değerlere sahip satırları filtrele
eksik_veriler = df_train[df_train["nezamanbırakmısGÜN"].isnull()]
# Eksik verilerin bilgisi
eksik_veriler


# In[11]:


# Sadece sigara bırakmış (2) olan kişilerin medyanını hesapla 
medyan_nezaman_birakmis1 = df_train.loc[df_train['sıgarakullanımı'] == 2, 'nezamanbırakmısGÜN'].median()

# Sigara bırakmış (2) olan kişiler için nezamanbırakmısGÜN'deki eksik değerleri doldur 
df_train.loc[(df_train['sıgarakullanımı'] == 2) & (df_train['nezamanbırakmısGÜN'].isnull()), 
             'nezamanbırakmısGÜN'] = medyan_nezaman_birakmis1


# In[17]:


df_train[df_train.isnull().any(axis=1)]

# 454 ve 497. indeksler için NaN değerleri "bilgiyok" ile doldur
train.loc[[454, 497]] = train.loc[[454, 497]].fillna("bilgiyok")
train.loc[[454, 497]]
# In[18]:


df_train[df_train.isnull().any(axis=1)]


# 89.index bilgisine baktığımızda ailede koah veya astım tanılı kimse yokmuş. o zaman buna bağlı olarak "varsakimde" değişkenlerini sıfırla dolduralım.
# 287 ve 303.indexteki ssatıra baktığımızdaysa ailede kkoah veya astım tanılı biri var ama kimde olduğu bilgisi girilmemiş.onları "bilgiyok" string ifadesiyle dolduralım.

# In[12]:


# 89. indeks için NaN değerlerini 0 ile doldur
df_train.loc[89] = df_train.loc[89].fillna(0)

# 287 ve 303. indeksler için NaN değerlerini "bilgiyok" string ifadesiyle doldur
df_train.loc[287] = df_train.loc[287].fillna("bilgiyok")
df_train.loc[303] = df_train.loc[303].fillna("bilgiyok")


# In[13]:


df_train[df_train.isnull().any(axis=1)]


# In[21]:


df_train[df_train['ailedekoahveyaastımTanılıHastavarmı'] == 2]


# In[22]:


# Koşulları yazıyoruz
filtered_train = df_train[
    (df_train['ailedekoahveyaastımTanılıHastavarmı'] == 2) &  # ailede koah veya astım var olanlar
    ~(  # Negasyon ile en az birinin 1'e eşit olduğu satırları hariç tutuyoruz
        (df_train['varsakımde ANNE'] == 1) |
        (df_train['varsakımde BABA'] == 1) |
        (df_train['varsakımde KARDES'] == 1) |
        (df_train['varsakımde DİĞER'] == 1)
    )
]

# Filtrelenmiş veriyi kontrol et
filtered_train


# görüldüğü üzere AİLEDE KİMSEDE ('varsakımde ANNE', 'varsakımde BABA', 'varsakımde KARDES', 'varsakımde DİGER' sütunlarının 0 OLMASI KİMSEDE HASTALIĞIN OLMADIĞINI GÖSTERİR) HASTALIK YOK İKEN AİLEDEKOAHVEYAASTIMTANILIHASTAVARMI DEĞİŞKENİ VAR (2) OLARAK GÖZÜKÜYOR.  "varskimde" değişkenleri sıfır olduğu için 'ailedekoahveyaastımTanılıHastavarmı değişkeni 1 olsun

# In[14]:


# Şüpheli durumları tespit etmek için koşul
condition = (
    (df_train['ailedekoahveyaastımTanılıHastavarmı'] == 2) &  # ailede hasta var olarak işaretlenmiş
    (df_train['varsakımde ANNE'] == 0) & 
    (df_train['varsakımde BABA'] == 0) & 
    (df_train['varsakımde KARDES'] == 0) & 
    (df_train['varsakımde DİĞER'] == 0)  # ancak ailede kimse hasta değil
)

# Bu koşulu sağlayan satırlarda ailedekoahveyaastımTanılıHastavarmı sütununu 1 olarak güncelle
df_train.loc[condition, 'ailedekoahveyaastımTanılıHastavarmı'] = 1

# Doğrulama: Değişiklik yapılan satırları kontrol et
df_train


# In[15]:


df_train[df_train.isnull().any(axis=1)]


# ## test kümesi eksik değerler

# In[16]:


na_columns = missing_values_table(df_test, na_name=True) 


# In[17]:


# "sıgarakullanımı" sütununda 1 (Hiç içmemiş) ve 3 (Halen içiyor) olan satırlara 0 yazmak (sıgarayıbırakannekadarGÜNıcmıs)
df_test.loc[
    df_test['sıgarakullanımı'].isin([1, 3]), 'sıgarayıbırakannekadarGÜNıcmıs'] = 0

# Kontrol etmek için güncellenmiş değerleri görüntüleme
güncellenmis_11 = df_test[
    df_test['sıgarakullanımı'].isin([1, 3])][['sıgarakullanımı', 'sıgarayıbırakannekadarGÜNıcmıs']]

print(güncellenmis_11.head())

# "sıgarakullanımı" sütununda 1 (Hiç içmemiş) ve 3 (Halen içiyor) olan satırlara 0 yazmak (sıgarabırakangundekacadetıcmıs)
df_test.loc[
    df_test['sıgarakullanımı'].isin([1, 3]), 'sıgarabırakangundekacadetıcmıs'] = 0

# Kontrol etmek için güncellenmiş değerleri görüntüleme
güncellenmis_22 = df_test[
    df_test['sıgarakullanımı'].isin([1, 3])][['sıgarakullanımı', 'sıgarabırakangundekacadetıcmıs']]

print(güncellenmis_22.head())


# "sıgarakullanımı" sütununda 1 (Hiç içmemiş) ve 3 (Halen içiyor) olan satırlara 0 yazmak (nezamanbırakmısGÜN)
df_test.loc[
    df_test['sıgarakullanımı'].isin([1, 3]), 'nezamanbırakmısGÜN'] = 0

# Kontrol etmek için güncellenmiş değerleri görüntüleme
güncellenmis_33 = df_test[
    df_test['sıgarakullanımı'].isin([1, 3])][['sıgarakullanımı', 'nezamanbırakmısGÜN']]

print(güncellenmis_33.head())

# "sıgarayadevamedengundekacadetıcıyo" sütununda 1 (Hiç içmemiş) ve 2 (Bırakmış) olan satırlara 0 yazmak
df_test.loc[
    df_test['sıgarakullanımı'].isin([1, 2]), 'sıgarayadevamedengundekacadetıcıyo'] = 0

# Kontrol etmek için güncellenmiş değerleri görüntüleme
güncellenmis_44 = df_test[
    df_test['sıgarakullanımı'].isin([1, 2])][['sıgarakullanımı', 'sıgarayadevamedengundekacadetıcıyo']]

print(güncellenmis_44.head())


# In[18]:


na_columns = missing_values_table(df_test, na_name=True) 


# In[19]:


# Sayısal değişkenler için median ile doldurma
medyan_doldurma=["YAŞ","FEV1","FEV1 %","PEF %","PEF", "FEV1/FVC Değeri", "tanısuresıay"]

for col in medyan_doldurma:
    if col in df_test.columns:  # Sütunun test veri setinde olup olmadığını kontrol et
        medyan_degeri = df_test[col].median()  # median değerini al
        df_test[col].fillna(medyan_degeri, inplace=True)  # Test kümesindeki eksik değerleri median ile doldur


# In[20]:


# Sadece sigara bırakmış (2) olan kişilerin medyanını hesapla
medyan_nezaman_birakmis2 = df_test.loc[df_test['sıgarakullanımı'] == 2, 'nezamanbırakmısGÜN'].median()

# Sigara bırakmış (2) olan kişiler için nezamanbırakmısGÜN'deki eksik değerleri doldur
df_test.loc[(df_test['sıgarakullanımı'] == 2) & (df_test['nezamanbırakmısGÜN'].isnull()), 
            'nezamanbırakmısGÜN'] = medyan_nezaman_birakmis2


# In[30]:


df_test[df_test.isnull().any(axis=1)]


# In[31]:


na_columns = missing_values_table(df_test, na_name=True) 


# In[32]:


df_test[df_test["ailedekoahveyaastımTanılıHastavarmı"].isnull()]


# varsakimde KARDES VE DİĞER bilgisi 1 yani var olarak girildiyse bu eksik değer "2" yani ailede koah veya astım tanılı biei var.

# In[21]:


df_test.loc[59] = df_test.loc[59].fillna(2)


# In[22]:


df_test[df_test["ailedekoahveyaastımTanılıHastavarmı"].isnull()]


# In[23]:


# Koşulları yazıyoruz
filtered_test = df_test[
    (test['ailedekoahveyaastımTanılıHastavarmı'] == 2) &  # ailede koah veya astım var olanlar
    ~(  # Negasyon ile en az birinin 1'e eşit olduğu satırları hariç tutuyoruz
        (df_test['varsakımde ANNE'] == 1) |
        (df_test['varsakımde BABA'] == 1) |
        (df_test['varsakımde KARDES'] == 1) |
        (df_test['varsakımde DİĞER'] == 1)
    )
]

# Filtrelenmiş veriyi kontrol et
filtered_test


# bazı yanlışlıklar var. ve bazılarının da bilgisi yok.

# In[24]:


# Şüpheli durumları tespit etmek için koşul
condition2 = (
    (df_test['ailedekoahveyaastımTanılıHastavarmı'] == 2) &  # ailede hasta var olarak işaretlenmiş
    (df_test['varsakımde ANNE'] == 0) & 
    (df_test['varsakımde BABA'] == 0) & 
    (df_test['varsakımde KARDES'] == 0) & 
    (df_test['varsakımde DİĞER'] == 0)) # ancak ailede kimse hasta değil

# Bu koşulu sağlayan satırlarda ailedekoahveyaastımTanılıHastavarmı sütununu 1 olarak güncelle
df_test.loc[condition2, 'ailedekoahveyaastımTanılıHastavarmı'] = 1.0

# Doğrulama: Değişiklik yapılan satırları kontrol et
df_test[condition2]


# In[25]:


# İlgili sütunlar
columns_to_update = ["varsakımde ANNE", "varsakımde BABA", "varsakımde KARDES", "varsakımde DİĞER"]

# Şartlı sıfırlama işlemi (yalnızca "ailedekoahveyaastımTanılıHastavarmı" == 1 için)
df_test.loc[df_test["ailedekoahveyaastımTanılıHastavarmı"] == 1, columns_to_update] = \
    df_test.loc[df_test["ailedekoahveyaastımTanılıHastavarmı"] == 1, columns_to_update].fillna(0)

# Sonuç
df_test[df_test.isnull().any(axis=1)]


# In[26]:


# Eğer 'ailedekoahveyaastımTanılıHastavarmı' == 2 ise ve ilgili sütun NaN ise, 'Bilgi Yok' ile doldur
columns_to_fill = ['varsakımde ANNE','varsakımde BABA', 'varsakımde KARDES', 'varsakımde DİĞER']

for col in columns_to_fill:
    df_test.loc[(df_test['ailedekoahveyaastımTanılıHastavarmı'] == 2) & (df_test[col].isnull()), col]="bilgiyok" 
# İşlemin doğruluğunu kontrol et
df_test[columns_to_fill].isnull().sum()  # Tüm sütunlarda eksik değerlerin sıfır olduğundan emin olun


# In[39]:


na_columns = missing_values_table(df_test, na_name=True)


# In[40]:


df_test


# ### aykırı değer analizi 

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# ## hedef değişken 

# Hedef Değişkenin Dağılımı
# Veri setimizdeki "tanı" değişkeninin dağılımını analiz etmek, hem dengesiz veri problemi olup olmadığını tespit etmemizi sağlar, hem de modelimizin bu dağılımı ne kadar iyi tahmin edebileceğine dair bize bir fikir verir. Eğer hedef değişkenimiz dengesizse (örneğin, bir sınıf diğerlerine göre çok daha fazla gözleme sahipse), modelimiz bu dengesizliği öğrenebilir ve yanlış tahminlerde bulunabilir.
# 

# In[22]:


#hedef değişkenin incelenmesi
# "tanı" değişkeni için countplot
sns.countplot(data=df_train, x="tanı", palette="flare")

# Başlık ve etiketler ekleme
plt.title("Tanı Değişkeninin Sıklık Dağılımı", fontsize=14)
plt.xlabel("Tanı", fontsize=12)
plt.ylabel("Sıklık", fontsize=12)

# Grafiği gösterme
plt.show()


# hedef değişkenimiz dengeli dağılıyor.

# In[335]:


#Kategorik Değişken Analizi (Analysis of Categorical Variables)

def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))

    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe, palette="mako")
        plt.show()


for col in categorical_columns:
    cat_summary(df_train, col)


# In[336]:


#test kümesi kategoriklerin analizi
for col in categorical_columns:
    if col in df_test.columns:
        cat_summary(df_test, col, plot=True)
    else:
        print(f"Sütun bulunamadı: {col}")


# In[38]:


# 3. Sayısal Değişken Analizi (Analysis of Numerical Variables)
######################################

def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist(bins=50)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show()

    print("#####################################")


for col in numeric_columns:
    num_summary(df_train, col, True)


# servisetoplamyatışsüesisaat ve yoğunbakımyatışsüresisaat değişkenleri full sıfırdan oluşuyor. bu değişkenleri ilerde drop edelim.

# In[121]:


#grafiklere te tek değil de birlikte bakmak istersek
df_train[numeric_columns].hist(figsize=(16, 20), bins=50, xlabelsize=8, ylabelsize=8)


# In[27]:


# NUMERİK DEĞİŞKENLERİN TARGET GÖRE ANALİZİ
def target_summary_with_num(dataframe, target, numerical_col):
    print(dataframe.groupby(target).agg({numerical_col: "mean"}), end="\n\n\n")

for col in numeric_columns:
    target_summary_with_num(df_train, "tanı", col)


# In[71]:


#test kümesi sayısalların analizi
for col in numeric_columns:
    num_summary(df_test, col, True)


# In[ ]:


yogumbakımayatıssayısı,yogumbakımatoplamyatıssuresısaat,yogumbakımatoplamyatıssüresıgun,servıseoplamyatıssuresısaat


# In[73]:


#grafiklere te tek değil de birlikte bakmak istersek
df_test[numeric_columns].hist(figsize=(16, 20), bins=50, xlabelsize=8, ylabelsize=8)


# değişkenler genelde sıfırda toplanmış ve çarpık dağılmış haldeler.

# In[43]:


#kategorik değişkenlerin hedef değişkene göre analizi
def target_summary_with_cat(dataframe, target, categorical_col):
    """
    Bağımlı değişken kategorik olduğunda, bağımsız değişkenin her bir kategorisine göre bağımlı değişkenin
    dağılımını gösteren bir özet çıkarır.

    :param dataframe: pandas DataFrame
    :param target: kategorik bağımlı değişken
    :param categorical_col: analiz edilecek kategorik bağımsız değişken
    """
    # Çapraz tablo ile bağımsız ve bağımlı değişken ilişkisini inceleme
    cross_tab = pd.crosstab(dataframe[categorical_col], dataframe[target], normalize="index") * 100
    print(f"--- {categorical_col} için Tanı Dağılımı ---")
    print(cross_tab)
    print("\n")


# In[44]:


# Tüm kategorik sütunlar için hedef değişken analizi
for col in categorical_columns:
    target_summary_with_cat(df_train, "tanı", col)

#kategorik değişkenlerin hedef değişkene göre analizi
def target_summary_with_cat(dataframe, target, categorical_col):
    print(categorical_col)
    print(pd.DataFrame({"TARGET_MEAN": dataframe.groupby(categorical_col)[target].mean(),
                        "Count": dataframe[categorical_col].value_counts(),
                        "Ratio": 100 * dataframe[categorical_col].value_counts() / len(dataframe)}), end="\n\n\n")

for col in categorical_columns:
    target_summary_with_cat(df_train, "tanı", col)
yogumbakımayatıssayısı,yogumbakımatoplamyatıssuresısaat,yogumbakımatoplamyatıssüresıgun,servıseoplamyatıssuresısaat
# In[28]:


#sıfır sütünlarını ve hastano , tarih değişkenlerini drop edelim.'
df_train = df_train.drop(['yogumbakımatoplamyatıssuresısaat', 'servıseoplamyatıssuresısaat', 'hastaNo','basvurutarıhı'], axis=1)


# In[29]:


df_test= df_test.drop(['yogumbakımatoplamyatıssuresısaat', 'servıseoplamyatıssuresısaat', 'hastaNo','basvurutarıhı'], axis=1)


# In[30]:


numeric_columns = [col for col in numeric_columns if col in df_train.columns]
numeric_columns = [col for col in numeric_columns if col in df_test.columns]


# In[26]:


# Korelasyon Matrisi
f, ax = plt.subplots(figsize=[18, 13])
sns.heatmap(df_train[numeric_columns].corr(), annot=True, fmt=".2f", ax=ax, cmap="flare")
ax.set_title("Correlation Matrix", fontsize=20)
plt.show()


# In[ ]:





# In[27]:


df_train.corr(method = 'pearson',numeric_only = True).unstack().idxmin()


# In[28]:


df_train.corr(method = 'pearson',numeric_only = True).unstack().idxmax() 


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# # veri görselleştirme

# In[76]:


def draw_boxplots_for_numeric_columns(df, figsize=(10, 5), palette="Set2"):
    """
    Tüm numerik değişkenler için kutu grafiği çizer.

    Args:
        df (pd.DataFrame): Veri çerçevesi (DataFrame).
        figsize (tuple): Grafik boyutu (default: (10, 5)).
        palette (str): Grafik renk paleti (default: "Set2").
    """

    for col in numeric_columns:
        plt.figure(figsize=figsize)
        sns.boxplot(data=df, x=col, palette=palette)
        plt.title(f'Boxplot for {col}')
        plt.xlabel('')
        plt.ylabel('Values')
        plt.show()
        
draw_boxplots_for_numeric_columns(df_train)      


# Kutu grafiklerinde aykırı değerler görünse de bilgi kaybetmemek adına şimdilik dokunmayacağım, ileride kullanılacak kompleks modeller zaten yüksek oranda baş edebiliyor aykırı değerler ile.

# In[78]:


from itertools import combinations

def numcols_target_corr(df, target="tanı"):

    
    # Sayısal sütunların ikili kombinasyonlarını oluştur
    numvar_combinations = list(combinations(numeric_columns, 2))
    
    # Her kombinasyon için scatter plot oluştur
    for item in numvar_combinations:
        plt.subplots(figsize=(14, 8))
        sns.scatterplot(x=df[item[0]], y=df[item[1]], hue=df[target], palette="Set2")\
            .set_title(f'{item[0]}   &   {item[1]}')
        plt.grid(True)
        plt.show()

# Fonksiyonu çağır
numcols_target_corr(df_train, target="tanı")


# In[324]:


# Sayısal - Sayısal İlişki: Scatterplot
sns.pairplot(df_train[['sıgarayıbırakannekadarGÜNıcmıs','sıgarabırakangundekacadetıcmıs',
                       'nezamanbırakmısGÜN',]], diag_kind='kde', corner=True)

plt.show()


# In[88]:


sns.pairplot(df_train[['FEV1','PEF','FEV1 %','PEF %',]], diag_kind='kde', corner=True)
plt.show()


# In[89]:


# Kategorik - Kategorik İlişki: Çapraz Tablo
kategori_tablosu = pd.crosstab(df_train['tanı'], df_train['sıgarakullanımı'])
print(kategori_tablosu)

sns.heatmap(kategori_tablosu, annot=True, cmap='YlGnBu', fmt='d')
plt.title('tanı ve sigara kullanımı İlişkisi')
plt.show()


# In[333]:


# Kategorik - Kategorik İlişki: Çapraz Tablo
kategori_tablosu = pd.crosstab(df_train['cınsıyet'], df_train['tanı'])
print(kategori_tablosu)

sns.heatmap(kategori_tablosu, annot=True, cmap='pink', fmt='d')
plt.title('tanı ve cinsiyet kullanımı İlişkisi')
plt.show()


# In[90]:


#çok değişkenli analiz: kategorik-sayısal

for col in ['nabız', 'solunumsayısı']:
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df_train, x='tanı', y=col, palette='coolwarm')
    plt.title(f'{col} by tanı')
    plt.show()


# In[66]:


import itertools
# İkili kombinasyonlar
combinations = list(itertools.combinations(df_train[numeric_columns], 2))


# In[67]:


# Density plot çizimi
for x, y in combinations:
    plt.figure(figsize=(6, 4))
    sns.kdeplot(data=df_train, x=x, y=y, cmap="Blues", fill=True)
    plt.title(f'Density Plot: {x} vs {y}')
    plt.show()


# In[339]:


import seaborn as sns
import matplotlib.pyplot as plt

# Grafik boyutu
plt.figure(figsize=(15, 8))

# Örnek sütunlardan birini seçelim
sns.violinplot(data=df_train, x="tanı", y="YAŞ", palette="coolwarm")

# Başlık ve eksen etiketleri
plt.title("tanıya Göre YAŞ Dağılımı (Violin Plot)", fontsize=14)
plt.xlabel("tanı")
plt.ylabel("YAŞ")

# Grafiği göster
plt.show()


# In[172]:


df_train.columns


# ## base model kurumu1

# In[64]:


train_copy2= df_train.copy()
test_copy2=df_test.copy()


# ## one-hot encoding
import pandas as pd
# Eğitim kümesindeki kategorik değişkenler
def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe


# Eğitim setinde one-hot encoding
train_copy2 = one_hot_encoder(train2, categorical_columns, drop_first=True)

# Test setinde one-hot encoding
test_copy2 = one_hot_encoder(test2, categorical_columns, drop_first=True)

train_copy2.head(3)test_copy2.head(3)from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import numpy as np

# Orijinal sütunlar: Kategorik sütunlar
categorical_columns = ['varsakımde ANNE', 'varsakımde BABA', 'varsakımde KARDES', 'varsakımde DİĞER']

# "bilgiyok" değerlerini -1 ile değiştirme
for col in categorical_columns:
    train_copy2[col] = train_copy2[col].replace('bilgiyok', -1)
    test_copy2[col] = test_copy2[col].replace('bilgiyok', -1)

# Kategorik sütunların veri tipini 'str' yapma (OneHotEncoder ile uyum için)
train_copy2[categorical_columns] = train_copy2[categorical_columns].astype(str)
test_copy2[categorical_columns] = test_copy2[categorical_columns].astype(str)

# OneHotEncoder oluşturma
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

# Eğitim setinde fit ve transform işlemleri
train_encoded = encoder.fit_transform(train_copy2[categorical_columns])

# Test setinde sadece transform işlemi
test_encoded = encoder.transform(test_copy2[categorical_columns])

# Yeni sütun isimlerini belirleme
encoded_columns = encoder.get_feature_names_out(categorical_columns)

# Dönüştürülen veriyi DataFrame'e çevirme
train_encoded_df = pd.DataFrame(train_encoded, columns=encoded_columns)
test_encoded_df = pd.DataFrame(test_encoded, columns=encoded_columns)

# Orijinal index'leri koruma
train_encoded_df.index = train_copy2.index
test_encoded_df.index = test_copy2.index

# Orijinal veri setlerine OneHotEncoded sütunları ekleme
train_copy2 = pd.concat([train_copy2.drop(columns=categorical_columns), train_encoded_df], axis=1)
test_copy2 = pd.concat([test_copy2.drop(columns=categorical_columns), test_encoded_df], axis=1)

# Sonuç kontrolü
print("Eğitim seti şekli:", train_copy2.shape)
print("Test seti şekli:", test_copy2.shape)

# #### aykırı değer 
def outlier_thresholds(dataframe, col_name, q1=0.05, q3=0.95):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False
    
for col in numeric_columns:
    print(col, check_outlier(train_copy2, col))
    # Aykırı değerlerin baskılanması
def replace_with_thresholds(dataframe, variable, q1=0.05, q3=0.95):
    low_limit, up_limit = outlier_thresholds(dataframe, variable, q1=0.05, q3=0.95)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit
    
# Aykırı Değer Analizi 
for col in numeric_columns:
    print(col, check_outlier(train_copy2,col))
    if check_outlier(train_copy2, col):
        replace_with_thresholds(train_copy2, col)# Aykırı Değer Analizi -test
for col in numeric_columns:
    print(col, check_outlier(test_copy2,col))
    if check_outlier(test_copy2, col):
        replace_with_thresholds(test_copy2, col)### yeni değişken üretme def feature_engineering(df):
    # Tanı toplam süresini ay olarak hesapla
    df['tanı_toplam_sure_ay'] = df['tanısuresıyıl'] * 12 + df['tanısuresıay']
    
    # Toplam yatış sayısını hesapla
    df['toplam_yatış_sayısı'] = (
        df['acılservıseyatıssayısı'] + 
        df['yogumbakımayatıssayısı'] + 
        df['servıseyatıssayısı']
    )
    
    # Toplam yatış süresini gün olarak hesapla
    df['toplam_yatış_suresi'] = (
        df['acılservısetoplamyatıssuresısaat'] / 24 +  # Saat cinsinden olanı güne çevir
        df['acilservistoplamyatışsüresigün'] +
        df['yogumbakımatoplamyatıssüresıgun'] +
        df['servisetoplamyatıssüresıgun']
    )
    
    
    
    df['solunum_toplam'] = df[['FEV1', 'PEF', 'FEV1/FVC Değeri']].sum(axis=1)
    df['gunluk_sigara_ortalama'] = df[['sıgarabırakangundekacadetıcmıs', 'sıgarayadevamedengundekacadetıcıyo']].mean(axis=1)
    df['pef_fev1_orani'] = df['PEF'] / df['FEV1']

    
    return df
# Eğitim kümesi üzerinde dönüşüm
train_copy2 = feature_engineering(train_copy2)

# Test kümesi üzerinde dönüşüm
test_copy2 = feature_engineering(test_copy2)

# Kontrol
train_copy2.head()
# ## z skore
# Sayısal sütunları seçme
#standard = train_copy2.select_dtypes(include=['int64', 'float64']).columns

# StandardScaler oluşturma
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

# Eğitim setinde fit ve transform
train_copy2[numeric_columns] = scaler.fit_transform(train_copy2[numeric_columns])

# Test setinde aynı dönüşümü uygulama
test_copy2[numeric_columns] = scaler.transform(test_copy2[numeric_columns])

# Kontrol
print("Eğitim Verisi Standartlaştırılmış İlk Satırlar:")
train_copy2.head()
#Z-skore yerine min-max deneyelim
# Sayısal sütunları seçme
standard = train_copy2.select_dtypes(include=['int64', 'float64']).columns

# StandardScaler oluşturma
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

# Eğitim setinde fit ve transform
train_copy2[standard] = scaler.fit_transform(train_copy2[standard])

# Test setinde aynı dönüşümü uygulama
test_copy2[standard] = scaler.transform(test_copy2[standard])

# Kontrol
print("Eğitim Verisi Standartlaştırılmış İlk Satırlar:")
train_copy2.head()from sklearn.preprocessing import OneHotEncoder
import pandas as pd
# 'sütun_adi' sütunundaki "bilgiyok" değerini -1 ile değiştir

# OneHotEncoder oluşturma
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

# Kategorik sütunları belirleme
categorical_columns0= ['varsakımde ANNE', 'varsakımde BABA', 'varsakımde KARDES', 'varsakımde DİĞER']
#categorical_columns = train_copy2.select_dtypes(include=['object']).columns.tolist()
categorical_columns0 =[col for col in categorical_columns0 ] .replace('bilgiyok', -1).astype(int)

train_copy2[categorical_columns] = train_copy2[categorical_columns].astype('str')
test_copy2[categorical_columns] = test_copy2[categorical_columns].astype('str')

# Sadece eğitim seti ile fit etme
train_encoded = encoder.fit_transform(train_copy2[categorical_columns])

# Test setini dönüştürme (transform)
test_encoded = encoder.transform(test_copy2[categorical_columns])

# Sonuçları DataFrame olarak dönüştürme
train_encoded_df = pd.DataFrame(train_encoded, columns=encoder.get_feature_names_out(categorical_columns))
test_encoded_df = pd.DataFrame(test_encoded, columns=encoder.get_feature_names_out(categorical_columns))

# Orijinal veri setine birleştirme
train_copy2 = pd.concat([train_copy2.reset_index(drop=True), train_encoded_df.reset_index(drop=True)], axis=1)
test_copy2 = pd.concat([test_copy2.reset_index(drop=True), test_encoded_df.reset_index(drop=True)], axis=1)

# Orijinal kategorik sütunları kaldırma
train_copy2.drop(columns=categorical_columns, inplace=True)
test_copy2.drop(columns=categorical_columns, inplace=True)
train_copy2.head(2)
# (label_encoder), yalnızca 2 değeri olan kategorik (binary) değişkenleri sayısal verilere dönüştürür.
# İkinci fonksiyon (one_hot_encoder), birden fazla kategoriye sahip sütunları One-Hot Encoding yöntemiyle dönüştürür, yani her kategoriye karşılık gelen yeni sütunlar ekler.

# In[47]:


#!pip install lightgbm
#!pip install catboost
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor  
from catboost import CatBoostRegressor


# Log dönüşümünün gerçekleştirilmesi
X = train_copy2.drop(columns=["tanı"], errors="ignore")  # 'tanı' hedef değişken ve 'basvurutarıhı' tarih
y = train_copy2["tanı"]

# Verinin eğitim ve tet verisi olarak bölünmesi
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=17)

models = [('LR', LinearRegression()),
          ("Ridge", Ridge()),
          ("Lasso", Lasso()),
          ("ElasticNet", ElasticNet()),
          ('KNN', KNeighborsRegressor()),
          ('CART', DecisionTreeRegressor()),
          ('RF', RandomForestRegressor()),
          #('SVR', SVR()),
          ('GBM', GradientBoostingRegressor()),
          #("XGBoost", XGBRegressor(objective='reg:squarederror')),
          #("LightGBM", LGBMRegressor())]
          ("CatBoost", CatBoostRegressor(verbose=False))]

for name, regressor in models:
    rmse = np.mean(np.sqrt(-cross_val_score(regressor, X, y, cv=5, scoring="neg_mean_squared_error")))
    print(f"RMSE: {round(rmse, 4)} ({name}) ")


# standartlaştırma yapınca modeller biraz daha iyi performans gösterdi

# ## MODELLEMELER

# In[89]:


train_model=df_train.copy()
test_model=df_test.copy()


# In[90]:


def feature_engineering(df):
    # Tanı toplam süresini ay olarak hesapla
    df['tanı_toplam_sure_ay'] = df['tanısuresıyıl'] * 12 + df['tanısuresıay']
    
    # Toplam yatış sayısını hesapla
    df['toplam_yatış_sayısı'] = (
        df['acılservıseyatıssayısı'] + 
        df['yogumbakımayatıssayısı'] + 
        df['servıseyatıssayısı'] )
    
    # Toplam yatış süresini gün olarak hesapla
    df['toplam_yatış_suresi'] = (
        df['acılservısetoplamyatıssuresısaat'] / 24 +  # Saat cinsinden olanı güne çevir
        df['acilservistoplamyatışsüresigün'] +
        df['yogumbakımatoplamyatıssüresıgun'] +
        df['servisetoplamyatıssüresıgun'])
    
    df['solunum_toplam'] = df[['FEV1', 'PEF', 'FEV1/FVC Değeri']].sum(axis=1)
    df['gunluk_sigara_ortalama'] = df[['sıgarabırakangundekacadetıcmıs', 'sıgarayadevamedengundekacadetıcıyo']].mean(axis=1)
    df['pef_fev1_orani'] = df['PEF'] / df['FEV1']

    return df
# Eğitim kümesi üzerinde dönüşüm
train_model = feature_engineering(train_model)

# Test kümesi üzerinde dönüşüm
test_model = feature_engineering(test_model)

# Kontrol
train_model.head()


# In[91]:


test_model.head(3)


# In[92]:


import pandas as pd
# Eğitim kümesindeki kategorik değişkenler
onehot_columns = ['varsakımde ANNE', 'varsakımde BABA', 'varsakımde KARDES', 'varsakımde DİĞER']

def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe

# Eğitim setinde one-hot encoding
train_model = one_hot_encoder(train_model, onehot_columns ,drop_first=True)

# Test setinde one-hot encoding
test_model = one_hot_encoder(test_model, onehot_columns, drop_first=True)

train_model.head(3)


# In[93]:


# Sayısal sütunları seçme
#standard = train_copy2.select_dtypes(include=['int64', 'float64']).columns

# StandardScaler oluşturma
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

# Eğitim setinde fit ve transform
train_model[numeric_columns] = scaler.fit_transform(train_model[numeric_columns])

# Test setinde aynı dönüşümü uygulama
test_model[numeric_columns] = scaler.transform(test_model[numeric_columns])

# Kontrol
print("Eğitim Verisi Standartlaştırılmış İlk Satırlar:")
train_model.head()


# In[94]:


#!pip install yellowbrick

import numpy as np
import warnings
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler, RobustScaler

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, GridSearchCV, cross_validate, cross_val_score, validation_curve

from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.impute import KNNImputer


from sklearn.exceptions import ConvergenceWarning

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action="ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)


# In[98]:


# Sınıflama modelleri için skorlama yapar.

def score_me(y_test, predicts):
    print('Accuracy: ', metrics.accuracy_score(y_test, predicts))
    print('Precision: ', metrics.precision_score(y_test, predicts))
    print('Recall: ', metrics.recall_score(y_test, predicts))
    
#ROC Curve çizdirir ve AUC Skoru verir.

def plot_roc_curve(y_test, y_pred_prob):
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
    auc_score = roc_auc_score(y_test, y_pred_prob)
    
    plt.plot(fpr, tpr, label = 'data 1, AUC= ' + str(auc_score))
    plt.legend(loc=4)
    plt.show()


# In[95]:


#Veri setini hazırlama
x = train_model.drop(columns=["tanı"], errors="ignore")  # Bağımsız değişkenler
y = train_model["tanı"]  # Bağımlı değişken

# Eğitim ve test kümelerine ayırma
x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size=0.2, random_state=17)


# ### KNN 

# In[110]:


def give_me_knn(neighbor):
    knn = KNeighborsClassifier(n_neighbors = neighbor)
    
    knn.fit(x_train, y_train)
    
    return (knn.score(x_valid, y_valid), neighbor)


# In[111]:


accuracy = []

for i in range(1, 20):
    accuracy.append(give_me_knn(i))

accuracy_df = pd.DataFrame(accuracy, columns = ['Score', 'Neighbor'])

accuracy_df = accuracy_df.sort_values(by = 'Score', ascending = False)
accuracy_df


# In[157]:


knn = KNeighborsClassifier(n_neighbors = 11)
knn.fit(x_train, y_train)
knn_best=knn.score(x_valid, y_valid)
knn_best


# In[116]:


rmse = np.sqrt(mean_squared_error(y_valid, y_pred))
rmse


# In[139]:


from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
knn_predicts = knn.predict(x_valid)
conf_matrix = confusion_matrix(y_valid, knn_predicts)
cmdisplay = ConfusionMatrixDisplay(conf_matrix, display_labels = ['0', '1'])
cmdisplay.plot()


# In[117]:


knn_predict_proba = knn.predict_proba(x_valid)
knn_predict_proba_one = knn_predict_proba[:, 1]
plot_roc_curve(y_valid, knn_predict_proba_one)


# In[102]:


#2.yol
# . Veriyi ölçeklendirme (KNN için gereklidir)
#scaler = StandardScaler()
#X_train_scaled = scaler.fit_transform(x_train)
#X_valid_scaled = scaler.transform(x_valid)

# 3. KNN modeli oluşturma ve en iyi K değerini bulma
knn = KNeighborsRegressor()

# GridSearchCV ile K parametresi için optimizasyon
param_grid = {'n_neighbors': range(1, 21)}  # 1'den 20'ye kadar K değerleri denenecek
grid_search = GridSearchCV(knn, param_grid, cv=5, scoring='neg_mean_squared_error', verbose=1)
grid_search.fit(X_train_scaled, y_train)

# En iyi K değerini alma
best_k = grid_search.best_params_['n_neighbors']
print(f"En iyi K değeri: {best_k}")

# 4. En iyi K değeri ile modelin eğitilmesi
knn_best = KNeighborsRegressor(n_neighbors=best_k)
knn_best.fit(X_train_scaled, y_train)

# 5. Modelin test verisinde değerlendirilmesi
y_pred = knn_best.predict(X_valid_scaled)
rmse = np.sqrt(mean_squared_error(y_valid, y_pred))

print(f"Test RMSE: {rmse:.4f}")


# In[140]:


print(classification_report(y_valid, knn_predicts))


# In[120]:


#Scores DataFrame Initializing
classification_results = pd.DataFrame([], columns = ['Accuracy Score'])

#Adding First Results into results DF
classification_results.loc['K-Nearest Neighbor'] = accuracy_df['Score'].iloc[0]
classification_results


# ### Naive Bayes

# In[159]:


nb = GaussianNB()

nb.fit(x_train, y_train)

nb.score(x_valid, y_valid)

classification_results.loc['Naive Bayes'] = nb.score(x_valid, y_valid)
classification_results


# In[141]:


nb_predicts = nb.predict(x_valid)

conf_matrix = confusion_matrix(y_valid, nb_predicts)
cmdisplay = ConfusionMatrixDisplay(conf_matrix, display_labels = ['1', '2'])
cmdisplay.plot()


# In[131]:


mse = mean_squared_error(y_valid, nb_predicts)  # Mean Squared Error
rmse = np.sqrt(mse)
rmse


# In[132]:


print(classification_report(y_valid, nb_predicts))


# In[126]:


nb_predict_proba = nb.predict_proba(x_valid)
nb_predict_proba_one = nb_predict_proba[:, 1]
plot_roc_curve(y_valid, nb_predict_proba_one)


# ## Logistic Regression

# In[162]:


logreg = LogisticRegression(max_iter = 10000, penalty = None)

logreg.fit(x_train, y_train)

logreg.score(x_valid, y_valid)

classification_results.loc['Logistic Regression'] = logreg.score(x_valid, y_valid)
classification_results


# In[142]:


logreg_predicts = logreg.predict(x_valid)

conf_matrix = confusion_matrix(y_valid, logreg_predicts)
cmdisplay = ConfusionMatrixDisplay(conf_matrix, display_labels = ['1', '2'])
cmdisplay.plot()


# In[135]:


mse = mean_squared_error(y_valid, logreg_predicts)  # Mean Squared Error
rmse = np.sqrt(mse)
rmse


# In[136]:


print(classification_report(y_valid, logreg_predicts))


# In[138]:


logreg_predict_proba = logreg.predict_proba(x_valid)
logreg_predict_proba_one = logreg_predict_proba[:, 1]
plot_roc_curve(y_valid, logreg_predict_proba_one)


# ### Decision Tree Classifier

# In[164]:


dtc = DecisionTreeClassifier(random_state=17)

dtc.fit(x_train, y_train)

dtc.score(x_valid, y_valid)

classification_results.loc['Decision Tree Classifier'] = dtc.score(x_valid, y_valid)
#classification_results
dtc.score(x_valid, y_valid)


# In[165]:


dtc_predicts = dtc.predict(x_valid)
conf_matrix = confusion_matrix(y_valid, dtc_predicts)
cmdisplay = ConfusionMatrixDisplay(conf_matrix, display_labels = ['1', '2'])
cmdisplay.plot()


# In[166]:


mse = mean_squared_error(y_valid, dtc_predicts)  # Mean Squared Error
rmse = np.sqrt(mse)
rmse


# In[167]:


print(classification_report(y_valid, dtc_predicts))


# In[168]:


dtc_predict_proba = dtc.predict_proba(x_valid)
dtc_predict_proba_one = dtc_predict_proba[:, 1]
plot_roc_curve(y_valid, dtc_predict_proba_one)


# ## Random Forest Classifier

# In[170]:


rfc = RandomForestClassifier(random_state=17)

rfc.fit(x_train, y_train)

rfc.score(x_valid, y_valid)

classification_results.loc['Random Forest Classifier'] = rfc.score(x_valid, y_valid)

#classification_results
rfc.score(x_valid, y_valid)


# In[173]:


rfc_predicts = rfc.predict(x_valid)

conf_matrix = confusion_matrix(y_valid, rfc_predicts)
cmdisplay = ConfusionMatrixDisplay(conf_matrix, display_labels = ['1', '2'])
cmdisplay.plot()


# In[174]:


mse = mean_squared_error(y_valid, rfc_predicts )  # Mean Squared Error
rmse = np.sqrt(mse)
rmse


# In[175]:


print(classification_report(y_valid, rfc_predicts))


# In[176]:


rfc_predict_proba = rfc.predict_proba(x_valid)
rfc_predict_proba_one = rfc_predict_proba[:, 1]
plot_roc_curve(y_valid, rfc_predict_proba_one)


# ## Bagging Classifier (RFC)

# In[180]:


rfc = RandomForestClassifier()
bag = BaggingClassifier(rfc, n_estimators=100, random_state=17,oob_score=True)

bag.fit(x_train, y_train)

classification_results.loc['Bagging Classifier (RFC)'] = bag.score(x_valid, y_valid)

classification_results


# In[181]:


bag_predicts = bag.predict(x_valid)

conf_matrix = confusion_matrix(y_valid, bag_predicts)
cmdisplay = ConfusionMatrixDisplay(conf_matrix, display_labels = ['1', '2'])
cmdisplay.plot()


# In[182]:


mse = mean_squared_error(y_valid, bag_predicts )  # Mean Squared Error
rmse = np.sqrt(mse)
rmse


# In[183]:


print(classification_report(y_valid, bag_predicts))


# In[184]:


bag_predict_proba = bag.predict_proba(x_valid)
bag_predict_proba_one = bag_predict_proba[:, 1]
plot_roc_curve(y_valid, bag_predict_proba_one)


# In[ ]:





# ## Gradient Boosting Classifier

# In[186]:


gbc = GradientBoostingClassifier(learning_rate = 0.01, n_estimators = 5000)

gbc.fit(x_train, y_train)

classification_results.loc['Gradient Boosting Classifier'] = gbc.score(x_valid, y_valid)
classification_results


# In[187]:


gbc_predicts = gbc.predict(x_valid)

conf_matrix = confusion_matrix(y_valid, gbc_predicts)
cmdisplay = ConfusionMatrixDisplay(conf_matrix, display_labels = ['1', '2'])
cmdisplay.plot()


# In[188]:


mse = mean_squared_error(y_valid, gbc_predicts )  # Mean Squared Error
rmse = np.sqrt(mse)
rmse


# In[189]:


print(classification_report(y_valid, gbc_predicts))


# In[190]:


gbc_predict_proba = gbc.predict_proba(x_valid)
gbc_predict_proba_one = gbc_predict_proba[:, 1]
plot_roc_curve(y_valid, gbc_predict_proba_one)


# ## catboost

# In[199]:


from catboost import CatBoostRegressor
from sklearn.model_selection import GridSearchCV, cross_val_score
import numpy as np

# CatBoostRegressor modelini başlatıyoruz
catboost_model = CatBoostRegressor(random_state=17, verbose=False)

# Hiperparametre  grid
catboost_params = {"iterations": [200, 500],
                   "learning_rate": [0.01, 0.1],
                   "depth": [3, 6]}

# GridSearchCV ile en iyi parametreyi buluyoruz
catboost_best_grid = GridSearchCV(catboost_model, catboost_params, cv=5, n_jobs=-1, verbose=True).fit(x, y)

# En iyi parametreleri alıyoruz ve final modelimizi oluşturuyoruz
catboost_final = catboost_model.set_params(**catboost_best_grid.best_params_, random_state=17).fit(x, y)

# Modelin R² skoru (başarı ölçütü)
r2_score = catboost_final.score(x, y)

# RMSE (Root Mean Squared Error) hesaplama
rmse = np.mean(np.sqrt(-cross_val_score(catboost_final, x, y, cv=10, scoring="neg_mean_squared_error")))

# Sonuçları yazdıralım
print(f"Best Parameters: {catboost_best_grid.best_params_}")
print(f"R² Score: {r2_score}")
print(f"RMSE: {rmse}")


# In[202]:


classification_results.loc['catboost'] = r2_score
classification_results


# # AdaBoostClassifier
# 

# In[208]:


ada = AdaBoostClassifier(learning_rate = 0.01, n_estimators = 5000)

ada.fit(x_train, y_train)

classification_results.loc['Ada Boosting Classifier'] = ada.score(x_valid, y_valid)
classification_results


# In[204]:


ada_predicts = ada.predict(x_valid)

conf_matrix = confusion_matrix(y_valid, ada_predicts)
cmdisplay = ConfusionMatrixDisplay(conf_matrix, display_labels = ['1', '2'])
cmdisplay.plot()


# In[207]:


mse = mean_squared_error(y_valid, ada_predicts )  # Mean Squared Error
rmse = np.sqrt(mse)
rmse


# In[205]:


print(classification_report(y_valid, ada_predicts))


# In[206]:


ada_predict_proba = ada.predict_proba(x_valid)
ada_predict_proba_one = ada_predict_proba[:, 1]
plot_roc_curve(y_valid, ada_predict_proba_one)


# In[ ]:





# In[209]:


classification_results.sort_values(by = 'Accuracy Score', ascending = False)


# In[210]:


# En iyi parametrelerle final model
catboost_final = CatBoostRegressor(**catboost_best_grid.best_params_, random_state=17)
catboost_final.fit(x_train, y_train)


# In[233]:


import pandas as pd
from catboost import CatBoostClassifier

# 1. CatBoost Modelini Sınıflandırma için Eğit
catboost_final = CatBoostClassifier(**catboost_best_grid.best_params_, random_state=17, verbose=False)
catboost_final.fit(x, y)  # Eğitim verisi ile model eğitimi

# 2. Test Verisini Ön İşleme (Aynı şekilde işlem uygulandı)
X_test_processed = preprocess.transform(test_model)  # Test verisini aynı şekilde işleme

# 3. Test Verisi Üzerinde Tahmin Yap
y_test_predictions = catboost_final.predict(X_test_processed)  # Modelin tahmin ettiği sınıflar (1 veya 2)

# 4. Test Veri Setine Tahmin Sonuçlarını Ekleyin
test_model["Predictions"] = y_test_predictions  # Tahmin edilen sonuçları ekle

# 5. Yeni bir DataFrame oluşturun ve sadece "Hastano" ve "Predictions" sütunlarını seçin
odev = pd.DataFrame({
    "Hastano": test["hastaNo"],  # Hastane numarası
    "Tanı": test_model["Predictions"]  # Modelin tahmin ettiği tanı (1 veya 2)
})

# 6. Yeni Excel dosyasına kaydetme
odev.to_excel("odev_predictions.xlsx", index=False)

# Dosyayı kaydettikten sonra ilk birkaç satırı kontrol edebilirsiniz
print(odev.head())


# In[234]:


import os
print(os.getcwd())


# In[235]:


odev.to_excel(r"C:\Users\HANDENUR\Downloads\odev_predictions.xlsx", index=False)


# In[ ]:




