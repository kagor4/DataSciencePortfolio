#!/usr/bin/env python
# coding: utf-8

# ### Прогнозирование успешности стартапов с помощью машинного обучения  
# 
# ## Описание проекта  
# 
# **Цель:** Разработка модели машинного обучения для предсказания вероятности закрытия стартапов на основе исторических данных.  
# 
# **Особенности задачи:**  
# - Работа с псевдо-реальными данными (1980-2018 гг.)  
# - Участие в Kaggle-соревновании  
# - Практическое применение методов Data Science  
# 
# ## Ключевые аспекты  
# 
# **Данные:**  
# - Исторические показатели стартапов  
# - Признаки деятельности компаний  
# - Целевая переменная: факт закрытия (бинарная классификация)  
# 
# **Техническая реализация:**  
# - Предобработка данных:  
#   - Работа с пропусками  
#   - Кодирование категориальных признаков  
#   - Feature engineering  
# - Моделирование:  
#   - Классические ML-алгоритмы (Random Forest, XGBoost)  
#   - Оптимизация гиперпараметров  
#   - Ансамблирование моделей  
# - Оценка:  
#   - Метрика ROC-AUC  
#   - Feature importance analysis  
# 
# **Стек технологий:**  
# - Python (Pandas, Scikit-learn)  
# - XGBoost/LightGBM/CatBoost  
# - Matplotlib/Seaborn для визуализации  
# 
# ## Практическая ценность  
# 
# **Применение результатов:**  
# - Для инвесторов: оценка рисков венчурных инвестиций  
# - Для основателей: выявление ключевых факторов успеха  
# - Для акселераторов: отбор перспективных проектов  
# 
# > **Примечание:** Проект демонстрирует навыки работы с Kaggle и решение задач бинарной классификации в условиях соревнования.  

# In[ ]:


get_ipython().run_line_magic('pip', 'install -q phik')


# In[541]:


get_ipython().run_line_magic('pip', 'install -q category_encoders')


# In[542]:


get_ipython().run_line_magic('pip', 'install -q optuna')


# In[543]:


import pandas as pd
import os
import numpy as np
import optuna
from datetime import datetime
import matplotlib.pyplot as plt
import phik
import seaborn as sns
from sklearn.utils import shuffle
from sklearn.dummy import DummyClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler, OrdinalEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score, recall_score, roc_curve
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, cross_val_predict
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import make_column_selector, make_column_transformer, ColumnTransformer
from sklearn.impute import SimpleImputer 


# Создадим функции для визуализации распределения данных и построения ящика с усами.

# In[544]:


def unique_distribution_plots(data, name, title):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Уникальное распределение ' + name, fontweight='bold', fontsize=20)
    
    sns.distplot(data, ax=axes[0], color='skyblue', hist_kws={'edgecolor':'black'})
    axes[0].set_ylabel('Плотность')
    axes[0].set_title(title, fontsize=18)
    
    sns.boxplot(data, ax=axes[1], color='salmon')
    axes[1].set_title(title, fontsize=18)
    
    plt.show()


# In[545]:


def unique_distribution(data, name):
    columns = [i for i in df_train.columns if (df_train[i].nunique() > 2) & (df_train[i].dtypes != 'object')]
    for column in columns:
        unique_distribution_plots(data[column], name, column)


# Создам функцию для апсемплинга

# In[546]:


def upsample(features, target, repeat):
    features_zeros = features[target == 0]
    features_ones = features[target == 1]
    target_zeros = target[target == 0]
    target_ones = target[target == 1]

    features_upsampled = pd.concat([features_ones] + [features_zeros] * repeat)
    target_upsampled = pd.concat([target_ones] + [target_zeros] * repeat)
    
    features_upsampled, target_upsampled = shuffle(
        features_upsampled, target_upsampled, random_state=12345)
    
    return features_upsampled, target_upsampled


# Создам функцию для сверки результатов

# In[547]:


def evaluate_model(model, features, target):
    prediction = model.predict(features)
    proba_one = model.predict_proba(features)[:, 1]
    fpr, tpr, threshold = roc_curve(target, proba_one)
    plot_roc_auc(fpr, tpr)
    
    print("Accuracy:", round(accuracy_score(target, prediction), 3))
    print("F1:", round(f1_score(target, prediction), 3))
    print("Precision:", round(precision_score(target, prediction), 3))
    print("Recall:", round(recall_score(target, prediction), 3))
    
    # Построение ROC-кривой
    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc_score(target, proba_one))
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.show()
    
    return prediction


# Создам функцию для постройки ROC-AUC графика

# In[548]:


def plot_roc_auc(fpr, tpr):
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label='ROC curve')
    plt.plot([0, 1], [0, 1], 'k--', label='Random guess')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.show()


# Создадим функцию графика кросс-валидации, чтобы оценить стабильность и надежность результатов нашей модели. Это позволит убедиться, что полученные метрики не являются случайными и отражают устойчивое поведение модели на различных блоках данных.

# In[549]:


def plot_cross_val_recall(best_model, features, target):
    cv_recall_scores = cross_val_score(best_model, features, target, cv=10, scoring='recall', n_jobs=-1)
    plt.plot(cv_recall_scores)
    plt.title('Recall Scores across Cross-validation Folds')
    plt.xlabel('Fold')
    plt.ylabel('Recall Metric')
    plt.xticks(range(10))
    plt.show()


# In[550]:


def optimize_decision_tree(trial):
    criterion = trial.suggest_categorical('criterion', ['entropy'])
    max_depth = trial.suggest_int('max_depth', 2, 10)
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 30, 60, 10)
    model_dt = DecisionTreeClassifier(random_state=1, criterion=criterion, 
                                     max_depth=max_depth, 
                                     min_samples_leaf=min_samples_leaf,
                                     class_weight='balanced'
                                     )
    model_dt.fit(features_train, target_train)
    
    trial.set_user_attr(key="best_booster", value=model_dt)
    
    cross_valid = cross_val_score(model_dt, features_train, target_train, cv=5, scoring='recall', n_jobs=-1).mean()
    return cross_valid

def study_callback(study, trial):
    if study.best_trial.number == trial.number:
        study.set_user_attr(key="best_booster", value=trial.user_attrs["best_booster"])


# In[551]:


def optimize_random_forest(trial):
    criterion = trial.suggest_categorical('criterion', ['entropy'])
    n_estimators = trial.suggest_int('n_estimators', 3, 9, 3)
    max_depth = trial.suggest_int('max_depth', 25, 30)
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 10, 50, 10)
    rf_model = RandomForestClassifier(random_state=1, criterion=criterion, 
                                     max_depth=max_depth, 
                                     n_estimators=n_estimators, 
                                     min_samples_leaf=min_samples_leaf,
                                     class_weight='balanced'
                                     )
    rf_model.fit(features_train, target_train)
    trial.set_user_attr(key="best_booster", value=rf_model)
    cross_valid = cross_val_score(rf_model, features_train, target_train, cv=5, scoring='recall', n_jobs=-1).mean()
    return cross_valid

def study_callback(study, trial):
    if study.best_trial.number == trial.number:
        study.set_user_attr(key="best_booster", value=trial.user_attrs["best_booster"])


# In[552]:


data_train = 'C:/Users/Егор/Downloads/kaggle_startups_train_01.csv'

if os.path.exists(data_train):
    df_train = pd.read_csv(data_train)
else:
    print('Something is wrong')


# In[553]:


data_test = 'C:/Users/Егор/Downloads/kaggle_startups_test_01.csv'

if os.path.exists(data_test):
    df_test = pd.read_csv(data_test)
else:
    print('Something is wrong')


# In[554]:


data_test_2nd = 'C:/Users/Егор/Downloads/kaggle_startups_test_01.csv'

if os.path.exists(data_test_2nd):
    df_test_2nd = pd.read_csv(data_test_2nd)
else:
    print('Something is wrong')


# In[555]:


df_train.head()


# In[556]:


df_test.head()


# In[557]:


df_train.info()


# In[558]:


df_train = df_train.astype({'funding_total_usd': 'float32'})
df_train = df_train.astype({'funding_rounds': 'int32'})
df_train.info()


# In[559]:


df_test = df_test.astype({'funding_total_usd': 'float32'})
df_test = df_test.astype({'funding_rounds': 'int32'})
df_test.info()


# In[560]:


df_train.dropna(subset=['name', 'first_funding_at'], inplace=True)
df_test.dropna(subset=['name', 'first_funding_at'], inplace=True)


# In[561]:


df_train.describe()


# In[562]:


df_train.duplicated().sum()


# In[563]:


df_test.duplicated().sum()


# In[564]:


df_train.shape


# In[565]:


df_train['category_list'].unique()


# In[566]:


len(df_train['category_list'].unique())


# Промежуточный вывод:
# Явных дубликатов и аномалий в данных не обнаружил. 
# 
# 
# Займемся разделением категория стартапов и изменим тип столбца статуса. Я решил выделить 10 категорий стартапов по присутствующим словам в столбце category_list

# In[567]:


df_train['status'] = df_train['status'].replace({'operating': 1, 'closed': 0})


# In[568]:


data_train = df_train['status']
data_train = df_train.drop(['status'], axis=1)


# In[569]:


technology_and_it = ['technology', 'it', 'cyber'],
healthcare_and_medicine = ['health', 'medical', 'healthcare', 'medicine', 'diabetes', 'nutrition', 'neuroscience'],
industry = ['manufacturing', 'industrial', 'mining', 'chemicals'],
finance_and_business = ['financial', 'business', 'insurance', 'banking'],
education = ['education', 'college', 'learning', 'teaching', 'edu', 'stem'],
retail_trade = ['retail', 'shopping', 'e-commerce'],
entertainment_and_games = ['entertainment', 'games', 'gaming', 'video_games', 'online_gaming', 'casual_games'],
media_and_communication = ['media', 'communication', 'social_media', 'broadcasting'],
tourism_and_travel = ['tourism', 'travel', 'hospitality'],
energy_and_ecology = ['energy', 'green', 'solar', 'wind', 'carbon', 'clean_technology']


# In[570]:


df_train['category_list'] = df_train['category_list'].fillna('')
df_test['category_list'] = df_test['category_list'].fillna('')


# Создам новые категории

# In[571]:


df_train['technology_and_it'] = 0
tech_and_it_regex = '|'.join('|'.join(words) for words in technology_and_it)
df_train.loc[df_train['category_list'].str.contains(tech_and_it_regex), 'technology_and_it'] = 1

df_test['technology_and_it'] = 0
tech_and_it_regex = '|'.join('|'.join(words) for words in technology_and_it)
df_test.loc[df_test['category_list'].str.contains(tech_and_it_regex), 'technology_and_it'] = 1


# In[572]:


df_train['healthcare_and_medicine'] = 0
healthcare_and_medicine_regex = '|'.join('|'.join(words) for words in healthcare_and_medicine)
df_train.loc[df_train['category_list'].str.contains(healthcare_and_medicine_regex), 'healthcare_and_medicine'] = 1

df_test['healthcare_and_medicine'] = 0
healthcare_and_medicine_regex = '|'.join('|'.join(words) for words in healthcare_and_medicine)
df_test.loc[df_test['category_list'].str.contains(healthcare_and_medicine_regex), 'healthcare_and_medicine'] = 1


# In[573]:


df_train['industry'] = 0
industry_regex = '|'.join('|'.join(words) for words in industry)
df_train.loc[df_train['category_list'].str.contains(industry_regex), 'industry'] = 1

df_test['industry'] = 0
industry_regex = '|'.join('|'.join(words) for words in industry)
df_test.loc[df_test['category_list'].str.contains(industry_regex), 'industry'] = 1


# In[574]:


df_train['finance_and_business'] = 0
finance_and_business_regex = '|'.join('|'.join(words) for words in finance_and_business)
df_train.loc[df_train['category_list'].str.contains(finance_and_business_regex), 'finance_and_business'] = 1

df_test['finance_and_business'] = 0
finance_and_business_regex = '|'.join('|'.join(words) for words in finance_and_business)
df_test.loc[df_test['category_list'].str.contains(finance_and_business_regex), 'finance_and_business'] = 1


# In[575]:


df_train['education'] = 0
education_regex = '|'.join('|'.join(words) for words in education)
df_train.loc[df_train['category_list'].str.contains(education_regex), 'education'] = 1

df_test['education'] = 0
education_regex = '|'.join('|'.join(words) for words in education)
df_test.loc[df_test['category_list'].str.contains(education_regex), 'education'] = 1


# In[576]:


df_train['retail_trade'] = 0
retail_trade_regex = '|'.join('|'.join(words) for words in retail_trade)
df_train.loc[df_train['category_list'].str.contains(retail_trade_regex), 'retail_trade'] = 1

df_test['retail_trade'] = 0
retail_trade_regex = '|'.join('|'.join(words) for words in retail_trade)
df_test.loc[df_test['category_list'].str.contains(retail_trade_regex), 'retail_trade'] = 1


# In[577]:


df_train['entertainment_and_games'] = 0
entertainment_and_games_regex = '|'.join('|'.join(words) for words in entertainment_and_games)
df_train.loc[df_train['category_list'].str.contains(entertainment_and_games_regex), 'entertainment_and_games'] = 1

df_test['entertainment_and_games'] = 0
entertainment_and_games_regex = '|'.join('|'.join(words) for words in entertainment_and_games)
df_test.loc[df_test['category_list'].str.contains(entertainment_and_games_regex), 'entertainment_and_games'] = 1


# In[578]:


df_train['media_and_communication'] = 0
media_and_communication_regex = '|'.join('|'.join(words) for words in media_and_communication)
df_train.loc[df_train['category_list'].str.contains(media_and_communication_regex), 'media_and_communication'] = 1

df_test['media_and_communication'] = 0
media_and_communication_regex = '|'.join('|'.join(words) for words in media_and_communication)
df_test.loc[df_test['category_list'].str.contains(media_and_communication_regex), 'media_and_communication'] = 1


# In[579]:


df_train['tourism_and_travel'] = 0
tourism_and_travel_regex = '|'.join('|'.join(words) for words in tourism_and_travel)
df_train.loc[df_train['category_list'].str.contains(tourism_and_travel_regex), 'tourism_and_travel'] = 1

df_test['tourism_and_travel'] = 0
tourism_and_travel_regex = '|'.join('|'.join(words) for words in tourism_and_travel)
df_test.loc[df_test['category_list'].str.contains(tourism_and_travel_regex), 'tourism_and_travel'] = 1


# In[580]:


df_train['energy_and_ecology'] = 0
energy_and_ecology_regex = '|'.join('|'.join(words) for words in energy_and_ecology)
df_train.loc[df_train['category_list'].str.contains(energy_and_ecology_regex), 'energy_and_ecology'] = 1

df_test['energy_and_ecology'] = 0
energy_and_ecology_regex = '|'.join('|'.join(words) for words in energy_and_ecology)
df_test.loc[df_test['category_list'].str.contains(energy_and_ecology_regex), 'energy_and_ecology'] = 1


# In[581]:


df_train.drop('category_list', axis=1, inplace=True)
df_test.drop('category_list', axis=1, inplace=True)


# Посчитаем сколько дней открыт стартап, а так же сколько дней прошло с момента открытия до первого и последнего раунда финансирования

# In[582]:


df_train['founded_at'] = pd.to_datetime(df_train['founded_at'])
df_train['closed_at'] = pd.to_datetime(df_train['closed_at'])
df_train['first_funding_at'] = pd.to_datetime(df_train['first_funding_at'])
df_train['last_funding_at'] = pd.to_datetime(df_train['last_funding_at'])
today = datetime.now()
df_train['closed_at'].fillna(today, inplace=True)
df_train['days_open'] = (df_train['closed_at'] - df_train['founded_at']).dt.days
df_train['first_funding_days'] = (df_train['first_funding_at'] - df_train['founded_at']).dt.days
df_train['last_funding_days'] = (df_train['last_funding_at'] - df_train['founded_at']).dt.days
df_train['first_funding_days'] = df_train['first_funding_days'].astype('Int32')
df_train.drop(['founded_at', 'closed_at', 'first_funding_at', 'last_funding_at'], axis=1, inplace=True)
df_train.head()


# In[583]:


df_test['founded_at'] = pd.to_datetime(df_test['founded_at'])
df_test['closed_at'] = pd.to_datetime(df_test['closed_at'])
df_test['first_funding_at'] = pd.to_datetime(df_test['first_funding_at'])
df_test['last_funding_at'] = pd.to_datetime(df_test['last_funding_at'])
today = datetime.now()
df_test['closed_at'].fillna(today, inplace=True)
df_test['days_open'] = (df_test['closed_at'] - df_test['founded_at']).dt.days
df_test['first_funding_days'] = (df_test['first_funding_at'] - df_test['founded_at']).dt.days
df_test['last_funding_days'] = (df_test['last_funding_at'] - df_test['founded_at']).dt.days
df_test['first_funding_days'] = df_test['first_funding_days'].astype('Int32')
df_test.drop(['founded_at', 'closed_at', 'first_funding_at', 'last_funding_at'], axis=1, inplace=True)
df_test.head()


# In[584]:


df_train.info()


# В следующем этапе анализа мы рассмотрим целевой признак "status". Проведем проверку на наличие выбросов, оценим его распределение и баланс классов.

# In[585]:


status_counts = df_train['status'].value_counts()
plt.figure(figsize=(8, 8))
status_counts.plot(kind='pie', autopct='%1.1f%%', colors=['lightblue', 'lightgreen'])
plt.title('Распределение целевого признака')
plt.ylabel('')  # Убираем подпись оси y
plt.legend(['Работает', 'Закрыт'], loc='best')  # Добавляем легенду
plt.show()


# Используем OrdinalEncoder для кодирования данных

# In[586]:


cat_columns = df_train.select_dtypes(include='object').columns.tolist()
num_columns = df_train.select_dtypes(include='number').columns.tolist()
cat_columns


# In[587]:


# Создаем экземпляр OrdinalEncoder
encoder = OrdinalEncoder()

# Применяем OrdinalEncoder только к категориальным столбцам
df_train = df_train.copy()
df_train[cat_columns] = encoder.fit_transform(df_train[cat_columns])

df_test = df_test.copy()
df_test[cat_columns] = encoder.fit_transform(df_test[cat_columns])


# In[588]:


df_train.isna().sum()


# На этом этапе нужно избавить от NaN. Столбцы 'country_code', 'state_code', 'region', 'city' я заполню уникальными числовыми значениями, которые еще не встречались в этом столбце. Столбец funding_total_usd я принял решение заполнить средним значением по каждой стране, в том случае если данных по стране нет, то заполняем просто средним.

# In[589]:


columns_to_replace = ['country_code', 'state_code', 'region', 'city']

unique_numeric_values = set(range(1, len(df_train) + 1))

for column in columns_to_replace:
    unique_values = df_train[column].dropna().unique()
    
    unused_numeric_value = min(unique_numeric_values - set(unique_values))
    
    df_train[column].fillna(unused_numeric_value, inplace=True)
    
df_train.isna().sum()


# In[590]:


mean_funding_by_country = df_train.groupby('country_code')['funding_total_usd'].mean()

df_train['funding_total_usd'] = df_train.apply(
    lambda row: mean_funding_by_country[row['country_code']] if pd.notna(row['funding_total_usd']) else np.nan,
    axis=1
)

overall_mean = df_train['funding_total_usd'].mean()
df_train['funding_total_usd'].fillna(overall_mean, inplace=True)


# In[591]:


columns_to_replace = ['country_code', 'state_code', 'region', 'city']

unique_numeric_values = set(range(1, len(df_test) + 1))

for column in columns_to_replace:
    unique_values = df_test[column].dropna().unique()
    
    unused_numeric_value = min(unique_numeric_values - set(unique_values))
    
    df_test[column].fillna(unused_numeric_value, inplace=True)
    
df_test.isna().sum()


# In[592]:


mean_funding_by_country = df_test.groupby('country_code')['funding_total_usd'].mean()

df_test['funding_total_usd'] = df_test.apply(
    lambda row: mean_funding_by_country[row['country_code']] if pd.notna(row['funding_total_usd']) else np.nan,
    axis=1
)

overall_mean = df_test['funding_total_usd'].mean()
df_test['funding_total_usd'].fillna(overall_mean, inplace=True)


# In[593]:


df_train.shape


# In[594]:


plt.figure(figsize=(10, 6))

phik_matrix = df_train.phik_matrix()


sns.heatmap(
    phik_matrix[['status']].sort_values('status', ascending=False),
    annot=True, annot_kws={"size": 12}, cmap='Blues'
)
plt.title('hotel_train', fontweight="bold")
plt.show()


# In[595]:


filtered_columns = df_train.columns[(df_train.min() != 0) | (df_train.max() != 1)]

phik_matrix = df_train[filtered_columns].phik_matrix()

plt.figure(figsize=(10, 8))
sns.heatmap(phik_matrix, annot=True, fmt=".2f", cmap='coolwarm')
plt.title('Матрица корреляции Phik')
plt.xlabel('Признаки')
plt.ylabel('Признаки')
plt.show()


# Исходя из Phik Matrix, можно сделать следующие выводы:
# 
# - Переменная "status" сильно коррелирует с переменными "energy_and_ecology", "state_code", "city" и "region". Это означает, что эти переменные могут оказывать значительное влияние на статус объекта.
# - Переменные "funding_rounds", "country_code" и "days_open" имеют умеренную корреляцию с "status". Это может указывать на то, что количество раундов финансирования, код страны и количество открытых дней также могут оказывать влияние на статус объекта.
# - Остальные переменные имеют незначительную или отсутствующую корреляцию с "status", что может означать их менее значимость при предсказании статуса объекта.

# In[596]:


unique_distribution(df_train, 'df_train')


# Проведем скалирование и апсемплинг

# In[597]:


df_train, df_target = df_train.drop(columns='status', axis=1), df_train['status']


# In[598]:


num_columns.remove('status')

for column in num_columns:
    unique_values = df_train[column].unique()
    if len(unique_values) == 2 and 0 in unique_values and 1 in unique_values:
        num_columns.remove(column)

scaler = StandardScaler()

df_train = df_train.copy()
df_train[num_columns] = scaler.fit_transform(df_train[num_columns])

df_test = df_test.copy()
df_test[num_columns] = scaler.fit_transform(df_test[num_columns])


# In[599]:


df_target.value_counts()


# In[600]:


df_train, df_target = upsample(df_train, df_target, 10)


# In[601]:


df_train.shape[0] / df_test.shape[0]


# In[602]:


df_target.value_counts()


# In[624]:


features_train, features_test, target_train, target_test = train_test_split(df_train, df_target, test_size=0.2, random_state=1234)


# In[604]:


target_train = np.squeeze(target_train)
target_test = np.squeeze(target_test)


# In[631]:


features_train.shape


# In[632]:


target_train.shape


# In[633]:


features_test.shape


# In[634]:


target_test.shape


# In[635]:


random_model = DummyClassifier(random_state=1)
random_model.fit(features_train, target_train)
predictions_random = random_model.predict(features_test)

random_proba = random_model.predict_proba(features_test)[:, 1]

print("Accuracy:", accuracy_score(target_test, predictions_random))
print("F1:", f1_score(target_test, predictions_random))
print("ROC_auc:", roc_auc_score(target_test, random_proba))

fpr_random, tpr_random, threshold_random = roc_curve(target_test, random_proba)


# In[636]:


dummy_predictions = evaluate_model(random_model, features_train, target_train)


# Исследуем логистическую регрессию

# In[637]:


lr_model = LogisticRegression(random_state=1, solver='liblinear', n_jobs=-1, class_weight='balanced')
params = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100],
    'penalty': ['l1', 'l2']
}
grid_search = GridSearchCV(lr_model, params, cv=5, scoring='recall')
grid_search.fit(features_train, target_train)

best_lr_model = grid_search.best_estimator_
best_params = grid_search.best_params_
best_score = grid_search.best_score_

print("Лучшие параметры:", best_params)
print("Лучший Recall на кросс-валидации:", best_score)

train_recall = best_lr_model.score(features_train, target_train)
print(f'Лучший Recall на тренировочной выборке: {train_recall}')


# In[612]:


lr_model.fit(features_train, target_train)
prediction_lr = evaluate_model(lr_model, features_train, target_train)


# In[613]:


plot_cross_val_recall(lr_model, features_test, target_test)


# In[614]:


lr_model.class_weight


# Исследуем дерево решений

# In[615]:


study = optuna.create_study(direction='maximize')
study.optimize(optimize_decision_tree, n_trials=100, callbacks=[study_callback])

best_dt_model = study.best_trial.user_attrs['best_booster']
best_dt_params = study.best_params

cross_val_recall = cross_val_score(best_dt_model, features_train, target_train, cv=5, scoring='recall').mean()
print('Recall:', cross_val_recall)

print('Лучшие параметры:', best_dt_params)


# In[616]:


best_dt = DecisionTreeClassifier(criterion='entropy', max_depth=8, min_samples_leaf=30, random_state=1, class_weight='balanced')
best_dt.fit(features_train, target_train)
prediction_dt = evaluate_model(best_dt, features_train, target_train)


# In[617]:


plot_cross_val_recall(best_dt, features_test, target_test)


# Исследуем модель случайного леса

# In[618]:


study_rf = optuna.create_study(direction='maximize')
study_rf.optimize(optimize_random_forest, n_trials=50, callbacks=[study_callback])
best_rf_model = study_rf.best_trial.user_attrs["best_booster"]

print('Recall:', study_rf.best_value, 'с параметрами:', study_rf.best_params)


# In[619]:


prediction_rf = evaluate_model(best_rf_model, features_train, target_train)


# In[620]:


plot_cross_val_recall(best_rf_model, features_test, target_test)


# Проведен анализ и сравнение эффективности трех моделей - логистической регрессии, дерева решений и случайного леса. Все три модели показали хорошие результаты на тренировочной выборке.
# 
# Логистическая регрессия:
# Accuracy: 1.0
# F1: 1.0
# Precision: 1.0
# Recall: 1.0
# 
# Дерево решений:
# Accuracy: 0.997
# F1: 0.997
# Precision: 0.993
# Recall: 1.0
# 
# Случайный лес:
# Accuracy: 1.0
# F1: 1.0
# Precision: 1.0
# Recall: 1.0
# 
# Кросс-валидация показала следующие результаты:
# Логистическая регрессия: Recall: 1
# Дерево решений: Recall: 1
# Случайный лес: Recall: 1
# 
# Получились крайне высокие показатели, но я не смог найти утечку или в где допустил ошибку. Был бы рад если подсказали мне.

# In[638]:


predictions = best_dt.predict(df_test)


# In[639]:


df_test_2nd = df_test_2nd.loc[:, ['name']]
df_test_2nd.head()


# In[640]:


predictions_df = pd.DataFrame(predictions, index=df_test_2nd.index, columns=['prediction'])
result = df_test_2nd.join(predictions_df)
result['prediction'] = result['prediction'].replace({1: 'operating', 0: 'closed'})
result.head()

