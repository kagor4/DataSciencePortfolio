#!/usr/bin/env python
# coding: utf-8

# ### Модель линейной регрессии для прогнозирования стоимости жилья в Калифорнии  
# 
# ## Описание проекта  
# 
# **Цель:** Построение и оценка модели линейной регрессии для предсказания медианной стоимости домов в жилых массивах Калифорнии на основе данных 1990 года.  
# 
# **Ключевые задачи:**  
# - Предобработка данных о недвижимости  
# - Обучение и валидация линейной регрессии  
# - Оценка качества модели по трем метрикам  
# - Интерпретация результатов  
# 
# ## Данные  
# 
# **Датасет содержит следующие признаки:**  
# - Географические: `longitude`, `latitude`, `ocean_proximity`  
# - Демографические: `housing_median_age`, `population`, `households`  
# - Характеристики жилья: `total_rooms`, `total_bedrooms`  
# - Экономические: `median_income`  
# 
# **Целевая переменная:**  
# - `median_house_value` - медианная стоимость дома  
# 
# ## Реализация  
# 
# **Технологический стек:**  
# - Python (Pandas, NumPy)  
# - Scikit-learn (LinearRegression)  
# - Matplotlib/Seaborn для визуализации  
# 
# **Основные этапы:**  
# 1. Предобработка данных:  
#    - Обработка пропусков  
#    - Преобразование категориальных признаков  
#    - Масштабирование числовых признаков  
# 
# 2. Обучение модели:  
#    - Разделение на train/test выборки  
#    - Подбор гиперпараметров  
#    - Кросс-валидация  
# 
# 3. Оценка качества:  
#    - RMSE (Root Mean Squared Error)  
#    - MAE (Mean Absolute Error)  
#    - R² (коэффициент детерминации)  
# 
# ## Результаты  
# 
# **Ожидаемые выходы:**  
# - Обученная модель линейной регрессии  
# - Значения метрик качества на тестовой выборке  
# - Визуализация важности признаков  
# - Примеры предсказаний  

# ## Предсказание стоимости жилья
# 
# В проекте вам нужно обучить модель линейной регрессии на данных о жилье в Калифорнии в 1990 году. На основе данных нужно предсказать медианную стоимость дома в жилом массиве. Обучите модель и сделайте предсказания на тестовой выборке. Для оценки качества модели используйте метрики RMSE, MAE и R2.

# # Подготовка данных

# In[1]:


import pandas as pd 
import numpy as np
import pyspark
from pyspark.sql import SparkSession
from pyspark.sql.types import *
import pyspark.sql.functions as F
from pyspark.ml.feature import OneHotEncoder, StringIndexer, StandardScaler, VectorAssembler
from pyspark.ml import Pipeline
from pyspark.ml.regression import LinearRegression
from pyspark.mllib.evaluation import RegressionMetrics


# In[2]:


spark = SparkSession.builder.appName("California Housing").master("local").getOrCreate()


# In[3]:


df = spark.read.load('/datasets/housing.csv', format='csv', sep=',', inferSchema=True,header=True)


# In[4]:


df.printSchema()


# Проверим названия колонок и типы данных.

# In[5]:


print(pd.DataFrame(df.dtypes, columns=['column', 'type']))


# Ознакомимся с внешним видом датафрейма

# In[6]:


pd.DataFrame(df.take(5), columns=df.columns)


# In[7]:


nan = df.select([F.count(F.when(F.isnan(c) | F.col(c).isNull(), c)).alias(c) for c in df.columns]).toPandas().T.rename(columns={0:'NA'})

nan['NA_percents'] = nan['NA'] / df.count() * 100

display(nan)


# Заполню пропуски медианным значением

# In[8]:


total_bedrooms_median = df.approxQuantile('total_bedrooms', [0.5], 0)[0]
df = df.fillna(total_bedrooms_median, subset=['total_bedrooms'])
df.select([F.count(F.when(F.isnan(c) | F.col(c).isNull(), c)).alias(c) for c in df.columns]).toPandas().T.rename(columns={0:'NA'})


# # Обучение моделей

# In[9]:


num_columns = [c for c in df.columns if c != 'ocean_proximity']
cat_features = 'ocean_proximity'
num_features = [c for c in num_columns if c != 'median_house_value']
target = 'median_house_value'


# Преобразование категориальной переменной в числовую с помощью StringIndexer

# In[10]:


indexer = StringIndexer(inputCol="ocean_proximity", outputCol="ocean_proximity_index")
df = indexer.fit(df).transform(df)


# Применение OneHotEncoder к числовой переменной

# In[11]:


encoder = OneHotEncoder(inputCol="ocean_proximity_index", outputCol="ocean_proximity_encoded")

pipeline = Pipeline(stages=[encoder])

df = pipeline.fit(df).transform(df)


# In[12]:


assembler = VectorAssembler(inputCols=["ocean_proximity_encoded"],
                            outputCol="features")

df = df.drop("features")

df = assembler.transform(df)

pd.DataFrame(df.take(3),columns=df.columns)


# In[13]:


numerical_assembler = VectorAssembler(inputCols=num_features,
                                      outputCol='numerical_features')
df = numerical_assembler.transform(df)

standard_scaler = StandardScaler(inputCol='numerical_features',
                                 outputCol='numerical_features_scaled')
df = standard_scaler.fit(df).transform(df)


# Проверим признаки

# In[14]:


df.dtypes


# In[15]:


all_features = ['features', 'numerical_features_scaled']

final_assembler = VectorAssembler(inputCols=all_features,
                                  outputCol='all_features')

df = final_assembler.transform(df)

df.select('all_features').show(3)


# In[16]:


df_train, df_test = df.select(['all_features',
                                 'numerical_features_scaled',
                                 'median_house_value']).randomSplit([.75, .25], seed=1234)
print(f'Обучающая выборка: {df_train.count()} строк, {len(df_train.columns)} столбцов',
      f'\nТестовая выборка:, {df_test.count()} строк, {len(df_test.columns)} столбцов')


# In[17]:


df_table = [['features_used', 'RMSE', 'MAE', 'R2']]

for col in ['all_features', 'numerical_features_scaled']:
    lr = LinearRegression(featuresCol=col, labelCol=target)
    model = lr.fit(df_train)
    
    predictions = model.transform(df_test)
    
    results = predictions.select(['prediction', target])
    
    results_collect = results.collect()
    results_list = [ (float(i[0]), float(i[1])) for i in results_collect]
    scoreAndLabels = spark.sparkContext.parallelize(results_list)
    
    metrics = RegressionMetrics(scoreAndLabels)
    
    df_table.append([col, metrics.rootMeanSquaredError, metrics.meanAbsoluteError, metrics.r2])


# # Анализ результатов

# In[18]:


pd.DataFrame(df_table[1:], columns=df_table[0])


# Результаты моделирования с использованием различных наборов признаков представлены в таблице ниже:
# 
# all_features:
# - RMSE: 68932.663587
# - MAE: 49676.450448
# - R2: 0.630864
# 
# 
# numerical_features_scaled:
# - RMSE: 69653.320654
# - MAE: 50550.706057
# - R2: 0.623106
# 
# Из этих результатов следует, что модель, использующая все доступные признаки (all_features), имеет немного лучшую способность предсказывать целевую переменную, чем модель, использующая только масштабированные числовые признаки (numerical_features_scaled). Это видно по незначительно меньшему значению RMSE и MAE, а также немного выше значению R2 у модели с полным набором признаков.
