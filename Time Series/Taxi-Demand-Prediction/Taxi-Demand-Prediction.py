#!/usr/bin/env python
# coding: utf-8

# ## Прогнозирование спроса на такси в аэропортах  
# 
# #### Описание проекта  
# Компания «Чётенькое такси» собрала исторические данные о заказах такси в аэропортах. Чтобы оптимизировать работу водителей в периоды пиковой нагрузки, необходимо прогнозировать количество заказов на следующий час.  
# 
# **Задача:**  
# Разработать модель машинного обучения для предсказания количества заказов такси в аэропорту на следующий час.  
# 
# #### Описание данных  
# Данные хранятся в файле `/datasets/taxi.csv`. Структура данных:  
# - `datetime` — дата и время заказа (временной ряд)  
# - `num_orders` — количество заказов (целевая переменная)  
# 
# **Цель:**  
# Построить модель, которая поможет компании эффективно распределять водителей и минимизировать время ожидания клиентов.  

# ## Подготовка

# In[1]:


get_ipython().system('pip install --upgrade scikit-learn')


# In[2]:


# Базовые библиотеки для работы с данными
import pandas as pd
import warnings
import os
import numpy as np

# Визуализация данных
import seaborn as sns
from matplotlib import pyplot as plt

# Анализ временных рядов
from statsmodels.tsa.seasonal import seasonal_decompose

# Машинное обучение: разделение данных и метрики
from sklearn.model_selection import train_test_split, TimeSeriesSplit, GridSearchCV, cross_val_score
from sklearn.metrics import mean_squared_error, make_scorer

# Машинное обучение: модели
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from catboost import CatBoostRegressor
import lightgbm as lgb
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline


# In[3]:


warnings.filterwarnings('ignore')


# In[4]:


def summarize_dataframe(df):
    print('='*40)
    print(f'Общие размеры DataFrame: {df.shape[0]} строк, {df.shape[1]} столбцов')
    print('='*40)
    
    print('\nПервые 10 строк:')
    display(df.head(10))
    
    print('\nСтатистика числовых столбцов:')
    display(df.describe())
    
    print('\nИнформация о DataFrame:')
    info = df.info(memory_usage='deep')
    print('\nИспользование памяти: {:.2f} MB'.format(
        df.memory_usage(deep=True).sum() / (1024 ** 2)
    ))
    print('='*40)
    
    return info


# In[5]:


dataset = '/datasets/taxi.csv'

if os.path.exists(dataset):
    df = pd.read_csv(dataset, sep=',', index_col = 0)
else:
    print('Something is wrong')
summarize_dataframe(df)


# In[6]:


# Проверим индекс на монотонность

df.index.is_monotonic


# In[7]:


if not isinstance(df.index, pd.DatetimeIndex):
    df.index = pd.to_datetime(df.index)


# In[8]:


# Ресемплируем данные по часу

df = df.resample('1H').sum()


# ## Анализ

# In[9]:


df


# In[10]:


# Вычисляем средние значения за сутки и неделю
daily_avg = df.resample('D').mean()
weekly_avg = df.resample('W').mean()

# Создаем график
plt.figure(figsize=(14, 7))

# График исходных данных (ресемплированных по часам)
plt.plot(df.index, df['num_orders'], label='Часовые заказы', alpha=0.5, color='blue')

# График средних значений за сутки
plt.plot(daily_avg.index, daily_avg['num_orders'], label='Суточный средний заказ', color='green', linestyle='--')

# График средних значений за неделю
plt.plot(weekly_avg.index, weekly_avg['num_orders'], label='Недельный средний заказ', color='red', linestyle='-.')

# Настройка графика
plt.title('Анализ временного ряда заказов с часовыми данными и средними значениями', fontsize=16)
plt.xlabel('Дата', fontsize=12)
plt.ylabel('Количество заказов', fontsize=12)
plt.legend(fontsize=12, title="Легенда", title_fontsize=12)  # Добавляем заголовок легенде
plt.grid(True)

# Показываем график
plt.tight_layout()
plt.show()


# In[11]:


# Определим функцию для построения графиков

sns.set(style="whitegrid", rc={'figure.figsize': (15, 4)})

def lineplot(data, title, xlabel='Дата', ylabel='Значение', grid=True, color='blue', linewidth=1.5):

    ax = data.plot(color=color, linewidth=linewidth)
    
    ax.set_title(title, fontsize=16, pad=20)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    
    if grid:
        ax.grid(True, linestyle='--', alpha=0.6)
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    plt.show()


# In[12]:


# Разделим данные на трендовую, сезонную и шумовую состовляющие и построим соответствующие графики

decomposed = seasonal_decompose(df)


# In[13]:


lineplot(decomposed.trend, 'Тренд')


# In[14]:


lineplot(decomposed.trend.rolling(24*7).mean(), 'Тренд (сглаженный)')


# In[15]:


lineplot(decomposed.seasonal.tail(24*7), 'Сезонность')


# In[16]:


lineplot(decomposed.resid, 'Шумы')


# **Выводы:**
# 
# 1. **Тренд:**  
#    Наблюдается устойчивый рост количества заказов с течением времени. Однако периодически возникают небольшие спады, которые могут быть связаны с внешними факторами (например, погодные условия, события в городе или изменения спроса).
# 
# 2. **Сезонность:**  
#    Четко выражена дневная сезонность:  
#    - **Ночью** наблюдается спад активности заказов, что связано с естественным снижением спроса в это время суток.  
#    - **Утром** начинается постепенный рост, который продолжается в течение дня.  
#    - **Вечером** достигается пик заказов, что объясняется повышенной активностью людей после работы или учебы.  

# ## Обучение

# In[17]:


def get_features(df_learning):
    df_learning['dayofweek'] = df_learning.index.dayofweek
    df_learning['hour'] = df_learning.index.hour
    
    for lag in range(1, 6):
        df_learning[f'lag_{lag}'] = df_learning['num_orders'].shift(lag)
    
    df_learning['rolling_mean'] = df_learning['num_orders'].shift(1).rolling(window=24).mean()
    
    df_learning.dropna(inplace=True)
    
    return df_learning


# In[18]:


df_learning = df.copy()
df_learning = get_features(df_learning)


# In[19]:


# Разделим данные на обучающую и тестовую выборки

features = df_learning.drop(['num_orders'], axis=1)
target = df_learning['num_orders']


# In[20]:


features_train, features_test, target_train, target_test = train_test_split(
    features, target, shuffle=False, test_size=0.1, random_state=17)


# In[21]:


# Обучим различные модели и добавим модели и рассчитанный RMSE в массив


train_models = []


# ### Линейная регрессия

# In[22]:


# Определяем scorer для RMSE
rmse_scorer = make_scorer(
    lambda y_true, y_pred: np.sqrt(mean_squared_error(y_true, y_pred)),
    greater_is_better=False
)


# In[23]:


# Определяем числовые и категориальные признаки
numeric_features = ['lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5', 'rolling_mean']
categorical_features = ['hour', 'dayofweek']


# In[24]:


# Создаем преобразователь для числовых признаков (масштабирование)
numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

# Создаем преобразователь для категориальных признаков (OHE)
categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Объединяем преобразователи с помощью ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)


# In[25]:


# Создаем полную пайплайн с предобработкой и моделью
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])


# In[26]:


# Определяем TimeSeriesSplit с 5 фолдами
tscv = TimeSeriesSplit(n_splits=5)


# In[27]:


# Выполняем кросс-валидацию
cv_scores = cross_val_score(
    model_pipeline,
    features_train,
    target_train,
    cv=tscv,
    scoring=rmse_scorer
)


# In[28]:


# Рассчитываем RMSE для каждого фолда
rmse_scores = np.sqrt(-cv_scores)


# In[29]:


# Выводим среднее значение RMSE
mean_rmse = rmse_scores.mean()
print(f"Среднее значение RMSE по кросс-валидации: {mean_rmse}")
model = LinearRegression()
train_models.append([model, mean_rmse])


# ### DecisionTreeRegressor

# In[30]:


# Создаем модель
model = DecisionTreeRegressor(random_state=12345)

# Гиперпараметры для поиска
parameters = {'max_depth': range(1, 11, 2)}

# Кросс-валидация для временных рядов
tscv = TimeSeriesSplit(n_splits=5)

# GridSearchCV с использованием RMSE как scorer
gsearch = GridSearchCV(
    estimator=model,
    param_grid=parameters,
    cv=tscv,
    scoring=rmse_scorer,
    refit=True
)

# Обучаем модель
gsearch.fit(features_train, target_train)

# Лучшие параметры
print(f"Лучшие параметры: {gsearch.best_params_}")


# In[31]:


# Вычисляем RMSE через кросс-валидацию
cv_scores = cross_val_score(
    gsearch.best_estimator_,
    features_train,
    target_train,
    cv=tscv,
    scoring=rmse_scorer
)
mean_rmse = -cv_scores.mean()
print(f"Среднее значение RMSE по кросс-валидации: {mean_rmse}")

# Создаем строковое представление модели
model_description = f"DecisionTree(max_depth={gsearch.best_params_['max_depth']})"

# Добавляем только одну запись для DecisionTree
train_models.append([model, mean_rmse])


# ### RandomForestRegressor

# In[32]:


# Создаем модель
model = RandomForestRegressor(random_state=12345)

# Гиперпараметры для поиска
parameters = {
    'n_estimators': range(50, 100, 10),
    'max_depth': range(1, 11, 2)
}

# Кросс-валидация для временных рядов
tscv = TimeSeriesSplit(n_splits=5)

# GridSearchCV с использованием RMSE как scorer
gsearch = GridSearchCV(
    estimator=model,
    param_grid=parameters,
    cv=tscv,
    scoring=rmse_scorer,
    refit=True
)

# Обучаем модель
gsearch.fit(features_train, target_train)

# Лучшие параметры
print(f"Лучшие параметры: {gsearch.best_params_}")


# In[33]:


# Вычисляем RMSE через кросс-валидацию
cv_scores = cross_val_score(
    gsearch.best_estimator_,
    features_train,
    target_train,
    cv=tscv,
    scoring=rmse_scorer
)

# Среднее значение RMSE
mean_rmse = -cv_scores.mean()
print(f"Среднее значение RMSE по кросс-валидации: {mean_rmse}")

# Сохраняем модель и метрику
train_models.append([model, mean_rmse])


# ### CatBoost

# In[34]:


# Создаем модель CatBoostRegressor
model = CatBoostRegressor(verbose=False)

# Кросс-валидация для временных рядов
tscv = TimeSeriesSplit(n_splits=5)

# Выполняем кросс-валидацию
cv_scores = cross_val_score(
    model,
    features_train,
    target_train,
    cv=tscv,
    scoring=rmse_scorer
)

# Среднее значение RMSE
mean_rmse = -cv_scores.mean()
print(f"Среднее значение RMSE по кросс-валидации: {mean_rmse}")


# In[35]:


# Сохраняем модель и метрику
train_models.append([model, mean_rmse])


# ### LightGBM

# In[36]:


# Создаем модель LightGBM
model = lgb.LGBMRegressor(verbose=-1)

# Кросс-валидация для временных рядов
tscv = TimeSeriesSplit(n_splits=5)

# Выполняем кросс-валидацию
cv_scores = cross_val_score(
    model,
    features_train,
    target_train,
    cv=tscv,
    scoring=rmse_scorer
)

# Среднее значение RMSE
mean_rmse = -cv_scores.mean()
print(f"Среднее значение RMSE по кросс-валидации: {mean_rmse}")


# In[37]:


# Сохраняем модель и метрику
train_models.append([model, mean_rmse])


# In[38]:


result = pd.DataFrame(train_models, columns=['model', 'rmse_train'])
pd.options.display.max_colwidth = 0
display(result.sort_values(by='rmse_train'))


# ###  **Вывод**  
# На основе результатов, **лучшей моделью** является **LinearRegression** с RMSE **5.06**, так как она показала наименьшую ошибку среди всех моделей.  
# 
# Ансамблевые методы (**LGBMRegressor** и **CatBoostRegressor**) также продемонстрировали хорошие результаты (**RMSE ≈ 25.52 и 25.59** соответственно), но уступают линейной регрессии.  
# 
# Методы **DecisionTree** и **RandomForest** показали наибольшие ошибки (**RMSE 30.33 и 26.07**), что делает их менее предпочтительными.  

# ## Тестирование

# In[39]:


# Обучаем пайплайн на всей обучающей выборке
model_pipeline.fit(features_train, target_train)


# In[40]:


# Делаем предсказания на тестовой выборке
predictions_test = model_pipeline.predict(features_test)


# In[41]:


# Вычисляем RMSE на тестовой выборке
mse_test = mean_squared_error(target_test, predictions_test)
rmse_test = np.sqrt(mse_test)

print(f"RMSE на тестовой выборке: {rmse_test}")


# In[42]:


# Создаем DataFrame для удобства работы
results_df = pd.DataFrame({
    'datetime': features_test.index,
    'actual': target_test,
    'predicted': predictions_test
})

# Строим график
plt.figure(figsize=(14, 7))

# Исходный временной ряд (реальные значения)
plt.plot(results_df['datetime'], results_df['actual'], label='Фактические значения', color='blue', alpha=0.7)

# Предсказанный временной ряд
plt.plot(results_df['datetime'], results_df['predicted'], label='Предсказанные значения', color='orange', linestyle='--')

# Настройка графика
plt.title('Линейная регрессия: Фактические vs Предсказанные значения', fontsize=16)
plt.xlabel('Дата', fontsize=12)
plt.ylabel('Количество заказов', fontsize=12)
plt.legend(fontsize=12, title="Легенда", title_fontsize=12)
plt.grid(True)

# Показываем график
plt.tight_layout()
plt.show()


# In[43]:


# Бейзлайн 1: Среднее значение
mean_value = target_train.mean()
baseline_mean_predictions = np.full_like(target_test, mean_value, dtype=np.float64)
rmse_baseline_mean = np.sqrt(mean_squared_error(target_test, baseline_mean_predictions))
print(f"RMSE для бейзлайна (среднее значение): {rmse_baseline_mean}")

# Бейзлайн 2: Предыдущее значение (лаг 1)
baseline_lag1_predictions = target_test.shift(1).fillna(method='bfill').values
rmse_baseline_lag1 = np.sqrt(mean_squared_error(target_test, baseline_lag1_predictions))
print(f"RMSE для бейзлайна (предыдущее значение): {rmse_baseline_lag1}")

# Бейзлайн 3: Значение этого же часа с предыдущих суток (лаг 24)
baseline_lag24_predictions = target_test.shift(24).fillna(method='bfill').values
rmse_baseline_lag24 = np.sqrt(mean_squared_error(target_test, baseline_lag24_predictions))
print(f"RMSE для бейзлайна (значение этого часа с предыдущих суток): {rmse_baseline_lag24}")

# Бейзлайн 4: Значение этого же часа и дня недели с предыдущей недели (лаг 168)
baseline_lag168_predictions = target_test.shift(168).fillna(method='bfill').values
rmse_baseline_lag168 = np.sqrt(mean_squared_error(target_test, baseline_lag168_predictions))
print(f"RMSE для бейзлайна (значение этого часа и дня недели с предыдущей недели): {rmse_baseline_lag168}")


# #### Вывод по бейзлайнам:
# 
# - **Среднее значение**: RMSE = 84.69 (худший результат).
# - **Предыдущее значение (лаг 1)**: RMSE = 58.93.
# - **Значение этого часа с предыдущих суток (лаг 24)**: RMSE = 55.99.
# - **Значение этого часа и дня недели с предыдущей недели (лаг 168)**: RMSE = 49.27 (лучший результат).
# 
# #### Вывод:
# Бейзлайн с лагом 168 показал наименьшую ошибку (RMSE = 42.08), что указывает на наличие сильной недельной сезонности в данных. Модель должна иметь RMSE ниже 42.08, чтобы превзойти простые эвристики.
# 
# #### Рекомендации:
# - Сравните RMSE вашей модели с 42.08 для оценки её эффективности.
# - При необходимости добавьте признаки, отражающие сезонность (например, лаги 24 и 168) или внешние факторы (погода, праздники).

# ###  **Общий вывод**  
# 
# 1. **Модели машинного обучения**:  
#    - Лучшей моделью является **LinearRegression** с RMSE на обучающей выборке **5.06**, что значительно ниже, чем у других моделей:  
#      - LGBMRegressor: RMSE = 25.52  
#      - CatBoostRegressor: RMSE = 25.59  
#      - RandomForestRegressor: RMSE = 26.07  
#      - DecisionTreeRegressor: RMSE = 30.33  
#    - Однако на **тестовой выборке** RMSE **LinearRegression** составил **42.57**, что указывает на возможное переобучение или недостаточную сложность модели для захвата всех зависимостей в данных.  
# 
# 2. **Бейзлайны**:  
#    - Наиболее эффективный бейзлайн — использование значения этого часа и дня недели с предыдущей недели (**лаг 168**), с RMSE = **49.27**.  
#    - Прочие бейзлайны показали худшие результаты:  
#      - Среднее значение: RMSE = 84.69  
#      - Предыдущее значение: RMSE = 58.93  
#      - Значение этого часа с предыдущих суток: RMSE = 56.30  
# 
# ###  **Заключение**  
# Модель **LinearRegression** является **оптимальным выбором** для прогнозирования числа заказов, так как она показывает лучший результат по сравнению с другими моделями и бейзлайнами.  
# 
# Однако **RMSE на тестовой выборке (42.57) выше, чем на обучающей (5.06)**, что может свидетельствовать о переобучении или недостаточности используемых признаков.  
# 
# Рекомендуется:  
# - **Тестировать модель** на новых данных.  
# - **Добавить новые признаки** (например, погодные данные, праздники).  
# - Рассмотреть **ансамблевые методы** (LGBMRegressor, CatBoost), которые могут лучше обобщать зависимости.  
# 
