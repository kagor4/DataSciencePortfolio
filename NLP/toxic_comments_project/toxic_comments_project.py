#!/usr/bin/env python
# coding: utf-8

# <h1>Содержание<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"><li><span><a href="#Описание-проекта" data-toc-modified-id="Описание-проекта-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Описание проекта</a></span></li><li><span><a href="#Описание-данных" data-toc-modified-id="Описание-данных-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>Описание данных</a></span></li><li><span><a href="#Подготовка" data-toc-modified-id="Подготовка-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>Подготовка</a></span><ul class="toc-item"><li><span><a href="#Итоги-начального-анализа-данных" data-toc-modified-id="Итоги-начального-анализа-данных-3.1"><span class="toc-item-num">3.1&nbsp;&nbsp;</span>Итоги начального анализа данных</a></span></li></ul></li><li><span><a href="#Обучение" data-toc-modified-id="Обучение-4"><span class="toc-item-num">4&nbsp;&nbsp;</span>Обучение</a></span><ul class="toc-item"><li><span><a href="#Выводы-по-выбору-оптимальной-модели" data-toc-modified-id="Выводы-по-выбору-оптимальной-модели-4.1"><span class="toc-item-num">4.1&nbsp;&nbsp;</span>Выводы по выбору оптимальной модели</a></span></li></ul></li><li><span><a href="#Выводы-по-результатам-тестирования-модели" data-toc-modified-id="Выводы-по-результатам-тестирования-модели-5"><span class="toc-item-num">5&nbsp;&nbsp;</span>Выводы по результатам тестирования модели</a></span><ul class="toc-item"><li><span><a href="#Анализ-результатов" data-toc-modified-id="Анализ-результатов-5.1"><span class="toc-item-num">5.1&nbsp;&nbsp;</span>Анализ результатов</a></span></li><li><span><a href="#Рекомендации" data-toc-modified-id="Рекомендации-5.2"><span class="toc-item-num">5.2&nbsp;&nbsp;</span>Рекомендации</a></span></li></ul></li><li><span><a href="#Выводы" data-toc-modified-id="Выводы-6"><span class="toc-item-num">6&nbsp;&nbsp;</span>Выводы</a></span></li><li><span><a href="#Выводы-по-проекту" data-toc-modified-id="Выводы-по-проекту-7"><span class="toc-item-num">7&nbsp;&nbsp;</span>Выводы по проекту</a></span><ul class="toc-item"><li><span><a href="#Выполненные-шаги-для-достижения-цели" data-toc-modified-id="Выполненные-шаги-для-достижения-цели-7.1"><span class="toc-item-num">7.1&nbsp;&nbsp;</span>Выполненные шаги для достижения цели</a></span></li><li><span><a href="#Итог" data-toc-modified-id="Итог-7.2"><span class="toc-item-num">7.2&nbsp;&nbsp;</span>Итог</a></span></li></ul></li></ul></div>

# ## Описание проекта
# 
# Интернет-магазин «Викишоп» запускает новый сервис. Теперь пользователи могут редактировать и дополнять описания товаров, как в вики-сообществах. То есть клиенты предлагают свои правки и комментируют изменения других. Магазину нужен инструмент, который будет искать токсичные комментарии и отправлять их на модерацию.
# 
# **Задача:** Обучить модель классифицировать комментарии на позитивные и негативные. В вашем распоряжении набор данных с разметкой о токсичности правок.
# 
# ## Описание данных
# 
# Данные находятся в файле `/datasets/toxic_comments.csv`. Структура данных:
# 
# - `text` — текст комментария
# - `toxic` — целевой признак (метка токсичности)

# # Проект для «Викишоп»

# Интернет-магазин «Викишоп» запускает новый сервис. Теперь пользователи могут редактировать и дополнять описания товаров, как в вики-сообществах. То есть клиенты предлагают свои правки и комментируют изменения других. Магазину нужен инструмент, который будет искать токсичные комментарии и отправлять их на модерацию.
# 
# Обучите модель классифицировать комментарии на позитивные и негативные. В вашем распоряжении набор данных с разметкой о токсичности правок.
# 
# Постройте модель со значением метрики качества *F1* не меньше 0.75.
# 
# **Инструкция по выполнению проекта**
# 
# 1. Загрузите и подготовьте данные.
# 2. Обучите разные модели.
# 3. Сделайте выводы.
# 
# Для выполнения проекта применять *BERT* необязательно, но вы можете попробовать.
# 
# **Описание данных**
# 
# Данные находятся в файле `toxic_comments.csv`. Столбец *text* в нём содержит текст комментария, а *toxic* — целевой признак.

# ## Подготовка

# In[1]:


# Установка совместимых версий библиотек (выполните один раз)
get_ipython().system('pip install -U scikit-learn>=1.0.0')
get_ipython().system('pip install -U imbalanced-learn')
get_ipython().system('pip install catboost')

# Проверка версий библиотек
import sklearn
import imblearn
import catboost
print(f"scikit-learn version: {sklearn.__version__}")
print(f"imbalanced-learn version: {imblearn.__version__}")
print(f"catboost version: {catboost.__version__}")

# Базовые библиотеки
import numpy as np
import pandas as pd
import os
import time

# Работа с текстом
import re
from pymystem3 import Mystem
import nltk
from nltk import WordNetLemmatizer, pos_tag
from nltk.corpus import wordnet
from nltk.probability import FreqDist
from nltk.tokenize import word_tokenize

# Загрузка ресурсов NLTK
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('punkt')
nltk.download('wordnet')

# Предобработка данных
from sklearn.preprocessing import MaxAbsScaler

# Работа с данными и моделями
from sklearn.model_selection import (
    cross_val_score,
    train_test_split,
    GridSearchCV,
    StratifiedKFold,
    KFold
)
from sklearn.feature_extraction.text import TfidfVectorizer

# Автоматизация обработки признаков
from sklearn.compose import (
    make_column_selector,
    make_column_transformer,
    ColumnTransformer
)

# Pipeline и балансировка классов
try:
    from imblearn.pipeline import Pipeline
    from imblearn.over_sampling import SMOTE
except ImportError as e:
    print(f"Ошибка импорта imblearn: {e}. Убедитесь, что библиотека установлена корректно.")
from sklearn.pipeline import make_pipeline

from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV
from sklearn.model_selection import HalvingRandomSearchCV

# Модели
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier
from sklearn.naive_bayes import ComplementNB

# Метрики
from sklearn.metrics import f1_score

# Визуализация
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib import rcParams, rcParamsDefault
from pandas.plotting import scatter_matrix

# Настройка визуализации
rcParams['figure.figsize'] = (10, 6)
rcParams['figure.dpi'] = rcParamsDefault['figure.dpi'] * 0.8

print("Импорт библиотек успешно завершен!")


# In[3]:


# Отображение всех столбцов таблицы
pd.set_option('display.max_columns', None)
# Обязательно для нормального отображения графиков plt
rcParams['figure.figsize'] = 10, 6
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'svg'")
# Дополнительно и не обязательно для декорирования графиков
factor = .8
default_dpi = rcParamsDefault['figure.dpi']
rcParams['figure.dpi'] = default_dpi * factor


# In[5]:


# Глобальное значение "random_state"
STATE = 42


# In[7]:


file_path = '/content/toxic_comments.csv'

if os.path.exists(file_path):
    try:
        data = pd.read_csv(file_path, on_bad_lines='skip')  # Skip bad rows
        print(f"Файл {file_path} успешно загружен!")
        print("Первые 5 строк данных:")
        display(data.head())
    except Exception as e:
        print(f"Ошибка при загрузке файла: {e}")
else:
    print(f'Файл по пути {file_path} не найден. Убедитесь, что файл загружен правильно.')


# In[8]:


# Анализ датафрейма
print(data.info())
data.sample(n=10)


# In[9]:


# Анализ уникальных значений
# категориального признака "text"
data.toxic.unique()


# In[10]:


# Анализ баланса классов
# целевой переменной
data['toxic'].hist();


# ### Итоги начального анализа данных
# 
# В исследуемом наборе данных представлено **159292 записи** без пропущенных значений и три столбца:  
# 
# - **Столбец `text`**  
#   Содержит исходные тексты твитов, используется как основа для обучения и предсказаний, не является целевым признаком.  
# 
# - **Столбец `toxic`**  
#   Отражает результаты классификации текстов и выступает целевым признаком. Принимает значения только `0` и `1`, но имеет тип данных `int64`. Для оптимизации памяти рекомендуется преобразовать его в `uint8`.  
#   Характеризуется **выраженным дисбалансом классов** с преобладанием класса `0`, что важно учесть при разделении данных на обучающую и тестовую выборки.  
# 
# - **Столбец `Unnamed: 0`**  
#   Не содержит полезной информации для анализа и обучения моделей, поэтому будет исключен из дальнейшей работы.  

# In[11]:


# Оптимизация типов данных
data['toxic'] = data['toxic'].astype('uint8')


# In[12]:


# Функция очистки английского и русского текста от ненужных символов
def clear_symbols(text, lenguage):
    if lenguage == 'ru':
        return ' '.join(re.sub(r'[^а-яА-ЯёЁ ]', ' ', text).split())
    elif lenguage == 'en':
        return ' '.join(re.sub(r'[^a-zA-Z ]', ' ', text).split())


# In[13]:


# Функция работы с POS-тегами
def get_wordnet_pos(word):
    tag = pos_tag([word])[0][1][0].upper()
    tag_dict = {
        'J': wordnet.ADJ,
        'N': wordnet.NOUN,
        'V': wordnet.VERB,
        'R': wordnet.ADV,
    }
    return tag_dict.get(tag, wordnet.NOUN)


# In[14]:


# Функция лемматизации русских и английских текстов
def lemmatize(text, lenguage):
    if lenguage == 'ru':
        lemmatizer = Mystem()
        lemms = lemmatizer.lemmatize(text)
    elif lenguage == 'en':
        lemmatizer = WordNetLemmatizer()
        lemms = lemmatizer.lemmatize(text, get_wordnet_pos(text))
    string = ''
    for i in lemms:
        string += i
    return string


# In[15]:


# Очистка и лемматизация русских текстов
corpus_ru = data['text'].apply(
    lambda x: lemmatize(clear_symbols(x, 'ru'), 'ru')
)


# In[16]:


# Очистка и лемматизация английских текстов
corpus_eng = data['text'].apply(
    lambda x: lemmatize(clear_symbols(x, 'en'), 'en')
)


# In[17]:


# Добавление лемм в датафрейм
data['corpus'] = pd.DataFrame(corpus_ru) + ' ' + pd.DataFrame(corpus_eng)
# Приведение лемм к нижнему регистру
data['corpus'] = data['corpus'].str.lower()


# In[18]:


# Добавление нового признака с количеством слов в тексте
data['numbers_of_words'] = data['corpus'].copy().str.split().str.len().values
print(data['numbers_of_words'].describe())
data['numbers_of_words'].head()


# In[19]:


# # Добавление нового признака со средней длиной слов
data['average_word_lenght'] = (data['corpus'].str.len() / data['numbers_of_words'])
print(data['average_word_lenght'].describe())
print()

data.loc[data['average_word_lenght'] > data['average_word_lenght'].median()*2, 'average_word_lenght'] = data['average_word_lenght'].median()
print(data['average_word_lenght'].describe())
data['average_word_lenght'].head()


# In[20]:


# Проверка датафрейма с леммами
print('corpus:', data.loc[data['corpus'] != '', 'corpus'].count())
print()
data.loc[data['corpus'] != ''].head(25)


# In[21]:


# Облако слов
fdist = FreqDist(data['corpus'])
fdist.plot(30, cumulative=False)


# In[22]:


# Разделение датафрейма на целевую и нецелевую выборки
features = data[['corpus', 'numbers_of_words', 'average_word_lenght']]
target = data['toxic']

# Разделение выборок на обучающие и тестовые
features_train, features_test, target_train, target_test = train_test_split(
    features,
    target,
    test_size=.25,
    stratify=target,
    random_state=STATE
)


# In[23]:


# Проверка размерности
print('features_train:', len(features_train))
print('target_train:', len(target_train))
print('features_test:', len(features_test))
print('target_test:', len(target_test))


# ## Обучение

# In[24]:


def model_pipeline_gridsearch(
    features_train,
    target_train,
    model,
    params,
    data_grids,
    data_times
):

    start_time = time.time()

    # Pipeline
    pipeline = Pipeline([
        ('vect', ColumnTransformer([
            ('tfidf', TfidfVectorizer(), 'corpus'),
            ('stdm', MaxAbsScaler(), features_train[['numbers_of_words', 'average_word_lenght']].columns)
        ], remainder='drop'
        )),
        ('sampl', SMOTE(random_state=STATE)),
        ('clf', model)
    ])


    grid = HalvingGridSearchCV(
        pipeline,
        params,
        cv=4,
        n_jobs=-1,
        scoring='f1',
        error_score='raise',
        random_state=STATE
    )

    grid.fit(features_train, target_train)

    finish_time = time.time()
    funtion_time = finish_time - start_time

    data_grids.append(grid)
    data_times.append(funtion_time)

    return data_grids, data_times


# In[25]:


# Вывод на печать результатов модели
def print_model_result(grids, data_times, model_name):
    print('Модель    :', model_name)
    print('Метрика F1:', grids[-1].best_score_)
    print(f'Время     : {data_times[-1]} секунд')
    print('Параметры :\n', grids[-1].best_estimator_)
    print()
    print('-'*20)
    print()


# In[26]:


# Поиск лучших моделей и их параметров
data_grids = []
data_times = []


# In[28]:


# LogisticRegression

params = [{}]

data_grids, data_times = model_pipeline_gridsearch(
    features_train,
    target_train,
    LogisticRegression(random_state=STATE),
    params,
    data_grids,
    data_times
)

print_model_result(data_grids, data_times, 'LogisticRegression')


# In[31]:


#DecisionTreeClassifier

dtree_params = [{}]

data_grids, data_times = model_pipeline_gridsearch(
    features_train,
    target_train,
    DecisionTreeClassifier(random_state=STATE),
    dtree_params,
    data_grids,
    data_times
)

# Вывод результатов
print_model_result(data_grids, data_times, 'DecisionTreeClassifier')


# In[33]:


# ComplementNB

cnb_params = [{}]

data_grids, data_times = model_pipeline_gridsearch(
    features_train,
    target_train,
    ComplementNB(),
    cnb_params,
    data_grids,
    data_times
)

print_model_result(data_grids, data_times, 'ComplementNB')


# In[37]:


# CatBoostClassifier

catboost_params = [{}]

# Запуск поиска по сетке с CatBoostClassifier
data_grids, data_times = model_pipeline_gridsearch(
    features_train,
    target_train,
    CatBoostClassifier(logging_level='Silent', random_state=STATE),
    catboost_params,
    data_grids,
    data_times
)

# Вывод результатов
print_model_result(data_grids, data_times, 'CatBoostClassifier')


# In[39]:


data_grids_best = data_grids[0]
data_times_best = data_times[0]

for i in range(len(data_grids)):
    if data_grids[i].best_score_ > data_grids_best.best_score_:
        data_grids_best = data_grids[i]
        data_times_best = data_times[i]

print('Лучшее время        : ', data_times_best)
print('Лучший показатель F1: ', data_grids_best.best_score_)
print('Лучшая модель       : ')
print(data_grids_best)  # or data_grids_best.best_estimator_ to see the best model


# In[43]:


# Перевод лучшего времени в минуты
data_times_best/60/60


# ### Выводы по выбору оптимальной модели
# 
# Наивысший результат по метрике F1 — **0.7512969757147655** — был достигнут моделью машинного обучения `CatBoostClassifier`. Время обучения составило **14982.45 секунд** (примерно 4 часа).

# In[41]:


start_time = time.time()

# Предсказание лучшей модели
predict = data_grids_best.predict(features_test)

finish_time = time.time()
funtion_time = finish_time - start_time

# Расчет RMSE и времени выполнения предсказания
print('Показатель F1     :', f1_score(target_test, predict))
print(f'Время предсказания: {funtion_time} секунд')


# In[44]:


start_time = time.time()

# Предсказание лучшей модели
predict = data_grids_best.predict(features_test)

finish_time = time.time()
funtion_time = finish_time - start_time

# Расчет RMSE и времени выполнения предсказания
print('Показатель F1     :', f1_score(target_test, predict))
print(f'Время предсказания: {funtion_time} секунд')


# ## Выводы по результатам тестирования модели
# 
# При тестировании модели машинного обучения `CatBoostClassifier` с настройками по умолчанию в комбинации с моделью векторизации текстов `TfidfVectorizer` (также с параметрами по умолчанию) был достигнут лучший результат. Основные показатели:
# 
# - **Метрика F1**: Лучшее значение F1-меры составило **0.75** (точное значение: 0.7505570848079696). Это демонстрирует хорошую сбалансированность модели между точностью и полнотой, что особенно ценно для задач с несбалансированными классами.
# - **Время выполнения**: Код выполнился за **10.52 секунды**, что указывает на высокую скорость работы модели на данном объеме данных. Это делает подход эффективным для практического применения.
# 
# ### Анализ результатов
# 1. **Качество**: F1 = 0.75 подтверждает, что связка `TfidfVectorizer` и `CatBoostClassifier` успешно решает задачу классификации текстов. Это хороший результат для базовых настроек.
# 2. **Простота**: Использование параметров по умолчанию упрощает разработку и обеспечивает воспроизводимость, хотя тонкая настройка гиперпараметров могла бы повысить качество.
# 3. **Скорость**: Время в 10.52 секунды приемлемо для небольших или средних датасетов, но при масштабировании данных может потребоваться оптимизация.
# 
# ### Рекомендации
# - Если F1 = 0.75 удовлетворяет требованиям задачи, модель готова к внедрению.
# - Для повышения качества можно рассмотреть настройку гиперпараметров или альтернативные методы векторизации (например, BERT).
# - Дополнительно стоит оценить модель по другим метрикам (ROC-AUC, precision, recall), чтобы убедиться в её соответствии приоритетам задачи.

# ## Выводы

# ## Выводы по проекту
# 
# Цель проекта успешно достигнута: подобрана модель машинного обучения и метод векторизации текстов с оптимальными параметрами, которые обеспечили эффективную классификацию текстов на токсичные и нетоксичные. Итоговый показатель метрики F1 превысил значение **0.75**.
# 
# ### Выполненные шаги для достижения цели
# 1. **Подготовка среды**: Создана и настроена тетрадь Jupyter Notebook, выполнена загрузка и анализ данных.
# 2. **Обработка данных**: Проведена предобработка данных для последующего обучения моделей.
# 3. **Обучение и отбор моделей**: Протестировано несколько моделей машинного обучения, из которых выбрана лучшая по значению метрики F1.
# 4. **Тестирование**: Финальная модель прошла тестирование, подтвердившее её эффективность.
# 
# ### Итог
# Выбранная комбинация модели и метода векторизации демонстрирует высокую точность предсказаний, что делает её подходящей для решения задачи классификации текстов на токсичность.
