#!/usr/bin/env python
# coding: utf-8

# ### Оценка риска ДТП в реальном времени для каршеринговых сервисов  
# 
# ## Описание проекта  
# 
# **Цель:** Разработка системы оценки риска ДТП для выбранного маршрута в режиме реального времени. При высоком уровне риска система предупреждает водителя и предлагает альтернативные пути.  
# 
# **Ключевые задачи:**  
# 1. Построить модель предсказания ДТП с учетом:  
#    - Только автомобилей как виновников (`at_fault` = 1)  
#    - Повреждений транспортных средств (кроме царапин)  
#    - Данных за 2012 год (наиболее актуальные)  
#    - Фактора возраста автомобиля  
# 2. Проанализировать важность факторов ДТП  
# 3. Дать рекомендации по:  
#    - Возможности создания системы оценки риска  
#    - Дополнительным факторам для учета  
#    - Необходимости дополнительного оборудования (датчики, камеры)  
# 
# ## Описание данных  
# 
# **Основные таблицы:**  
# - `collisions` (информация о ДТП):  
#   - `case_id` - уникальный идентификатор ДТП  
#   - Содержит данные о месте и времени происшествия  
# 
# - `parties` (участники ДТП):  
#   - `case_id` + `party_number` - составной ключ  
#   - Поля: роль участника, виновность (`at_fault`)  
# 
# - `vehicles` (пострадавшие ТС):  
#   - Данные о транспортных средствах  
#   - Возраст автомобиля и другие характеристики  
# 
# **Стек технологий:** Python, Pandas, Scikit-learn, CatBoost/XGBoost, SHAP, Streamlit (для демо)  
# 
# **Метрики оценки:** ROC-AUC, Precision-Recall, Feature Importance  
# 
# > **Примечание:** Проект находится на стадии исследования возможностей. Требуется анализ данных и проверка гипотез о предсказуемости ДТП.  

# ## 1. Подключитесь к базе. Загрузите таблицы sql

# In[1]:


# Обновление библиотек
get_ipython().system('pip install --upgrade scikit-learn')
get_ipython().system('pip install torchmetrics')
get_ipython().system('pip install psycopg2')
get_ipython().system('pip install catboost')
get_ipython().system('pip install phik')


# In[2]:


# Стандартные библиотеки
import warnings
import os
import gc

# Библиотеки для работы с SQL
import psycopg2
from psycopg2 import extras

# Научные вычисления и обработка данных
import pandas as pd
import numpy as np
from math import ceil

# Графическая визуализация
import seaborn as sns
import matplotlib.pyplot as plt

# PyTorch для машинного обучения
import torch
import torch.nn as nn
from torchmetrics.classification import BinaryF1Score

# Scikit-learn для работы с данными и метриками
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, precision_score, recall_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier
from sklearn.impute import SimpleImputer

# Phik для расчёта корреляции
import phik
from phik import phik_matrix


# In[ ]:


# Функция создания и возврата соединения с базой даннх и курсора

def create_connection_and_cursor(db_config):
    connection_string = 'postgresql://{}:{}@{}:{}/{}'.format(
        db_config['user'],
        db_config['pwd'],
        db_config['host'],
        db_config['port'],
        db_config['db']
    )
    try:
        connection = psycopg2.connect(connection_string)
        cursor = connection.cursor(cursor_factory=extras.DictCursor)
        return connection, cursor
    except Exception as e:
        print(f"Ошибка подключения: {e}")
        return None, None


# In[ ]:


# Функция закрытия курсора и соединения с базой данных

def close_resources(cursor, connection):
    if cursor:
        cursor.close()
        print("Курсор закрыт")
    if connection:
        connection.close()
        print("Соединение закрыто")


# In[ ]:


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


# In[ ]:


# Функция для вычисления процента пропусков в данных
def compute_missing_data(df):
    missing_percentages = {}
    total_rows = len(df)

    for column in df.columns:
        null_count = df[column].isnull().sum()
        missing_percentage = round((null_count / total_rows) * 100, 2)

        missing_percentages[column] = missing_percentage

    for column, percentage in missing_percentages.items():
        print(f"Количество пропусков в столбце {column} = {percentage}%")


# In[ ]:


def calculate_total_missing_percentage(df):
    # Подсчет общего количества пропущенных значений в DataFrame
    total_missing = df.isnull().sum().sum()

    # Расчет общего процента пропущенных значений
    total_percent_missing = (total_missing / len(df)) * 100

    # Округление до двух знаков после запятой
    total_percent_missing_rounded = round(total_percent_missing, 2)

    return total_percent_missing_rounded


# In[ ]:


def replace_spaces_with_underscores(df):
    for column in df.columns:
        if df[column].dtype == 'object':
            df[column] = df[column].str.replace(' ', '_')
    return df


# In[ ]:


# Функция для инициализации весов нейросети
def init_weights(layer):
    """
    Инициализация весов слоя нейросети.
    Используется метод Kaiming для слоев Linear и 0 для bias.
    """
    if isinstance(layer, nn.Linear):
        nn.init.kaiming_uniform_(
            layer.weight,
            mode='fan_in',
            nonlinearity='relu'
        )
        if layer.bias is not None:
            nn.init.zeros_(layer.bias)


# In[ ]:


def plot_confusion_matrix_nn(model, X_test, y_test, title="Confusion Matrix"):
    """
    Функция для построения и отображения матрицы ошибок для нейронной сети, 
    а также вычисления Precision и Recall.

    Parameters:
    - model: Обученная нейронная сеть.
    - X_test: Тестовые данные.
    - y_test: Истинные метки классов для тестовых данных.
    - title: Заголовок для графика.
    """
    # Перевод модели в режим оценки
    model.eval()

    # Преобразование тестовых данных в тензоры
    X_test_torch = torch.FloatTensor(X_test.values)
    y_test_torch = torch.FloatTensor(y_test.values)

    # Получение предсказаний от модели
    with torch.no_grad():
        net_preds = model(X_test_torch).flatten()
        net_preds = net_preds.round().detach().numpy()  # Округляем для бинарной классификации

    # Вычисление матрицы ошибок
    cm = confusion_matrix(y_test, net_preds)

    # Определение значений из матрицы ошибок
    tn, fp, fn, tp = cm.ravel()  # Распаковка матрицы ошибок в TP, FP, FN, TN

    # Отображение матрицы ошибок
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Class 0', 'Class 1'])
    disp.plot(cmap='Blues', values_format='d')

    plt.title(title)
    plt.show()

    # Расчет и вывод Precision и Recall
    precision = tp / (tp + fp) if tp + fp != 0 else 0
    recall = tp / (tp + fn) if tp + fn != 0 else 0

    print(f"Precision = {precision:.3f}")
    print(f"Recall = {recall:.3f}")

    return cm, precision, recall


# In[ ]:


def check_table_exists(cursor, table_name):
    """
    Проверяет наличие таблицы в базе данных.

    :param cursor: курсор для выполнения SQL-запросов
    :param table_name: имя таблицы
    :return: True, если таблица существует, иначе False
    """
    try:
        cursor.execute("""
            SELECT COUNT(*)
            FROM information_schema.tables
            WHERE table_name = %s;
        """, (table_name,))
        return cursor.fetchone()[0] > 0
    except Exception as e:
        print(f"Ошибка при проверке таблицы {table_name}: {e}")
        return False


# In[ ]:


def encode_categorical_features(df, encoder_type='ordinal'):
    if encoder_type == 'ordinal':
        encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    elif encoder_type == 'onehot':
        encoder = OneHotEncoder(handle_unknown='ignore', drop='first')
    else:
        raise ValueError("Unsupported encoder type")

    categorical_columns = df.select_dtypes(include=['object']).columns
    encoded_data = encoder.fit_transform(df[categorical_columns])

    # Преобразование результата в DataFrame для one-hot encoding
    if encoder_type == 'onehot':
        encoded_data = pd.DataFrame(encoded_data.toarray(), columns=encoder.get_feature_names_out(categorical_columns))

    # Добавление закодированных данных обратно в исходный DataFrame
    df[categorical_columns] = encoded_data

    return df


# In[ ]:


# Включим вывод предупреждений
warnings.filterwarnings('default')


# In[ ]:


# Подключение к базе данных
db_config = {
'user': 'praktikum_student', # имя пользователя,
'pwd': 'Sdf4$2;d-d30pp', # пароль,
'host': 'rc1b-wcoijxj3yxfsf3fs.mdb.yandexcloud.net',
'port': 6432, # порт подключения,
'db': 'data-science-vehicle-db' # название базы данных,
}


# ## 2. Первичное исследование таблиц

# In[ ]:


connection, cursor = create_connection_and_cursor(db_config)

if cursor:
    try:
        # Список таблиц для проверки
        required_tables = ['case_ids', 'parties', 'collisions', 'vehicles']

        # Проверяем наличие каждой таблицы
        missing_tables = [
            table for table in required_tables
            if not check_table_exists(cursor, table)
        ]

        if missing_tables:
            print(f"Отсутствуют следующие таблицы в базе данных: {', '.join(missing_tables)}")
        else:
            print("Все необходимые таблицы присутствуют в базе данных.")

    except Exception as e:
        print(f"Ошибка при проверке таблиц: {e}")


# In[ ]:


if cursor:
    try:
        limit = 1000

        cursor.execute(f'SELECT * FROM case_ids LIMIT {limit};')
        case_ids = pd.DataFrame(cursor.fetchall(), columns=[desc[0] for desc in cursor.description])

        print(f'Таблица case_ids успешно загружена! Показаны первые {limit} строк.')
        print(summarize_dataframe(case_ids))

    except Exception as e:
        print(f"Ошибка при загрузке таблицы case_ids: {e}")


# In[ ]:


# Удалю датафрейм, так как умирайет Kernel, далее при ниобходимости загружу или загружу часть таблицы
del case_ids
gc.collect()


# In[ ]:


if cursor:
    try:
        limit = 1000
        cursor.execute(f'SELECT * FROM parties LIMIT {limit};;')

        parties = pd.DataFrame(cursor.fetchall(), columns=[desc[0] for desc in cursor.description])
        print('Таблица case_ids успешно загружена!')
        print(summarize_dataframe(parties))

    except Exception as e:
        print(f"Ошибка при загрузке таблицы case_ids: {e}")


# In[ ]:


# Удалю датафрейм, так как умирайет Kernel, далее при ниобходимости загружу или загружу часть таблицы
del parties
gc.collect()


# In[ ]:


if cursor:
    try:
        limit = 1000
        cursor.execute(f'SELECT * FROM collisions LIMIT {limit};;')

        collisions = pd.DataFrame(cursor.fetchall(), columns=[desc[0] for desc in cursor.description])
        print('Таблица case_ids успешно загружена!')
        print(summarize_dataframe(collisions))

    except Exception as e:
        print(f"Ошибка при загрузке таблицы case_ids: {e}")


# In[ ]:


# Удалю датафрейм, так как умирайет Kernel, далее при ниобходимости загружу или загружу часть таблицы
del collisions
gc.collect()


# In[ ]:


if cursor:
    try:
        limit = 1000
        cursor.execute(f'SELECT * FROM vehicles LIMIT {limit};;')

        vehicles = pd.DataFrame(cursor.fetchall(), columns=[desc[0] for desc in cursor.description])
        print('Таблица case_ids успешно загружена!')
        print(summarize_dataframe(vehicles))

    except Exception as e:
        print(f"Ошибка при загрузке таблицы case_ids: {e}")


# In[ ]:


# Удалю датафрейм, так как умирайет Kernel, далее при ниобходимости загружу или загружу часть таблицы
del vehicles
gc.collect()


# In[ ]:


query_min_max = 'SELECT min(COLLISION_DATE), max(COLLISION_DATE) FROM collisions'
cursor.execute(query_min_max)
result = cursor.fetchone()
print(f"Минимальная дата: {result[0]}, Максимальная дата: {result[1]}")


# <div>
# 
# ### **Промежуточный анализ таблиц**
# 
# Исходя из информации о таблицах (`case_ids`, `collisions`, `parties`, `vehicles`), сделаны следующие выводы:
# 
# ---
# 
# #### **Общие характеристики:**
# - **Связующее поле:** Все таблицы содержат столбец `case_id`, который, вероятно, используется для объединения данных о ДТП между таблицами.
# - **Типы данных:**
#   - Категориальные (`object`): описывают качественные характеристики (например, тип транспортного средства, погодные условия).
#   - Числовые (`int64`, `float64`): содержат количественные показатели (например, расстояния, суммы убытков).
# - **Размерность таблиц:**
#   - Таблица `vehicles`: 1,021,234 записей.
#   - Таблица `parties`: 2,752,408 записей.
#   - Другие таблицы варьируются в размерах между ними, отражая различный уровень детализации.
# 
# ---
# 
# #### **Специфика таблиц:**
# - **`case_ids`:**  
#   Содержит базовую информацию, например, идентификаторы случаев и их годы. Таблица может быть полезна для фильтрации или группировки данных по годам.
# 
# - **`collisions`:**  
#   Самая подробная таблица, включающая географию ДТП, погодные условия, типы участников и характер повреждений. Наибольшее количество столбцов и записей среди всех таблиц делает её центральным элементом анализа.
# 
# - **`parties`:**  
#   Описывает участников ДТП, их роли и дополнительные параметры (например, страховые данные). Таблица среднего размера, указывающая на наличие дополнительных данных для каждого участника.
# 
# - **`vehicles`:**  
#   Содержит данные о транспортных средствах, участвующих в ДТП (тип, возраст, характеристики). Имеет меньшее количество записей, возможно из-за того, что один случай включает несколько транспортных средств.
# 
# ---
# 
# #### **Заключение:**
# - Эти четыре таблицы представляют собой комплексную базу данных о ДТП, охватывая разные аспекты инцидентов.
# - Их совместное использование позволяет глубоко анализировать причины и характеристики происшествий.
# - **Рекомендация:** Использовать `case_id` как связующее поле для объединения данных и выявления взаимосвязей между участниками, транспортными средствами и обстоятельствами ДТП.
# 
# </div>
# 

# ## 3. Проведите статистический анализ факторов ДТП

# ### 3.1. Количество аварий по месяцам

# In[ ]:


# Создание соединения и выполнение запроса
query_average_month = '''
SELECT CAST(EXTRACT(MONTH FROM COLLISION_DATE) AS INTEGER) AS month_number,
       INITCAP(TO_CHAR(COLLISION_DATE, 'Month')) AS month_name,
       COUNT(*) AS collision_count
FROM collisions
GROUP BY month_name, month_number
ORDER BY month_number;
'''

cursor.execute(query_average_month)
rows = cursor.fetchall()
columns = [desc[0] for desc in cursor.description]


# Создание DataFrame
df = pd.DataFrame(rows, columns=columns)

# Настройка графика
sns.set_theme(style="whitegrid")
plt.figure(figsize=(12, 6))
bar = sns.barplot(
    x="month_name",
    y="collision_count",
    data=df,
    palette="viridis"
)

# Добавление подписей к столбцам
for i, value in enumerate(df['collision_count']):
    bar.text(i, value + 50, f'{value:,}', ha='center', fontsize=9)

# Настройки осей и заголовка
plt.title("Количество аварий по месяцам", fontsize=14, fontweight='bold')
plt.xlabel("Месяц", fontsize=12)
plt.ylabel("Количество аварий", fontsize=12)
plt.xticks(rotation=45, ha='right')

# Показ графика
plt.tight_layout()
plt.show()

# Печать таблицы
print("Данные по авариям по месяцам:")
print(df.to_string(index=False))


# #### Вывод:
# - **Самый высокий показатель аварий:** В марте (139,581 случаев).
# - **Самый низкий показатель аварий:** В июле (102,227 случаев).
# - **Заметное снижение аварий летом (июнь, июль, август)**, что может быть связано с погодными условиями или снижением трафика.

# ### 3.2. Задачи для коллег

# #### 3.2.1. Анализ самых частых факторов ДТП
# Задача заключается в выявлении самых частых основных факторов, приводящих к ДТП. Для этого необходимо проанализировать данные из таблицы `collisions`, где основной фактор аварии указан в параметре `primary_collision_factor`.

# In[ ]:


query_top_violation_categories = '''
SELECT pcf_violation_category, COUNT(*) AS collision_count
FROM collisions
GROUP BY pcf_violation_category
ORDER BY collision_count DESC
LIMIT 10;
'''

# Выполнение SQL запроса и загрузка результатов в DataFrame
cursor.execute(query_top_violation_categories)
rows = cursor.fetchall()
columns = [desc[0] for desc in cursor.description]
df_top_violation_categories = pd.DataFrame(rows, columns=columns)

# Вывод таблицы с результатами
print(df_top_violation_categories)

# Настройка графика
plt.figure(figsize=(10, 6))
sns.barplot(x='collision_count', y='pcf_violation_category', data=df_top_violation_categories, palette='Blues_d')

# Добавление аннотаций для количества на столбцах
for index, value in enumerate(df_top_violation_categories['collision_count']):
    plt.text(value, index, f'{value:,}', va='center', fontsize=10)

# Настройки осей и заголовка
plt.title('Топ-10 категорий нарушений по количеству ДТП', fontsize=14, fontweight='bold')
plt.xlabel('Количество ДТП', fontsize=12)
plt.ylabel('Категория нарушения', fontsize=12)
plt.tight_layout()

# Показать график
plt.show()


# #### Вывод:
# - **Основная причина аварий:** Превышение скорости (speeding), которое является причиной 438,439 аварий.
# - **Другие распространенные нарушения:** Неправильный поворот (improper turning) и нарушение правил права проезда (automobile right of way) занимают второе и третье места соответственно.
# - **Менее частые нарушения:** Такие категории, как "несоответствующие действия на дороге" и "нарушение правил парковки", имеют гораздо меньшую частоту, но все равно способствуют значительному числу ДТП.
# - **Неопределенные данные:** В категории "unknown" зафиксировано 39,558 случаев, что может означать отсутствие информации о причине нарушения в данных.

# #### 3.2.2. Анализ зависимости степени трезвости участника и основного фактора аварии
# Нужно проанализировать зависимость между степенью трезвости участника ДТП и основным факторов аварии.
# Необходимые для задачи параметры: `primary_collision_factor` (таблица `collisions`) и `party_sobriety` (таблица `parties`).

# In[ ]:


query = '''
WITH collision_damage_table AS
    (SELECT case_id,
        collision_damage
    FROM collisions),
number_of_coll AS
    (SELECT p.case_id,
        p.party_drug_physical,
        cdt.collision_damage,
        COUNT(*) OVER (PARTITION BY p.party_drug_physical, cdt.collision_damage)
    FROM parties AS p
    JOIN collision_damage_table AS cdt ON p.case_id = cdt.case_id
    WHERE p.party_type = 'car')


SELECT party_drug_physical,
    collision_damage,
    count
FROM number_of_coll
WHERE party_drug_physical NOT IN ('G', 'not applicable', 'None')
GROUP BY party_drug_physical, collision_damage, count;
'''

# Выполнение SQL запроса
cursor.execute(query)
rows = cursor.fetchall()

# Извлекаем названия столбцов из описания курсора
columns = [desc[0] for desc in cursor.description]

# Создаем DataFrame из полученных данных
query_df = pd.DataFrame(rows, columns=columns)

# Для удобства заменим английские значения на русские
query_df.replace('impairment - physical', 'Ухудшение состояния', inplace=True)
query_df.replace('sleepy/fatigued', 'Сонный/Усталый', inplace=True)
query_df.replace('under drug influence', 'Под воздействием лекарств', inplace=True)

query_df.replace('fatal', 'Не подлежит восстановлению', inplace=True)
query_df.replace('middle damage', 'Машина в целом на ходу', inplace=True)
query_df.replace('scratch', 'Царапина', inplace=True)
query_df.replace('severe damage', 'Серьезное повреждение', inplace=True)
query_df.replace('small damage', 'Отедльный элемент под замену/покраску', inplace=True)

# Отображаем DataFrame
display(query_df)


# In[ ]:


# Устанавливаем стиль для более приятного внешнего вида
sns.set(style="whitegrid")

# Создаем график зависимости между состоянием участника и серьезностью повреждения
plt.figure(figsize=(8, 8))

# Визуализация с использованием scatter plot
scatter = plt.scatter(
    query_df['party_drug_physical'],  # состояние участника
    query_df['collision_damage'],     # серьезность повреждения
    s=query_df['count'] * 2.4,        # уменьшение размера точек на 20%
    c=query_df['count'],              # цвет точек зависит от количества
    cmap='viridis',                   # палитра для цветов
    edgecolors='w',                   # белая рамка для улучшения видимости точек
    alpha=0.6                         # уменьшение прозрачности точек
)

# Добавление заголовка и подписей к осям
plt.title('Зависимость между состоянием участника и серьезностью повреждения', fontsize=18)
plt.xlabel('Состояние участника', fontsize=14)
plt.ylabel('Серьезность повреждения', fontsize=14)

# Настройка размера шрифтов на осях
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

# Добавление цветовой шкалы для графика
plt.colorbar(scatter, label='Количество происшествий')

# Поворот подписей на оси X, если они слишком длинные
plt.xticks(rotation=45)

# Отображение графика
plt.tight_layout()
plt.show()


# ### Выводы:
# 
# На основе полученных результатов можно выделить несколько ключевых наблюдений:
# 
# 1. **Фатальные случаи**: Часто происходят, когда на участника ДТП воздействуют лекарства (категория "under drug influence" на оси X и "fatal" на оси Y).
# 2. **Небольшие повреждения**: Такие случаи распределяются относительно равномерно между тремя состояниями участников: 1) человек уснул, 2) ухудшение состояния водителя, 3) воздействие лекарств. Однако преобладает состояние усталости среди этих факторов для данного типа серьезности повреждения.
# 3. **Другие типы повреждений**: Для остальных типов повреждений количество ДТП распределено более равномерно между тремя типами состояния участников.

# #### 3.2.3. Влияние состояния участника на серьезность повреждений транспортного средства
# Необходимо исследовать, как состояние участника ДТП влияет на степень повреждения его транспортного средства. Для этого нужно использовать параметры `collision_damage` (таблица `collisions`) и `party_drug_physical` (таблица `parties`). При этом следует исключить аварии с неизвестным или не оцененным состоянием участников.
# 
# #### 3.2.4. Связь между типом аварии и числом участников
# Тип аварии влияет на количество участников ДТП. Например, при лобовом столкновении (Head-On) обычно больше участников, чем при столкновении сзади (Rear End). Создать SQL-запрос, который соединяет таблицы `collisions` и `parties` и извлекает данные о типе аварии `type_of_collision` и числе участников `party_count`.
# 
# #### 3.2.5. Анализ зависимости категории нарушения и состояния участника
# Необходимо исследовать, как состояние участника ДТП связано с категорией нарушения. Возможно, определенные состояния участников приводят к изменениям в типах нарушений. Для этого необходимо использовать параметры `pcf_violation_category` (таблица `collisions`) и `party_drug_physical` (таблица `parties`).
# 
# #### 3.2.6. Влияние погоды на частоту ДТП
# Предполагается, что погода может влиять на частоту ДТП. Задача заключается в проверке гипотезы о том, что плохая погода увеличивает частоту ДТП. Для анализа используются параметры `at_fault` (таблица `parties`) и `weather_1` (таблица `collisions`).
# 
# #### 3.2.7. Влияние состояния дороги на серьезность повреждений
# Состояние дороги влияет на серьезность повреждений в ДТП. Например, при плохих дорожных условиях (например, "скользкая дорога" или "ямы") повреждения транспортных средств будут более серьезными. Создайте SQL-запрос, который соединяет таблицы `collisions` и `parties` и извлекает данные о состоянии дороги `road_surface` и серьезности повреждений `collision_damage`.

# ## Создайте модель для оценки водительского риска

# In[ ]:


# SQL-запрос для извлечения данных
query = '''
SELECT *
FROM parties p
JOIN vehicles v
    ON p.case_id = v.case_id AND p.party_number = v.party_number
LEFT JOIN collisions c
    ON p.case_id = c.case_id
WHERE p.case_id IN (
    SELECT case_id
    FROM parties
    WHERE party_type = 'car' AND at_fault = 1
)
AND LOWER(c.collision_damage) != 'scratch'
AND EXTRACT(YEAR FROM c.collision_date) = 2012;
'''

# Выполняем запрос и загружаем данные в DataFrame
df = pd.read_sql_query(query, connection)

# Вывод информации о результатах
print(f"Загружено {len(df)} записей из базы данных.")
display(df.head())


# In[ ]:


# SQL-запрос для получения всех уникальных значений столбца COLLISION_DAMAGE
query_unique_collision_damage = '''
SELECT DISTINCT collision_damage
FROM collisions
ORDER BY collision_damage;
'''

# Выполняем запрос и загружаем результат в DataFrame
unique_collision_damage_df = pd.read_sql_query(query_unique_collision_damage, connection)

# Вывод результатов с пояснением
print(f"Найдено {len(unique_collision_damage_df)} уникальных значений в столбце 'collision_damage'.")
display(unique_collision_damage_df)


# In[ ]:


# SQL-запрос для объединения данных из таблиц и фильтрации результатов
optimized_query = '''
SELECT *
FROM parties AS p
INNER JOIN vehicles AS v ON p.case_id = v.case_id AND p.party_number = v.party_number
INNER JOIN collisions AS c ON p.case_id = c.case_id
WHERE c.collision_damage != 'scratch'
      AND EXTRACT(YEAR FROM c.collision_date) = 2012
      AND p.case_id IN (
          SELECT case_id
          FROM parties
          WHERE party_type = 'car' AND at_fault = 1
      );
'''

# Загружаем результаты запроса в DataFrame и выводим первые строки
result_df = pd.read_sql_query(optimized_query, connection)
print(f"Получено {len(result_df)} записей. Отображаем первые строки:")
display(result_df.head())

# Закрываем подключение и курсор
close_resources(cursor, connection)


# ### Первичный отбор факторов, необходимых для построения модели
# 
# #### Признаки, которые не подходят для анализа:
# - **`id`**: Уникальный идентификатор записи, не имеет влияния на вероятность или характер ДТП.
# - **`case_id`**: Идентификатор конкретного случая происшествия, служит лишь связующим элементом между таблицами.
# - **`party_number`**: Номер участника происшествия, не влияет на причины или обстоятельства ДТП.
# - **`insurance_premium`**: Сумма страховой премии, не связана с вероятностью или обстоятельствами аварий.
# - **`party_drug_physical`**: Физическое состояние участника, неприменимо в контексте анализа транспортных средств как виновников.
# - **`cellphone_in_use`**: Использование мобильного телефона, относится скорее к действиям водителя, чем к характеристикам машины.
# - **`type_of_collision`**: Тип столкновения может быть полезен, но требует дополнительной категоризации для применения.
# - **`motor_vehicle_involved_with`**: Сведения о других участниках ДТП, не касаются напрямую транспортного средства виновника.
# - **`road_surface`, `road_condition_1`, `control_device`, `lighting`**: Дорожные и погодные условия могут влиять на аварийность, но не являются характеристиками конкретного транспортного средства.
# - **`collision_date`, `collision_time`**: Временные метки происшествия не дают прямой информации о технических характеристиках или виновности транспортного средства.
# 
# #### Признаки с ограниченной значимостью для модели:
# - **`primary_collision_factor`**: Основная причина аварии, полезна для общего анализа, но не критична для анализа машины как виновника.
# - **`pcf_violation_category`**: Категория нарушения, связана с поведением водителя, но не с характеристиками транспортного средства.
# 
# #### Признаки с потенциальным косвенным влиянием:
# - **`party_sobriety`**: Уровень трезвости водителя может быть индикатором его способности к управлению.
# - **`party_type`**: Тип участника происшествия, потенциально влияет на поведение и его последствия для ДТП.
# 
# #### Наиболее значимые признаки:
# - **`AT_FAULT`**: Ответственность за ДТП, позволяет разделить данные на виновных и невиновных.
# - **`VEHICLE_TYPE`**: Тип транспортного средства, напрямую влияет на поведение и характеристики на дороге.
# - **`COLLISION_DAMAGE`**: Уровень повреждений, отражает тяжесть происшествия и потенциальное влияние дорожных условий или технического состояния машины.
# 

# In[ ]:


# Шаг 1: Фильтрация данных по типу участника
# Оставляем только те строки, где тип участника — автомобиль.
filtered_df = result_df[result_df['party_type'] == 'car']

# Шаг 2: Удаление признаков, не влияющих на анализ
# Исключаем из набора данных столбцы, которые признаны малозначимыми.
columns_to_drop = [
    'id',
    'case_id',
    'party_number',
    'insurance_premium',
    'party_drug_physical',
    'cellphone_in_use',
    'type_of_collision',
    'motor_vehicle_involved_with',
    'road_surface',
    'road_condition_1',
    'control_device',
    'lighting',
    'collision_date',
    'collision_time'
]
analyze_df = filtered_df.drop(columns=columns_to_drop, axis=1)

# Шаг 3: Проверка итогового набора данных
# Просмотр первых строк после фильтрации
print("Первые строки обработанного набора данных:")
display(analyze_df.head())

# Информация о структуре DataFrame, включая количество ненулевых значений и типы данных
print("Информация о DataFrame после удаления ненужных столбцов:")
display(analyze_df.info())

# Статистическое описание числовых столбцов
print("Статистические показатели:")
display(analyze_df.describe())


# In[ ]:


cleaned_df = analyze_df.copy()

columns_to_drop = [
    'party_count',
    'intersection',
    'collision_damage',
    'primary_collision_factor',
    'pcf_violation_category',
    'party_type',
    'party_sobriety'
]
cleaned_df = cleaned_df.drop(columns=columns_to_drop, axis=1)


# ### Cтатистическое исследование отобранных факторов

# #### Количественные признаки:
# - **distance**: Реальное расстояние до места происшествия. Может влиять на скорость и внимание водителя.
# 
# #### Категориальные признаки:
# - **at_fault**: Указывает на виновника происшествия. Ключевой признак для классификации случаев по ответственности.
# - **vehicle_type**: Категория транспортного средства (например, легковой автомобиль, грузовик, мотоцикл). Определяет технические характеристики, которые могут повлиять на поведение на дороге.
# - **vehicle_transmission**: Тип трансмиссии (автоматическая, механическая). Может косвенно влиять на управляемость автомобиля и поведение водителя в сложных ситуациях.
# - **vehicle_age**: Возраст транспортного средства. Старые автомобили могут быть менее безопасными из-за износа систем безопасности и потенциальных неисправностей.
# - **county_city_location**: Близость к городской черте. Может повлиять на дорожные условия и правила, например, ограничение скорости.
# - **county_location**: Расстояние до ближайших населенных пунктов. Указывает на возможные изменения в плотности движения и инфраструктуре.
# - **weather_1**: Погодные условия во время ДТП (например, дождь, снег, ясная погода). Влияют на сцепление с дорогой и видимость.
# - **location_type**: Тип местоположения (город, пригород, шоссе). Может повлиять на интенсивность движения и уровень риска.
# - **direction**: Направление движения на момент ДТП. Может влиять на траекторию движения и взаимодействие с другими участниками.

# In[ ]:


compute_missing_data(cleaned_df)


# In[ ]:


result_total_missing = calculate_total_missing_percentage(cleaned_df)
print(f"Общее количество пропусков в сформированном датафрейме составляет {result_total_missing:.2f}%")


# In[ ]:


has_spaces_columns = []
for column in cleaned_df.columns:
    if cleaned_df[column].dtype == 'object':
        has_spaces = cleaned_df[column].str.contains(' ', na=False).any()
        if has_spaces:
            has_spaces_columns.append(column)

if has_spaces_columns:
    for column in has_spaces_columns:
        print(f"В столбце {column} есть пробелы, тип: {cleaned_df[column].dtype}")
else:
    print("В датафрейме пробелов нет")


# In[ ]:


df = replace_spaces_with_underscores(cleaned_df)


# In[ ]:


# повторная проверка на наличие пробелов в столбцах
has_spaces_columns = []
for column in df.columns:
    if df[column].dtype == 'object':
        has_spaces = df[column].str.contains(' ', na=False).any()
        if has_spaces:
            has_spaces_columns.append(column)

if has_spaces_columns:
    for column in has_spaces_columns:
        print(f"В столбце {column} есть пробелы, тип: {df[column].dtype}")
else:
    print("В датафрейме пробелов нет")


# In[ ]:


# Проверим уникальные значения

for column in df.columns:
    print(f"Unique values in column {column}: {df[column].unique()}")
    print()


# In[ ]:


# Заменяем отсутствующие значения в столбце 'location_type'
df['location_type'].fillna('no_info', inplace=True)


# In[ ]:


display(df.head())
display(df.info())


# In[ ]:


compute_missing_data(df)


# In[ ]:


result_total_missing = calculate_total_missing_percentage(df)
print(f"Общее количество пропусков в сформированном датафрейме составляет {result_total_missing:.2f}%")


# In[ ]:


# Проверка на дубликаты
print("Количество дубликатов:", df.duplicated().sum())
print('')
# Удаление дубликатов
df = df.drop_duplicates()
print(df.info())


# In[ ]:


result_total_missing = calculate_total_missing_percentage(df)
print(f"Общее количество пропусков в сформированном датафрейме составляет {result_total_missing:.2f}%")


# In[ ]:


# Создание копии df для создания графика после обучения модели
df_copy = df.copy()


# <div>    
# 
# **Вывод** <br>
# 
# В результате выполнения ряда процедур очистки и трансформации данных в датафрейме, можно выделить несколько ключевых шагов, которые существенно повысили качество и достоверность исходных данных.
# 
# 1. **Стандартизация значений**: Все пробелы в категориальных столбцах были заменены на подчеркивания. Эта мера направлена на унификацию формата данных, что упрощает их обработку и уменьшает риск возникновения ошибок при анализе. Такой подход способствует улучшению сортировки и более точному сравнению данных.
# 
# 2. **Заполнение пропусков**: Для столбца 'location_type' отсутствующие значения были заменены на 'no_info'. Это необходимая мера для устранения пропущенных данных, которые могли бы повлиять на результаты анализа. Такая замена помогает сохранить целостность данных, не теряя информации о присутствии или отсутствии этих параметров.
# 
# 3. **Удаление дубликатов**: Все повторяющиеся записи были удалены из датафрейма. Этот шаг необходим для обеспечения уникальности каждой строки данных, предотвращая ошибочное удвоение значений. Удаление дубликатов способствует упрощению анализа и уменьшению объема данных, что важно для точных выводов.
# 
# После выполнения этих шагов датафрейм стал значительно более чистым и готовым для проведения дальнейшего анализа. Структура данных стала более организованной и надежной, что открывает возможности для глубокого анализа влияния различных факторов на происшествия, таких как возраст автомобилей, тип местоположения, погодные условия и другие параметры. Также теперь возможно применение более сложных методов анализа, включая машинное обучение, для предсказания и классификации различных аспектов дорожных инцидентов.
# 
# </div>
# 

# ### Визуализация и анализ данных (EDA)
# 
# После предварительной обработки данных, следующим шагом является визуализация и анализ данных (Exploratory Data Analysis, EDA). Это поможет нам лучше понять распределение и взаимосвязи между различными переменными в нашем наборе данных.

# In[ ]:


# Построение графика распределения возраста автомобилей
plt.figure(figsize=(12, 7))
sns.histplot(df['vehicle_age'], bins=25, kde=True, color='skyblue', edgecolor='black')
plt.title('Распределение возраста автомобилей', fontsize=16)
plt.xlabel('Возраст автомобиля (годы)', fontsize=12)
plt.ylabel('Количество автомобилей', fontsize=12)
plt.grid(True)
plt.show()

# Расчет статистического описания для отфильтрованных данных
vehicle_age_stats = df['vehicle_age'].describe()
print("Статистическое описание распределения возраста автомобилей:")
print(vehicle_age_stats)


# #### Влияние погоды на количество ДТП

# In[ ]:


# Построение графика зависимости числа ДТП от погодных условий
weather_distribution = df["weather_1"].value_counts()
plt.figure(figsize=(12, 7))
sns.barplot(x=weather_distribution.index, y=weather_distribution.values, palette="coolwarm", alpha=0.85)
plt.title('Влияние погодных условий на частоту ДТП', fontsize=16)
plt.xlabel('Тип погоды', fontsize=12)
plt.ylabel('Число ДТП', fontsize=12)
plt.xticks(rotation=45, ha="right", fontsize=10)
plt.tight_layout()
plt.show()


# #### Изучение связи между возрастом автомобиля и его типом

# In[ ]:


# Построение боксплота для анализа связи между типом автомобиля и его возрастом
plt.figure(figsize=(12, 7))
sns.boxplot(x='vehicle_type', y='vehicle_age', data=filtered_df, palette="Set2")
plt.title('Влияние типа автомобиля на его возраст', fontsize=16)
plt.xlabel('Тип транспортного средства', fontsize=12)
plt.ylabel('Возраст транспортного средства (годы)', fontsize=12)
plt.xticks(rotation=45, ha="right", fontsize=10)
plt.tight_layout()
plt.show()


# In[ ]:


# Список названий непрерывных признаков (количественные)
interval_cols = ['vehicle_age', 'distance']

# Вычисление матрицы корреляций Phik с указанием интервалов
phik_overview = df.phik_matrix(interval_cols=interval_cols)


# In[ ]:


phik_overview.round(2)

# Построение корреляционной матрицы
plt.figure(figsize=(14, 10))
sns.heatmap(
    phik_overview.values,
    xticklabels=phik_overview.columns,
    yticklabels=phik_overview.index,
    cmap='coolwarm',  # Можно выбрать другую палитру
    vmin=0, vmax=1,
    annot=True,  # Если нужно отображать значения в ячейках
    fmt='.2f',  # Форматирование чисел
    cbar_kws={'label': 'Correlation (Phik)'}  # Подпись для цветовой шкалы
)

plt.title('Phik Correlation Matrix')
plt.show()


# <div>
# 
# **Вывод**  
# 
# <br>**Создание гистограммы распределения возраста автомобилей**<br>  
# Возраст автомобилей в основном варьируется от 0 до 20 лет, с пиком в 5 лет. Это может свидетельствовать о том, что большинство автомобилей на дороге достаточно новые, что отражает тенденцию обновления автопарка.
# 
# <br>**Влияние погоды на количество ДТП**<br>  
# ДТП чаще происходят при ясной погоде, в то время как наименьшее количество инцидентов наблюдается при снегопаде. Это может говорить о том, что ясная погода, несмотря на лучшие видимость и условия, способствует более интенсивному движению, что увеличивает вероятность столкновений.
# 
# <br>**Изучение связи между возрастом автомобиля и его типом**<br>  
# Существует четкая связь между возрастом автомобиля и его типом. Старые автомобили чаще всего оказываются легковыми, в то время как новые автомобили — грузовыми. Это может свидетельствовать о предпочтении приобретения новых грузовиков для коммерческого использования, в отличие от легковых автомобилей, которые часто используются в частных целях.
# 
# <br>**Изучение матрицы корреляции**<br>  
# Наиболее коррелирущими факторами с целевым - являются состояние трезвости водителя и ичсло участников ДТП.
#     
# Эти факторы могут быть связаны как с человеческим фактором, так и с погодными условиями и типом автомобиля, что подчеркивает необходимость комплексного подхода к анализу и улучшению дорожной безопасности.
# 
# </div>
# 

# #### Разделение данных на обучающую и тестовую выборки

# In[ ]:


# Разделение данных на обучающую и тестовую выборки
X = df.drop(columns=['at_fault'])
y = df['at_fault']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Удаление аномальных значений в обучающей выборке
# Использование метода .loc для безопасности и читаемости
X_train = X_train[X_train['vehicle_age'] != 161]
y_train = y_train[X_train.index]

# Проверка количества удаленных строк
print(f"Удалено {len(X_train[X_train['vehicle_age'] == 161])} строк с аномальными значениями.")


# #### Обработка пропущенных значений после разделения
# Теперь, когда данные разделены, мы можем безопасно заменить пропущенные значения, используя агрегированные значения из обучающей выборки.

# In[ ]:


# Импутация для количественных столбцов
quantitative_columns = ['distance']
imputer_quant = SimpleImputer(strategy='median')  # Используем медиану для импутации

# Применение импутации для обучающей и тестовой выборок
X_train[quantitative_columns] = imputer_quant.fit_transform(X_train[quantitative_columns])
X_test[quantitative_columns] = imputer_quant.transform(X_test[quantitative_columns])

# Проверка завершения импутации
print("Импутация завершена для столбцов:", quantitative_columns)


# In[ ]:


# Импутация для категориальных столбцов
categorical_columns = [
    'vehicle_type', 'vehicle_transmission', 'vehicle_age', 
    'county_city_location', 'county_location', 'weather_1', 
    'location_type', 'direction'
]

# Создаем импутер для категориальных данных
imputer_cat = SimpleImputer(strategy='most_frequent')  # Используем наиболее частое значение для импутации

# Применение импутации для обучающей и тестовой выборок
X_train[categorical_columns] = imputer_cat.fit_transform(X_train[categorical_columns])
X_test[categorical_columns] = imputer_cat.transform(X_test[categorical_columns])

# Вывод завершения импутации
print("Импутация завершена для категориальных столбцов:", categorical_columns)


# In[ ]:


# Масштабирование количественных данных
scaler = StandardScaler()

# Применение масштабирования для обучающей и тестовой выборок
X_train[quantitative_columns] = scaler.fit_transform(X_train[quantitative_columns])
X_test[quantitative_columns] = scaler.transform(X_test[quantitative_columns])

# Вывод завершения масштабирования
print(f"Масштабирование завершено для столбцов: {quantitative_columns}")


# In[ ]:


# Пример использования для кодирования с использованием ordinal и onehot
X_train_encoded_ordinal = encode_categorical_features(X_train, encoder_type='ordinal')
X_test_encoded_ordinal = encode_categorical_features(X_test, encoder_type='ordinal')

X_train_encoded_onehot = encode_categorical_features(X_train, encoder_type='onehot')
X_test_encoded_onehot = encode_categorical_features(X_test, encoder_type='onehot')


# ### Задача бизнес-анализа: минимизация риска попадания в ДТП
# 
# В рамках данной бизнес-задачи необходимо минимизировать риск попадания в ДТП на маршруте движения. Задача является **бинарной классификацией**, где целевой показатель принимает значения: **"Виновен"** и **"Не виновен"**.
# 
# Для решения этой задачи необходимо:
# 
# - **Минимизировать ложноотрицательные значения** с целью снижения возможного риска попадания в ДТП.
# - **Минимизировать ложноположительные значения**, чтобы обеспечить возможность быстрого добирания до пункта назначения.
# 
# Эти цели можно достичь с помощью использования **F1-меры**. Она позволяет балансировать между точностью и полнотой, что особенно важно в таких ситуациях.
# 
# #### Метрика для оценки качества модели:
# 
# - **F1-мера**  

# ### Настройка гиперпараметров логистической регрессии и тестирование модели на тренировочных данных

# In[ ]:


get_ipython().run_cell_magic('time', '', '\n# Инициализация классификатора логистической регрессии с фиксированным random_state\nlog_reg = LogisticRegression(random_state=42)\n\n# Определение диапазона гиперпараметров для настройки\nparam_grid = {\n    \'penalty\': [\'l2\', \'elasticnet\'],  # Тип регуляризации\n    \'C\': [0.1, 1, 10, 100],  # Параметр регуляризации\n    \'solver\': [\'lbfgs\', \'liblinear\'],  # Алгоритм оптимизации\n    \'max_iter\': [100, 200, 300]  # Количество итераций для сходимости\n}\n\n# Поиск оптимальных гиперпараметров с использованием GridSearchCV с кросс-валидацией\ngrid_search_log_reg = GridSearchCV(estimator=log_reg,\n                                   param_grid=param_grid,\n                                   scoring=\'f1\',  # Оценка по F1-мере\n                                   refit=\'f1\',  # Повторное обучение на лучших параметрах с использованием F1-меры\n                                   cv=5,  # Количество фолдов для кросс-валидации\n                                   verbose=1)  # Включение подробного вывода для мониторинга процесса\n\n# Обучение модели с оптимизацией гиперпараметров\ngrid_search_log_reg.fit(X_train, y_train)\n\n# Извлечение лучших параметров модели\nbest_log_reg_model = grid_search_log_reg.best_estimator_\n\n# Проверим все доступные метрики в результатах\nprint(grid_search_log_reg.cv_results_.keys())  # Для диагностики доступных метрик\n\n# Получение метрик для лучшей модели\nmetrics_columns = [\'mean_test_score\']  # Используем mean_test_score для получения итоговой метрики\nbest_log_reg_metrics = pd.DataFrame(grid_search_log_reg.cv_results_)[metrics_columns].iloc[grid_search_log_reg.best_index_]\n\n# Сохраняем лучшую F1-метрику в отдельную переменную с новым именем\nlog_reg_f1 = best_log_reg_metrics[\'mean_test_score\']\n\n# Вывод лучших параметров и метрик\nprint(f"Лучший классификатор: {best_log_reg_model}")\nprint(f"Лучшая F1-метрика: {log_reg_f1}")\ndisplay(best_log_reg_metrics)')


# ### Результаты настройки гиперпараметров логистической регрессии
# 
# **Процесс обучения**:  
# Фиттинг на 5 фолдах для каждого из 30 кандидатов, всего 150 обучений.
# 
# **Лучший классификатор**:  
# `LogisticRegression(C=1, max_iter=200, random_state=42)`
# 
# **Метрики для лучшей модели**:
# - **mean_test_f1**: 0.65881
# 
# **Время выполнения**:
# - **CPU время**: 18.6 секунды
# - **Wall time**: 19 секунд

# ### Настройка гиперпараметров случайного леса и тестирование модели на тренировочных данных

# In[ ]:


get_ipython().run_cell_magic('time', '', '\n# Инициализация классификатора случайного леса с фиксированным random_state\nrf_classifier = RandomForestClassifier(random_state=42)\n\n# Определение диапазона гиперпараметров для настройки\nparam_grid = {\n    \'max_depth\': range(10, 21, 2),  # Диапазон глубины деревьев\n    \'n_estimators\': range(150, 351, 50)  # Диапазон количества деревьев в лесу\n}\n\n# Поиск оптимальных гиперпараметров с использованием GridSearchCV с кросс-валидацией\ngrid_search_rf = GridSearchCV(estimator=rf_classifier,\n                              param_grid=param_grid,\n                              scoring=\'f1\',  # Оценка по F1-мере\n                              refit=\'f1\',  # Повторное обучение на лучших параметрах с использованием F1-меры\n                              cv=5,  # Количество фолдов для кросс-валидации\n                              verbose=1)  # Включение подробного вывода для мониторинга процесса\n\n# Обучение модели с оптимизацией гиперпараметров\ngrid_search_rf.fit(X_train, y_train)\n\n# Извлечение лучших параметров модели\nbest_rf_model = grid_search_rf.best_estimator_\n\n# Проверим все доступные метрики в результатах\nprint(grid_search_rf.cv_results_.keys())  # Для диагностики доступных метрик\n\n# Получение метрик для лучшей модели\nmetrics_columns = [\'mean_test_score\']  # Используем mean_test_score для получения итоговой метрики\nbest_rf_metrics = pd.DataFrame(grid_search_rf.cv_results_)[metrics_columns].iloc[grid_search_rf.best_index_]\n\n# Сохраняем лучшую F1-метрику в переменную forest_f1\nforest_f1 = best_rf_metrics[\'mean_test_score\']\n\n# Вывод лучших параметров и метрик\nprint(f"Лучший классификатор: {best_rf_model}")\nprint(f"Лучшая F1-метрика: {forest_f1}")\ndisplay(best_rf_metrics)')


# ### Результаты настройки гиперпараметров для случайного леса
# 
# **Процесс обучения**:  
# Фиттинг на 5 фолдах для каждого из 30 кандидатов, всего 150 обучений.
# 
# **Лучший классификатор**:  
# `RandomForestClassifier(max_depth=10, n_estimators=350, random_state=42)`
# 
# **Метрики для лучшей модели**:
# - **Лучшая F1-метрика**: 0.6597687518703677
# - **Средний тестовый F1 (mean_test_score)**: 0.659769
# 
# **Время выполнения**:
# - **CPU время**: 16 минут 34 секунды
# - **Wall time**: 16 минут 41 секунда

# ### Настройка гиперпараметров CatBoost и тестирование модели на тренировочных данных

# In[ ]:


get_ipython().run_cell_magic('time', '', '\n# Инициализация классификатора CatBoost с фиксированным random_state\ncatboost_classifier = CatBoostClassifier(random_state=42, verbose=0)\n\n# Определение диапазона гиперпараметров для настройки\nparam_grid = {\n    \'depth\': [6, 8, 10],  # Глубина дерева\n    \'learning_rate\': [0.01, 0.05, 0.1],  # Коэффициент обучения\n    \'iterations\': [500, 1000, 1500],  # Количество итераций (деревьев)\n}\n\n# Поиск оптимальных гиперпараметров с использованием GridSearchCV с кросс-валидацией\ngrid_search_catboost = GridSearchCV(estimator=catboost_classifier,\n                                    param_grid=param_grid,\n                                    scoring=\'f1\',  # Оценка по F1-мере\n                                    refit=\'f1\',  # Повторное обучение на лучших параметрах с использованием F1-меры\n                                    cv=5,  # Количество фолдов для кросс-валидации\n                                    verbose=1)  # Включение подробного вывода для мониторинга процесса\n\n# Обучение модели с оптимизацией гиперпараметров\ngrid_search_catboost.fit(X_train, y_train)\n\n# Извлечение лучших параметров модели\nbest_catboost_params = grid_search_catboost.best_params_\n\n# Извлечение лучших метрик модели\nmetrics_columns = [\'mean_test_score\']  # Используем mean_test_score для получения итоговой метрики\nbest_catboost_metrics = pd.DataFrame(grid_search_catboost.cv_results_)[metrics_columns].iloc[grid_search_catboost.best_index_]\n\n# Сохраняем лучшую F1-метрику в переменную catboost_f1\ncatboost_f1 = best_catboost_metrics[\'mean_test_score\']\n\n# Вывод лучших гиперпараметров и метрик\nprint(f"Лучшие гиперпараметры: {best_catboost_params}")\nprint(f"Лучшая F1-метрика: {catboost_f1}")\ndisplay(best_catboost_metrics)')


# ### Результаты настройки гиперпараметров для классификатора CatBoost
# 
# **Процесс обучения**:  
# Фиттинг на 5 фолдах для каждого из 27 кандидатов, всего 135 обучений.
# 
# **Лучшие гиперпараметры**:  
# - **depth**: 6  
# - **iterations**: 500  
# - **learning_rate**: 0.01
# 
# **Метрики для лучшей модели**:  
# - **mean_test_f1**: 0.664909
# 
# **Время выполнения**:  
# - **CPU время**: 3 часа 27 минут  
# - **Wall time**: 38 минут 45 секунд

# ### Настройка, обучения нейронной сети и тестирование на тренировочных данных

# In[ ]:


# Установка начального random seed для воспроизводимости
torch.manual_seed(42)
np.random.seed(42)

# Параметры архитектуры нейронной сети
input_size = X_train.shape[1]  # Размерность входных данных
hidden_layer_1 = 16  # Число нейронов в первом скрытом слое
hidden_layer_2 = 8   # Число нейронов во втором скрытом слое
output_size = 1      # Число выходных нейронов

# Определение архитектуры нейронной сети
net = nn.Sequential(
    nn.Linear(input_size, hidden_layer_1),
    nn.ReLU(),  # Используем ReLU вместо Sigmoid для более эффективного обучения
    nn.BatchNorm1d(hidden_layer_1),  # Нормализация первого скрытого слоя
    nn.Linear(hidden_layer_1, hidden_layer_2),
    nn.ReLU(),
    nn.BatchNorm1d(hidden_layer_2),  # Нормализация второго скрытого слоя
    nn.Linear(hidden_layer_2, output_size),
    nn.Sigmoid()  # Для задач бинарной классификации
)

# Настройка оптимизатора и функции потерь
learning_rate = 1e-3  # Скорость обучения
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)  # Оптимизатор Adam
criterion = nn.BCELoss()  # Функция потерь для бинарной классификации


# In[ ]:


get_ipython().run_cell_magic('time', '', '\n# Старт кросс-валидации\nn_splits = 5  # Количество фолдов для кросс-валидации\nskf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)\n\n# Количество эпох и размер батча\nnum_epochs = 200\nbatch_size = 128\nnum_batches = ceil(len(X_train) / batch_size)\n\n# Хранение результатов для каждого фолда\nf1_scores = []\nbest_f1 = 0\n\n# Кросс-валидация\nfor fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):\n    print(f"Fold {fold + 1}/{n_splits}")\n    \n    # Разделение данных на тренировочные и валидационные\n    X_train_fold, X_val_fold = X_train.iloc[train_idx], X_train.iloc[val_idx]\n    y_train_fold, y_val_fold = y_train.iloc[train_idx], y_train.iloc[val_idx]\n\n    # Преобразование данных в тензоры\n    X_train_fold_torch = torch.FloatTensor(X_train_fold.values)\n    y_train_fold_torch = torch.FloatTensor(y_train_fold.values)\n    X_val_fold_torch = torch.FloatTensor(X_val_fold.values)\n    y_val_fold_torch = torch.FloatTensor(y_val_fold.values)\n    \n    # Переинициализация модели и оптимизатора для каждого фолда\n    net.apply(init_weights)  # Функция для инициализации весов\n    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)\n    \n    # Обучение модели на текущем фолде\n    for epoch in range(num_epochs):\n        order = np.random.permutation(len(X_train_fold_torch))\n        for batch_idx in range(num_batches):\n            start_idx = batch_idx * batch_size\n            batch_indexes = order[start_idx:start_idx + batch_size]\n            X_batch = X_train_fold_torch[batch_indexes]\n            y_batch = y_train_fold_torch[batch_indexes]\n\n            # Обучение нейросети\n            optimizer.zero_grad()\n            preds = net(X_batch).flatten()\n\n            loss_value = criterion(preds, y_batch)\n            loss_value.backward()\n            optimizer.step()\n\n        # Тестирование модели на валидационном наборе\n        net.eval()\n        with torch.no_grad():\n            net_preds = net(X_val_fold_torch).flatten()\n            current_f1 = f1_score(y_val_fold, net_preds.round().detach().numpy())\n            net_auc_roc = roc_auc_score(y_val_fold, net_preds.detach().numpy())\n        \n        print(f"Epoch {epoch + 1}/{num_epochs}")\n        print(f"F1-мера для фолда {fold + 1}: {round(current_f1, 3)}")\n        print("-" * 40)\n\n        # Обновление лучшей F1-меры, если текущая выше\n        if current_f1 > best_f1:\n            best_f1 = current_f1\n            print(f"Лучшее значение F1-меры для фолда {fold + 1}: {best_f1}")\n\n    # Сохранение метрик для текущего фолда\n    f1_scores.append(best_f1)\n\n# Вывод средних значений по всем фолдам\nprint(f"Средняя F1-мера по всем фолдам: {round(np.mean(f1_scores), 3)}")\nprint(f"Лучшее значение F1-меры: {best_f1}")')


# ### Результаты обучения нейронной сети
# 
# **Средняя F1-мера по всем фолдам:** 0.71  
# **Лучшее значение F1-меры:** 0.710
# 
# **Время выполнения:**
# - CPU time: 41min 25s
# - Wall time: 6min 55s

# In[ ]:


models_metrics = pd.DataFrame(
    data=[[0.659, 0.66, 0.665, 0.71],
         ],
    columns=['LogisticRegression', 'RandomForestClassifier', 'CatBoostClassifier', 'NeuralNet'],
    index=['F1']
)

display(models_metrics)


# Лучшую F1 метрику показала логистическая регрессия.

# In[ ]:


get_ipython().run_cell_magic('time', '', '\n# Преобразование данных в тензоры\nX_train_torch = torch.FloatTensor(X_train.values)  # Используем .values для получения NumPy массива\nX_test_torch = torch.FloatTensor(X_test.values)\ny_train_torch = torch.FloatTensor(y_train.values)\ny_test_torch = torch.FloatTensor(y_test.values)\n\n# Количество эпох, размер одного батча и их количество\nnum_epochs = 200\nbatch_size = 128\nnum_batches = ceil(len(X_train_torch) / batch_size)\n\n# Список для хранения значений F1-меры\nf1_scores = []\n\n# Обучение нейросети\nfor epoch in range(num_epochs):\n    order = np.random.permutation(len(X_train_torch))\n    for batch_idx in range(num_batches):\n        start_idx = batch_idx * batch_size\n        \n        optimizer.zero_grad()\n\n        batch_indexes = order[start_idx:start_idx + batch_size]\n        X_batch = X_train_torch[batch_indexes]\n        y_batch = y_train_torch[batch_indexes]\n\n        # Обучение нейросети\n        preds = net.forward(X_batch).flatten()\n\n        loss_value = criterion(preds, y_batch)  # Используем правильную переменную для потерь\n        loss_value.backward()\n        \n        optimizer.step()\n    \n    # Тестирование нейросети\n    if epoch % 1 == 0:\n        net.eval()\n        with torch.no_grad():\n            net_preds = net.forward(X_test_torch).flatten()\n            net_f1 = round(f1_score(y_test, net_preds.round().detach().numpy()), 3)\n            net_auc_roc = round(roc_auc_score(y_test, net_preds.round().detach().numpy()), 3)\n            f1_scores.append(net_f1)  # Добавляем значение F1 в список\n            print(f\'Epoch: {epoch}\')\n            print(f\'F1-мера = {net_f1}\')\n            print(f\'AUC-ROC = {net_auc_roc}\\n\')\n\n# Средняя F1-мера по всем эпохам\naverage_f1 = round(np.mean(f1_scores), 3)\nprint(f"Средняя F1-мера: {average_f1}")')


# ### Выводы по моделям
# 
# По результатам сравнительного анализа F1-меры можно отметить, что модели демонстрируют разные уровни эффективности, особенно на тренировочных и тестовых выборках.
# 
# - **Логистическая регрессия** показала результат F1 на тренировочных данных 0.659, что является достаточно стабильным, но уступает остальным моделям. Несмотря на это, её результаты могут быть полезны в задачах с простыми зависимостями между признаками, где высокая интерпретируемость модели является важным фактором.
# 
# - **Случайный лес** с F1-мерой на уровне 0.66 продемонстрировал хорошую сбалансированность между точностью и полнотой. Эта модель является довольно устойчивой, особенно при наличии большого объема данных и в задачах с высокой вариативностью признаков.
# 
# - **CatBoost**, с F1-мерой 0.665, также показал хорошие результаты, немного превосходя случайный лес. Это может свидетельствовать о его способности более эффективно работать с нелинейными зависимостями в данных и обеспечивать более стабильные результаты в разнообразных задачах.
# 
# - **Нейронная сеть**, с F1-мерой 0.71 на тренировочных данных, показала наилучший результат среди всех моделей. На тестовых данных F1-мера составила 0.674. Это говорит о высокой гибкости модели в работе с данными и способности более точно захватывать сложные зависимости. Однако, нейронные сети требуют более сложной настройки гиперпараметров и более длительного времени на обучение, что может стать недостатком в условиях ограниченных ресурсов. 
# 
# ### Результаты моделей:
# 
# - **Logistic Regression:**  
#   - Cross-validated F1: 0.659 
# 
# - **Random Forest Classifier:**  
#   - Cross-validated F1: 0.66
# 
# - **CatBoost Classifier:**  
#   - Cross-validated F1: 0.665 
# 
# - **Neural Network:**      
#   - Cross-validated F1: 0.71
#   - F1-мера на тестовых данных: 0.674
# 
# ### Заключение
# 
# Сравнив результаты моделей, можно сделать вывод, что **Neural Network** показывает лучший результат по F1-мере на тренировочных данных, что делает её предпочтительным выбором для более сложных задач. Однако, в случае ограничений по времени или ресурсу, **Random Forest** и **CatBoost** остаются хорошими альтернативами. **Logistic Regression** хотя и уступает по результатам, всё же может быть полезна в задачах, требующих высокой интерпретируемости и простоты.

# ## Проведите анализ важности факторов ДТП

# In[ ]:


# Матрица ошибок Neural Network
cm, precision, recall = plot_confusion_matrix_nn(net, X_test, y_test, title="Матрица ошибок Neural Network")


# ### Вывод по матрице ошибок моделей
# 
# Давайте рассмотрим результаты матриц ошибок для модели **Neural Network**. Это важно для принятия решений, направленных на улучшение безопасности дорожного движения.
# 
# #### Основные метрики матрицы ошибок:
# 
# 1. **True Positives (TP)** — правильно предсказанные случаи ДТП. Модель верно предсказала, что ДТП произойдет, и оно действительно произошло.
# 2. **True Negatives (TN)** — правильно предсказанные случаи, когда ДТП не произошло. Модель верно предсказала, что авария не состоится.
# 3. **False Positives (FP)** — ложные положительные случаи. Модель ошибочно предсказала, что ДТП произойдет, хотя его не было.
# 4. **False Negatives (FN)** — ложные отрицательные случаи. Модель ошибочно предсказала, что ДТП не произойдет, хотя оно случилось.
# 
# #### Пример для **Neural Network**:
# 
# - **True Positives (TP)**: 3829 — модель правильно предсказала 3829 случая ДТП.
# - **True Negatives (TN)**: 2123 — модель правильно предсказала 2123 случая, когда ДТП не произошло.
# - **False Positives (FP)**: 2243 — модель ошибочно предсказала 2243 случаев ДТП, которых не было.
# - **False Negatives (FN)**: 1469 — модель ошибочно не предсказала 1469 случаев ДТП.
# 
# #### Интерпретация метрик:
# 
# 1. **Precision (Точность)**: 63.1% — из всех предсказанных случаев ДТП, 63.1% действительно являются авариями. Это важно для минимизации ложных срабатываний, например, в ситуациях, когда необходимо избежать ненужных вмешательств или решений.
#    
# 2. **Recall (Полнота)**: 72.3% — из всех настоящих случаев ДТП модель смогла предсказать 72.3%. Это означает, что модель эффективно обнаруживает большинство аварий, что критично для предсказания серьезных инцидентов, требующих вмешательства.
# 
# 3. **F1 Score**: 0.674 — метрика, которая объединяет точность и полноту. В нашем случае она составляет 0.674, что говорит о сбалансированном подходе модели к обнаружению аварий и минимизации ложных тревог.
# 
# #### Бизнес-интерпретация:
# 
# - **Precision (Точность)** для бизнеса можно интерпретировать как **стоимость ложных срабатываний**. В контексте ДТП это может означать излишние действия, такие как проверка или реагирование на ложные тревоги, что приведет к лишним затратам ресурсов.
#   
# - **Recall (Полнота)** интерпретируется как **потери из-за пропущенных случаев**. Это важно в случаях, когда необходимо как можно скорее выявить потенциально опасные ситуации, например, для предотвращения аварий в критичных точках на дорогах.
# 
# ### Заключение
# 
# - **Logistic Regression** показала отличные результаты по **Recall**, что означает высокую способность модели находить все реальные ДТП.
# - **Logistic Regression** будет полезной, если приоритетом является нахождение всех потенциальных случаев ДТП, даже если это приведет к некоторым ложным срабатываниям.
# 
# Эти выводы помогают в выборе подходящей модели для прогнозирования вероятности ДТП в зависимости от бизнес-целей: будь то минимизация ложных срабатываний или максимизация охвата критичных случаев.
# 

# In[ ]:


# Получение весов из первого скрытого слоя
hidden_layer_weights = net[0].weight.detach().numpy()  # Получаем веса первого слоя

# Суммируем веса по всем нейронам скрытого слоя, чтобы получить вклад каждого признака
importance = np.abs(hidden_layer_weights).sum(axis=0)

# Создание Series для удобства визуализации
importance_series = pd.Series(importance, index=df.drop('at_fault', axis=1).columns)

# Настройка графика
plt.figure(figsize=(12, 8))  # Размер графика
importance_series.sort_values(ascending=True).plot(kind='barh', color=plt.cm.viridis((importance_series.sort_values(ascending=True) - importance_series.min()) / (importance_series.max() - importance_series.min())), grid=True)

# Настройка оформления графика
plt.title('Важность признаков для предсказания ДТП через скрытый слой нейронной сети', fontsize=18, weight='bold')  # Заголовок
plt.xlabel('Абсолютные веса признаков', fontsize=14)  # Подпись оси X
plt.ylabel('Признаки', fontsize=14)  # Подпись оси Y
plt.xticks(fontsize=12)  # Размер шрифта для оси X
plt.yticks(fontsize=12)  # Размер шрифта для оси Y

# Добавление подписей с коэффициентами на графике
for index, value in enumerate(importance_series.sort_values(ascending=True)):
    plt.text(value, index, f'{value:.3f}', va='center', fontsize=12, color='black')

plt.tight_layout()  # Оптимизация размещения элементов

# Отображение графика
plt.show()


# Для рассмотрения графика зависимости самого важного фактора, влияющее на предсказание модели - `distance` - воспользуемся корреляцией phik, которая помогает определить коэффициент корреляции с категориальными параметрами. Рассмотрение количества ДТП в зависимости от Номера географического района не представляется возможным, поскольку количеством таких районов 498.

# In[ ]:


# Построение box plot без точек (выбросов)
plt.figure(figsize=(10, 6))
sns.boxplot(x='at_fault', y='distance', data=df, palette='Set2', showfliers=False)

# Настройка меток оси X
plt.xticks([0, 1], ['Без ДТП', 'ДТП'])

# Настройка заголовков и подписей
plt.title('Распределение вероятности ДТП от расстояния до главной дороги', fontsize=16)
plt.xlabel('Вероятность ДТП (at_fault)', fontsize=12)
plt.ylabel('Расстояние от главной дороги (distance)', fontsize=12)

# Отображение графика
plt.show()


# **Промежуточный вывод**
# 
# Анализируя график, отображающий зависимость вероятности ДТП от расстояния до главной дороги, можно выделить несколько ключевых наблюдений о влиянии этого фактора на безопасность на дорогах:
# 
# 1. Медианное значение расстояния от главной дороги в случаях с ДТП и без ДТП примерно одинаковое.
# 2. Если расстояние от главной дороги больше — вероятность ДТП выше.
# 
# В целом, расстояние от главной дороги не оказывает значительного влияния на вероятность ДТП.

# ## Выводы

# ### Общий вывод по модели
# 
# В данной задаче лучшей моделью оказалась **Neural Network**, которая продемонстрировал высокую F1-меру на тестовой выборке, что делает его эффективным инструментом для предсказания вероятности ДТП.
# 
# Однако важно отметить, что точность любой модели машинного обучения напрямую зависит от качества и полноты данных, а также от выбора признаков. Это подчеркивает важность тщательной подготовки данных и их анализа для получения наилучших результатов.
# 
# Для дальнейшего улучшения модели можно рассмотреть следующие шаги:
# - Уточнение критериев выбора признаков, включая анализ корреляции между признаками.
# - Применение более сложных моделей машинного обучения, например, ансамблевых методов.
# - Регулярное обновление данных и пересмотр модели по мере появления новых данных или изменений в законодательстве.
# 
# ### Анализ важности факторов
# 
# Анализ важности факторов, выполненный с помощью функции `feature_importances_` после обучения модели, позволил выделить ключевые признаки, наиболее сильно влияющие на вероятность ДТП. Наибольшее значение имеет **расстояние от главной дороги** (`distance`). Также значимым фактором является **тип кузова** (`vehicle_type`).
# 
# ### Заключение
# 
# Таким образом, создание эффективной системы оценки риска для разрешения использовать авто крайне важно, но требует постоянного обновления данных и анализа новых факторов. Для улучшения модели необходимо уточнить критерии выбора признаков, рассмотреть использование более сложных методов машинного обучения и регулярно обновлять данные для актуальности модели.
# 
