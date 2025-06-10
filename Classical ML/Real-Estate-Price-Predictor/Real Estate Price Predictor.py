#!/usr/bin/env python
# coding: utf-8

# ### Анализ рыночной стоимости недвижимости в Санкт-Петербурге и ЛО  
# 
# ## Описание проекта  
# 
# **Цель:**  
# Разработка автоматизированной системы оценки рыночной стоимости квартир на основе данных Яндекс.Недвижимости для выявления аномалий и мошеннических объявлений.  
# 
# **Источник данных:**  
# Архив объявлений о продаже квартир в Санкт-Петербурге и Ленинградской области за несколько лет.  
# 
# **Типы данных:**  
# 1. **Пользовательские:**  
#    - Площадь, количество комнат  
#    - Этаж, высота потолков  
#    - Состояние ремонта  
# 
# 2. **Автоматические (геосервисы):**  
#    - Расстояние до центра/аэропорта  
#    - Количество парков/водоёмов в районе  
#    - Близость к метро  
# 
# ## Ключевые задачи  
# 
# 1. **Исследовательский анализ:**  
#    - Выявление корреляции параметров с ценой  
#    - Анализ распределения цен по районам  
#    - Обнаружение аномальных значений  
# 
# 2. **Построение модели:**  
#    - Предсказание рыночной стоимости  
#    - Определение значимости параметров  
# 
# 3. **Визуализация:**  
#    - Геораспределение цен  
#    - Топ-факторов стоимости  

# ### Откройте файл с данными и изучите общую информацию. 

# In[1]:


import pandas as pd


# In[2]:


try:
    data = pd.read_csv('/datasets/real_estate_data.csv', sep='\t')
except:
    data = pd.read_csv('https://code.s3.yandex.net/datasets/real_estate_data.csv', sep='\t')


# In[3]:


data.head()


# In[4]:


data.info()


# In[5]:


data.hist(figsize=(15, 20))


# На гистограммах виден сильный сдвиг влево, это означает большое количество аномалий и выбросов Кроме того, можно предположить, то что наибольшее количество квартир имеют минимальные значения своих характеристик (общая площадь, жилая площадь, этаж, количество комнат и тд). 

# ### Предобработка данных

# In[6]:


data.isna().sum()


# In[7]:


# Заполняем пропуски высоты потолков медианным значением
data['ceiling_height'] = data['ceiling_height'].fillna(data['ceiling_height'].median())


# In[8]:


# Заполняем пропуски количества этажей в здании значением не ниже этажа квартиры, соответственно просто этажом квартиры.
data['floors_total'] = data['floors_total'].dropna()


# In[9]:


# Заполняем пропуски площади кухни нулем. Кухня может отсутствовать в квартире, например если это студия.
data['kitchen_area'] = data['kitchen_area'].fillna(0)


# In[10]:


# Заполняем пропуск площади жилой зоны разницей между общей площадью и площадью кухни
data['living_area'] = data['living_area'].fillna(data['total_area'] - data['kitchen_area'])


# In[11]:


# Заполняем пропуски информации по балконам медианным значением.
data['balcony'] = data['balcony'].fillna(data['balcony'].median())


# In[12]:


# Удаляем пропуски информации по локациям, так как их всего 49.
data = data.dropna(subset=['locality_name'])


# In[13]:


# Для того чтобы заполнить пропуски по ближайшему аэропорту, 
# берем медианное значение расстояния до ближайшего аэропорта для каждого населенного пункта. 
data['airports_nearest'] = data['airports_nearest'].fillna(data
                                                           .groupby('locality_name')['airports_nearest']
                                                           .transform('median'))
# Как выясняется, во многих населенных пунктов нет ни одного значения до ближайшего аэропорта.


# In[14]:


# Воспользуемся медианным значением медиан расстояний до аэропортов, 
# чтобы заполнить пропуски для населенных пунктов, где отсутствует какая либо информация.
airports_nearest_median = data.groupby('locality_name')['airports_nearest'].median()
medians_of_airports_nearest_median = airports_nearest_median.median()
data['airports_nearest'] = data['airports_nearest'].fillna(medians_of_airports_nearest_median)


# In[15]:


# Для того чтобы заполнить пропуски до центра Санкт-Петербурга, 
# берем медианное значение расстояния для каждого населенного пункта. 
data['cityCenters_nearest'] = data['cityCenters_nearest'].fillna(data
                                                                 .groupby('locality_name')['cityCenters_nearest']
                                                                 .transform('median'))
# Как выясняется, во многих населенных пунктов нет ни одного значения.


# In[16]:


# Воспользуемся медианным значением медиан расстояний до центра Санкт-Петербурга, 
# чтобы заполнить пропуски для населенных пунктов, где отсутствует какая либо информация.
cityCenters_nearest_median = data.groupby('locality_name')['cityCenters_nearest'].median()
medians_of_cityCenters_nearest_median = cityCenters_nearest_median.median()
data['cityCenters_nearest'] = data['cityCenters_nearest'].fillna(medians_of_cityCenters_nearest_median)


# In[17]:


# Для того чтобы заполнить пропуски есть ли парк на расстоянии до 3км или нет, 
# берем медианное значение расстояния для каждого населенного пункта. 
data['parks_around3000'] = data['parks_around3000'].fillna(data
                                                    .groupby('locality_name')['parks_around3000']
                                                    .transform('median'))
# В датафрейме много сёл. Далеко не в каждом селе есть парки, 
# да и не только в таких населенных пунктах, как сёла могут они отсутствовать, 
# а уже тем более на расстоянии до 3км. 
# По этому можем предположить, что пропущенные значения обозначают отсутствие парков поблизости. 
# Заменим оставшиеся пропущенные значения на ноль


# In[18]:


data['parks_around3000'] = data['parks_around3000'].fillna(0)


# In[19]:


for t in data['locality_name'].unique():
    data.loc[(data['locality_name'] == t) & 
             (data['parks_nearest'].isna()) & 
             (data['parks_around3000'] >= 1), 'parks_nearest'] = \
    data.loc[(data['locality_name'] == t), 'parks_nearest'].median()


# In[20]:


# Для того чтобы заполнить пропуски есть ли пруд на расстоянии до 3км или нет, 
# берем медианное значение расстояния для каждого населенного пункта. 
data['ponds_around3000'] = data['ponds_around3000'].fillna(data
                                                    .groupby('locality_name')['ponds_around3000']
                                                    .transform('median'))
# В датафрейме много сёл. Далеко не в каждом селе есть пруд, 
# да и не только в таких населенных пунктах, как сёла могут они отсутствовать, 
# а уже тем более на расстоянии до 3км. 
# По этому можем предположить, что пропущенные значения обозначают отсутствие прудов поблизости. 
# Заменим оставшиеся пропущенные значения на ноль


# In[21]:


data['ponds_around3000'] = data['ponds_around3000'].fillna(0)


# In[22]:


data['is_apartment'] = data['is_apartment'].fillna(False)


# In[23]:


for t in data['locality_name'].unique():
    data.loc[(data['locality_name'] == t) & 
             (data['ponds_nearest'].isna()) & 
             (data['ponds_around3000'] >= 1), 'ponds_nearest'] = \
    data.loc[(data['locality_name'] == t), 'ponds_nearest'].median()


# Пропуски в данных могут возникать по разным причинам:
# 
# 1. Некоторые значения могут не быть доступны из-за ошибок ввода или технических проблем при сборе данных.
# 
# 2. Некоторые значения могут быть пропущены, потому что люди не хотят или не могут предоставить информацию. Например, владельцы квартир могут не знать точного расстояния до ближайшего аэропорта, парка или водоема.
# 
# 3. Пропуски могут также возникать из-за ошибок в обработке данных, например, из-за ошибок при переносе данных из одной базы данных в другую.
# 
# 4. Некоторые значения могут быть пропущены намеренно, так как они не имеют значения для анализа. Например, если апартаменты не указаны, то можно считать, что это обычная квартира.

# In[24]:


data['first_day_exposition'] = pd.to_datetime(data['first_day_exposition'], format='%Y-%m-%dT%H:%M:%S')


# In[25]:


print('Неявные дубликаты до обработки:', data['locality_name'].nunique())


# Изменение типа данных коснулось только столбца: first_day_exposition (сколько дней было размещено объявление (от публикации до снятия)). Причиной такого решения стало то, что для дальнейшего анализа нам будет нужен формат datetime64 в этом столбце. Изменять тип данных с float64 на int64 я не стал, так как необходимые нам вычислительные операции в дальнейшем - доступны для обоих типов данных.

# In[26]:


locality_name_unique = ['коттеджный посёлок', 'посёлок городского типа', 
                        'городской поселок', 'поселок городского типа', 
                        'городской посёлок', 'посёлок при железнодорожной станции', 
                        'посёлок при железнодорожной станции',  'посёлок']

data['locality_name'] = data['locality_name'].replace(locality_name_unique,'поселок',regex=True)


# In[27]:


print('Неявные дубликаты после обработки:', data['locality_name'].nunique())


# In[28]:


data = data[(data['ceiling_height'] < 100) & (data['ceiling_height'] > 1.2)]
data.loc[data['ceiling_height'] >= 20 , 'ceiling_height'] /= 10
data = data[data['ceiling_height'] < 8]


# In[29]:


data = data[data['airports_nearest'] != 0]


# Основной объем аномальных данных содержал столбец с информацией о высоте потолка (ceiling_height). Там где было возможно логически понять, то что произошла ошибка в указании плавающей точки (в данных указана высота 25 метров, но логически можно понять, то что это 2.5 метра) было исправлено это. Данные которые не поддавались логическому объяснению - удалил.
# 
# Еще нашел аномалию в расстояние до аэропорта. Удалил строку, содержащую расстояние до аэропорта - 0.
# 

# ### Посчитайте и добавьте в таблицу новые столбцы

# In[30]:


data['price_per_sqm'] = data['last_price'] / data['total_area']


# In[31]:


data['weekday'] = data['first_day_exposition'].dt.weekday


# In[32]:


data['month'] = data['first_day_exposition'].dt.month


# In[33]:


data['year'] = data['first_day_exposition'].dt.year


# In[34]:


def get_floor_type(row):
    if row['floor'] == 1:
        return 'первый'
    elif row['floor'] == row['floors_total']:
        return 'последний'
    else:
        return 'другой'
    
data['floor_type'] = data.apply(get_floor_type, axis=1)


# In[35]:


data['city_centers_km'] = (data['cityCenters_nearest'] / 1000).round()


# ### Проведите исследовательский анализ данных

# In[36]:


data_filtered = data


# In[37]:


# Избавляемся от выбросов в total_area.
print('Количество строк до обработки выбросов:', len(data['total_area']))
# Сохраним в переменной t количество строк до обработки выбросов.
t = len(data['total_area'])
data_filtered = data_filtered[data_filtered['total_area'] < 200]
print('Количество строк после обработки выбросов:', len(data_filtered['total_area']))
print('Процент обработанных выбросов:', 1 - len(data_filtered['total_area']) / len(data['total_area']))
data = data[data['total_area'] < 200]


# In[38]:


# Избавляемся от выбросов в living_area.
print('Количество строк до обработки выбросов:', len(data['living_area']))
data_filtered = data_filtered[data_filtered['living_area'] < 150]
print('Количество строк после обработки выбросов:', len(data_filtered['living_area']))
print('Процент обработанных выбросов:', 1 - len(data_filtered['living_area']) / len(data['living_area']))
data = data[data['living_area'] < 150]


# In[39]:


# Избавляемся от выбросов в kitchen_area.
print('Количество строк до обработки выбросов:', len(data['kitchen_area']))
data_filtered = data_filtered[data_filtered['kitchen_area'] < 50]
print('Количество строк после обработки выбросов:', len(data_filtered['kitchen_area']))
print('Процент обработанных выбросов:', 1 - len(data_filtered['kitchen_area']) / len(data['kitchen_area']))
data = data[data['kitchen_area'] < 50]


# In[40]:


# Избавляемся от выбросов в last_price.
print('Количество строк до обработки выбросов:', len(data['last_price']))
data_filtered = data_filtered[data_filtered['last_price'] < 40000000]
print('Количество строк после обработки выбросов:', len(data_filtered['last_price']))
print('Процент обработанных выбросов:', 1 - len(data_filtered['last_price']) / len(data['last_price']))
data = data[data['last_price'] < 40000000]


# In[41]:


# Избавляемся от выбросов в rooms.
print('Количество строк до обработки выбросов:', len(data['rooms']))
data_filtered = data_filtered[data_filtered['rooms'] < 6]
print('Количество строк после обработки выбросов:', len(data_filtered['rooms']))
print('Процент обработанных выбросов:', 1 - len(data_filtered['rooms']) / len(data['rooms']))
data = data[data['rooms'] < 6]


# In[42]:


# Избавляемся от выбросов в ceiling_height.
print('Количество строк до обработки выбросов:', len(data['ceiling_height']))
data_filtered = data_filtered[data_filtered['ceiling_height'] < 5]
print('Количество строк после обработки выбросов:', len(data_filtered['ceiling_height']))
print('Процент обработанных выбросов:', 1 - len(data_filtered['ceiling_height']) / len(data['ceiling_height']))
data = data[data['ceiling_height'] < 5]


# In[43]:


# Избавляемся от выбросов в floor.
print('Количество строк до обработки выбросов:', len(data['floor']))
data_filtered = data_filtered[data_filtered['floor'] < 25]
print('Количество строк после обработки выбросов:', len(data_filtered['floor']))
print('Процент обработанных выбросов:', 1 - len(data_filtered['floor']) / len(data['floor']))
data = data[data['floor'] < 25]


# In[44]:


# Избавляемся от выбросов в floors_total.
print('Количество строк до обработки выбросов:', len(data['floors_total']))
data_filtered = data_filtered[data_filtered['floors_total'] < 30]
print('Количество строк после обработки выбросов:', len(data_filtered['floors_total']))
print('Процент обработанных выбросов:', 1 - len(data_filtered['floors_total']) / len(data['floors_total']))
data = data[data['floors_total'] < 30]


# In[45]:


# Избавляемся от выбросов в airports_nearest.
print('Количество строк до обработки выбросов:', len(data['airports_nearest']))
data_filtered = data_filtered[data_filtered['airports_nearest'] < 70000]
print('Количество строк после обработки выбросов:', len(data_filtered['airports_nearest']))
print('Процент обработанных выбросов:', 1 - len(data_filtered['airports_nearest']) / len(data['airports_nearest']))
data = data[data['airports_nearest'] < 70000]


# In[46]:


# Избавляемся от выбросов в parks_nearest.
print('Количество строк до обработки выбросов:', len(data['parks_nearest']))
data_filtered = data_filtered[(data_filtered['parks_nearest'] < 1500) | (data_filtered['parks_nearest'].isna())]
print('Количество строк после обработки выбросов:', len(data_filtered['parks_nearest']))
print('Процент обработанных выбросов:', 1 - len(data_filtered['parks_nearest']) / len(data['parks_nearest']))
data = data[(data['parks_nearest'] < 1500) | (data['parks_nearest'].isna())]


# In[47]:


print('Процент обработанных выбросов составил:', 1 - len(data['parks_nearest']) / t)


# In[48]:


data['total_area'].hist(bins=50)


# In[49]:


data['living_area'].hist(bins=50)


# In[50]:


data['kitchen_area'].hist(bins=50)


# In[51]:


data['last_price'].hist(bins=100)


# In[52]:


data['rooms'].hist(bins=10)


# In[53]:


data['ceiling_height'].hist(bins=10)


# In[54]:


data['floor'].hist(bins=20)


# In[55]:


data['floor_type'].hist(bins=5)


# In[56]:


data['floors_total'].hist(bins=20)


# In[57]:


data['cityCenters_nearest'].hist(bins=10)


# In[58]:


data['airports_nearest'].hist(bins=10)


# In[59]:


data['parks_nearest'].hist(bins=50)


# In[60]:


data['weekday'].hist(bins=7)


# In[61]:


data['month'].hist(bins=12)


# Из приведенных выше гистограмм мы видим, то что у большинства квартир на рынке средняя общая площадь составляет от 30 до 50 квадратных метров. Жилая зона в среднем составляет  до 50 квадратных метров. Цена за усредненную квартиру составляет до 5 миллионов рублей. Количество комнат - от 1 до 3. Высота потолка 2.5 метра. Квартира находится в с 1 до 7 этажа, но чаще всего это пятиэтажный дом. На втором месте по частоте - девятиэтажка. Расстояние до центра Санкт-Петербурга в среднем 15 км. Расстояние до ближайшего аэропорта - 45 км. Жилое помещение часто находится недалеко от парка, обычно расстояние составляет около 500 метров. Объявление о продаже квартир чаще всего размещают в будние дни и в начале и конце года. 

# In[62]:


data['days_exposition'].hist(bins=50)


# In[63]:


print('Среднее количество дней, которое существует объявление:', data['days_exposition'].mean())
print('Медианное  количество дней, которое существует объявление:', data['days_exposition'].median())


# Средний срок продажи квартир составляет 180 дней. Медианный срок - 95 дней. Быстрыми продажами квартир можно считать продажи до 200 дней после публикации объявления. Необычно долгими можно считать продажи квартир, чьи объявления созданы более 600 дней назад.

# In[64]:


data_total_area_last_price = data.pivot_table(index='total_area', values='last_price')
data_total_area_last_price = data_total_area_last_price.query('total_area < 400 and last_price < 20000000')
data_total_area_last_price['total_area'] = data_total_area_last_price.index
data_total_area_last_price.plot(x='total_area', y='last_price', kind='scatter', alpha=0.3)
print('Коэффициент корреляции между общей площадью и ценой недвижимости равен', 
      data_total_area_last_price['total_area'].corr(data_total_area_last_price['last_price']))


# Видна сильная зависимость цены от общей площади.
# 

# In[65]:


data_living_area_last_price = data.pivot_table(index='living_area', values='last_price')
data_living_area_last_price = data_living_area_last_price.query('living_area < 300 and last_price < 20000000')
data_living_area_last_price['living_area'] = data_living_area_last_price.index
data_living_area_last_price.plot(x='living_area', y='last_price', kind='scatter', alpha=0.3)
print('Коэффициент корреляции между жилой площадью и ценой недвижимости равен', 
      data_living_area_last_price['living_area'].corr(data_living_area_last_price['last_price']))


# Видна сильная зависимость цены от жилой площади.

# In[66]:


data_kitchen_area_last_price = data.pivot_table(index='kitchen_area', values='last_price')
data_kitchen_area_last_price = data_kitchen_area_last_price.query('kitchen_area < 60 and last_price < 20000000')
data_kitchen_area_last_price['kitchen_area'] = data_kitchen_area_last_price.index
data_kitchen_area_last_price.plot(x='kitchen_area', y='last_price', kind='scatter', alpha=0.3)
print('Коэффициент корреляции между площадью кухни и ценой недвижимости равен', 
      data_kitchen_area_last_price['kitchen_area'].corr(data_kitchen_area_last_price['last_price']))


# Видна сильная зависимость цены от площади кухни.

# In[67]:


data_rooms_last_price = data.pivot_table(index='rooms', values='last_price')
data_rooms_last_price = data_rooms_last_price.query('rooms < 10 and last_price < 20000000')
data_rooms_last_price['rooms'] = data_rooms_last_price.index
data_rooms_last_price.plot(x='rooms', y='last_price')
print('Коэффициент корреляции между количеством комнат и ценой недвижимости равен', 
      data_rooms_last_price['rooms'].corr(data_rooms_last_price['last_price']))


# Видна сильная зависимость цены от количества комнат.

# In[68]:


data_floor_type_last_price = data.query('last_price < 200000000')
data_floor_type_last_price.plot(x='floor_type', y='last_price', kind='scatter', alpha=0.3)


# Самые дешевые квартиры наблюдаются на первом этаже. Далее по цене идут квартиры на последнем этаже. Самые дорогие квартиры находятся между первым и последним этажом. 

# In[69]:


data_weekday = data.query('last_price < 20000000')
data_weekday.plot(x='weekday', y='last_price', kind='hexbin', gridsize=20, figsize=(8, 6), sharex=False, grid=True)


# Больше всего объявлений создается в будние дни.

# In[70]:


data_month = data.query('last_price < 20000000')
data_month.plot(x='month', y='last_price', kind='hexbin', gridsize=20, figsize=(8, 6), sharex=False, grid=True)


# Больше всего объявлений создается вначале и вконце года.

# In[71]:


data_year = data.query('last_price < 20000000')
data_year.plot(x='year', y='last_price', kind='hexbin', gridsize=20, figsize=(8, 6), sharex=False, grid=True)


# Больше всего объявлений создалось в 2017 и в 2018 году.

# In[72]:


top_price_per_sqm = data.pivot_table(index='locality_name', values='price_per_sqm', aggfunc=['mean', 'count'])
top_price_per_sqm.columns = ['price_per_sqm', 'count']
top_price_per_sqm['price_per_sqm'] = top_price_per_sqm['price_per_sqm'].round()
top_price_per_sqm = top_price_per_sqm.sort_values(by='price_per_sqm', ascending=False).head(10)
print(top_price_per_sqm)
highest_price_locality = top_price_per_sqm.sort_values(by='price_per_sqm', ascending=False).index[0]
print('Населенный пункт с самой высокой стоимостью квадратного метра:', highest_price_locality)
lower_price_locality = top_price_per_sqm.sort_values(by='price_per_sqm', ascending=True).index[0]
print('Населенный пункт с самой низкой стоимостью квадратного метра:', lower_price_locality)


# Как и ожидалось - самая большая цена за квадратный метр в Санкт-Петербурге. Закрывает десятку по дороговизне квадратного метра жилья - Красное Село

# In[73]:


spb_data = data[data['locality_name'] == 'Санкт-Петербург']
spb_data_pivot = spb_data.pivot_table(index='city_centers_km', values='last_price', aggfunc='mean')
spb_data_pivot['last_price'] = spb_data_pivot['last_price'].round(2)
spb_data_pivot['mean_price_per_km'] = spb_data_pivot['last_price'] / spb_data_pivot.index
spb_data_pivot['mean_price_per_km'] = spb_data_pivot['mean_price_per_km'].round(2)
spb_data_pivot['mean_price_per_km'].loc[0] = spb_data_pivot['last_price'].loc[0]
spb_data_pivot['city_centers_km'] = spb_data_pivot.index
spb_data_pivot.plot(x='city_centers_km', y='mean_price_per_km')


# Зависимость цены на квартиру от расстояния до центра Санкт-Петербурга сильно коррелирует в том случае, если до центра 10 км. Стоимость квартиры, находящиеся от центра на расстояние 10 км и далее не так сильно зависит от этого фактора.

# ### Общий вывод

# Задача: Поиска интересных особенностей и зависимостей, которые существуют на рынке недвижимости в Санкт-Петербурге и соседних населенных пунктов.
# 
# Первым делом необходимо узнать какой тип данных у нас есть в наличие. Узнаем тип данных и приступаем к дальнейшим шагам. Большинство типов данных - float64 и int64. Это означает то что с ними можно выполнять математические операции для обработки данных. Необходимость изменения типа данных object рассмотрим позже. 
# 
# Вторым этапом исследование является обработка пропущенных значений. Основная часть пропущенных значений была заполнена медианными значениями соответствующих столбцов, но были и другие случаи. Пропущенные значения жилых зон заполнил разницей между общей площадью и площадью кухни. Пропуски в расстоянии до ближайшего парка заполнялись медианным значением расстояния от квартиры до парка для каждого населенного пункта отдельно. Там где не было данных об этом расстояние - пришлось оставить пропуски. 
# 
# Изменение типа данных коснулось только столбца: first_day_exposition (сколько дней было размещено объявление (от публикации до снятия)). Причиной такого решения стало то, что для дальнейшего анализа нам будет нужен формат datetime64 в этом столбце.
# 
# Проведенные исследование показывается зависимость цены на квартиру в Санкт-Петербурге и соседних населенных пунктах от совершенно разных параметрах. Стоимость может зависеть от площадей помещений в квартире (жилая зона, кухонная зона, общая и тд), от количества комнат, от этажа на котором расположена квартира и других факторов. Мы узнали, то что самые распространенные квартиры на рынке от 30 до 50 квадратных метров, с одной до трех комнат. Дома в которых продают квартиры чаще всего пятиэтажные, на втором месте - десятиэтажные. В Санкт-Петербурге и соседних населенных пунктах много парков = этот вывод можно сделать из того, что у большинства квартир расстояние до ближайшего парка занимает около 500 метров. Средний срок размещения объявление о продаже квартиры составляет от 200 до 600 дней. Чаще всего объявления размещают в буднии дни и в начале и конце года.
# 

# In[ ]:




