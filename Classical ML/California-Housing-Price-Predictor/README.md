# Прогнозирование стоимости жилья в Калифорнии

Модель линейной регрессии для предсказания медианной стоимости домов в жилых массивах Калифорнии на основе данных 1990 года.

## 🎯 Цель проекта

- Разработать модель для точного предсказания стоимости жилья
- Оценить качество модели с помощью метрик RMSE, MAE и R²
- Интерпретировать влияние признаков на стоимость недвижимости

## 💡 Использованные технологии

- Python 3.x
- pandas, numpy
- pyspark
- OneHotEncoder, StringIndexer, StandardScaler, VectorAssembler
- Jupyter Notebook

## 🧪 Как запустить проект

```bash
git clone https://github.com/kagor4/California-Housing-Price-Predictor.git
cd California-Housing-Price-Predictor
pip install -r requirements.txt
```

Затем откройте и запустите файл `California Housing Price Predictor.py` или ноутбук с аналогичным кодом в Jupyter. Убедитесь, что датасет `housing.csv` доступен в папке `/datasets/`.

## 📊 Описание данных

Датасет содержит информацию о жилье в Калифорнии (1990 год):
- **Признаки**:
  - Географические: `longitude`, `latitude`, `ocean_proximity`
  - Демографические: `housing_median_age`, `population`, `households`
  - Характеристики жилья: `total_rooms`, `total_bedrooms`
  - Экономические: `median_income`
- **Целевая переменная**: `median_house_value` (медианная стоимость дома)

Предобработка включала:
- Заполнение пропусков в `total_bedrooms` медианным значением
- Кодирование категориального признака `ocean_proximity` (StringIndexer + OneHotEncoder)
- Масштабирование числовых признаков (StandardScaler)

## 🔍 Краткие результаты

- Лучшая модель: Линейная регрессия с использованием всех признаков (`all_features`)
- Метрики (тестовая выборка):
  - RMSE: 68932.66
  - MAE: 49676.45
  - R²: 0.631
- Сравнение:
  - Модель с числовыми признаками (`numerical_features_scaled`):
    - RMSE: 69653.32
    - MAE: 50550.71
    - R²: 0.623
- Основные выводы:
  - Полный набор признаков (`all_features`) даёт лучшую точность
  - Категориальный признак `ocean_proximity` улучшает предсказания
  - Модель объясняет ~63% вариации цен
- Рекомендации:
  - Добавить нелинейные модели (Random Forest, Gradient Boosting)
  - Исследовать новые признаки (например, инфраструктура района)
  - Увеличить объём данных для повышения качества

## 📁 Структура проекта

```
📦 California-Housing-Price-Predictor/
├── California Housing Price Predictor.py  # анализ и обучение модели
├── requirements.txt                      # зависимости
└── README.md                             # описание проекта
```

## ✅ TODO

- Исследовать нелинейные модели (Random Forest, XGBoost)
- Добавить признаки (инфраструктура, транспортная доступность)
- Провести визуализацию предсказаний
- Разработать Streamlit-приложение для демонстрации

## © Автор

Автор: [kagor4](https://github.com/kagor4)

```
