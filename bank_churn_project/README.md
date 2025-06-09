# bank_churn_project

Проект по прогнозированию ухода клиентов из банка на основе исторических данных.  
Цель — построить модель с максимальной F1-мерой (целевой минимум: **0.59**) и протестировать различные подходы к устранению дисбаланса классов.

---
 
## 🎯 Цель проекта

- Предсказать уход клиента в ближайшее время
- Использовать методы борьбы с дисбалансом: upsample, downsample, class_weight
- Сравнить 3 модели: DecisionTree, RandomForest, LogisticRegression

---

## 🧰 Используемые технологии

- Python 3.x
- pandas, numpy, matplotlib
- scikit-learn
- tqdm

---

## 📁 Структура проекта

```
bank_churn_project/
├── data/                     # CSV-файлы и базы (если есть)
├── review_homework_id_1032049.py   # основной анализ и код
├── requirements.txt          # зависимости
└── README.md                 # описание проекта
```

---

## 📈 Модели и метрики

**Лучшие параметры:**

- RandomForestClassifier  
  `n_estimators=100`, `max_depth=15`, `min_samples_leaf=8`

**Тестовая выборка:**

- Accuracy: `0.82`
- Precision: `0.57`
- Recall: `0.66`
- F1 Score: `0.61`
- ROC AUC: `0.85`

---

## ⚖️ Работа с дисбалансом классов

Методы, которые были протестированы:
- **Upsample** (увеличение миноритарного класса)
- **Downsample** (уменьшение мажоритарного класса)
- **Class weights** (взвешивание классов в моделях)

---

## 🧪 Как запустить

```bash
git clone https://github.com/your_username/bank_churn_project.git
cd bank_churn_project
pip install -r requirements.txt
```

---

## ✅ Вывод

Модель RandomForestClassifier показала лучший баланс между Precision и Recall.  
F1-мера превышает целевое значение (0.59), проект достиг поставленной цели.

---

## © Автор

Автор: [Ваше имя или GitHub](https://github.com/your_username)
