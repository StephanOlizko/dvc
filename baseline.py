import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib

# Загрузка данных
df = pd.read_csv("data/iris.csv")
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# Разделение данных
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

# Тренировка модели
model = LogisticRegression(max_iter=3, random_state=42)
model.fit(X_train, y_train)

# Оценка модели
y_pred = model.predict(X_test)

acc = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average="micro")
recall = recall_score(y_test, y_pred, average="micro")
f1 = f1_score(y_test, y_pred, average="micro")

print(f"Accuracy: {acc}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1: {f1}")

# Сохранение модели
joblib.dump(model, "model/model.pkl")

import json

# В конце функции run_experiment добавьте сохранение в JSON
metrics = {
    "Accuracy": acc,
    "F1": f1,
    "Precision": precision,
    "Recall": recall,
}
with open("metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)
