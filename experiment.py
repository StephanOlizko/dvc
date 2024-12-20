import argparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import pandas as pd

def run_experiment(n_estimators, max_depth, output_path):
    # Загрузка данных
    data = pd.read_csv("data/iris.csv")
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]

    # Разделение на тренировочные и тестовые данные
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Обучение модели
    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    model.fit(X_train, y_train)

    # Предсказания
    y_pred = model.predict(X_test)

    # Вычисление метрик
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="macro")
    precision = precision_score(y_test, y_pred, average="macro")
    recall = recall_score(y_test, y_pred, average="macro")

    import json

    # В конце функции run_experiment добавьте сохранение в JSON
    metrics = {
        "Accuracy": accuracy,
        "F1": f1,
        "Precision": precision,
        "Recall": recall,
    }
    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_estimators", type=int, required=True, help="Number of trees in the forest")
    parser.add_argument("--max_depth", type=int, required=True, help="Maximum depth of the trees")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save metrics")
    args = parser.parse_args()

    run_experiment(args.n_estimators, args.max_depth, args.output_path)


