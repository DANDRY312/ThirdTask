import os
import yaml
import joblib
import json
import pandas as pd
import mlflow
import mlflow.sklearn
import dagshub
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Инициализация DagsHub (автоматически настроит MLflow на облако)
dagshub.init(
    repo_owner="DANDRY312",
    repo_name="ThirdTask",
    mlflow=True,
    token=os.getenv("DAGSHUB_TOKEN"),
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def train_model():
    # Настройка эксперимента
    mlflow.set_experiment("Iris_Classification")

    # Чтение параметров
    params_path = os.path.join(BASE_DIR, "..", "params.yaml")
    with open(params_path, "r") as f:
        params = yaml.safe_load(f)["train"]

    # Подготовка данных
    data_path = os.path.join(BASE_DIR, "..", "data", "iris.csv")
    df = pd.read_csv(data_path)
    X = df.drop("target", axis=1)
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Начало эксперимента
    with mlflow.start_run():
        # Автоматическое логирование параметров и модели
        mlflow.sklearn.autolog()

        model = RandomForestClassifier(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            random_state=42,
        )

        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        acc = accuracy_score(y_test, predictions)

        # Сохранение метрик для DVC
        metrics_path = os.path.join(BASE_DIR, "..", "metrics.json")
        with open(metrics_path, "w") as f:
            json.dump({"accuracy": acc}, f)

        # Сохранение модели для DVC
        model_dir = os.path.join(BASE_DIR, "..", "models")
        os.makedirs(model_dir, exist_ok=True)
        joblib.dump(model, os.path.join(model_dir, "model.pkl"))

        print(f"Модель обучена. Accuracy: {acc}")


if __name__ == "__main__":
    train_model()
