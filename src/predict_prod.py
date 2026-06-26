import mlflow.pyfunc
import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
db_path = os.path.join(BASE_DIR, "..", "mlflow.db")
sqlite_uri = f"sqlite:///{db_path}"

tracking_uri = os.getenv("MLFLOW_TRACKING_URI", sqlite_uri)
mlflow.set_tracking_uri(tracking_uri)

print(f"Подключаемся к реестру: {tracking_uri}")

model_name = "Iris_RF_Model"
alias = "champion"

model_uri = f"models:/{model_name}@{alias}"
print(f"Скачиваем модель по URI: {model_uri}")

model = mlflow.pyfunc.load_model(model_uri)

dummy_data = pd.DataFrame(
    [[5.1, 3.5, 1.4, 0.2]],
    columns=[
        "sepal length (cm)",
        "sepal width (cm)",
        "petal length (cm)",
        "petal width (cm)",
    ],
)

prediction = model.predict(dummy_data)
print(f"\n Предсказание боевой модели: Класс {prediction[0]}")
