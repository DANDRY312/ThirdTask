import joblib
import os
import numpy as np  # Добавляем для работы с массивами
from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from pydantic import BaseModel

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "model.pkl")

# Добавляем базовую проверку существования файла
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Модель не найдена по пути: {MODEL_PATH}")

model = joblib.load(MODEL_PATH)

app = FastAPI(title="Iris ML API")


class IrisFeatures(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float


@app.get("/")
def read_root():
    return RedirectResponse(url="/docs")


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/predict")
def predict(features: IrisFeatures):
    try:
        # Преобразуем в массив numpy — это стандарт для sklearn
        data = np.array(
            [
                [
                    features.sepal_length,
                    features.sepal_width,
                    features.petal_length,
                    features.petal_width,
                ]
            ]
        )

        prediction = model.predict(data)
        return {"predicted_class": int(prediction[0])}

    except Exception as e:
        # В случае ошибки внутри модели вернем 500
        raise HTTPException(status_code=500, detail=str(e))
