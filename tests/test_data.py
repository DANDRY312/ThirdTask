import pandas as pd
import pandera as pa
from pandera import Column, Check
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "..", "data", "iris.csv")


def test_data_schema():
    # Проверяем, существует ли файл (скачался ли он через DVC)
    assert os.path.exists(DATA_PATH), "Файл данных не найден!"

    df = pd.read_csv(DATA_PATH)

    # Описываем строгий контракт (схему) данных
    schema = pa.DataFrameSchema(
        {
            "sepal length (cm)": Column(float, Check.ge(0)),  # Значения >= 0
            "sepal width (cm)": Column(float, Check.ge(0)),
            "petal length (cm)": Column(float, Check.ge(0)),
            "petal width (cm)": Column(float, Check.ge(0)),
            "target": Column(int, Check.isin([0, 1, 2])),  # Только 3 класса!
        }
    )

    # Валидация данных (если что-то не так, тест упадет)
    schema.validate(df)
