import traceback
from fastapi import FastAPI
import numpy as np
import pickle
import scipy.sparse
import os

if not os.path.exists("model.pkl") or not os.path.exists("interaction_matrix.npz"):
    raise FileNotFoundError("Файлы модели отсутствуют! Проверь загрузку model.pkl и interaction_matrix.npz")

# Инициализируем FastAPI
app = FastAPI()

# Загружаем обученную модель ALS (должен быть файл `model.pkl` в проекте)
try:
    print("🔄 Загрузка модели ALS...")
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    
    print("🔄 Загрузка interaction_matrix...")
    interaction_matrix = scipy.sparse.load_npz("interaction_matrix.npz")

    if model is None:
        raise ValueError("❌ Ошибка: model загрузилась как None!")
    if interaction_matrix is None:
        raise ValueError("❌ Ошибка: interaction_matrix загрузилась как None!")
        
    print("✅ Модель и матрица загружены успешно!")
except Exception as e:
    print(f"❌ Ошибка загрузки модели: {e}")
    model = None
    interaction_matrix = None



# Эндпоинт для получения рекомендаций
@app.get("/recommend/{visitorid}")
def get_recommendations(visitorid: int, N: int = 5):
    try:
        if model is None or interaction_matrix is None:
            return {"error": "Модель не загружена"}

        if visitorid not in range(interaction_matrix.shape[0]):
            return {"error": "Пользователь не найден"}

        recommended_items = model.recommend(visitorid, interaction_matrix[visitorid], N=N)

        recommendations = [{"categoryid": int(item[0]), "score": float(item[1])} for item in recommended_items]

        return {"visitorid": visitorid, "recommendations": recommendations}

    except Exception as e:
        error_message = traceback.format_exc()
        return {"error": "Внутренняя ошибка сервера", "details": error_message}
