from fastapi import FastAPI
import numpy as np
import uvicorn

# Инициализируем FastAPI
app = FastAPI()

# Загружаем обученную модель ALS (уже обучена в твоём Colab)
model = model  # Используем обученную модель
interaction_matrix = interaction_sparse  # Матрица взаимодействий пользователей

# Обработчик запроса на рекомендации
@app.get("/recommend/{visitorid}")
def get_recommendations(visitorid: int, N: int = 5):
    try:
        if visitorid not in range(interaction_matrix.shape[0]):
            return {"error": "Пользователь не найден"}

        # Получаем рекомендации
        recommended_items = model.recommend(visitorid, interaction_matrix[visitorid], N=N)

        # Формируем список рекомендаций
        recommendations = [{"categoryid": int(item[0]), "score": float(item[1])} for item in recommended_items]

        return {"visitorid": visitorid, "recommendations": recommendations}

    except Exception as e:
        return {"error": str(e)}

# Запускаем сервер (для локального запуска)
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
