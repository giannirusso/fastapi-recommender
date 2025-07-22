from fastapi import FastAPI, Query
import joblib
import numpy as np

app = FastAPI()
model = joblib.load("app/model.pkl")  # user-item matrix and item similarity

@app.get("/recommend")
def recommend(user_id: int = Query(..., description="User ID")):
    recommendations = model.recommend(user_id)
    return {
        "user_id": user_id,
        "recommended_items": recommendations
    }
