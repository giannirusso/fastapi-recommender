# 🛍️ E-commerce Recommender System

A simple item-based collaborative filtering recommendation system with FastAPI and Docker.

## 📌 Project Goal
Recommend relevant products to users based on their historical interactions.

## 🧠 Model
- Technique: Item-based Collaborative Filtering
- Similarity: Cosine similarity
- Stack: pandas, scipy, FastAPI, Docker

## 🧪 API Example

`GET /recommend?user_id=123`

Response:
```json
{
  "user_id": 123,
  "recommended_items": [104, 218, 150]
}
```

## 🚀 Run Locally (Docker)

```bash
docker build -t recommender-api .
docker run -p 8000:8000 recommender-api
```

## 📂 Project Structure

- `notebooks/`: EDA and model prototyping
- `src/`: Python scripts for recommendation logic
- `app/`: FastAPI app
