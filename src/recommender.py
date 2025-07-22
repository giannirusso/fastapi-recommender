import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity

class Recommender:
    def __init__(self, user_item_matrix):
        self.user_item = user_item_matrix
        self.similarity = cosine_similarity(user_item_matrix.T)

    def recommend(self, user_id, top_k=5):
        user_vector = self.user_item[user_id]
        scores = user_vector @ self.similarity
        scores[user_vector.nonzero()] = 0
        recommended_items = np.argsort(-scores)[:top_k]
        return recommended_items.tolist()
