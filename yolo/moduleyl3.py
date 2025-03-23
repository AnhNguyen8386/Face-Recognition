import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class FaceComparator:
    def __init__(self, face_db):
        self.face_db = face_db
        self.names = list(face_db.keys())
        self.features = np.array(list(face_db.values()))

    def compare(self, face_vector, threshold=0.6):
        if len(self.features) == 0:
            return "Unknown", 0.0
        similarity_scores = cosine_similarity([face_vector], self.features)[0]
        best_match_idx = np.argmax(similarity_scores)
        best_score = similarity_scores[best_match_idx]
        return (self.names[best_match_idx], best_score) if best_score > threshold else ("Unknown", best_score)
