import torch
from torch.nn.functional import cosine_similarity

class FaceComparator:
    def __init__(self, face_db, threshold=0.4):
        self.threshold = threshold
        if face_db:
            self.names = list(face_db.keys())
            self.db_tensor = torch.stack([v.float() for v in face_db.values()])
        else:
            self.names = []
            self.db_tensor = torch.empty((0, 512))
            print("Database trống")

    def compare(self, webcam_feature):
        if self.db_tensor.shape[0] == 0:
            return None, 1
        similarities = cosine_similarity(webcam_feature.unsqueeze(0), self.db_tensor)
        distances = 1 - similarities.squeeze(0)
        best_distance, best_idx = distances.min(dim=0)
        if best_distance.item() < self.threshold:
            return self.names[best_idx.item()], best_distance.item()
        return None, best_distance.item()
