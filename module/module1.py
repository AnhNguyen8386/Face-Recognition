import cv2
import os
import torch
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1
import torchvision.transforms as transforms
class FaceDatabase:
    def __init__(self):
        self.mtcnn = MTCNN(image_size=160, margin=20)
        self.facenet = InceptionResnetV1(pretrained="vggface2").eval()
        self.face_db = {}

    def create_database(self, folder="faces"):
        self.face_db = {}
        for filename in os.listdir(folder):
            name, ext = os.path.splitext(filename)
            path = os.path.join(folder, filename)
            img = cv2.imread(path)
            if img is None:
                continue
            face = self.detect_face(img)
            if face is None:
                continue
            feature = self.extract_features(face)
            if feature is not None:
                self.face_db[name] = feature
        return self.face_db
    def detect_face(self, img):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        face = self.mtcnn(img_pil)
        return face


    def extract_features(self, face):
        if face is None:
            return None
        face = face.unsqueeze(0)
        with torch.no_grad():
            embedding = self.facenet(face)
        return embedding.squeeze(0)
    def get_features(self):
        return self.face_db
