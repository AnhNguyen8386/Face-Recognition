import os
import torch
import cv2
import numpy as np
from facenet_pytorch import InceptionResnetV1
from torchvision import transforms

class FaceDatabase:
    def __init__(self, model_path="face_detection/face_detection_yolov5s.pt"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path).to(self.device)
        self.facenet = InceptionResnetV1(pretrained="vggface2").eval().to(self.device)
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.face_db = {}

    def extract_faces(self, img):
        results = self.model(img)
        faces = []
        for *xyxy, conf, cls in results.xyxy[0]:
            x1, y1, x2, y2 = map(int, xyxy)
            face = img[y1:y2, x1:x2]
            faces.append(face)
        return faces

    def extract_features(self, face):
        face = cv2.resize(face, (160, 160))
        face = self.transform(face).unsqueeze(0).to(self.device)
        with torch.no_grad():
            feature = self.facenet(face).cpu().numpy().flatten()
        return feature

    def create_database(self, folder="faces"):
        self.face_db = {}
        for filename in os.listdir(folder):
            name, _ = os.path.splitext(filename)
            img_path = os.path.join(folder, filename)
            img = cv2.imread(img_path)
            faces = self.extract_faces(img)
            if not faces:
                continue
            feature = self.extract_features(faces[0])
            self.face_db[name] = feature
        return self.face_db

    def get_features(self):
        return self.face_db
