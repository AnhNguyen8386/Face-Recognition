import cv2
import torch
import numpy as np
from facenet_pytorch import InceptionResnetV1
from torchvision import transforms

class WebcamRecognizer:
    def __init__(self, model_path="face_detection/face_detection_yolov5s.pt"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path).to(self.device)
        self.facenet = InceptionResnetV1(pretrained="vggface2").eval().to(self.device)
        self.transform = transforms.Compose([transforms.ToTensor()])

    def detect_face(self, frame):
        results = self.model(frame)
        for *xyxy, conf, cls in results.xyxy[0]:
            x1, y1, x2, y2 = map(int, xyxy)
            face = frame[y1:y2, x1:x2]
            face = cv2.resize(face, (160, 160))
            face = self.transform(face).unsqueeze(0).to(self.device)
            with torch.no_grad():
                feature = self.facenet(face).cpu().numpy().flatten()
            return (x1, y1, x2, y2), feature
        return None, None

    def recognize_face_from_webcam(self):
        cap = cv2.VideoCapture(0)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            box, face_vector = self.detect_face(frame)
            if box:
                x1, y1, x2, y2 = box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.imshow("Webcam", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        cap.release()
        cv2.destroyAllWindows()
