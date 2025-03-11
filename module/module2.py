import cv2
import torch
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1

class WebcamRecognizer:
    def __init__(self):
        self.mtcnn = MTCNN(image_size=160, keep_all=False)
        self.facenet = InceptionResnetV1(pretrained="vggface2").eval()
    def recognize_face_from_webcam(self):
        cap = cv2.VideoCapture(0)
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Lỗi khi lấy hình ảnh từ webcam")
                break
            face, box = self.detect_face(frame)
            if face is not None:
                feature = self.extract_features(face)
                if feature is not None:
                    yield feature, frame, box
            self.show_frame(frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        cap.release()
        cv2.destroyAllWindows()

    def detect_face(self, img):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        box, _ = self.mtcnn.detect(img_pil)
        face = self.mtcnn(img_pil)
        if face is not None and box is not None:
            return face, list(map(int, box[0]))
        return None, None

    def extract_features(self, face):
        face = face.unsqueeze(0)
        with torch.no_grad():
            return self.facenet(face).squeeze(0)

    def draw_box(self, frame, box, name=None):
        if box:
            x1, y1, x2, y2 = box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            if name:
                cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    def show_frame(self, frame):
        cv2.imshow("Webcam Face Recognition", frame)
