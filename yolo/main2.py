from yolo.moduleyl1 import FaceDatabase
from yolo.moduleyl2 import WebcamRecognizer
from yolo.moduleyl3 import FaceComparator
import cv2

face_db = FaceDatabase()
database = face_db.create_database("faces")

webcam = WebcamRecognizer()
comparator = FaceComparator(database)

cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    box, face_vector = webcam.detect_face(frame)
    if face_vector is not None:
        name, score = comparator.compare(face_vector)
        label = f"{name} ({score:.2f})" if name != "Unknown" else "Unknown"
        if box:
            x1, y1, x2, y2 = box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    cv2.imshow("Face Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
cap.release()
cv2.destroyAllWindows()
