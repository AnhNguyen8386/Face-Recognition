from module.module1 import FaceDatabase
from module.module2 import WebcamRecognizer
from module.module3 import FaceComparator
db = FaceDatabase()
face_db = db.create_database("faces")
webcam = WebcamRecognizer()
comparator = FaceComparator(face_db)
last_recognized_name = None
for feature, frame, box in webcam.recognize_face_from_webcam():
    name, distance = comparator.compare(feature)
    if name:
        last_recognized_name = name
        print(f"Nhận diện: {name} (Khoảng cách: {distance:.4f})", end="\r")
    else:
        print(f"Unknow!", end="\r")
    webcam.draw_box(frame, box, name)
    webcam.show_frame(frame)
if last_recognized_name:
    print(f"Kết quả cuối cùng: Nhận diện {last_recognized_name}")
else:
    print("Unknow")
