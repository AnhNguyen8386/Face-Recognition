import torch
import cv2
import numpy as np
MODEL_PATH = "face_detection/face_detection_yolov5s.pt"
IMAGE_PATH = "faces/Khanh.jpg"
device = torch.device("cpu")
model = torch.hub.load("ultralytics/yolov5", "custom", path=MODEL_PATH, force_reload=True)
model.to(device)
model.eval()

def detect_faces(image_path):
    img = cv2.imread(image_path)
    results = model(img)
    boxes = results.pandas().xyxy[0]  # Lấy kết quả
    return boxes

def draw_boxes(image_path, boxes):
    img = cv2.imread(image_path)
    for _, row in boxes.iterrows():
        x1, y1, x2, y2, conf = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax']), row['confidence']
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, f"{conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow("Face Detection", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



boxes = detect_faces(IMAGE_PATH)
draw_boxes(IMAGE_PATH, boxes)
