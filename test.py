import pickle

file_path = "face_db_facedetection.pkl"
with open(file_path, "rb") as f:
    data = pickle.load(f)
print("Dữ liệu trong file:", list(data.keys()))
print("Đã lưu dữ liệu thành công! Số lượng ID:", len(data))
