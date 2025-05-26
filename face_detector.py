from ultralytics import YOLO
import cv2

class FaceDetector:
    def __init__(self, model_path='model/yolov8s.pt'):
        self.model = YOLO(model_path)  # Load YOLOv8 face detector

    def detect_faces(self, image):
        """Returns list of bounding boxes in format [x1, y1, x2, y2]"""
        results = self.model(image)
        return results[0].boxes.xyxy.cpu().numpy()  # Convert to numpy array