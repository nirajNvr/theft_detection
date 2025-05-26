import cv2
import numpy as np
from face_detector import FaceDetector
from face_recognizer import FaceRecognizer
from database_manager import FaceDatabase

def register_face(image_path, name):
    # Initialize components
    detector = FaceDetector()
    recognizer = FaceRecognizer()
    database = FaceDatabase()

    # Read image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image {image_path}")
        return False

    print(f"Image loaded successfully. Shape: {image.shape}")

    # Detect faces
    boxes = detector.detect_faces(image)
    print(f"Detected {len(boxes)} faces")
    
    if len(boxes) != 1:  # Only register if exactly 1 face is detected
        print(f"Error: Found {len(boxes)} faces. Please use an image with exactly one face.")
        return False

    # Do not crop, use the whole image
    face_img = image
    print(f"Using the whole image for recognition. Shape: {face_img.shape}")
    
    embedding = recognizer.get_embedding(face_img)
    
    if embedding is not None:
        database.add_face(name, embedding)
        print(f"Successfully registered {name}!")
        return True
    else:
        print("Error: Could not extract face features")
        return False

if __name__ == "__main__":
    # Register Elon Musk's face
    register_face("srujan.jpg", "srujan")