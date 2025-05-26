import cv2
import numpy as np
from face_detector import FaceDetector
from face_recognizer import FaceRecognizer
from database_manager import FaceDatabase

def process_image(image_path: str='sovit_test.jpg', output_path: str = 'sovit_output.jpg') -> None:
    """
    Process an image to detect and recognize faces.
    
    Args:
        image_path (str): Path to the input image
        output_path (str): Path where the processed image will be saved
    """
    detector = FaceDetector()
    recognizer = FaceRecognizer()
    database = FaceDatabase()

    image = cv2.imread(image_path)
    if image is None:
        print(f"🚨 Error: Could not read image at {image_path}")
        return

    # Resize large images to avoid memory issues
    height, width = image.shape[:2]
    if max(height, width) > 2000:
        image = cv2.resize(image, (width // 2, height // 2))

    # Get face embeddings directly from the recognizer
    faces = recognizer.model.get(image)
    print(f"🔍 Detected {len(faces)} faces")

    for i, face in enumerate(faces):
        # Get the bounding box
        bbox = face.bbox.astype(int)
        x1, y1, x2, y2 = bbox
        
        # Get the embedding
        embedding = face.embedding
        name = "Unknown"
        if embedding is not None:
            name = database.recognize_face(embedding)
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            cv2.putText(image, name, (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        print(f"👤 Face {i}: {name}")

    cv2.imwrite(output_path, image)
    print(f"💾 Results saved to {output_path}")

if __name__ == "__main__":
    process_image('second.jpg', 'output.jpg')