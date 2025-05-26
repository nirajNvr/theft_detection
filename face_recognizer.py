import insightface
import numpy as np
import cv2
from sklearn.metrics.pairwise import cosine_similarity

class FaceRecognizer:
    def __init__(self, model_name='buffalo_l'):
        self.model = insightface.app.FaceAnalysis(name=model_name)
        self.model.prepare(ctx_id=0)  # Use ctx_id=1 for GPU if available
    
    def get_embedding(self, face_image):
        """Returns 512-D face embedding with error handling"""
        try:
            # Convert to RGB if needed (ArcFace expects RGB)
            if face_image.shape[2] == 1:  # Grayscale
                face_image = cv2.cvtColor(face_image, cv2.COLOR_GRAY2RGB)
            elif face_image.shape[2] == 4:  # RGBA
                face_image = cv2.cvtColor(face_image, cv2.COLOR_RGBA2RGB)
            
            faces = self.model.get(face_image)
            if len(faces) == 0:
                print("⚠️ No face found in the cropped image!")
                return None
            return faces[0].embedding
        except Exception as e:
            print(f"❌ Error in feature extraction: {str(e)}")
            return None

    def compare_faces(self, embedding1, embedding2, threshold=0.6):
        if embedding1 is None or embedding2 is None:
            return False
        similarity = cosine_similarity([embedding1], [embedding2])[0][0]
        return similarity > threshold