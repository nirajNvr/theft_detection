import numpy as np
import os
from sklearn.metrics.pairwise import cosine_similarity

class FaceDatabase:
    def __init__(self, db_folder='face_db'):
        self.db_folder = db_folder
        os.makedirs(db_folder, exist_ok=True)
        self.known_faces = self._load_known_faces()

    def _load_known_faces(self):
        """Loads all .npy files from face_db folder"""
        db = {}
        for file in os.listdir(self.db_folder):
            if file.endswith('.npy'):
                name = os.path.splitext(file)[0]
                db[name] = np.load(os.path.join(self.db_folder, file))
        return db

    def add_face(self, name, embedding):
        """Adds new face to database"""
        np.save(os.path.join(self.db_folder, f"{name}.npy"), embedding)
        self.known_faces[name] = embedding

    def recognize_face(self, embedding, threshold=0.6):
        """Returns recognized name or 'Unknown'"""
        for name, known_embedding in self.known_faces.items():
            similarity = cosine_similarity([embedding], [known_embedding])[0][0]
            if similarity > threshold:
                return name
        return "Unknown"