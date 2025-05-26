# import cv2
# import numpy as np
# from face_detector import FaceDetector
# from face_recognizer import FaceRecognizer
# from database_manager import FaceDatabase
# from zones import ZONES, ACCESS_LEVELS

# def draw_zones(frame):
#     """Draw zone boundaries on frame"""
#     # Soft zone (blue)
#     cv2.rectangle(frame, ZONES["soft"][0], ZONES["soft"][1], (255, 0, 0), 2)
#     # Hard zone (red)
#     cv2.rectangle(frame, ZONES["hard"][0], ZONES["hard"][1], (0, 0, 255), 2)
#     return frame

# def check_access(name, position):
#     """Check zone violations"""
#     if name == "Unknown":
#         return "🚨 INTRUDER ALERT"
    
#     access = ACCESS_LEVELS.get(name, "general")
#     x, y = position
    
#     # Check hard zone
#     if (ZONES["hard"][0][0] <= x <= ZONES["hard"][1][0] and 
#         ZONES["hard"][0][1] <= y <= ZONES["hard"][1][1]):
#         if access != "main":
#             return f"⚠️ {name} in RESTRICTED ZONE"
#     return None

# def main():
#     # Initialize components
#     detector = FaceDetector()
#     recognizer = FaceRecognizer()
#     database = FaceDatabase()
    
#     # Camera setup
#     cap = cv2.VideoCapture(0)
#     cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
#     cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break
            
#         # Process frame
#         frame = draw_zones(frame)
#         faces = recognizer.model.get(frame)
        
#         for face in faces:
#             # Get face info
#             bbox = face.bbox.astype(int)
#             x, y = bbox[:2]  # Top-left position
#             name = database.recognize_face(face.embedding)
            
#             # Check access
#             alert = check_access(name, (x, y))
#             if alert:
#                 cv2.putText(frame, alert, (50, 50), 
#                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
#             # Draw face box
#             color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
#             cv2.rectangle(frame, bbox[:2], bbox[2:], color, 2)
#             cv2.putText(frame, name, (x, y-10), 
#                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        
#         cv2.imshow("Warehouse Security", frame)
#         if cv2.waitKey(1) == ord('q'):
#             break
    
#     cap.release()
#     cv2.destroyAllWindows()

# if __name__ == "__main__":
#     main()
import cv2
import numpy as np
from face_detector import FaceDetector
from face_recognizer import FaceRecognizer
from database_manager import FaceDatabase
from zones import ZONES, ZONE_COLORS, ACCESS_LEVELS

def draw_zones(frame):
    """Draw polyzonal boundaries with labels"""
    # General zones (yellow)
    for zone in ZONES["general"]:
        cv2.rectangle(frame, zone[0], zone[1], ZONE_COLORS["general"], 2)
    
    # Restricted zones (red)
    for zone in ZONES["restricted"]:
        cv2.rectangle(frame, zone[0], zone[1], ZONE_COLORS["restricted"], 2)
    
    # Zone labels
    cv2.putText(frame, "GENERAL (All Employees)", (50, 50), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, ZONE_COLORS["general"], 2)
    cv2.putText(frame, "RESTRICTED (Admins Only)", (1100, 50), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, ZONE_COLORS["restricted"], 2)
    return frame

def check_access(name, bbox):
    """Enforce zone permissions"""
    x_center = (bbox[0] + bbox[2]) // 2
    y_center = (bbox[1] + bbox[3]) // 2
    
    for zone in ZONES["restricted"]:
        if (zone[0][0] <= x_center <= zone[1][0] and 
            zone[0][1] <= y_center <= zone[1][1]):
            if ACCESS_LEVELS.get(name, "general") != "admin":
                return f"VIOLATION: {name} in restricted zone"
    return None

def main():
    detector = FaceDetector()
    recognizer = FaceRecognizer()
    database = FaceDatabase()
    
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame = draw_zones(frame)
        faces = recognizer.model.get(frame)
        
        for face in faces:
            bbox = face.bbox.astype(int)
            name = database.recognize_face(face.embedding)
            
            # Zone access check
            alert = check_access(name, bbox)
            if alert:
                cv2.putText(frame, alert, (50, 100), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            
            # Draw face box
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            cv2.rectangle(frame, bbox[:2], bbox[2:], color, 2)
            cv2.putText(frame, name, (bbox[0], bbox[1]-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        cv2.imshow("Theft Detection - PolyZonal Monitoring", frame)
        if cv2.waitKey(1) == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()