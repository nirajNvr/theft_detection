import cv2
import numpy as np
from face_detector import FaceDetector
from face_recognizer import FaceRecognizer
from database_manager import FaceDatabase
from threading import Lock

# Global configuration
ACCESS_LEVELS = {
    "niraj": "general",  # Changed from admin to general
    "sovit": "general",
    "Unknown": "general"
}

# Zone management
current_zones = {
    "general": [np.array([[0, 0], [400, 0], [400, 600], [0, 600]], dtype=np.int32)],  # Left half
    "restricted": [np.array([[400, 0], [800, 0], [800, 600], [400, 600]], dtype=np.int32)],  # Right half
    "veil": []  # New zone type for no-detection areas
}
zone_lock = Lock()
edit_mode = False
current_zone_type = "general"
dragging = False
selected_zone_idx = -1
selected_corner_idx = -1

def draw_zones(frame):
    """Visualize zones with transparency"""
    overlay = frame.copy()
    with zone_lock:
        for zone_type, polygons in current_zones.items():
            if zone_type == "general":
                color = (0, 255, 255)  # Yellow
                for poly in polygons:
                    cv2.fillPoly(overlay, [poly], color)
                    cv2.polylines(overlay, [poly], True, (255,255,255), 2)
            elif zone_type == "restricted":
                color = (0, 0, 255)  # Red
                for poly in polygons:
                    cv2.fillPoly(overlay, [poly], color)
                    cv2.polylines(overlay, [poly], True, (255,255,255), 2)
            else:  # veil
                # Draw veil zones with higher opacity
                for poly in polygons:
                    cv2.fillPoly(overlay, [poly], (0, 0, 0))
                    cv2.polylines(overlay, [poly], True, (255,255,255), 2)
    
    # Blend with original - different blend for veil zones
    # First blend normal zones
    cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
    
    # Then add veil zones with higher opacity
    veil_overlay = frame.copy()
    with zone_lock:
        for poly in current_zones["veil"]:
            cv2.fillPoly(veil_overlay, [poly], (0, 0, 0))
            cv2.polylines(veil_overlay, [poly], True, (255,255,255), 2)
    cv2.addWeighted(veil_overlay, 0.7, frame, 0.3, 0, frame)
    
    # Draw zone labels
    cv2.putText(frame, "GENERAL (Drag to edit)", (20, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(frame, "RESTRICTED (Press 'R' to edit)", (420, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(frame, "VEIL (Press 'V' to edit)", (20, 60), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    
    if edit_mode:
        cv2.putText(frame, "EDIT MODE: Drag corners | DEL to remove | ESC to save", 
                   (50, 570), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    return frame

def get_face_center(face):
    """Calculate center point of face bounding box"""
    bbox = face.bbox.astype(int)
    x1, y1, x2, y2 = bbox
    return (x1 + x2) // 2, (y1 + y2) // 2

def trigger_alert(name):
    """Display alert for unauthorized access"""
    return f"VIOLATION: {name} in restricted zone"

def mouse_callback(event, x, y, flags, param):
    """Handle mouse events for zone editing"""
    global dragging, selected_zone_idx, selected_corner_idx
    
    if not edit_mode:
        return
    
    with zone_lock:
        zones = current_zones[current_zone_type]
    
    # Find nearest zone/corner
    if event == cv2.EVENT_LBUTTONDOWN:
        min_dist = 20
        for zi, zone in enumerate(zones):
            for ci, corner in enumerate(zone):
                dist = np.sqrt((x - corner[0])**2 + (y - corner[1])**2)
                if dist < min_dist:
                    selected_zone_idx = zi
                    selected_corner_idx = ci
                    dragging = True
                    min_dist = dist
    
    # Drag corner
    elif event == cv2.EVENT_MOUSEMOVE and dragging:
        zones[selected_zone_idx][selected_corner_idx] = [x, y]
    
    # Release
    elif event == cv2.EVENT_LBUTTONUP:
        dragging = False

def draw_help_overlay(frame):
    """Draw help overlay with all available controls"""
    # Create semi-transparent overlay
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (790, 590), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    
    # Title
    cv2.putText(frame, "WAREHOUSE SECURITY SYSTEM - CONTROLS", (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    # Basic Controls
    cv2.putText(frame, "BASIC CONTROLS:", (50, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    controls = [
        ("Q", "Quit application"),
        ("H", "Toggle this help overlay"),
        ("ESC", "Exit edit mode and save changes")
    ]
    
    y_pos = 140
    for key, desc in controls:
        cv2.putText(frame, f"{key}: {desc}", (70, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y_pos += 30
    
    # Zone Editing Controls
    cv2.putText(frame, "ZONE EDITING CONTROLS:", (50, 240),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    zone_controls = [
        ("E", "Toggle edit mode"),
        ("G", "Switch to general zone editing"),
        ("R", "Switch to restricted zone editing"),
        ("V", "Switch to veil zone editing"),
        ("N", "Add new zone"),
        ("DELETE", "Remove last zone of current type"),
        ("Mouse", "Drag corners to adjust zone shapes")
    ]
    
    y_pos = 280
    for key, desc in zone_controls:
        cv2.putText(frame, f"{key}: {desc}", (70, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y_pos += 30
    
    # Zone Information
    cv2.putText(frame, "ZONE INFORMATION:", (50, 460),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    zone_info = [
        ("General Zone", "Yellow - Allowed for everyone"),
        ("Restricted Zone", "Red - Admin access only"),
        ("Veil Zone", "Black - No face detection"),
        ("Face Box", "Green: Allowed, Red: Restricted, Purple: Admin")
    ]
    
    y_pos = 500
    for key, desc in zone_info:
        cv2.putText(frame, f"{key}: {desc}", (70, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y_pos += 30
    
    return frame

def get_access_status(name, is_in_restricted):
    """Determine access status for a person"""
    # First check which zone the person is in
    if is_in_restricted:
        # In restricted zone - only admin allowed
        if ACCESS_LEVELS.get(name) == "admin":
            return "ALLOWED", (255, 0, 255)  # Purple for admin in restricted
        else:
            return "RESTRICTED", (0, 0, 255)  # Red for unauthorized in restricted
    else:
        # In general zone - everyone allowed
        return "ALLOWED", (0, 255, 0)  # Green for authorized in general

def is_in_veil_zone(x, y):
    """Check if a point is in any veil zone"""
    with zone_lock:
        for zone in current_zones["veil"]:
            point = np.array([x, y], dtype=np.float32)
            if cv2.pointPolygonTest(zone, (point[0], point[1]), False) >= 0:
                return True
    return False

def process_webcam(ip_address=None):
    """Main webcam processing function"""
    global edit_mode, current_zone_type
    
    # Initialize components
    detector = FaceDetector()
    recognizer = FaceRecognizer()
    database = FaceDatabase()
    
    # Initialize webcam with provided IP or default
    if ip_address:
        cap = cv2.VideoCapture(ip_address)
    else:
        cap = cv2.VideoCapture(0)
        
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    cv2.namedWindow("Warehouse Security")
    cv2.setMouseCallback("Warehouse Security", mouse_callback)
    
    show_help = True  # Start with help visible
    edit_mode = False  # Start in non-edit mode
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame from webcam")
            break
        
        # Resize frame
        frame = cv2.resize(frame, (800, 600))
        vis_frame = frame.copy()
        
        # Draw zones
        vis_frame = draw_zones(vis_frame)
        
        # Process faces (only when not editing for better performance)
        if not edit_mode:
            try:
                faces = recognizer.model.get(frame)
                for face in faces:
                    x_center, y_center = get_face_center(face)
                    
                    # Skip detection if face is in veil zone
                    if is_in_veil_zone(x_center, y_center):
                        continue
                    
                    name = database.recognize_face(face.embedding)
                    
                    # Draw face box and name
                    bbox = face.bbox.astype(int)
                    x1, y1, x2, y2 = bbox
                    
                    # First check which zone the person is in
                    is_in_restricted = False
                    with zone_lock:
                        for zone in current_zones["restricted"]:
                            point = np.array([x_center, y_center], dtype=np.float32)
                            if cv2.pointPolygonTest(zone, (point[0], point[1]), False) >= 0:
                                is_in_restricted = True
                                break
                    
                    # Get access status based on zone first, then check permissions
                    status, color = get_access_status(name, is_in_restricted)
                    
                    # Draw face box
                    cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 2)
                    
                    # Draw name and status
                    name_text = f"{name}"
                    status_text = f"{status}"
                    
                    # Calculate text positions
                    name_y = y1 - 10
                    status_y = y1 - 35
                    
                    # Draw name
                    cv2.putText(vis_frame, name_text, (x1, name_y), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                    
                    # Draw status with background for better visibility
                    (status_w, status_h), _ = cv2.getTextSize(status_text, 
                                                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                    cv2.rectangle(vis_frame, 
                                (x1, status_y - status_h - 5),
                                (x1 + status_w, status_y + 5),
                                (0, 0, 0), -1)
                    cv2.putText(vis_frame, status_text, (x1, status_y), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                    
                    # Log violation if needed
                    if status == "RESTRICTED" and is_in_restricted:
                        print(f"Alert: {name} in restricted zone")
            except Exception as e:
                print(f"Error processing faces: {str(e)}")
                import traceback
                traceback.print_exc()
        
        # Show help overlay if enabled
        if show_help:
            vis_frame = draw_help_overlay(vis_frame)
        
        # Display frame
        cv2.imshow("Warehouse Security", vis_frame)
        
        # Handle key controls
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        elif key == ord('h'):
            show_help = not show_help
        elif key == ord('e'):
            edit_mode = not edit_mode
            print(f"Edit mode {'enabled' if edit_mode else 'disabled'}")
        elif key == 27:  # ESC key
            if edit_mode:
                edit_mode = False
                print("Edit mode disabled - changes saved")
        elif key == ord('g'):
            current_zone_type = "general"
            print("Switched to general zone editing")
        elif key == ord('r'):
            current_zone_type = "restricted"
            print("Switched to restricted zone editing")
        elif key == ord('v'):  # New key for veil zones
            current_zone_type = "veil"
            print("Switched to veil zone editing")
        elif key in [8, 46]:  # Both Backspace (8) and Delete (46) keys
            if edit_mode:  # Only allow deletion in edit mode
                with zone_lock:
                    if current_zones[current_zone_type]:
                        current_zones[current_zone_type].pop(0)  # Remove from front (top layer)
                        print(f"Removed top {current_zone_type} zone")
        elif key == ord('n'):  # New zone
            if edit_mode:  # Only allow new zones in edit mode
                with zone_lock:
                    # Create a new polygonal zone
                    center_x = 400
                    center_y = 300
                    radius = 100
                    num_points = 6  # Hexagon
                    points = []
                    for i in range(num_points):
                        angle = 2 * np.pi * i / num_points
                        x = center_x + radius * np.cos(angle)
                        y = center_y + radius * np.sin(angle)
                        points.append([x, y])
                    
                    new_zone = np.array(points, dtype=np.int32)
                    
                    # Insert at the beginning of the list to make it appear on top
                    current_zones[current_zone_type].insert(0, new_zone)
                    print(f"Added new {current_zone_type} zone on top")
        elif key == ord('c'):  # Clear all zones of current type
            if edit_mode:  # Only allow clearing in edit mode
                with zone_lock:
                    current_zones[current_zone_type] = []
                    print(f"Cleared all {current_zone_type} zones")
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    process_webcam()