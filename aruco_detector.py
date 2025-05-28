# aruco_detector.py
import cv2
import cv2.aruco as aruco
import numpy as np
import time

class ArucoDetector:
    def __init__(self, dictionary_type=aruco.DICT_4X4_50):
        """Initialize ArUco detector with specified dictionary."""
        # Set up ArUco detection
        self.aruco_dict = aruco.Dictionary_get(dictionary_type)
        self.parameters = aruco.DetectorParameters_create()
        
        # Enhance parameters for better detection
        self.parameters.adaptiveThreshConstant = 7
        self.parameters.adaptiveThreshWinSizeMin = 3
        self.parameters.adaptiveThreshWinSizeMax = 23
        self.parameters.adaptiveThreshWinSizeStep = 10
        self.parameters.minMarkerPerimeterRate = 0.03
        self.parameters.maxMarkerPerimeterRate = 0.5
        
        # Stats for performance tracking
        self.detection_count = 0
        self.frame_count = 0
        self.last_detection_time = 0
        self.fps = 0
        self.last_fps_update = time.time()
    
    def detect_markers(self, frame):
        """Detect ArUco markers in the given frame."""
        # Update frame count
        self.frame_count += 1
        
        # Update FPS calculation every second
        current_time = time.time()
        if current_time - self.last_fps_update >= 1.0:
            self.fps = self.frame_count / (current_time - self.last_fps_update)
            self.frame_count = 0
            self.last_fps_update = current_time
        
        # Start with a copy of the frame for drawing
        output_frame = frame.copy()
        
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply image enhancement for better detection
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # Detect markers
        corners, ids, rejected = aruco.detectMarkers(enhanced, self.aruco_dict, parameters=self.parameters)
        
        # Prepare detection data
        detections = []
        
        # Process detected markers
        if ids is not None and len(ids) > 0:
            # Draw detected markers
            aruco.drawDetectedMarkers(output_frame, corners, ids)
            
            # Process each detected marker
            for i, corner in enumerate(corners):
                # Extract points
                pts = corner.reshape(4, 2)
                
                # Calculate center
                center_x = int(np.mean(pts[:, 0]))
                center_y = int(np.mean(pts[:, 1]))
                center = (center_x, center_y)
                
                # Calculate bounding box
                x_min, y_min = np.min(pts, axis=0).astype(int)
                x_max, y_max = np.max(pts, axis=0).astype(int)
                width = x_max - x_min
                height = y_max - y_min
                
                # Draw center crosshair
                cv2.line(output_frame, (center_x - 10, center_y), (center_x + 10, center_y), (0, 0, 255), 2)
                cv2.line(output_frame, (center_x, center_y - 10), (center_x, center_y + 10), (0, 0, 255), 2)
                
                # Draw marker ID and position info
                label = f"ID: {ids[i][0]}"
                cv2.putText(output_frame, label, (x_min, y_min - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                position = f"Pos: ({center_x}, {center_y})"
                cv2.putText(output_frame, position, (x_min, y_min - 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                # Add fancy targeting display
                # Corner indicators
                cv2.line(output_frame, tuple(pts[0].astype(int)), (int(pts[0][0] + 20), int(pts[0][1])), (0, 255, 255), 2)
                cv2.line(output_frame, tuple(pts[0].astype(int)), (int(pts[0][0]), int(pts[0][1] + 20)), (0, 255, 255), 2)
                
                cv2.line(output_frame, tuple(pts[2].astype(int)), (int(pts[2][0] - 20), int(pts[2][1])), (0, 255, 255), 2)
                cv2.line(output_frame, tuple(pts[2].astype(int)), (int(pts[2][0]), int(pts[2][1] - 20)), (0, 255, 255), 2)
                
                # Approximate distance calculation
                # This is a simple approximation - real distance would need camera calibration
                marker_size_pixels = np.linalg.norm(pts[0] - pts[1])
                distance_estimate = int(1000 / max(1, marker_size_pixels) * 100)  # Arbitrary scale
                
                # Add distance info
                cv2.putText(output_frame, f"~{distance_estimate}cm", (x_min, y_min - 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                # Save detection info
                detections.append({
                    'id': int(ids[i][0]),
                    'center': center,
                    'bbox': (int(x_min), int(y_min), int(width), int(height)),
                    'corners': pts.tolist(),
                    'distance': distance_estimate
                })
            
            # Update detection stats
            self.detection_count += 1
            self.last_detection_time = time.time()
            
            # Add detection status banner
            cv2.rectangle(output_frame, (0, 0), (200, 40), (0, 255, 0), -1)
            cv2.putText(output_frame, "DETECTED", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        else:
            # Add "no detection" banner
            cv2.rectangle(output_frame, (0, 0), (250, 40), (0, 0, 255), -1)
            cv2.putText(output_frame, "NO DETECTION", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Add performance stats
        detection_rate = 100 * self.detection_count / max(1, self.frame_count)
        cv2.putText(output_frame, f"FPS: {self.fps:.1f}", (10, output_frame.shape[0] - 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        cv2.putText(output_frame, f"Detection Rate: {detection_rate:.1f}%", (10, output_frame.shape[0] - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Add timestamp
        timestamp = time.strftime("%H:%M:%S")
        cv2.putText(output_frame, timestamp, (output_frame.shape[1] - 100, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return len(detections) > 0, output_frame, detections

def generate_marker(marker_id, size=500, filename=None):
    """Generate an ArUco marker image."""
    aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_50)
    marker_image = aruco.drawMarker(aruco_dict, marker_id, size)
    
    # Add white border for better detection
    border_size = int(size * 0.1)  # 10% border
    bordered_marker = np.ones((size + 2*border_size, size + 2*border_size), dtype=np.uint8) * 255
    bordered_marker[border_size:border_size+size, border_size:border_size+size] = marker_image
    
    if filename:
        cv2.imwrite(filename, bordered_marker)
        
    return bordered_marker