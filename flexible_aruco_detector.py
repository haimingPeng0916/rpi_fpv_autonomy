# flexible_aruco_detector.py - Support multiple ArUco dictionaries
import cv2
import cv2.aruco as aruco
import numpy as np
import time

class FlexibleArucoDetector:
    def __init__(self, dictionary_types=None):
        """Initialize ArUco detector with multiple dictionary support."""
        if dictionary_types is None:
            # Try multiple common dictionaries
            dictionary_types = [
                aruco.DICT_4X4_50,
                aruco.DICT_6X6_250,
                aruco.DICT_5X5_100,
                aruco.DICT_ARUCO_ORIGINAL
            ]
        
        self.dictionaries = []
        self.dict_names = []
        
        # Initialize all dictionaries
        dict_name_map = {
            aruco.DICT_4X4_50: "4x4_50",
            aruco.DICT_6X6_250: "6x6_250", 
            aruco.DICT_5X5_100: "5x5_100",
            aruco.DICT_ARUCO_ORIGINAL: "ARUCO_ORIGINAL"
        }
        
        for dict_type in dictionary_types:
            try:
                aruco_dict = aruco.Dictionary_get(dict_type)
                self.dictionaries.append(aruco_dict)
                self.dict_names.append(dict_name_map.get(dict_type, f"DICT_{dict_type}"))
                print(f"✅ Loaded dictionary: {dict_name_map.get(dict_type, f'DICT_{dict_type}')}")
            except Exception as e:
                print(f"❌ Failed to load dictionary {dict_type}: {e}")
        
        if not self.dictionaries:
            raise ValueError("No ArUco dictionaries could be loaded!")
        
        # Create detection parameters
        self.parameters = aruco.DetectorParameters_create()
        
        # Enhanced parameters for Pi camera
        self.parameters.adaptiveThreshConstant = 7
        self.parameters.adaptiveThreshWinSizeMin = 3
        self.parameters.adaptiveThreshWinSizeMax = 23
        self.parameters.adaptiveThreshWinSizeStep = 10
        self.parameters.minMarkerPerimeterRate = 0.03
        self.parameters.maxMarkerPerimeterRate = 0.5
        self.parameters.polygonalApproxAccuracyRate = 0.03
        self.parameters.minCornerDistanceRate = 0.05
        self.parameters.minDistanceToBorder = 3
        self.parameters.minMarkerDistanceRate = 0.05
        
        # Stats
        self.detection_count = 0
        self.frame_count = 0
        self.fps = 0
        self.last_fps_update = time.time()
        
        print(f"FlexibleArucoDetector initialized with {len(self.dictionaries)} dictionaries")
    
    def detect_markers(self, frame):
        """Detect ArUco markers using all available dictionaries."""
        self.frame_count += 1
        
        # Update FPS
        current_time = time.time()
        if current_time - self.last_fps_update >= 1.0:
            self.fps = self.frame_count / (current_time - self.last_fps_update)
            self.frame_count = 0
            self.last_fps_update = current_time
        
        output_frame = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Enhanced image processing for Pi camera
        enhanced = self._enhance_image_for_aruco(gray)
        
        all_detections = []
        successful_dict = None
        
        # Try each dictionary
        for i, (aruco_dict, dict_name) in enumerate(zip(self.dictionaries, self.dict_names)):
            try:
                corners, ids, rejected = aruco.detectMarkers(enhanced, aruco_dict, parameters=self.parameters)
                
                if ids is not None and len(ids) > 0:
                    print(f"🎯 Found {len(ids)} markers using {dict_name} dictionary")
                    successful_dict = dict_name
                    
                    # Process detections
                    for j, corner in enumerate(corners):
                        pts = corner.reshape(4, 2)
                        center_x = int(np.mean(pts[:, 0]))
                        center_y = int(np.mean(pts[:, 1]))
                        center = (center_x, center_y)
                        
                        x_min, y_min = np.min(pts, axis=0).astype(int)
                        x_max, y_max = np.max(pts, axis=0).astype(int)
                        width = x_max - x_min
                        height = y_max - y_min
                        
                        marker_size_pixels = np.linalg.norm(pts[0] - pts[1])
                        distance_estimate = int(1000 / max(1, marker_size_pixels) * 100)
                        
                        detection = {
                            'id': int(ids[j][0]),
                            'center': center,
                            'bbox': (int(x_min), int(y_min), int(width), int(height)),
                            'corners': pts.tolist(),
                            'distance': distance_estimate,
                            'marker_size_pixels': marker_size_pixels,
                            'dictionary': dict_name
                        }
                        all_detections.append(detection)
                    
                    # Draw markers
                    aruco.drawDetectedMarkers(output_frame, corners, ids)
                    
                    # Draw detection info
                    for j, detection in enumerate(all_detections):
                        self._draw_marker_info(output_frame, detection, j)
                    
                    break  # Found markers, no need to try other dictionaries
                    
            except Exception as e:
                print(f"❌ Error with {dict_name}: {e}")
                continue
        
        # Add debug information
        self._add_debug_info(output_frame, enhanced, successful_dict, len(all_detections))
        
        if all_detections:
            self.detection_count += 1
            return True, output_frame, all_detections
        else:
            return False, output_frame, []
    
    def _enhance_image_for_aruco(self, gray):
        """Enhanced image processing for better detection."""
        # Multiple enhancement techniques
        enhanced = gray.copy()
        
        # 1. Histogram equalization
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(enhanced)
        
        # 2. Noise reduction
        enhanced = cv2.medianBlur(enhanced, 3)
        
        # 3. Slight gaussian blur to help with focus issues
        enhanced = cv2.GaussianBlur(enhanced, (3, 3), 0)
        
        # 4. Sharpening to enhance edges
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        enhanced = cv2.filter2D(enhanced, -1, kernel)
        
        # 5. Ensure values are in valid range
        enhanced = np.clip(enhanced, 0, 255).astype(np.uint8)
        
        return enhanced
    
    def _draw_marker_info(self, frame, detection, index):
        """Draw information for each detected marker."""
        center = detection['center']
        marker_id = detection['id']
        dict_name = detection['dictionary']
        
        # Draw center crosshair
        cv2.line(frame, (center[0] - 15, center[1]), (center[0] + 15, center[1]), (0, 0, 255), 3)
        cv2.line(frame, (center[0], center[1] - 15), (center[0], center[1] + 15), (0, 0, 255), 3)
        
        # Draw marker info
        bbox = detection['bbox']
        x, y = bbox[0], bbox[1]
        
        # Use different colors for different dictionaries
        colors = [(0, 255, 0), (255, 0, 0), (0, 255, 255), (255, 0, 255)]
        color = colors[index % len(colors)]
        
        cv2.putText(frame, f"ID: {marker_id}", (x, y - 35), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.putText(frame, f"Dict: {dict_name}", (x, y - 15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        cv2.putText(frame, f"Pos: ({center[0]}, {center[1]})", (x, y - 55), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        # Draw corners with different colors
        corners = np.array(detection['corners']).astype(int)
        corner_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]
        for i, pt in enumerate(corners):
            cv2.circle(frame, tuple(pt), 5, corner_colors[i], -1)
    
    def _add_debug_info(self, frame, enhanced, successful_dict, num_detections):
        """Add debug information to frame."""
        h, w = frame.shape[:2]
        
        # Detection status
        if num_detections > 0:
            cv2.rectangle(frame, (0, 0), (300, 40), (0, 255, 0), -1)
            cv2.putText(frame, f"DETECTED: {num_detections} markers", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        else:
            cv2.rectangle(frame, (0, 0), (250, 40), (0, 0, 255), -1)
            cv2.putText(frame, "NO MARKERS DETECTED", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Dictionary info
        if successful_dict:
            cv2.putText(frame, f"Dictionary: {successful_dict}", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Performance stats
        cv2.putText(frame, f"FPS: {self.fps:.1f}", (10, h - 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Show enhanced image
        try:
            small_enhanced = cv2.resize(enhanced, (160, 120))
            small_enhanced_colored = cv2.cvtColor(small_enhanced, cv2.COLOR_GRAY2BGR)
            frame[h-130:h-10, w-170:w-10] = small_enhanced_colored
            cv2.rectangle(frame, (w-170, h-130), (w-10, h-10), (255, 255, 255), 2)
            cv2.putText(frame, "Enhanced", (w-160, h-135), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        except:
            pass

def generate_test_markers():
    """Generate test markers in different formats."""
    print("Generating test markers in multiple formats...")
    
    # Dictionary types to test
    test_dicts = [
        (aruco.DICT_4X4_50, "4x4_50"),
        (aruco.DICT_6X6_250, "6x6_250"),
        (aruco.DICT_5X5_100, "5x5_100")
    ]
    
    for dict_type, name in test_dicts:
        try:
            aruco_dict = aruco.Dictionary_get(dict_type)
            
            # Generate markers 0, 1, 2 for each dictionary
            for marker_id in [0, 1, 2]:
                marker_image = aruco.drawMarker(aruco_dict, marker_id, 200)
                
                # Add border
                border_size = 30
                bordered = np.ones((260, 260), dtype=np.uint8) * 255
                bordered[30:230, 30:230] = marker_image
                
                filename = f"marker_{name}_id_{marker_id}.png"
                cv2.imwrite(filename, bordered)
                print(f"Generated: {filename}")
                
        except Exception as e:
            print(f"Error generating {name}: {e}")

def test_flexible_detector():
    """Test the flexible detector."""
    print("=== Testing Flexible ArUco Detector ===")
    
    detector = FlexibleArucoDetector()
    
    # Test with a black frame (should detect nothing)
    test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(test_frame, "Test frame - no markers", (50, 240), 
               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    detected, frame, detections = detector.detect_markers(test_frame)
    print(f"Test result: detected={detected}, detections={len(detections)}")

if __name__ == "__main__":
    generate_test_markers()
    test_flexible_detector()