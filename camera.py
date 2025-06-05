# camera.py - Enhanced for better 6x6 ArUco detection
import cv2
import numpy as np
import time
from picamera2 import Picamera2

class Camera:
    def __init__(self, width=640, height=480):
        """Initialize the camera and detection settings."""
        self.width = width
        self.height = height
        self.camera = None
        self.running = False
        self.last_frame = None
        self.last_detection = None
        
        # Detection mode: 'box', 'aruco', or 'none'
        self.detection_mode = 'aruco'  # Default to ArUco detection
        
        # Box detection parameters
        self.threshold_value = 127
        self.min_area = 1000
        self.aspect_ratio_min = 0.8
        self.aspect_ratio_max = 1.2
        
        # ArUco detection
        self.aruco_detector = None
        self._init_aruco_detector()
        
        # Visual settings
        self.show_debug_info = True
        self.show_enhanced_view = False
        
        # Performance tracking
        self.fps_counter = 0
        self.last_fps_time = time.time()
        self.current_fps = 0
    
    def _init_aruco_detector(self):
        """Initialize ArUco detector with enhanced parameters for 6x6 markers."""
        self.aruco_dicts = []
        self.dict_names = []
        self.aruco_params = None
        
        try:
            print("Initializing Enhanced ArUco detector...")
            import cv2.aruco as aruco
            
            # Prioritize 6x6 dictionaries first
            dict_types = [
                (aruco.DICT_6X6_50, "6x6_50"),
                (aruco.DICT_6X6_100, "6x6_100"),
                (aruco.DICT_6X6_250, "6x6_250"),
                (aruco.DICT_6X6_1000, "6x6_1000"),
                # Fallback options
                (aruco.DICT_4X4_50, "4x4_50"),
                (aruco.DICT_5X5_100, "5x5_100"),
            ]
            
            for dict_type, name in dict_types:
                try:
                    aruco_dict = aruco.Dictionary_get(dict_type)
                    self.aruco_dicts.append(aruco_dict)
                    self.dict_names.append(name)
                    print(f"✅ Loaded ArUco dictionary: {name}")
                except Exception as e:
                    print(f"❌ Failed to load {name}: {e}")
            
            if self.aruco_dicts:
                # Create optimized detection parameters for 6x6 markers
                self.aruco_params = aruco.DetectorParameters_create()
                self._optimize_aruco_params()
                
                print(f"✅ ArUco detector initialized with {len(self.aruco_dicts)} dictionaries")
                return True
            else:
                print("❌ No ArUco dictionaries available, falling back to box detection")
                self.detection_mode = 'box'
                return False
                
        except ImportError as e:
            print(f"❌ ArUco import failed: {e}")
            print("OpenCV ArUco module not available")
            self.detection_mode = 'box'
            return False
        except Exception as e:
            print(f"❌ ArUco initialization failed: {e}")
            import traceback
            traceback.print_exc()
            self.detection_mode = 'box'
            return False
    
    def _optimize_aruco_params(self):
        """Optimize ArUco parameters specifically for 6x6 markers on Pi Camera."""
        # Adaptive thresholding - crucial for varying lighting
        self.aruco_params.adaptiveThreshWinSizeMin = 3
        self.aruco_params.adaptiveThreshWinSizeMax = 23
        self.aruco_params.adaptiveThreshWinSizeStep = 10
        self.aruco_params.adaptiveThreshConstant = 7
        
        # Corner refinement - improves accuracy
        self.aruco_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        self.aruco_params.cornerRefinementWinSize = 5
        self.aruco_params.cornerRefinementMaxIterations = 30
        self.aruco_params.cornerRefinementMinAccuracy = 0.1
        
        # Marker detection parameters
        self.aruco_params.minMarkerPerimeterRate = 0.03
        self.aruco_params.maxMarkerPerimeterRate = 4.0
        self.aruco_params.polygonalApproxAccuracyRate = 0.03
        self.aruco_params.minCornerDistanceRate = 0.05
        self.aruco_params.minDistanceToBorder = 3
        
        # Bit extraction - important for 6x6 markers
        self.aruco_params.markerBorderBits = 1
        self.aruco_params.perspectiveRemovePixelPerCell = 8
        self.aruco_params.perspectiveRemoveIgnoredMarginPerCell = 0.13
        
        # Error correction - helps with partial occlusions
        self.aruco_params.maxErroneousBitsInBorderRate = 0.35
        self.aruco_params.errorCorrectionRate = 0.6
        
        # Contour filtering
        self.aruco_params.minOtsuStdDev = 5.0
        self.aruco_params.minMarkerDistanceRate = 0.05
    
    def set_detection_mode(self, mode):
        """Set detection mode: 'box', 'aruco', or 'none'"""
        if mode == 'aruco' and not self.aruco_dicts:
            print("ArUco not available, staying in current mode")
            return False
        
        if mode in ['box', 'aruco', 'none']:
            self.detection_mode = mode
            print(f"Detection mode set to: {mode}")
            return True
        return False
    
    def set_visual_settings(self, show_debug=True, show_enhanced=False):
        """Configure visual overlays."""
        self.show_debug_info = show_debug
        self.show_enhanced_view = show_enhanced
    
    def start(self):
        """Start the camera with optimized settings for ArUco detection."""
        if self.camera is not None:
            print("Camera already started")
            return True
        
        try:
            self.camera = Picamera2()
            
            # Configure camera for better ArUco detection
            config = self.camera.create_preview_configuration(
                main={"size": (self.width, self.height)},
                controls={
                    "ExposureTime": 20000,  # Fixed exposure for consistent detection
                    "AnalogueGain": 4.0,    # Moderate gain to reduce noise
                    "Contrast": 1.2,        # Slightly increased contrast
                }
            )
            self.camera.configure(config)
            self.camera.start()
            time.sleep(1)
            
            self.running = True
            print(f"✅ Camera started in {self.detection_mode} mode")
            return True
            
        except Exception as e:
            print(f"❌ Error starting camera: {e}")
            self.running = False
            return False
    
    def stop(self):
        """Stop the camera."""
        try:
            self.running = False
            if self.camera:
                self.camera.stop()
                time.sleep(0.1)
                self.camera.close()
                self.camera = None
            print("✅ Camera stopped")
        except Exception as e:
            print(f"❌ Error stopping camera: {e}")
    
    def _enhance_image(self, gray):
        """Enhanced image preprocessing for better 6x6 ArUco detection."""
        # Apply CLAHE for better contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # Bilateral filter to reduce noise while keeping edges
        enhanced = cv2.bilateralFilter(enhanced, 9, 75, 75)
        
        # Optional: Sharpen the image slightly
        kernel = np.array([[-1,-1,-1],
                          [-1, 9,-1],
                          [-1,-1,-1]])
        enhanced = cv2.filter2D(enhanced, -1, kernel)
        
        return enhanced
    
    def detect_aruco(self, frame):
        """Enhanced ArUco detection with multi-pass approach for 6x6 markers."""
        if not self.aruco_dicts or not self.aruco_params:
            print("ArUco not initialized, falling back to box detection")
            return self.detect_box(frame)
        
        try:
            output_frame = frame.copy()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # First pass: Try with enhanced image
            enhanced = self._enhance_image(gray)
            
            found_detections = []
            successful_dict = None
            
            import cv2.aruco as aruco
            
            # Try each dictionary with both enhanced and original image
            for aruco_dict, dict_name in zip(self.aruco_dicts, self.dict_names):
                # Skip non-6x6 dictionaries if we've already found 6x6 markers
                if successful_dict and successful_dict.startswith("6x6") and not dict_name.startswith("6x6"):
                    continue
                
                try:
                    # First attempt with enhanced image
                    corners, ids, rejected = aruco.detectMarkers(enhanced, aruco_dict, parameters=self.aruco_params)
                    
                    # If no detection, try with original grayscale
                    if ids is None or len(ids) == 0:
                        corners, ids, rejected = aruco.detectMarkers(gray, aruco_dict, parameters=self.aruco_params)
                    
                    if ids is not None and len(ids) > 0:
                        successful_dict = dict_name
                        print(f"🎯 ArUco: Found {len(ids)} markers using {dict_name}")
                        
                        # Draw markers with enhanced visualization
                        aruco.drawDetectedMarkers(output_frame, corners, ids)
                        
                        # Process each detection
                        for i, corner in enumerate(corners):
                            pts = corner.reshape(4, 2)
                            center_x = int(np.mean(pts[:, 0]))
                            center_y = int(np.mean(pts[:, 1]))
                            center = (center_x, center_y)
                            
                            # Bounding box
                            x_min, y_min = np.min(pts, axis=0).astype(int)
                            x_max, y_max = np.max(pts, axis=0).astype(int)
                            
                            marker_id = ids[i][0]
                            
                            # Enhanced visualization
                            # Draw corner points
                            for pt in pts:
                                cv2.circle(output_frame, tuple(pt.astype(int)), 3, (255, 0, 255), -1)
                            
                            # Center crosshair
                            cv2.line(output_frame, (center_x - 15, center_y), (center_x + 15, center_y), (0, 0, 255), 2)
                            cv2.line(output_frame, (center_x, center_y - 15), (center_x, center_y + 15), (0, 0, 255), 2)
                            
                            # ID with background for visibility
                            label = f"ID: {marker_id}"
                            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                            cv2.rectangle(output_frame, (x_min, y_min - h - 5), (x_min + w, y_min), (0, 0, 0), -1)
                            cv2.putText(output_frame, label, (x_min, y_min - 5), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                            
                            # Marker quality indicator (based on corner variance)
                            corner_distances = [np.linalg.norm(pts[i] - pts[(i+1)%4]) for i in range(4)]
                            quality = 100 - min(np.std(corner_distances) * 10, 100)
                            quality_color = (0, int(quality * 2.55), int((100-quality) * 2.55))
                            cv2.putText(output_frame, f"Q: {int(quality)}%", (x_min, y_max + 20), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, quality_color, 2)
                            
                            # Distance estimate
                            marker_size = np.mean(corner_distances)
                            distance = int(1000 / max(1, marker_size) * 100)
                            cv2.putText(output_frame, f"{distance}cm", (x_min, y_min - 25), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
                            
                            found_detections.append({
                                'id': int(marker_id),
                                'center': center,
                                'bbox': (x_min, y_min, x_max - x_min, y_max - y_min),
                                'distance': distance,
                                'dictionary': dict_name,
                                'quality': int(quality),
                                'corners': pts.tolist()
                            })
                        
                        # If we found 6x6 markers, stop searching
                        if dict_name.startswith("6x6"):
                            break
                            
                except Exception as dict_error:
                    print(f"❌ Error with {dict_name}: {dict_error}")
                    continue
            
            # Update FPS
            self._update_fps()
            
            # Enhanced status display
            if found_detections:
                # Success banner with more info
                cv2.rectangle(output_frame, (0, 0), (350, 35), (0, 150, 0), -1)
                status_text = f"ARUCO {successful_dict}: {len(found_detections)} detected"
                cv2.putText(output_frame, status_text, (10, 25), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # FPS display
                cv2.putText(output_frame, f"FPS: {self.current_fps:.1f}", 
                           (output_frame.shape[1] - 100, 25), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                # Save detection
                self.last_detection = {
                    'center': found_detections[0]['center'],
                    'bbox': found_detections[0]['bbox'],
                    'id': found_detections[0]['id'],
                    'distance': found_detections[0]['distance'],
                    'timestamp': time.time(),
                    'all_detections': found_detections
                }
                
                return True, output_frame, found_detections[0]['center'], found_detections[0]['bbox']
            else:
                # No detection banner with rejected count
                cv2.rectangle(output_frame, (0, 0), (250, 35), (0, 0, 150), -1)
                cv2.putText(output_frame, f"NO ARUCO (rejected: {len(rejected)})", (10, 25), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # Show debug view if enabled
                if self.show_enhanced_view and len(rejected) > 0:
                    cv2.aruco.drawDetectedMarkers(output_frame, rejected, borderColor=(100, 0, 255))
                
                return False, output_frame, None, None
                
        except Exception as e:
            print(f"❌ ArUco detection error: {e}")
            import traceback
            traceback.print_exc()
            cv2.putText(frame, "ARUCO ERROR", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            return False, frame, None, None
    
    def _update_fps(self):
        """Update FPS counter."""
        self.fps_counter += 1
        current_time = time.time()
        if current_time - self.last_fps_time > 1.0:
            self.current_fps = self.fps_counter / (current_time - self.last_fps_time)
            self.fps_counter = 0
            self.last_fps_time = current_time
    
    def detect_box(self, frame):
        """Simple box detection with clean visualization."""
        try:
            output_frame = frame.copy()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, self.threshold_value, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            found = False
            center = None
            bbox = None
            
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                area = cv2.contourArea(largest_contour)
                
                if area >= self.min_area:
                    x, y, w, h = cv2.boundingRect(largest_contour)
                    center = (x + w//2, y + h//2)
                    bbox = (x, y, w, h)
                    found = True
                    
                    # Clean visualization
                    cv2.rectangle(output_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    cv2.circle(output_frame, center, 5, (0, 0, 255), -1)
                    cv2.putText(output_frame, f"Box: {w}x{h}", (x, y-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Status
            if found:
                cv2.rectangle(output_frame, (0, 0), (150, 35), (0, 150, 0), -1)
                cv2.putText(output_frame, "BOX DETECTED", (10, 25), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            else:
                cv2.rectangle(output_frame, (0, 0), (120, 35), (0, 0, 150), -1)
                cv2.putText(output_frame, "NO BOX", (10, 25), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            return found, output_frame, center, bbox
            
        except Exception as e:
            print(f"❌ Box detection error: {e}")
            return False, frame, None, None
    
    def get_frame(self):
        """Get frame with clean detection visualization."""
        if not self.running or self.camera is None:
            frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            cv2.putText(frame, "Camera not running", (50, self.height//2), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            return False, frame, None, None
        
        try:
            # Capture frame
            frame = self.camera.capture_array()
            
            # Handle format conversion
            if len(frame.shape) == 2:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            elif frame.shape[2] == 4:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            # IMPORTANT FIX: Handle RGB to BGR conversion for Picamera2
            elif frame.shape[2] == 3:
                # Picamera2 returns RGB, OpenCV expects BGR
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            self.last_frame = frame.copy()
            
            # Apply detection based on mode
            if self.detection_mode == 'aruco':
                success, processed_frame, center, bbox = self.detect_aruco(frame)
            elif self.detection_mode == 'box':
                success, processed_frame, center, bbox = self.detect_box(frame)
            else:  # mode == 'none'
                processed_frame = frame.copy()
                cv2.putText(processed_frame, "Detection: OFF", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (128, 128, 128), 2)
                success, center, bbox = False, None, None
            
            # Add minimal debug info if enabled
            if self.show_debug_info:
                cv2.putText(processed_frame, f"Mode: {self.detection_mode.upper()}", 
                        (10, processed_frame.shape[0] - 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # IMPORTANT: Return True for frame capture success (not detection success)
            # The first return value should indicate if we got a valid frame, not if we detected something
            frame_captured_successfully = processed_frame is not None
            return frame_captured_successfully, processed_frame, center, bbox
                
        except Exception as e:
            print(f"❌ Error in get_frame(): {e}")
            if self.last_frame is not None:
                return False, self.last_frame, None, None
            else:
                frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
                cv2.putText(frame, "CAMERA ERROR", (50, self.height//2), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                return False, frame, None, None
    
    def update_settings(self, settings):
        """Update detection settings."""
        if 'detection_mode' in settings:
            self.set_detection_mode(settings['detection_mode'])
        if 'threshold_value' in settings:
            self.threshold_value = settings['threshold_value']
        if 'show_debug_info' in settings:
            self.show_debug_info = settings['show_debug_info']
        if 'show_enhanced_view' in settings:
            self.show_enhanced_view = settings['show_enhanced_view']
    
    def calibrate_camera(self, num_images=20, pattern_size=(9, 6)):
        """Camera calibration helper for better ArUco pose estimation."""
        print("Starting camera calibration...")
        # This would be a separate calibration routine
        # For now, return None
        return None