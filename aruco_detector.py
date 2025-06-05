# aruco_detector.py - Enhanced for 6x6 ArUco markers
import cv2
import numpy as np
import time

class ArucoDetector:
    """Enhanced ArUco detector optimized for 6x6 markers in FPV applications."""
    
    def __init__(self):
        """Initialize the ArUco detector with multiple dictionaries."""
        self.aruco_dicts = {}
        self.aruco_params = None
        self.last_detection_time = 0
        self.detection_history = []
        self.max_history = 5
        
        # Initialize ArUco detection
        self._init_aruco()
        
        # Tracking for smooth detection
        self.tracked_markers = {}
        self.tracking_threshold = 50  # pixels
        
    def _init_aruco(self):
        """Initialize ArUco dictionaries with 6x6 priority."""
        try:
            import cv2.aruco as aruco
            
            # Priority order: 6x6 dictionaries first
            dict_configs = [
                (aruco.DICT_6X6_50, "6x6_50"),
                (aruco.DICT_6X6_100, "6x6_100"),
                (aruco.DICT_6X6_250, "6x6_250"),
                (aruco.DICT_6X6_1000, "6x6_1000"),
                (aruco.DICT_4X4_50, "4x4_50"),
                (aruco.DICT_5X5_100, "5x5_100"),
            ]
            
            for dict_type, name in dict_configs:
                try:
                    self.aruco_dicts[name] = aruco.Dictionary_get(dict_type)
                    print(f"✅ Loaded ArUco dictionary: {name}")
                except Exception as e:
                    print(f"❌ Failed to load {name}: {e}")
            
            # Create optimized detector parameters
            self.aruco_params = aruco.DetectorParameters_create()
            self._optimize_parameters()
            
            print(f"ArUco detector initialized with {len(self.aruco_dicts)} dictionaries")
            
        except ImportError:
            print("❌ OpenCV ArUco module not available")
            raise
    
    def _optimize_parameters(self):
        """Optimize detection parameters for 6x6 markers on RPi camera."""
        # Adaptive thresholding
        self.aruco_params.adaptiveThreshWinSizeMin = 3
        self.aruco_params.adaptiveThreshWinSizeMax = 23
        self.aruco_params.adaptiveThreshWinSizeStep = 10
        self.aruco_params.adaptiveThreshConstant = 7
        
        # Corner refinement for accuracy
        self.aruco_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        self.aruco_params.cornerRefinementWinSize = 5
        self.aruco_params.cornerRefinementMaxIterations = 30
        self.aruco_params.cornerRefinementMinAccuracy = 0.1
        
        # Detection parameters
        self.aruco_params.minMarkerPerimeterRate = 0.03
        self.aruco_params.maxMarkerPerimeterRate = 4.0
        self.aruco_params.polygonalApproxAccuracyRate = 0.03
        self.aruco_params.minCornerDistanceRate = 0.05
        self.aruco_params.minDistanceToBorder = 3
        
        # Bit extraction
        self.aruco_params.markerBorderBits = 1
        self.aruco_params.perspectiveRemovePixelPerCell = 8
        self.aruco_params.perspectiveRemoveIgnoredMarginPerCell = 0.13
        
        # Error correction
        self.aruco_params.maxErroneousBitsInBorderRate = 0.35
        self.aruco_params.errorCorrectionRate = 0.6
        
        # Additional optimizations
        self.aruco_params.minOtsuStdDev = 5.0
        self.aruco_params.minMarkerDistanceRate = 0.05
    
    def preprocess_frame(self, frame):
        """Preprocess frame for better detection."""
        # Convert to grayscale if needed
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame
        
        # Apply CLAHE for contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # Denoise while preserving edges
        denoised = cv2.bilateralFilter(enhanced, 9, 75, 75)
        
        return denoised
    
    def detect_markers(self, frame):
        """
        Detect ArUco markers with enhanced processing.
        
        Returns:
            detected (bool): Whether any markers were detected
            output_frame (numpy.ndarray): Annotated frame
            detections (list): List of detection dictionaries
        """
        import cv2.aruco as aruco
        
        output_frame = frame.copy()
        all_detections = []
        
        # Preprocess frame
        gray = self.preprocess_frame(frame)
        
        # Try detection with each dictionary (6x6 first)
        for dict_name, aruco_dict in self.aruco_dicts.items():
            # Skip non-6x6 if we already found 6x6 markers
            if all_detections and any(d['dictionary'].startswith('6x6') for d in all_detections):
                if not dict_name.startswith('6x6'):
                    continue
            
            try:
                # Detect markers
                corners, ids, rejected = aruco.detectMarkers(
                    gray, aruco_dict, parameters=self.aruco_params
                )
                
                # If no detection with preprocessed, try original
                if ids is None and dict_name.startswith('6x6'):
                    gray_original = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    corners, ids, rejected = aruco.detectMarkers(
                        gray_original, aruco_dict, parameters=self.aruco_params
                    )
                
                if ids is not None and len(ids) > 0:
                    # Process each detected marker
                    for i, (corner, marker_id) in enumerate(zip(corners, ids.flatten())):
                        detection = self._process_detection(
                            corner, marker_id, dict_name, frame.shape
                        )
                        all_detections.append(detection)
                    
                    print(f"Found {len(ids)} markers with {dict_name}")
                    
                    # Draw on output frame
                    self._draw_detections(output_frame, corners, ids, dict_name)
                    
                    # Stop if we found 6x6 markers
                    if dict_name.startswith('6x6'):
                        break
                        
            except Exception as e:
                print(f"Error detecting with {dict_name}: {e}")
                continue
        
        # Update tracking
        self._update_tracking(all_detections)
        
        # Draw status overlay
        self._draw_status_overlay(output_frame, all_detections)
        
        # Update detection history
        self.detection_history.append(len(all_detections))
        if len(self.detection_history) > self.max_history:
            self.detection_history.pop(0)
        
        detected = len(all_detections) > 0
        return detected, output_frame, all_detections
    
    def _process_detection(self, corner, marker_id, dict_name, frame_shape):
        """Process a single marker detection."""
        pts = corner.reshape(4, 2)
        
        # Calculate center
        center_x = int(np.mean(pts[:, 0]))
        center_y = int(np.mean(pts[:, 1]))
        center = (center_x, center_y)
        
        # Calculate bounding box
        x_min, y_min = np.min(pts, axis=0).astype(int)
        x_max, y_max = np.max(pts, axis=0).astype(int)
        bbox = (x_min, y_min, x_max - x_min, y_max - y_min)
        
        # Estimate distance (rough approximation)
        marker_size_pixels = np.mean([np.linalg.norm(pts[i] - pts[(i+1)%4]) for i in range(4)])
        distance_cm = int(5000 / max(marker_size_pixels, 1))  # Assumes 5cm marker
        
        # Calculate quality metric
        corner_distances = [np.linalg.norm(pts[i] - pts[(i+1)%4]) for i in range(4)]
        quality = 100 - min(np.std(corner_distances) * 10, 100)
        
        # Calculate pose angles (simplified)
        # This is a rough estimation - use cv2.aruco.estimatePoseSingleMarkers for accurate pose
        width = pts[1][0] - pts[0][0]
        height = pts[2][1] - pts[1][1]
        angle = np.degrees(np.arctan2(pts[1][1] - pts[0][1], pts[1][0] - pts[0][0]))
        
        return {
            'id': int(marker_id),
            'center': center,
            'bbox': bbox,
            'corners': pts.tolist(),
            'distance': distance_cm,
            'quality': int(quality),
            'angle': angle,
            'dictionary': dict_name,
            'timestamp': time.time(),
            'frame_position': (center_x / frame_shape[1], center_y / frame_shape[0])  # Normalized
        }
    
    def _update_tracking(self, detections):
        """Update marker tracking for smooth detection."""
        current_ids = {d['id']: d for d in detections}
        
        # Update existing tracked markers
        for marker_id in list(self.tracked_markers.keys()):
            if marker_id in current_ids:
                # Update position with smoothing
                old_pos = self.tracked_markers[marker_id]['center']
                new_pos = current_ids[marker_id]['center']
                
                # Simple exponential smoothing
                alpha = 0.7
                smooth_x = int(alpha * new_pos[0] + (1 - alpha) * old_pos[0])
                smooth_y = int(alpha * new_pos[1] + (1 - alpha) * old_pos[1])
                
                self.tracked_markers[marker_id] = current_ids[marker_id]
                self.tracked_markers[marker_id]['center'] = (smooth_x, smooth_y)
                self.tracked_markers[marker_id]['tracked_frames'] = \
                    self.tracked_markers[marker_id].get('tracked_frames', 0) + 1
            else:
                # Remove if not detected
                del self.tracked_markers[marker_id]
        
        # Add new markers
        for marker_id, detection in current_ids.items():
            if marker_id not in self.tracked_markers:
                self.tracked_markers[marker_id] = detection
                self.tracked_markers[marker_id]['tracked_frames'] = 1
    
    def _draw_detections(self, frame, corners, ids, dict_name):
        """Draw detected markers on frame."""
        import cv2.aruco as aruco
        
        # Draw detected markers
        aruco.drawDetectedMarkers(frame, corners, ids)
        
        # Add custom annotations
        for i, (corner, marker_id) in enumerate(zip(corners, ids.flatten())):
            pts = corner.reshape(4, 2)
            center = pts.mean(axis=0).astype(int)
            
            # Draw center crosshair
            cv2.drawMarker(frame, tuple(center), (0, 0, 255), 
                          cv2.MARKER_CROSS, 20, 2)
            
            # Add ID label with background
            label = f"ID:{marker_id} ({dict_name})"
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            x_min = int(pts[:, 0].min())
            y_min = int(pts[:, 1].min()) - 10
            
            cv2.rectangle(frame, (x_min, y_min - h - 4), 
                         (x_min + w + 4, y_min), (0, 0, 0), -1)
            cv2.putText(frame, label, (x_min + 2, y_min - 2), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Draw corner points
            for j, pt in enumerate(pts):
                cv2.circle(frame, tuple(pt.astype(int)), 4, (255, 0, 255), -1)
                cv2.putText(frame, str(j), tuple(pt.astype(int) + 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
    
    def _draw_status_overlay(self, frame, detections):
        """Draw status information overlay."""
        h, w = frame.shape[:2]
        
        # Status background
        overlay_h = 40
        cv2.rectangle(frame, (0, h - overlay_h), (w, h), (0, 0, 0), -1)
        cv2.rectangle(frame, (0, h - overlay_h), (w, h), (0, 0, 0), 3)
        
        # Detection status
        if detections:
            status = f"ArUco: {len(detections)} detected"
            color = (0, 255, 0)
            
            # Add marker IDs
            ids_str = ", ".join([f"{d['id']}" for d in detections[:5]])
            if len(detections) > 5:
                ids_str += "..."
            status += f" | IDs: {ids_str}"
        else:
            status = "ArUco: No markers detected"
            color = (0, 0, 255)
        
        # Detection stability indicator
        if len(self.detection_history) > 2:
            stability = np.std(self.detection_history)
            if stability < 0.5:
                status += " | Stable"
            else:
                status += " | Unstable"
        
        cv2.putText(frame, status, (10, h - 15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # FPS indicator (if available)
        current_time = time.time()
        if hasattr(self, 'last_fps_time'):
            fps = 1.0 / (current_time - self.last_fps_time)
            cv2.putText(frame, f"FPS: {fps:.1f}", (w - 100, h - 15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        self.last_fps_time = current_time
    
    def get_tracked_markers(self):
        """Get currently tracked markers with smoothed positions."""
        return self.tracked_markers.copy()
    
    def reset_tracking(self):
        """Reset marker tracking."""
        self.tracked_markers.clear()
        self.detection_history.clear()