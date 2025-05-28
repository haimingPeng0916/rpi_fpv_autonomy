# opencv_processor.py - Fixed Version
import cv2
import numpy as np
import time
import math
from aruco_detector import ArucoDetector

class OpenCVProcessor:
    """Handles OpenCV processing and visualization."""
    
    def __init__(self):
        """Initialize the OpenCV processor."""
        self.aruco_detector = ArucoDetector()
        self.modes = ["basic", "edge", "contour", "grid", "aruco"]
        self.current_mode = "basic"
        
        # Edge detection parameters
        self.edge_low = 50
        self.edge_high = 150
        
        # Grid parameters
        self.grid_size = 50
        self.grid_color = (0, 100, 0)
        
        # Contour parameters
        self.contour_threshold = 100
    
    def set_mode(self, mode):
        """Set the current visualization mode."""
        if mode in self.modes:
            self.current_mode = mode
            print(f"OpenCV mode changed to: {mode}")
            return True
        print(f"Invalid mode: {mode}, available modes: {self.modes}")
        return False
    
    def process_frame(self, frame, flight_data):
        """Process a frame with the current visualization mode."""
        # Make a copy to avoid modifying the original
        output = frame.copy()
        
        try:
            # Apply different processing based on mode
            if self.current_mode == "edge":
                output = self._apply_edge_detection(output)
            elif self.current_mode == "contour":
                output = self._apply_contour_detection(output)
            elif self.current_mode == "grid":
                output = self._apply_grid(output)
            elif self.current_mode == "aruco":
                output = self._apply_aruco_detection(output)
            # For "basic" mode, no processing needed
            
            # Always add flight data overlay regardless of mode
            output = self._add_flight_data_overlay(output, flight_data)
            
            return output
            
        except Exception as e:
            print(f"Error in OpenCV processing (mode: {self.current_mode}): {e}")
            # Return original frame with flight data overlay on error
            return self._add_flight_data_overlay(frame.copy(), flight_data)
    
    def _apply_edge_detection(self, frame):
        """Apply Canny edge detection."""
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            edges = cv2.Canny(blurred, self.edge_low, self.edge_high)
            
            # Convert back to color for overlay
            edges_color = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
            
            # Blend with original
            result = cv2.addWeighted(frame, 0.7, edges_color, 0.3, 0)
            
            return result
        except Exception as e:
            print(f"Error in edge detection: {e}")
            return frame
    
    def _apply_contour_detection(self, frame):
        """Apply contour detection."""
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            _, thresh = cv2.threshold(blurred, self.contour_threshold, 255, cv2.THRESH_BINARY)
            
            # Find contours
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Draw all contours on a copy of the frame
            result = frame.copy()
            cv2.drawContours(result, contours, -1, (0, 255, 0), 2)
            
            return result
        except Exception as e:
            print(f"Error in contour detection: {e}")
            return frame
    
    def _apply_grid(self, frame):
        """Apply a grid overlay."""
        try:
            result = frame.copy()
            h, w = frame.shape[:2]
            
            # Draw vertical lines
            for x in range(0, w, self.grid_size):
                cv2.line(result, (x, 0), (x, h), self.grid_color, 1)
            
            # Draw horizontal lines
            for y in range(0, h, self.grid_size):
                cv2.line(result, (0, y), (w, y), self.grid_color, 1)
            
            return result
        except Exception as e:
            print(f"Error in grid overlay: {e}")
            return frame
    
    def _apply_aruco_detection(self, frame):
        """Apply ArUco marker detection."""
        try:
            detected, processed_frame, detections = self.aruco_detector.detect_markers(frame)
            
            if detected:
                print(f"ArUco: {len(detections)} markers detected")
                for detection in detections:
                    print(f"  Marker ID {detection['id']} at {detection['center']}")
            
            return processed_frame
            
        except Exception as e:
            print(f"Error in ArUco detection: {e}")
            return frame
    
    def _add_flight_data_overlay(self, frame, flight_data):
        """Add flight data overlay to the frame."""
        try:
            # Add semi-transparent overlay at the top
            h, w = frame.shape[:2]
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (w, 80), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
            
            # Add flight data text
            # Row 1
            cv2.putText(frame, f"Roll: {flight_data['attitude']['roll']:.1f}°", (10, 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(frame, f"Pitch: {flight_data['attitude']['pitch']:.1f}°", (160, 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(frame, f"Yaw: {flight_data['attitude']['yaw']:.1f}°", (310, 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(frame, f"Alt: {flight_data['altitude']:.1f}m", (460, 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Row 2
            cv2.putText(frame, f"Batt: {flight_data['battery']:.1f}V", (10, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(frame, f"Mode: {flight_data['mode']}", (160, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Armed status with color
            status_color = (0, 255, 0) if flight_data['armed'] else (0, 0, 255)
            status_text = "ARMED" if flight_data['armed'] else "DISARMED"
            cv2.putText(frame, status_text, (310, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 1)
            
            # Add current processing mode indicator
            cv2.putText(frame, f"Mode: {self.current_mode.upper()}", (10, h - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            
            # Add artificial horizon indicator
            self._draw_attitude_indicator(frame, flight_data['attitude'])
            
            return frame
            
        except Exception as e:
            print(f"Error adding flight data overlay: {e}")
            return frame
    
    def _draw_attitude_indicator(self, frame, attitude):
        """Draw a simple attitude indicator (artificial horizon)."""
        try:
            h, w = frame.shape[:2]
            center_x, center_y = w - 80, 140
            radius = 50
            
            # Background circle
            cv2.circle(frame, (center_x, center_y), radius, (0, 0, 0), -1)
            cv2.circle(frame, (center_x, center_y), radius, (255, 255, 255), 1)
            
            # Calculate roll and pitch offsets
            roll_rad = math.radians(attitude['roll'])
            pitch_offset = attitude['pitch'] * 0.5  # Scale pitch for better visualization
            
            # Draw horizon line
            horizon_length = radius - 10
            x1 = center_x - horizon_length * math.cos(roll_rad)
            y1 = center_y - horizon_length * math.sin(roll_rad) + pitch_offset
            x2 = center_x + horizon_length * math.cos(roll_rad)
            y2 = center_y + horizon_length * math.sin(roll_rad) + pitch_offset
            
            cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            
            # Draw center dot
            cv2.circle(frame, (center_x, center_y), 3, (0, 255, 0), -1)
            
            # Draw roll indicator
            roll_indicator_length = radius - 5
            roll_x = center_x + roll_indicator_length * math.sin(roll_rad)
            roll_y = center_y - roll_indicator_length * math.cos(roll_rad)
            cv2.line(frame, (center_x, center_y - radius + 5), 
                    (int(roll_x), int(roll_y)), (255, 0, 0), 2)
            
        except Exception as e:
            print(f"Error drawing attitude indicator: {e}")
        
        return frame