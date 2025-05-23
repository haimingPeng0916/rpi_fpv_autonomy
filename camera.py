# camera.py
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
        
        # Detection parameters
        self.threshold_value = 127
        self.min_area = 1000
        self.aspect_ratio_min = 0.8
        self.aspect_ratio_max = 1.2
    
    def start(self):
        """Start the camera."""
        if self.camera is not None:
            return
        
        try:
            self.camera = Picamera2()
            config = self.camera.create_preview_configuration(main={"size": (self.width, self.height)})
            self.camera.configure(config)
            self.camera.start()
            self.running = True
            print("Camera started successfully")
        except Exception as e:
            print(f"Error starting camera: {e}")
            self.running = False
    
    def stop(self):
        """Stop the camera."""
        if self.camera is not None:
            self.camera.stop()
            self.camera = None
            self.running = False
    
    # def detect_box(self, frame):
    #     """Detect a box in the given frame."""
    #     # Make a copy for drawing
    #     output_frame = frame.copy()
        
    #     # Convert to grayscale
    #     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
    #     # Apply thresholding
    #     _, thresh = cv2.threshold(gray, self.threshold_value, 255, cv2.THRESH_BINARY)
        
    #     # Find contours
    #     contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
    #     # Result variables
    #     found = False
    #     center = None
    #     bbox = None
        
    #     # Find largest contour
    #     if contours:
    #         largest_contour = max(contours, key=cv2.contourArea)
            
    #         # Approximate the contour
    #         perimeter = cv2.arcLength(largest_contour, True)
    #         approx = cv2.approxPolyDP(largest_contour, 0.04 * perimeter, True)
            
    #         # If it has 4 points, it's likely our box
    #         if len(approx) == 4:
    #             # Get coordinates
    #             x, y, w, h = cv2.boundingRect(approx)
    #             area = cv2.contourArea(largest_contour)
    #             aspect_ratio = float(w) / h
                
    #             # If it meets our criteria
    #             if (self.aspect_ratio_min <= aspect_ratio <= self.aspect_ratio_max and 
    #                 area >= self.min_area):
                    
    #                 center = (x + w//2, y + h//2)
    #                 bbox = (x, y, w, h)
    #                 found = True
                    
    #                 # Draw on the output frame
    #                 cv2.drawContours(output_frame, [approx], 0, (0, 255, 0), 3)
    #                 cv2.circle(output_frame, center, 5, (0, 0, 255), -1)
    #                 cv2.putText(output_frame, f"Box: ({center[0]}, {center[1]})", 
    #                            (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    
    #                 # Save detection info
    #                 self.last_detection = {
    #                     'center': center,
    #                     'bbox': bbox,
    #                     'timestamp': time.time()
    #                 }
        
    #     # Add debug info to the frame
    #     cv2.putText(output_frame, f"Threshold: {self.threshold_value}", (10, 20), 
    #                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
    #     # Overlay threshold image in small corner for debugging
    #     small_thresh = cv2.resize(cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR), 
    #                              (160, 120))
    #     output_frame[10:10+120, output_frame.shape[1]-160-10:output_frame.shape[1]-10] = small_thresh
        
    #     return found, output_frame, center, bbox
    
    def detect_box(self, frame):
        """Detect a box in the given frame with enhanced visualization."""
        # Make a copy for drawing
        output_frame = frame.copy()
        
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply thresholding
        _, thresh = cv2.threshold(gray, self.threshold_value, 255, cv2.THRESH_BINARY)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Result variables
        found = False
        center = None
        bbox = None
        
        # Find largest contour
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Approximate the contour
            perimeter = cv2.arcLength(largest_contour, True)
            approx = cv2.approxPolyDP(largest_contour, 0.04 * perimeter, True)
            
            # If it has 4 points, it's likely our box
            if len(approx) == 4:
                # Get coordinates
                x, y, w, h = cv2.boundingRect(approx)
                area = cv2.contourArea(largest_contour)
                aspect_ratio = float(w) / h
                
                # If it meets our criteria
                if (self.aspect_ratio_min <= aspect_ratio <= self.aspect_ratio_max and 
                    area >= self.min_area):
                    
                    center = (x + w//2, y + h//2)
                    bbox = (x, y, w, h)
                    found = True
                    
                    # --- Enhanced visualization starts here ---
                    
                    # 1. Draw the exact contour in green (thin line)
                    cv2.drawContours(output_frame, [approx], 0, (0, 255, 0), 2)
                    
                    # 2. Draw a clear bounding rectangle in blue (thicker line)
                    cv2.rectangle(output_frame, (x, y), (x+w, y+h), (255, 0, 0), 3)
                    
                    # 3. Draw corner markers at each corner of the bounding box
                    corner_size = 10
                    # Top-left
                    cv2.line(output_frame, (x, y), (x + corner_size, y), (0, 255, 255), 3)
                    cv2.line(output_frame, (x, y), (x, y + corner_size), (0, 255, 255), 3)
                    # Top-right
                    cv2.line(output_frame, (x+w, y), (x+w - corner_size, y), (0, 255, 255), 3)
                    cv2.line(output_frame, (x+w, y), (x+w, y + corner_size), (0, 255, 255), 3)
                    # Bottom-left
                    cv2.line(output_frame, (x, y+h), (x + corner_size, y+h), (0, 255, 255), 3)
                    cv2.line(output_frame, (x, y+h), (x, y+h - corner_size), (0, 255, 255), 3)
                    # Bottom-right
                    cv2.line(output_frame, (x+w, y+h), (x+w - corner_size, y+h), (0, 255, 255), 3)
                    cv2.line(output_frame, (x+w, y+h), (x+w, y+h - corner_size), (0, 255, 255), 3)
                    
                    # 4. Draw crosshairs at the center
                    cv2.line(output_frame, (center[0] - 10, center[1]), (center[0] + 10, center[1]), (0, 0, 255), 2)
                    cv2.line(output_frame, (center[0], center[1] - 10), (center[0], center[1] + 10), (0, 0, 255), 2)
                    
                    # 5. Add more informative text
                    # Position text
                    cv2.putText(output_frame, f"Center: ({center[0]}, {center[1]})", 
                            (x, y-30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    # Size text
                    cv2.putText(output_frame, f"Size: {w}x{h} px", 
                            (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    
                    # 6. Add detection status indicator
                    cv2.rectangle(output_frame, (10, 10), (200, 50), (0, 0, 0), -1)  # Black background
                    cv2.putText(output_frame, "BOX DETECTED", (20, 35),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                            
                    # Save detection info
                    self.last_detection = {
                        'center': center,
                        'bbox': bbox,
                        'size': (w, h),
                        'timestamp': time.time()
                    }
                
        # If no box was found, add a "no detection" indicator
        if not found:
            cv2.rectangle(output_frame, (10, 10), (250, 50), (0, 0, 0), -1)  # Black background
            cv2.putText(output_frame, "NO BOX DETECTED", (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        # Add debug info to the frame
        cv2.putText(output_frame, f"Threshold: {self.threshold_value}", (10, 70), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # --- Fix for the channel mismatch issue starts here ---
        try:
            # Get output frame dimensions and channels
            output_channels = output_frame.shape[2] if len(output_frame.shape) > 2 else 1
            
            # Create an appropriately sized and formatted small threshold image
            small_thresh = cv2.resize(thresh, (160, 120))
            
            # Convert to match output frame channel count
            if output_channels == 3:
                small_thresh_colored = cv2.cvtColor(small_thresh, cv2.COLOR_GRAY2BGR)
            elif output_channels == 4:
                small_thresh_colored = cv2.cvtColor(small_thresh, cv2.COLOR_GRAY2BGRA)
            else:
                # Unexpected channel count, skip overlay
                raise ValueError(f"Unexpected channel count: {output_channels}")
            
            # Calculate ROI coordinates
            roi_y1, roi_y2 = 10, 10 + small_thresh_colored.shape[0]
            roi_x1 = output_frame.shape[1] - small_thresh_colored.shape[1] - 10
            roi_x2 = output_frame.shape[1] - 10
            
            # Create the ROI
            roi = output_frame[roi_y1:roi_y2, roi_x1:roi_x2]
            
            # Verify ROI and overlay have the same shape
            if roi.shape == small_thresh_colored.shape:
                output_frame[roi_y1:roi_y2, roi_x1:roi_x2] = small_thresh_colored
        except Exception as e:
            # If overlay fails, don't crash the whole detection
            print(f"Warning: Could not overlay threshold image: {e}")
        
        return found, output_frame, center, bbox

    def get_frame(self):
        """Get a frame from the camera and process it."""
        if not self.running or self.camera is None:
            # Return a black frame if camera is not running
            frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            cv2.putText(frame, "Camera not running", (50, self.height//2), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            return False, frame, None, None
        
        try:
            # Capture frame
            frame = self.camera.capture_array()
            self.last_frame = frame.copy()
            
            # Process frame
            return self.detect_box(frame)
        except Exception as e:
            print(f"Error capturing frame: {e}")
            # Return last frame or black frame
            if self.last_frame is not None:
                return False, self.last_frame, None, None
            else:
                frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
                cv2.putText(frame, "Error capturing frame", (50, self.height//2), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                return False, frame, None, None
    
    def update_settings(self, settings):
        """Update detection settings."""
        if 'threshold_value' in settings:
            self.threshold_value = settings['threshold_value']
        if 'min_area' in settings:
            self.min_area = settings['min_area']
        if 'aspect_ratio_min' in settings:
            self.aspect_ratio_min = settings['aspect_ratio_min']
        if 'aspect_ratio_max' in settings:
            self.aspect_ratio_max = settings['aspect_ratio_max']