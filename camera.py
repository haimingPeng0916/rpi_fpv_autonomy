# camera.py - Pi AI Camera Configuration Fix
import cv2
import numpy as np
import time
import subprocess

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
        """Start the Pi AI camera with proper configuration."""
        if self.camera is not None:
            return
        
        try:
            from picamera2 import Picamera2
            
            self.camera = Picamera2()
            
            # Get camera info for debugging
            try:
                cameras = Picamera2.global_camera_info()
                print(f"Available cameras: {cameras}")
            except Exception as e:
                print(f"Could not list cameras: {e}")
            
            # Configure camera specifically for Pi AI camera
            # Use a configuration that works with the native resolution
            config = self.camera.create_preview_configuration(
                main={
                    "size": (self.width, self.height), 
                    "format": "RGB888"
                },
                # Add buffer configuration for better performance
                buffer_count=4
            )
            
            print(f"Camera configuration: {config}")
            
            # Apply configuration
            self.camera.configure(config)
            
            # Start camera with longer timeout
            print("Starting Pi AI camera...")
            self.camera.start()
            
            # Wait longer for camera to stabilize (Pi AI camera needs more time)
            print("Waiting for camera to stabilize...")
            time.sleep(3)
            
            # Test capture with multiple attempts
            test_successful = False
            for attempt in range(5):
                try:
                    print(f"Test capture attempt {attempt + 1}...")
                    test_frame = self.camera.capture_array()
                    
                    if test_frame is not None and test_frame.size > 0:
                        print(f"Test capture successful! Frame shape: {test_frame.shape}, dtype: {test_frame.dtype}")
                        test_successful = True
                        break
                    else:
                        print(f"Test capture attempt {attempt + 1} returned empty frame")
                        time.sleep(1)
                        
                except Exception as e:
                    print(f"Test capture attempt {attempt + 1} failed: {e}")
                    time.sleep(1)
            
            if test_successful:
                self.running = True
                print(f"Pi AI Camera started successfully - Final resolution: {test_frame.shape}")
            else:
                raise Exception("All test captures failed")
                
        except ImportError:
            print("Picamera2 not available")
            self.running = False
            
        except Exception as e:
            print(f"Error starting Pi AI camera: {e}")
            print("Trying alternative configuration...")
            
            # Try alternative configuration
            try:
                if self.camera:
                    self.camera.stop()
                
                self.camera = Picamera2()
                
                # Alternative config - use smaller resolution that might work better
                alt_config = self.camera.create_video_configuration(
                    main={"size": (640, 480), "format": "RGB888"},
                    buffer_count=4
                )
                
                print(f"Trying alternative configuration: {alt_config}")
                self.camera.configure(alt_config)
                self.camera.start()
                
                time.sleep(3)
                
                # Test this configuration
                test_frame = self.camera.capture_array()
                if test_frame is not None and test_frame.size > 0:
                    self.running = True
                    print(f"Alternative configuration successful! Frame shape: {test_frame.shape}")
                else:
                    raise Exception("Alternative configuration also failed")
                    
            except Exception as e2:
                print(f"Alternative configuration failed: {e2}")
                self.running = False
                if self.camera:
                    try:
                        self.camera.stop()
                    except:
                        pass
                    self.camera = None
    
    def stop(self):
        """Stop the camera."""
        if self.camera is not None:
            try:
                self.camera.stop()
                print("Pi AI Camera stopped")
            except Exception as e:
                print(f"Error stopping camera: {e}")
            finally:
                self.camera = None
                self.running = False
    
    def get_frame(self):
        """Get a frame from the camera and process it."""
        if not self.running or self.camera is None:
            # Return a diagnostic frame
            frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            cv2.putText(frame, "Pi AI Camera not running", (50, self.height//2 - 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(frame, "Check console for error details", (50, self.height//2), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            return False, frame, None, None
        
        try:
            # Capture frame from Pi AI camera
            frame = self.camera.capture_array()
            
            if frame is None or frame.size == 0:
                raise Exception("Camera returned empty frame")
            
            # Handle different color formats
            if len(frame.shape) == 3:
                if frame.shape[2] == 3:
                    # RGB format from Picamera2 - convert to BGR for OpenCV
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                elif frame.shape[2] == 4:
                    # RGBA format - convert to BGR
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
            elif len(frame.shape) == 2:
                # Grayscale - convert to BGR
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            
            # Ensure frame is correct size
            if frame.shape[:2] != (self.height, self.width):
                frame = cv2.resize(frame, (self.width, self.height))
            
            self.last_frame = frame.copy()
            
            # Process frame for box detection
            box_detected, processed_frame, center, bbox = self.detect_box(frame)
            
            # Return TRUE for camera success, regardless of whether box was detected
            # The first parameter indicates camera is working, not whether box was found
            return True, processed_frame, center, bbox
            
        except Exception as e:
            print(f"Error capturing frame from Pi AI camera: {e}")
            
            # Return last frame or error frame
            if self.last_frame is not None:
                return False, self.last_frame, None, None
            else:
                frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
                cv2.putText(frame, f"Capture Error", (50, self.height//2 - 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.putText(frame, f"{str(e)[:40]}", (50, self.height//2 + 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                return False, frame, None, None
    
    def detect_box(self, frame):
        """Detect a box in the given frame with enhanced visualization."""
        # Your existing detect_box implementation stays exactly the same
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
                    
                    # Enhanced visualization
                    cv2.drawContours(output_frame, [approx], 0, (0, 255, 0), 2)
                    cv2.rectangle(output_frame, (x, y), (x+w, y+h), (255, 0, 0), 3)
                    
                    # Corner markers
                    corner_size = 10
                    cv2.line(output_frame, (x, y), (x + corner_size, y), (0, 255, 255), 3)
                    cv2.line(output_frame, (x, y), (x, y + corner_size), (0, 255, 255), 3)
                    cv2.line(output_frame, (x+w, y), (x+w - corner_size, y), (0, 255, 255), 3)
                    cv2.line(output_frame, (x+w, y), (x+w, y + corner_size), (0, 255, 255), 3)
                    cv2.line(output_frame, (x, y+h), (x + corner_size, y+h), (0, 255, 255), 3)
                    cv2.line(output_frame, (x, y+h), (x, y+h - corner_size), (0, 255, 255), 3)
                    cv2.line(output_frame, (x+w, y+h), (x+w - corner_size, y+h), (0, 255, 255), 3)
                    cv2.line(output_frame, (x+w, y+h), (x+w, y+h - corner_size), (0, 255, 255), 3)
                    
                    # Crosshairs at center
                    cv2.line(output_frame, (center[0] - 10, center[1]), (center[0] + 10, center[1]), (0, 0, 255), 2)
                    cv2.line(output_frame, (center[0], center[1] - 10), (center[0], center[1] + 10), (0, 0, 255), 2)
                    
                    # Text info
                    cv2.putText(output_frame, f"Center: ({center[0]}, {center[1]})", 
                            (x, y-30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    cv2.putText(output_frame, f"Size: {w}x{h} px", 
                            (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    
                    # Detection status
                    cv2.rectangle(output_frame, (10, 10), (200, 50), (0, 0, 0), -1)
                    cv2.putText(output_frame, "BOX DETECTED", (20, 35),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                            
                    # Save detection info
                    self.last_detection = {
                        'center': center,
                        'bbox': bbox,
                        'size': (w, h),
                        'timestamp': time.time()
                    }
                
        # If no box was found
        if not found:
            cv2.rectangle(output_frame, (10, 10), (250, 50), (0, 0, 0), -1)
            cv2.putText(output_frame, "NO BOX DETECTED", (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        # Debug info
        cv2.putText(output_frame, f"Threshold: {self.threshold_value}", (10, 70), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Add Pi AI camera status indicator
        cv2.putText(output_frame, "Pi AI Camera Active", (10, output_frame.shape[0] - 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Overlay threshold image
        try:
            small_thresh = cv2.resize(thresh, (160, 120))
            small_thresh_colored = cv2.cvtColor(small_thresh, cv2.COLOR_GRAY2BGR)
            
            roi_y1, roi_y2 = 10, 10 + small_thresh_colored.shape[0]
            roi_x1 = output_frame.shape[1] - small_thresh_colored.shape[1] - 10
            roi_x2 = output_frame.shape[1] - 10
            
            roi = output_frame[roi_y1:roi_y2, roi_x1:roi_x2]
            
            if roi.shape == small_thresh_colored.shape:
                output_frame[roi_y1:roi_y2, roi_x1:roi_x2] = small_thresh_colored
        except Exception as e:
            pass  # Silently skip overlay if it fails
        
        return found, output_frame, center, bbox

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