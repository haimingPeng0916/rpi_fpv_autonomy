# camera_system.py
import cv2
import numpy as np
import time
from picamera2 import Picamera2

class CameraSystem:
    def __init__(self, width=640, height=480):
        """Initialize the camera system."""
        self.width = width
        self.height = height
        self.camera = None
        self.running = False
        self.last_frame = None
    
    def start(self):
        """Start the camera using PiCamera2."""
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
            print("Camera stopped")
    
    def get_frame(self):
        """Get a frame from the camera."""
        if not self.running or self.camera is None:
            # Return a black frame if camera is not running
            frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            cv2.putText(frame, "Camera not running", (50, self.height//2), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            return False, frame
        
        try:
            # Capture frame using PiCamera2
            frame = self.camera.capture_array()
            self.last_frame = frame.copy()
            return True, frame
        except Exception as e:
            print(f"Error capturing frame: {e}")
            # Return last frame or black frame
            if self.last_frame is not None:
                return False, self.last_frame
            else:
                frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
                cv2.putText(frame, "Error capturing frame", (50, self.height//2), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                return False, frame