# app.py
from flask import Flask, Response, render_template, jsonify
import time
import threading
import os
import json
import cv2
import math
# from camera_system import Camera
from camera import Camera
from flight_controller import FlightController
from opencv_processor import OpenCVProcessor

# Create application
app = Flask(__name__)
camera = Camera()
flight_controller = FlightController()
cv_processor = OpenCVProcessor()

# Create directories if they don't exist
os.makedirs('static', exist_ok=True)
os.makedirs('templates', exist_ok=True)

# Routes
@app.route('/')
def index():
    """Render the main page."""
    return render_template('index.html')

def generate_frames():
    """Generate video frames with overlays."""
    while True:
        # Get raw frame from camera
        success, frame, _, _ = camera.get_frame()
        if not success:
            time.sleep(0.1)
            continue
        
        # Get flight data
        flight_data = flight_controller.get_telemetry()
        
        # Process frame with OpenCV
        frame = cv_processor.process_frame(frame, flight_data)
        
        # Convert to JPEG
        ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ret:
            continue
            
        # Yield frame in MJPEG format
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')



@app.route('/video_feed')
def video_feed():
    """Video streaming route."""
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/flight_data')
def get_flight_data():
    """Get current flight data."""
    return jsonify(flight_controller.get_telemetry())

@app.route('/api/visualization_mode/<mode>')
def set_visualization_mode(mode):
    """Set the current visualization mode."""
    success = cv_processor.set_mode(mode)
    return jsonify({"success": success, "mode": cv_processor.current_mode})

def main():
    # Start camera system
    print("Starting camera system...")
    camera.start()
    
    # Start flight controller update thread
    print("Starting flight controller update...")
    flight_controller.start_updates()
    
    # Run the app
    print("Starting web server...")
    try:
        app.run(host='0.0.0.0', port=5000, threaded=True, debug=False)
    except KeyboardInterrupt:
        print("Shutting down...")
    finally:
        camera.stop()
        flight_controller.stop_updates()

if __name__ == '__main__':
    main()