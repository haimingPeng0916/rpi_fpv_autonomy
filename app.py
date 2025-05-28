# app.py - Debug Version
from flask import Flask, Response, render_template, jsonify
import time
import threading
import os
import json
import cv2
import math
import numpy as np
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
    """Generate video frames with overlays - DEBUG VERSION."""
    frame_count = 0
    print("=== GENERATE_FRAMES STARTED ===")
    
    while True:
        frame_count += 1
        print(f"\n--- Frame {frame_count} ---")
        
        try:
            # Get raw frame from camera
            print(f"Frame {frame_count}: Calling camera.get_frame()...")
            success, frame, center, bbox = camera.get_frame()
            print(f"Frame {frame_count}: camera.get_frame() returned:")
            print(f"  - success: {success}")
            print(f"  - frame shape: {frame.shape if frame is not None else 'None'}")
            print(f"  - frame dtype: {frame.dtype if frame is not None else 'None'}")
            print(f"  - center: {center}")
            print(f"  - bbox: {bbox}")
            
            if not success:
                print(f"Frame {frame_count}: Camera failed, using error frame")
                # Create a error frame with useful info
                if frame is None:
                    frame = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(frame, f"Camera Error - Frame {frame_count}", (50, 240), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.putText(frame, f"Check console output", (50, 280), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            else:
                print(f"Frame {frame_count}: Camera success! Frame shape: {frame.shape}")
            
            # Get flight data
            print(f"Frame {frame_count}: Getting flight data...")
            flight_data = flight_controller.get_telemetry()
            print(f"Frame {frame_count}: Got flight data: battery={flight_data.get('battery', 'N/A')}V")
            
            # Process frame with OpenCV
            print(f"Frame {frame_count}: Processing with OpenCV...")
            processed_frame = cv_processor.process_frame(frame, flight_data)
            print(f"Frame {frame_count}: OpenCV processing complete, frame shape: {processed_frame.shape}")
            
            # Convert to JPEG
            print(f"Frame {frame_count}: Encoding to JPEG...")
            ret, buffer = cv2.imencode('.jpg', processed_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if not ret:
                print(f"Frame {frame_count}: JPEG encoding FAILED!")
                continue
            else:
                print(f"Frame {frame_count}: JPEG encoding success, buffer size: {len(buffer)}")
                
            print(f"Frame {frame_count}: Yielding frame to browser...")
            # Yield frame in MJPEG format
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            
            print(f"Frame {frame_count}: Frame successfully sent to browser")
            
            # Add small delay to prevent flooding console
            time.sleep(0.033)  # ~30 FPS
            
        except Exception as e:
            print(f"Frame {frame_count}: EXCEPTION occurred: {e}")
            import traceback
            traceback.print_exc()
            
            # Create error frame
            error_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(error_frame, f"Exception in Frame {frame_count}", (50, 240), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(error_frame, f"{str(e)[:40]}", (50, 280), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            
            # Try to send error frame
            try:
                ret, buffer = cv2.imencode('.jpg', error_frame)
                if ret:
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            except:
                pass
                
            time.sleep(0.5)  # Wait longer on error

@app.route('/video_feed')
def video_feed():
    """Video streaming route."""
    print("=== VIDEO_FEED ROUTE CALLED ===")
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

@app.route('/debug/camera_status')
def camera_status():
    """Debug route to check camera status."""
    status = {
        "camera_running": camera.running,
        "camera_object": str(type(camera.camera)),
        "last_frame_shape": camera.last_frame.shape if camera.last_frame is not None else None,
        "last_detection": camera.last_detection
    }
    return jsonify(status)

@app.route('/debug/test_frame')
def test_single_frame():
    """Debug route to test a single frame capture."""
    try:
        success, frame, center, bbox = camera.get_frame()
        if success and frame is not None:
            # Save frame for debugging
            cv2.imwrite('debug_frame.jpg', frame)
            return jsonify({
                "success": True,
                "frame_shape": frame.shape,
                "frame_dtype": str(frame.dtype),
                "center": center,
                "bbox": bbox,
                "message": "Frame saved as debug_frame.jpg"
            })
        else:
            return jsonify({
                "success": False,
                "frame_shape": frame.shape if frame is not None else None,
                "message": "Frame capture failed"
            })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e),
            "message": "Exception during frame capture"
        })

def main():
    # Start camera system
    print("Starting camera system...")
    camera.start()
    
    # Give camera more time to initialize
    print("Waiting for camera to fully initialize...")
    time.sleep(5)
    
    # Test one frame capture before starting web server
    print("Testing frame capture...")
    try:
        success, test_frame, center, bbox = camera.get_frame()
        print(f"Test capture: success={success}")
        if success and test_frame is not None:
            print(f"Frame shape: {test_frame.shape}")
            print(f"Frame dtype: {test_frame.dtype}")
            print(f"Frame min/max values: {test_frame.min()}/{test_frame.max()}")
            # Save test frame
            cv2.imwrite('test_frame_startup.jpg', test_frame)
            print("Test frame saved as test_frame_startup.jpg")
        else:
            print("Warning: Test frame capture failed")
            print(f"Frame is None: {test_frame is None}")
            if test_frame is not None:
                print(f"Frame shape: {test_frame.shape}")
    except Exception as e:
        print(f"Test capture error: {e}")
        import traceback
        traceback.print_exc()
    
    # Start flight controller update thread
    print("Starting flight controller...")
    flight_controller.start_updates()
    
    # Run the app
    print("Starting web server...")
    print("Debug routes available:")
    print("  http://localhost:5000/debug/camera_status")
    print("  http://localhost:5000/debug/test_frame")
    try:
        app.run(host='0.0.0.0', port=5000, threaded=True, debug=False)
    except KeyboardInterrupt:
        print("Shutting down...")
    finally:
        camera.stop()
        flight_controller.stop_updates()

if __name__ == '__main__':
    main()