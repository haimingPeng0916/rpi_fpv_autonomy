# app.py - Production-ready version with real flight data
from flask import Flask, Response, render_template, jsonify, request
from flask_socketio import SocketIO, emit
import time
import threading
import os
import json
import cv2
import numpy as np
from datetime import datetime
import logging
from queue import Queue, Empty
from camera import Camera
from flight_controller import FlightController
from opencv_processor import OpenCVProcessor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create Flask application with SocketIO for real-time communication
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# Initialize components
camera = Camera()
cv_processor = OpenCVProcessor()

# Flight controller - set simulation=False for real hardware
# Change port to match your setup (e.g., /dev/ttyUSB0, /dev/ttyACM0)
flight_controller = FlightController(
    simulation=False,  # Set to False for real flight controller
    port='/dev/ttyAMA10',  # Adjust to your FC's serial port
    baudrate=115200  # Standard MSP baudrate
)

# Frame buffer for low-latency streaming
frame_buffer = Queue(maxsize=2)  # Small buffer to reduce latency
telemetry_buffer = Queue(maxsize=10)

# Performance monitoring
performance_stats = {
    'frame_rate': 0,
    'telemetry_rate': 0,
    'frame_latency': 0,
    'telemetry_latency': 0,
    'dropped_frames': 0
}

# Create directories if they don't exist
os.makedirs('static', exist_ok=True)
os.makedirs('templates', exist_ok=True)

class VideoStreamProducer(threading.Thread):
    """Dedicated thread for video frame production."""
    
    def __init__(self):
        super().__init__(daemon=True)
        self.running = False
        self.frame_count = 0
        self.last_fps_time = time.time()
        
    def run(self):
        """Main video production loop."""
        self.running = True
        logger.info("Video stream producer started")
        
        while self.running:
            try:
                # Get frame from camera
                success, frame, center, bbox = camera.get_frame()
                
                if success and frame is not None:
                    # Get latest telemetry (non-blocking)
                    telemetry = flight_controller.get_telemetry()
                    
                    # Process frame with OpenCV
                    processed_frame = cv_processor.process_frame(frame, telemetry)
                    
                    # Add performance overlay
                    self._add_performance_overlay(processed_frame)
                    
                    # Encode frame
                    encode_start = time.time()
                    ret, buffer = cv2.imencode('.jpg', processed_frame, 
                                             [cv2.IMWRITE_JPEG_QUALITY, 80])  # Lower quality for speed
                    encode_time = time.time() - encode_start
                    
                    if ret:
                        # Try to add to buffer (non-blocking)
                        try:
                            frame_buffer.put_nowait({
                                'buffer': buffer,
                                'timestamp': time.time(),
                                'encode_time': encode_time
                            })
                        except:
                            # Buffer full, drop oldest frame
                            try:
                                frame_buffer.get_nowait()
                                frame_buffer.put_nowait({
                                    'buffer': buffer,
                                    'timestamp': time.time(),
                                    'encode_time': encode_time
                                })
                                performance_stats['dropped_frames'] += 1
                            except:
                                pass
                    
                    # Update FPS
                    self.frame_count += 1
                    current_time = time.time()
                    if current_time - self.last_fps_time >= 1.0:
                        performance_stats['frame_rate'] = self.frame_count / (current_time - self.last_fps_time)
                        self.frame_count = 0
                        self.last_fps_time = current_time
                else:
                    # Camera error - wait a bit
                    time.sleep(0.1)
                    
            except Exception as e:
                logger.error(f"Video producer error: {e}")
                time.sleep(0.1)
    
    def _add_performance_overlay(self, frame):
        """Add performance statistics overlay."""
        h, w = frame.shape[:2]
        
        # Performance stats in bottom right
        stats_text = [
            f"FPS: {performance_stats['frame_rate']:.1f}",
            f"Telemetry: {performance_stats['telemetry_rate']:.1f}Hz",
            f"Dropped: {performance_stats['dropped_frames']}"
        ]
        
        y_offset = h - 80
        for text in stats_text:
            cv2.putText(frame, text, (w - 200, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            y_offset += 20
    
    def stop(self):
        """Stop the video producer."""
        self.running = False

class TelemetryBroadcaster(threading.Thread):
    """Dedicated thread for telemetry broadcasting via WebSocket."""
    
    def __init__(self):
        super().__init__(daemon=True)
        self.running = False
        self.update_count = 0
        self.last_rate_time = time.time()
        
    def run(self):
        """Main telemetry broadcast loop."""
        self.running = True
        logger.info("Telemetry broadcaster started")
        
        while self.running:
            try:
                # Get telemetry
                telemetry = flight_controller.get_telemetry()
                
                # Add freshness information
                freshness = flight_controller.get_telemetry_freshness()
                telemetry['data_age'] = freshness
                
                # Add performance stats
                telemetry['performance'] = performance_stats.copy()
                
                # Broadcast via WebSocket
                socketio.emit('telemetry_update', telemetry, namespace='/')
                
                # Update rate counter
                self.update_count += 1
                current_time = time.time()
                if current_time - self.last_rate_time >= 1.0:
                    performance_stats['telemetry_rate'] = self.update_count / (current_time - self.last_rate_time)
                    self.update_count = 0
                    self.last_rate_time = current_time
                
                # Target 50Hz update rate
                time.sleep(0.02)
                
            except Exception as e:
                logger.error(f"Telemetry broadcaster error: {e}")
                time.sleep(0.1)
    
    def stop(self):
        """Stop the telemetry broadcaster."""
        self.running = False

# Routes
@app.route('/')
def index():
    """Render the main page."""
    return render_template('index.html')

def generate_frames():
    """Generate video frames for streaming."""
    logger.info("Starting frame generation")
    
    while True:
        try:
            # Get frame from buffer (blocking with timeout)
            frame_data = frame_buffer.get(timeout=0.5)
            
            # Calculate latency
            latency = time.time() - frame_data['timestamp']
            performance_stats['frame_latency'] = latency
            
            # Yield frame
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + 
                   frame_data['buffer'].tobytes() + b'\r\n')
                   
        except Empty:
            # No frame available - send placeholder
            logger.warning("Frame buffer empty")
            placeholder = create_placeholder_frame("No video signal")
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + placeholder + b'\r\n')
        except Exception as e:
            logger.error(f"Frame generation error: {e}")
            time.sleep(0.1)

@app.route('/video_feed')
def video_feed():
    """Video streaming route."""
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/flight_data')
def get_flight_data():
    """Get current flight data (REST endpoint for compatibility)."""
    telemetry = flight_controller.get_telemetry()
    telemetry['freshness'] = flight_controller.get_telemetry_freshness()
    return jsonify(telemetry)

@app.route('/api/visualization_mode/<mode>')
def set_visualization_mode(mode):
    """Set the current visualization mode."""
    success = cv_processor.set_mode(mode)
    return jsonify({"success": success, "mode": cv_processor.current_mode})

@app.route('/api/detection_mode/<mode>')
def set_detection_mode(mode):
    """Set camera detection mode."""
    success = camera.set_detection_mode(mode)
    return jsonify({"success": success, "mode": camera.detection_mode})

@app.route('/api/system_status')
def get_system_status():
    """Get comprehensive system status."""
    status = {
        "camera": {
            "running": camera.running,
            "detection_mode": camera.detection_mode,
            "last_detection": camera.last_detection
        },
        "flight_controller": {
            "connected": not flight_controller.simulation,
            "simulation": flight_controller.simulation,
            "statistics": flight_controller.get_statistics()
        },
        "performance": performance_stats,
        "opencv_mode": cv_processor.current_mode,
        "timestamp": time.time()
    }
    return jsonify(status)

@app.route('/api/send_command', methods=['POST'])
def send_command():
    """Send command to flight controller."""
    try:
        data = request.json
        command = data.get('command')
        params = data.get('params', {})
        
        if command == 'arm':
            success = flight_controller.arm_motors()
        elif command == 'disarm':
            success = flight_controller.disarm_motors()
        elif command == 'set_rc':
            channels = params.get('channels', [])
            success = flight_controller.send_rc_channels(channels)
        else:
            return jsonify({"success": False, "error": "Unknown command"})
            
        return jsonify({"success": success})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

# WebSocket event handlers
@socketio.on('connect')
def handle_connect():
    """Handle client connection."""
    logger.info(f"Client connected: {request.sid}")
    emit('connected', {'status': 'Connected to FPV Control Station'})

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection."""
    logger.info(f"Client disconnected: {request.sid}")

@socketio.on('control_command')
def handle_control_command(data):
    """Handle control commands via WebSocket."""
    command = data.get('command')
    params = data.get('params', {})
    
    try:
        if command == 'set_mode':
            mode = params.get('mode')
            if params.get('type') == 'visualization':
                success = cv_processor.set_mode(mode)
            elif params.get('type') == 'detection':
                success = camera.set_detection_mode(mode)
            else:
                success = False
        else:
            # Forward to flight controller
            success = flight_controller.send_command(command, params)
            
        emit('command_response', {
            'command': command,
            'success': success,
            'timestamp': time.time()
        })
    except Exception as e:
        logger.error(f"Control command error: {e}")
        emit('command_response', {
            'command': command,
            'success': False,
            'error': str(e)
        })

def create_placeholder_frame(text):
    """Create a placeholder frame when video is unavailable."""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(frame, text, (150, 240), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    _, buffer = cv2.imencode('.jpg', frame)
    return buffer.tobytes()

# Global components
video_producer = None
telemetry_broadcaster = None

def initialize_system():
    """Initialize all system components."""
    global video_producer, telemetry_broadcaster
    
    logger.info("Initializing FPV Control Station...")
    
    # Connect to flight controller
    if not flight_controller.simulation:
        logger.info("Attempting to connect to flight controller...")
        if not flight_controller.connect_to_fc():
            logger.warning("Failed to connect to flight controller, falling back to simulation")
            flight_controller.simulation = True
    
    # Start flight controller updates
    flight_controller.start_updates()
    
    # Start camera
    logger.info("Starting camera system...")
    if not camera.start():
        logger.error("Failed to start camera")
        return False
    
    # Wait for camera stabilization
    time.sleep(2)
    
    # Start video producer
    video_producer = VideoStreamProducer()
    video_producer.start()
    
    # Start telemetry broadcaster
    telemetry_broadcaster = TelemetryBroadcaster()
    telemetry_broadcaster.start()
    
    logger.info("System initialization complete")
    return True

def shutdown_system():
    """Shutdown all system components."""
    global video_producer, telemetry_broadcaster
    
    logger.info("Shutting down FPV Control Station...")
    
    # Stop threads
    if video_producer:
        video_producer.stop()
        video_producer.join(timeout=2)
        
    if telemetry_broadcaster:
        telemetry_broadcaster.stop()
        telemetry_broadcaster.join(timeout=2)
    
    # Stop components
    camera.stop()
    flight_controller.stop_updates()
    flight_controller.disconnect()
    
    logger.info("Shutdown complete")

def main():
    """Main entry point."""
    try:
        # Initialize system
        if not initialize_system():
            logger.error("System initialization failed")
            return
        
        # Print startup information
        print("\n" + "="*50)
        print("FPV Control Station - READY")
        print("="*50)
        print(f"Web Interface: http://localhost:5000")
        print(f"Flight Controller: {'CONNECTED' if not flight_controller.simulation else 'SIMULATION MODE'}")
        print(f"Camera: {'ACTIVE' if camera.running else 'INACTIVE'}")
        print("="*50 + "\n")
        
        # Run the application with SocketIO
        socketio.run(app, host='0.0.0.0', port=5000, debug=False)
        
    except KeyboardInterrupt:
        print("\n\nReceived interrupt signal...")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        shutdown_system()

if __name__ == '__main__':
    main()