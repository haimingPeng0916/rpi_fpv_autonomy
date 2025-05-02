import asyncio
import websockets
import json
import time
import threading
import socket
import os
import logging
import signal
import sys
from http.server import SimpleHTTPRequestHandler, HTTPServer
import math
from threading import Thread, Event

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
STREAM_PORT = 8000  # MJPEG streaming port
HTTP_PORT = 8080    # Web interface port
WS_PORT = 8765      # WebSocket port for flight data
stop_event = Event()

# Connected WebSocket clients
connected_clients = set()

class FlightData:
    """Simulates or reads flight controller data"""
    def __init__(self):
        self.roll = 0
        self.pitch = 0
        self.yaw = 0
        self.altitude = 0
        self.battery = 75
        
    def update(self):
        """Update flight data (simulation for now)"""
        # Simulate dynamic data
        self.roll = 45 * math.sin(time.time() * 0.5)
        self.pitch = 30 * math.sin(time.time() * 0.3)
        self.yaw = (time.time() * 10) % 360
        self.altitude = 10 + 2 * math.sin(time.time() / 5)
        self.battery = max(0, self.battery - 0.01)
        
    def to_json(self):
        """Convert to JSON format"""
        return {
            "type": "sensor_data",
            "roll": round(self.roll, 1),
            "pitch": round(self.pitch, 1),
            "yaw": round(self.yaw, 1),
            "altitude": round(self.altitude, 1),
            "battery": round(self.battery, 1)
        }

class CameraStreamer:
    """Handles camera capture and MJPEG streaming"""
    def __init__(self, port=8000):
        self.port = port
        self.server_socket = None
        self.connections = []
        self.camera = None
        
    def start(self):
        """Initialize camera and start streaming server"""
        # Try to initialize camera
        try:
            # First try to use PiCamera if available
            try:
                from picamera2 import Picamera2
                self.camera = Picamera2()
                # Configure with a reasonable resolution
                self.camera.configure(self.camera.create_preview_configuration(
                    main={"size": (640, 480), "format": "RGB888"}
                ))
                self.camera.start()
                logger.info("Using Raspberry Pi Camera module")
            except (ImportError, Exception) as e:
                logger.info(f"PiCamera not available: {e}, trying OpenCV")
                # Fallback to OpenCV
                import cv2
                # Try different camera devices
                for i in range(3):  # Try cameras 0, 1, 2
                    try:
                        self.camera = cv2.VideoCapture(i)
                        if self.camera.isOpened():
                            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                            logger.info(f"Using OpenCV with camera at index {i}")
                            break
                    except Exception as e:
                        logger.warning(f"Failed to open camera at index {i}: {e}")
                
                if not self.camera or not getattr(self.camera, 'isOpened', lambda: False)():
                    raise Exception("No camera available")
        except Exception as e:
            logger.error(f"Failed to initialize any camera: {e}")
            return False
                    
        logger.info("Camera initialized successfully")
        
        # Start the server in a separate thread
        server_thread = Thread(target=self._run_server)
        server_thread.daemon = True
        server_thread.start()
        
        logger.info(f"MJPEG stream server started on port {self.port}")
        return True
        
    def _run_server(self):
        """Run the streaming server"""
        try:
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.server_socket.bind(('0.0.0.0', self.port))
            self.server_socket.listen(5)
            
            while not stop_event.is_set():
                try:
                    conn, addr = self.server_socket.accept()
                    logger.info(f"New stream connection from {addr}")
                    
                    # Handle each connection in a separate thread
                    client_thread = Thread(target=self._handle_client, args=(conn,))
                    client_thread.daemon = True
                    client_thread.start()
                    
                    self.connections.append((conn, client_thread))
                    
                except Exception as e:
                    logger.error(f"Error accepting connection: {e}")
                    
        except Exception as e:
            logger.error(f"Server socket error: {e}")
        finally:
            self._cleanup()
    
    def _handle_client(self, conn):
        """Handle a single client connection"""
        try:
            # Send HTTP response header
            response = (
                b'HTTP/1.0 200 OK\r\n'
                b'Server: Camera MJPEG Server\r\n'
                b'Content-Type: multipart/x-mixed-replace; boundary=FRAME\r\n'
                b'Cache-Control: no-cache\r\n'
                b'Connection: close\r\n\r\n'
            )
            conn.sendall(response)
            
            # Check what type of camera we're using
            is_picamera = hasattr(self.camera, 'capture_array')
            
            # Stream frames
            while not stop_event.is_set():
                # Capture a frame
                if is_picamera:
                    frame = self.camera.capture_array()
                    import cv2
                    ret, jpeg_buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                    if not ret:
                        logger.warning("Failed to encode frame to JPEG")
                        continue
                    jpeg_bytes = jpeg_buffer.tobytes()
                else:
                    ret, frame = self.camera.read()
                    if not ret:
                        logger.warning("Failed to read frame from camera")
                        time.sleep(0.1)
                        continue
                    
                    # Convert to JPEG
                    import cv2
                    ret, jpeg_buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                    if not ret:
                        logger.warning("Failed to encode frame to JPEG")
                        continue
                    
                    # Get the byte array
                    jpeg_bytes = jpeg_buffer.tobytes()
                
                # Send the frame with MJPEG header
                try:
                    conn.sendall(
                        b'--FRAME\r\n'
                        b'Content-Type: image/jpeg\r\n'
                        b'Content-Length: ' + str(len(jpeg_bytes)).encode() + b'\r\n\r\n'
                    )
                    conn.sendall(jpeg_bytes)
                    conn.sendall(b'\r\n')
                except:
                    # Client likely disconnected
                    break
                
                # Limit frame rate
                time.sleep(0.05)  # ~20fps
                
        except Exception as e:
            logger.error(f"Client connection error: {e}")
        finally:
            try:
                conn.close()
            except:
                pass
    
    def _cleanup(self):
        """Clean up resources"""
        # Close all connections
        for conn, _ in self.connections:
            try:
                conn.close()
            except:
                pass
                
        # Close server socket
        if self.server_socket:
            try:
                self.server_socket.close()
            except:
                pass
                
        # Release camera
        if self.camera:
            if hasattr(self.camera, 'stop'):
                self.camera.stop()
            elif hasattr(self.camera, 'release'):
                self.camera.release()
            
        logger.info("MJPEG stream server shut down")

# Web server for hosting the dashboard
class DashboardHandler(SimpleHTTPRequestHandler):
    """Serves the web dashboard files"""
    def log_message(self, format, *args):
        logger.info(format % args)
    
    def end_headers(self):
        # Ensure proper content type for HTML files
        if self.path.endswith('.html'):
            self.send_header('Content-Type', 'text/html')
        elif self.path.endswith('.js'):
            self.send_header('Content-Type', 'application/javascript')
        elif self.path.endswith('.css'):
            self.send_header('Content-Type', 'text/css')
        super().end_headers()

def start_http_server(port=8080):
    """Start HTTP server to serve the dashboard"""
    server = HTTPServer(('0.0.0.0', port), DashboardHandler)
    logger.info(f"HTTP server started on port {port}")
    
    # Create HTML file if it doesn't exist
    os.makedirs("fpv/web", exist_ok=True)
    
    if not os.path.exists("fpv/web/index.html"):
        with open("fpv/web/index.html", "w") as f:
            f.write("""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>FPV Dashboard</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            background-color: #222;
            color: #fff;
            margin: 0;
            padding: 20px;
        }
        .container {
            display: flex;
            flex-wrap: wrap;
            gap: 20px;
        }
        .panel {
            background-color: #333;
            border-radius: 8px;
            padding: 15px;
            margin-bottom: 20px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.3);
        }
        h1, h2 {
            color: #ddd;
        }
        .video-panel {
            flex: 1 1 60%;
            min-width: 300px;
        }
        .data-panel {
            flex: 1 1 30%;
            min-width: 250px;
        }
        .data-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 10px;
        }
        .metric {
            background-color: #444;
            padding: 10px;
            border-radius: 5px;
        }
        .metric-label {
            font-size: 0.9em;
            color: #aaa;
            margin-bottom: 5px;
        }
        .metric-value {
            font-size: 1.8em;
            font-weight: bold;
        }
        #videoFeed {
            width: 100%;
            border-radius: 5px;
            background-color: #000;
        }
    </style>
</head>
<body>
    <h1>FPV Dashboard</h1>
    
    <div class="container">
        <div class="panel video-panel">
            <h2>Camera Feed</h2>
            <img id="videoFeed" src="http://window.location.hostname:8000/" alt="Camera Feed">
        </div>
        
        <div class="panel data-panel">
            <h2>Flight Data</h2>
            <div class="data-grid">
                <div class="metric">
                    <div class="metric-label">Roll</div>
                    <div class="metric-value" id="roll">0.0°</div>
                </div>
                <div class="metric">
                    <div class="metric-label">Pitch</div>
                    <div class="metric-value" id="pitch">0.0°</div>
                </div>
                <div class="metric">
                    <div class="metric-label">Yaw</div>
                    <div class="metric-value" id="yaw">0.0°</div>
                </div>
                <div class="metric">
                    <div class="metric-label">Altitude</div>
                    <div class="metric-value" id="altitude">0.0 m</div>
                </div>
                <div class="metric">
                    <div class="metric-label">Battery</div>
                    <div class="metric-value" id="battery">0.0%</div>
                </div>
            </div>
        </div>
    </div>

    <script>
        // Fix the video feed URL to use the current hostname
        document.getElementById('videoFeed').src = `http://${window.location.hostname}:8000/`;
        
        // WebSocket for flight data
        const ws = new WebSocket(`ws://${window.location.hostname}:8765`);
        
        ws.onopen = function() {
            console.log('Connected to the server');
        };
        
        ws.onmessage = function(event) {
            const data = JSON.parse(event.data);
            
            if (data.type === 'sensor_data') {
                // Update flight data
                document.getElementById('roll').textContent = `${data.roll.toFixed(1)}°`;
                document.getElementById('pitch').textContent = `${data.pitch.toFixed(1)}°`;
                document.getElementById('yaw').textContent = `${data.yaw.toFixed(1)}°`;
                document.getElementById('altitude').textContent = `${data.altitude.toFixed(1)} m`;
                document.getElementById('battery').textContent = `${data.battery.toFixed(1)}%`;
            }
        };
        
        ws.onclose = function() {
            console.log('Disconnected from the server');
            // Try to reconnect after 5 seconds
            setTimeout(function() {
                location.reload();
            }, 5000);
        };
    </script>
</body>
</html>""")
        logger.info("Created default index.html file")
    
    # Run in a thread
    thread = threading.Thread(target=server.serve_forever)
    thread.daemon = True
    thread.start()
    
    return server

async def websocket_handler(websocket, path):
    """Handle WebSocket connections for flight data"""
    # Register client
    connected_clients.add(websocket)
    logger.info(f"Client connected. Total clients: {len(connected_clients)}")
    
    try:
        while not stop_event.is_set():
            # Keep the connection alive, client will only receive data
            try:
                await asyncio.wait_for(websocket.recv(), timeout=1.0)
            except asyncio.TimeoutError:
                # This is expected - we use timeout to check stop_event periodically
                pass
            except websockets.exceptions.ConnectionClosed:
                break
    finally:
        # Unregister client
        connected_clients.remove(websocket)
        logger.info(f"Client disconnected. Total clients: {len(connected_clients)}")

async def flight_data_broadcast():
    """Broadcast flight data to all connected clients"""
    flight_data = FlightData()
    
    while not stop_event.is_set():
        # Update flight data
        flight_data.update()
        
        # Broadcast to all connected clients
        if connected_clients:
            message = json.dumps(flight_data.to_json())
            await asyncio.gather(
                *[client.send(message) for client in connected_clients],
                return_exceptions=True
            )
        
        # Update at 10Hz
        await asyncio.sleep(0.1)

async def start_websocket_server():
    """Start the WebSocket server"""
    async with websockets.serve(websocket_handler, "0.0.0.0", WS_PORT):
        logger.info(f"WebSocket server started on port {WS_PORT}")
        
        # Start flight data broadcast
        broadcast_task = asyncio.create_task(flight_data_broadcast())
        
        # Run until stopped
        while not stop_event.is_set():
            await asyncio.sleep(1)
            
        # Cancel broadcast task
        broadcast_task.cancel()

def cleanup():
    """Clean up resources when stopping"""
    logger.info("Shutting down...")
    
    # Set stop event
    stop_event.set()
    
    # Give threads time to clean up
    time.sleep(1)
    
    sys.exit(0)

def main():
    """Main function"""
    # Print startup message
    logger.info("Starting FPV Dashboard Server")
    
    # Start camera streaming
    camera_streamer = CameraStreamer(STREAM_PORT)
    camera_streaming_active = camera_streamer.start()
    if not camera_streaming_active:
        logger.warning("Camera streaming not available")
    
    # Start HTTP server
    http_server = start_http_server(HTTP_PORT)
    
    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, lambda *args: cleanup())
    signal.signal(signal.SIGTERM, lambda *args: cleanup())
    
    # Print access URLs
    local_ip = socket.gethostbyname(socket.gethostname())
    logger.info(f"Access the dashboard at: http://{local_ip}:{HTTP_PORT}/fpv/web/static/index.html")
    if camera_streaming_active:
        logger.info(f"Direct camera stream: http://{local_ip}:{STREAM_PORT}/")
    
    # Start WebSocket server in asyncio event loop
    asyncio.run(start_websocket_server())

if __name__ == "__main__":
    main()