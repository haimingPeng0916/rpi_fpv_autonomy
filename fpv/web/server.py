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
import math
from http.server import SimpleHTTPRequestHandler, HTTPServer
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
FC_PORT = '/dev/ttyS0'  # UART0 (GPIO 14/15)
FC_BAUD = 115200    # Baud rate for flight controller
stop_event = Event()

# Connected WebSocket clients
connected_clients = set()

class FlightController:
    """Interface to the flight controller using MSP protocol"""
    def __init__(self, port=FC_PORT, baudrate=FC_BAUD):
        self.port = port
        self.baudrate = baudrate
        self.serial = None
        self.connected = False
        
    def connect(self):
        """Connect to the flight controller"""
        try:
            import serial
            self.serial = serial.Serial(self.port, self.baudrate, timeout=1)
            self.connected = True
            logger.info(f"Connected to flight controller on {self.port}")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to flight controller: {e}")
            self.connected = False
            return False
    
    def disconnect(self):
        """Disconnect from the flight controller"""
        if self.serial:
            self.serial.close()
            self.connected = False
    
    def msp_send(self, cmd, data=None):
        """Send an MSP command to the flight controller"""
        if not self.connected:
            return False
        
        if data is None:
            data = []
        
        # MSP protocol format: $M<[data_length][cmd][data][checksum]
        # where checksum = XOR of data_length, cmd and all data bytes
        
        try:
            # Create the packet
            packet = bytearray()
            packet.extend(b'$M<')  # Header
            
            # Add data length
            data_length = len(data)
            packet.append(data_length)
            
            # Add command
            packet.append(cmd)
            
            # Calculate checksum (start with data_length and cmd)
            checksum = data_length ^ cmd
            
            # Add data and update checksum
            for b in data:
                packet.append(b)
                checksum ^= b
            
            # Add checksum
            packet.append(checksum)
            
            # Send the packet
            self.serial.write(packet)
            return True
            
        except Exception as e:
            logger.error(f"Error sending MSP command: {e}")
            return False
    
    def msp_read(self):
        """Read and parse an MSP response"""
        if not self.connected:
            return None
        
        try:
            # Wait for the header
            header = self.serial.read(3)
            if len(header) != 3 or header != b'$M>':
                return None
            
            # Read data length
            data_length = ord(self.serial.read(1))
            
            # Read command
            cmd = ord(self.serial.read(1))
            
            # Read data
            data = self.serial.read(data_length)
            
            # Read checksum
            checksum = ord(self.serial.read(1))
            
            # Verify checksum
            calc_checksum = data_length ^ cmd
            for b in data:
                calc_checksum ^= b
                
            if calc_checksum != checksum:
                logger.warning("MSP checksum error")
                return None
                
            return {'cmd': cmd, 'data': data}
            
        except Exception as e:
            logger.error(f"Error reading MSP response: {e}")
            return None
    
    def get_attitude(self):
        """Get attitude data (roll, pitch, yaw)"""
        if not self.connected:
            return None
            
        # MSP_ATTITUDE command (108)
        self.msp_send(108)
        
        # Wait for response
        response = self.msp_read()
        if response is None or response['cmd'] != 108:
            return None
            
        data = response['data']
        if len(data) < 6:
            return None
            
        # Parse attitude data (Betaflight/iNav format)
        # Values are in 1/10 degrees
        roll = (data[0] | (data[1] << 8))
        if roll > 32767:
            roll -= 65536
        roll = roll / 10.0
        
        pitch = (data[2] | (data[3] << 8))
        if pitch > 32767:
            pitch -= 65536
        pitch = pitch / 10.0
        
        yaw = (data[4] | (data[5] << 8))
        if yaw > 32767:
            yaw -= 65536
        
        return {'roll': roll, 'pitch': pitch, 'yaw': yaw}
    
    def get_altitude(self):
        """Get altitude data"""
        # MSP_ALTITUDE command (109)
        self.msp_send(109)
        
        response = self.msp_read()
        if response is None or response['cmd'] != 109:
            return None
            
        data = response['data']
        if len(data) < 4:
            return None
            
        # Parse altitude in cm, convert to meters
        altitude_cm = (data[0] | (data[1] << 8) | (data[2] << 16) | (data[3] << 24))
        if altitude_cm > 2147483647:
            altitude_cm -= 4294967296
        
        return altitude_cm / 100.0  # Convert to meters
    
    def get_battery(self):
        """Get battery status"""
        # MSP_ANALOG command (110)
        self.msp_send(110)
        
        response = self.msp_read()
        if response is None or response['cmd'] != 110:
            return None
            
        data = response['data']
        if len(data) < 3:
            return None
            
        # Parse battery voltage (0.1V units)
        voltage = data[0] / 10.0
        
        # Convert to percentage (adjust for your battery)
        # Assuming 3S LiPo (min 9V, max 12.6V)
        min_voltage = 9.0
        max_voltage = 12.6
        percentage = (voltage - min_voltage) / (max_voltage - min_voltage) * 100
        percentage = max(0, min(100, percentage))
        
        return percentage
    
    def is_armed(self):
        """Check if the flight controller is armed"""
        # MSP_STATUS command (101)
        self.msp_send(101)
        
        response = self.msp_read()
        if response is None or response['cmd'] != 101:
            return False
            
        data = response['data']
        if len(data) < 2:
            return False
            
        # Parse flags
        flags = data[0] | (data[1] << 8)
        
        # Check armed bit (usually bit 0)
        return (flags & 1) != 0
    
    def get_flight_mode(self):
        """Get current flight mode"""
        # MSP_STATUS command (101)
        self.msp_send(101)
        
        response = self.msp_read()
        if response is None or response['cmd'] != 101:
            return "UNKNOWN"
            
        data = response['data']
        if len(data) < 6:
            return "UNKNOWN"
            
        # Parse flight mode flags
        flags = data[4] | (data[5] << 8)
        
        # Map flags to flight modes (may vary by firmware)
        if flags & (1 << 0):
            return "ANGLE"
        elif flags & (1 << 1):
            return "HORIZON"
        elif flags & (1 << 2):
            return "NAV ALTHOLD"
        elif flags & (1 << 3):
            return "HEADING HOLD"
        elif flags & (1 << 4):
            return "NAV RTH"
        elif flags & (1 << 5):
            return "NAV POSHOLD"
        elif flags & (1 << 6):
            return "NAV WP"
        elif flags & (1 << 7):
            return "HEADFREE"
        else:
            return "MANUAL"
    
    def get_gps_data(self):
        """Get GPS position data"""
        # MSP_RAW_GPS command (106)
        self.msp_send(106)
        
        response = self.msp_read()
        if response is None or response['cmd'] != 106:
            return None
            
        data = response['data']
        if len(data) < 14:
            return None
            
        # Parse GPS fix status
        fix = data[0]
        
        # Parse coordinates (degrees * 10,000,000)
        lat = (data[1] | (data[2] << 8) | (data[3] << 16) | (data[4] << 24)) / 10000000.0
        lon = (data[5] | (data[6] << 8) | (data[7] << 16) | (data[8] << 24)) / 10000000.0
        
        return {'fix': fix != 0, 'lat': lat, 'lon': lon}
    
    def execute_command(self, command_str):
        """Execute a command from the web interface"""
        if not self.connected:
            return "Flight controller not connected"
            
        # Parse the command
        parts = command_str.lower().split()
        if not parts:
            return "Invalid command"
            
        # Handle different commands
        if parts[0] == "arm":
            # MSP_SET_RAW_RC command (200)
            # Set throttle low, yaw right to arm
            channels = [1500, 1500, 1000, 1900, 1500, 1500, 1500, 1500]
            data = []
            for ch in channels:
                data.append(ch & 0xFF)
                data.append((ch >> 8) & 0xFF)
            
            if self.msp_send(200, data):
                return "Armed successfully"
            else:
                return "Arming failed"
                
        elif parts[0] == "disarm":
            # Set throttle low, yaw left to disarm
            channels = [1500, 1500, 1000, 1100, 1500, 1500, 1500, 1500]
            data = []
            for ch in channels:
                data.append(ch & 0xFF)
                data.append((ch >> 8) & 0xFF)
            
            if self.msp_send(200, data):
                return "Disarmed successfully"
            else:
                return "Disarming failed"
                
        elif parts[0] == "set" and len(parts) >= 3:
            try:
                channel = parts[1]
                value = int(parts[2])
                
                # Ensure value is within safe limits
                value = max(1000, min(2000, value))
                
                # Default RC channels
                channels = [1500, 1500, 1000, 1500, 1500, 1500, 1500, 1500]
                
                # Set the specified channel
                if channel == "roll":
                    channels[0] = value
                elif channel == "pitch":
                    channels[1] = value
                elif channel == "throttle":
                    channels[2] = value
                elif channel == "yaw":
                    channels[3] = value
                elif channel.startswith("aux") and len(channel) > 3:
                    try:
                        aux_num = int(channel[3:])
                        if 1 <= aux_num <= 4:
                            channels[3 + aux_num] = value
                        else:
                            return f"Invalid aux channel: {channel}"
                    except ValueError:
                        return f"Invalid aux channel: {channel}"
                else:
                    return f"Unknown channel: {channel}"
                
                # Send RC channel values
                data = []
                for ch in channels:
                    data.append(ch & 0xFF)
                    data.append((ch >> 8) & 0xFF)
                
                if self.msp_send(200, data):
                    return f"Set {channel} to {value}"
                else:
                    return f"Failed to set {channel}"
                    
            except ValueError:
                return f"Invalid value: {parts[2]}"
        else:
            return f"Unknown command: {command_str}"

class PiCameraStreamer:
    """Handles Raspberry Pi camera streaming using MJPEG"""
    def __init__(self, port=8000):
        self.port = port
        self.server_socket = None
        self.connections = []
        self.camera = None
        self.frame_queue = None
        self.frame_queue_lock = threading.Lock()
    
    def start(self):
        """Initialize camera and start streaming server"""
        try:
            # Try to use picamera2 (optimized for Pi Camera)
            from picamera2 import Picamera2
            import numpy as np
            
            # Initialize the camera (cam_num=1 for second camera)
            camera_num = 1  # Use CAM 1 as specified
            self.camera = Picamera2(camera_num)
            
            # Configure camera
            config = self.camera.create_preview_configuration(
                main={"size": (640, 480), "format": "RGB888"}
            )
            self.camera.configure(config)
            
            # Start the camera
            self.camera.start()
            logger.info(f"Initialized Pi Camera on camera port {camera_num}")
            
            # Initialize frame queue for future CV processing
            self.frame_queue = []
            
            # Start separate threads for streaming and frame processing
            streaming_thread = Thread(target=self._run_streaming_server)
            streaming_thread.daemon = True
            streaming_thread.start()
            
            # Start frame capture thread (for future CV processing)
            capture_thread = Thread(target=self._capture_frames)
            capture_thread.daemon = True
            capture_thread.start()
            
            logger.info(f"Camera streaming server started on port {self.port}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize Pi Camera: {e}")
            return False
    
    def _capture_frames(self):
        """Continuously capture frames for processing (for future CV use)"""
        if not self.camera:
            return
            
        while not stop_event.is_set():
            try:
                # Capture a frame
                frame = self.camera.capture_array()
                
                # Store the frame for future processing (keep only the most recent)
                with self.frame_queue_lock:
                    self.frame_queue = [frame]  # Only keep latest frame
                
                # Sleep to control capture rate
                time.sleep(0.03)  # ~30 fps
                
            except Exception as e:
                logger.error(f"Error capturing frame: {e}")
                time.sleep(0.1)
    
    def _run_streaming_server(self):
        """Run the MJPEG streaming server"""
        try:
            # Create server socket
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.server_socket.bind(('0.0.0.0', self.port))
            self.server_socket.listen(5)
            
            while not stop_event.is_set():
                try:
                    # Accept client connections
                    conn, addr = self.server_socket.accept()
                    logger.info(f"New stream connection from {addr}")
                    
                    # Handle each client in a separate thread
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
            # Send HTTP response header for MJPEG stream
            response = (
                b'HTTP/1.0 200 OK\r\n'
                b'Server: Pi Camera MJPEG Server\r\n'
                b'Content-Type: multipart/x-mixed-replace; boundary=FRAME\r\n'
                b'Cache-Control: no-cache\r\n'
                b'Connection: close\r\n\r\n'
            )
            conn.sendall(response)
            
            # Import OpenCV for JPEG encoding
            import cv2
            
            # Stream frames
            while not stop_event.is_set() and self.camera:
                try:
                    # Capture frame directly for streaming
                    frame = self.camera.capture_array()
                    
                    # Convert to JPEG
                    ret, jpeg_buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                    if not ret:
                        logger.warning("Failed to encode frame to JPEG")
                        continue
                    
                    # Get the byte array
                    jpeg_bytes = jpeg_buffer.tobytes()
                    
                    # Send the frame with MJPEG header
                    conn.sendall(
                        b'--FRAME\r\n'
                        b'Content-Type: image/jpeg\r\n'
                        b'Content-Length: ' + str(len(jpeg_bytes)).encode() + b'\r\n\r\n'
                    )
                    conn.sendall(jpeg_bytes)
                    conn.sendall(b'\r\n')
                    
                    # Control streaming rate
                    time.sleep(0.05)  # ~20 fps for streaming
                    
                except ConnectionError:
                    # Client disconnected
                    break
                except Exception as e:
                    logger.error(f"Error streaming frame: {e}")
                    time.sleep(0.1)
                    
        except Exception as e:
            logger.error(f"Client handling error: {e}")
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
        
        # Stop camera
        if self.camera:
            self.camera.stop()
            
        logger.info("Camera streaming server shut down")
    
    def get_latest_frame(self):
        """Get the latest frame (for future CV processing)"""
        with self.frame_queue_lock:
            if self.frame_queue:
                return self.frame_queue[0].copy()
            else:
                return None

class DashboardHandler(SimpleHTTPRequestHandler):
    """Serves the web dashboard files"""
    def log_message(self, format, *args):
        logger.info(format % args)
    
    def end_headers(self):
        # Set proper content types
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
    
    # Run in a thread
    thread = threading.Thread(target=server.serve_forever)
    thread.daemon = True
    thread.start()
    
    return server

async def websocket_handler(websocket, path, flight_controller):
    """Handle WebSocket connections for flight data and commands"""
    # Register client
    connected_clients.add(websocket)
    logger.info(f"Client connected. Total clients: {len(connected_clients)}")
    
    try:
        while not stop_event.is_set():
            try:
                # Wait for messages from clients with a timeout
                message = await asyncio.wait_for(websocket.recv(), timeout=1.0)
                
                # Parse the command
                try:
                    data = json.loads(message)
                    
                    if data.get('type') == 'command':
                        command = data.get('command', '')
                        logger.info(f"Received command: {command}")
                        
                        # Process the command
                        if flight_controller and flight_controller.connected:
                            response = flight_controller.execute_command(command)
                        else:
                            response = "Flight controller not connected"
                            
                        # Send response back to client
                        await websocket.send(json.dumps({
                            "type": "command_response",
                            "message": response
                        }))
                        
                except json.JSONDecodeError:
                    logger.error(f"Invalid JSON: {message}")
                except Exception as e:
                    logger.error(f"Error processing command: {e}")
                    
            except asyncio.TimeoutError:
                # This is fine - it's just our timeout to check stop flag
                pass
            except websockets.exceptions.ConnectionClosed:
                logger.info("Client disconnected")
                break
    finally:
        # Unregister client
        connected_clients.remove(websocket)
        logger.info(f"Client disconnected. Total clients: {len(connected_clients)}")

async def flight_data_broadcast(flight_controller):
    """Broadcast flight data to all connected clients"""
    # Create simulated data for fallback
    simulated_data = {
        "roll": 0.0,
        "pitch": 0.0,
        "yaw": 0.0,
        "altitude": 0.0,
        "battery": 100.0,
        "gps_fix": False,
        "lat": 0.0,
        "lon": 0.0,
        "armed": False,
        "flight_mode": "IDLE"
    }
    
    while not stop_event.is_set():
        try:
            # Get data either from flight controller or simulation
            if flight_controller and flight_controller.connected:
                # Try to get real flight data
                attitude = flight_controller.get_attitude()
                if attitude:
                    simulated_data["roll"] = attitude["roll"]
                    simulated_data["pitch"] = attitude["pitch"]
                    simulated_data["yaw"] = attitude["yaw"]
                
                altitude = flight_controller.get_altitude()
                if altitude is not None:
                    simulated_data["altitude"] = altitude
                
                battery = flight_controller.get_battery()
                if battery is not None:
                    simulated_data["battery"] = battery
                
                gps_data = flight_controller.get_gps_data()
                if gps_data:
                    simulated_data["gps_fix"] = gps_data["fix"]
                    simulated_data["lat"] = gps_data["lat"]
                    simulated_data["lon"] = gps_data["lon"]
                
                simulated_data["armed"] = flight_controller.is_armed()
                simulated_data["flight_mode"] = flight_controller.get_flight_mode()
            else:
                # Update simulated data
                simulated_data["roll"] = 45 * math.sin(time.time() * 0.5)
                simulated_data["pitch"] = 30 * math.sin(time.time() * 0.3)
                simulated_data["yaw"] = (time.time() * 10) % 360
                simulated_data["altitude"] = 10 + 2 * math.sin(time.time() / 5)
                simulated_data["battery"] = max(0, 100.0 - ((time.time() % 300) / 3))
                simulated_data["gps_fix"] = True
                simulated_data["lat"] = 35.689842 + (math.sin(time.time() * 0.1) * 0.001)
                simulated_data["lon"] = 139.691951 + (math.cos(time.time() * 0.1) * 0.001)
            
            # Prepare message
            message = {
                "type": "sensor_data",
                "roll": round(simulated_data["roll"], 1),
                "pitch": round(simulated_data["pitch"], 1),
                "yaw": round(simulated_data["yaw"], 1),
                "altitude": round(simulated_data["altitude"], 1),
                "battery": round(simulated_data["battery"], 1),
                "gps_fix": simulated_data["gps_fix"],
                "lat": simulated_data["lat"],
                "lon": simulated_data["lon"],
                "armed": simulated_data["armed"],
                "flight_mode": simulated_data["flight_mode"]
            }
            
            # Broadcast to all connected clients
            if connected_clients:
                await asyncio.gather(
                    *[client.send(json.dumps(message)) for client in connected_clients],
                    return_exceptions=True
                )
                
            # Update at 10Hz
            await asyncio.sleep(0.1)
            
        except Exception as e:
            logger.error(f"Error in flight data broadcast: {e}")
            await asyncio.sleep(1)

async def start_websocket_server(flight_controller):
    """Start the WebSocket server"""
    async with websockets.serve(
        lambda ws, path: websocket_handler(ws, path, flight_controller), 
        "0.0.0.0", 
        WS_PORT
    ):
        logger.info(f"WebSocket server started on port {WS_PORT}")
        
        # Start flight data broadcast
        broadcast_task = asyncio.create_task(flight_data_broadcast(flight_controller))
        
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

async def main():
    # Print startup message
    logger.info("Starting FPV Dashboard Server")
    
    # Connect to flight controller
    flight_controller = FlightController(FC_PORT, FC_BAUD)
    fc_connected = flight_controller.connect()
    
    # Start camera streaming
    camera_streamer = PiCameraStreamer(STREAM_PORT)
    camera_streaming_active = camera_streamer.start()
    
    # Start HTTP server for dashboard
    http_server = start_http_server(HTTP_PORT)
    
    # Check if dashboard exists
    if not os.path.exists("fpv/web/index.html"):
        logger.warning("Dashboard HTML file not found at fpv/web/index.html")
    
    # Print access URLs
    try:
        local_ip = socket.gethostbyname(socket.gethostname())
    except:
        local_ip = "localhost"
        
    logger.info(f"Access the dashboard at: http://{local_ip}:{HTTP_PORT}/fpv/web/index.html")
    if camera_streaming_active:
        logger.info(f"Direct camera stream: http://{local_ip}:{STREAM_PORT}/")
    
    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, lambda *args: cleanup())
    signal.signal(signal.SIGTERM, lambda *args: cleanup())
    
    # Start WebSocket server
    await start_websocket_server(flight_controller)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        cleanup()