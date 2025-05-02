import asyncio
import websockets
import json
import time
import serial
import struct
from picamera2 import Picamera2
from picamera2.encoders import JpegEncoder
from picamera2.outputs import FileOutput
from threading import Thread
import socket
import os
import logging
import signal
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Flight controller serial connection
FC_PORT = '/dev/ttyS0'  # UART port, adjust as needed
FC_BAUD = 115200

# WebSocket server port
WS_PORT = 8765

# MJPEG streaming port
STREAM_PORT = 8000

# Global variables
connected_clients = set()
fc_serial = None
stop_event = False

# MSP command constants
MSP_SET_RAW_RC = 200
MSP_RC_TUNING = 111
MSP_ATTITUDE = 108
MSP_ALTITUDE = 109
MSP_ANALOG = 110
MSP_STATUS = 101

class MSPProtocol:
    def __init__(self, serial_port):
        self.serial = serial_port
        
    def msp_send(self, command, data=None):
        if data is None:
            data = []
        
        total_size = len(data)
        checksum = 0
        
        # MSP header: $M
        packet = ['$'.encode('ascii'), 'M'.encode('ascii'), '<'.encode('ascii')]
        
        # Size
        packet.append(struct.pack('B', total_size))
        checksum ^= total_size
        
        # Command
        packet.append(struct.pack('B', command))
        checksum ^= command
        
        # Data
        for i in range(total_size):
            packet.append(struct.pack('B', data[i]))
            checksum ^= data[i]
        
        # Checksum
        packet.append(struct.pack('B', checksum))
        
        # Send the packet
        for p in packet:
            self.serial.write(p)
            
        return True
    
    def msp_parse_response(self):
        header = self.serial.read(3)
        if len(header) != 3:
            return None
        
        if header[0] != ord('$') or header[1] != ord('M') or header[2] != ord('>'):
            return None
        
        size = ord(self.serial.read(1))
        cmd = ord(self.serial.read(1))
        data = self.serial.read(size)
        checksum = ord(self.serial.read(1))
        
        # Verify checksum
        calculated_checksum = size ^ cmd
        for b in data:
            calculated_checksum ^= b
            
        if calculated_checksum != checksum:
            return None
            
        return {'cmd': cmd, 'data': data}
    
    def set_raw_rc(self, channels):
        """
        Send raw RC values (typically between 1000-2000) to the flight controller
        channels: List of channel values [ch1, ch2, ch3, ch4, ...]
        """
        data = []
        for ch in channels:
            data.append(ch & 0xFF)        # low byte
            data.append((ch >> 8) & 0xFF) # high byte
            
        return self.msp_send(MSP_SET_RAW_RC, data)
    
    def arm(self):
        """Arms the flight controller"""
        # Typically, arming is done by setting throttle low and yaw right
        # This implementation may vary depending on your FC
        channels = [1500, 1500, 1000, 1900] + [1500] * 4  # Roll, Pitch, Throttle, Yaw + 4 aux channels
        return self.set_raw_rc(channels)
    
    def disarm(self):
        """Disarms the flight controller"""
        # Typically, disarming is done by setting throttle low and yaw left
        channels = [1500, 1500, 1000, 1100] + [1500] * 4
        return self.set_raw_rc(channels)
    
    def send_command(self, command_str):
        """
        Parse and send various commands to the flight controller
        Format examples:
        - "arm" - arms the motors
        - "disarm" - disarms the motors
        - "set throttle 1500" - sets throttle to 1500
        - "set roll 1200" - sets roll to 1200
        """
        parts = command_str.lower().split()
        response = ""
        
        if not parts:
            return "Empty command"
        
        if parts[0] == "arm":
            if self.arm():
                response = "Armed successfully"
            else:
                response = "Failed to arm"
                
        elif parts[0] == "disarm":
            if self.disarm():
                response = "Disarmed successfully"
            else:
                response = "Failed to disarm"
                
        elif parts[0] == "set" and len(parts) >= 3:
            channel = parts[1]
            try:
                value = int(parts[2])
                # Ensure value is within safe limits
                value = max(1000, min(2000, value))
                
                # Default channels
                channels = [1500, 1500, 1000, 1500] + [1500] * 4
                
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
                
                if self.set_raw_rc(channels):
                    response = f"Set {channel} to {value}"
                else:
                    response = f"Failed to set {channel}"
            except ValueError:
                response = f"Invalid value: {parts[2]}"
        else:
            response = f"Unknown command: {command_str}"
            
        return response

class SensorData:
    def __init__(self):
        self.roll = 0
        self.pitch = 0
        self.yaw = 0
        self.altitude = 0
        self.battery = 0
        self.gps_fix = False
        self.lat = 0
        self.lon = 0
        self.armed = False
        self.flight_mode = "IDLE"
        
    def to_json(self):
        return {
            "type": "sensor_data",
            "roll": self.roll,
            "pitch": self.pitch,
            "yaw": self.yaw,
            "altitude": self.altitude,
            "battery": self.battery,
            "gps_fix": self.gps_fix,
            "lat": self.lat,
            "lon": self.lon,
            "armed": self.armed,
            "flight_mode": self.flight_mode
        }

class MJPEGStreamServer:
    def __init__(self, port=8000):
        self.port = port
        self.server_socket = None
        self.connections = []
        self.camera = None
        
    def start(self):
        # Initialize camera
        self.camera = Picamera2()
        self.camera.configure(self.camera.create_video_configuration(main={"size": (640, 480)}))
        self.camera.start()
        
        # Start the server in a separate thread
        server_thread = Thread(target=self._run_server)
        server_thread.daemon = True
        server_thread.start()
        
        logger.info(f"MJPEG stream server started on port {self.port}")
        
    def _run_server(self):
        try:
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.server_socket.bind(('0.0.0.0', self.port))
            self.server_socket.listen(5)
            
            while not stop_event:
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
        try:
            # Send HTTP response header
            response = (
                b'HTTP/1.0 200 OK\r\n'
                b'Server: PiCamera MJPEG Server\r\n'
                b'Content-Type: multipart/x-mixed-replace; boundary=FRAME\r\n'
                b'Cache-Control: no-cache\r\n'
                b'Connection: close\r\n\r\n'
            )
            conn.sendall(response)
            
            # Stream frames
            while not stop_event:
                # Capture a frame from the camera
                frame = self.camera.capture_array()
                
                # Convert to JPEG
                jpeg_buffer = self.camera.capture_buffer("main", format="jpeg")
                
                # Send the frame with MJPEG header
                conn.sendall(
                    b'--FRAME\r\n'
                    b'Content-Type: image/jpeg\r\n'
                    b'Content-Length: ' + str(len(jpeg_buffer)).encode() + b'\r\n\r\n'
                )
                conn.sendall(jpeg_buffer)
                conn.sendall(b'\r\n')
                
                # Limit frame rate
                time.sleep(0.03)  # ~30fps
                
        except Exception as e:
            logger.error(f"Client connection error: {e}")
        finally:
            try:
                conn.close()
            except:
                pass
    
    def _cleanup(self):
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
            
        logger.info("MJPEG stream server shut down")

async def websocket_server(stop):
    async def handler(websocket, path):
        # Register client
        connected_clients.add(websocket)
        logger.info(f"Client connected. Total clients: {len(connected_clients)}")
        
        try:
            while not stop.is_set():
                # Wait for messages from clients
                message = await asyncio.wait_for(websocket.recv(), timeout=1.0)
                
                # Parse the command
                try:
                    data = json.loads(message)
                    
                    if data.get('type') == 'command':
                        command = data.get('command', '')
                        logger.info(f"Received command: {command}")
                        
                        # Process the command and get response
                        if fc_serial and msp:
                            response = msp.send_command(command)
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
        finally:
            # Unregister client
            connected_clients.remove(websocket)
            logger.info(f"Client disconnected. Total clients: {len(connected_clients)}")
    
    # Start WebSocket server
    async with websockets.serve(handler, "0.0.0.0", WS_PORT):
        logger.info(f"WebSocket server started on port {WS_PORT}")
        while not stop.is_set():
            await asyncio.sleep(1)

async def sensor_data_broadcast(stop):
    sensor_data = SensorData()
    
    # Simulate some sensor data for testing if no flight controller is connected
    simulate_data = fc_serial is None
    
    while not stop.is_set():
        try:
            # If flight controller is connected, read real sensor data
            if fc_serial and msp and not simulate_data:
                # Here you would add code to read real sensor data from the flight controller
                # using the MSP protocol
                pass
            else:
                # Simulate data for testing
                sensor_data.roll += 0.1
                if sensor_data.roll > 45:
                    sensor_data.roll = -45
                    
                sensor_data.pitch += 0.2
                if sensor_data.pitch > 30:
                    sensor_data.pitch = -30
                    
                sensor_data.yaw += 0.5
                if sensor_data.yaw > 360:
                    sensor_data.yaw = 0
                    
                sensor_data.altitude = 10 + 2 * Math.sin(time.time() / 5)
                sensor_data.battery = 70 - time.time() % 30
            
            # Broadcast sensor data to all connected clients
            if connected_clients:
                message = json.dumps(sensor_data.to_json())
                await asyncio.gather(
                    *[client.send(message) for client in connected_clients],
                    return_exceptions=True
                )
                
            # Update at 10Hz
            await asyncio.sleep(0.1)
            
        except Exception as e:
            logger.error(f"Error in sensor data broadcast: {e}")
            await asyncio.sleep(1)

def cleanup(signal_received=None, frame=None):
    global stop_event
    logger.info("Shutting down...")
    
    # Set stop event
    stop_event = True
    
    # Stop the asyncio event loop
    asyncio.get_event_loop().stop()
    
    # Close serial connection
    if fc_serial:
        fc_serial.close()
        
    sys.exit(0)

async def main():
    global fc_serial, msp
    stop = asyncio.Event()
    
    # Initialize serial connection to flight controller
    try:
        fc_serial = serial.Serial(FC_PORT, FC_BAUD, timeout=1)
        msp = MSPProtocol(fc_serial)
        logger.info(f"Connected to flight controller on {FC_PORT}")
    except Exception as e:
        logger.warning(f"Failed to connect to flight controller: {e}")
        logger.warning("Running in simulation mode")
        fc_serial = None
        msp = None
    
    # Start MJPEG streaming server
    mjpeg_server = MJPEGStreamServer(STREAM_PORT)
    mjpeg_server.start()
    
    # Start tasks
    tasks = [
        websocket_server(stop),
        sensor_data_broadcast(stop)
    ]
    
    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, cleanup)
    signal.signal(signal.SIGTERM, cleanup)
    
    await asyncio.gather(*tasks)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        cleanup()