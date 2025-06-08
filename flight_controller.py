# flight_controller_enhanced.py
import time
import threading
import math
import random
import struct
import serial
import copy
from collections import deque
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class MSPMessage:
    """MSP message structure"""
    code: int
    data: bytes
    timestamp: float

class FlightController:
    """Enhanced flight controller with optimized real-time MSP communication."""
    
    # MSP protocol constants
    MSP_HEADER = b'$M<'
    MSP_ATTITUDE = 108
    MSP_ALTITUDE = 109
    MSP_ANALOG = 110
    MSP_STATUS = 101
    MSP_RAW_GPS = 106
    MSP_COMP_GPS = 107
    MSP_SET_RAW_RC = 200
    MSP_RC = 105
    
    # Performance tuning
    TELEMETRY_UPDATE_RATE = 50  # Hz
    SERIAL_TIMEOUT = 0.02  # 20ms timeout for low latency
    MAX_RETRY_COUNT = 3
    MESSAGE_QUEUE_SIZE = 100
    
    def __init__(self, simulation=True, port='/dev/ttyUSB0', baudrate=115200):
        """Initialize enhanced flight controller."""
        self.simulation = simulation
        self.port = port
        self.baudrate = baudrate
        self.serial = None
        
        # Threading components
        self.data_lock = threading.Lock()
        self.serial_lock = threading.Lock()
        self.update_thread = None
        self.serial_thread = None
        self.running = False
        
        # Message queuing for low latency
        self.rx_queue = deque(maxlen=self.MESSAGE_QUEUE_SIZE)
        self.tx_queue = deque(maxlen=self.MESSAGE_QUEUE_SIZE)
        
        # Telemetry with timestamp for freshness checking
        self.telemetry = self._init_telemetry()
        self.telemetry_timestamps = {}
        
        # Performance monitoring
        self.stats = {
            'messages_sent': 0,
            'messages_received': 0,
            'errors': 0,
            'avg_latency': 0,
            'update_rate': 0
        }
        self.last_update_time = time.time()
        self.update_count = 0
        
    def _init_telemetry(self) -> Dict[str, Any]:
        """Initialize telemetry structure."""
        return {
            'attitude': {'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0},
            'altitude': 0.0,
            'battery': 12.6,
            'armed': False,
            'mode': 'ANGLE',
            'rssi': 80,
            'airspeed': 0.0,
            'ground_speed': 0.0,
            'home_distance': 0.0,
            'timestamp': time.time(),
            'gps': {
                'fix': False,
                'satellites': 0,
                'latitude': 0.0,
                'longitude': 0.0,
                'altitude': 0.0,
                'speed': 0.0,
                'course': 0.0
            },
            'rc_channels': [1500] * 16,
            'status': {
                'cycle_time': 0,
                'i2c_errors': 0,
                'sensors': []
            }
        }
    
    def connect_to_fc(self) -> bool:
        """Connect to flight controller with optimized settings."""
        if self.simulation:
            logger.info("Running in simulation mode")
            return True
            
        try:
            # Configure serial with low latency settings
            self.serial = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                timeout=self.SERIAL_TIMEOUT,
                write_timeout=self.SERIAL_TIMEOUT,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                xonxoff=False,
                rtscts=False,
                dsrdtr=False
            )
            
            # Set low latency mode on Linux
            try:
                import fcntl
                import struct
                # ASYNC_LOW_LATENCY = 0x2000
                fcntl.ioctl(self.serial.fd, 0x541F, struct.pack('I', 0x2000))
            except:
                pass  # Not critical if it fails
            
            # Clear buffers
            self.serial.reset_input_buffer()
            self.serial.reset_output_buffer()
            
            logger.info(f"Connected to {self.port} at {self.baudrate} baud")
            
            # Test connection
            if self._test_connection():
                logger.info("Flight controller connection verified")
                return True
            else:
                logger.error("Failed to verify connection")
                self.disconnect()
                return False
                
        except serial.SerialException as e:
            logger.error(f"Serial connection failed: {e}")
            return False
    
    def _test_connection(self) -> bool:
        """Test connection by requesting status."""
        for _ in range(3):
            if self._send_msp_request(self.MSP_STATUS):
                response = self._read_msp_response(self.MSP_STATUS, timeout=0.1)
                if response:
                    return True
            time.sleep(0.1)
        return False
    
    def disconnect(self):
        """Disconnect from flight controller."""
        self.stop_updates()
        
        with self.serial_lock:
            if self.serial and self.serial.is_open:
                try:
                    self.serial.close()
                    logger.info("Disconnected from flight controller")
                except Exception as e:
                    logger.error(f"Error during disconnect: {e}")
    
    def start_updates(self) -> bool:
        """Start telemetry update threads."""
        if self.running:
            logger.warning("Updates already running")
            return False
            
        self.running = True
        
        # Start serial communication thread (real mode only)
        if not self.simulation:
            self.serial_thread = threading.Thread(target=self._serial_loop, daemon=True)
            self.serial_thread.start()
        
        # Start update thread
        self.update_thread = threading.Thread(target=self._update_loop, daemon=True)
        self.update_thread.start()
        
        logger.info("Started flight controller threads")
        return True
    
    def stop_updates(self):
        """Stop all update threads."""
        self.running = False
        
        # Wait for threads to stop
        if self.serial_thread and self.serial_thread.is_alive():
            self.serial_thread.join(timeout=1.0)
            
        if self.update_thread and self.update_thread.is_alive():
            self.update_thread.join(timeout=1.0)
            
        logger.info("Stopped flight controller threads")
    
    def _serial_loop(self):
        """Dedicated thread for serial communication."""
        logger.info("Serial communication thread started")
        
        while self.running:
            try:
                # Process outgoing messages
                if self.tx_queue:
                    msg = self.tx_queue.popleft()
                    self._send_msp_command_raw(msg.code, msg.data)
                
                # Check for incoming data
                if self.serial.in_waiting > 0:
                    self._process_incoming_data()
                    
                # Small sleep to prevent CPU spinning
                time.sleep(0.001)
                
            except Exception as e:
                logger.error(f"Serial loop error: {e}")
                self.stats['errors'] += 1
                time.sleep(0.01)
    
    def _process_incoming_data(self):
        """Process any available serial data."""
        try:
            # Read available data
            data = self.serial.read(self.serial.in_waiting)
            
            # Parse MSP messages (simplified for performance)
            # In production, you'd want a proper state machine parser
            if len(data) >= 6:  # Minimum MSP message size
                # Look for MSP header
                idx = data.find(b'$M>')
                if idx >= 0:
                    # Extract message (simplified)
                    # Full implementation would properly parse multiple messages
                    self.rx_queue.append(MSPMessage(
                        code=data[idx+4] if idx+4 < len(data) else 0,
                        data=data[idx+5:],
                        timestamp=time.time()
                    ))
        except Exception as e:
            logger.error(f"Error processing incoming data: {e}")
    
    def _update_loop(self):
        """Main update loop for telemetry requests."""
        logger.info("Telemetry update loop started")
        
        # Request cycle for different data types
        request_cycle = [
            (self.MSP_ATTITUDE, 50),    # 50Hz
            (self.MSP_ALTITUDE, 10),    # 10Hz
            (self.MSP_ANALOG, 10),      # 10Hz
            (self.MSP_RAW_GPS, 5),      # 5Hz
            (self.MSP_STATUS, 2),       # 2Hz
        ]
        
        cycle_timers = {cmd: 0 for cmd, _ in request_cycle}
        last_cycle_time = time.time()
        
        while self.running:
            try:
                current_time = time.time()
                cycle_dt = current_time - last_cycle_time
                last_cycle_time = current_time
                
                if self.simulation:
                    self._simulate_telemetry()
                else:
                    # Send requests based on their update rates
                    for cmd, rate in request_cycle:
                        cycle_timers[cmd] += cycle_dt
                        if cycle_timers[cmd] >= 1.0 / rate:
                            self._request_telemetry(cmd)
                            cycle_timers[cmd] = 0
                    
                    # Process received messages
                    self._process_received_messages()
                
                # Update statistics
                self._update_statistics()
                
                # Target loop rate
                elapsed = time.time() - current_time
                sleep_time = max(0, (1.0 / self.TELEMETRY_UPDATE_RATE) - elapsed)
                if sleep_time > 0:
                    time.sleep(sleep_time)
                    
            except Exception as e:
                logger.error(f"Update loop error: {e}")
                self.stats['errors'] += 1
                time.sleep(0.1)
    
    def _request_telemetry(self, msg_code: int):
        """Queue telemetry request."""
        self.tx_queue.append(MSPMessage(msg_code, b'', time.time()))
    
    def _process_received_messages(self):
        """Process messages from receive queue."""
        while self.rx_queue:
            try:
                msg = self.rx_queue.popleft()
                self._handle_msp_message(msg)
            except Exception as e:
                logger.error(f"Error processing message: {e}")
    
    def _handle_msp_message(self, msg: MSPMessage):
        """Handle received MSP message."""
        data = msg.data
        
        with self.data_lock:
            if msg.code == self.MSP_ATTITUDE and len(data) >= 6:
                self.telemetry['attitude']['roll'] = self._bytes_to_int16(data[0:2]) / 10.0
                self.telemetry['attitude']['pitch'] = self._bytes_to_int16(data[2:4]) / 10.0
                self.telemetry['attitude']['yaw'] = self._bytes_to_int16(data[4:6])
                self.telemetry_timestamps['attitude'] = msg.timestamp
                
            elif msg.code == self.MSP_ALTITUDE and len(data) >= 6:
                self.telemetry['altitude'] = self._bytes_to_int32(data[0:4]) / 100.0
                vario = self._bytes_to_int16(data[4:6]) / 100.0
                self.telemetry_timestamps['altitude'] = msg.timestamp
                
            elif msg.code == self.MSP_ANALOG and len(data) >= 7:
                self.telemetry['battery'] = data[0] / 10.0
                self.telemetry['rssi'] = data[3] if len(data) > 3 else 0
                self.telemetry_timestamps['analog'] = msg.timestamp
                
            elif msg.code == self.MSP_RAW_GPS and len(data) >= 16:
                self.telemetry['gps']['fix'] = data[0] > 0
                self.telemetry['gps']['satellites'] = data[1]
                self.telemetry['gps']['latitude'] = self._bytes_to_int32(data[2:6]) / 10000000.0
                self.telemetry['gps']['longitude'] = self._bytes_to_int32(data[6:10]) / 10000000.0
                self.telemetry['gps']['altitude'] = self._bytes_to_int16(data[10:12])
                self.telemetry['gps']['speed'] = self._bytes_to_int16(data[12:14]) / 100.0
                self.telemetry['gps']['course'] = self._bytes_to_int16(data[14:16]) / 10.0
                self.telemetry_timestamps['gps'] = msg.timestamp
            
            # Update main timestamp
            self.telemetry['timestamp'] = time.time()
            
        self.stats['messages_received'] += 1
    
    def _send_msp_request(self, msg_code: int) -> bool:
        """Send MSP request message."""
        return self._send_msp_command_raw(msg_code, b'')
    
    def _send_msp_command_raw(self, msg_code: int, data: bytes) -> bool:
        """Send raw MSP command."""
        if not self.serial or not self.serial.is_open:
            return False
            
        try:
            # Build MSP packet
            size = len(data)
            checksum = size ^ msg_code
            for byte in data:
                checksum ^= byte
            
            packet = bytearray(self.MSP_HEADER)
            packet.extend([size, msg_code])
            packet.extend(data)
            packet.append(checksum)
            
            # Send with lock
            with self.serial_lock:
                self.serial.write(packet)
                self.serial.flush()
                
            self.stats['messages_sent'] += 1
            return True
            
        except Exception as e:
            logger.error(f"Error sending MSP command: {e}")
            self.stats['errors'] += 1
            return False
    
    def _read_msp_response(self, expected_code: int, timeout: float = 0.05) -> Optional[bytes]:
        """Read MSP response (legacy method for compatibility)."""
        # In the new architecture, responses are handled by the serial thread
        # This method checks the receive queue for matching messages
        start_time = time.time()
        
        while (time.time() - start_time) < timeout:
            for msg in list(self.rx_queue):
                if msg.code == expected_code:
                    self.rx_queue.remove(msg)
                    return msg.data
            time.sleep(0.001)
            
        return None
    
    def get_telemetry(self) -> Dict[str, Any]:
        """Get current telemetry data (thread-safe)."""
        with self.data_lock:
            return copy.deepcopy(self.telemetry)
    
    def get_telemetry_freshness(self) -> Dict[str, float]:
        """Get age of telemetry data in seconds."""
        current_time = time.time()
        with self.data_lock:
            return {
                key: current_time - timestamp 
                for key, timestamp in self.telemetry_timestamps.items()
            }
    
    def send_rc_channels(self, channels: list) -> bool:
        """Send RC channel values."""
        if len(channels) < 4:
            logger.error("At least 4 RC channels required")
            return False
            
        # Pack channel data (16-bit values)
        data = bytearray()
        for i in range(min(16, len(channels))):
            value = max(1000, min(2000, channels[i]))
            data.extend(struct.pack('<H', value))
            
        # Queue the command
        self.tx_queue.append(MSPMessage(self.MSP_SET_RAW_RC, bytes(data), time.time()))
        return True
    
    def arm_motors(self) -> bool:
        """Arm the motors."""
        # Typical arm sequence: throttle low, yaw right
        return self.send_rc_channels([1500, 1500, 1000, 2000, 1000, 1000, 1000, 1000])
    
    def disarm_motors(self) -> bool:
        """Disarm the motors."""
        # Typical disarm sequence: throttle low, yaw left
        return self.send_rc_channels([1500, 1500, 1000, 1000, 1000, 1000, 1000, 1000])
    
    def _update_statistics(self):
        """Update performance statistics."""
        self.update_count += 1
        current_time = time.time()
        
        if current_time - self.last_update_time >= 1.0:
            self.stats['update_rate'] = self.update_count / (current_time - self.last_update_time)
            self.update_count = 0
            self.last_update_time = current_time
            
            # Log statistics periodically
            if self.stats['update_rate'] > 0:
                logger.debug(f"Update rate: {self.stats['update_rate']:.1f}Hz, "
                           f"Messages: {self.stats['messages_sent']}/{self.stats['messages_received']}, "
                           f"Errors: {self.stats['errors']}")
    
    def _simulate_telemetry(self):
        """Generate simulated flight data."""
        t = time.time()
        
        with self.data_lock:
            # Realistic flight simulation
            self.telemetry['attitude']['roll'] = 15 * math.sin(t / 2)
            self.telemetry['attitude']['pitch'] = 10 * math.cos(t / 3)
            self.telemetry['attitude']['yaw'] = (self.telemetry['attitude']['yaw'] + 1) % 360
            
            # Altitude with some variation
            self.telemetry['altitude'] = 50 + 10 * math.sin(t / 10)
            
            # Battery drain simulation
            self.telemetry['battery'] = max(10.0, 12.6 - (t % 600) / 100)
            
            # Arm status changes
            if int(t) % 20 == 0:
                self.telemetry['armed'] = not self.telemetry['armed']
            
            # GPS simulation
            self.telemetry['gps']['fix'] = True
            self.telemetry['gps']['satellites'] = random.randint(8, 12)
            self.telemetry['gps']['latitude'] += random.uniform(-0.00001, 0.00001)
            self.telemetry['gps']['longitude'] += random.uniform(-0.00001, 0.00001)
            
            # Speed simulation
            self.telemetry['airspeed'] = max(0, 15 + 5 * math.sin(t / 5))
            self.telemetry['ground_speed'] = self.telemetry['airspeed'] * random.uniform(0.9, 1.1)
            
            # Update timestamp
            self.telemetry['timestamp'] = t
            
            # Simulate all telemetry being fresh
            current_time = time.time()
            for key in ['attitude', 'altitude', 'analog', 'gps']:
                self.telemetry_timestamps[key] = current_time
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get performance statistics."""
        return copy.deepcopy(self.stats)
    
    # Helper methods
    def _bytes_to_int16(self, byte_array: bytes) -> int:
        """Convert 2 bytes to signed 16-bit integer (little endian)."""
        return struct.unpack('<h', bytes(byte_array))[0]
    
    def _bytes_to_int32(self, byte_array: bytes) -> int:
        """Convert 4 bytes to signed 32-bit integer (little endian)."""
        return struct.unpack('<i', bytes(byte_array))[0]
    
    def _bytes_to_uint16(self, byte_array: bytes) -> int:
        """Convert 2 bytes to unsigned 16-bit integer (little endian)."""
        return struct.unpack('<H', bytes(byte_array))[0]
    
    def _bytes_to_uint32(self, byte_array: bytes) -> int:
        """Convert 4 bytes to unsigned 32-bit integer (little endian)."""
        return struct.unpack('<I', bytes(byte_array))[0]