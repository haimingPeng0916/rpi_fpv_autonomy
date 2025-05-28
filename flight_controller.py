# flight_controller.py
import time
import threading
import math
import random
import struct
import serial
import copy

class FlightController:

    # MSP command constants
    MSP_ATTITUDE = 108  # Get attitude data (roll, pitch, yaw)
    MSP_ALTITUDE = 109  # Get altitude data
    MSP_ANALOG = 110    # Get battery and RSSI data
    MSP_STATUS = 101    # Get status info
    MSP_SET_RAW_RC = 200  # Set RC channel values

    def __init__(self, simulation=True, port='/dev/ttyS0', baudrate=115200):
        """Initialize flight controller interface."""
        self.simulation = simulation
        self.port = port 
        self.baudrate = baudrate
        self.serial = None
        
        # Flight data that gets updated by background thread
        self.telemetry = {
            'attitude': {
                'roll': 0.0,
                'pitch': 0.0,
                'yaw': 0.0
            },
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
                'longitude': 0.0
            },
        }

        # Threading components
        self.data_lock = threading.Lock()  # Protects telemetry data
        self.command_lock = threading.Lock()  # Protects serial port for commands
        self.update_thread = None
        self.running = False
    
    def connect_to_fc(self):
        """Connect to a real flight controller"""
        if self.simulation:
            print("Running in simulation mode - no serial connection needed")
            return True
            
        try:
            self.serial = serial.Serial(self.port, self.baudrate, timeout=1.0)
            print(f"Connected to {self.port} at {self.baudrate} baud")
            return True
        except serial.SerialException as e:
            print(f"Serial port error: {e}")
            return False
    
    def disconnect(self):
        """Disconnect from flight controller"""
        try:
            if self.serial and self.serial.is_open:
                self.serial.close()
                print(f"Disconnected from {self.port}")
        except Exception as e:
            print(f"Exception while disconnecting: {e}")

    def send_msp_request(self, msg_id): 
        """Send MSP request (for data queries)"""
        if not self.serial or not self.serial.is_open:
            return False

        data_buffer = bytearray([0x24, 0x4D, 0x3C])  # $M<
        size = 0
        checksum = size ^ msg_id
        data_buffer.append(size)
        data_buffer.append(msg_id)
        data_buffer.append(checksum)

        try:
            bytes_written = self.serial.write(data_buffer)
            self.serial.flush()
            return bytes_written == len(data_buffer)
        except Exception as e:
            print(f"Exception while sending msp request: {e}")
            return False

    def send_msp_command(self, msg_id, data):
        """Send MSP command with payload (for control commands)"""
        if not self.serial or not self.serial.is_open:
            return False
        
        if not isinstance(data, (list, bytearray, bytes)):
            print("Error: data must be a list, bytearray or bytes")
            return False
        
        # Calculate payload size and checksum
        size = len(data)
        checksum = size ^ msg_id
        for byte in data:
            checksum ^= byte
        
        # Create packet: $M<[size][msg_id][data...][checksum]
        packet = bytearray([0x24, 0x4D, 0x3C, size, msg_id])
        packet.extend(data)
        packet.append(checksum)
        
        try:
            bytes_written = self.serial.write(packet)
            self.serial.flush()
            return bytes_written == len(packet)
        except Exception as e:
            print(f"Error sending MSP command: {e}")
            return False

    def read_msp_response(self, msg_id, timeout=0.1):
        """Read MSP response from serial port"""
        if not self.serial or not self.serial.is_open:
            return None
        
        # Clear any existing data in the buffer
        self.serial.reset_input_buffer()
        
        start_time = time.time()
        buffer = bytearray()
        state = 'IDLE'
        message_length = 0
        message_code = 0
        checksum = 0
        
        while (time.time() - start_time) < timeout:
            if self.serial.in_waiting > 0:
                byte = ord(self.serial.read(1))
                
                if state == 'IDLE' and byte == 36:  # $ character
                    state = 'HEADER_START'
                    
                elif state == 'HEADER_START' and byte == 77:  # M character
                    state = 'HEADER_M'
                    
                elif state == 'HEADER_M' and byte == 62:  # > character (response)
                    state = 'HEADER_ARROW'
                    
                elif state == 'HEADER_ARROW':
                    message_length = byte
                    checksum = byte
                    state = 'PAYLOAD_SIZE'
                    
                elif state == 'PAYLOAD_SIZE':
                    message_code = byte
                    checksum ^= byte
                    state = 'PAYLOAD_CODE'
                    
                elif state == 'PAYLOAD_CODE':
                    buffer.append(byte)
                    checksum ^= byte
                    
                    if len(buffer) == message_length:
                        state = 'CHECKSUM'
                        
                elif state == 'CHECKSUM':
                    if checksum == byte:
                        # Valid message received
                        return self.parse_msp_data(message_code, buffer)
                    else:
                        print(f"Checksum error: calculated={checksum}, received={byte}")
                        return None
            else:
                # No data available, sleep briefly
                time.sleep(0.001)
        
        # Timeout occurred
        return None

    def parse_msp_data(self, msg_id, data):
        """Parse MSP response data into meaningful values"""
        if msg_id == self.MSP_ATTITUDE:
            if len(data) >= 6:
                # Convert bytes to values (roll, pitch, yaw)
                roll = self._bytes_to_int16(data[0:2]) / 10.0
                pitch = self._bytes_to_int16(data[2:4]) / 10.0
                yaw = self._bytes_to_int16(data[4:6])
                return {'roll': roll, 'pitch': pitch, 'yaw': yaw}
        
        elif msg_id == self.MSP_ANALOG:
            if len(data) >= 3:
                battery = data[0] / 10.0  # Convert to volts
                current = data[1] if len(data) > 1 else 0
                rssi = data[2] if len(data) > 2 else 0
                return {'battery': battery, 'current': current, 'rssi': rssi}
        
        elif msg_id == self.MSP_ALTITUDE:
            if len(data) >= 4:
                altitude = self._bytes_to_int32(data[0:4]) / 100.0  # Convert to meters
                return {'altitude': altitude}
        
        # Return raw data if we don't know how to parse it
        return list(data)

    def _bytes_to_int16(self, byte_array):
        """Convert 2 bytes to signed 16-bit integer (little endian)"""
        return struct.unpack('<h', bytes(byte_array))[0]
    
    def _bytes_to_int32(self, byte_array):
        """Convert 4 bytes to signed 32-bit integer (little endian)"""
        return struct.unpack('<i', bytes(byte_array))[0]

    def start_updates(self):
        """Start flight data update thread."""
        if self.update_thread is not None and self.update_thread.is_alive():
            print("Update thread is already running")
            return False
            
        self.running = True
        self.update_thread = threading.Thread(target=self._update_loop)
        self.update_thread.daemon = True
        self.update_thread.start()
        print("Started flight controller update thread")
        return True
    
    def stop_updates(self):
        """Stop flight data updates."""
        if self.running:
            self.running = False
            if self.update_thread:
                self.update_thread.join(timeout=2.0)
                if self.update_thread.is_alive():
                    print("Warning: Update thread did not terminate properly")
                else:
                    print("Stopped flight controller update thread")
                self.update_thread = None
    
    def _update_loop(self):
        """Background update loop - runs in separate thread"""
        print("MSP update loop started")
        
        while self.running:
            if self.simulation:
                self._simulate_telemetry()
            else:
                self._read_real_telemetry()
            
            # Update at 20Hz (50ms intervals)
            time.sleep(0.05)
        
        print("MSP update loop stopped")
    
    def _read_real_telemetry(self):
        """Read telemetry from real flight controller using MSP"""
        try:
            # Use command_lock to prevent interference with manual commands
            with self.command_lock:
                # Request attitude data
                if self.send_msp_request(self.MSP_ATTITUDE):
                    attitude_data = self.read_msp_response(self.MSP_ATTITUDE, timeout=0.05)
                    if attitude_data:
                        with self.data_lock:
                            self.telemetry['attitude'].update(attitude_data)
                
                # Request analog data (battery, RSSI)
                if self.send_msp_request(self.MSP_ANALOG):
                    analog_data = self.read_msp_response(self.MSP_ANALOG, timeout=0.05)
                    if analog_data:
                        with self.data_lock:
                            self.telemetry['battery'] = analog_data.get('battery', self.telemetry['battery'])
                            self.telemetry['rssi'] = analog_data.get('rssi', self.telemetry['rssi'])
                
                # Request altitude data
                if self.send_msp_request(self.MSP_ALTITUDE):
                    altitude_data = self.read_msp_response(self.MSP_ALTITUDE, timeout=0.05)
                    if altitude_data:
                        with self.data_lock:
                            self.telemetry['altitude'] = altitude_data.get('altitude', self.telemetry['altitude'])
                
                # Update timestamp
                with self.data_lock:
                    self.telemetry['timestamp'] = time.time()
                    
        except Exception as e:
            print(f"Error reading telemetry: {e}")
    
    def _simulate_telemetry(self):
        """Generate simulated flight data."""
        t = time.time()
        
        # Thread-safe update of telemetry data
        with self.data_lock:
            # Create somewhat realistic looking data
            self.telemetry['attitude']['roll'] = 15 * math.sin(t / 2)
            self.telemetry['attitude']['pitch'] = 10 * math.cos(t / 3)
            self.telemetry['attitude']['yaw'] = (self.telemetry['attitude']['yaw'] + 1) % 360
            
            self.telemetry['altitude'] = 50 + 10 * math.sin(t / 10)
            self.telemetry['battery'] = 12.6 - (t % 100) / 100
            
            # Toggle armed status every 10 seconds for demonstration
            if int(t) % 10 == 0:
                self.telemetry['armed'] = not self.telemetry['armed']
            
            # Add some realism to GPS
            self.telemetry['gps']['fix'] = True
            self.telemetry['gps']['satellites'] = 8 + random.randint(0, 4)
            self.telemetry['gps']['latitude'] += random.uniform(-0.0001, 0.0001)
            self.telemetry['gps']['longitude'] += random.uniform(-0.0001, 0.0001)
            
            # Calculate airspeed and ground speed
            wind_factor = random.uniform(0.8, 1.2)
            self.telemetry['airspeed'] = 5 + 2 * math.sin(t / 5) 
            self.telemetry['ground_speed'] = self.telemetry['airspeed'] * wind_factor
            
            # Home distance increases slightly over time
            self.telemetry['home_distance'] = 10 + t % 100
            
            # Update timestamp
            self.telemetry['timestamp'] = t
    
    def get_telemetry(self):
        """Get current telemetry data (thread-safe)."""
        with self.data_lock:
            return copy.deepcopy(self.telemetry)
    
    def send_command(self, command_type, data=None):
        """Send a command to the flight controller (thread-safe)."""
        if self.simulation:
            print(f"Simulated command: {command_type}, data: {data}")
            return True
        else:
            # Use command_lock to prevent interference with background telemetry reading
            with self.command_lock:
                if command_type == "set_rc_channels":
                    # Example: Set RC channel values
                    # data should be a list of channel values (1000-2000)
                    if data and isinstance(data, list):
                        rc_data = []
                        for value in data:
                            rc_data.append(value & 0xFF)        # Low byte
                            rc_data.append((value >> 8) & 0xFF) # High byte
                        return self.send_msp_command(self.MSP_SET_RAW_RC, rc_data)
                
                # Add more command types as needed
                print(f"Unknown command type: {command_type}")
                return False

    def arm_motors(self):
        """Arm the motors (example control command)"""
        # This is an example - actual implementation depends on your flight controller
        # Usually involves setting specific RC channel values
        rc_channels = [1500, 1500, 1000, 2000, 1000, 1000, 1000, 1000]  # Example values
        return self.send_command("set_rc_channels", rc_channels)
    
    def disarm_motors(self):
        """Disarm the motors (example control command)"""
        # This is an example - actual implementation depends on your flight controller
        rc_channels = [1500, 1500, 1000, 1000, 1000, 1000, 1000, 1000]  # Example values
        return self.send_command("set_rc_channels", rc_channels)