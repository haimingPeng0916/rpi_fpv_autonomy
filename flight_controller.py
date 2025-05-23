# flight_controller.py
import time
import threading
import math
import random
import struct
import serial

class FlightController:

    # MSP command constants
    MSP_ATTITUDE = 108  # Get attitude data (roll, pitch, yaw)
    MSP_ALTITUDE = 109  # Get altitude data
    MSP_ANALOG = 110    # Get battery and RSSI data


    def __init__(self, simulation=True, port='/dev/ttyS0', baudrate=115200):
        """Initialize flight controller interface."""
        self.simulation = simulation
        self.port = port 
        self.baudrate = baudrate
        self.serial = None
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

        self.data_lock = threading.Lock()
        self.update_thread = None
        self.running = False
    
    def connect_to_fc(self):
        """Connect to a real flight controller"""
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
            if serial.is_open and self.serial:
                serial.close()
                print(f"Disconnected from {self.port}")
        except Exception as e:
            print(f"Exception while disconecting: {e}")


    # def send_msp_request(self, msg_id): 

    #     if not self.serial or not self.serial.is_open:
    #         return False

        # data_buffer = bytearray([0x24, 0x4D, 0x3C]) #$M
        # size = 0
        # checksum = size ^ msg_id
        # data_buffer.append(size)
        # data_buffer.append(msg_id)
        # data_buffer.append(checksum)

        # # try send the request
        # try:
        #     bytes_written = self.serial.write(data_buffer)
        #     self.serial.flush()
        #     return bytes_written == len(data_buffer)

        # except Exception as e:
        #     print(f"Exception while sending msp request: {e}")
        #     return False


    # def read_msp_response(self, msg_id, timeout=0.1):
    #     """read the msp response on the serial port"""
        
    #     if not self.serial or not self.serial.is_open:
    #         return None
        



    def start_updates(self):
        """Start flight data update thread."""
        if self.update_thread is not None:
            return
            
        self.running = True
        self.update_thread = threading.Thread(target=self._update_loop)
        self.update_thread.daemon = True
        self.update_thread.start()
    
    def stop_updates(self):
        """Stop flight data updates."""
        self.running = False
        if self.update_thread:
            self.update_thread.join(timeout=1.0)
            self.update_thread = None
    
    def _update_loop(self):
        """Background update loop."""
        while self.running:
            if self.simulation:
                self._simulate_telemetry()
            else:
                self._read_telemetry()
            time.sleep(0.05)  # Update at 20Hz
    
    def _simulate_telemetry(self):
        """Generate simulated flight data."""
        t = time.time()
        
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
    
    def _read_telemetry(self):
        """Read telemetry from actual flight controller (override for real implementation)."""
        # Here you would implement actual MSP protocol communication
        # For example:
        # 1. Send MSP_ATTITUDE command
        # 2. Read response
        # 3. Parse data and update self.telemetry
        pass
    
    def get_telemetry(self):
        """Get current telemetry data."""
        return self.telemetry.copy()
    
    def send_command(self, command_type, data=None):
        """Send a command to the flight controller."""
        if self.simulation:
            print(f"Simulated command: {command_type}, data: {data}")
            return True
        else:
            # Here you would implement actual command sending
            pass