#!/usr/bin/env python3
import serial
import struct
import time
import threading
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MSPClient:
    """
    Client for communicating with flight controllers using the MultiWii Serial Protocol (MSP).
    This implementation supports basic MSP commands for fetching attitude and IMU data.
    """
    # MSP message indicators
    MSP_HEADER = b'$M'
    MSP_DIRECTION_TO_FC = b'<'  # To flight controller
    MSP_DIRECTION_FROM_FC = b'>'  # From flight controller
    
    # Common MSP command codes
    MSP_ATTITUDE = 108
    MSP_RAW_IMU = 102
    MSP_ALTITUDE = 109
    MSP_STATUS = 101
    
    def __init__(self, port, baudrate=115200, timeout=1.0):
        """
        Initialize the MSP client.
        
        Args:
            port (str): Serial port name (e.g., '/dev/ttyAMA0')
            baudrate (int): Baud rate for serial communication
            timeout (float): Serial read timeout in seconds
        """
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.serial = None
        self.running = False
        self.lock = threading.Lock()
        self.data_cache = {}
        self.reader_thread = None
        
        # For keeping track of in-flight requests
        self.pending_requests = set()
    
    def start(self):
        """
        Start the MSP client and the background reading thread.
        
        Returns:
            bool: True if started successfully, False otherwise
        """
        try:
            # Open serial connection
            self.serial = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                timeout=self.timeout
            )
            
            if not self.serial.is_open:
                self.serial.open()
            
            # Start reader thread
            self.running = True
            self.reader_thread = threading.Thread(target=self._reader_thread_func)
            self.reader_thread.daemon = True
            self.reader_thread.start()
            
            logger.info(f"MSP client started on {self.port} at {self.baudrate} baud")
            return True
        except Exception as e:
            logger.error(f"Failed to start MSP client: {str(e)}")
            self.stop()
            return False
    
    def stop(self):
        """Stop the MSP client and close the serial connection."""
        self.running = False
        
        # Wait for reader thread to terminate
        if self.reader_thread and self.reader_thread.is_alive():
            self.reader_thread.join(timeout=1.0)
        
        # Close serial connection
        if self.serial and self.serial.is_open:
            self.serial.close()
        
        logger.info("MSP client stopped")
    
    def _reader_thread_func(self):
        """Background thread that reads and processes incoming MSP messages."""
        logger.info("Reader thread started")
        
        while self.running:
            try:
                # Look for header
                header = self.serial.read(2)
                if header != self.MSP_HEADER:
                    continue
                
                # Read direction
                direction = self.serial.read(1)
                if direction != self.MSP_DIRECTION_FROM_FC:
                    continue
                
                # Read payload size
                size = self.serial.read(1)
                if not size:
                    continue
                size = struct.unpack('B', size)[0]
                
                # Read command
                cmd = self.serial.read(1)
                if not cmd:
                    continue
                cmd = struct.unpack('B', cmd)[0]
                
                # Read payload
                payload = self.serial.read(size)
                if len(payload) != size:
                    logger.warning(f"Incomplete payload: expected {size}, got {len(payload)}")
                    continue
                
                # Read checksum
                checksum = self.serial.read(1)
                if not checksum:
                    continue
                checksum = struct.unpack('B', checksum)[0]
                
                # Verify checksum
                calculated_checksum = size ^ cmd
                for b in payload:
                    calculated_checksum ^= b
                
                if checksum != calculated_checksum:
                    logger.warning(f"Invalid checksum for command {cmd}")
                    continue
                
                # Process the message
                self._process_message(cmd, payload)
                
            except serial.SerialException as e:
                logger.error(f"Serial error: {str(e)}")
                self.running = False
                break
            except Exception as e:
                logger.error(f"Error in reader thread: {str(e)}")
                time.sleep(0.1)  # Avoid tight error loops
    
    def _process_message(self, cmd, payload):
        """
        Process a received MSP message and store the data.
        
        Args:
            cmd (int): MSP command code
            payload (bytes): Message payload
        """
        try:
            # Process based on command type
            if cmd == self.MSP_ATTITUDE:
                data = self._parse_attitude(payload)
            elif cmd == self.MSP_RAW_IMU:
                data = self._parse_raw_imu(payload)
            elif cmd == self.MSP_ALTITUDE:
                data = self._parse_altitude(payload)
            elif cmd == self.MSP_STATUS:
                data = self._parse_status(payload)
            else:
                # Unknown command, store raw payload
                data = {'raw': list(payload)}
            
            # Store the processed data
            with self.lock:
                self.data_cache[cmd] = data
                # Mark this request as completed
                if cmd in self.pending_requests:
                    self.pending_requests.remove(cmd)
            
        except Exception as e:
            logger.error(f"Error processing message {cmd}: {str(e)}")
    
    def _parse_attitude(self, payload):
        """
        Parse attitude data from payload.
        
        Args:
            payload (bytes): Raw payload
            
        Returns:
            dict: Parsed attitude data (roll, pitch, yaw)
        """
        if len(payload) < 6:
            return None
        
        # Decode values from payload (roll, pitch, yaw)
        roll = struct.unpack('<h', payload[0:2])[0] / 10.0
        pitch = struct.unpack('<h', payload[2:4])[0] / 10.0
        yaw = struct.unpack('<h', payload[4:6])[0]
        
        return {
            'roll': roll,
            'pitch': pitch,
            'yaw': yaw
        }
    
    def _parse_raw_imu(self, payload):
        """
        Parse raw IMU data from payload.
        
        Args:
            payload (bytes): Raw payload
            
        Returns:
            dict: Parsed IMU data (accelerometer, gyroscope, magnetometer)
        """
        if len(payload) < 18:
            return None
        
        # Decode 9 short values (3 accel, 3 gyro, 3 mag)
        accel_x = struct.unpack('<h', payload[0:2])[0]
        accel_y = struct.unpack('<h', payload[2:4])[0]
        accel_z = struct.unpack('<h', payload[4:6])[0]
        
        gyro_x = struct.unpack('<h', payload[6:8])[0]
        gyro_y = struct.unpack('<h', payload[8:10])[0]
        gyro_z = struct.unpack('<h', payload[10:12])[0]
        
        mag_x = struct.unpack('<h', payload[12:14])[0]
        mag_y = struct.unpack('<h', payload[14:16])[0]
        mag_z = struct.unpack('<h', payload[16:18])[0]
        
        # Scale factors vary by board, these are typical values
        accel_scale = 512.0   # For ±2g sensitivity
        gyro_scale = 16.4     # For ±2000°/s sensitivity
        
        return {
            'accelerometer': {
                'x': accel_x / accel_scale,
                'y': accel_y / accel_scale,
                'z': accel_z / accel_scale
            },
            'gyroscope': {
                'x': gyro_x / gyro_scale,
                'y': gyro_y / gyro_scale,
                'z': gyro_z / gyro_scale
            },
            'magnetometer': {
                'x': mag_x,
                'y': mag_y,
                'z': mag_z
            }
        }
    
    def _parse_altitude(self, payload):
        """
        Parse altitude data from payload.
        
        Args:
            payload (bytes): Raw payload
            
        Returns:
            dict: Parsed altitude data
        """
        if len(payload) < 6:
            return None
        
        # Decode values
        altitude = struct.unpack('<i', payload[0:4])[0] / 100.0  # cm to m
        vario = struct.unpack('<h', payload[4:6])[0] / 100.0     # cm/s to m/s
        
        return {
            'altitude': altitude,
            'vario': vario
        }
    
    def _parse_status(self, payload):
        """
        Parse status data from payload.
        
        Args:
            payload (bytes): Raw payload
            
        Returns:
            dict: Parsed status data
        """
        if len(payload) < 10:
            return None
        
        # This is a simplified implementation - actual parsing depends on your FC
        cycle_time = struct.unpack('<h', payload[0:2])[0]
        i2c_errors = struct.unpack('<h', payload[2:4])[0]
        sensors = struct.unpack('<h', payload[4:6])[0]
        
        return {
            'cycle_time': cycle_time,
            'i2c_errors': i2c_errors,
            'sensors': sensors
        }
    
    def request_data(self, cmd):
        """
        Send a request for data to the flight controller.
        
        Args:
            cmd (int): MSP command code
            
        Returns:
            bool: True if request sent successfully, False otherwise
        """
        try:
            with self.lock:
                self.pending_requests.add(cmd)
            
            # Construct MSP message
            msg = bytearray()
            msg.extend(self.MSP_HEADER)
            msg.extend(self.MSP_DIRECTION_TO_FC)
            msg.append(0)  # Zero payload size
            msg.append(cmd)
            
            # Checksum is XOR of size, cmd, and payload (empty in this case)
            checksum = 0 ^ cmd
            msg.append(checksum)
            
            # Send the message
            self.serial.write(msg)
            return True
        except Exception as e:
            logger.error(f"Error requesting data (cmd={cmd}): {str(e)}")
            return False
    
    def get_data(self, cmd, timeout=0.5):
        """
        Get data for a specific command, requesting it if not available.
        
        Args:
            cmd (int): MSP command code
            timeout (float): Maximum time to wait for data
            
        Returns:
            dict: Command data or None if not available
        """
        # Check if data is already available
        with self.lock:
            if cmd in self.data_cache:
                return self.data_cache[cmd]
        
        # Request data
        self.request_data(cmd)
        
        # Wait for data to arrive
        start_time = time.time()
        while time.time() - start_time < timeout:
            with self.lock:
                if cmd in self.data_cache and cmd not in self.pending_requests:
                    return self.data_cache[cmd]
            time.sleep(0.01)
        
        # Timeout reached
        logger.warning(f"Timeout waiting for data (cmd={cmd})")
        return None

# Example usage
if __name__ == "__main__":
    # Create MSP client
    client = MSPClient('/dev/ttyAMA0', baudrate=115200)
    
    # Start client
    if not client.start():
        logger.error("Failed to start MSP client")
        exit(1)
    
    try:
        # Main loop
        while True:
            # Request and get attitude data
            attitude = client.get_data(MSPClient.MSP_ATTITUDE)
            if attitude:
                print(f"Roll: {attitude['roll']:.1f}°, Pitch: {attitude['pitch']:.1f}°, Yaw: {attitude['yaw']}°")
            
            # Request and get IMU data
            imu = client.get_data(MSPClient.MSP_RAW_IMU)
            if imu:
                accel = imu['accelerometer']
                print(f"Accel: X={accel['x']:.2f}, Y={accel['y']:.2f}, Z={accel['z']:.2f}")
            
            # Sleep to control loop rate
            time.sleep(0.1)
    
    except KeyboardInterrupt:
        print("Exiting...")
    finally:
        # Stop client
        client.stop()