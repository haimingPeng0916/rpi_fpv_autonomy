import time
import signal
import sys
from fpv.msp_client import MSPClient

# Initialize components
msp_client = MSPClient('/dev/ttyAMA0', baudrate=115200)

# Handle graceful shutdown
def signal_handler(sig, frame):
    print("Shutting down...")
    msp_client.stop()
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# Start the system
def main():
    # Start the MSP client
    if not msp_client.start():
        print("Failed to start MSP client")
        return 1
    
    print("FPV autonomy system running...")
    
    # Main loop
    while True:
        # Get flight data
        attitude = msp_client.get_data(MSPClient.MSP_ATTITUDE)
        
        # Get IMU data
        imu = msp_client.get_data(MSPClient.MSP_RAW_IMU)
        
        if attitude:
            print(f"Roll: {attitude.get('roll', 0):.1f}°, "
                  f"Pitch: {attitude.get('pitch', 0):.1f}°, "
                  f"Yaw: {attitude.get('yaw', 0)}°")
        if imu:
            accel = imu.get('accelerometer', {})
            gyro = imu.get('gyroscope', {})
            print(f"Accel: X={accel.get('x', 0)}, Y={accel.get('y', 0)}, Z={accel.get('z', 0)}")
            print(f"Gyro: X={gyro.get('x', 0)}, Y={gyro.get('y', 0)}, Z={gyro.get('z', 0)}")
        
        # Request new data for next iteration
        msp_client.request_data(MSPClient.MSP_ATTITUDE)
        time.sleep(0.05)  # Small delay between requests
        msp_client.request_data(MSPClient.MSP_RAW_IMU)
        
        # Sleep to control loop rate
        time.sleep(0.1)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())