#!/usr/bin/env python3
import os
import json
import asyncio
import logging
import time
import signal
import sys
from aiohttp import web
import socketio

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add the project root to Python's path
# This is crucial for finding the fpv package
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
logger.info(f"Adding project root to path: {project_root}")
sys.path.insert(0, project_root)

# Create Socket.IO server
sio = socketio.AsyncServer(cors_allowed_origins='*', async_mode='aiohttp')
app = web.Application()
sio.attach(app)

# Now try to import the MSPClient
try:
    from fpv.msp_client import MSPClient
    logger.info("Successfully imported MSPClient")
except ImportError as e:
    logger.error(f"Failed to import MSPClient: {e}")
    # Print the current Python path for debugging
    logger.error(f"Current Python path: {sys.path}")
    raise

# Global variables
msp_client = None
should_run = True

# Format MSP data function
def format_msp_data():
    """Format MSP data for the dashboard"""
    try:
        # Get data using the same approach as in your main.py
        attitude = msp_client.get_data(MSPClient.MSP_ATTITUDE)
        imu = msp_client.get_data(MSPClient.MSP_RAW_IMU)
        
        # Format data for the dashboard
        data = {
            'timestamp': time.strftime('%H:%M:%S'),
            'altitude': 0,  # If you have altitude data, add it here
            'orientation': {
                'roll': attitude.get('roll', 0) if attitude else 0,
                'pitch': attitude.get('pitch', 0) if attitude else 0,
                'yaw': attitude.get('yaw', 0) if attitude else 0
            },
            'accel': {
                'x': imu.get('accelerometer', {}).get('x', 0) if imu else 0,
                'y': imu.get('accelerometer', {}).get('y', 0) if imu else 0,
                'z': imu.get('accelerometer', {}).get('z', 0) if imu else 0
            }
        }
        
        # Request new data for next iteration (as you do in main.py)
        msp_client.request_data(MSPClient.MSP_ATTITUDE)
        msp_client.request_data(MSPClient.MSP_RAW_IMU)
        
        return data
    except Exception as e:
        logger.error(f"Error formatting MSP data: {str(e)}")
        return None

# Task to send data updates to clients
async def send_data_updates():
    """Send data updates to connected clients"""
    global should_run
    
    while should_run:
        try:
            # Get formatted data
            data = format_msp_data()
            
            if data:
                # Send to all connected clients
                await sio.emit('fpv_data', data)
            
            # Update rate: 10 times per second
            await asyncio.sleep(0.1)
            
        except Exception as e:
            logger.error(f"Error sending data updates: {str(e)}")
            await asyncio.sleep(1)  # Wait before retrying

# Socket.IO connection event
@sio.event
async def connect(sid, environ):
    logger.info(f"Client connected: {sid}")

# Socket.IO disconnection event
@sio.event
async def disconnect(sid):
    logger.info(f"Client disconnected: {sid}")

# Handle graceful shutdown
async def shutdown(app):
    """Shutdown the server gracefully"""
    global should_run, msp_client
    
    logger.info("Shutting down server...")
    should_run = False
    
    # Stop the MSP client if it's running
    if msp_client:
        msp_client.stop()
    
    # Close all socket connections
    for sid in list(sio.sockets.keys()):
        await sio.disconnect(sid)

# Main function
async def start_server(port='/dev/ttyAMA0', baud=115200, http_port=8080):
    """Start the web server"""
    global msp_client, should_run
    
    # Initialize the MSP client
    msp_client = MSPClient(port, baudrate=baud)
    
    # Start the MSP client
    if not msp_client.start():
        logger.error("Failed to start MSP client")
        return 1
    
    # Setup shutdown handler
    app.on_shutdown.append(shutdown)
    
    # Set up routes for serving static files
    static_path = os.path.join(os.path.dirname(__file__), 'static')
    app.router.add_static('/', path=static_path, name='static')
    
    # Start the data update task
    asyncio.create_task(send_data_updates())
    
    # Start the server
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, '0.0.0.0', http_port)
    await site.start()
    logger.info(f"Server started at http://0.0.0.0:{http_port}")
    
    # Keep the server running
    while should_run:
        await asyncio.sleep(1)
    
    # Clean up
    await runner.cleanup()
    logger.info("Server shutdown complete")

# Entry point
def main():
    """Entry point for the web server"""
    # Signal handlers for graceful shutdown
    def signal_handler(sig, frame):
        global should_run
        logger.info("Received shutdown signal, stopping server...")
        should_run = False
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Parse command line arguments (you could add argparse here)
    http_port = 8080
    port = '/dev/ttyAMA0'  # Default to Raspberry Pi UART
    baud = 115200
    
    # Start the server
    try:
        asyncio.run(start_server(port, baud, http_port))
    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    main()