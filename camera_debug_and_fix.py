# camera_debug_and_fix.py - Debug camera issues and test the fix
import subprocess
import time

def check_camera_status():
    """Check if camera is already in use."""
    print("=== Camera Status Check ===")
    
    try:
        # Check if any process is using the camera
        result = subprocess.run(['sudo', 'lsof', '/dev/video*'], 
                              capture_output=True, text=True)
        if result.stdout:
            print("📹 Camera is in use by:")
            print(result.stdout)
        else:
            print("📹 Camera appears to be free")
            
        # Check available cameras
        from picamera2 import Picamera2
        camera_info = Picamera2.global_camera_info()
        print(f"📷 Available cameras: {len(camera_info)}")
        for i, info in enumerate(camera_info):
            print(f"  Camera {i}: {info}")
            
    except Exception as e:
        print(f"❌ Error checking camera status: {e}")

def test_simple_camera():
    """Test basic camera functionality."""
    print("\n=== Simple Camera Test ===")
    
    try:
        from picamera2 import Picamera2
        import cv2
        
        # Get camera info first
        camera_info = Picamera2.global_camera_info()
        if not camera_info:
            print("❌ No cameras available")
            return False
        
        print(f"Using camera 0: {camera_info[0]}")
        
        # Initialize camera with explicit camera number
        camera = Picamera2(0)  # Explicitly use camera 0
        
        # Simple configuration
        config = camera.create_preview_configuration(main={"size": (640, 480)})
        print(f"Configuration: {config}")
        
        camera.configure(config)
        camera.start()
        
        print("✅ Camera started successfully")
        time.sleep(2)
        
        # Test capture
        frame = camera.capture_array()
        print(f"✅ Frame captured: {frame.shape}")
        
        # Test ArUco detection
        if len(frame.shape) == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        elif frame.shape[2] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        
        # Save test frame
        cv2.imwrite("simple_camera_test.jpg", frame)
        print("✅ Saved test frame: simple_camera_test.jpg")
        
        camera.stop()
        camera.close()
        print("✅ Camera stopped")
        
        return True
        
    except Exception as e:
        print(f"❌ Simple camera test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_aruco_on_saved_frame():
    """Test ArUco detection on the saved frame."""
    print("\n=== ArUco Test on Saved Frame ===")
    
    try:
        import cv2
        import cv2.aruco as aruco
        
        # Load the frame we just saved
        frame = cv2.imread("simple_camera_test.jpg")
        if frame is None:
            print("❌ Could not load saved frame")
            return
        
        print(f"✅ Loaded frame: {frame.shape}")
        
        # Test with DICT_6X6_50 (the one that works)
        aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_50)
        parameters = aruco.DetectorParameters_create()
        
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect markers
        corners, ids, rejected = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
        
        if ids is not None and len(ids) > 0:
            print(f"🎯 SUCCESS! Detected {len(ids)} markers")
            for marker_id in ids:
                print(f"   ID: {marker_id[0]}")
            
            # Draw detection
            result = frame.copy()
            aruco.drawDetectedMarkers(result, corners, ids)
            
            # Add labels
            for i, marker_id in enumerate(ids):
                if len(corners) > i:
                    center = corners[i][0].mean(axis=0).astype(int)
                    cv2.putText(result, f"ID:{marker_id[0]}", 
                               tuple(center), cv2.FONT_HERSHEY_SIMPLEX, 
                               0.8, (0, 255, 0), 2)
            
            cv2.imwrite("aruco_detection_success.jpg", result)
            print("✅ Saved: aruco_detection_success.jpg")
            
        else:
            print("❌ No ArUco markers detected")
            
    except Exception as e:
        print(f"❌ ArUco test failed: {e}")
        import traceback
        traceback.print_exc()

def kill_camera_processes():
    """Kill any processes that might be using the camera."""
    print("\n=== Killing Camera Processes ===")
    
    try:
        # Kill any Python processes that might be using the camera
        subprocess.run(['pkill', '-f', 'python.*app.py'], capture_output=True)
        subprocess.run(['pkill', '-f', 'libcamera'], capture_output=True)
        time.sleep(1)
        print("✅ Killed potential camera processes")
        
    except Exception as e:
        print(f"⚠️ Error killing processes: {e}")

def update_main_camera_code():
    """Show how to update the main camera.py with the working dictionary."""
    print("\n=== Camera Code Update Instructions ===")
    
    print("""
🔧 To fix your main application:

1. STOP your Flask app (Ctrl+C if it's running)

2. Update your camera.py file:
   Find this line in _init_aruco_detector():
   
   dict_types = [
       (aruco.DICT_4X4_50, "4x4_50"),
       ...
   ]
   
   Change it to:
   
   dict_types = [
       (aruco.DICT_6X6_50, "6x6_50"),      # ← ADD THIS FIRST!
       (aruco.DICT_4X4_50, "4x4_50"),
       (aruco.DICT_6X6_250, "6x6_250"),
       (aruco.DICT_5X5_100, "5x5_100"),
   ]

3. Save the file and restart your Flask app:
   python3 app.py

4. Your marker (ID 1) should now be detected! 🎯
""")

if __name__ == "__main__":
    print("🚨 Camera Debugging Tool")
    
    # Check what's using the camera
    check_camera_status()
    
    # Kill any competing processes
    kill_camera_processes()
    
    # Wait a moment
    time.sleep(2)
    
    # Test basic camera functionality
    if test_simple_camera():
        # Test ArUco on the captured frame
        test_aruco_on_saved_frame()
        
        # Show update instructions
        update_main_camera_code()
    else:
        print("\n❌ Camera test failed. Possible issues:")
        print("  1. Your Flask app might still be running (kill it)")
        print("  2. Another process is using the camera")
        print("  3. Camera hardware issue")
        print("\nTry: pkill -f python3 && pkill -f libcamera")