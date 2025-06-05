# test_fixed_camera.py - Test the updated camera with DICT_6X6_50
from camera import Camera
import time
import cv2

def test_updated_camera():
    """Test the updated camera system."""
    print("=== Testing Updated Camera System ===")
    
    # Create camera instance
    camera = Camera()
    
    try:
        # Start camera
        print("1. Starting camera...")
        success = camera.start()
        if not success:
            print("❌ Failed to start camera")
            return
        
        print("✅ Camera started successfully")
        print(f"Detection mode: {camera.detection_mode}")
        
        # Check if DICT_6X6_50 is loaded
        if hasattr(camera, 'aruco_dicts') and hasattr(camera, 'dict_names'):
            print(f"Available dictionaries: {camera.dict_names}")
            if "6x6_50" in camera.dict_names:
                print("✅ DICT_6X6_50 is available!")
            else:
                print("❌ DICT_6X6_50 not found in available dictionaries")
        
        # Test detection
        print("\n2. Testing detection...")
        print("Hold up your ArUco marker (ID 1, 6x6_50 dictionary)")
        
        for i in range(10):
            print(f"\nTest {i+1}/10:")
            
            success, frame, center, bbox = camera.get_frame()
            
            if success:
                print(f"  🎯 DETECTION SUCCESS!")
                print(f"     Center: {center}")
                print(f"     Bbox: {bbox}")
                
                # Check if we have detection details
                if camera.last_detection:
                    detection = camera.last_detection
                    if 'id' in detection:
                        print(f"     Marker ID: {detection['id']}")
                    if 'distance' in detection:
                        print(f"     Distance: {detection['distance']}cm")
                    if 'all_detections' in detection:
                        print(f"     Total markers: {len(detection['all_detections'])}")
                
                # Save successful detection
                cv2.imwrite(f"camera_success_test_{i}.jpg", frame)
                print(f"     Saved: camera_success_test_{i}.jpg")
                
            else:
                print("  ❌ No detection")
                # Save frame for debugging
                cv2.imwrite(f"camera_no_detection_{i}.jpg", frame)
            
            time.sleep(1)
    
    finally:
        print("\n3. Stopping camera...")
        camera.stop()
        print("✅ Camera stopped")

def test_specific_6x6_detection():
    """Test detection specifically with DICT_6X6_50."""
    print("\n=== Testing DICT_6X6_50 Detection ===")
    
    try:
        import cv2.aruco as aruco
        from picamera2 import Picamera2
        
        # Initialize camera
        camera = Picamera2()
        config = camera.create_preview_configuration(main={"size": (640, 480)})
        camera.configure(config)
        camera.start()
        time.sleep(1)
        
        # Initialize ArUco with the working dictionary
        aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_50)
        parameters = aruco.DetectorParameters_create()
        
        print("Camera ready. Testing DICT_6X6_50 detection...")
        
        for i in range(5):
            print(f"Test {i+1}/5:")
            
            # Capture frame
            frame = camera.capture_array()
            
            # Handle format
            if len(frame.shape) == 2:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            elif frame.shape[2] == 4:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            
            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect markers
            corners, ids, rejected = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
            
            if ids is not None and len(ids) > 0:
                print(f"  🎯 SUCCESS! Detected {len(ids)} markers")
                for marker_id in ids:
                    print(f"     ID: {marker_id[0]}")
                
                # Draw detection
                result = frame.copy()
                aruco.drawDetectedMarkers(result, corners, ids)
                
                # Add ID labels
                for j, marker_id in enumerate(ids):
                    if len(corners) > j:
                        center = cv2.mean(corners[j][0], axis=0)[:2]
                        center = tuple(map(int, center))
                        cv2.putText(result, f"ID:{marker_id[0]}", 
                                   (center[0]-20, center[1]-20), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                
                cv2.imwrite(f"direct_6x6_success_{i}.jpg", result)
                print(f"     Saved: direct_6x6_success_{i}.jpg")
            else:
                print("  ❌ No detection")
                cv2.imwrite(f"direct_6x6_frame_{i}.jpg", frame)
            
            time.sleep(1)
        
        camera.stop()
        camera.close()
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("🧪 Testing Fixed Camera System")
    print("Your marker: ID 1, DICT_6X6_50")
    
    # Test the updated camera class
    test_updated_camera()
    
    # Test direct detection
    test_specific_6x6_detection()
    
    print("\n📁 Check these files:")
    print("  - camera_success_test_*.jpg: Successful detections from Camera class")
    print("  - direct_6x6_success_*.jpg: Direct DICT_6X6_50 detections")
    
    print("\n🎯 If this works, replace your current camera.py with the updated version!")