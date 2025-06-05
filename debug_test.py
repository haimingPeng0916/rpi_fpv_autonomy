# comprehensive_aruco_debug.py - Debug ArUco detection issues
import cv2
import cv2.aruco as aruco
import numpy as np
import time
from picamera2 import Picamera2

def test_all_dictionaries_on_image(image_path):
    """Test all ArUco dictionaries on a saved image."""
    print(f"\n=== Testing All Dictionaries on {image_path} ===")
    
    try:
        # Load the image
        image = cv2.imread(image_path)
        if image is None:
            print(f"❌ Could not load {image_path}")
            return
        
        print(f"Image loaded: {image.shape}")
        
        # Try all available dictionaries
        dict_types = [
            (aruco.DICT_4X4_50, "DICT_4X4_50"),
            (aruco.DICT_4X4_100, "DICT_4X4_100"),
            (aruco.DICT_4X4_250, "DICT_4X4_250"),
            (aruco.DICT_4X4_1000, "DICT_4X4_1000"),
            (aruco.DICT_5X5_50, "DICT_5X5_50"),
            (aruco.DICT_5X5_100, "DICT_5X5_100"),
            (aruco.DICT_5X5_250, "DICT_5X5_250"),
            (aruco.DICT_5X5_1000, "DICT_5X5_1000"),
            (aruco.DICT_6X6_50, "DICT_6X6_50"),
            (aruco.DICT_6X6_100, "DICT_6X6_100"),
            (aruco.DICT_6X6_250, "DICT_6X6_250"),
            (aruco.DICT_6X6_1000, "DICT_6X6_1000"),
            (aruco.DICT_7X7_50, "DICT_7X7_50"),
            (aruco.DICT_7X7_100, "DICT_7X7_100"),
            (aruco.DICT_7X7_250, "DICT_7X7_250"),
            (aruco.DICT_7X7_1000, "DICT_7X7_1000"),
            (aruco.DICT_ARUCO_ORIGINAL, "DICT_ARUCO_ORIGINAL"),
        ]
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        for dict_type, dict_name in dict_types:
            try:
                # Get dictionary
                aruco_dict = aruco.Dictionary_get(dict_type)
                parameters = aruco.DetectorParameters_create()
                
                # Try different parameter settings
                param_sets = [
                    ("default", {}),
                    ("relaxed", {
                        "adaptiveThreshConstant": 7,
                        "minMarkerPerimeterRate": 0.01,
                        "maxMarkerPerimeterRate": 0.8,
                        "polygonalApproxAccuracyRate": 0.05,
                    }),
                    ("strict", {
                        "adaptiveThreshConstant": 7,
                        "minMarkerPerimeterRate": 0.05,
                        "maxMarkerPerimeterRate": 0.3,
                        "polygonalApproxAccuracyRate": 0.01,
                    })
                ]
                
                for param_name, param_updates in param_sets:
                    params = aruco.DetectorParameters_create()
                    for key, value in param_updates.items():
                        setattr(params, key, value)
                    
                    # Detect markers
                    corners, ids, rejected = aruco.detectMarkers(gray, aruco_dict, parameters=params)
                    
                    if ids is not None and len(ids) > 0:
                        print(f"🎯 SUCCESS: {dict_name} with {param_name} params")
                        print(f"   Found {len(ids)} markers: {[id[0] for id in ids]}")
                        
                        # Draw detection result
                        result = image.copy()
                        aruco.drawDetectedMarkers(result, corners, ids)
                        
                        # Save result
                        result_filename = f"detection_success_{dict_name}_{param_name}.jpg"
                        cv2.imwrite(result_filename, result)
                        print(f"   Saved: {result_filename}")
                        
                        return dict_name, param_name  # Return first successful detection
                    
            except Exception as e:
                print(f"❌ Error with {dict_name}: {e}")
                continue
        
        print("❌ No markers detected with any dictionary")
        return None, None
        
    except Exception as e:
        print(f"❌ Error testing image: {e}")
        return None, None

def enhanced_live_detection():
    """Enhanced live detection with multiple techniques."""
    print("\n=== Enhanced Live Detection ===")
    
    try:
        # Initialize camera
        camera = Picamera2()
        config = camera.create_preview_configuration(main={"size": (640, 480)})
        camera.configure(config)
        camera.start()
        time.sleep(2)  # Longer wait for stabilization
        
        print("Camera started. Hold up your ArUco marker...")
        
        # Try multiple dictionaries simultaneously
        dict_types = [
            (aruco.DICT_4X4_50, "4x4_50"),
            (aruco.DICT_5X5_100, "5x5_100"),
            (aruco.DICT_6X6_250, "6x6_250"),
            (aruco.DICT_ARUCO_ORIGINAL, "original"),
        ]
        
        aruco_dicts = {}
        for dict_type, name in dict_types:
            try:
                aruco_dicts[name] = aruco.Dictionary_get(dict_type)
            except:
                print(f"❌ Could not load {name}")
        
        # Enhanced parameters
        parameters = aruco.DetectorParameters_create()
        parameters.adaptiveThreshConstant = 7
        parameters.adaptiveThreshWinSizeMin = 3
        parameters.adaptiveThreshWinSizeMax = 23
        parameters.adaptiveThreshWinSizeStep = 10
        parameters.minMarkerPerimeterRate = 0.01  # Very relaxed
        parameters.maxMarkerPerimeterRate = 0.8   # Very relaxed
        parameters.polygonalApproxAccuracyRate = 0.05
        parameters.minCornerDistanceRate = 0.01
        parameters.minDistanceToBorder = 1
        parameters.minMarkerDistanceRate = 0.01
        
        for test_num in range(10):
            print(f"\nTest {test_num + 1}/10:")
            
            # Capture frame
            frame = camera.capture_array()
            
            # Handle different frame formats
            if len(frame.shape) == 2:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            elif frame.shape[2] == 4:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            
            # Save raw frame for analysis
            cv2.imwrite(f"debug_frame_{test_num}.jpg", frame)
            
            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Try multiple image enhancement techniques
            enhanced_images = {
                "original": gray,
                "clahe": cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(gray),
                "gaussian": cv2.GaussianBlur(gray, (5, 5), 0),
                "median": cv2.medianBlur(gray, 5),
                "bilateral": cv2.bilateralFilter(gray, 9, 75, 75)
            }
            
            found_any = False
            
            # Try each dictionary with each enhancement
            for dict_name, aruco_dict in aruco_dicts.items():
                for enhance_name, enhanced_gray in enhanced_images.items():
                    try:
                        corners, ids, rejected = aruco.detectMarkers(enhanced_gray, aruco_dict, parameters=parameters)
                        
                        if ids is not None and len(ids) > 0:
                            print(f"   🎯 FOUND with {dict_name} + {enhance_name}!")
                            print(f"      Markers: {[id[0] for id in ids]}")
                            
                            # Draw result
                            result = frame.copy()
                            aruco.drawDetectedMarkers(result, corners, ids)
                            
                            # Add labels
                            for i, marker_id in enumerate(ids):
                                if len(corners) > i:
                                    center = np.mean(corners[i][0], axis=0).astype(int)
                                    cv2.putText(result, f"ID:{marker_id[0]}", tuple(center), 
                                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                            
                            # Save successful detection
                            success_filename = f"success_{test_num}_{dict_name}_{enhance_name}.jpg"
                            cv2.imwrite(success_filename, result)
                            print(f"      Saved: {success_filename}")
                            
                            found_any = True
                            
                    except Exception as e:
                        print(f"      Error with {dict_name}+{enhance_name}: {e}")
                        continue
            
            if not found_any:
                print("   ❌ No markers detected")
            
            time.sleep(1)
        
        camera.stop()
        camera.close()
        print("\n✅ Enhanced detection test complete")
        
    except Exception as e:
        print(f"❌ Enhanced detection failed: {e}")
        import traceback
        traceback.print_exc()

def analyze_marker_manually(image_path):
    """Manually analyze the marker pattern."""
    print(f"\n=== Manual Marker Analysis: {image_path} ===")
    
    try:
        image = cv2.imread(image_path)
        if image is None:
            print(f"Could not load {image_path}")
            return
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Find contours to locate potential markers
        _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        print(f"Found {len(contours)} contours")
        
        # Look for rectangular contours
        for i, contour in enumerate(contours):
            # Approximate contour
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            area = cv2.contourArea(contour)
            
            if len(approx) == 4 and area > 1000:  # Rectangular and large enough
                print(f"Potential marker contour {i}: area={area}, vertices={len(approx)}")
                
                # Draw this contour
                result = image.copy()
                cv2.drawContours(result, [approx], -1, (0, 255, 0), 3)
                cv2.imwrite(f"potential_marker_{i}.jpg", result)
                
                # Extract the marker region
                rect = cv2.boundingRect(approx)
                x, y, w, h = rect
                marker_region = gray[y:y+h, x:x+w]
                
                # Resize to standard size and analyze
                if w > 50 and h > 50:
                    resized = cv2.resize(marker_region, (100, 100))
                    cv2.imwrite(f"marker_region_{i}.jpg", resized)
                    
                    # Try to determine the grid size
                    print(f"   Extracted marker region: {w}x{h} -> saved as marker_region_{i}.jpg")
    
    except Exception as e:
        print(f"Error in manual analysis: {e}")

if __name__ == "__main__":
    print("🔍 Comprehensive ArUco Debug Tool")
    print("This will test all dictionaries and enhancement techniques")
    
    # First, capture some frames for analysis
    print("\n1. Capturing frames for offline analysis...")
    enhanced_live_detection()
    
    # Analyze the first captured frame
    print("\n2. Testing all dictionaries on captured frame...")
    test_all_dictionaries_on_image("debug_frame_0.jpg")
    
    # Manual analysis
    print("\n3. Manual marker analysis...")
    analyze_marker_manually("debug_frame_0.jpg")
    
    print("\n🎯 Debug complete! Check the generated images:")
    print("   - debug_frame_*.jpg: Raw camera frames")
    print("   - success_*.jpg: Successful detections (if any)")
    print("   - detection_success_*.jpg: Dictionary-specific successes")
    print("   - potential_marker_*.jpg: Manually found marker regions")