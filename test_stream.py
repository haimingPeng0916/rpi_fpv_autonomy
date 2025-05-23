from flask import Flask, Response, render_template
import cv2
import time

app = Flask(__name__)

def generate_frames():
    """Generate frames directly from camera."""
    # Try different camera indices if 0 doesn't work
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("ERROR: Could not open camera!")
        return
    
    print("Camera opened successfully")
    
    try:
        while True:
            # Capture frame
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                time.sleep(0.1)
                continue
            
            # Print frame shape occasionally
            if int(time.time()) % 5 == 0:
                print(f"Frame captured: {frame.shape}")
            
            # Convert to JPEG
            ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if not ret:
                continue
            
            # Yield frame
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
    except Exception as e:
        print(f"Error in generate_frames: {e}")
    finally:
        cap.release()
        print("Camera released")

@app.route('/')
def index():
    """Render simple test page."""
    return """
    <html>
      <head><title>Camera Test</title></head>
      <body>
        <h1>Camera Test</h1>
        <img src="/video_feed" width="640" height="480" />
      </body>
    </html>
    """

@app.route('/video_feed')
def video_feed():
    """Video streaming route."""
    print("Video feed requested!")
    return Response(generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    print("Starting test server...")
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)