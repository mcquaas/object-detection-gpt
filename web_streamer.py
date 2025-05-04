from flask import Flask, render_template_string, Response
import cv2
from ultralytics import YOLO
import time
import threading

# --- Configuration ---
RTSP_URL = "rtsp://rtspstream:wehcE7kcENhSZhhJQGu5k@zephyr.rtsp.stream/traffic"
YOLO_MODEL = 'yolov8n.pt' # Or choose a different model like yolov8s.pt, yolov8m.pt
FRAME_SKIP = 2 # Process every Nth frame to potentially reduce load
RECONNECT_DELAY_SEC = 5 # Delay before attempting to reconnect to RTSP stream
# -------------------

app = Flask(__name__)

# Global variables
model = None
output_frame = None
lock = threading.Lock() # To control access to output_frame
stream_active = True # Flag to control the streaming thread

def load_model():
    """Loads the YOLO model."""
    global model
    try:
        print(f"Loading YOLO model: {YOLO_MODEL}...")
        model = YOLO(YOLO_MODEL)
        print("YOLO model loaded successfully.")
    except Exception as e:
        print(f"Error loading YOLO model: {e}")
        # Handle error appropriately, maybe exit or retry
        exit()

def video_stream_capture():
    """Reads frames from RTSP, processes with YOLO, and updates the global output_frame."""
    global output_frame, lock, stream_active
    
    cap = None
    last_connection_attempt = 0

    while stream_active:
        try:
            # Attempt to connect or reconnect if needed
            if cap is None or not cap.isOpened():
                if time.time() - last_connection_attempt < RECONNECT_DELAY_SEC:
                    time.sleep(1) # Wait before retrying connection
                    continue
                
                print(f"Attempting to open RTSP stream: {RTSP_URL}")
                last_connection_attempt = time.time()
                cap = cv2.VideoCapture(RTSP_URL)
                if not cap.isOpened():
                    print(f"Error: Could not open RTSP stream. Retrying in {RECONNECT_DELAY_SEC} seconds...")
                    cap = None # Ensure cap is None so it retries
                    time.sleep(RECONNECT_DELAY_SEC)
                    continue
                else:
                    print("RTSP stream opened successfully.")

            # Read frame
            success, frame = cap.read()
            if not success:
                print("Warning: Failed to read frame from stream. Attempting to reconnect...")
                cap.release()
                cap = None # Force reconnect attempt
                time.sleep(1) # Brief pause before trying again
                continue

            # Process the frame with YOLOv8
            try:
                results = model.predict(frame, verbose=False, stream=False) # Process single frame
                result = results[0]
                class_names = result.names

                # Draw bounding boxes
                for box in result.boxes.data:
                    x1, y1, x2, y2 = map(int, box[:4])
                    confidence = float(box[4])
                    class_id = int(box[5])
                    label = class_names[class_id]
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"{label} {confidence:.2f}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                                
                processed_frame = frame

            except Exception as e:
                print(f"Error during YOLO prediction: {e}")
                processed_frame = frame # Show raw frame on error

            # Encode frame as JPEG and update the global variable
            (flag, encoded_image) = cv2.imencode(".jpg", processed_frame)
            if not flag:
                continue # Skip if encoding failed

            with lock:
                output_frame = encoded_image.tobytes()
                
        except Exception as e:
            print(f"Error in video stream loop: {e}")
            if cap is not None:
                cap.release()
            cap = None
            time.sleep(RECONNECT_DELAY_SEC) # Wait before trying to reconnect
            
    # Cleanup when loop exits
    if cap is not None:
        cap.release()
    print("Video stream capture thread stopped.")

def generate_frames():
    """Generator function to yield frames for the MJPEG stream."""
    global output_frame, lock
    while True:
        with lock:
            if output_frame is None:
                # Send a placeholder or wait if no frame is available yet
                # For simplicity, we'll just wait briefly
                time.sleep(0.1) 
                continue
            
            frame_bytes = output_frame
            
        # Yield the frame in the required format for MJPEG
        yield (
            b'--frame\r\n' + 
            b'Content-Type: image/jpeg\r\n\r\n' + 
            frame_bytes + 
            b'\r\n'
        )
        
        # Control frame rate slightly to prevent overwhelming the client/network
        time.sleep(0.03) # Adjust as needed (~30 FPS target)


@app.route('/')
def index():
    """Serves the main HTML page."""
    # Simple HTML page with an img tag pointing to the video feed
    return render_template_string("""
        <html>
        <head><title>YOLOv8 RTSP Stream</title></head>
        <body>
            <h1>YOLOv8 Object Detection from RTSP Stream</h1>
            <img src="{{ url_for('video_feed') }}" width="800"> 
            <p>Attempting to stream from: """ + RTSP_URL + """</p>
        </body>
        </html>
    """)

@app.route('/video_feed')
def video_feed():
    """Route that serves the MJPEG stream."""
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    # Load the model first
    load_model()
    
    # Start the background thread for capturing and processing video
    print("Starting video stream capture thread...")
    video_thread = threading.Thread(target=video_stream_capture)
    video_thread.daemon = True # Allows thread to exit when main app exits
    video_thread.start()
    
    # Run the Flask app
    print("Starting Flask server on http://0.0.0.0:5000")
    # Use host='0.0.0.0' to make it accessible on the network
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True) 

    # Signal the video thread to stop when Flask exits (though daemon=True often suffices)
    stream_active = False
    video_thread.join(timeout=2) # Wait briefly for the thread to finish
    print("Flask server stopped.") 