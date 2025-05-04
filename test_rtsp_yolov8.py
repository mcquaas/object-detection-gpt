import cv2
from ultralytics import YOLO
import time

# Initialize YOLOv8 model
try:
    model = YOLO('yolov8n.pt')
    print("YOLOv8 model loaded successfully.")
except Exception as e:
    print(f"Error loading YOLOv8 model: {e}")
    exit()

# RTSP stream URL provided by the user
rtsp_url = "rtsp://rtspstream:wehcE7kcENhSZhhJQGu5k@zephyr.rtsp.stream/traffic"

print(f"Attempting to open RTSP stream: {rtsp_url}")

# Attempt to open the RTSP stream
cap = cv2.VideoCapture(rtsp_url)

# Check if the stream was opened successfully
if not cap.isOpened():
    print(f"Error: Could not open RTSP stream.")
    exit()
else:
    print("RTSP stream opened successfully.")

# --- Optional: Get stream properties (might not work with all streams) ---
# try:
#     fps = cap.get(cv2.CAP_PROP_FPS)
#     width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#     print(f"Stream properties: {width}x{height} @ {fps:.2f} FPS")
# except Exception as e:
#     print(f"Could not get stream properties: {e}")
# ------------------------------------------------------------------------

frame_count = 0
start_time = time.time()

while True:
    # Read a frame from the stream
    success, frame = cap.read()

    # Check if a frame was read successfully
    if not success:
        print("Error: Failed to read frame from stream or stream ended.")
        break

    # Perform YOLOv8 prediction
    try:
        results = model.predict(frame, verbose=False) # verbose=False reduces console output
        result = results[0] # Get the first result
        
        # Get class names
        class_names = result.names

        # Draw bounding boxes on the frame
        for box in result.boxes.data:
            x1, y1, x2, y2 = map(int, box[:4])
            confidence = float(box[4])
            class_id = int(box[5])
            label = class_names[class_id]

            # Draw rectangle and label
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} {confidence:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    except Exception as e:
        print(f"Error during YOLOv8 prediction or drawing: {e}")
        # Continue to the next frame or break depending on the error
        continue 

    # --- Calculate and display FPS ---
    frame_count += 1
    elapsed_time = time.time() - start_time
    if elapsed_time > 1: # Update FPS display every second
        fps = frame_count / elapsed_time
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        # Reset counter and timer
        frame_count = 0
        start_time = time.time()
    # ---------------------------------

    # Display the processed frame
    # Note: cv2.imshow might not work in a remote environment like SSH without X11 forwarding.
    # If it doesn't work, we might need to save frames to files instead for verification.
    try:
        cv2.imshow("RTSP Stream with YOLOv8 Detection", frame)
    except cv2.error as e:
        if "DISPLAY" in str(e):
            print("Error: cv2.imshow() failed. This usually means you are in an environment without a display (like a headless server or SSH session without X11 forwarding).")
            print("Consider saving frames to files instead for verification.")
        else:
            print(f"cv2.imshow error: {e}")
        break # Exit if display fails

    # Check for 'q' key press to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Exiting...")
        break

# Release the video capture object and destroy windows
cap.release()
try:
    cv2.destroyAllWindows()
    print("Resources released.")
except cv2.error as e:
     # Ignore errors if destroyAllWindows fails (e.g., no window was shown)
     if "NULL object" not in str(e):
        print(f"Error closing OpenCV windows: {e}")

print("Script finished.") 