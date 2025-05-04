import streamlit as st
import cv2
from ultralytics import YOLO
import time
from datetime import datetime
import collections

# --- Configuration ---
YOLO_MODEL = 'yolov8n.pt'
DEFAULT_RTSP_URL = "rtsp://rtsp.passtrack.mx/test"
MAX_LOG_ENTRIES = 50 # Max number of log entries to keep
# -------------------

st.set_page_config(layout="wide", page_title="RTSP YOLOv8 Streamlit")

@st.cache_resource # Cache the model loading
def load_yolo_model(model_path):
    try:
        model = YOLO(model_path)
        print("YOLO model loaded successfully using cache.")
        return model
    except Exception as e:
        st.error(f"Error loading YOLO model: {e}")
        return None

model = load_yolo_model(YOLO_MODEL)

st.title("RTSP Stream Object Detection with YOLOv8")

# --- Session State Initialization ---
if 'rtsp_url' not in st.session_state:
    st.session_state.rtsp_url = DEFAULT_RTSP_URL
if 'cap' not in st.session_state:
    st.session_state.cap = None
if 'run_stream' not in st.session_state:
    st.session_state.run_stream = False
if 'log_history' not in st.session_state:
    # Use a deque to automatically limit the size
    st.session_state.log_history = collections.deque(maxlen=MAX_LOG_ENTRIES)
if 'frame_count' not in st.session_state: # Add frame counter
    st.session_state.frame_count = 0
# ------------------------------------

def add_log(message):
    """Adds a timestamped message to the log history."""
    timestamp = datetime.now().strftime("%H:%M:%S")
    st.session_state.log_history.appendleft(f"{timestamp} - {message}") # Add to the beginning

def stop_stream():
    if st.session_state.cap is not None:
        st.session_state.cap.release()
        st.session_state.cap = None
        add_log("Video stream stopped.")
    st.session_state.run_stream = False

def start_stream():
    stop_stream() # Ensure any previous stream is stopped
    url = st.session_state.rtsp_url
    if not url:
        st.warning("Please enter an RTSP URL.")
        return
        
    try:
        add_log(f"Attempting to connect to: {url}")
        st.session_state.cap = cv2.VideoCapture(url)
        if not st.session_state.cap.isOpened():
            st.error(f"Error: Could not open RTSP stream at {url}")
            add_log(f"Failed to connect to: {url}")
            st.session_state.cap = None
            st.session_state.run_stream = False
        else:
            add_log("RTSP stream opened successfully.")
            st.session_state.run_stream = True
            # Force a rerun to start processing immediately
            # Note: Newer versions use st.rerun()
            st.rerun() # Use the current standard method
    except Exception as e:
        st.error(f"Exception occurred while opening stream: {e}")
        add_log(f"Exception connecting to stream: {e}")
        st.session_state.cap = None
        st.session_state.run_stream = False

# --- UI Layout ---
st.text_input("RTSP URL:", key='rtsp_url')

col_buttons = st.columns(2)
with col_buttons[0]:
    st.button("Start Stream", on_click=start_stream, type="primary")
with col_buttons[1]:
    st.button("Stop Stream", on_click=stop_stream)

st.divider()
st.subheader("RTSP Stream Output")
# Removed column/placeholder creation here - moved inside if/else
# placeholder_col1, placeholder_col2 = st.columns([2, 3])
# log_placeholder = placeholder_col2.empty()
# frame_count_placeholder = placeholder_col1.caption(f"Processed frames: {st.session_state.frame_count}")

# --- Main Processing Loop and Display Area ---
if st.session_state.run_stream and st.session_state.cap is not None and model is not None:
    # Create columns *only* when running
    col1, col2 = st.columns([2, 3]) 
    
    success, frame = st.session_state.cap.read()
    
    if success:
        st.session_state.frame_count += 1 
        col1.caption(f"Processed frames: {st.session_state.frame_count}") # Draw caption directly
        
        try:
            # Process the frame with YOLOv8
            results = model.predict(frame, verbose=False, stream=False) 
            result = results[0]
            class_names = result.names
            detected_objects = []

            # Draw bounding boxes and collect info for log
            for box in result.boxes.data:
                x1, y1, x2, y2 = map(int, box[:4])
                confidence = float(box[4])
                class_id = int(box[5])
                label = class_names[class_id]
                
                detected_objects.append(f"{label} ({confidence:.2f})")
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{label} {confidence:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Log detected objects for this frame (if any)
            if detected_objects:
                add_log(f"Detected: {', '.join(detected_objects)}")

            # Convert frame to RGB for Streamlit display
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Draw image directly in the column
            col1.image(frame_rgb, caption="Live Stream")

            # Draw log display directly in the column
            col2.text_area("Log", "\n".join(st.session_state.log_history), height=400, key="log_active") # Use key to prevent state loss on rerun

        except Exception as e:
            add_log(f"Error during prediction/display: {e}")
            st.error(f"Error during prediction/display: {e}")
            # Show the raw frame even if processing failed
            col1.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), caption="Error during processing") 
            # Draw log display directly in the column
            col2.text_area("Log", "\n".join(st.session_state.log_history), height=400, key="log_error")
            
    else: # Frame read failed
        add_log("Failed to read frame from stream. Stopping.")
        st.warning("Failed to read frame. Stream might have ended or there was an error.")
        # Display message in column 1 and current log in column 2
        col1.warning("Stream stopped or failed to read frame.")
        col2.text_area("Log", "\n".join(st.session_state.log_history), height=400, key="log_stopped")
        stop_stream()
        st.rerun()
        
    # --- Auto-refresh mechanism --- 
    time.sleep(0.01) 
    st.rerun()
    # -----------------------------
else:
    # Create columns and Show the default state when the stream is not running
    col1, col2 = st.columns([2, 3])
    col1.image("https://streamlit.io/images/brand/streamlit-mark-color.png", caption="Stream paused")
    col1.caption(f"Processed frames: {st.session_state.frame_count}") # Show last count
    # Draw log display directly in the column
    col2.text_area("Log", "\n".join(st.session_state.log_history), height=400, key="log_paused")

# Removed placeholder updates from here

# Keep the script running if the stream is active
# The rerun mechanism handles frame updates when active. 