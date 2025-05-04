import gradio as gr
import cv2
# No longer need numpy explicitly here if not used elsewhere after changes
# import numpy as np
from ultralytics import YOLO

# Initialize YOLOv8 model (e.g., yolov8n.pt)
# The model will be downloaded automatically on the first run
model = YOLO('yolov8n.pt')

# No longer need YOLOv3 paths or loading class names manually
# weights_path = './yolo3/yolov3.weights'
# config_path = './yolo3/yolov3.cfg'
# names_path = './yolo3/coco.names'
# net = cv2.dnn.readNet(weights_path, config_path)
# with open(names_path, 'r') as f:
#     labels = f.read().strip().split('\n')

def detect_objects(image):
    # YOLOv8 prediction
    # Setting verbose=False to avoid printing detailed logs for each prediction
    results = model.predict(image, verbose=False)

    # Get the first result object
    result = results[0]
    
    # Get class names from the model
    class_names = result.names

    # Iterate through detected boxes
    for box in result.boxes.data:
        # Extract coordinates, confidence, and class ID
        x1, y1, x2, y2 = map(int, box[:4]) # Bounding box coordinates (xyxy format)
        confidence = float(box[4]) # Confidence score
        class_id = int(box[5]) # Class ID
        
        # Filter detections by confidence threshold (optional, YOLOv8 predict has its own confidence threshold)
        # if confidence > 0.5: # You can adjust this threshold if needed

        label = class_names[class_id]

        # Draw the rectangle
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # Display the label
        cv2.putText(image, f"{label} {confidence:.2f}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    
    # No need for explicit NMS, YOLOv8 handles it
    # boxes = []
    # confidences = []
    # class_ids = []
    # ... (YOLOv3 processing logic removed) ...
    # indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    # ... (YOLOv3 drawing logic removed) ...

    return image

# Define the Gradio Blocks interface
with gr.Blocks() as demo:
    gr.Markdown("## Object Detection with YOLOv8 and Gradio") # Added a title
    with gr.Row():
        # Changed input to Image for webcam or file upload flexibility
        # webcam_input = gr.Video(label="Webcam Input") 
        input_source = gr.Image(label="Input Image/Webcam", sources=["upload", "webcam"], type="numpy")
        output_image = gr.Image(label="Detection Output", type="numpy")

    # Define update function - now directly takes the numpy image
    def update(frame_rgb):
        # Ensure the frame is not None
        if frame_rgb is None:
            return None
        # No need to decode video, input is now numpy array
        # cap = cv2.VideoCapture(video)
        # success, frame = cap.read()
        # if not success:
        #     cap.release()
        #     return None
        # cap.release()
        # frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # Input should be RGB if coming from Gradio Image component
        
        # Perform detection
        processed_image = detect_objects(frame_rgb)
        return processed_image

    # Link input source to the update function
    # Use stream() for webcam or change() for upload
    input_source.stream(fn=update, inputs=input_source, outputs=output_image)
    input_source.change(fn=update, inputs=input_source, outputs=output_image)


# Launch the Gradio app
# share=True allows creating a public link (optional)
demo.launch(share=True)
