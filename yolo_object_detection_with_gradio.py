import gradio as gr
import cv2
import numpy as np

# Paths to YOLOv3 model files
weights_path = './yolo3/yolov3.weights'
config_path = './yolo3/yolov3.cfg'
names_path = './yolo3/coco.names'

# Load YOLOv3 model
net = cv2.dnn.readNet(weights_path, config_path)

# Load class names
with open(names_path, 'r') as f:
    labels = f.read().strip().split('\n')

def detect_objects(image):
    height, width = image.shape[:2]
    # Create a blob and perform forward propagation
    blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    detections = net.forward(output_layers)
    
    boxes = []
    confidences = []
    class_ids = []

    # Process the detections
    for output in detections:
        for detection in output:
            scores = detection[5:]  # Extract the confidence scores
            class_id = np.argmax(scores)  # Get the class with the highest score
            confidence = scores[class_id]  # Get the confidence value
            if confidence > 0.5:  # Threshold to filter weak detections
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Calculate the box coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    
    # Apply Non-Maximum Suppression to remove overlapping boxes
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # Draw bounding boxes and labels on the image
    if len(indices) > 0:
        for i in indices.flatten():
            x, y, w, h = boxes[i]
            label = str(labels[class_ids[i]])
            confidence = confidences[i]

            # Draw the rectangle
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            # Display the label
            cv2.putText(image, f"{label} {confidence:.2f}", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return image

# Define the Gradio Blocks interface
with gr.Blocks() as demo:
    with gr.Row():
        webcam_input = gr.Video(label="Webcam Input")
        output_image = gr.Image(label="Detection Output")

    # Define update function
    def update(video):
        # Decode the video frame from the file path
        cap = cv2.VideoCapture(video)
        success, frame = cap.read()
        if not success:
            cap.release()
            return None
        cap.release()

        # Ensure the frame is in RGB format
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return detect_objects(frame_rgb)

    # Link webcam input to the update function
    webcam_input.change(fn=update, inputs=webcam_input, outputs=output_image)

# Launch the Gradio app
demo.launch()
