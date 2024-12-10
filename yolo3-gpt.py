import cv2
import gradio as gr
import numpy as np
from PIL import Image, ImageDraw
import re
import io
import base64
import requests

# YOLOv3 setup
weights_path = "yolo3/yolov3.weights"
config_path = "yolo3/yolov3.cfg"
names_path = "yolo3/coco.names"

# Load YOLO model
net = cv2.dnn.readNet(weights_path, config_path)

# Load YOLO labels
with open(names_path, 'r', encoding='utf-8') as file:
    labels = file.read().strip().split('\n')

# Azure GPT and Speech setup
OPENAI_ENDPOINT = "Your_OpenAI_Endpoint"
OPENAI_API_KEY = "Your_OpenAI_API_Key"
DEPLOYMENT_NAME = "Your_OpenAI_Model"

SPEECH_ENDPOINT = "Your_Speech_Endpoint"
SPEECH_API_KEY = "Your_Speech_API_Key"

def request_gpt(image_array):
    """Send a request to Azure GPT with the detected image."""
    endpoint = f"{OPENAI_ENDPOINT}/openai/deployments/{DEPLOYMENT_NAME}/chat/completions?api-version=2024-08-01-preview"
    headers = {"Content-Type": "application/json", "api-key": OPENAI_API_KEY}

    # Convert image to base64
    image = Image.fromarray(image_array)
    buffered_io = io.BytesIO()
    image.save(buffered_io, format='PNG')
    base64_image = base64.b64encode(buffered_io.getvalue()).decode('utf-8')

    # GPT request payload
    message_list = [
        {"role": "system", "content": "You are a bot analyzing objects detected in the image."},
        {"role": "user", "content": f"Analyze this image: {base64_image}"}
    ]

    payload = {"messages": message_list, "temperature": 0.7, "top_p": 0.95, "max_tokens": 1500}
    response = requests.post(endpoint, headers=headers, json=payload)

    if response.status_code == 200:
        response_json = response.json()
        return response_json['choices'][0]['message']['content']
    else:
        return f"Error: {response.status_code}, {response.text}"

def request_tts(text):
    """Convert text to speech using Azure Speech service."""
    headers = {
        "Ocp-Apim-Subscription-Key": SPEECH_API_KEY,
        "Content-Type": "application/ssml+xml",
        "X-Microsoft-OutputFormat": "audio-16khz-128kbitrate-mono-mp3"
    }
    payload = f"""
    <speak version='1.0' xml:lang='en-US'>
        <voice xml:lang='en-US' xml:gender='Female' name='en-US-AvaMultilingualNeural'>{text}</voice>
    </speak>
    """
    response = requests.post(SPEECH_ENDPOINT, headers=headers, data=payload)

    if response.status_code == 200:
        file_name = "response_audio.mp3"
        with open(file_name, "wb") as audio_file:
            audio_file.write(response.content)
        return file_name
    else:
        return None

def detect_object(image):
    """Detect objects in an image using YOLOv3."""
    height, width = image.shape[:2]
    blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    detections = net.forward(output_layers)
    result_image = Image.fromarray(image.copy())
    draw = ImageDraw.Draw(result_image)

    for detection in detections:
        for det in detection:
            scores = det[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                box = det[:4] * np.array([width, height, width, height])
                center_x, center_y, w, h = box.astype("int")
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                draw.rectangle([x, y, x + w, y + h], outline="red", width=2)
                draw.text((x, y), f"{labels[class_id]}: {confidence:.2f}", fill="yellow")
    return result_image

def click_capture(image):
    return image

def click_send_gpt(image, chat_history):
    if image is None:
        chat_history.append(["User", "No image captured for analysis."])
        return chat_history

    if isinstance(image, Image.Image):
        image = np.array(image)

    gpt_response = request_gpt(image)
    chat_history.append(["Assistant", gpt_response])
    return chat_history

def change_chatbot(chat_history):
    if chat_history and chat_history[-1][1]:
        tts_file = request_tts(chat_history[-1][1])
        return tts_file
    return None

def stream_webcam(image):
    if image is None:
        return None
    return detect_object(image)

# Gradio app
with gr.Blocks() as demo:
    with gr.Row():
        webcam_input = gr.Image(label="Webcam", source="webcam", type="numpy")
        detected_image = gr.Image(label="Detected Image", type="pil")
        captured_image = gr.Image(label="Captured Image", type="pil")

    with gr.Row():
        capture_btn = gr.Button("Capture")
        send_gpt_btn = gr.Button("Send to GPT")

    chatbot = gr.Chatbot(label="Chat").style(color_map=("blue", "green"))
    audio_output = gr.Audio(label="GPT Audio", type="filepath", autoplay=True)

    webcam_input.stream(fn=stream_webcam, inputs=webcam_input, outputs=detected_image)
    capture_btn.click(fn=click_capture, inputs=detected_image, outputs=captured_image)
    send_gpt_btn.click(fn=click_send_gpt, inputs=[captured_image, chatbot], outputs=chatbot)
    chatbot.change(fn=change_chatbot, inputs=chatbot, outputs=audio_output)

demo.launch()
