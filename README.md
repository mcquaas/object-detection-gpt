# Object Detection with YOLOv3 and GPT Integration

This project demonstrates how to use the YOLOv3 model for object detection, integrated with a Gradio interface for real-time webcam input. Additionally, it includes GPT-assisted object detection functionalities for enhanced analysis and context generation.

## Features
  ![Gradio Interface Example](demo/demo_1.png)
  ![YOLOv3 Detection Example](demo/demo_2.png)
  ![GPT Interface Example](demo/demo_3.png)
  
- **YOLOv3 Integration**: Uses the YOLOv3 model for accurate object detection.
- **Gradio Interface**: Provides an easy-to-use web interface to display detection results.
- **Real-Time Webcam Support**: Processes video input from a webcam and displays detection results live.
- **GPT-Assisted Analysis**: Enhances object detection with contextual insights and analysis using GPT.

## Requirements

Before running the project, ensure you have the following installed:

- Python 3.8 or higher
- OpenCV (`cv2`)
- Gradio (`gradio`)
- Numpy (`numpy`)
- OpenAI GPT API (`openai`)

You can install the dependencies with:

```bash
pip install -r requirements.txt
```

## File Structure

```bash
├── yolo3/
│   ├── yolov3.weights    # YOLOv3 pre-trained weights file
│   ├── yolov3.cfg        # YOLOv3 configuration file
│   ├── coco.names        # File containing class names
├── demo/
│   ├── demo_1.png
│   ├── demo_2.png    
├── yolo_object_detection_with_gradio.py  # Main script for Gradio-based detection
├── yolo3-gpt.py           # Script for GPT-assisted detection
├── requirements.txt  
├── README.md              # Project documentation
```

> **Note**: You need to download the YOLOv3 weights and configuration files from [Darknet YOLO](https://pjreddie.com/darknet/yolo/) and place them in the `yolo3/` folder.

## Usage

### Gradio-based Object Detection
1. Clone the repository:
  ```bash
  git clone https://github.com/seonokkim/object-detection-gradio.git
  cd object-detection-gradio
  ```
2. Ensure all dependencies are installed:
  ```bash
  pip install -r requirements.txt  
  ```
3. Run the application:
  ```bash
  python yolo_object_detection_with_gradio.py
  ```
4. Open the Gradio interface in your browser (it will provide a local or public link).

### GPT-Assisted Object Detection
1. Clone the repository (if not already done):
  ```bash
  git clone https://github.com/seonokkim/object-detection-gradio.git
  cd object-detection-gradio
  ```
2. Ensure all dependencies are installed:
  ```bash
  pip install -r requirements.txt  
  ```
3. Set up your OpenAI API key:
  ```bash
  export OPENAI_API_KEY="your_api_key"
  ```
4. Run the application:
  ```bash
  python yolo3-gpt.py
  ```

## Acknowledgments

- [YOLOv3](https://arxiv.org/pdf/1804.02767) - You Only Look Once, for real-time object detection.
- [Gradio](https://gradio.app/) - For creating a simple web interface for ML applications.
- [OpenAI GPT](https://openai.com/) - For enhanced context generation and analysis.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
