# FemSecure

**FemSecure** is an AI-powered Women Safety Analytics platform designed to enhance safety and awareness in public spaces. This system uses advanced computer vision and AI techniques to detect, analyze, and act upon scenarios that could pose risks to women's safety.

## Features

1. **Person Detection**
   - Detects individuals in real-time using a YOLO-based object detection model.

2. **Gender Classification**
   - Classifies detected individuals as male or female using a VGG19-based gender detection model.

3. **Gender Distribution Analysis**
   - Calculates and displays the ratio of men to women in a given area.

4. **SOS Gesture Recognition**
   - Recognizes specific hand gestures (e.g., SOS signals) using MediaPipe and OpenCV to detect distress signals.

## Technologies Used

- **YOLOv8**: For real-time object and face detection.
- **VGG19**: For gender classification.
- **OpenCV**: For video processing and image analysis.
- **MediaPipe**: For hand gesture recognition.
- **Python**: Primary programming language for development.

