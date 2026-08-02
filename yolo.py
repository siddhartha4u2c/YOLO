import gradio as gr
from ultralytics import YOLO
import cv2
import numpy as np

# Load YOLO model once when the app starts
model = YOLO("yolo26n.pt")


def detect_objects(image):
    """
    Runs YOLO detection on an uploaded image or webcam frame.
    """
    if image is None:
        return None

    # Convert RGB (Gradio) -> BGR (OpenCV)
    frame = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Run prediction
    results = model.predict(
        source=frame,
        imgsz=640,
        conf=0.4,
        verbose=False
    )

    # Draw bounding boxes
    output = results[0].plot()

    # Convert BGR -> RGB for Gradio
    output = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)

    return output


demo = gr.Interface(
    fn=detect_objects,
    inputs=gr.Image(type="numpy", label="Upload Image or Use Webcam"),
    outputs=gr.Image(type="numpy", label="Detection Result"),
    title="YOLO Live Object Detection",
    description="Upload an image or use your webcam for real-time object detection.",
    allow_flagging="never"
)
import os

demo.launch(
    server_name="0.0.0.0",
    server_port=int(os.environ.get("PORT", 7860))
)
