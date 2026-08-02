import gradio as gr
from ultralytics import YOLO
import cv2
import numpy as np
import os

# Load model once
model = YOLO("yolo26n.pt")


def detect_frame(frame):
    """
    Receives webcam frame continuously
    and returns detected frame.
    """

    if frame is None:
        return None

    # Gradio gives RGB
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    # YOLO prediction
    results = model.predict(
        source=frame,
        imgsz=640,
        conf=0.4,
        verbose=False
    )

    # Draw detections
    annotated = results[0].plot()

    # Convert back to RGB
    annotated = cv2.cvtColor(
        annotated,
        cv2.COLOR_BGR2RGB
    )

    return annotated


with gr.Blocks(title="YOLO26 Live Detection") as demo:

    gr.Markdown(
        """
        # YOLO26 Real-Time Object Detection
        
        Allow webcam access and see live detections.
        """
    )

    webcam = gr.Image(
        sources=["webcam"],
        streaming=True,
        type="numpy",
        label="Webcam"
    )

    output = gr.Image(
        label="Detection Output"
    )

    webcam.stream(
        fn=detect_frame,
        inputs=webcam,
        outputs=output,
        time_limit=60
    )


if __name__ == "__main__":

    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860))
    )
