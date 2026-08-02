import gradio as gr
from ultralytics import YOLO
import cv2
import os
import time

model = YOLO("yolo26n.pt")


def webcam_detection(video):
    """
    Receives webcam stream and yields processed frames.
    """

    cap = cv2.VideoCapture(video)

    while cap.isOpened():

        success, frame = cap.read()

        if not success:
            break

        # YOLO inference
        results = model(
            frame,
            imgsz=320,
            conf=0.4,
            verbose=False
        )

        # Draw boxes
        annotated = results[0].plot()

        # Convert BGR -> RGB
        annotated = cv2.cvtColor(
            annotated,
            cv2.COLOR_BGR2RGB
        )

        yield annotated

        # Control FPS
        time.sleep(0.03)

    cap.release()


with gr.Blocks() as demo:

    gr.Markdown(
        """
        # YOLO26 Live Webcam Detection
        """
    )

    input_video = gr.Video(
        sources=["webcam"],
        streaming=True,
        label="Camera"
    )

    output_video = gr.Image(
        label="YOLO Detection"
    )

    input_video.stream(
        fn=webcam_detection,
        inputs=input_video,
        outputs=output_video
    )


demo.launch(
    server_name="0.0.0.0",
    server_port=int(os.environ.get("PORT",7860))
)
