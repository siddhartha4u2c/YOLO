import gradio as gr
from ultralytics import YOLO
import cv2
import os

model = YOLO("yolo26n.pt")


def detect_frame(frame):
    if frame is None:
        return None

    # Gradio gives RGB image
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    results = model.predict(
        source=frame,
        imgsz=320,
        conf=0.4,
        verbose=False
    )

    annotated = results[0].plot()

    # Back to RGB for Gradio
    annotated = cv2.cvtColor(
        annotated,
        cv2.COLOR_BGR2RGB
    )

    return annotated


with gr.Blocks() as demo:

    gr.Markdown("# YOLO26 Live Webcam Detection")

    with gr.Row():

        webcam = gr.Image(
            sources=["webcam"],
            streaming=True,
            type="numpy",
            label="Webcam"
        )

        output = gr.Image(
            label="Detection"
        )


    webcam.stream(
        fn=detect_frame,
        inputs=webcam,
        outputs=output
    )


demo.launch(
    server_name="0.0.0.0",
    server_port=int(os.environ.get("PORT", 7860))
)
