import cv2
from ultralytics import YOLO

def open_camera():
    for index in [0, 1, 2, 3]:
        print(f"Trying camera index {index}...")
        camera = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if not camera.isOpened():
            camera.release()
            continue
        # Request a common USB-camera format.
        camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        camera.set(cv2.CAP_PROP_FPS, 30)
        success, frame = camera.read()
        if success and frame is not None and frame.size > 0:
            print(f"Camera {index} opened successfully.")
            return camera
        camera.release()
    return None

def main():
    model = YOLO("yolo26n.pt")
    camera = open_camera()
    if camera is None:
        print("Error: No working camera was found.")
        return
    print("Live object detection started.")
    print("Press Q or ESC to quit.")
    try:
        while True:
            try:
                success, frame = camera.read()
            except cv2.error as error:
                print("OpenCV camera error:", error)
                break
            if not success or frame is None or frame.size == 0:
                print("The camera returned an invalid frame.")
                break
            results = model.predict(
                source=frame,
                imgsz=640,
                conf=0.4,
                verbose=False,
            )
            output = results[0].plot()
            cv2.imshow("YOLO26 Live Detection", output)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()