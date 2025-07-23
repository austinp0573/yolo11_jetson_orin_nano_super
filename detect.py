import cv2
from ultralytics import YOLO

def gstreamer_pipeline(
    sensor_id=0,
    capture_width=1920,
    capture_height=1080,
    display_width=960,
    display_height=540,
    framerate=30,
    flip_method=0,
):
    return (
        "nvarguscamerasrc sensor-id=%d ! "
        "video/x-raw(memory:NVMM), width=(int)%d, height=(int)%d, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink"
        % (
            sensor_id,
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,
        )
    )

def show_camera_with_yolo():
    window_title = "YOLOv11 Detection"

    try:
        model = YOLO('yolo11n.pt')
        print("YOLO model 'yolo11n.pt' loaded successfully.")
    except Exception as e:
        print(f"Error loading YOLO model: {e}")
        print("Please ensure the 'yolo11n.pt' file is accessible.")
        return

    video_capture = cv2.VideoCapture(gstreamer_pipeline(flip_method=0), cv2.CAP_GSTREAMER)

    if not video_capture.isOpened():
        print("Error: Unable to open camera.")
        return
    
    print("Camera opened successfully. Starting detection stream...")
    try:
        cv2.namedWindow(window_title, cv2.WINDOW_AUTOSIZE)
        
        while True:
            ret_val, frame = video_capture.read()
            if not ret_val:
                print("Warning: Failed to grab frame.")
                break

            results = model(frame, verbose=False) 

            annotated_frame = results[0].plot()

            if cv2.getWindowProperty(window_title, cv2.WND_PROP_AUTOSIZE) >= 0:
                cv2.imshow(window_title, annotated_frame)
            else:
                print("Window closed by user.")
                break

            keyCode = cv2.waitKey(10) & 0xFF
            if keyCode == 27 or keyCode == ord('q'):
                print("Exit key pressed. Shutting down.")
                break
    finally:
        video_capture.release()
        cv2.destroyAllWindows()
        print("Resources released.")

if __name__ == "__main__":
    show_camera_with_yolo()