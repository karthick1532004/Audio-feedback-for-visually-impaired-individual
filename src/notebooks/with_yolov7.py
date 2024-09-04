import cv2
import numpy as np
from path_planning_new import binary_image
from center_line_1 import *
from getting_trajectory import * 
import sys

sys.path.append(r'D:\FIVETH SEM\system thinking\system_thinking_1\system_thinking\src\notebooks\yolov7-main')
from detect import detect  # Import the detect function from detect.py

centerpoints = []

center = (0 + int(640 / 2), 0 + int(480 / 2))

# Open the video file for video capture
video_path = r"D:\FIVETH SEM\system thinking\test_video.mp4"
video_capture = cv2.VideoCapture(video_path)

if not video_capture.isOpened():
    print("Error: Cannot open the video file.")
    exit()

while True:
    # Read a frame from the video
    ret, frame = video_capture.read()

    if not ret:
        print("End of video.")
        break
    frame = cv2.resize(frame, (640, 480))
    # frame = cv2.rotate(frame, cv2.ROTATE_180)
    masked_image, mask = binary_image(frame)
    centerpoints, output_image = centre(masked_image, frame)

    if len(centerpoints) > 5:
        # Call YOLOv7 object detection
        detect_results = detect(
            weights='path/to/your/model.weights',  # Specify the path to your YOLOv7 model weights
            source=frame,  # Pass the current frame as the source
            img_size=640,  # Set the image size
            conf_thres=0.25,  # Set confidence threshold
            iou_thres=0.45,  # Set IOU threshold for NMS
            device='0',  # Specify the device (0 for GPU, 'cpu' for CPU)
            view_img=False,  # Do not display detection results here
            save_txt=False,  # Do not save results to a text file
            save_conf=False,  # Do not save confidences in labels
            nosave=True,  # Do not save images/videos
            classes=None,  # Specify classes to filter by (if needed)
            agnostic_nms=False,  # Set class-agnostic NMS
            augment=False,  # Do not use augmented inference
            update=False,  # Do not update the model
            project='runs/detect',  # Specify the project directory
            name='exp',  # Specify the project name
            exist_ok=False,  # Do not increment the project name
            no_trace=False  # Do not disable model tracing
        )

        # Extract detected objects and their bounding boxes from the results
        detections, bbox = detect_results[0], detect_results[1]

        # Further process the detected objects and update 'img' if needed

    # Display the processed frame
    cv2.imshow('Video Frame', output_image)
    cv2.imshow('orginal Frame', mask)
    cv2.imshow('binary image', masked_image)

    # Exit the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all windows
video_capture.release()
cv2.destroyAllWindows()