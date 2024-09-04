import cv2
import numpy as np
import pyttsx3
from path_planning_new import binary_image
from center_line_1 import *
#from getting_trajectory import * 
from stering_1 import *
import serial  # Import the serial module

# Initialize the text-to-speech engine
engine = pyttsx3.init()

centerpoints = []
center = (0 + int(640 / 2), 0 + int(480 / 2))

# Open the video file for video capture
video_path = r"/media/boobalan/New Volume/college/FIVETH SEM/system thinking/system_thinking_1/system_thinking/test_video.mp4"
video_capture = cv2.VideoCapture(video_path)

if not video_capture.isOpened():
    print("Error: Cannot open the video file.")
    exit()
    
# Initialize a serial connection to your Arduino
#arduino_port = 'COM8'  # Replace 'COMX' with your Arduino's COM port
#baud_rate = 9600  # Make sure this matches the baud rate in your Arduino code
#ser = serial.Serial(arduino_port, baud_rate)

while True:
    # Read a frame from the video
    ret, frame = video_capture.read()
    
    if not ret:
        print("End of video.")
        break
    
    frame = cv2.resize(frame, (640, 480))
    #frame = cv2.rotate(frame, cv2.ROTATE_180)
    masked_image, mask = binary_image(frame)
    centerpoints, output_image = centre(masked_image, frame)
    
    if len(centerpoints) > 5:
        img, angle, delta = getSteeringCommand(center, centerpoints, output_image)
        
        if angle < 90:
            direction = "move left"
        elif angle > 90:
            direction = "move right"
        else:
            direction = "go straight"
        
        # Speak the direction
        engine.say(direction)
        engine.runAndWait()
        
        # Send the delta value to Arduino
        #ser.write(f"{angle}\n".encode())
        print(angle)
        cv2.imshow("delta_OUT", img)

    # Display the processed frame
    cv2.imshow('Video Frame', output_image)
    cv2.imshow('Original Frame', mask)
    cv2.imshow('Binary Image', masked_image)
    
    # Exit the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all windows
video_capture.release()
cv2.destroyAllWindows()
