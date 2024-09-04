import cv2
from path_planning_new import binary_image
from lines import *
from getting_trajectory import *

centerpoints = []

center =(0+int(640 / 2),0+int(480 / 2))
# Open the video file for video capture
video_path = r"/media/boobalan/New Volume/college/FIVETH SEM/system thinking/system_thinking_1/system_thinking/test_video.mp4"
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

    # Process the frame
    frame = cv2.resize(frame, (640, 480))
    masked_image, mask = binary_image(frame)
    
    array = np.array(masked_image)
    #img = cv2.imread(r"C:\Users\hhara\Downloads\WhatsApp Image 2023-08-16 at 12.43.31 PM.jpeg",1)
    img = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    
    centerpoints, img = getLines(array, img)
    
    cv2.imshow("line_OUT", img)
    
    if len(centerpoints) > 5:
            img = getSteeringCommand(center, centerpoints, img)
            

    cv2.imshow("OUT", img)
    #cv2.imshow('out', output_image)
    cv2.imshow('segm', mask)
    cv2.imshow('binary', masked_image)
    
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()

