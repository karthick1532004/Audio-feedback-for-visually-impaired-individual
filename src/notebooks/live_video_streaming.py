import cv2

# Initialize the camera capture
live_video = cv2.VideoCapture("http://192.168.222.182:8080/video")

while True:
    # Read a frame from the camera
    ret, frame = live_video.read()

    if not ret:
        break

    # Assuming the depth camera frame can be accessed using a function like `get_depth_frame`
    depth_frame = get_depth_frame()

    # Process the depth frame to obtain the depth map
    depth_map = process_depth_frame(depth_frame)

    # Display the color frame and depth map
    cv2.imshow('Color Frame', frame)
    cv2.imshow('Depth Map', depth_map)

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) == ord('q'):
        break

# Release video capture and close windows
live_video.release()
cv2.destroyAllWindows()
