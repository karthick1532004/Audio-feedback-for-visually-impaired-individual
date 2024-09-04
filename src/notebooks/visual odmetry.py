import cv2
import numpy as np

# Initialize parameters
MIN_NUM_FEATURES = 2000  # Minimum number of features to track
SCALE_FACTOR = 0.5  # Scale factor for resizing the input frames
FEATURE_PARAMS = dict(maxCorners=1000, qualityLevel=0.3, minDistance=7, blockSize=7)
LK_PARAMS = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

def preprocess_image(image):
    # Resize the image for faster processing
    height, width = image.shape[:2]
    new_height, new_width = int(height * SCALE_FACTOR), int(width * SCALE_FACTOR)
    resized_image = cv2.resize(image, (new_width, new_height))
    return resized_image

def feature_tracking(prev_img, curr_img, prev_points):
    curr_points, status, _ = cv2.calcOpticalFlowPyrLK(prev_img, curr_img, prev_points, None, **LK_PARAMS)
    status = status[:, 0]  # Flatten the status array to 1D

    # Filter points and status based on the valid status
    prev_points = prev_points[status == 1]
    curr_points = curr_points[status == 1]

    # Reinitialize feature points if the number of points becomes too small
    if len(prev_points) < MIN_NUM_FEATURES:
        prev_points = cv2.goodFeaturesToTrack(curr_img, mask=None, **FEATURE_PARAMS)

    return curr_points, prev_points


def main():
    # Initialize video capture
    video_path = "D:/FIVETH SEM/system thinking/test video.mp4"
    cap = cv2.VideoCapture(video_path)

    # Initialize the first frame and feature points
    ret, prev_frame = cap.read()
    if not ret:
        print("Error: Unable to read the video.")
        return

    prev_frame = preprocess_image(prev_frame)
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    prev_points = cv2.goodFeaturesToTrack(prev_gray, mask=None, **FEATURE_PARAMS)

    # Initialize camera pose
    trajectory = []
    pose = np.eye(4)

    while True:
        ret, curr_frame = cap.read()
        if not ret:
            break

        # Preprocess current frame
        curr_frame = preprocess_image(curr_frame)
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)

        # Track feature points using Lucas-Kanade
        curr_points, prev_points = feature_tracking(prev_gray, curr_gray, prev_points)

        # Estimate essential matrix and recover pose
        essential_matrix, _ = cv2.findEssentialMat(curr_points, prev_points)
        _, rotation, translation, _ = cv2.recoverPose(essential_matrix, curr_points, prev_points)

         # Update camera pose
        pose[:3, :3] = rotation.dot(pose[:3, :3])
        pose[:3, 3] += pose[:3, :3].dot(np.reshape(translation, (3, 1)))[:, 0]


        # Store camera trajectory
        trajectory.append(pose[:3, 3])

        # Draw the camera trajectory on the current frame
        for i in range(1, len(trajectory)):
            x1, y1 = map(int, trajectory[i - 1][:2])
            x2, y2 = map(int, trajectory[i][:2])
            cv2.line(curr_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)


        # Display the current frame
        cv2.imshow("Monocular Visual Odometry", curr_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # Update previous frame and points for the next iteration
        prev_gray = curr_gray.copy()
        prev_points = curr_points.copy()

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
