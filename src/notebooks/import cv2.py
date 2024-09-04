import cv2
import numpy as np
import sys

sys.path.append('D:/FIVETH SEM/system thinking/vision_based_robot_navigation-master/src/')

from system_thinking.src.notebooks.models import net  # Assuming 'net' is the class or function you want to import from 'models.py'
import torch
from torch.autograd import Variable

# Pre-processing and post-processing constants #
CMAP = np.load('D:/FIVETH SEM/system thinking/vision_based_robot_navigation-master/src/cmap_nyud.npy')
DEPTH_COEFF = 5000.  # to convert into meters
HAS_CUDA = torch.cuda.is_available()
IMG_SCALE = 1. / 255
IMG_MEAN = np.array([0.485, 0.456, 0.406]).reshape((1, 1, 3))
IMG_STD = np.array([0.229, 0.224, 0.225]).reshape((1, 1, 3))
MAX_DEPTH = 8.
MIN_DEPTH = 0.
NUM_CLASSES = 40
NUM_TASKS = 2  # segm + depth

def prepare_img(img):
    return (img * IMG_SCALE - IMG_MEAN) / IMG_STD

model = net(num_classes=NUM_CLASSES, num_tasks=NUM_TASKS)
if HAS_CUDA:
    _ = model.cuda()
_ = model.eval()

ckpt = torch.load('D:/FIVETH SEM/system thinking/vision_based_robot_navigation-master/weights/ExpNYUD_joint.ckpt')
model.load_state_dict(ckpt['state_dict'])

# Function to draw a line between two points
def draw_line(img, point1, point2, color, thickness):
    return cv2.line(img, point1, point2, color, thickness)

# Function to handle mouse events
def mouse_callback(event, x, y, flags, param):
    global start_point, end_point, draw_line_flag

    if event == cv2.EVENT_LBUTTONDOWN:
        start_point = (x, y)
        draw_line_flag = True

    elif event == cv2.EVENT_LBUTTONUP:
        end_point = (x, y)
        draw_line_flag = False

# Set up the video capture from the URL
video_path = 'D:/FIVETH SEM/system thinking/test video.mp4'
cap = cv2.VideoCapture(video_path)

# Set up the mouse callback
cv2.namedWindow('Segmentation Mask with X and Y Axes and Marked Points')
cv2.setMouseCallback('Segmentation Mask with X and Y Axes and Marked Points', mouse_callback)

# Initialize the points and line drawing flag
start_point = None
end_point = None
draw_line_flag = False

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        break

    # Flip the frame vertically to correct orientation
    frame = cv2.flip(frame, 0)

    # Preprocess the frame
    img = cv2.resize(frame, (640, 480))  # Resize the frame for faster processing
    with torch.no_grad():
        img_var = Variable(torch.from_numpy(prepare_img(img).transpose(2, 0, 1)[None]), requires_grad=False).float()
        if HAS_CUDA:
            img_var = img_var.cuda()
        segm, depth = model(img_var)
        segm = cv2.resize(segm[0, :NUM_CLASSES].cpu().data.numpy().transpose(1, 2, 0),
                          frame.shape[:2][::-1],
                          interpolation=cv2.INTER_CUBIC)
        # depth = cv2.resize(depth[0, 0].cpu().data.numpy(),
        #                    img.shape[:2][::-1],
        #                    interpolation=cv2.INTER_CUBIC)
        segm = CMAP[segm.argmax(axis=2) + 1].astype(np.uint8)

    # Assuming 'segm' contains the segmentation output where green color represents the floor
    floor_class = 1  # Replace this with the class index of the floor in the segmentation output

    # Create a binary mask for the floor region
    floor_mask = (segm == CMAP[floor_class]).all(axis=2)

    # Initialize the 'mapping' array with white color
    mapping = np.full_like(frame, (255, 255, 255), dtype=np.uint8)

    # Set the floor area in the mapping to black color
    mapping[floor_mask] = (0, 0, 0)

    # Find contours in the binary floor mask
    contours, _ = cv2.findContours(floor_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Get the bounding rectangle of the floor region
    if len(contours) > 0:
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)

    # Convert the 'segm' to a binary mask
    segm_binary_mask = (segm == CMAP[floor_class]).all(axis=2)
    segm_binary_mask = segm_binary_mask.astype(np.uint8) * 255  # Convert True/False to 1/0 and then scale to 0/255
    
    edge_detected_image = cv2.Canny(segm_binary_mask, threshold1=30, threshold2=100, apertureSize=3)

    # Find contours in the edge-detected image
    contours, _ = cv2.findContours(edge_detected_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Get the largest contour (presumably the road)
    largest_contour = max(contours, key=cv2.contourArea)

    # Fit a line to the largest contour using least squares
    [vx, vy, x, y] = cv2.fitLine(largest_contour, cv2.DIST_L2, 0, 0.01, 0.01)

    # Calculate two points to define the centerline
    left_x = int(x - 1000 * vx)
    left_y = int(y - 1000 * vy)
    right_x = int(x + 1000 * vx)
    right_y = int(y + 1000 * vy)

    # Draw the centerline on the edge-detected image
    centerline_img = cv2.cvtColor(edge_detected_image, cv2.COLOR_GRAY2BGR)
    cv2.line(centerline_img, (left_x, left_y), (right_x, right_y), (0, 0, 255), 3)  # Red color
    centerline_img_resize = cv2.resize(centerline_img, (640, 480))  # Resize the frame for faster processing
    segm_resize = cv2.resize(segm, (640, 480))  # Resize the frame for faster processing
    segm_binary_mask_resize = cv2.resize(segm_binary_mask, (640, 480))  # Resize the frame for faster processing

    # Display the original image
    #
    cv2.imshow('Original Image', frame)

    # Display the segmentation output
    cv2.imshow('Segmentation Output', segm_resize)

    # Display the floor mapping (without red grid lines) with the X and Y axes and numbers
    #cv2.imshow('Floor Mapping with X and Y Axes', mapping)

    # Display the binary segmentation mask (black and white) with X and Y axes, numbers, and marked points
    cv2.imshow('Segmentation Mask with X and Y Axes and Marked Points', segm_binary_mask_resize)
    cv2.imshow('centre line drawn',centerline_img_resize )

    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit the video
        break

cap.release()
cv2.destroyAllWindows()
