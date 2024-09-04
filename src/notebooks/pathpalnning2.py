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

# Load the image
image_path = 'D:/FIVETH SEM/system thinking/test5.jpg'  # Replace with the path to your image
frame = cv2.imread(image_path)
img_path = cv2.resize(frame, (640, 480))
# img = np.array(Image.open(img_path))
img = np.array(img_path)

# Perform segmentation on the image
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

# Draw X and Y axes with tick marks and numbers on the binary mask image
scale_interval = 50  # Interval between tick marks
scale_length = 10  # Length of each tick mark
scale_font = cv2.FONT_HERSHEY_SIMPLEX
scale_font_size = 0.4
scale_font_thickness = 1

# Draw X axis with tick marks and numbers
for i in range(x, x + w, scale_interval):
    cv2.line(segm_binary_mask, (i - x, y + h), (i - x, y + h - scale_length), (255, 255, 255), 1)
    cv2.putText(segm_binary_mask, str(i - x), (i - x, y + h - scale_length - 2), scale_font, scale_font_size, (255, 255, 255), scale_font_thickness, cv2.LINE_AA)

# Draw Y axis with tick marks and numbers
for j in range(y, y + h, scale_interval):
    cv2.line(segm_binary_mask, (x - x, j - y), (x - x + scale_length, j - y), (255, 255, 255), 1)
    cv2.putText(segm_binary_mask, str(j - y), (x - x + scale_length + 2, j - y), scale_font, scale_font_size, (255, 255, 255), scale_font_thickness, cv2.LINE_AA)

# Mark two points on the free space (replace these with the desired points)
point1 = (300,300)
point2 = (300, 500)

# Draw circles at the specified points on the binary mask image
circle_radius = 5
circle_color = (255, 255, 255)  # Green color for the circles
cv2.circle(segm_binary_mask, point1, circle_radius, circle_color, -1)  # -1 indicates filled circle
cv2.circle(segm_binary_mask, point2, circle_radius, circle_color, -1)  # -1 indicates filled circle

# Draw a line connecting the two points on the binary mask image (color: white)
line_color = (255, 255, 255)  # White color for the line
line_thickness = 1
cv2.line(segm_binary_mask, point1, point2, line_color, line_thickness)

# ... (Rest of the code remains the same)

# Display the original image
cv2.imshow('Original Image', frame)

# Display the segmentation output
cv2.imshow('Segmentation Output', segm)

# Display the floor mapping (without red grid lines) with the X and Y axes and numbers
cv2.imshow('Floor Mapping with X and Y Axes', mapping)

# Display the binary segmentation mask (black and white) with X and Y axes, numbers, and marked points
cv2.imshow('Segmentation Mask with X and Y Axes and Marked Points', segm_binary_mask)
cv2.imwrite('final_image_with_centerline1.png', mapping)

cv2.waitKey(0)
cv2.destroyAllWindows()