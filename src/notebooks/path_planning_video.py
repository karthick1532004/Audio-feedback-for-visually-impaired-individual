import cv2
import numpy as np
import sys

sys.path.append('D:/FIVETH SEM/system thinking/system_thinking/src/')

from system_thinking.src.notebooks.models import net  # Assuming 'net' is the class or function you want to import from 'models.py'
import torch
from torch.autograd import Variable

# Pre-processing and post-processing constants #
CMAP = np.load('D:/FIVETH SEM/system thinking/system_thinking/src/cmap_nyud.npy')
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

ckpt = torch.load('D:/FIVETH SEM/system thinking/system_thinking/weights/ExpNYUD_joint.ckpt')
model.load_state_dict(ckpt['state_dict'])

# Load the video
video_path = 'D:/FIVETH SEM/system thinking/test_video.mp4'
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video file.")
    sys.exit()

batch_size = 16  # Adjust as needed
frames_buffer = []

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        break

    frames_buffer.append(frame)

    if len(frames_buffer) == batch_size:
        with torch.no_grad():
            batch = np.stack([prepare_img(f).transpose(2, 0, 1) for f in frames_buffer])
            batch_var = Variable(torch.from_numpy(batch), requires_grad=False).float()
            if HAS_CUDA:
                batch_var = batch_var.cuda()
            segm_batch, _ = model(batch_var)

        for i, segm in enumerate(segm_batch):
            segm = cv2.resize(segm[:NUM_CLASSES].cpu().data.numpy().transpose(1, 2, 0),
                              frames_buffer[i].shape[:2][::-1],
                              interpolation=cv2.INTER_CUBIC)
            segm = CMAP[segm.argmax(axis=2) + 1].astype(np.uint8)

    # Assuming 'segm' contains the segmentation output where green color represents the floor
    floor_class = 2 # Replace this with the class index of the floor in the segmentation output

    # Create a binary mask for the floor region
    floor_mask = (segm == CMAP[floor_class]).all(axis=2)

    # Initialize the 'mapping' array with white color
    mapping = np.full_like(frame, (255, 255, 255), dtype=np.uint8)

    # Set the floor area in the mapping to black color
    mapping[floor_mask] = (0, 0, 0)

    # Display the original frame
    cv2.imshow('Original Frame', frame)

    # Display the segmentation output
    cv2.imshow('Segmentation Output', segm)

    # Display the floor mapping with the red lines dividing the white space
    cv2.imshow('Floor Mapping with Grid Lines', mapping)
    
    frames_buffer = []

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()