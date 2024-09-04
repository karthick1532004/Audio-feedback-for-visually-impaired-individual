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
#image_path = 'D:/FIVETH SEM/system thinking/test_1.jpeg'  # Replace with the path to your image
image_path = 'D:/FIVETH SEM/system thinking/test5.jpg'  # Replace with the path to your image
#image_path = 'D:/FIVETH SEM/system thinking/test_1.jpeg'  # Replace with the path to your image
frame = cv2.imread(image_path)

# Figure 2-top row
# img_path = '../../examples/ExpNYUD_joint/000464.png'
# img_path = '../../img/000281.jpg'
img_path = cv2.resize(frame, (640, 480))
# img = np.array(Image.open(img_path))
img = np.array(img_path)

with torch.no_grad():
    img_var = Variable(torch.from_numpy(prepare_img(img).transpose(2, 0, 1)[None]), requires_grad=False).float()
    if HAS_CUDA:
        img_var = img_var.cuda()
    segm, depth = model(img_var)
    segm = cv2.resize(segm[0, :NUM_CLASSES].cpu().data.numpy().transpose(1, 2, 0),
                      img.shape[:2][::-1],
                      interpolation=cv2.INTER_CUBIC)
    # depth = cv2.resize(depth[0, 0].cpu().data.numpy(),
    #                    img.shape[:2][::-1],
    #                    interpolation=cv2.INTER_CUBIC)
    segm = CMAP[segm.argmax(axis=2) + 1].astype(np.uint8)
    # depth = np.abs(depth)

cv2.imshow('segmentation', segm)
cv2.imshow('input', cv2.resize(frame, (640, 480)))
cv2.waitKey(0)
cv2.destroyAllWindows()
