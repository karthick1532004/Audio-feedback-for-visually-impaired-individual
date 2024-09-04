import cv2
import numpy as np
import sys

sys.path.append(r'C:\Users\hhara\OneDrive\Desktop\system_thinking/src/')

from system_thinking.src.notebooks.models import net  # Assuming 'net' is the class or function you want to import from 'models.py'
import torch
from torch.autograd import Variable

def binary_image(image_path):
    # Pre-processing and post-processing constants #
    CMAP = np.load(r'C:\Users\hhara\OneDrive\Desktop\system_thinking/src/cmap_nyud.npy')
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

    ckpt = torch.load(r'C:\Users\hhara\OneDrive\Desktop\system_thinking/weights/ExpNYUD_joint.ckpt')
    model.load_state_dict(ckpt['state_dict'])

    # Load the image
    #image_path = r"C:\Users\hhara\Downloads\WhatsApp Image 2023-08-16 at 12.43.31 PM.jpeg"  # Replace with the path to your image
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

    # Display the original image
    '''cv2.imshow('Original Image', frame)
    
    # Display the segmentation output
    cv2.imshow('Segmentation Output', segm)
    
    # Display the floor mapping with the red lines dividing the white space
    cv2.imshow('Floor Mapping with Grid Lines', mapping)
    cv2.imwrite('output1_binary.png', mapping)
    
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()'''
