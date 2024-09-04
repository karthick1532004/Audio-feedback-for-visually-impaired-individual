import cv2
import numpy as np
from path_planning_new import binary_image
from center_line_1 import *
from getting_trajectory import * 

centerpoints = []

center =(0+int(640 / 2),0+int(480 / 2))

# Load the image
frame = cv2.imread(r"D:\FIVETH SEM\system thinking\system_thinking_1\system_thinking\test10.jpg")
frame = cv2.resize(frame, (640, 480))



masked_image, mask = binary_image(frame)
centerpoints,output_image = centre(masked_image, frame)
    
if len(centerpoints) > 5:
       img = getSteeringCommand(center, centerpoints, output_image)

# Display the processed image
cv2.imshow('out', output_image)
cv2.imshow('segm', mask)
cv2.imshow('binary', masked_image)

# Save the processed images in a specific directory
cv2.imwrite('D:/FIVETH SEM/system thinking/system_thinking_1/out.png', output_image)
cv2.imwrite('D:/FIVETH SEM/system thinking/system_thinking_1/segm.png', mask)
cv2.imwrite('D:/FIVETH SEM/system thinking/system_thinking_1/binary.png', masked_image)


cv2.waitKey(0)
cv2.destroyAllWindows()
